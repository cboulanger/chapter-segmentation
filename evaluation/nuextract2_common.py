"""Shared, dependency-light building blocks for the NuExtract-2.0-4B
fine-tuning pilot -- data-prep, training, merge, and evaluation scripts
all import from here rather than duplicating the chat-prompt/target-
JSON/split/tokenization logic across four scripts. See design spec
docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md.

None of the functions below import torch/transformers/peft -- every
heavy dependency (a real tokenizer's apply_chat_template, a real
encode function, a real eos_token_id) is injected as a plain
callable/value by the calling script, so this module (and its tests)
stay usable without the `nuextract-finetune` optional dependency group.

Distinct from evaluation/nuextract_baseline.py, which targets
NuExtract-1.5-tiny's raw <|input|>/<|output|> prompt convention via
Ollama -- NuExtract-2.0-4B uses transformers' chat-template convention
instead. NUEXTRACT_TEMPLATE/parse_response/score_book are schema-
identical between the two models' TocEntry-shaped output, so scripts
using this module import those three straight from
evaluation.nuextract_baseline rather than duplicating them here.
"""

import json
import random
from typing import Callable

from evaluation.nuextract_baseline import _expected_start_page

BASE_MODEL_REPO = "numind/NuExtract-2.0-4B"
GGUF_REPO = "numind/NuExtract-2.0-4B-GGUF"
GGUF_FILENAME = "NuExtract-2.0-4B-Q4_K_M.gguf"

DEFAULT_SPLIT_SEED = 42
DEFAULT_EVAL_COUNTS = {"open-access": 8, "copyrighted-scans": 3}


def build_chat_prompt(text: str, template: dict, apply_chat_template: Callable[..., str]) -> str:
    """Builds the chat-template prompt NuExtract-2.0-4B expects: a single
    user turn holding the scan-window text, with `template`
    (NUEXTRACT_TEMPLATE-shaped) passed through apply_chat_template's own
    `template` kwarg. `apply_chat_template` is injected rather than
    importing transformers here -- pass
    `processor.tokenizer.apply_chat_template` at the call site."""
    messages = [{"role": "user", "content": text}]
    return apply_chat_template(
        messages, template=json.dumps(template), tokenize=False, add_generation_prompt=True,
    )


def build_target(chapters: list[dict]) -> dict:
    """Converts a book's .expected.json "chapters" list into the
    NUEXTRACT_TEMPLATE-shaped target a fine-tuned model should learn to
    produce. printed_page_number is the start of citation_pages, as a
    string (matching how a real filled template represents every field),
    or None when citation_pages itself is null/unparseable -- the model
    should learn to say "I don't know" here, not be forced to guess, so
    the field is left null rather than the chapter being dropped from
    the target entirely (title/authors are still real, useful
    supervision)."""
    entries = []
    for chapter in chapters:
        start = _expected_start_page(chapter.get("citation_pages"))
        entries.append({
            "title": chapter["title"],
            "authors": chapter.get("authors", []),
            "printed_page_number": str(start) if start is not None else None,
        })
    return {"chapters": entries}


def stratified_split(
    corpus_stems: dict[str, list[str]],
    eval_counts: dict[str, int] = DEFAULT_EVAL_COUNTS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, list[tuple[str, str]]]:
    """Splits each corpus's book stems independently (so both corpora's
    very different zero-shot baselines land on both sides of the split,
    not just one), then pools the per-corpus train/eval picks into two
    flat (corpus, stem) lists. Deterministic for a given seed -- sorting
    before shuffling means the result depends only on the input sets and
    the seed, not filesystem iteration order. A corpus absent from
    eval_counts contributes 0 books to eval (all of it goes to train). If
    eval_counts[corpus] exceeds that corpus's own book count, the
    `stems[:n_eval]`/`stems[n_eval:]` slicing degrades gracefully: all of
    that corpus's books go to eval and none to train."""
    rng = random.Random(seed)
    train: list[tuple[str, str]] = []
    eval_: list[tuple[str, str]] = []
    for corpus in sorted(corpus_stems):
        stems = sorted(corpus_stems[corpus])
        rng.shuffle(stems)
        n_eval = eval_counts.get(corpus, 0)
        eval_.extend((corpus, stem) for stem in stems[:n_eval])
        train.extend((corpus, stem) for stem in stems[n_eval:])
    return {"train": train, "eval": eval_}


def tokenize_training_example(
    text: str,
    target: dict,
    apply_chat_template: Callable[..., str],
    encode: Callable[[str], list[int]],
    eos_token_id: int,
) -> dict:
    """Builds one training example's input_ids/labels/attention_mask as
    plain Python lists (no torch dependency -- see pad_batch for
    batching, and evaluation/scripts/finetune_nuextract.py for where
    these get wrapped in tensors). The prompt half is masked out of the
    loss (label=-100, HF's cross-entropy "ignore this position"
    sentinel); only the target JSON + EOS token contribute to the loss,
    so the model is trained to produce the completion, not to predict
    its own prompt back. `apply_chat_template` and `encode` are injected
    so this stays testable without a real tokenizer -- pass
    `processor.tokenizer.apply_chat_template` and
    `lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]` at
    the call site. Uses a blank NUEXTRACT_TEMPLATE-shaped template
    (matching evaluation.nuextract_baseline.NUEXTRACT_TEMPLATE) rather
    than importing it, to avoid a second import path for the same
    literal -- callers needing the real constant should import it from
    evaluation.nuextract_baseline directly."""
    template = {"chapters": [{"title": "", "authors": [""], "printed_page_number": ""}]}
    prompt = build_chat_prompt(text, template, apply_chat_template)
    completion = json.dumps(target)

    prompt_ids = encode(prompt)
    completion_ids = encode(completion) + [eos_token_id]

    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
    }


def pad_batch(examples: list[dict], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pads a batch of tokenize_training_example outputs to the
    longest example's length -- plain nested lists, not tensors, so this
    stays torch-free and unit-testable; the training script wraps each
    field in torch.tensor(...) itself. Expects a non-empty `examples`
    list -- its only caller is finetune_nuextract.py's Trainer data
    collator, whose DataLoader never yields an empty batch."""
    max_len = max(len(ex["input_ids"]) for ex in examples)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for ex in examples:
        pad_len = max_len - len(ex["input_ids"])
        batch["input_ids"].append(ex["input_ids"] + [pad_token_id] * pad_len)
        batch["labels"].append(ex["labels"] + [-100] * pad_len)
        batch["attention_mask"].append(ex["attention_mask"] + [0] * pad_len)
    return batch
