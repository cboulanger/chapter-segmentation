# NuExtract-2.0-4B LoRA Fine-Tuning Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the committed tooling (data prep, LoRA training, merge, held-out evaluation) needed to answer one question: does LoRA fine-tuning on the existing ~50-book ground-truth corpus measurably close NuExtract-2.0-4B's dominant "title-correct, page-number-null" failure mode, without generating any new ground truth.

**Architecture:** A shared, dependency-light module (`evaluation/nuextract2_common.py`) holds every pure function (prompt building, target-JSON construction, the stratified train/eval split, and tokenization/padding) so it can be unit-tested without the heavy ML stack. Four thin, real-model-calling scripts under `evaluation/scripts/` orchestrate the pilot end-to-end: prepare data -> train a LoRA adapter (`transformers`+`peft`, `mps`) -> merge it into a standalone checkpoint -> convert/quantize to GGUF (manual `llama.cpp` step, documented not scripted) -> score the held-out split (`llama.cpp` only, never `transformers`/MPS, per the backend-parity bug already documented in `evaluation/RESULTS.md`).

**Tech Stack:** Python 3.12, `transformers`, `peft`, `accelerate`, `torch` (`mps` backend), `llama-cpp-python`, `huggingface_hub` -- all behind a new `nuextract-finetune` optional-dependency group so the default `uv sync`/`pytest` workflow stays free of multi-GB ML dependencies.

---

## Spec coverage self-review

Checked against `docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md` section by section:

- "Training data" (target-JSON shape, no new ground truth, training files not committed) -> Tasks 2, 3.
- "Train/eval split" (stratified, ~39/11, single split default) -> Task 2 (`stratified_split`).
- "Training mechanics" (LoRA rank/alpha/dropout/lr/epochs, masked loss, per-epoch held-out check) -> Task 4. (Per-epoch *held-out f1* specifically is out of scope for the training script itself -- see "Decision criteria"/backend-split note: only `llama.cpp` numbers are trustworthy, and `llama.cpp` has no training-time hook, so the training script tracks training loss per epoch, and the real held-out f1 check happens after merge+convert via Task 6, not during training. This is called out explicitly in Task 4.)
- "Backend split" (train via transformers/peft, eval via llama.cpp only) -> Tasks 4, 5, 6.
- "Deliverables" (`prepare_nuextract_finetune_data.py`, `finetune_nuextract.py`, documented merge/convert/quantize) -> Tasks 3, 4, 5, 7.
- "Non-goals" (no production wiring, no CV, no cloud GPU by default, text-only, no CI) -> respected throughout; nothing in this plan touches `segmentation.py` or adds CI config.
- "Decision criteria" (compare held-out f1 via llama.cpp against current zero-shot baseline) -> Task 6's script produces exactly this number, comparable against `evaluation/RESULTS.md`'s existing baseline table.

No gaps found. Placeholder scan: none of the code blocks below contain TODO/TBD; every script is complete, runnable code.

---

### Task 1: Dependency group and gitignore entry

**Files:**
- Modify: `pyproject.toml`
- Modify: `evaluation/.gitignore`

- [ ] **Step 1: Add the `nuextract-finetune` optional-dependency group**

In `pyproject.toml`, in the `[project.optional-dependencies]` table (currently `kreuzberg`, `tesseract`, `llm-eval`), add a fourth group:

```toml
nuextract-finetune = [
    "torch>=2.4.0",
    "transformers>=4.46.0",
    "peft>=0.13.0",
    "accelerate>=0.34.0",
    "llama-cpp-python>=0.3.0",
    "huggingface-hub>=0.25.0",
]
```

The full `[project.optional-dependencies]` table should read:

```toml
[project.optional-dependencies]
kreuzberg = ["httpx>=0.27.0"]
tesseract = ["pytesseract>=0.3.10", "pymupdf>=1.24.0", "pillow>=10.0.0"]
llm-eval = ["openai>=1.0.0", "httpx>=0.27.0"]
nuextract-finetune = [
    "torch>=2.4.0",
    "transformers>=4.46.0",
    "peft>=0.13.0",
    "accelerate>=0.34.0",
    "llama-cpp-python>=0.3.0",
    "huggingface-hub>=0.25.0",
]
```

- [ ] **Step 2: Verify the TOML is still valid**

Run: `uv run python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Gitignore the fine-tuning data/artifact directory**

In `evaluation/.gitignore` (currently `*.pdf`, `manifest.local.json`, `.ocr-cache/`), add a fourth line:

```
finetune/
```

Full file:

```
*.pdf
manifest.local.json
.ocr-cache/
finetune/
```

This is where Task 3's data-prep script, Task 4's trained adapter, and Task 5's merged checkpoint all write their output -- none of it may be committed, since `copyrighted-scans/` books' scan-window text is real, non-redistributable extracted PDF text (the same reason `*.pdf`/`manifest.local.json` are already ignored here).

- [ ] **Step 4: Verify the ignore rule works**

Run:
```bash
mkdir -p evaluation/finetune/data
echo '{}' > evaluation/finetune/data/train.jsonl
git status --porcelain evaluation/finetune/
```
Expected: no output (the file is ignored, not listed as untracked).

Clean up the probe file:
```bash
rm -rf evaluation/finetune
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml evaluation/.gitignore
git commit -m "chore: add nuextract-finetune optional dependency group and gitignore its data dir"
```

---

### Task 2: Shared pure helpers (`evaluation/nuextract2_common.py`)

**Files:**
- Create: `evaluation/nuextract2_common.py`
- Test: `tests/test_nuextract2_common.py`

This module holds every function the rest of the pilot needs that does **not** require `torch`/`transformers`/`peft` to be installed or a real model to be loaded -- prompt building (given an injected `apply_chat_template` callable), target-JSON construction from ground truth, the stratified train/eval split, and training-example tokenization/padding (given injected encode/eos-token-id values). Keeping these free of heavy imports means they run under the default `uv run pytest` (no `nuextract-finetune` extras needed) exactly like every other test in this repo.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nuextract2_common.py`:

```python
"""Unit tests for evaluation/nuextract2_common.py -- the NuExtract-2.0-4B
fine-tuning pilot's shared, dependency-light helpers. See design spec
docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md.

None of these tests import torch/transformers/peft -- every heavy
dependency (a real tokenizer, a real chat template) is injected as a
plain callable/value, so this file runs under the default `uv run pytest`
with no `nuextract-finetune` extras installed."""

import json
import unittest

from evaluation.nuextract2_common import (
    build_chat_prompt,
    build_target,
    pad_batch,
    stratified_split,
    tokenize_training_example,
)


class TestBuildChatPrompt(unittest.TestCase):
    def test_passes_single_user_message_with_text(self):
        captured = {}

        def fake_apply_chat_template(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "PROMPT"

        result = build_chat_prompt("scan window text", {"chapters": []}, fake_apply_chat_template)

        self.assertEqual(result, "PROMPT")
        self.assertEqual(captured["messages"], [{"role": "user", "content": "scan window text"}])
        self.assertEqual(captured["kwargs"]["tokenize"], False)
        self.assertEqual(captured["kwargs"]["add_generation_prompt"], True)
        self.assertIn('"chapters"', captured["kwargs"]["template"])


class TestBuildTarget(unittest.TestCase):
    def test_uses_start_of_citation_pages_as_page_number(self):
        chapters = [{"title": "Introduction", "authors": ["A"], "citation_pages": "1-31"}]
        self.assertEqual(
            build_target(chapters),
            {"chapters": [{"title": "Introduction", "authors": ["A"], "printed_page_number": "1"}]},
        )

    def test_null_citation_pages_becomes_null_page_number(self):
        chapters = [{"title": "Foreword", "authors": [], "citation_pages": None}]
        self.assertEqual(
            build_target(chapters),
            {"chapters": [{"title": "Foreword", "authors": [], "printed_page_number": None}]},
        )

    def test_handles_roman_numeral_citation_pages(self):
        chapters = [{"title": "Foreword", "authors": [], "citation_pages": "vii-ix"}]
        self.assertEqual(build_target(chapters)["chapters"][0]["printed_page_number"], "7")

    def test_defaults_missing_authors_to_empty_list(self):
        chapters = [{"title": "Index", "citation_pages": None}]
        self.assertEqual(build_target(chapters)["chapters"][0]["authors"], [])


class TestStratifiedSplit(unittest.TestCase):
    def test_splits_each_corpus_independently_by_eval_count(self):
        corpus_stems = {
            "open-access": [f"oa{i}" for i in range(10)],
            "copyrighted-scans": [f"cs{i}" for i in range(5)],
        }
        result = stratified_split(
            corpus_stems, eval_counts={"open-access": 3, "copyrighted-scans": 2}, seed=1,
        )

        eval_by_corpus = {"open-access": 0, "copyrighted-scans": 0}
        for corpus, _stem in result["eval"]:
            eval_by_corpus[corpus] += 1
        self.assertEqual(eval_by_corpus, {"open-access": 3, "copyrighted-scans": 2})
        self.assertEqual(len(result["train"]), 10)

    def test_train_and_eval_are_disjoint_and_cover_every_book(self):
        corpus_stems = {"open-access": [f"oa{i}" for i in range(6)]}
        result = stratified_split(corpus_stems, eval_counts={"open-access": 2}, seed=7)

        train_set = set(result["train"])
        eval_set = set(result["eval"])
        self.assertEqual(train_set & eval_set, set())
        self.assertEqual(train_set | eval_set, {("open-access", f"oa{i}") for i in range(6)})

    def test_deterministic_for_same_seed(self):
        corpus_stems = {"open-access": [f"oa{i}" for i in range(8)]}
        result_a = stratified_split(corpus_stems, eval_counts={"open-access": 3}, seed=99)
        result_b = stratified_split(corpus_stems, eval_counts={"open-access": 3}, seed=99)
        self.assertEqual(result_a, result_b)

    def test_corpus_absent_from_eval_counts_goes_entirely_to_train(self):
        corpus_stems = {"pending": ["p0", "p1"]}
        result = stratified_split(corpus_stems, eval_counts={"open-access": 8}, seed=1)
        self.assertEqual(result["eval"], [])
        self.assertEqual(len(result["train"]), 2)


class TestTokenizeTrainingExample(unittest.TestCase):
    @staticmethod
    def _fake_apply_chat_template(messages, **kwargs):
        return f"PROMPT[{messages[0]['content']}]"

    @staticmethod
    def _fake_encode(text):
        return [ord(c) for c in text]  # one fake token id per character

    def test_masks_prompt_and_keeps_completion_in_labels(self):
        result = tokenize_training_example(
            "abc", {"chapters": []}, self._fake_apply_chat_template, self._fake_encode, eos_token_id=999,
        )
        prompt_len = len(self._fake_encode("PROMPT[abc]"))
        completion_ids = self._fake_encode(json.dumps({"chapters": []}))

        self.assertEqual(result["labels"][:prompt_len], [-100] * prompt_len)
        self.assertEqual(result["labels"][prompt_len:-1], completion_ids)
        self.assertEqual(result["labels"][-1], 999)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))
        self.assertEqual(result["attention_mask"], [1] * len(result["input_ids"]))


class TestPadBatch(unittest.TestCase):
    def test_pads_to_longest_example(self):
        examples = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [1], "labels": [1], "attention_mask": [1]},
        ]
        batch = pad_batch(examples, pad_token_id=0)
        self.assertEqual(batch["input_ids"], [[1, 2, 3], [1, 0, 0]])
        self.assertEqual(batch["labels"], [[-100, 2, 3], [1, -100, -100]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1], [1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract2_common.py -v`
Expected: `ModuleNotFoundError: No module named 'evaluation.nuextract2_common'` (or `ImportError`) on collection -- the module doesn't exist yet.

- [ ] **Step 3: Implement the module**

Create `evaluation/nuextract2_common.py`:

```python
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

from evaluation.nuextract_baseline import _expected_start_page

BASE_MODEL_REPO = "numind/NuExtract-2.0-4B"
GGUF_REPO = "numind/NuExtract-2.0-4B-GGUF"
GGUF_FILENAME = "NuExtract-2.0-4B-Q4_K_M.gguf"

DEFAULT_SPLIT_SEED = 42
DEFAULT_EVAL_COUNTS = {"open-access": 8, "copyrighted-scans": 3}


def build_chat_prompt(text: str, template: dict, apply_chat_template) -> str:
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
    eval_counts contributes 0 books to eval (all of it goes to train)."""
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
    apply_chat_template,
    encode,
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
    field in torch.tensor(...) itself."""
    max_len = max(len(ex["input_ids"]) for ex in examples)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for ex in examples:
        pad_len = max_len - len(ex["input_ids"])
        batch["input_ids"].append(ex["input_ids"] + [pad_token_id] * pad_len)
        batch["labels"].append(ex["labels"] + [-100] * pad_len)
        batch["attention_mask"].append(ex["attention_mask"] + [0] * pad_len)
    return batch
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract2_common.py -v`
Expected: all tests `PASSED`.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: same pass count as before plus the new tests, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add evaluation/nuextract2_common.py tests/test_nuextract2_common.py
git commit -m "feat: add shared pure helpers for the NuExtract-2.0-4B fine-tuning pilot"
```

---

### Task 3: Data-prep script (`prepare_nuextract_finetune_data.py`)

**Files:**
- Create: `evaluation/scripts/prepare_nuextract_finetune_data.py`

No new heavy dependency -- this script only reads the existing corpus (PDFs, `.expected.json`) via `evaluation.harness` and `chapter_segmentation.segmentation`, exactly like the zero-shot baseline scripts already do. It has no dedicated unit test (same convention as `evaluate_nuextract_baseline.py`, `evaluate_chapter_segmentation_strategies.py`, etc. -- integration-only scripts that need the real evaluation PDFs, verified by manual run + `py_compile`, not `pytest`); its non-trivial logic (`build_target`, `stratified_split`) is already unit-tested in Task 2.

- [ ] **Step 1: Write the script**

Create `evaluation/scripts/prepare_nuextract_finetune_data.py`:

```python
#!/usr/bin/env python3
"""Builds train/eval JSONL fine-tuning data for the NuExtract-2.0-4B LoRA
pilot from the existing ground-truth corpus. See design spec
docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md.

Output goes to evaluation/finetune/data/ -- gitignored (see
evaluation/.gitignore) because copyrighted-scans books' scan-window text
is real extracted PDF text that can't be redistributed, the same reason
*.pdf and manifest.local.json are gitignored elsewhere in evaluation/.
Rerunning this script is the only way to reproduce the data locally; the
split itself is deterministic (see nuextract2_common.stratified_split's
seed), so anyone with the same corpus PDFs gets the same train/eval
assignment.

Each JSONL line has:
  - "text": the scan-window text (finetune_nuextract.py's model input)
  - "target": the NUEXTRACT_TEMPLATE-shaped training target (build_target)
  - "expected_chapters": the raw .expected.json chapters (citation_pages-
    shaped, for evaluate_nuextract_finetune.py's score_book calls -- NOT
    the same shape as "target", which is why both are stored)

Run (needs the evaluation PDFs -- see evaluation/README.md's "Fetching
the PDFs"):

    uv run python evaluation/scripts/prepare_nuextract_finetune_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.segmentation import _llm_scan_indices
from evaluation.harness import analysis_pages_for, available_books, list_corpora
from evaluation.nuextract2_common import DEFAULT_EVAL_COUNTS, DEFAULT_SPLIT_SEED, build_target, stratified_split

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "finetune" / "data"


def _book_text(corpus: str, pdf_path: Path) -> str | None:
    file_bytes = pdf_path.read_bytes()
    pages = analysis_pages_for(corpus, file_bytes)
    if pages is None:
        return None
    scan_indices = _llm_scan_indices(pages)
    if not scan_indices:
        return None
    return "\n\n".join(pages[i] for i in scan_indices)


def _main() -> int:
    corpus_books: dict[str, dict[str, tuple[Path, Path]]] = {}
    for corpus in list_corpora():
        corpus_books[corpus] = {
            pdf_path.stem: (pdf_path, expected_path)
            for pdf_path, expected_path, _manifest_entry in available_books(corpus)
        }

    corpus_stems = {corpus: list(books) for corpus, books in corpus_books.items()}
    split = stratified_split(corpus_stems, eval_counts=DEFAULT_EVAL_COUNTS, seed=DEFAULT_SPLIT_SEED)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_manifest = {"seed": DEFAULT_SPLIT_SEED, "eval_counts": DEFAULT_EVAL_COUNTS, "train": [], "eval": []}

    for split_name in ("train", "eval"):
        out_path = _OUTPUT_DIR / f"{split_name}.jsonl"
        written = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for corpus, stem in split[split_name]:
                pdf_path, expected_path = corpus_books[corpus][stem]
                text = _book_text(corpus, pdf_path)
                if text is None:
                    print(f"{corpus}/{stem}: SKIPPED (needs OCR or no TOC-scan pages -- see evaluation/README.md)")
                    continue
                chapters = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
                target = build_target(chapters)
                fh.write(json.dumps({
                    "corpus": corpus,
                    "stem": stem,
                    "text": text,
                    "target": target,
                    "expected_chapters": chapters,
                }) + "\n")
                split_manifest[split_name].append({"corpus": corpus, "stem": stem})
                written += 1
        print(f"{split_name}: wrote {written} books to {out_path}")

    (_OUTPUT_DIR / "split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(f"split manifest: {_OUTPUT_DIR / 'split.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 2: Syntax-check the script**

Run: `uv run python -m py_compile evaluation/scripts/prepare_nuextract_finetune_data.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Run it for real against the evaluation corpus**

Run: `uv run python evaluation/scripts/prepare_nuextract_finetune_data.py`
Expected: prints a `SKIPPED` line for any book needing OCR/with no scan-index pages, then `train: wrote N books to .../train.jsonl` (N around 39) and `eval: wrote M books to .../eval.jsonl` (M around 11), then the split-manifest path. Confirm `evaluation/finetune/data/train.jsonl`, `eval.jsonl`, and `split.json` now exist and `eval.jsonl` has one line per book listed in `split.json`'s `"eval"` array.

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/prepare_nuextract_finetune_data.py
git commit -m "feat: add NuExtract-2.0-4B fine-tuning data-prep script"
```

---

### Task 4: LoRA training script (`finetune_nuextract.py`)

**Files:**
- Create: `evaluation/scripts/finetune_nuextract.py`

Needs the `nuextract-finetune` optional dependency group. This script is thin glue over Task 2's already-tested `tokenize_training_example`/`pad_batch`/`build_chat_prompt` -- it has no dedicated unit test of its own (it needs the real ~7.5GB model and a real `mps`/GPU device, the same reason `evaluate_nuextract_baseline.py` and the other model-calling scripts in this repo have none either); its correctness rests on Task 2's tests plus the manual smoke-run below. Per the spec, held-out f1 is **not** computed here -- `llama.cpp` is the only backend this project trusts for a real accuracy number on this model (see `evaluation/RESULTS.md`'s transformers/MPS-vs-llama.cpp finding), and `llama.cpp` has no training path, so this script only tracks training loss per epoch; the real held-out check happens in Task 6, after merge + GGUF conversion.

- [ ] **Step 1: Install the optional dependency group**

Run: `uv pip install --python .venv -e ".[nuextract-finetune]"`
Expected: `torch`, `transformers`, `peft`, `accelerate`, `llama-cpp-python`, `huggingface-hub` install successfully (Metal-accelerated `llama-cpp-python` build on this machine, per this session's earlier setup).

- [ ] **Step 2: Write the script**

Create `evaluation/scripts/finetune_nuextract.py`:

```python
#!/usr/bin/env python3
"""LoRA fine-tuning for the NuExtract-2.0-4B pilot -- trains on
evaluation/finetune/data/train.jsonl (see
prepare_nuextract_finetune_data.py). Held-out f1 is NOT computed here --
only llama.cpp/GGUF produces a trustworthy accuracy number for this
model (see evaluation/RESULTS.md's transformers/MPS-vs-llama.cpp
finding), and llama.cpp has no training path, so the real held-out check
happens after merge_nuextract_lora.py + a manual GGUF conversion, via
evaluate_nuextract_finetune.py. See design spec
docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md.

Requires the `nuextract-finetune` optional dependency group:

    uv pip install --python .venv -e ".[nuextract-finetune]"

Run:

    uv run python evaluation/scripts/finetune_nuextract.py --output-dir evaluation/finetune/adapter
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

from evaluation.nuextract2_common import BASE_MODEL_REPO, pad_batch, tokenize_training_example

_TRAIN_PATH = Path(__file__).resolve().parent.parent / "finetune" / "data" / "train.jsonl"
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collate(batch: list[dict], pad_token_id: int) -> dict:
    padded = pad_batch(batch, pad_token_id)
    return {key: torch.tensor(value) for key, value in padded.items()}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Where to save the trained LoRA adapter")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--device", default="mps", help="mps, cuda, or cpu")
    args = parser.parse_args()

    if not _TRAIN_PATH.exists():
        print(f"No training data at {_TRAIN_PATH} -- run prepare_nuextract_finetune_data.py first")
        return 1

    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_REPO, token=token, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_REPO, token=token, trust_remote_code=True, torch_dtype=torch.float16,
    ).to(args.device)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_examples = _load_examples(_TRAIN_PATH)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    examples = [
        tokenize_training_example(
            ex["text"], ex["target"], tokenizer.apply_chat_template,
            lambda s: tokenizer(s, add_special_tokens=False)["input_ids"],
            tokenizer.eos_token_id,
        )
        for ex in raw_examples
    ]
    print(f"Loaded {len(examples)} training examples from {_TRAIN_PATH}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=lambda batch: _collate(batch, pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 3: Syntax-check the script**

Run: `uv run python -m py_compile evaluation/scripts/finetune_nuextract.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Smoke-test with real training data**

Requires Task 3's `evaluation/finetune/data/train.jsonl` to exist. Run a short training pass to confirm the pipeline runs end to end without crashing (full 4-epoch training is the real pilot run, done once this smoke test passes):

```bash
uv run python evaluation/scripts/finetune_nuextract.py \
    --output-dir evaluation/finetune/adapter-smoke-test --epochs 0.1
```

Expected: `Loaded <N> training examples...`, `trainable params: ...` from `print_trainable_parameters()`, a short training log (loss decreasing is not required for a 0.1-epoch smoke test, just that it runs), and `Adapter saved to evaluation/finetune/adapter-smoke-test`. Remove the smoke-test output afterward: `rm -rf evaluation/finetune/adapter-smoke-test`.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/finetune_nuextract.py
git commit -m "feat: add LoRA training script for the NuExtract-2.0-4B fine-tuning pilot"
```

---

### Task 5: Merge script (`merge_nuextract_lora.py`)

**Files:**
- Create: `evaluation/scripts/merge_nuextract_lora.py`

- [ ] **Step 1: Write the script**

Create `evaluation/scripts/merge_nuextract_lora.py`:

```python
#!/usr/bin/env python3
"""Merges a trained LoRA adapter (see finetune_nuextract.py) into the
base NuExtract-2.0-4B weights and saves a standalone merged checkpoint,
ready for GGUF conversion. See design spec
docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md.

llama.cpp's convert_hf_to_gguf.py (a separate checkout, not a pip
dependency of this repo) is run manually afterward -- see
evaluation/README.md's "NuExtract-2.0-4B fine-tuning pilot" section for
the exact commands. This script only does the transformers/peft-side
merge, the part that needs this repo's Python environment.

Requires the `nuextract-finetune` optional dependency group (see
finetune_nuextract.py's docstring).

Run:

    uv run python evaluation/scripts/merge_nuextract_lora.py \\
        --adapter-dir evaluation/finetune/adapter \\
        --output-dir evaluation/finetune/merged
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.nuextract2_common import BASE_MODEL_REPO


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_REPO, token=token, trust_remote_code=True, torch_dtype=torch.float16,
    )
    merged = PeftModel.from_pretrained(base_model, args.adapter_dir).merge_and_unload()
    merged.save_pretrained(args.output_dir)
    AutoProcessor.from_pretrained(
        BASE_MODEL_REPO, token=token, trust_remote_code=True,
    ).save_pretrained(args.output_dir)
    print(f"Merged checkpoint saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 2: Syntax-check the script**

Run: `uv run python -m py_compile evaluation/scripts/merge_nuextract_lora.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add evaluation/scripts/merge_nuextract_lora.py
git commit -m "feat: add LoRA merge script for the NuExtract-2.0-4B fine-tuning pilot"
```

(Run against a real trained adapter as part of Task 7's end-to-end pilot execution, once Task 4's full training run has produced `evaluation/finetune/adapter`.)

---

### Task 6: Held-out evaluation script (`evaluate_nuextract_finetune.py`)

**Files:**
- Create: `evaluation/scripts/evaluate_nuextract_finetune.py`

This is the script that produces the pilot's actual answer: held-out f1 via `llama.cpp`, directly comparable to `evaluation/RESULTS.md`'s existing zero-shot baseline table. It loads a real tokenizer (`AutoProcessor`) only to build the chat-template prompt string -- generation itself always goes through `llama_cpp.Llama`, never `transformers`, so the MPS decoding bug documented in `RESULTS.md` (which is specific to `transformers`-driven *generation*) does not apply to this script.

- [ ] **Step 1: Write the script**

Create `evaluation/scripts/evaluate_nuextract_finetune.py`:

```python
#!/usr/bin/env python3
"""Scores a NuExtract-2.0-4B checkpoint (base or fine-tuned) against the
held-out eval split written by prepare_nuextract_finetune_data.py, via
llama.cpp/GGUF only. See design spec
docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md
and evaluation/RESULTS.md's transformers/MPS-vs-llama.cpp finding for
why this never generates through transformers -- AutoProcessor is loaded
here only to build the chat-template prompt string; all actual
generation goes through llama_cpp.Llama.

Requires the `nuextract-finetune` optional dependency group (see
finetune_nuextract.py's docstring).

Run against a fine-tuned model (after merge_nuextract_lora.py + a manual
llama.cpp GGUF conversion -- see evaluation/README.md's "NuExtract-2.0-4B
fine-tuning pilot" section):

    uv run python evaluation/scripts/evaluate_nuextract_finetune.py \\
        --gguf-path evaluation/finetune/merged.Q4_K_M.gguf

Run against the unmodified base model instead (a same-split, same-
scoring-code sanity check against evaluation/RESULTS.md's full-corpus
zero-shot number -- downloads the published GGUF if not already cached):

    uv run python evaluation/scripts/evaluate_nuextract_finetune.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoProcessor

from evaluation.metrics import MicroAggregate
from evaluation.nuextract2_common import BASE_MODEL_REPO, GGUF_FILENAME, GGUF_REPO, build_chat_prompt
from evaluation.nuextract_baseline import NUEXTRACT_TEMPLATE, parse_response, score_book

_EVAL_JSONL_PATH = Path(__file__).resolve().parent.parent / "finetune" / "data" / "eval.jsonl"


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf-path", help="Local GGUF file (default: download the published base-model GGUF)")
    parser.add_argument("--n-ctx", type=int, default=40960)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--gpu-layers", type=int, default=-1, help="-1 = full Metal offload, 0 = CPU-only")
    args = parser.parse_args()

    if not _EVAL_JSONL_PATH.exists():
        print(f"No eval data at {_EVAL_JSONL_PATH} -- run prepare_nuextract_finetune_data.py first")
        return 1

    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_REPO, token=token, trust_remote_code=True)
    apply_chat_template = processor.tokenizer.apply_chat_template

    gguf_path = args.gguf_path or hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILENAME)
    print(f"Loading {gguf_path} ...")
    llm = Llama(model_path=gguf_path, n_gpu_layers=args.gpu_layers, n_ctx=args.n_ctx, verbose=False)

    total = MicroAggregate()
    for line in _EVAL_JSONL_PATH.read_text(encoding="utf-8").splitlines():
        example = json.loads(line)
        prompt = build_chat_prompt(example["text"], NUEXTRACT_TEMPLATE, apply_chat_template)
        try:
            out = llm.create_completion(prompt, max_tokens=args.max_tokens, temperature=0.0)
        except ValueError as exc:
            print(f"{example['corpus']}/{example['stem']}: SKIPPED ({exc})")
            continue
        predicted = parse_response(out["choices"][0]["text"])
        metrics = score_book(predicted, example["expected_chapters"])
        total.add(metrics)
        print(
            f"{example['corpus']}/{example['stem']}: precision={metrics.precision:.2f} "
            f"recall={metrics.recall:.2f} f1={metrics.f1:.2f} "
            f"({metrics.true_positives}/{metrics.found_count} found, "
            f"{metrics.true_positives}/{metrics.expected_count} expected)"
        )

    agg = total.compute()
    print(f"\n=== eval split: precision={agg.precision:.2f} recall={agg.recall:.2f} f1={agg.f1:.2f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 2: Syntax-check the script**

Run: `uv run python -m py_compile evaluation/scripts/evaluate_nuextract_finetune.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Sanity-check against the base model**

Run it with no `--gguf-path` (downloads and scores the unmodified base model on the eval split):

```bash
uv run python evaluation/scripts/evaluate_nuextract_finetune.py
```

Expected: one line per eval-split book, then `=== eval split: precision=... recall=... f1=... ===`. This number is the same measurement as `evaluation/RESULTS.md`'s full-corpus baseline (same scoring code, same backend) but restricted to the ~11-book eval split -- it won't match the full-corpus f1=0.39 exactly (different, smaller sample) but should be in a broadly similar range. This run is the pilot's actual pre-fine-tuning reference point.

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/evaluate_nuextract_finetune.py
git commit -m "feat: add held-out evaluation script for the NuExtract-2.0-4B fine-tuning pilot"
```

---

### Task 7: Documentation and end-to-end pilot run

**Files:**
- Modify: `evaluation/README.md`

- [ ] **Step 1: Document the workflow**

In `evaluation/README.md`, insert a new `### NuExtract-2.0-4B fine-tuning pilot` subsection immediately after the existing `### NuExtract baseline spike` subsection (before `### Strategy-pipeline evaluation`):

```markdown
### NuExtract-2.0-4B fine-tuning pilot

A go/no-go check on whether LoRA fine-tuning closes NuExtract-2.0-4B's
dominant zero-shot failure mode (titles/authors correct,
`printed_page_number` null -- see `RESULTS.md`'s failure-mode
breakdown), using only the existing ~50-book corpus, no new ground
truth. See
`docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`
for the full rationale and decision criteria.

One-time setup -- install the fine-tuning dependency group:

```bash
uv pip install --python .venv -e ".[nuextract-finetune]"
```

1. **Prepare data** (needs the evaluation PDFs -- see "Fetching the PDFs"
   above; no ML dependencies needed for this step):

   ```bash
   uv run python evaluation/scripts/prepare_nuextract_finetune_data.py
   ```

   Writes `evaluation/finetune/data/{train,eval}.jsonl` and `split.json`
   -- all gitignored (copyrighted-scans books' scan-window text can't be
   redistributed, same reason `*.pdf` is gitignored).

2. **Train the LoRA adapter** (`transformers`+`peft` on this machine's
   `mps` backend):

   ```bash
   uv run python evaluation/scripts/finetune_nuextract.py \
       --output-dir evaluation/finetune/adapter
   ```

3. **Merge the adapter into a standalone checkpoint**:

   ```bash
   uv run python evaluation/scripts/merge_nuextract_lora.py \
       --adapter-dir evaluation/finetune/adapter \
       --output-dir evaluation/finetune/merged
   ```

4. **Convert to GGUF and quantize**, via a local `llama.cpp` checkout
   (not a dependency of this repo -- clone it separately):

   ```bash
   python /path/to/llama.cpp/convert_hf_to_gguf.py \
       evaluation/finetune/merged \
       --outfile evaluation/finetune/merged.fp16.gguf
   /path/to/llama.cpp/llama-quantize \
       evaluation/finetune/merged.fp16.gguf \
       evaluation/finetune/merged.Q4_K_M.gguf \
       Q4_K_M
   ```

5. **Score the held-out split** and compare against the base model:

   ```bash
   # fine-tuned
   uv run python evaluation/scripts/evaluate_nuextract_finetune.py \
       --gguf-path evaluation/finetune/merged.Q4_K_M.gguf

   # base model, same split, same code -- the number to beat
   uv run python evaluation/scripts/evaluate_nuextract_finetune.py
   ```

Not a pytest test and not part of any CI workflow -- a manual, one-off
pilot, same operational pattern as the NuExtract baseline spike above.
Record the result (both f1 numbers, and whether the null-page-number
rate dropped on the held-out books) in `RESULTS.md`.
```

- [ ] **Step 2: Proofread against the actual scripts**

Re-read the four command blocks above against Tasks 3-6's actual `argparse` flags (`--output-dir`, `--adapter-dir`, `--gguf-path`) to confirm every flag name matches exactly what each script defines.

- [ ] **Step 3: Commit**

```bash
git add evaluation/README.md
git commit -m "docs: document the NuExtract-2.0-4B fine-tuning pilot workflow"
```

- [ ] **Step 4: Run the full pilot end to end**

With Tasks 1-6 committed and the optional dependency group installed, run the five numbered steps in the new README section for real: prepare data, train (full `--epochs 4.0`, not the Task 4 smoke test), merge, convert/quantize via a local `llama.cpp` checkout, then score both the fine-tuned and base checkpoints on the held-out split. This is the actual pilot result the design spec's "Decision criteria" section asks for -- expect the full run (training + both eval passes) to take on the order of a few hours on this machine, dominated by the ~11-book `llama.cpp` eval passes (minutes each) and the training run itself (tens of minutes for ~39 examples at 4 epochs).

- [ ] **Step 5: Record the result in RESULTS.md**

Add a dated section to `evaluation/RESULTS.md` (following this document's own established pattern for every prior NuExtract finding) reporting: both f1 numbers (fine-tuned vs. base, same held-out split), whether the null-page-number rate specifically dropped on the held-out books (inspect a couple of previously-null-scoring held-out books by hand, the same way earlier failure-mode entries in this document were verified), and a go/no-go conclusion per the design spec's decision criteria (a clear, not-explained-by-noise improvement concentrated in the diagnosed failure mode = go; anything within a few points of baseline = no-go / treat as noise). Commit this alongside a final `uv run pytest -q` confirmation that the rest of the suite is unaffected.

---
