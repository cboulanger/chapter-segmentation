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
    parser.add_argument("--gpu-layers", type=int, default=-1, help="-1 = full GPU offload, 0 = CPU-only")
    parser.add_argument(
        "--repeat-penalty", type=float, default=1.1,
        help="llama-cpp-python's own default is 1.0 (off) -- greedy decoding (temperature=0.0, no "
        "penalty) got stuck in an infinite <diacritic><diacritic>... repetition loop on at least one "
        "held-out book (open-access/9781800641648, a Hebrew-linguistics text), burning the full "
        "--max-tokens budget without ever closing the JSON, so parse_response saw truncated/invalid "
        "JSON and scored 0/0 found despite several early chapters being extracted correctly. 1.1 is "
        "llama.cpp's own commonly-used default in its other tooling.",
    )
    parser.add_argument(
        "--repeat-last-n", type=int, default=16,
        help="Lookback window (in tokens) the repeat penalty above applies over -- passed to Llama's "
        "own last_n_tokens_size (default 64). 64 was too wide for this task: our output is a JSON "
        "LIST of chapter dicts, each repeating the same field names (\"title\"/\"authors\"/"
        "\"printed_page_number\") every ~20-40 tokens -- a 64-token window reaches back into the "
        "PREVIOUS chapter entry and penalizes that legitimate, required repetition, not just a "
        "genuine degenerate loop. Confirmed empirically: --repeat-penalty 1.1 at the 64-token default "
        "fixed the two collapsed books but broke four others that were previously correct (aggregate "
        "f1 0.59 -> 0.41, worse overall). The pathological loops seen so far repeat every 1-4 tokens "
        "(e.g. a single repeated diacritic pair), so a much shorter window should still suppress them "
        "while staying short enough to never reach the previous chapter entry's field names.",
    )
    parser.add_argument(
        "--dump-dir",
        help="Write one JSON file per book here with its raw completion text, finish_reason, parsed "
        "chapters, and expected_chapters -- for inspecting *why* a book scored the way it did (e.g. "
        "a 0/0-found book: empty generation? truncated at --max-tokens? malformed JSON?), which the "
        "console's precision/recall/f1 summary line alone can't show. Not written by default.",
    )
    args = parser.parse_args()
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    if not _EVAL_JSONL_PATH.exists():
        print(f"No eval data at {_EVAL_JSONL_PATH} -- run prepare_nuextract_finetune_data.py first")
        return 1

    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_REPO, token=token, trust_remote_code=True)
    # See finetune_nuextract.py's identical fallback: AutoProcessor.from_pretrained
    # returns a bare tokenizer (no `.tokenizer` attribute) on some transformers
    # versions, a composite processor wrapping one on others.
    tokenizer = getattr(processor, "tokenizer", processor)
    apply_chat_template = tokenizer.apply_chat_template

    gguf_path = args.gguf_path or hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILENAME)
    print(f"Loading {gguf_path} ...")
    llm = Llama(
        model_path=gguf_path, n_gpu_layers=args.gpu_layers, n_ctx=args.n_ctx,
        last_n_tokens_size=args.repeat_last_n, verbose=False,
    )

    total = MicroAggregate()
    for line in _EVAL_JSONL_PATH.read_text(encoding="utf-8").splitlines():
        example = json.loads(line)
        prompt = build_chat_prompt(example["text"], NUEXTRACT_TEMPLATE, apply_chat_template)
        try:
            out = llm.create_completion(
                prompt, max_tokens=args.max_tokens, temperature=0.0, repeat_penalty=args.repeat_penalty,
            )
        except ValueError as exc:
            print(f"{example['corpus']}/{example['stem']}: SKIPPED ({exc})")
            continue
        raw_text = out["choices"][0]["text"]
        finish_reason = out["choices"][0].get("finish_reason")
        predicted = parse_response(raw_text)
        metrics = score_book(predicted, example["expected_chapters"])
        total.add(metrics)
        print(
            f"{example['corpus']}/{example['stem']}: precision={metrics.precision:.2f} "
            f"recall={metrics.recall:.2f} f1={metrics.f1:.2f} "
            f"({metrics.true_positives}/{metrics.found_count} found, "
            f"{metrics.true_positives}/{metrics.expected_count} expected, "
            f"finish_reason={finish_reason})"
        )
        if dump_dir:
            dump_path = dump_dir / f"{example['corpus']}__{example['stem']}.json"
            dump_path.write_text(json.dumps({
                "corpus": example["corpus"],
                "stem": example["stem"],
                "finish_reason": finish_reason,
                "raw_text": raw_text,
                "predicted_chapters": predicted,
                "expected_chapters": example["expected_chapters"],
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            }, indent=2), encoding="utf-8")

    agg = total.compute()
    print(f"\n=== eval split: precision={agg.precision:.2f} recall={agg.recall:.2f} f1={agg.f1:.2f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
