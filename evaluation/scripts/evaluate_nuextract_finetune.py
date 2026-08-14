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
    args = parser.parse_args()

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
