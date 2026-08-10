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
