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
    # AutoProcessor.from_pretrained's return type for this model varies by
    # transformers version: some return a composite processor wrapping a
    # `.tokenizer`, others return the bare tokenizer itself (no `.tokenizer`
    # attribute) -- handle both rather than assuming one.
    tokenizer = getattr(processor, "tokenizer", processor)
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
