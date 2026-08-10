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
