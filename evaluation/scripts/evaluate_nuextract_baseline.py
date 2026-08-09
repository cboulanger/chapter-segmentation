#!/usr/bin/env python3
"""Zero-shot NuExtract-1.5-tiny TOC-extraction baseline over the
chapter-segmentation evaluation corpora. See design spec
docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md.

Scores only the TOC-*listing* step (title + printed_page_number pairs)
llm_extract_toc_entries (segmentation.py) would replace -- not full
chapter-boundary localization. This script never runs
_locate_toc_entries/_chapters_from_located, and does not touch
segmentation.py, cli.py, or any production strategy code.

One-time setup: pull the model into a local Ollama server. There is no
official Ollama tag for NuExtract-1.5-tiny -- Ollama's own "nuextract" tag
is a different, older, larger model (the original phi-3-based NuExtract
v1). Pull a third-party GGUF build straight from Hugging Face instead:

    ollama pull hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0

Then run (needs the evaluation PDFs -- see evaluation/README.md's
"Fetching the PDFs"):

    uv run python evaluation/scripts/evaluate_nuextract_baseline.py
    uv run python evaluation/scripts/evaluate_nuextract_baseline.py --corpus open-access
    uv run python evaluation/scripts/evaluate_nuextract_baseline.py --model hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q4_K_M
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.segmentation import _llm_scan_indices
from evaluation.harness import analysis_pages_for, available_books, list_corpora
from evaluation.metrics import MicroAggregate
from evaluation.nuextract_baseline import build_prompt, call_ollama, parse_response, score_book

_DEFAULT_MODEL = "hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _run_corpus(corpus: str, model: str, ollama_url: str, timeout: float, grand_total: MicroAggregate) -> None:
    triples = available_books(corpus)
    if not triples:
        return
    print(f"=== {corpus} ===")
    corpus_total = MicroAggregate()
    for pdf_path, expected_path, _book in triples:
        expected_chapters = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
        file_bytes = pdf_path.read_bytes()
        pages = analysis_pages_for(corpus, file_bytes)
        if pages is None:
            print(f"{pdf_path.name}: SKIPPED (needs OCR -- see evaluation/README.md)")
            continue
        scan_indices = _llm_scan_indices(pages)
        if not scan_indices:
            print(f"{pdf_path.name}: SKIPPED (no TOC-scan pages found)")
            continue
        prompt = build_prompt(pages, scan_indices)
        try:
            raw = call_ollama(ollama_url, model, prompt, timeout=timeout)
        except Exception as exc:
            print(f"{pdf_path.name}: FAILED ({exc})")
            continue
        predicted = parse_response(raw)
        metrics = score_book(predicted, expected_chapters)
        corpus_total.add(metrics)
        grand_total.add(metrics)
        print(
            f"{pdf_path.name}: precision={metrics.precision:.2f} recall={metrics.recall:.2f} "
            f"({metrics.true_positives}/{metrics.found_count} found, "
            f"{metrics.true_positives}/{metrics.expected_count} expected)"
        )
    agg = corpus_total.compute()
    print(f"[{corpus}] aggregate: precision={agg.precision:.2f} recall={agg.recall:.2f} f1={agg.f1:.2f}\n")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", help="Only evaluate this corpus (default: every corpus)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"Ollama model tag (default: {_DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=_DEFAULT_OLLAMA_URL, help=f"Ollama server base URL (default: {_DEFAULT_OLLAMA_URL})")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-book request timeout in seconds (default: 300)")
    args = parser.parse_args()

    corpora = [args.corpus] if args.corpus else list_corpora()
    if not any(available_books(corpus) for corpus in corpora):
        print("No evaluation PDFs present -- run: uv run python evaluation/scripts/fetch_evaluation_pdfs.py")
        return 1

    grand_total = MicroAggregate()
    for corpus in corpora:
        _run_corpus(corpus, args.model, args.ollama_url, args.timeout, grand_total)

    total = grand_total.compute()
    print(f"=== TOTAL: precision={total.precision:.2f} recall={total.recall:.2f} f1={total.f1:.2f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
