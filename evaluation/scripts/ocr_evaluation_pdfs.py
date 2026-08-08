#!/usr/bin/env python3
"""OCR the evaluation books whose text layer is absent or degenerate, into
each corpus's gitignored OCR cache (evaluation/corpus/<name>/.ocr-cache/,
content-hash keyed), so the accuracy harness and evaluation scripts can
analyze them the way a real caller would.

Uses KreuzbergOcrBackend by default (pass --ocr-backend tesseract for the
local-binary path instead). Books already cached are skipped instantly, so
re-runs are cheap; the first run over several full scanned books takes a
long time.

    uv run python evaluation/scripts/ocr_evaluation_pdfs.py
    uv run python evaluation/scripts/ocr_evaluation_pdfs.py --corpus copyrighted
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.ocr import detect_language, ocr_pdf_pages
from chapter_segmentation.segmentation import extract_page_texts_for_analysis, pages_need_ocr
from evaluation.harness import available_books, list_corpora, ocr_cache_dir


async def _main(ocr_backend_name: str, kreuzberg_url: str, corpus_filter: str | None) -> int:
    if ocr_backend_name == "tesseract":
        from chapter_segmentation.ocr_backends.tesseract import TesseractOcrBackend
        backend = TesseractOcrBackend()
    else:
        from chapter_segmentation.ocr_backends.kreuzberg import KreuzbergOcrBackend
        backend = KreuzbergOcrBackend(kreuzberg_url=kreuzberg_url)

    corpora = [corpus_filter] if corpus_filter else list_corpora()
    for corpus in corpora:
        for pdf_path, _expected_path, book in available_books(corpus):
            file_bytes = pdf_path.read_bytes()
            pages, _layout_used = extract_page_texts_for_analysis(file_bytes)
            if not pages_need_ocr(pages):
                print(f"{corpus}/{pdf_path.name}: text layer usable, no OCR needed")
                continue
            language = detect_language(book.get("language"), book.get("title", ""))
            print(f"{corpus}/{pdf_path.name}: OCR-ing {len(pages)} pages (language={language}) ...", flush=True)
            try:
                page_texts = await ocr_pdf_pages(
                    file_bytes, backend=backend, cache_dir=ocr_cache_dir(corpus), language=language,
                )
            except Exception as exc:
                print(f"{corpus}/{pdf_path.name}: FAILED ({exc}) -- skipping, will retry on next run", flush=True)
                continue
            print(f"{corpus}/{pdf_path.name}: done, {sum(len(p) for p in page_texts)} chars cached", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-backend", choices=["kreuzberg", "tesseract"], default="kreuzberg")
    parser.add_argument("--kreuzberg-url", default="http://localhost:8100")
    parser.add_argument("--corpus", help="Only OCR this corpus (default: every corpus under evaluation/corpus/)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args.ocr_backend, args.kreuzberg_url, args.corpus)))
