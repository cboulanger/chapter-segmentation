"""Standalone CLI: `chapter-segmentation analyze <pdf>` -- no Zotero, no
RAG app, just a PDF in and a chapter list out. Defaults to TesseractOcrBackend
(no container required) when the PDF needs OCR; pass --ocr-backend kreuzberg
to use a running Kreuzberg sidecar instead.
"""

import argparse
import asyncio
import json
import sys

from chapter_segmentation.ocr import ocr_pdf_pages
from chapter_segmentation.segmentation import (
    analyze_attachment,
    extract_page_texts_for_analysis,
    pages_need_ocr,
)


def _build_ocr_backend(name: str, kreuzberg_url: str):
    if name == "tesseract":
        from chapter_segmentation.ocr_backends.tesseract import TesseractOcrBackend
        return TesseractOcrBackend()
    if name == "kreuzberg":
        from chapter_segmentation.ocr_backends.kreuzberg import KreuzbergOcrBackend
        return KreuzbergOcrBackend(kreuzberg_url=kreuzberg_url)
    raise ValueError(f"Unknown --ocr-backend {name!r}")


async def _analyze(pdf_path: str, ocr_backend_name: str, kreuzberg_url: str, ocr_cache_dir: str) -> dict:
    file_bytes = open(pdf_path, "rb").read()
    pages, _layout_used = extract_page_texts_for_analysis(file_bytes)

    if pages_need_ocr(pages):
        from pathlib import Path
        backend = _build_ocr_backend(ocr_backend_name, kreuzberg_url)
        pages = await ocr_pdf_pages(file_bytes, backend=backend, cache_dir=Path(ocr_cache_dir), language="eng")

    return analyze_attachment(pages)


def main() -> None:
    parser = argparse.ArgumentParser(prog="chapter-segmentation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Detect chapter boundaries in a PDF")
    analyze_parser.add_argument("pdf", help="Path to the PDF file")
    analyze_parser.add_argument(
        "--ocr-backend", choices=["tesseract", "kreuzberg"], default="tesseract",
        help="OCR backend to use if the PDF needs OCR (default: tesseract, no container required)",
    )
    analyze_parser.add_argument(
        "--kreuzberg-url", default="http://localhost:8100",
        help="Kreuzberg sidecar URL (only used with --ocr-backend kreuzberg)",
    )
    analyze_parser.add_argument("--ocr-cache-dir", default=".chapter-segmentation-ocr-cache")

    args = parser.parse_args()
    if args.command == "analyze":
        result = asyncio.run(_analyze(args.pdf, args.ocr_backend, args.kreuzberg_url, args.ocr_cache_dir))
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
