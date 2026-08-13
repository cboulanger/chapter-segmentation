#!/usr/bin/env python3
"""Retrofits existing evaluation/corpus/*/*.expected.json files with a
"toc" field, using the same structural TOC-page detection
ground_truth_helper.py already uses to exclude TOC pages from chapter-start
search. See
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md.

Auto-written entries still need a spot-check (open the PDF at
toc_start_index/toc_end_index, confirm) before being trusted -- this script
finds the best-scoring structural match, not necessarily the correct one,
same caveat as ground_truth_helper.py's chapter-boundary drafts.

Usage:
    uv run python evaluation/scripts/add_toc_ground_truth.py
    uv run python evaluation/scripts/add_toc_ground_truth.py --force
    uv run python evaluation/scripts/add_toc_ground_truth.py --corpus pending
    uv run python evaluation/scripts/add_toc_ground_truth.py --corpus pending copyrighted-scans
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader

from evaluation.scripts.ground_truth_helper import find_toc_pages, toc_page_range

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
_DEFAULT_CORPORA = ["open-access", "copyrighted-scans"]


def retrofit_book(pages: list[str], expected: dict, force: bool) -> tuple[dict | None, str]:
    """Returns (updated expected dict or None if unchanged, status message).
    Pure function -- no file I/O -- so it's independently testable."""
    if "toc" in expected and not force:
        return None, "SKIP: already has a toc field"

    toc_pages = find_toc_pages(pages)
    toc_range = toc_page_range(toc_pages)

    if toc_pages and toc_range is None:
        return None, f"NEEDS REVIEW: non-contiguous TOC-like pages found: {sorted(toc_pages)}"

    updated = dict(expected)
    if toc_range is None:
        updated["toc"] = None
        return updated, "OK: no TOC page found, wrote toc=null"

    updated["toc"] = {"toc_start_index": toc_range[0], "toc_end_index": toc_range[1]}
    return updated, f"OK: wrote toc={updated['toc']}"


def _load_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="Re-run books that already have a toc field")
    parser.add_argument(
        "--corpus",
        nargs="+",
        metavar="NAME",
        default=None,
        help=f"Corpus/corpora to process, e.g. 'pending'. Defaults to {_DEFAULT_CORPORA}.",
    )
    args = parser.parse_args()

    needs_review = []
    n_written = 0
    n_skipped = 0

    for corpus in args.corpus or _DEFAULT_CORPORA:
        corpus_dir = _CORPUS_DIR / corpus
        if not corpus_dir.is_dir():
            print(f"[{corpus}] SKIP: no such corpus directory: {corpus_dir}")
            continue
        for expected_path in sorted(corpus_dir.glob("*.expected.json")):
            key = expected_path.name.removesuffix(".expected.json")
            pdf_path = corpus_dir / f"{key}.pdf"
            if not pdf_path.exists():
                print(f"[{corpus}/{key}] SKIP: no PDF found at {pdf_path}")
                n_skipped += 1
                continue

            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            pages = _load_pages(pdf_path)
            updated, message = retrofit_book(pages, expected, args.force)
            print(f"[{corpus}/{key}] {message}")

            if updated is not None:
                expected_path.write_text(
                    json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                n_written += 1
            elif message.startswith("NEEDS REVIEW"):
                needs_review.append(f"{corpus}/{key}")
            else:
                n_skipped += 1

    print(f"\n{n_written} book(s) written, {n_skipped} skipped, {len(needs_review)} need manual review")
    if needs_review:
        print("Needs manual review:")
        for entry in needs_review:
            print(f"  - {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
