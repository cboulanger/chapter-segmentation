#!/usr/bin/env python3
"""Download the open-access chapter-segmentation evaluation PDFs on demand.

The PDFs themselves are gitignored (evaluation/corpus/*/*.pdf) so they
aren't shipped in the repo. This script reads each corpus's manifest.json
under evaluation/corpus/ (see evaluation/README.md) and downloads each
entry with "oa": true into that same corpus directory if not already
present.

Non-OA books ("oa": false) are perfectly welcome in a manifest — they just
can't be auto-downloaded. If one is missing locally, this script prints its
DOI and the exact path to save it to, so you can fetch it manually through
your institution's legal access and drop it in yourself.

Usage:
    uv run python evaluation/scripts/fetch_evaluation_pdfs.py
    uv run python evaluation/scripts/fetch_evaluation_pdfs.py --force            # re-download even if present
    uv run python evaluation/scripts/fetch_evaluation_pdfs.py --corpus open-access
"""

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir, list_corpora, load_manifest_books


def fetch_corpus(corpus: str, force: bool) -> None:
    cdir = corpus_dir(corpus)
    missing_non_oa = []

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for book in load_manifest_books(corpus):
            target = cdir / book["filename"]
            if not book["oa"]:
                if not target.exists():
                    missing_non_oa.append(book)
                continue
            if target.exists() and not force:
                print(f"[skip] {corpus}/{book['filename']} already present")
                continue
            print(f"[fetch] {corpus}/{book['filename']} <- {book['download_url']}")
            response = client.get(book["download_url"])
            response.raise_for_status()
            target.write_bytes(response.content)
            print(f"        wrote {len(response.content):,} bytes")

    if missing_non_oa:
        print(f"\n[{corpus}] Not auto-downloaded (OA: No — legally cannot be fetched automatically).")
        print("Download each one manually via your institution's access to the DOI below,")
        print("then place it at the path shown, and re-run this script (or just run the tests):\n")
        for book in missing_non_oa:
            doi = book.get("doi")
            doi_url = f"https://doi.org/{doi}" if doi else "(no DOI on file — see manifest.json)"
            print(f"  - {book['filename']}")
            print(f"      DOI: {doi_url}")
            print(f"      Save to: {cdir / book['filename']}")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-download even if the PDF already exists")
    parser.add_argument("--corpus", help="Only fetch this corpus (default: every corpus under evaluation/corpus/)")
    args = parser.parse_args()

    corpora = [args.corpus] if args.corpus else list_corpora()
    for corpus in corpora:
        fetch_corpus(corpus, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
