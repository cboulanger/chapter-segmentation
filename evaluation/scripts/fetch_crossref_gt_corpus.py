#!/usr/bin/env python3
"""Fetch the Crossref-sourced ground-truth corpus.

Reads evaluation/crossref_gt/manifest.json (the curated list of open-access
books — see evaluation/crossref_gt/README.md) and, for each book, downloads
its PDF and its Crossref-registered book-chapter metadata into that same
directory. Both steps are independently skipped if their target file
already exists, so a partial prior run resumes cleanly; --force refetches
everything.

This corpus is standalone -- it is not yet consumed by
tests/test_segmentation_accuracy.py or evaluation/generate_report.py.

Usage:
    uv run python evaluation/scripts/fetch_crossref_gt_corpus.py
    uv run python evaluation/scripts/fetch_crossref_gt_corpus.py --force
    uv run python evaluation/scripts/fetch_crossref_gt_corpus.py --contact-email you@example.org
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "crossref_gt"
_DEFAULT_CONTACT_EMAIL = "boulanger@lhlt.mpg.de"


def _crossref_book_chapters(isbn: str, client: httpx.Client, contact_email: Optional[str]) -> list[dict]:
    """GET .../works?filter=isbn:{isbn}, returning raw type=="book-chapter"
    items. Any network/HTTP/JSON failure is printed and treated as an empty
    result -- never raises, so one bad book never aborts the batch."""
    params: dict[str, str | int] = {
        "filter": f"isbn:{isbn}",
        "select": "DOI,title,subtitle,author,page,type,container-title,published,ISBN",
        "rows": 100,
    }
    if contact_email:
        params["mailto"] = contact_email

    response = None
    for _attempt in range(_MAX_RETRIES):
        try:
            response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
        except httpx.HTTPError as exc:
            print(f"  [warn] network error fetching Crossref metadata for {isbn}: {exc}")
            return []
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
        time.sleep(delay)
    else:
        print(f"  [warn] exhausted retries (429) fetching Crossref metadata for {isbn}")
        return []

    try:
        response.raise_for_status()
        items = response.json()["message"]["items"]
    except Exception as exc:
        print(f"  [warn] bad Crossref response for {isbn}: {exc}")
        return []

    return [item for item in items if item.get("type") == "book-chapter"]


def _normalize_chapter(item: dict) -> dict:
    """Projects a raw Crossref book-chapter item to {title, authors,
    chapter_doi, citation_pages}. title follows the same title+subtitle
    join convention as CrossrefMetadataStrategy's _parse_crossref_item
    (src/chapter_segmentation/evidence/crossref_strategy.py) -- Crossref
    splits a chapter's real printed heading into separate title/subtitle
    fields, so title alone is often a truncated fragment."""
    titles = item.get("title") or []
    subtitles = item.get("subtitle") or []
    title = " ".join(part for part in (titles[0] if titles else "", subtitles[0] if subtitles else "") if part)
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in item.get("author", []) if a.get("family")
    ]
    return {
        "title": title,
        "authors": authors,
        "chapter_doi": item.get("DOI"),
        "citation_pages": item.get("page"),
    }


def _fetch_crossref_metadata(
    isbn: str, client: httpx.Client, contact_email: Optional[str], target: Path, force: bool
) -> tuple[int, int]:
    """Writes target (<isbn>.crossref.json) unless it already exists and
    not force. Returns (chapter_count, chapters_missing_page_range) either
    way, so the caller can flag books needing review even on a skipped
    (already-fetched) run. A corrupted/truncated cache file (e.g. from a
    prior run interrupted mid-write) is treated as a cache miss -- never
    raises, so one bad book never aborts the batch."""
    if target.exists() and not force:
        try:
            cached = json.loads(target.read_text(encoding="utf-8"))
            chapters = cached["chapters"]
            return len(chapters), sum(1 for c in chapters if not c["citation_pages"])
        except Exception as exc:
            print(f"  [warn] corrupted cache file {target.name}, refetching: {exc}")

    raw_items = _crossref_book_chapters(isbn, client, contact_email)
    chapters = [_normalize_chapter(item) for item in raw_items]
    payload = {
        "isbn": isbn,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw_items": raw_items,
        "chapters": chapters,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(chapters), sum(1 for c in chapters if not c["citation_pages"])


def _download_pdf(book: dict, client: httpx.Client, target: Path, force: bool) -> str:
    """Returns 'skip', 'downloaded', or 'failed'. Never raises -- a failed
    download for one book must not abort the batch."""
    if target.exists() and not force:
        return "skip"
    try:
        response = client.get(book["download_url"], timeout=120)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  [warn] failed to download PDF for {book['isbn']}: {exc}")
        return "failed"
    target.write_bytes(response.content)
    return "downloaded"


def fetch_all(corpus_dir: Path, force: bool, contact_email: Optional[str]) -> int:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    books = manifest["books"]
    flagged: list[tuple[str, str]] = []
    total_chapters = 0
    downloaded = 0

    with httpx.Client(follow_redirects=True) as client:
        for book in books:
            isbn = book["isbn"]
            print(f"[{isbn}] {book['title']}")

            pdf_status = _download_pdf(book, client, corpus_dir / f"{isbn}.pdf", force)
            print(f"  pdf: {pdf_status}")
            if pdf_status == "downloaded":
                downloaded += 1

            n_chapters, n_missing_page = _fetch_crossref_metadata(
                isbn, client, contact_email, corpus_dir / f"{isbn}.crossref.json", force
            )
            print(f"  crossref: {n_chapters} chapter(s), {n_missing_page} missing page range")
            total_chapters += n_chapters
            if n_chapters == 0:
                flagged.append((isbn, "zero book-chapter records"))
            elif n_missing_page:
                flagged.append((isbn, f"{n_missing_page} chapter(s) missing page range"))

    print(f"\n{len(books)} book(s) processed, {downloaded} PDF(s) newly downloaded, {total_chapters} chapter(s) fetched")
    if flagged:
        print("\nFlagged for curation review:")
        for isbn, reason in flagged:
            print(f"  - {isbn}: {reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", default=str(_CORPUS_DIR), help="Directory containing manifest.json")
    parser.add_argument("--force", action="store_true", help="Refetch PDFs and Crossref metadata even if present")
    parser.add_argument("--contact-email", default=_DEFAULT_CONTACT_EMAIL, help="Crossref polite-pool contact email")
    args = parser.parse_args()
    return fetch_all(Path(args.corpus_dir), args.force, args.contact_email)


if __name__ == "__main__":
    raise SystemExit(main())
