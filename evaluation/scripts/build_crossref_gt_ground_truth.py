#!/usr/bin/env python3
"""One-time reconciliation: turn evaluation/crossref_gt/ (Crossref-sourced
book-chapter metadata -- printed citation_pages ranges only, no PDF-relative
indices) into real evaluation/corpus/open-access/ ground truth, completing
the "Follow-up work" explicitly deferred in
docs/superpowers/specs/2026-08-08-crossref-gt-corpus-design.md.

Approach: for each book, scan every page for a printed page number (reusing
evaluation/scripts/ground_truth_helper.py's extract_printed_number) and
derive the constant PDF-index-minus-printed-number offset by consensus vote
across all pages (arabic and roman pagination get their own offset, since
front matter is often numbered separately from the body). This is the same
"never assume pdf_index == printed_page_number" caution evaluation/CLAUDE.md
describes, just solved by measuring the real, book-specific offset instead
of assuming it's zero. Each chapter's Crossref citation_pages start then
maps directly to a candidate PDF index; a small window search around that
candidate confirms it's really that chapter's title/byline page (not a
continuation page) via fuzzy content match, the same signal
ground_truth_helper.py uses for hand-built ground truth. Two independent
sources agreeing -- Crossref's registered page number and the PDF's own
printed page number -- is what stands in for the hand-verification
evaluation/CLAUDE.md normally requires for 896 chapters across 46 books.

A book where the offset can't be derived (too few pages carry a printed
number pypdf/extract_printed_number can read), or where too few chapters
land a confident content match at their candidate page, is left untouched
-- its crossref_gt/ files stay put for manual curation later, rather than
polluting the harness with unverified chapter boundaries.

Usage:
    uv run python evaluation/scripts/build_crossref_gt_ground_truth.py --dry-run
    uv run python evaluation/scripts/build_crossref_gt_ground_truth.py
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
from rapidfuzz import fuzz

from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range

_CROSSREF_DIR = Path(__file__).resolve().parent.parent / "crossref_gt"
_OPEN_ACCESS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "open-access"

_MIN_OFFSET_VOTES = 5  # pages agreeing on the same offset, before it's trusted
_WINDOW = 6  # +/- pages searched around each offset-derived candidate index
_MIN_MATCH_SCORE = 85.0
_MIN_CONFIRMED_FRACTION = 0.8
_MIN_CONFIRMED_CHAPTERS = 3


def _citation_start(citation_pages: str | None) -> str | None:
    if not citation_pages:
        return None
    start, _, _ = citation_pages.partition("-")
    return start.strip() or None


def _derive_offset(pages: list[str], roman: bool) -> int | None:
    """Consensus PDF-index-minus-printed-number offset across every page
    whose printed number is (roman if `roman` else arabic) -- front matter
    and body are frequently numbered independently, so they get separate
    offsets. None if too few pages carry a readable number of that kind to
    trust the result."""
    counter: Counter[int] = Counter()
    for index, text in enumerate(pages):
        raw = extract_printed_number(text)
        if raw is None or raw.isdigit() == roman:
            continue
        value = _parse_toc_page_number(raw)
        if value is None or value <= 0:
            continue
        counter[index - value] += 1
    if not counter:
        return None
    offset, votes = counter.most_common(1)[0]
    return offset if votes >= _MIN_OFFSET_VOTES else None


def _locate_near(
    pages: list[str], toc_pages: set[int], title: str, authors: list[str], center: int, window: int
) -> tuple[int | None, float]:
    """Confirms a printed-number-derived candidate page is really this
    chapter's opening page (title/byline match), scanning a small window
    around it rather than ground_truth_helper's whole-book sequential
    search -- the printed number already narrows down *where*, this only
    needs to confirm *that it's the right page*."""
    last_names = [a.split()[-1].lower() for a in authors if a.strip()]
    best: tuple[int | None, float] = (None, 0.0)
    lo, hi = max(0, center - window), min(len(pages), center + window + 1)
    for index in range(lo, hi):
        if index in toc_pages:
            continue
        head = pages[index][:250].lower()
        score = fuzz.partial_ratio(title.lower(), head)
        if last_names and any(name in head for name in last_names):
            score = min(100.0, score + 5.0)
        if score > best[1]:
            best = (index, score)
    return best


def _toc_field_for(toc_pages: set[int]) -> tuple[dict | None, bool, str]:
    """Determines what (if anything) to write for the "toc" key, matching
    add_toc_ground_truth.py's retrofit_book() semantics exactly: an empty
    toc_pages means "confirmed no TOC" (write null), while a non-empty but
    non-contiguous toc_pages means "ambiguous, needs a human" -- writing a
    wrong null there would be actively incorrect *and* would permanently
    block add_toc_ground_truth.py's own "if 'toc' in expected: skip" check
    from ever revisiting this book later. Returns
    (field_value_if_any, should_write_key, status_message_for_logging)."""
    if not toc_pages:
        return None, True, "toc=null (no TOC found)"
    toc_range = toc_page_range(toc_pages)
    if toc_range is None:
        return None, False, f"toc NEEDS REVIEW: non-contiguous TOC-like pages {sorted(toc_pages)}"
    field = {"toc_start_index": toc_range[0], "toc_end_index": toc_range[1]}
    return field, True, f"toc={field}"


def _sanity_check(chapters: list[dict], total_pages: int) -> str | None:
    ranges = sorted((c["pdf_start_index"], c["pdf_end_index"]) for c in chapters)
    for start, end in ranges:
        if start > end:
            return f"start>end: {(start, end)}"
        if end >= total_pages:
            return f"end>=total_pages({total_pages}): {(start, end)}"
    for (_, end1), (start2, _) in zip(ranges, ranges[1:]):
        if start2 <= end1:
            return f"overlap: end {end1} vs next start {start2}"
    return None


def process_book(book: dict, dry_run: bool) -> tuple[str, str]:
    """Returns (isbn, outcome_message)."""
    isbn = book["isbn"]
    pdf_path = _CROSSREF_DIR / f"{isbn}.pdf"
    crossref_path = _CROSSREF_DIR / f"{isbn}.crossref.json"
    target_pdf = _OPEN_ACCESS_DIR / f"{isbn}.pdf"
    target_expected = _OPEN_ACCESS_DIR / f"{isbn}.expected.json"

    if not pdf_path.exists():
        return isbn, "SKIP: no PDF (fetch_crossref_gt_corpus.py first)"
    if not crossref_path.exists():
        return isbn, "SKIP: no crossref.json"
    if target_expected.exists():
        return isbn, "SKIP: already migrated (evaluation/corpus/open-access/*.expected.json exists)"

    chapters = json.loads(crossref_path.read_text(encoding="utf-8"))["chapters"]
    if not chapters:
        return isbn, "SKIP: zero chapters"

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    pages = [page.extract_text() or "" for page in reader.pages]
    toc_pages = find_toc_pages(pages)

    digit_offset = _derive_offset(pages, roman=False)
    roman_offset = _derive_offset(pages, roman=True)
    if digit_offset is None and roman_offset is None:
        return isbn, "SKIP: could not derive a printed-page-number offset (too few readable page numbers)"

    with_page = 0
    candidates: list[dict] = []
    for chapter in chapters:
        start = _citation_start(chapter["citation_pages"])
        if start is None:
            continue
        with_page += 1
        is_roman = not start.isdigit()
        offset = roman_offset if is_roman else digit_offset
        value = _parse_toc_page_number(start)
        if offset is None or value is None:
            continue
        center = offset + value
        if not (0 <= center < total_pages):
            continue
        index, score = _locate_near(pages, toc_pages, chapter["title"], chapter["authors"], center, _WINDOW)
        if index is None or score < _MIN_MATCH_SCORE:
            continue
        candidates.append({
            "title": chapter["title"],
            "authors": chapter["authors"],
            "citation_pages": chapter["citation_pages"],
            "pdf_start_index": index,
        })

    n_confirmed = len(candidates)
    fraction = (n_confirmed / with_page) if with_page else 0.0
    if n_confirmed < _MIN_CONFIRMED_CHAPTERS or fraction < _MIN_CONFIRMED_FRACTION:
        return isbn, (
            f"SKIP: only {n_confirmed}/{with_page} chapters confirmed "
            f"({fraction:.0%}, need >={_MIN_CONFIRMED_FRACTION:.0%} and >={_MIN_CONFIRMED_CHAPTERS})"
        )

    # Two chapters can legitimately share a printed page number (rare) but
    # never a PDF index -- collapse duplicates by keeping the higher-scoring
    # one implicitly via dict-by-index, then re-sort.
    by_index = {c["pdf_start_index"]: c for c in candidates}
    confirmed = sorted(by_index.values(), key=lambda c: c["pdf_start_index"])

    for i, chapter in enumerate(confirmed):
        next_start = confirmed[i + 1]["pdf_start_index"] if i + 1 < len(confirmed) else total_pages
        end = next_start - 1
        while end > chapter["pdf_start_index"] and len(pages[end].strip()) < 150:
            end -= 1
        chapter["pdf_end_index"] = end

    error = _sanity_check(confirmed, total_pages)
    if error:
        return isbn, f"SKIP: sanity check failed after reconciliation: {error}"

    toc_field, write_toc_key, toc_status = _toc_field_for(toc_pages)

    if dry_run:
        return isbn, (
            f"OK (dry-run): {n_confirmed}/{with_page} chapters ({fraction:.0%}) would be written, {toc_status}"
        )

    _OPEN_ACCESS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, target_pdf)
    payload = {"chapters": confirmed}
    if write_toc_key:
        payload["toc"] = toc_field
    target_expected.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_path = _OPEN_ACCESS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["books"].append({
        "filename": f"{isbn}.pdf",
        "title": book["title"],
        "language": book["language"],
        "extraction_type": "native",
        "embedded_toc": bool(reader.outline),
        "oa": True,
        "doi": book["doi"],
        "download_url": book["download_url"],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return isbn, f"OK: {n_confirmed}/{with_page} chapters ({fraction:.0%}) written, {toc_status}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing files")
    args = parser.parse_args()

    manifest = json.loads((_CROSSREF_DIR / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for book in manifest["books"]:
        result = process_book(book, args.dry_run)
        results.append(result)
        print(f"[{result[0]}] {result[1]}")

    n_ok = sum(1 for _, msg in results if msg.startswith("OK"))
    print(f"\n{n_ok}/{len(results)} book(s) {'would be ' if args.dry_run else ''}migrated to open-access/")

    needs_review = [isbn for isbn, msg in results if "toc NEEDS REVIEW" in msg]
    if needs_review:
        print(f"{len(needs_review)} migrated book(s) have an ambiguous toc field -- run add_toc_ground_truth.py to resolve:")
        for isbn in needs_review:
            print(f"  - {isbn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
