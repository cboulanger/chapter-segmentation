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

Migrated books land in evaluation/corpus/pending/, not open-access/ --
confirmation + novelty give high GT confidence, but open-access/ is what
this pilot's canonical numbers are measured against, so a human gets a
chance to spot-check (or evaluate in isolation via
evaluate_layout_toc_classifier.py's --corpora flag) before a book affects
those numbers. See
docs/superpowers/specs/2026-08-11-crossref-gt-corpus-expansion-design.md.

Usage:
    uv run python evaluation/scripts/build_crossref_gt_ground_truth.py --dry-run
    uv run python evaluation/scripts/build_crossref_gt_ground_truth.py
"""

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
from rapidfuzz import fuzz
from sklearn.preprocessing import StandardScaler

from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.scripts.evaluate_layout_toc_classifier import build_feature_table, load_book_corpus
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
from evaluation.scripts.layout_features import FEATURE_NAMES, extract_page_features
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary

_CROSSREF_DIR = Path(__file__).resolve().parent.parent / "crossref_gt"
_OPEN_ACCESS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "open-access"
_PENDING_DIR = Path(__file__).resolve().parent.parent / "corpus" / "pending"
_ALTO_CACHE_DIR = _CROSSREF_DIR / ".layout-cache"

_MIN_OFFSET_VOTES = 5  # pages agreeing on the same offset, before it's trusted
_WINDOW = 6  # +/- pages searched around each offset-derived candidate index
_MIN_MATCH_SCORE = 85.0
_MIN_CONFIRMED_FRACTION = 0.8
_MIN_CONFIRMED_CHAPTERS = 3
_DEFAULT_NOVELTY_PERCENTILE = 90.0  # see design spec's "Novelty metric" decision


def _nearest_neighbor_distance(vector: list[float], others: list[list[float]]) -> float:
    """Euclidean distance from `vector` to the closest point in `others`."""
    return min(math.dist(vector, other) for other in others)


def _novelty_threshold(vectors: list[list[float]], percentile: float) -> float:
    """The `percentile`-th percentile of `vectors`' own leave-one-out
    nearest-neighbor distances -- pages farther than this from the
    existing corpus sit outside how tightly that corpus already clusters.
    `vectors` must have at least 2 entries (a single vector has no
    leave-one-out neighbor to measure against)."""
    loo_distances = [
        _nearest_neighbor_distance(vector, vectors[:i] + vectors[i + 1 :])
        for i, vector in enumerate(vectors)
    ]
    return float(np.percentile(loo_distances, percentile))


def _is_novel(
    candidate_vectors: list[list[float]], existing_vectors: list[list[float]], threshold: float
) -> bool:
    """True if at least one candidate page's nearest-neighbor distance to
    the existing corpus meets or exceeds `threshold` -- the design spec's
    "keep if at least one page is novel" rule, since a book that's mostly
    ordinary but has one unusual chapter-opening is still worth keeping."""
    return any(
        _nearest_neighbor_distance(vector, existing_vectors) >= threshold
        for vector in candidate_vectors
    )


def _flatten_outline(outline: list, reader: PdfReader) -> list[tuple[str, int]]:
    """Flattens a (possibly nested) pypdf outline -- a list where nested
    lists represent nesting -- into (title, page_index) pairs in reading
    order. Skips entries with no title, or whose page pypdf can't resolve
    (get_destination_page_number raising means the destination is
    malformed or points outside this PDF -- skip it, don't crash the
    whole reconciliation over one bad bookmark)."""
    entries: list[tuple[str, int]] = []
    for item in outline:
        if isinstance(item, list):
            entries.extend(_flatten_outline(item, reader))
            continue
        title = getattr(item, "title", None)
        if not title:
            continue
        try:
            page_index = reader.get_destination_page_number(item)
        except Exception:
            continue
        entries.append((title, page_index))
    return entries


def _outline_agreement_report(
    outline_entries: list[tuple[str, int]], confirmed_chapters: list[dict]
) -> list[str]:
    """Diagnostic-only cross-check (see design spec's "Outline scope"
    decision): for each confirmed chapter, fuzzy-matches its title against
    outline_entries, and if the best confident match (score
    >= _MIN_MATCH_SCORE, the same bar the content-search confirmation
    itself uses) disagrees with the content-search-confirmed
    pdf_start_index, includes a line for it in the returned list. Returns
    an empty list when there's no outline, or when everything agrees --
    this has zero effect on the migration decision, purely evidence for a
    future decision on whether outline agreement should be allowed to
    rescue failed confirmations."""
    if not outline_entries:
        return []
    lines: list[str] = []
    for chapter in confirmed_chapters:
        best_page, best_score = None, 0.0
        for title, page_index in outline_entries:
            score = fuzz.partial_ratio(chapter["title"].lower(), title.lower())
            if score > best_score:
                best_page, best_score = page_index, score
        if best_score >= _MIN_MATCH_SCORE and best_page != chapter["pdf_start_index"]:
            lines.append(
                f'outline disagreement: chapter "{chapter["title"]}" '
                f"outline={best_page} content-search={chapter["pdf_start_index"]}"
            )
    return lines


@dataclass
class NoveltyBaseline:
    """Precomputed once per script run (not once per candidate book) --
    reloading and re-fitting against the whole open-access corpus for
    every candidate would be correct but wasteful."""

    pdfalto_bin: str
    scaler: StandardScaler
    chapter_first_vectors: list[list[float]]
    threshold: float


def _load_novelty_baseline(pdfalto_bin: str, novelty_percentile: float) -> NoveltyBaseline:
    """Loads the current evaluation/corpus/open-access/ corpus, fits a
    StandardScaler on its full feature-row set (the same normalization
    evaluate_layout_toc_classifier's own LogisticRegression classifier
    uses, so novelty distances are computed in the same space it reasons
    in), and returns the scaled chapter_first vectors plus the novelty
    distance threshold derived from their own leave-one-out
    nearest-neighbor distances."""

    def cache_dir_for(corpus: str) -> Path:
        return _OPEN_ACCESS_DIR.parent / corpus / ".layout-cache"

    books = load_book_corpus(corpora=["open-access"])
    rows = build_feature_table(books, cache_dir_for, pdfalto_bin)
    X = [[row["features"][name] for name in FEATURE_NAMES] for row in rows]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).tolist()
    chapter_first_vectors = [
        vector for vector, row in zip(X_scaled, rows) if row["label"] == LABEL_CHAPTER_FIRST
    ]
    threshold = _novelty_threshold(chapter_first_vectors, novelty_percentile)
    return NoveltyBaseline(pdfalto_bin, scaler, chapter_first_vectors, threshold)


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


def process_book(book: dict, dry_run: bool, novelty_baseline: NoveltyBaseline) -> tuple[str, str]:
    """Returns (isbn, outcome_message)."""
    isbn = book["isbn"]
    pdf_path = _CROSSREF_DIR / f"{isbn}.pdf"
    crossref_path = _CROSSREF_DIR / f"{isbn}.crossref.json"
    target_pdf = _PENDING_DIR / f"{isbn}.pdf"
    target_expected = _PENDING_DIR / f"{isbn}.expected.json"

    if not pdf_path.exists():
        return isbn, "SKIP: no PDF (fetch_crossref_gt_corpus.py first)"
    if not crossref_path.exists():
        return isbn, "SKIP: no crossref.json"
    if target_expected.exists():
        return isbn, "SKIP: already migrated (evaluation/corpus/pending/*.expected.json exists)"

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

    alto_path = ensure_alto_xml(pdf_path, _ALTO_CACHE_DIR, novelty_baseline.pdfalto_bin)
    page_features = extract_page_features(str(alto_path))
    candidate_rows = [
        [page_features[c["pdf_start_index"]][name] for name in FEATURE_NAMES]
        for c in confirmed
        if c["pdf_start_index"] in page_features
    ]
    if not candidate_rows:
        return isbn, "SKIP: no layout features extracted for confirmed chapter-first pages"
    candidate_vectors = novelty_baseline.scaler.transform(candidate_rows).tolist()
    if not _is_novel(candidate_vectors, novelty_baseline.chapter_first_vectors, novelty_baseline.threshold):
        return isbn, (
            f"SKIP: not novel (no confirmed chapter-first page's nearest-neighbor "
            f"distance reaches the {novelty_baseline.threshold:.3f} threshold)"
        )

    outline_entries = _flatten_outline(reader.outline or [], reader)
    for line in _outline_agreement_report(outline_entries, confirmed):
        print(f"  [{isbn}] {line}")

    toc_field, write_toc_key, toc_status = _toc_field_for(toc_pages)

    if dry_run:
        return isbn, (
            f"OK (dry-run): {n_confirmed}/{with_page} chapters ({fraction:.0%}) would be written, {toc_status}"
        )

    _PENDING_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, target_pdf)
    payload = {"chapters": confirmed}
    if write_toc_key:
        payload["toc"] = toc_field
    target_expected.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_path = _PENDING_DIR / "manifest.json"
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
    parser.add_argument("--pdfalto-bin", default=None, help="Path to the pdfalto binary (see pdfalto_runner.py)")
    parser.add_argument(
        "--novelty-percentile",
        type=float,
        default=_DEFAULT_NOVELTY_PERCENTILE,
        help=(
            "Percentile of the existing open-access chapter-first corpus's own "
            "leave-one-out nearest-neighbor distances used as the novelty threshold. "
            f"Default: {_DEFAULT_NOVELTY_PERCENTILE}."
        ),
    )
    args = parser.parse_args()
    pdfalto_bin = resolve_pdfalto_binary(args.pdfalto_bin)
    novelty_baseline = _load_novelty_baseline(pdfalto_bin, args.novelty_percentile)

    manifest = json.loads((_CROSSREF_DIR / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for book in manifest["books"]:
        result = process_book(book, args.dry_run, novelty_baseline)
        results.append(result)
        print(f"[{result[0]}] {result[1]}")

    n_ok = sum(1 for _, msg in results if msg.startswith("OK"))
    print(f"\n{n_ok}/{len(results)} book(s) {'would be ' if args.dry_run else ''}migrated to pending/")

    needs_review = [isbn for isbn, msg in results if "toc NEEDS REVIEW" in msg]
    if needs_review:
        print(f"{len(needs_review)} migrated book(s) have an ambiguous toc field -- run add_toc_ground_truth.py to resolve:")
        for isbn in needs_review:
            print(f"  - {isbn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
