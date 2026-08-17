"""Whole-book agreement gate for dnb-toc-only ground truth -- see design
spec docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4, and docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
for the follow-up redesign. Pure functions over TocEntry lists produced by
two independent vision_extract_toc_entries calls
(evaluation/dnb_toc_vision.py), which already return the identical
list[TocEntry] shape."""

from dataclasses import replace

from rapidfuzz import fuzz

from chapter_segmentation.segmentation import TocEntry, _parse_toc_page_number

# Same constant src/chapter_segmentation/evidence/fusion.py's _align uses
# for its own title-similarity matching.
_ALIGN_SCORE_THRESHOLD = 70.0


def _pages_equivalent(a: str | None, b: str | None) -> bool:
    """True when two entries' printed_page_number values represent the
    same page. None never matches None (or anything else) -- "unknown"
    on either side means there is nothing to compare, same policy the
    old -1-sentinel-skip already enforced. Otherwise: exact string match
    first (handles a shared alternate-scheme marker like "R42" directly,
    with no numeric parsing needed at all); then numeric equality via
    _parse_toc_page_number (handles a case difference in a roman numeral,
    "VII" vs "vii", or a leading zero, "07" vs "7"); then a
    case-insensitive string match (handles a case difference in a
    non-roman marker, "R42" vs "r42").

    Known limitation: the numeric tier does not consult printed_roman,
    so a roman marker can numerically collide with an unrelated arabic
    marker of the same value ("L" vs "50", both 50) -- accepted rather
    than threading printed_roman through this function and every
    align_toc_entries call site, since align_toc_entries' own
    title-similarity gate already makes a real false-positive rare (it
    additionally requires the two entries' titles to score >=
    _ALIGN_SCORE_THRESHOLD)."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    parsed_a, parsed_b = _parse_toc_page_number(a), _parse_toc_page_number(b)
    if parsed_a is not None and parsed_a == parsed_b:
        return True
    return a.casefold() == b.casefold()


def _candidate_titles(entry: TocEntry) -> tuple[str, ...]:
    """Every title reading worth trying for a fuzzy match: the primary
    title, plus any longer wrapped-title variants (TocEntry.title_variants
    -- see its own docstring in segmentation.py). An entry whose title
    wrapped across multiple TOC-page lines can have its FULL title only in
    title_variants, not title itself (a line-based capture may record just
    the last line, where the page number sits) -- comparing only .title
    against the other side's full, whole-read title would systematically
    under-score a real match. Found empirically: a real book's wrapped
    page-33 title matched its own title_variants near-verbatim but scored
    well below threshold against .title alone (2026-08-15 smoke test, book
    9783899718188). Note that of this module's current callers, only
    find_toc_candidates' regex path ever populates title_variants --
    _toc_items_to_entries (shared by both the text-LLM and vision-LLM
    paths) never does, so for two vision-extraction inputs this tuple is
    just (entry.title,) on both sides; the mechanism stays in place for
    whichever future extractor populates it."""
    return (entry.title,) + entry.title_variants


def align_toc_entries(a: list[TocEntry], b: list[TocEntry]) -> list[tuple[int, int]]:
    """Greedy, order-preserving alignment between two independently-
    produced TocEntry lists for the same TOC scan. A pair (i, j) counts as
    a match only when both sides have a KNOWN printed_page_number (neither
    is None) that's equivalent per _pages_equivalent (exact string match,
    numeric match, or case-insensitive string match), AND their
    titles score >= _ALIGN_SCORE_THRESHOLD on the better of rapidfuzz's
    token_sort_ratio and partial_ratio (the latter added 2026-08-16 to
    tolerate a trailing noise run on one side -- e.g. garbled OCR tokens
    following an otherwise-exact-match title -- without inflating false
    positives; see design spec
    docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
    section 1d/3.2 for the measurements behind this) -- mirrors evaluation/nuextract_baseline.py's
    match_toc_entries (page-number-first, then title) and
    src/chapter_segmentation/evidence/fusion.py's _align (greedy scan from
    the last matched b-index, "TOC order is book order"), but returns
    index PAIRS rather than a bare count, since the whole-book gate below
    needs to know exactly which entries agreed."""
    pairs: list[tuple[int, int]] = []
    last_j = -1
    for i, entry_a in enumerate(a):
        if entry_a.printed_page_number is None:
            continue
        best_j = None
        best_score = _ALIGN_SCORE_THRESHOLD
        for j in range(last_j + 1, len(b)):
            entry_b = b[j]
            if not _pages_equivalent(entry_a.printed_page_number, entry_b.printed_page_number):
                continue
            score = max(
                max(
                    fuzz.token_sort_ratio(title_a.lower(), title_b.lower()),
                    fuzz.partial_ratio(title_a.lower(), title_b.lower()),
                )
                for title_a in _candidate_titles(entry_a)
                for title_b in _candidate_titles(entry_b)
            )
            if score >= best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            pairs.append((i, best_j))
            last_j = best_j
    return pairs


def diff_toc_entries(
    a: list[TocEntry], b: list[TocEntry],
) -> tuple[list[tuple[TocEntry, TocEntry]], list[TocEntry], list[TocEntry]]:
    """Aligns a and b via align_toc_entries and returns (matched_pairs,
    only_in_a, only_in_b) -- matched_pairs holds the actual TocEntry
    objects (not indices) from each side for each matched line,
    only_in_a/only_in_b hold every entry from that side with no match on
    the other. Same underlying alignment gate_book uses to decide
    pass/fail; this exposes the full breakdown for a human (or Claude,
    arbitrating a below-threshold book) to review the actual
    disagreement -- see
    docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md
    section 4.1."""
    pairs = align_toc_entries(a, b)
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    matched_pairs = [(a[i], b[j]) for i, j in pairs]
    only_in_a = [entry for i, entry in enumerate(a) if i not in matched_a]
    only_in_b = [entry for j, entry in enumerate(b) if j not in matched_b]
    return matched_pairs, only_in_a, only_in_b


def gate_book(
    a: list[TocEntry], b: list[TocEntry], threshold: float = 0.90,
) -> tuple[bool, list[TocEntry]]:
    """Whole-book agreement gate (design spec section 4.2 of the original
    2026-08-15 design; the two inputs are now two independent vision-model
    extractions rather than a regex heuristic and a text-LLM pass -- see
    docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section
    3.1). agreement_rate = matched-pair count / max(len(a), len(b)).
    Below `threshold`, the book is rejected outright (passed=False,
    entries=[]) rather than trimmed down to just the agreeing entries -- a
    partially-agreeing book is exactly the case this design distrusts
    most, and a caller must not silently write a partial/incomplete
    result for it.

    At or above `threshold`, `entries` is the UNION of matched pairs (`a`'s
    title kept -- an arbitrary but deterministic choice between two
    equally-produced extractions -- falling back to `b`'s authors when
    `a`'s own are empty, in case one model dropped them) plus every
    singleton entry either side found alone, ordered by
    printed_page_number (an unknown, i.e. None, value sorts last). This is
    deliberate: once a book clears the trust bar, a line only one side
    caught is far likelier a real entry the other missed than a
    hallucination -- trimming it out would silently understate the page's
    real content, which is exactly the "incomplete training target"
    failure mode this design exists to avoid."""
    if not a and not b:
        return False, []
    matched_pairs, only_in_a, only_in_b = diff_toc_entries(a, b)
    agreement_rate = len(matched_pairs) / max(len(a), len(b))
    if agreement_rate < threshold:
        return False, []
    merged = [
        replace(entry_a, authors=entry_a.authors or entry_b.authors)
        for entry_a, entry_b in matched_pairs
    ]
    merged += only_in_a
    merged += only_in_b

    def _page_sort_key(entry: TocEntry) -> tuple:
        value = _parse_toc_page_number(entry.printed_page_number) if entry.printed_page_number else None
        return (entry.printed_page_number is None, value if value is not None else 0, entry.printed_page_number or "")

    merged.sort(key=_page_sort_key)
    return True, merged


def toc_entry_to_gt_dict(entry: TocEntry) -> dict:
    """Serializes one TocEntry to this corpus's <id>.expected.json entry
    shape (design spec section 2) -- printed_page_number is already
    str | None on TocEntry (see docs/superpowers/specs/2026-08-17-printed-page-number-string-design.md),
    so it's passed through as-is. Matches evaluation/nuextract2_common.py's
    build_target output shape directly (its primary downstream consumer,
    per the parent program spec's section 3), and mirrors how
    citation_pages is already stored as a string elsewhere in this
    project's ground truth.

    "skip" (added 2026-08-17, see TocEntry.skip's own docstring) records
    the vision extraction's own hint about whether this entry is a real
    chapter, but is NOT authoritative -- a downstream consumer that wants
    "which of these are real chapters" should be free to reclassify from
    the verbatim title/page data without needing new vision-model calls."""
    return {
        "title": entry.title,
        "authors": list(entry.authors),
        "printed_page_number": entry.printed_page_number,
        "skip": entry.skip,
    }
