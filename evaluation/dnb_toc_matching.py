"""Whole-book agreement gate for dnb-toc-only ground truth -- see design
spec docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4. Pure functions over TocEntry lists produced by the two
existing extractors (find_toc_candidates, llm_extract_toc_entries --
src/chapter_segmentation/segmentation.py), which already return the
identical list[TocEntry] shape."""

from rapidfuzz import fuzz

from chapter_segmentation.segmentation import TocEntry

# Same constant src/chapter_segmentation/evidence/fusion.py's _align uses
# for its own title-similarity matching.
_ALIGN_SCORE_THRESHOLD = 70.0


def align_toc_entries(a: list[TocEntry], b: list[TocEntry]) -> list[tuple[int, int]]:
    """Greedy, order-preserving alignment between two independently-
    produced TocEntry lists for the same TOC scan. A pair (i, j) counts as
    a match only when both sides have a KNOWN printed_page_number (neither
    is the -1 "unknown" sentinel) that's numerically equal, AND their
    titles score >= _ALIGN_SCORE_THRESHOLD on rapidfuzz's
    token_sort_ratio -- mirrors evaluation/nuextract_baseline.py's
    match_toc_entries (page-number-first, then title) and
    src/chapter_segmentation/evidence/fusion.py's _align (greedy scan from
    the last matched b-index, "TOC order is book order"), but returns
    index PAIRS rather than a bare count, since the whole-book gate below
    needs to know exactly which entries agreed."""
    pairs: list[tuple[int, int]] = []
    last_j = -1
    for i, entry_a in enumerate(a):
        if entry_a.printed_page_number == -1:
            continue
        best_j = None
        best_score = _ALIGN_SCORE_THRESHOLD
        for j in range(last_j + 1, len(b)):
            entry_b = b[j]
            if entry_b.printed_page_number != entry_a.printed_page_number:
                continue
            score = fuzz.token_sort_ratio(entry_a.title.lower(), entry_b.title.lower())
            if score >= best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            pairs.append((i, best_j))
            last_j = best_j
    return pairs
