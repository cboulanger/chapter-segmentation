"""Whole-book agreement gate for dnb-toc-only ground truth -- see design
spec docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4. Pure functions over TocEntry lists produced by the two
existing extractors (find_toc_candidates, llm_extract_toc_entries --
src/chapter_segmentation/segmentation.py), which already return the
identical list[TocEntry] shape."""

from dataclasses import replace

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


def gate_book(
    heuristic: list[TocEntry], llm: list[TocEntry], threshold: float = 0.90,
) -> tuple[bool, list[TocEntry]]:
    """Whole-book agreement gate (design spec section 4.2).
    agreement_rate = matched-pair count / max(len(heuristic), len(llm)).
    Below `threshold`, the book is rejected outright (passed=False,
    entries=[]) rather than trimmed down to just the agreeing entries -- a
    partially-agreeing book is exactly the case this design distrusts
    most, and a caller must not silently write a partial/incomplete
    result for it.

    At or above `threshold`, `entries` is the UNION of matched pairs
    (the heuristic's title kept, but falling back to the LLM's authors
    when the heuristic's own are empty -- the heuristic's title comes from
    structured regex capture rather than LLM reformatting, but the heuristic
    almost never populates authors except in a narrow marker-line case, so
    this preserves real author info the LLM extracted while preferring the
    more reliable regex-captured title) plus every singleton entry either
    extractor found alone, ordered by printed_page_number (the -1
    "unknown" sentinel sorts last). This is deliberate: once a book
    clears the trust bar, a line only one extractor caught is far likelier
    a real entry the other missed (OCR noise, an unusual title format)
    than a hallucination -- trimming it out would silently understate the
    page's real content, which is exactly the "incomplete training
    target" failure mode this design exists to avoid."""
    if not heuristic and not llm:
        return False, []
    pairs = align_toc_entries(heuristic, llm)
    agreement_rate = len(pairs) / max(len(heuristic), len(llm))
    if agreement_rate < threshold:
        return False, []
    matched_h = {i for i, _ in pairs}
    matched_l = {j for _, j in pairs}
    merged = [
        replace(heuristic[i], authors=heuristic[i].authors or llm[j].authors)
        for i, j in pairs
    ]
    merged += [entry for i, entry in enumerate(heuristic) if i not in matched_h]
    merged += [entry for j, entry in enumerate(llm) if j not in matched_l]
    merged.sort(key=lambda e: (e.printed_page_number == -1, e.printed_page_number))
    return True, merged


def toc_entry_to_gt_dict(entry: TocEntry) -> dict:
    """Serializes one TocEntry to this corpus's <id>.expected.json entry
    shape (design spec section 2) -- printed_page_number as a string, or
    None for the -1 "unknown" sentinel. Matches
    evaluation/nuextract2_common.py's build_target output shape directly
    (its primary downstream consumer, per the parent program spec's
    section 3), and mirrors how citation_pages is already stored as a
    string elsewhere in this project's ground truth."""
    return {
        "title": entry.title,
        "authors": list(entry.authors),
        "printed_page_number": str(entry.printed_page_number) if entry.printed_page_number != -1 else None,
    }
