"""Generates bulk-tier structured ground truth for dnb-toc-only (design
spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md).
For every manifest book not held out in eval_tier_ids.json (see
select_dnb_toc_eval_sample.py and evaluation/README.md's "Building
dnb-toc-only ground truth"), runs two independent extractors -- the regex
heuristic (find_toc_candidates) and a KISSKI LLM pass
(llm_extract_toc_entries) -- and writes <id>.expected.json with
"verified": false only when they agree well enough
(evaluation.dnb_toc_matching.gate_book, >=0.90 whole-book agreement).
Books that don't clear the gate are skipped and reported, not partially
written.

Spends real KISSKI API budget (one call per book, not per-model -- see
evaluation/refresh_llm_cache.py's docstring for the shared KISSKI_API_KEY
setup this script reuses). Not a pytest test.

    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 50   # smoke test
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py               # full corpus
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --spot-check 30
"""

from chapter_segmentation.segmentation import TocEntry, find_toc_candidates

# find_toc_candidates rejects any printed page number above
# len(pages) * _TOC_MAX_PAGE_NUMBER_RATIO (2.0, segmentation.py) -- a
# guard against mistaking book-internal noise for a real TOC elsewhere in
# a full book. A dnb-toc-only PDF *is* the TOC (1-3 pages) but prints page
# numbers from the ORIGINAL BOOK (which can run into the hundreds), so
# calling find_toc_candidates on it unpadded silently rejects nearly every
# real entry. Padding with harmless filler pages before the call -- same
# technique tests/test_segmentation.py's own _FILLER_PAGES fixture already
# uses -- raises the ratio guard's ceiling comfortably above any real
# book's page count without touching segmentation.py. The real content is
# always at the front of the padded list, well within
# find_toc_candidates' default 15% front-scan window regardless of how
# much padding is appended.
_PAGE_NUMBER_GUARD_PADDING = 1000


def _toc_entries_for_scan(pages: list[str]) -> list[TocEntry]:
    """Runs the heuristic regex extractor on a dnb-toc-only book's own
    page texts, working around _TOC_MAX_PAGE_NUMBER_RATIO's tiny-PDF
    false-rejection (see module docstring)."""
    padded = pages + ["Filler page, not part of the digitized TOC scan."] * _PAGE_NUMBER_GUARD_PADDING
    return find_toc_candidates(padded)
