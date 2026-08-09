"""NuExtract-1.5-tiny zero-shot TOC-extraction baseline spike. See design
spec docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md.

Scores only the TOC-*listing* step (title + printed_page_number pairs)
llm_extract_toc_entries (segmentation.py) would replace -- not full
chapter-boundary localization. This module never runs
_locate_toc_entries/_chapters_from_located.
"""

import json

import httpx
from rapidfuzz import fuzz

from chapter_segmentation._llm_json import parse_json_object
from chapter_segmentation.evidence.fusion import _ALIGN_SCORE_THRESHOLD
from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.metrics import Metrics

# Mirrors TocEntry's fields (segmentation.py) -- title/authors/
# printed_page_number, the same shape llm_extract_toc_entries asks a
# cloud LLM for, formatted as NuExtract's own template convention instead
# of a free-form instruction.
NUEXTRACT_TEMPLATE = {
    "chapters": [
        {"title": "", "authors": [""], "printed_page_number": ""},
    ],
}


def build_prompt(pages: list[str], scan_indices: list[int]) -> str:
    """Formats NuExtract's documented <|input|>/### Template/### Text/
    <|output|> input convention. Unlike _LLM_TOC_EXTRACTION_PROMPT
    (segmentation.py), no "[PAGE i]" markers are included -- NuExtract
    copies values verbatim from the text rather than following
    instructions, and this spike does not use page indices at all (see
    module docstring), only whatever printed page number already appears
    next to each title in the raw text."""
    text = "\n\n".join(pages[i] for i in scan_indices)
    template_json = json.dumps(NUEXTRACT_TEMPLATE)
    return f"<|input|>\n### Template:\n{template_json}\n### Text:\n{text}\n\n<|output|>"


def parse_response(raw: str) -> list[dict]:
    """Parses NuExtract's filled-template output into the "chapters" list.
    Returns [] for empty/unparseable output or a malformed "chapters"
    field -- treated as "no signal", mirroring llm_extract_toc_entries'
    own failure handling (segmentation.py) rather than raising."""
    try:
        data = parse_json_object(raw)
    except ValueError:
        return []
    chapters = data.get("chapters")
    return chapters if isinstance(chapters, list) else []


def _expected_start_page(citation_pages) -> int | None:
    """The chapter's own start page from an expected.json entry's
    "citation_pages" field (e.g. "1-31" -> 1, "vii-ix" -> 7). None if the
    field is null or has no "-" separator."""
    if not citation_pages or "-" not in citation_pages:
        return None
    start, _, _ = citation_pages.partition("-")
    return _parse_toc_page_number(start.strip())


def _predicted_page(printed_page_number) -> int | None:
    """The parsed page number from a NuExtract-predicted entry's
    "printed_page_number" field. None for a missing/null/empty value or
    one _parse_toc_page_number can't interpret."""
    if printed_page_number is None:
        return None
    text = str(printed_page_number).strip()
    if not text or text.lower() == "null":
        return None
    return _parse_toc_page_number(text)


def match_toc_entries(predicted: list[dict], expected: list[dict]) -> int:
    """Counts true positives: a predicted entry matches the next
    unmatched expected chapter whose start page (see
    _expected_start_page) is identical to the predicted entry's own page
    (see _predicted_page) AND whose title fuzzy-matches at or above
    fusion.py's own _ALIGN_SCORE_THRESHOLD. Greedy and order-preserving --
    once an expected index is matched, no earlier expected index can be
    matched again -- exactly mirroring fusion._align's "TOC listing order
    is book order" assumption. An entry on either side with no parseable
    page number can never match (see the two "null ... never matches"
    tests) -- this is a real scope limitation of this metric, not a bug:
    it means a chapter with no visible printed page number is
    unreachable by this scoring, consistent with the spec's "Scoring"
    section requiring an exact page match. A non-dict predicted item (a
    plausible malformed-model-output shape) is skipped, not raised on --
    mirrors llm_extract_toc_entries' own per-item defensive parsing in
    segmentation.py."""
    last_j = -1
    matched = 0
    for pred in predicted:
        if not isinstance(pred, dict):
            continue
        pred_page = _predicted_page(pred.get("printed_page_number"))
        pred_title = str(pred.get("title") or "").strip().lower()
        if pred_page is None or not pred_title:
            continue
        best_j = None
        best_score = 0.0
        for j in range(last_j + 1, len(expected)):
            exp_page = _expected_start_page(expected[j].get("citation_pages"))
            if exp_page is None or exp_page != pred_page:
                continue
            score = fuzz.token_sort_ratio(pred_title, str(expected[j].get("title") or "").strip().lower())
            if score >= _ALIGN_SCORE_THRESHOLD and score > best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            matched += 1
            last_j = best_j
    return matched
