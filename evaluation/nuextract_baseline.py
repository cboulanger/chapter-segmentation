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
