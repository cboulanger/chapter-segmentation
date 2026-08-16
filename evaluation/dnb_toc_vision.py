"""Vision-LLM TOC extraction for dnb-toc-only -- reads each book's page
images directly (no OCR, no text layer at all), per design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section 3.
dnb-toc-only's PDFs are pre-filtered to just their TOC pages during
acquisition (1-3 pages typically), so rendering every page unconditionally
is cheap and bounded -- this does NOT generalize to whole-book PDFs."""

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries

_VISION_TOC_EXTRACTION_PROMPT = """\
You are reading photographed/scanned page images of a book's table of \
contents. Some layouts use simple dotted leaders ("Title ..... 12"), \
others print the author's name on its own line above or below the title, \
or right-align the page number with no leader at all -- read the images \
directly rather than assuming one fixed layout.

Return ONLY a JSON array, one entry per real chapter -- skip \
acknowledgements, bibliography, index, and part-divider pages:
[{"title": "...", "authors": ["First Last", ...], "printed_page_number": "12"}]

printed_page_number is the page number exactly AS PRINTED on the page -- \
copy it verbatim, including roman numerals for front-matter chapters \
(e.g. "vii", not 7). If a chapter's printed page number is not visible, \
use null for printed_page_number. If authors are not identifiable, use an \
empty list."""

# Rendered image count this corpus's PDFs never exceed today (1-3 pages,
# per the acquisition pipeline's own TOC-only filtering) -- guards against
# silently building an arbitrarily large multi-image request if a
# mis-filtered outlier ever slips through (design spec section 5).
_MAX_VISION_PAGES = 20

# Mirrors segmentation.py's _LLM_TOC_RETRY_MAX_TOKENS escalation: a dense
# multi-page edited-volume TOC (60-80 entries, long German titles, author
# lists) can plausibly overrun the first attempt's budget. A truncated JSON
# array reliably fails parse_json_array (no closing "]") regardless of
# cause, so JSON-parseability alone is a sufficient, client-agnostic retry
# trigger -- without this, _call_with_retry's outer wrapper would reissue an
# IDENTICAL request at temperature=0.0 and fail the same way every time.
_VISION_MAX_TOKENS = 4096
_VISION_MAX_TOKENS_RETRY = 8192


def render_pages_to_images(pdf_path: Path, dpi: int = 200, pdftoppm_bin: str = "pdftoppm") -> list[bytes]:
    """Rasterizes every page of pdf_path to PNG bytes, in page order, via
    pdftoppm -- no OCR, no text extraction. Follows the same
    resolve_binary-able-external-tool and glob-then-sort conventions
    evaluation/scripts/clean_scanned_pdf.py already uses for pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        result = subprocess.run(
            [pdftoppm_bin, "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed on {pdf_path}: {result.stderr}")
        matches = sorted(Path(tmp).glob("page-*.png"))
        if not matches:
            raise RuntimeError(f"pdftoppm produced no output for {pdf_path}")
        return [p.read_bytes() for p in matches]


async def vision_extract_toc_entries(pdf_path: Path, model: str, client: Any, *, pdftoppm_bin: str = "pdftoppm") -> list[TocEntry]:
    """Renders every page of pdf_path and asks a vision-capable model
    (via an already-constructed openai.AsyncOpenAI-shaped `client`, model
    id `model`) to read the table of contents directly from the images.
    Same return shape as llm_extract_toc_entries, sharing its item-parsing
    tolerance logic (_toc_items_to_entries).

    Deliberately RAISES on any failure (network error, malformed JSON)
    rather than catching and returning [] the way llm_extract_toc_entries
    does -- that swallowing made the text pipeline's _call_with_retry
    wrapper dead code (llm_extract_toc_entries never actually raised to
    it). Here, the caller's retry wrapper does real work.

    Escalates max_tokens once (_VISION_MAX_TOKENS -> _VISION_MAX_TOKENS_RETRY)
    if the first response doesn't parse as a complete JSON array, mirroring
    segmentation.py's _extract_with_retry -- otherwise a truncated response
    would fail identically on every one of the caller's retry attempts
    (same images, same prompt, temperature=0.0)."""
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > _MAX_VISION_PAGES:
        raise ValueError(f"{pdf_path}: {page_count} pages exceeds vision-extraction cap of {_MAX_VISION_PAGES}")
    images = render_pages_to_images(pdf_path, pdftoppm_bin=pdftoppm_bin)
    content: list[dict] = [{"type": "text", "text": _VISION_TOC_EXTRACTION_PROMPT}]
    for image_bytes in images:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages = [{"role": "user", "content": content}]

    last_error: Exception | None = None
    for max_tokens in (_VISION_MAX_TOKENS, _VISION_MAX_TOKENS_RETRY):
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        try:
            items = parse_json_array(raw)
            return _toc_items_to_entries(items)
        except Exception as exc:  # noqa: BLE001 -- any parse failure triggers the escalation retry
            last_error = exc
    raise last_error
