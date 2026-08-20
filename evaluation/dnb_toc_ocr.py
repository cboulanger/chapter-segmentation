"""OCR'd-text TOC extraction for dnb-toc-only's vision+text-model pairing
-- feeds a text-only LLM the book's TOC pages reconstructed as plain text
via ocrmypdf + pdfalto, instead of vision_extract_toc_entries' page
images. See design spec
docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
section 3. Structurally parallel to evaluation/dnb_toc_vision.py, reusing
its cache_path/load_cached_llm_entries/write_cached_llm_entries directly
(both extraction paths share one cache, keyed by (book, model) -- see
that module's own "kind" field for how a cached entry records which
extraction path produced it)."""

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries
from evaluation.inference_endpoints import OpenAICompatibleLLMClient
from evaluation.scripts import pdfalto_runner

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"

# VPOS tolerance (ALTO points/px) for clustering <String> tokens into one
# reconstructed row -- see design spec section 3 and
# docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section
# 1b, which found pdfalto's own <TextBlock>/<TextLine> segmentation groups
# a dot-leader TOC's title column and page-number column into SEPARATE
# blocks (all titles, then all page numbers) rather than one block per
# printed line -- exactly the failure this raw-<String>-position
# reclustering avoids by ignoring ALTO's own TextBlock/TextLine nesting
# entirely and re-deriving rows purely from token geometry.
_ROW_VPOS_TOLERANCE = 8.0


def _rows_from_alto_xml(alto_path: Path) -> list[str]:
    """One reading-order-reconstructed text block per ALTO <Page>, in page
    order. Clusters every <String> token on a page by VPOS (tolerance
    _ROW_VPOS_TOLERANCE) into rows, sorts each row's tokens by HPOS, and
    joins them with a single space -- deliberately ignores ALTO's own
    <TextBlock>/<TextLine> nesting (see _ROW_VPOS_TOLERANCE's docstring for
    why trusting it doesn't work for this corpus's dot-leader TOCs). A page
    with no <String> tokens at all (e.g. a blank page pdfalto still emits
    an empty <Page> element for) produces an empty string, not an error --
    ocr_pages_to_rows's caller needs one entry per page to keep its
    per-page-list shape aligned with render_pages_to_images'."""
    root = ET.parse(alto_path).getroot()
    page_rows: list[str] = []
    for page in root.iter(_ALTO_NS + "Page"):
        tokens = sorted(
            (
                (float(string.get("VPOS", "0")), float(string.get("HPOS", "0")), string.get("CONTENT", ""))
                for string in page.iter(_ALTO_NS + "String")
                if string.get("CONTENT")
            ),
            key=lambda token: (token[0], token[1]),
        )
        clusters: list[list[tuple[float, float, str]]] = []
        for token in tokens:
            if clusters and token[0] - clusters[-1][0][0] <= _ROW_VPOS_TOLERANCE:
                clusters[-1].append(token)
            else:
                clusters.append([token])
        rows = [
            " ".join(content for _, _, content in sorted(cluster, key=lambda t: t[1]))
            for cluster in clusters
        ]
        page_rows.append("\n".join(rows))
    return page_rows


_TESSDATA_BEST_DIR_ENV_VAR = "TESSDATA_BEST_DIR"


def _resolve_tessdata_best_env(languages: tuple[str, ...] = ("deu", "eng")) -> dict[str, str] | None:
    """Resolves an optional subprocess environment override pointing
    ocrmypdf/tesseract at a tessdata_best directory instead of whatever
    ships by default (Homebrew's tesseract-lang formula ships
    tessdata_fast only -- there is no Homebrew formula for tessdata_best,
    so this is opt-in via the TESSDATA_BEST_DIR environment variable after
    a manual per-language download from
    https://github.com/tesseract-ocr/tessdata_best -- see
    evaluation/README.md's "Building dnb-toc-only ground truth"). Returns
    None (no env override -- ocrmypdf uses whatever tessdata is already on
    PATH/its default location) when the variable isn't set at all; purely
    opt-in, no default guessed. When it IS set, validates the directory
    actually contains every requested language's .traineddata file and
    raises RuntimeError naming exactly what's missing if not -- a
    misconfigured explicit request should fail loudly with an actionable
    message, not silently fall back to the default or surface as a
    cryptic tesseract error deep inside a subprocess (same
    raise-naming-what's-wrong convention
    evaluation/inference_endpoints.py's resolve_endpoint_from_env already
    established for a similar env-var-driven setup step)."""
    directory = os.environ.get(_TESSDATA_BEST_DIR_ENV_VAR)
    if not directory:
        return None
    if not Path(directory).is_dir():
        raise RuntimeError(f"{_TESSDATA_BEST_DIR_ENV_VAR}={directory} does not exist or is not a directory")
    missing = [lang for lang in languages if not (Path(directory) / f"{lang}.traineddata").exists()]
    if missing:
        raise RuntimeError(
            f"{_TESSDATA_BEST_DIR_ENV_VAR}={directory} is missing traineddata for: {', '.join(missing)} -- "
            f"download from https://github.com/tesseract-ocr/tessdata_best"
        )
    return {**os.environ, "TESSDATA_PREFIX": directory}


def ocr_pages_to_rows(pdf_path: Path, *, pdfalto_bin: str | None = None) -> list[str]:
    """Forces fresh OCR on pdf_path (ocrmypdf --force-ocr, unconditionally
    -- this corpus's PDFs are pre-filtered to 1-3 TOC pages, so re-OCRing
    even an already-text-layered PDF is cheap and keeps behavior uniform
    regardless of the source PDF's own text layer quality), then runs
    pdfalto and reconstructs reading order via _rows_from_alto_xml. Returns
    one string per page, in page order -- the same per-page-list shape
    render_pages_to_images (evaluation/dnb_toc_vision.py) returns for
    images, so the vision and text extraction paths stay visually parallel
    in any calling code. `pdfalto_bin` is passed straight through to
    pdfalto_runner.resolve_pdfalto_binary -- None (the default) resolves
    via the PDFALTO_BIN environment variable, then a bare "pdfalto" on
    PATH; pdfalto is a sibling checkout, not on PATH by default (see
    CLAUDE.local.md/evaluation/CLAUDE.md's pdfalto notes). Raises
    RuntimeError if ocrmypdf exits non-zero -- propagates to the caller
    exactly like any other extraction failure, no special-casing."""
    resolved_pdfalto_bin = pdfalto_runner.resolve_pdfalto_binary(pdfalto_bin)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ocr_pdf_path = tmp_dir / f"{pdf_path.stem}.ocr.pdf"
        result = subprocess.run(
            ["ocrmypdf", "--force-ocr", "-l", "deu+eng", str(pdf_path), str(ocr_pdf_path)],
            capture_output=True, text=True, env=_resolve_tessdata_best_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"ocrmypdf failed on {pdf_path}: {result.stderr}")
        alto_path = pdfalto_runner.ensure_alto_xml(ocr_pdf_path, tmp_dir, resolved_pdfalto_bin)
        return _rows_from_alto_xml(alto_path)
