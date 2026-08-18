#!/usr/bin/env python3
"""Measures real font-size contrast and per-line dispersion from the
dnb-toc-only corpus's ALTO output, to calibrate alto_scan_noise.py's
hand-picked synthetic constants (_CONTRAST_ALPHA, _FONT_JITTER) against
real scanned data -- see
docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md
section 3. Every page in this corpus is a confirmed TOC page by
construction (DNB only digitizes the TOC itself), so no per-page
labeling step is needed.

Usage:
    uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py
    uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py --pdfalto-bin ../pdfalto/pdfalto

The printed comparison table is the deliverable regardless of whether it
leads to changing alto_scan_noise.py's constants -- paste it into
evaluation/experiments/toc-classifier-pilot.md by hand as a new follow-up
subsection (this script does not write that file itself, matching every
other evaluation script in this directory)."""

import argparse
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir, list_corpora, load_manifest_books
from evaluation.scripts.alto_scan_noise import _CONTRAST_ALPHA, _FONT_JITTER
from evaluation.scripts.layout_features import extract_page_features
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"

# A line's font size counts as "body-like" (and so contributes a
# dispersion sample) when it sits within this fraction of the page's
# modal (most common) font size -- excludes titles/headers, which is
# exactly what _FONT_JITTER's per-clone noise is meant to model on body
# text, not title text (contrast compression is the separate,
# title-vs-body mechanism measured by contrast_ratios below).
_BODY_BAND = 0.1


def contrast_ratios(alto_path: Path) -> list[float]:
    """One font_size_max_ratio (max/modal font size) sample per non-empty
    page -- the real, uncompressed title/body contrast alto_scan_noise.py's
    _CONTRAST_ALPHA compresses synthetic (born-digital) ALTO toward."""
    features = extract_page_features(str(alto_path))
    return [f["font_size_max_ratio"] for f in features.values() if f["line_count"] > 0]


def body_line_dispersion_ratios(alto_path: Path) -> list[float]:
    """For each non-empty page, the ratio of every body-like line's font
    size (within _BODY_BAND of that page's modal size) to the modal size
    itself -- the real per-line size variation alto_scan_noise.py's
    _FONT_JITTER multiplicatively approximates on synthetic style clones."""
    tree = ET.parse(alto_path)
    root = tree.getroot()
    sizes_by_id = {
        style.get("ID"): float(style.get("FONTSIZE"))
        for style in root.iter(_ALTO_NS + "TextStyle")
        if style.get("ID") and style.get("FONTSIZE")
    }
    ratios: list[float] = []
    for page in root.iter(_ALTO_NS + "Page"):
        page_sizes = []
        for line in page.iter(_ALTO_NS + "TextLine"):
            string = line.find(_ALTO_NS + "String")
            if string is None:
                continue
            refs = (string.get("STYLEREFS") or "").split()
            if refs and refs[0] in sizes_by_id:
                page_sizes.append(sizes_by_id[refs[0]])
        if not page_sizes:
            continue
        modal = statistics.mode(page_sizes)
        if modal <= 0:
            continue
        ratios.extend(
            size / modal for size in page_sizes if abs(size / modal - 1.0) <= _BODY_BAND
        )
    return ratios


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", default="dnb-toc-only", help="Corpus to measure (default: dnb-toc-only)")
    parser.add_argument("--pdfalto-bin", help="Path to the pdfalto binary (see pdfalto_runner.py)")
    args = parser.parse_args()

    if args.corpus not in list_corpora(include_toc_only=True):
        print(f"Corpus '{args.corpus}' not found (or has no manifest.json).")
        return 1

    pdfalto_bin = resolve_pdfalto_binary(args.pdfalto_bin)
    cdir = corpus_dir(args.corpus)
    cache_dir = cdir / ".layout-cache"

    all_contrast: list[float] = []
    all_dispersion: list[float] = []
    for book in load_manifest_books(args.corpus):
        pdf_path = cdir / book["filename"]
        if not pdf_path.exists():
            print(f"[skip] {book['filename']}: PDF not present locally")
            continue
        alto_path = ensure_alto_xml(pdf_path, cache_dir, pdfalto_bin)
        all_contrast.extend(contrast_ratios(alto_path))
        all_dispersion.extend(body_line_dispersion_ratios(alto_path))

    contrast_stats = summarize(all_contrast)
    dispersion_stats = summarize(all_dispersion)

    print(f"\n=== {args.corpus}: real-scan measurements vs. alto_scan_noise.py constants ===\n")
    print(f"Title/body contrast ratio (font_size_max_ratio, n={contrast_stats.get('count', 0)}):")
    print(f"  measured: {contrast_stats}")
    print(f"  current _CONTRAST_ALPHA range: {_CONTRAST_ALPHA}")
    print(f"\nBody-line font-size dispersion (ratio to page modal size, within +/-{_BODY_BAND:.0%}, n={dispersion_stats.get('count', 0)}):")
    print(f"  measured: {dispersion_stats}")
    print(f"  current _FONT_JITTER range: {_FONT_JITTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
