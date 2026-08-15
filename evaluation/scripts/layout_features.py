"""Per-page geometric feature extraction from a pdfalto ALTO XML file --
the classifier's input for
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import re
import statistics
import xml.etree.ElementTree as ET

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"
# Digit branch: a plain page number. Roman-numeral branch: a proper
# roman-numeral grammar (not just "made of the right letters" -- that
# would also match ordinary English words like "mix", "did", "civic",
# "mild", "vivid", "livid"). The thousands/"cm" (900) group is
# deliberately omitted: real book front matter is never numbered past the
# low hundreds in roman numerals, and keeping it would let "mix" parse as
# the technically-valid-but-nonsensical M+IX (1009) and pass anyway.
_TRAILING_NUMERAL_RE = re.compile(
    r"^[0-9]{1,4}$|^(?=.)(cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)

# A chapter/part heading keyword (English/German/French), optionally
# followed by an arabic or roman number. The bare-number and bare-roman
# branches of heading detection reuse _TRAILING_NUMERAL_RE so its
# lookalike rejections ("mix", "did", "civic") carry over.
_HEADING_KEYWORD_RE = re.compile(
    r"^(?:chapter|kapitel|chapitre|part|teil|partie|§)\.?(?:\s+(?P<number>\S+))?$",
    re.IGNORECASE,
)


def _is_heading_line(text: str) -> bool:
    """Whether a line of text looks like a chapter/part heading: a heading
    keyword (optionally numbered), a bare arabic number, or a bare roman
    numeral. Content-based, so it survives OCR'd scans whose font metadata
    is unreliable."""
    stripped = text.strip().rstrip(".:").strip()
    if not stripped:
        return False
    if _TRAILING_NUMERAL_RE.match(stripped):
        return True
    match = _HEADING_KEYWORD_RE.match(stripped)
    if match is None:
        return False
    number = match.group("number")
    return number is None or _TRAILING_NUMERAL_RE.match(number) is not None

PAGE_FEATURE_NAMES = [
    "line_count",
    "width_mean",
    "width_var",
    "left_margin_mean",
    "left_margin_var",
    "trailing_number_fraction",
    "font_size_max_ratio",
    "top_block_is_large_font",
    "first_text_vpos_fraction",
    "line_density",
    "last_text_vpos_fraction",
    "top_line_heading_match",
]

# Computed by add_book_context_features (second pass), not by
# extract_page_features -- they need neighboring pages and book-level
# aggregates a single-page parse can't see.
CONTEXT_FEATURE_NAMES = [
    "prev_last_text_vpos_fraction",
    "prev_line_count_rel",
    "line_count_rel",
    "font_size_max_ratio_book",
    "edge_distance",
]

FEATURE_NAMES = PAGE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES


def _parse_font_sizes(root: ET.Element) -> dict[str, float]:
    """Maps each TextStyle ID to its FONTSIZE, from the document's Styles
    block."""
    sizes = {}
    for style in root.iter(_ALTO_NS + "TextStyle"):
        style_id = style.get("ID")
        font_size = style.get("FONTSIZE")
        if style_id and font_size:
            sizes[style_id] = float(font_size)
    return sizes


def _line_font_size(line: ET.Element, font_sizes: dict[str, float]) -> float | None:
    """Font size of a TextLine's first String, or None if unavailable."""
    string = line.find(_ALTO_NS + "String")
    if string is None:
        return None
    style_ref = string.get("STYLEREFS")
    if not style_ref:
        return None
    return font_sizes.get(style_ref.split()[0])


def _font_ratio_and_top_block_flag(
    lines: list[ET.Element], font_sizes: dict[str, float], page_height: float
) -> tuple[float, float, float, float]:
    """Ratio of the page's largest resolvable font size to its modal
    (most common, i.e. body-text) font size, whether the line with
    that largest font size sits in the top fifth of the page (title-block
    signal), and the raw max and modal font sizes. Defaults to
    (1.0, 0.0, 0.0, 0.0) when no line has a resolvable font size.

    Font size and VPOS are tracked together per line (rather than as two
    separately filtered/unfiltered parallel lists) so that a line with no
    resolvable font size -- no String child, no STYLEREFS, or an unknown
    style ID -- can't desynchronize the two and cause the max-font line's
    VPOS to be read from the wrong line."""
    resolvable = [
        (size, float(line.get("VPOS")))
        for line, size in ((line, _line_font_size(line, font_sizes)) for line in lines)
        if size is not None
    ]
    if not resolvable:
        return 1.0, 0.0, 0.0, 0.0

    modal_size = statistics.mode(size for size, _ in resolvable)
    max_size, max_size_vpos = max(resolvable, key=lambda pair: pair[0])
    font_size_max_ratio = max_size / modal_size if modal_size else 1.0
    top_block_is_large_font = float(
        font_size_max_ratio > 1.3 and max_size_vpos < page_height / 5
    )
    return font_size_max_ratio, top_block_is_large_font, max_size, modal_size


def extract_page_features(alto_xml_path: str) -> dict[int, dict[str, float]]:
    """Parses a pdfalto ALTO XML file into a per-page feature dict, keyed by
    0-based PDF page index (ALTO's PHYSICAL_IMG_NR is 1-based). A page with
    no text lines at all gets an all-zero feature vector."""
    tree = ET.parse(alto_xml_path)
    root = tree.getroot()
    font_sizes = _parse_font_sizes(root)

    features: dict[int, dict[str, float]] = {}
    for page in root.iter(_ALTO_NS + "Page"):
        page_index = int(page.get("PHYSICAL_IMG_NR")) - 1
        page_height = float(page.get("HEIGHT"))
        page_width = float(page.get("WIDTH"))
        lines = list(page.iter(_ALTO_NS + "TextLine"))

        if not lines:
            features[page_index] = {
                **{name: 0.0 for name in PAGE_FEATURE_NAMES},
                "_max_font_size": 0.0,
                "_modal_font_size": 0.0,
            }
            continue

        widths = [float(line.get("WIDTH")) / page_width for line in lines]
        left_margins = [float(line.get("HPOS")) / page_width for line in lines]
        vpositions = [float(line.get("VPOS")) for line in lines]

        trailing_hits = 0
        for line in lines:
            strings = line.findall(_ALTO_NS + "String")
            if strings and _TRAILING_NUMERAL_RE.match(strings[-1].get("CONTENT", "").strip()):
                trailing_hits += 1

        font_size_max_ratio, top_block_is_large_font, max_font_size, modal_font_size = (
            _font_ratio_and_top_block_flag(lines, font_sizes, page_height)
        )

        line_bottoms = [
            float(line.get("VPOS")) + float(line.get("HEIGHT", 0.0)) for line in lines
        ]
        top_line = min(lines, key=lambda line: float(line.get("VPOS")))
        top_line_text = " ".join(
            s.get("CONTENT", "") for s in top_line.findall(_ALTO_NS + "String")
        )

        features[page_index] = {
            "line_count": float(len(lines)),
            "width_mean": statistics.mean(widths),
            "width_var": statistics.variance(widths) if len(widths) > 1 else 0.0,
            "left_margin_mean": statistics.mean(left_margins),
            "left_margin_var": statistics.variance(left_margins) if len(left_margins) > 1 else 0.0,
            "trailing_number_fraction": trailing_hits / len(lines),
            "font_size_max_ratio": font_size_max_ratio,
            "top_block_is_large_font": top_block_is_large_font,
            "first_text_vpos_fraction": min(vpositions) / page_height,
            "line_density": len(lines) / page_height,
            "last_text_vpos_fraction": max(line_bottoms) / page_height,
            "top_line_heading_match": float(_is_heading_line(top_line_text)),
            "_max_font_size": max_font_size,
            "_modal_font_size": modal_font_size,
        }

    return features


def add_book_context_features(
    page_features: dict[int, dict[str, float]], total_pages: int
) -> dict[int, dict[str, float]]:
    """Second pass over one book's extract_page_features output: adds the
    CONTEXT_FEATURE_NAMES (previous-page context, book-relative
    normalization, distance to the nearer edge of the book) and strips the
    underscore-prefixed raw intermediates, returning vectors keyed exactly
    by FEATURE_NAMES.

    Book medians are computed over non-empty pages only (a scan's blank
    versos would otherwise drag them toward zero). The book-level body-font
    estimate is the median of per-page modal font sizes -- stable over the
    hundreds of near-identical jittery sizes OCR produces, where a single
    page's mode is arbitrary. Pages with no resolvable font get the neutral
    ratio 1.0, matching font_size_max_ratio's existing default."""
    non_empty = [f for f in page_features.values() if f["line_count"] > 0]
    median_line_count = (
        statistics.median(f["line_count"] for f in non_empty) if non_empty else 1.0
    ) or 1.0
    modal_sizes = [f["_modal_font_size"] for f in non_empty if f["_modal_font_size"] > 0]
    body_font_size = statistics.median(modal_sizes) if modal_sizes else 0.0

    result: dict[int, dict[str, float]] = {}
    for page_index, page in page_features.items():
        prev = page_features.get(page_index - 1)
        out = {name: page[name] for name in PAGE_FEATURE_NAMES}
        out["prev_last_text_vpos_fraction"] = (
            prev["last_text_vpos_fraction"] if prev is not None else 0.0
        )
        out["prev_line_count_rel"] = (
            prev["line_count"] / median_line_count if prev is not None else 0.0
        )
        out["line_count_rel"] = page["line_count"] / median_line_count
        max_font = page["_max_font_size"]
        out["font_size_max_ratio_book"] = (
            max_font / body_font_size if body_font_size > 0 and max_font > 0 else 1.0
        )
        # Absolute (not fractional) page count to the nearer of the book's
        # two edges -- real TOCs sit near the front or, in some traditions
        # (e.g. French-language books), near the back, but front-matter
        # length before a TOC starts doesn't scale with book length the way
        # a fractional distance would assume. An in-chapter mini-TOC has the
        # same local layout signature as a real TOC but sits deep in the
        # book, where this feature is large. Deliberately replaces the
        # earlier monotonic page_position_fraction: a fractional "closer to
        # page 1" feature learns to penalize genuine back-of-book TOCs
        # (rare in training data) hard enough to cancel out this feature's
        # own signal when both are present -- see the 2026-08-13 pilot
        # follow-up notes in evaluation/RESULTS.md.
        out["edge_distance"] = min(page_index, total_pages - 1 - page_index)
        result[page_index] = out
    return result
