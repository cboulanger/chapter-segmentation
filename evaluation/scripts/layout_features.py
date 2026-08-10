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

FEATURE_NAMES = [
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
]


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
) -> tuple[float, float]:
    """Ratio of the page's largest resolvable font size to its modal
    (most common, i.e. body-text) font size, and whether the line with
    that largest font size sits in the top fifth of the page (title-block
    signal). Defaults to (1.0, 0.0) when no line has a resolvable font
    size.

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
        return 1.0, 0.0

    modal_size = statistics.mode(size for size, _ in resolvable)
    max_size, max_size_vpos = max(resolvable, key=lambda pair: pair[0])
    font_size_max_ratio = max_size / modal_size if modal_size else 1.0
    top_block_is_large_font = float(
        font_size_max_ratio > 1.3 and max_size_vpos < page_height / 5
    )
    return font_size_max_ratio, top_block_is_large_font


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
            features[page_index] = {name: 0.0 for name in FEATURE_NAMES}
            continue

        widths = [float(line.get("WIDTH")) / page_width for line in lines]
        left_margins = [float(line.get("HPOS")) / page_width for line in lines]
        vpositions = [float(line.get("VPOS")) for line in lines]

        trailing_hits = 0
        for line in lines:
            strings = line.findall(_ALTO_NS + "String")
            if strings and _TRAILING_NUMERAL_RE.match(strings[-1].get("CONTENT", "").strip()):
                trailing_hits += 1

        font_size_max_ratio, top_block_is_large_font = _font_ratio_and_top_block_flag(
            lines, font_sizes, page_height
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
        }

    return features
