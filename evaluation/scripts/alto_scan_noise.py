"""Deterministic ALTO-level scan-noise augmentation: rewrites a cached
pdfalto ALTO XML file to look like the OCR output of a degraded scan,
so the born-digital open-access training pool can teach the layout
classifier scan-shaped feature distributions. The three perturbations
each mimic a property measured in the real copyrighted-scans ALTO (see
docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md):
font-size jitter into many near-identical style clones, title/body
contrast compression, and small geometry noise. All randomness is seeded
from the book key, so output is reproducible and cacheable."""

import random
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

_ALTO_NS_URI = "http://www.loc.gov/standards/alto/ns-v3#"
_ALTO_NS = "{" + _ALTO_NS_URI + "}"

_STYLE_CLONES = 4  # jittered copies per original TextStyle
_FONT_JITTER = (0.96, 1.04)  # per-clone multiplicative font-size noise
_CONTRAST_ALPHA = (0.3, 0.7)  # per-book pull of every size toward the body size
_GEOMETRY_JITTER = (0.99, 1.01)  # per-value multiplicative box noise
_PAGE_OFFSET = (-5.0, 5.0)  # per-page global drift, in ALTO points


def write_augmented_alto(source_path: Path, output_path: Path, book_key: str) -> Path:
    """Writes a scan-noise-augmented copy of source_path to output_path.
    Page/line/String structure and all CONTENT text are preserved -- only
    font styles and box geometry change -- so the source book's page labels
    apply to the augmented copy unchanged."""
    rng = random.Random(f"scan-noise:{book_key}")
    ET.register_namespace("", _ALTO_NS_URI)
    tree = ET.parse(source_path)
    root = tree.getroot()

    styles_parent = root.find(_ALTO_NS + "Styles")
    original_styles = (
        list(styles_parent.iter(_ALTO_NS + "TextStyle")) if styles_parent is not None else []
    )
    sizes_by_id = {
        style.get("ID"): float(style.get("FONTSIZE"))
        for style in original_styles
        if style.get("ID") and style.get("FONTSIZE")
    }

    # Body size = usage-weighted modal font size over String style refs,
    # so a heavily-used body style outweighs a rarely-used title style.
    used_sizes = []
    for string in root.iter(_ALTO_NS + "String"):
        refs = (string.get("STYLEREFS") or "").split()
        if refs and refs[0] in sizes_by_id:
            used_sizes.append(sizes_by_id[refs[0]])
    body_size = statistics.mode(used_sizes) if used_sizes else 0.0
    alpha = rng.uniform(*_CONTRAST_ALPHA)

    clone_ids: dict[str, list[str]] = {}
    for style in original_styles:
        style_id = style.get("ID")
        if style_id not in sizes_by_id:
            continue
        compressed = (
            body_size + (sizes_by_id[style_id] - body_size) * alpha
            if body_size > 0
            else sizes_by_id[style_id]
        )
        ids = []
        for i in range(_STYLE_CLONES):
            clone = ET.SubElement(styles_parent, _ALTO_NS + "TextStyle", dict(style.attrib))
            clone_id = f"{style_id}_aug{i}"
            clone.set("ID", clone_id)
            clone.set("FONTSIZE", f"{compressed * rng.uniform(*_FONT_JITTER):.3f}")
            ids.append(clone_id)
        clone_ids[style_id] = ids

    for string in root.iter(_ALTO_NS + "String"):
        refs = (string.get("STYLEREFS") or "").split()
        if refs and refs[0] in clone_ids:
            string.set("STYLEREFS", rng.choice(clone_ids[refs[0]]))

    for page in root.iter(_ALTO_NS + "Page"):
        page_dx = rng.uniform(*_PAGE_OFFSET)
        page_dy = rng.uniform(*_PAGE_OFFSET)
        for line in page.iter(_ALTO_NS + "TextLine"):
            for attr, drift in (("HPOS", page_dx), ("VPOS", page_dy)):
                value = line.get(attr)
                if value is not None:
                    jittered = (float(value) + drift) * rng.uniform(*_GEOMETRY_JITTER)
                    line.set(attr, f"{max(0.0, jittered):.2f}")
            width = line.get("WIDTH")
            if width is not None:
                line.set("WIDTH", f"{float(width) * rng.uniform(*_GEOMETRY_JITTER):.2f}")

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
