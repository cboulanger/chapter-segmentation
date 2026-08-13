"""Unit tests for evaluation/scripts/alto_scan_noise.py's deterministic
ALTO-level scan-noise augmentation, against the same style of hand-built
fixture used by tests/test_layout_features.py."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from evaluation.scripts.alto_scan_noise import write_augmented_alto
from evaluation.scripts.layout_features import extract_page_features

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"

_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Styles>
    <TextStyle ID="body" FONTFAMILY="serif" FONTSIZE="10.0" FONTTYPE="serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
    <TextStyle ID="title" FONTFAMILY="sans" FONTSIZE="24.0" FONTTYPE="sans-serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
  </Styles>
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p1_w1" CONTENT="Introduction" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p1_w2" CONTENT="Body" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="48" VPOS="215" WIDTH="340" HEIGHT="12">
            <String ID="p1_w3" CONTENT="More" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t4" HPOS="48" VPOS="230" WIDTH="340" HEIGHT="12">
            <String ID="p1_w4" CONTENT="Text" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""

# Fixture with tiny VPOS/HPOS coordinates (~1-2pt) so that multiplicative jitter
# (±1% of value) contributes ≤0.07pt of noise, while additive per-page offset
# spans ±5pt. Thus the observed mean VPOS shift per page is approximately the
# page's additive offset, cleanly decoupled from multiplicative jitter — isolates
# the per-page offset mechanism and kills mutants that remove or collapse offsets.
_TINY_VPOS_TWO_PAGE_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Styles>
    <TextStyle ID="body" FONTFAMILY="serif" FONTSIZE="10.0" FONTTYPE="serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
    <TextStyle ID="title" FONTFAMILY="sans" FONTSIZE="24.0" FONTTYPE="sans-serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
  </Styles>
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1" HPOS="1" VPOS="1" WIDTH="340" HEIGHT="12">
            <String ID="p1_w1" CONTENT="Line1" HPOS="1" WIDTH="340" HEIGHT="12" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="2" VPOS="2" WIDTH="340" HEIGHT="12">
            <String ID="p1_w2" CONTENT="Line2" HPOS="2" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="1" VPOS="1" WIDTH="340" HEIGHT="12">
            <String ID="p1_w3" CONTENT="Line3" HPOS="1" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t4" HPOS="2" VPOS="2" WIDTH="340" HEIGHT="12">
            <String ID="p1_w4" CONTENT="Line4" HPOS="2" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p2_b1">
          <TextLine ID="p2_t1" HPOS="1" VPOS="1" WIDTH="340" HEIGHT="12">
            <String ID="p2_w1" CONTENT="Page2L1" HPOS="1" WIDTH="340" HEIGHT="12" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p2_t2" HPOS="2" VPOS="2" WIDTH="340" HEIGHT="12">
            <String ID="p2_w2" CONTENT="Page2L2" HPOS="2" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p2_t3" HPOS="1" VPOS="1" WIDTH="340" HEIGHT="12">
            <String ID="p2_w3" CONTENT="Page2L3" HPOS="1" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p2_t4" HPOS="2" VPOS="2" WIDTH="340" HEIGHT="12">
            <String ID="p2_w4" CONTENT="Page2L4" HPOS="2" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


class TestWriteAugmentedAlto(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_path = Path(self._tmp.name)
        self.source = tmp_path / "book.alto.xml"
        self.source.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
        self.output = tmp_path / "book.aug.alto.xml"
        write_augmented_alto(self.source, self.output, "book-key")

    def test_same_seed_is_byte_identical(self):
        second = Path(self._tmp.name) / "again.aug.alto.xml"
        write_augmented_alto(self.source, second, "book-key")
        self.assertEqual(self.output.read_bytes(), second.read_bytes())

    def test_different_key_differs(self):
        other = Path(self._tmp.name) / "other.aug.alto.xml"
        write_augmented_alto(self.source, other, "another-key")
        self.assertNotEqual(self.output.read_bytes(), other.read_bytes())

    def test_page_line_and_string_counts_preserved(self):
        src_root = ET.parse(self.source).getroot()
        aug_root = ET.parse(self.output).getroot()
        for tag in ("Page", "TextLine", "String"):
            self.assertEqual(
                len(list(src_root.iter(_ALTO_NS + tag))),
                len(list(aug_root.iter(_ALTO_NS + tag))),
                tag,
            )

    def test_string_content_is_untouched(self):
        src_texts = [s.get("CONTENT") for s in ET.parse(self.source).getroot().iter(_ALTO_NS + "String")]
        aug_texts = [s.get("CONTENT") for s in ET.parse(self.output).getroot().iter(_ALTO_NS + "String")]
        self.assertEqual(src_texts, aug_texts)

    def test_title_contrast_is_compressed(self):
        # The fixture's title/body ratio is 24/10 = 2.4; augmentation must
        # strictly compress it (alpha < 1 pulls sizes toward the body size).
        src_ratio = extract_page_features(str(self.source))[0]["font_size_max_ratio"]
        aug_ratio = extract_page_features(str(self.output))[0]["font_size_max_ratio"]
        self.assertAlmostEqual(src_ratio, 2.4)
        self.assertLess(aug_ratio, src_ratio)
        self.assertGreater(aug_ratio, 1.0)

    def test_font_sizes_are_jittered_into_clones(self):
        aug_root = ET.parse(self.output).getroot()
        sizes = {s.get("FONTSIZE") for s in aug_root.iter(_ALTO_NS + "TextStyle")}
        # Original 2 styles plus jittered clones -> strictly more distinct sizes.
        self.assertGreater(len(sizes), 2)

    def test_geometry_is_perturbed_but_bounded(self):
        src_lines = list(ET.parse(self.source).getroot().iter(_ALTO_NS + "TextLine"))
        aug_lines = list(ET.parse(self.output).getroot().iter(_ALTO_NS + "TextLine"))
        moved = 0
        for src, aug in zip(src_lines, aug_lines):
            for attr in ("HPOS", "VPOS", "WIDTH"):
                src_val = float(src.get(attr))
                aug_val = float(aug.get(attr))
                self.assertGreaterEqual(aug_val, 0.0)
                self.assertLess(abs(aug_val - src_val), 20.0, attr)
                if aug_val != src_val:
                    moved += 1
        self.assertGreater(moved, 0)


class TestMultiPageAugmentation(unittest.TestCase):
    _KEYS = [f"book-{i}" for i in range(10)]

    def _page_mean_vpos_shifts(self, book_key: str) -> list[float]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "two-page.alto.xml"
            source.write_text(_TINY_VPOS_TWO_PAGE_FIXTURE_ALTO_XML, encoding="utf-8")
            output = tmp_path / "aug.alto.xml"
            write_augmented_alto(source, output, book_key)
            src_pages = list(ET.parse(source).getroot().iter(_ALTO_NS + "Page"))
            aug_pages = list(ET.parse(output).getroot().iter(_ALTO_NS + "Page"))
        shifts = []
        for src_page, aug_page in zip(src_pages, aug_pages):
            src_v = [float(l.get("VPOS")) for l in src_page.iter(_ALTO_NS + "TextLine")]
            aug_v = [float(l.get("VPOS")) for l in aug_page.iter(_ALTO_NS + "TextLine")]
            shifts.append(sum(a - s for s, a in zip(src_v, aug_v)) / len(src_v))
        return shifts

    def test_page_offsets_exist(self):
        # With tiny source coordinates the multiplicative jitter can move a
        # line by at most ~0.07pt, so a mean shift beyond 2pt can only come
        # from the additive per-page offset. Kills the "offsets removed"
        # mutant, which caps every shift near zero.
        max_abs_shift = max(
            abs(shift) for key in self._KEYS for shift in self._page_mean_vpos_shifts(key)
        )
        self.assertGreater(max_abs_shift, 2.0)

    def test_page_offsets_are_independent_per_page(self):
        # If both pages shared one global offset, the two pages' mean shifts
        # would agree to within jitter noise (~0.1pt) for EVERY key. Kills
        # the "single global offset" mutant. Deterministic: fixed keys, no
        # flakiness.
        max_between_page_gap = max(
            abs(shifts[0] - shifts[1])
            for shifts in (self._page_mean_vpos_shifts(key) for key in self._KEYS)
        )
        self.assertGreater(max_between_page_gap, 1.0)


if __name__ == "__main__":
    unittest.main()
