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

_TWO_PAGE_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
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
            <String ID="p1_w1" CONTENT="Chapter" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p1_w2" CONTENT="Text" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p2_b1">
          <TextLine ID="p2_t1" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p2_w1" CONTENT="Next" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p2_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p2_w2" CONTENT="Content" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
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
    def test_per_page_offsets_are_independent_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "two-page.alto.xml"
            source.write_text(_TWO_PAGE_FIXTURE_ALTO_XML, encoding="utf-8")
            out_a = tmp_path / "a.aug.alto.xml"
            out_b = tmp_path / "b.aug.alto.xml"
            write_augmented_alto(source, out_a, "book-key")
            write_augmented_alto(source, out_b, "book-key")
            self.assertEqual(out_a.read_bytes(), out_b.read_bytes())

            src_pages = list(ET.parse(source).getroot().iter(_ALTO_NS + "Page"))
            aug_pages = list(ET.parse(out_a).getroot().iter(_ALTO_NS + "Page"))
            self.assertEqual(len(aug_pages), 2)
            # Each page's lines must drift by a page-specific offset: compute
            # mean VPOS shift per page and require the two pages' shifts to
            # differ (independent per-page drift, not one global offset).
            shifts = []
            for src_page, aug_page in zip(src_pages, aug_pages):
                src_v = [float(l.get("VPOS")) for l in src_page.iter(_ALTO_NS + "TextLine")]
                aug_v = [float(l.get("VPOS")) for l in aug_page.iter(_ALTO_NS + "TextLine")]
                shifts.append(sum(a - s for s, a in zip(src_v, aug_v)) / len(src_v))
            self.assertNotAlmostEqual(shifts[0], shifts[1], places=3)


if __name__ == "__main__":
    unittest.main()
