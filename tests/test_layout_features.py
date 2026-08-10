"""Unit tests for evaluation/scripts/layout_features.py's ALTO-XML-to-
feature-vector parsing, against a small hand-built two-page fixture (one
TOC-shaped page, one chapter-opening-shaped page) with hand-computed
expected values."""

import tempfile
import unittest
from pathlib import Path

from evaluation.scripts.layout_features import extract_page_features

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
          <TextLine ID="p1_t1" HPOS="50" VPOS="100" WIDTH="100" HEIGHT="12">
            <String ID="p1_w1" CONTENT="Intro" HPOS="50" WIDTH="80" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w2" CONTENT="5" HPOS="130" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="50" VPOS="120" WIDTH="120" HEIGHT="12">
            <String ID="p1_w3" CONTENT="ChapterTwo" HPOS="50" WIDTH="100" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w4" CONTENT="12" HPOS="150" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="50" VPOS="140" WIDTH="140" HEIGHT="12">
            <String ID="p1_w5" CONTENT="ChapterThree" HPOS="50" WIDTH="120" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w6" CONTENT="99" HPOS="170" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p2_b1">
          <TextLine ID="p2_t1" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p2_w1" CONTENT="Introduction" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p2_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p2_w2" CONTENT="This" HPOS="48" WIDTH="30" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w3" CONTENT="is" HPOS="80" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w4" CONTENT="text" HPOS="102" WIDTH="290" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p2_t3" HPOS="48" VPOS="215" WIDTH="340" HEIGHT="12">
            <String ID="p2_w5" CONTENT="More" HPOS="48" WIDTH="40" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w6" CONTENT="body" HPOS="90" WIDTH="40" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w7" CONTENT="content" HPOS="132" WIDTH="260" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


class TestExtractPageFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.alto_path = Path(self._tmp.name) / "fixture.alto.xml"
        self.alto_path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
        self.features = extract_page_features(str(self.alto_path))

    def test_returns_zero_based_page_indices(self):
        self.assertEqual(set(self.features.keys()), {0, 1})

    def test_toc_like_page_features(self):
        f = self.features[0]
        self.assertEqual(f["line_count"], 3.0)
        self.assertAlmostEqual(f["width_mean"], 120.0)
        self.assertAlmostEqual(f["width_var"], 400.0)
        self.assertAlmostEqual(f["left_margin_mean"], 50.0)
        self.assertAlmostEqual(f["left_margin_var"], 0.0)
        self.assertEqual(f["trailing_number_fraction"], 1.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 1.0)
        self.assertEqual(f["top_block_is_large_font"], 0.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 100 / 600)
        self.assertAlmostEqual(f["line_density"], 3 / 600)

    def test_chapter_opening_page_features(self):
        f = self.features[1]
        self.assertEqual(f["line_count"], 3.0)
        self.assertAlmostEqual(f["width_mean"], 830 / 3)
        self.assertAlmostEqual(f["width_var"], 12033.333333333334, places=2)
        self.assertAlmostEqual(f["left_margin_mean"], 296 / 3)
        self.assertAlmostEqual(f["left_margin_var"], 7701.333333333333, places=2)
        self.assertEqual(f["trailing_number_fraction"], 0.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 2.4)
        self.assertEqual(f["top_block_is_large_font"], 1.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 50 / 600)


if __name__ == "__main__":
    unittest.main()
