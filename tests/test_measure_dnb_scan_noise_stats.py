"""Unit tests for evaluation/scripts/measure_dnb_scan_noise_stats.py's
ALTO-statistics helpers, against the same style of hand-built fixture used
by tests/test_alto_scan_noise.py."""

import tempfile
import unittest
from pathlib import Path

from evaluation.scripts.measure_dnb_scan_noise_stats import (
    body_line_dispersion_ratios,
    contrast_ratios,
    summarize,
)

# Page 1: one title line (24.0) and three body lines -- two at exactly
# 10.0 (giving statistics.mode an unambiguous winner) and one at 10.3
# (a body-like line within the +/-10% dispersion band). Page 2 is
# intentionally blank (no TextLine at all) to confirm empty pages are
# skipped by both measurements, matching layout_features.py's convention.
_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Styles>
    <TextStyle ID="title" FONTFAMILY="sans" FONTSIZE="24.0"/>
    <TextStyle ID="body" FONTFAMILY="serif" FONTSIZE="10.0"/>
    <TextStyle ID="body_jit" FONTFAMILY="serif" FONTSIZE="10.3"/>
  </Styles>
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p1_w1" CONTENT="Chapter One" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p1_w2" CONTENT="Text A" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="48" VPOS="215" WIDTH="340" HEIGHT="12">
            <String ID="p1_w3" CONTENT="Text B" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t4" HPOS="48" VPOS="230" WIDTH="340" HEIGHT="12">
            <String ID="p1_w4" CONTENT="Text C" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body_jit"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="500" HEIGHT="600">
      <PrintSpace/>
    </Page>
  </Layout>
</alto>
"""


class TestContrastRatios(unittest.TestCase):
    def test_one_ratio_per_non_empty_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.alto.xml"
            path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
            ratios = contrast_ratios(path)
        self.assertEqual(len(ratios), 1)
        self.assertAlmostEqual(ratios[0], 2.4, places=3)  # 24.0 / 10.0 modal


class TestBodyLineDispersionRatios(unittest.TestCase):
    def test_excludes_title_includes_body_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.alto.xml"
            path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
            ratios = body_line_dispersion_ratios(path)
        self.assertEqual(len(ratios), 3)  # two 10.0 lines + one 10.3 line; title excluded
        for ratio in ratios:
            self.assertLessEqual(abs(ratio - 1.0), 0.1)
        self.assertAlmostEqual(max(ratios), 1.03, places=3)


class TestSummarize(unittest.TestCase):
    def test_computes_expected_statistics(self):
        stats = summarize([1.0, 2.0, 3.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["median"], 2.0)
        self.assertAlmostEqual(stats["stdev"], 1.0)

    def test_empty_input(self):
        self.assertEqual(summarize([]), {"count": 0})


if __name__ == "__main__":
    unittest.main()
