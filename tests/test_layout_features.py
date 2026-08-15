"""Unit tests for evaluation/scripts/layout_features.py's ALTO-XML-to-
feature-vector parsing, against a small hand-built two-page fixture (one
TOC-shaped page, one chapter-opening-shaped page) with hand-computed
expected values."""

import tempfile
import unittest
from pathlib import Path

from evaluation.scripts.layout_features import (
    FEATURE_NAMES,
    PAGE_FEATURE_NAMES,
    _is_heading_line,
    add_book_context_features,
    extract_page_features,
)

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
        self.assertAlmostEqual(f["width_mean"], 120.0 / 500)
        self.assertAlmostEqual(f["width_var"], 400.0 / 500**2)
        self.assertAlmostEqual(f["left_margin_mean"], 50.0 / 500)
        self.assertAlmostEqual(f["left_margin_var"], 0.0 / 500**2)
        self.assertEqual(f["trailing_number_fraction"], 1.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 1.0)
        self.assertEqual(f["top_block_is_large_font"], 0.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 100 / 600)
        self.assertAlmostEqual(f["line_density"], 3 / 600)

    def test_chapter_opening_page_features(self):
        f = self.features[1]
        self.assertEqual(f["line_count"], 3.0)
        self.assertAlmostEqual(f["width_mean"], (830 / 3) / 500)
        self.assertAlmostEqual(f["width_var"], 12033.333333333334 / 500**2, places=6)
        self.assertAlmostEqual(f["left_margin_mean"], (296 / 3) / 500)
        self.assertAlmostEqual(f["left_margin_var"], 7701.333333333333 / 500**2, places=6)
        self.assertEqual(f["trailing_number_fraction"], 0.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 2.4)
        self.assertEqual(f["top_block_is_large_font"], 1.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 50 / 600)


# Regression fixture for the index-misalignment bug in
# top_block_is_large_font: a blank line with no resolvable font size
# (no String child) is interleaved with lines that do have one. The large-
# font line (VPOS=50, within the top fifth of a 600-tall page) sits after
# the blank line (VPOS=200) in document order, so a positionally-misaligned
# lookup into the unfiltered VPOS list would incorrectly read the blank
# line's VPOS=200 instead of the title line's VPOS=50, wrongly reporting
# top_block_is_large_font=0.0 instead of the correct 1.0.
_MISALIGNMENT_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Styles>
    <TextStyle ID="body" FONTFAMILY="serif" FONTSIZE="10.0" FONTTYPE="serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
    <TextStyle ID="title" FONTFAMILY="sans" FONTSIZE="24.0" FONTTYPE="sans-serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
  </Styles>
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1" HPOS="50" VPOS="200" WIDTH="100" HEIGHT="12"/>
          <TextLine ID="p1_t2" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p1_w1" CONTENT="Introduction" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="48" VPOS="300" WIDTH="340" HEIGHT="12">
            <String ID="p1_w2" CONTENT="Body" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t4" HPOS="48" VPOS="320" WIDTH="340" HEIGHT="12">
            <String ID="p1_w3" CONTENT="More" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


class TestTopBlockLargeFontIndexAlignment(unittest.TestCase):
    def test_uses_vpos_of_the_line_that_actually_has_the_max_font_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            alto_path = Path(tmp_dir) / "misalignment.alto.xml"
            alto_path.write_text(_MISALIGNMENT_FIXTURE_ALTO_XML, encoding="utf-8")
            features = extract_page_features(str(alto_path))

        f = features[0]
        self.assertAlmostEqual(f["font_size_max_ratio"], 2.4)
        self.assertEqual(f["top_block_is_large_font"], 1.0)


# Fixture for the roman-numeral trailing-token matcher: exercises a real
# digit page number, a real roman numeral, and two ordinary English words
# that happen to be spelled entirely with roman-numeral letters (i, v, x,
# l, c, d, m) but are not roman numerals -- these must not be counted.
_TRAILING_TOKEN_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Styles>
    <TextStyle ID="body" FONTFAMILY="serif" FONTSIZE="10.0" FONTTYPE="serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
  </Styles>
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1" HPOS="50" VPOS="100" WIDTH="100" HEIGHT="12">
            <String ID="p1_w1" CONTENT="Preface" HPOS="50" WIDTH="80" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w2" CONTENT="5" HPOS="130" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="50" VPOS="120" WIDTH="100" HEIGHT="12">
            <String ID="p1_w3" CONTENT="Appendix" HPOS="50" WIDTH="80" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w4" CONTENT="xii" HPOS="130" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="50" VPOS="140" WIDTH="100" HEIGHT="12">
            <String ID="p1_w5" CONTENT="The" HPOS="50" WIDTH="30" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w6" CONTENT="mix" HPOS="80" WIDTH="50" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t4" HPOS="50" VPOS="160" WIDTH="100" HEIGHT="12">
            <String ID="p1_w7" CONTENT="Say" HPOS="50" WIDTH="30" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w8" CONTENT="hello" HPOS="80" WIDTH="50" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


class TestTrailingNumberFraction(unittest.TestCase):
    def test_only_true_digits_and_true_roman_numerals_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            alto_path = Path(tmp_dir) / "trailing_tokens.alto.xml"
            alto_path.write_text(_TRAILING_TOKEN_FIXTURE_ALTO_XML, encoding="utf-8")
            features = extract_page_features(str(alto_path))

        # "5" (digit) and "xii" (real roman numeral) count; "mix" and
        # "hello" (ordinary English words, not numerals) do not.
        self.assertEqual(features[0]["trailing_number_fraction"], 0.5)


class TestIsHeadingLine(unittest.TestCase):
    def test_keyword_headings_match_in_all_three_languages(self):
        for text in ("Chapter 3", "KAPITEL 12", "Chapitre IV", "Part 2",
                     "Teil 1", "Partie 3", "§ 5", "Chapter", "Teil"):
            self.assertTrue(_is_heading_line(text), text)

    def test_bare_numbers_and_roman_numerals_match(self):
        for text in ("3", "12.", "IV", "xii", "1998"):
            self.assertTrue(_is_heading_line(text), text)

    def test_ordinary_text_and_roman_lookalikes_do_not_match(self):
        # "mix"/"did" are spelled entirely with roman-numeral letters but are
        # not roman numerals -- the existing _TRAILING_NUMERAL_RE grammar
        # rejects them and this feature must inherit that.
        for text in ("The Origins of Law", "mix", "did", "Introduction",
                     "Partial results", "Chapters and verses", ""):
            self.assertFalse(_is_heading_line(text), text)

    def test_keyword_followed_by_roman_lookalike_word_does_not_match(self):
        # The strict roman grammar must also apply to the token AFTER a
        # heading keyword, not only to bare tokens.
        for text in ("Chapter mix", "Chapter did", "Kapitel mild", "Part civic"):
            self.assertFalse(_is_heading_line(text), text)


class TestNewPageLocalFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.alto_path = Path(self._tmp.name) / "fixture.alto.xml"
        self.alto_path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
        self.features = extract_page_features(str(self.alto_path))

    def test_last_text_vpos_fraction(self):
        # Page 0: lowest line bottom = VPOS 140 + HEIGHT 12 = 152.
        self.assertAlmostEqual(self.features[0]["last_text_vpos_fraction"], 152 / 600)
        # Page 1: lowest line bottom = VPOS 215 + HEIGHT 12 = 227.
        self.assertAlmostEqual(self.features[1]["last_text_vpos_fraction"], 227 / 600)

    def test_top_line_heading_match_is_zero_for_both_fixture_pages(self):
        # Page 0's top line reads "Intro 5", page 1's reads "Introduction"
        # -- neither is a bare number, roman numeral, or heading keyword.
        self.assertEqual(self.features[0]["top_line_heading_match"], 0.0)
        self.assertEqual(self.features[1]["top_line_heading_match"], 0.0)

    def test_raw_font_intermediates_are_emitted(self):
        self.assertAlmostEqual(self.features[0]["_max_font_size"], 10.0)
        self.assertAlmostEqual(self.features[0]["_modal_font_size"], 10.0)
        self.assertAlmostEqual(self.features[1]["_max_font_size"], 24.0)
        self.assertAlmostEqual(self.features[1]["_modal_font_size"], 10.0)


def _synthetic_page(line_count: float, last_vpos: float, max_font: float, modal_font: float) -> dict:
    page = {name: 0.0 for name in PAGE_FEATURE_NAMES}
    page["line_count"] = line_count
    page["last_text_vpos_fraction"] = last_vpos
    page["_max_font_size"] = max_font
    page["_modal_font_size"] = modal_font
    return page


class TestAddBookContextFeatures(unittest.TestCase):
    def setUp(self):
        # Three-page book: a sparse chapter-opening-like page between two
        # dense body pages. Line counts 30/8/20 -> median 20; modal fonts
        # 10/10/12 -> body font median 10.
        self.pages = {
            0: _synthetic_page(30.0, 0.95, 10.0, 10.0),
            1: _synthetic_page(8.0, 0.4, 24.0, 10.0),
            2: _synthetic_page(20.0, 0.9, 12.0, 12.0),
        }
        self.result = add_book_context_features(self.pages, total_pages=3)

    def test_output_keys_are_exactly_feature_names(self):
        for page in self.result.values():
            self.assertEqual(set(page.keys()), set(FEATURE_NAMES))

    def test_page_zero_sentinels(self):
        self.assertEqual(self.result[0]["prev_last_text_vpos_fraction"], 0.0)
        self.assertEqual(self.result[0]["prev_line_count_rel"], 0.0)

    def test_prev_page_wiring(self):
        self.assertAlmostEqual(self.result[1]["prev_last_text_vpos_fraction"], 0.95)
        self.assertAlmostEqual(self.result[1]["prev_line_count_rel"], 30.0 / 20.0)
        self.assertAlmostEqual(self.result[2]["prev_last_text_vpos_fraction"], 0.4)
        self.assertAlmostEqual(self.result[2]["prev_line_count_rel"], 8.0 / 20.0)

    def test_book_relative_normalization(self):
        self.assertAlmostEqual(self.result[0]["line_count_rel"], 30.0 / 20.0)
        self.assertAlmostEqual(self.result[1]["line_count_rel"], 8.0 / 20.0)
        self.assertAlmostEqual(self.result[1]["font_size_max_ratio_book"], 24.0 / 10.0)
        self.assertAlmostEqual(self.result[2]["font_size_max_ratio_book"], 12.0 / 10.0)

    def test_edge_distance(self):
        # Three-page book (indices 0/1/2): edges are pages 0 and 2, so page 0
        # and page 2 are both distance 0 from their nearer edge, page 1 (the
        # middle) is distance 1 from either edge.
        self.assertEqual(self.result[0]["edge_distance"], 0)
        self.assertEqual(self.result[1]["edge_distance"], 1)
        self.assertEqual(self.result[2]["edge_distance"], 0)

    def test_edge_distance_picks_nearer_edge_in_a_longer_book(self):
        pages = {i: _synthetic_page(10.0, 0.9, 10.0, 10.0) for i in range(10)}
        result = add_book_context_features(pages, total_pages=10)
        self.assertEqual(result[0]["edge_distance"], 0)
        self.assertEqual(result[2]["edge_distance"], 2)
        self.assertEqual(result[5]["edge_distance"], 4)
        self.assertEqual(result[7]["edge_distance"], 2)
        self.assertEqual(result[9]["edge_distance"], 0)

    def test_empty_pages_are_excluded_from_book_medians(self):
        pages = {
            0: _synthetic_page(0.0, 0.0, 0.0, 0.0),  # blank page
            1: _synthetic_page(10.0, 0.9, 10.0, 10.0),
            2: _synthetic_page(20.0, 0.9, 10.0, 10.0),
        }
        result = add_book_context_features(pages, total_pages=3)
        # Median over non-empty pages only: (10+20)/2 = 15, not median(0,10,20)=10.
        self.assertAlmostEqual(result[1]["line_count_rel"], 10.0 / 15.0)
        # A page with no resolvable font gets the neutral 1.0, matching
        # font_size_max_ratio's existing default.
        self.assertEqual(result[0]["font_size_max_ratio_book"], 1.0)

    def test_all_empty_book_falls_back_without_dividing_by_zero(self):
        pages = {0: _synthetic_page(0.0, 0.0, 0.0, 0.0)}
        result = add_book_context_features(pages, total_pages=1)
        self.assertEqual(result[0]["line_count_rel"], 0.0)
        self.assertEqual(result[0]["font_size_max_ratio_book"], 1.0)

    def test_raw_keys_are_stripped(self):
        for page in self.result.values():
            self.assertNotIn("_max_font_size", page)
            self.assertNotIn("_modal_font_size", page)


if __name__ == "__main__":
    unittest.main()
