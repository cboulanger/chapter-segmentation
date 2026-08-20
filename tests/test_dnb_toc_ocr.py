"""Unit tests for evaluation/dnb_toc_ocr.py -- OCR'd-text TOC extraction
for dnb-toc-only's vision+text pairing, see design spec
docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
section 3. _rows_from_alto_xml is tested against a hand-written fixture
ALTO XML file (no real ocrmypdf/pdfalto dependency, matching how
evaluation/dnb_toc_vision.py's own render_pages_to_images test is the
only one of that module's tests that shells out to a real binary).
text_extract_toc_entries is tested with ocr_pages_to_rows mocked out and a
mocked OpenAI-shaped client, no real network call or OCR."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from evaluation.dnb_toc_ocr import _rows_from_alto_xml


_ALTO_NS_URI = "http://www.loc.gov/standards/alto/ns-v3#"


def _write_alto_fixture(path: Path) -> Path:
    # Page 1: a dot-leader TOC line whose title and page number pdfalto's
    # own TextBlock segmentation put in SEPARATE TextBlocks (mirroring the
    # real tesseract failure mode the 2026-08-16 investigation found) but
    # whose VPOS values (100, 102) are within the 8px tolerance -- the row
    # reconstruction must still merge them into one row, sorted by HPOS
    # regardless of which TextBlock each token came from.
    # Page 2: two genuinely separate rows (VPOS 200 and 260, 60px apart --
    # well outside tolerance), each with its title and number already in
    # the same TextLine, must NOT merge into one row.
    # Page 3: an empty PrintSpace (a blank page pdfalto still emits a bare
    # <Page> element for) must produce an empty string, not crash.
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{_ALTO_NS_URI}">
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="400" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1">
            <String CONTENT="Einleitung" HPOS="50" VPOS="100" WIDTH="80" HEIGHT="12"/>
          </TextLine>
        </TextBlock>
        <TextBlock ID="p1_b2">
          <TextLine ID="p1_t2">
            <String CONTENT="9" HPOS="300" VPOS="102" WIDTH="10" HEIGHT="12"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="400" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p2_b1">
          <TextLine ID="p2_t1">
            <String CONTENT="Schluss" HPOS="50" VPOS="200" WIDTH="80" HEIGHT="12"/>
            <String CONTENT="40" HPOS="300" VPOS="200" WIDTH="10" HEIGHT="12"/>
          </TextLine>
          <TextLine ID="p2_t2">
            <String CONTENT="Bibliographie" HPOS="50" VPOS="260" WIDTH="80" HEIGHT="12"/>
            <String CONTENT="45" HPOS="300" VPOS="260" WIDTH="10" HEIGHT="12"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page3" PHYSICAL_IMG_NR="3" WIDTH="400" HEIGHT="600">
      <PrintSpace/>
    </Page>
  </Layout>
</alto>
"""
    path.write_text(content, encoding="utf-8")
    return path


class TestRowsFromAltoXml(unittest.TestCase):
    def test_reconstructs_one_row_per_page_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")

            rows = _rows_from_alto_xml(alto_path)

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], "Einleitung 9")

    def test_tokens_across_different_text_blocks_merge_when_vpos_is_close(self):
        # The actual regression this function exists to fix: pdfalto's own
        # TextBlock boundaries put "Einleitung" and "9" in separate blocks,
        # but their VPOS values (100, 102) are within the 8px tolerance --
        # row reconstruction must ignore the TextBlock boundary entirely.
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")
            rows = _rows_from_alto_xml(alto_path)
            self.assertEqual(rows[0], "Einleitung 9")

    def test_rows_further_apart_than_tolerance_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")
            rows = _rows_from_alto_xml(alto_path)
            self.assertEqual(rows[1], "Schluss 40\nBibliographie 45")

    def test_empty_page_produces_an_empty_string_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")
            rows = _rows_from_alto_xml(alto_path)
            self.assertEqual(rows[2], "")
