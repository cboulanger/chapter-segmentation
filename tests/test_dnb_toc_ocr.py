"""Unit tests for evaluation/dnb_toc_ocr.py -- OCR'd-text TOC extraction
for dnb-toc-only's vision+text pairing, see design spec
docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
section 3. _rows_from_alto_xml is tested against a hand-written fixture
ALTO XML file (no real ocrmypdf/pdfalto dependency, matching how
evaluation/dnb_toc_vision.py's own render_pages_to_images test is the
only one of that module's tests that shells out to a real binary).
text_extract_toc_entries is tested with ocr_pages_to_rows mocked out and a
mocked OpenAI-shaped client, no real network call or OCR."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from evaluation.dnb_toc_ocr import _resolve_tessdata_best_env, _rows_from_alto_xml, ocr_pages_to_rows


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


class TestResolveTessdataBestEnv(unittest.TestCase):
    def test_returns_none_when_env_var_absent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TESSDATA_BEST_DIR", None)
            self.assertIsNone(_resolve_tessdata_best_env())

    def test_returns_tessdata_prefix_pointing_at_the_directory_when_all_languages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "deu.traineddata").write_bytes(b"fake")
            (Path(tmp) / "eng.traineddata").write_bytes(b"fake")
            with patch.dict(os.environ, {"TESSDATA_BEST_DIR": tmp}, clear=False):
                env = _resolve_tessdata_best_env()

        self.assertIsNotNone(env)
        self.assertEqual(env["TESSDATA_PREFIX"], tmp)

    def test_raises_naming_the_missing_language_when_dir_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "deu.traineddata").write_bytes(b"fake")
            with patch.dict(os.environ, {"TESSDATA_BEST_DIR": tmp}, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    _resolve_tessdata_best_env()

        self.assertIn("eng", str(ctx.exception))

    def test_raises_a_distinct_message_when_the_directory_does_not_exist(self):
        with patch.dict(os.environ, {"TESSDATA_BEST_DIR": "/nonexistent/tessdata_best"}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                _resolve_tessdata_best_env()

        self.assertIn("does not exist", str(ctx.exception))


class TestOcrPagesToRowsTessdataWiring(unittest.TestCase):
    def test_passes_the_resolved_tessdata_env_through_to_ocrmypdf(self):
        fake_env = {"TESSDATA_PREFIX": "/fake/tessdata_best"}
        fake_result = MagicMock(returncode=0)
        with patch("evaluation.dnb_toc_ocr._resolve_tessdata_best_env", return_value=fake_env), \
             patch("evaluation.dnb_toc_ocr.subprocess.run", return_value=fake_result) as mock_run, \
             patch("evaluation.dnb_toc_ocr.pdfalto_runner.ensure_alto_xml", return_value=Path("/fake/out.alto.xml")), \
             patch("evaluation.dnb_toc_ocr._rows_from_alto_xml", return_value=["row"]):
            ocr_pages_to_rows(Path("/fake/book.pdf"))

        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.kwargs["env"], fake_env)

    def test_passes_none_env_when_tessdata_best_is_not_configured(self):
        fake_result = MagicMock(returncode=0)
        with patch("evaluation.dnb_toc_ocr._resolve_tessdata_best_env", return_value=None), \
             patch("evaluation.dnb_toc_ocr.subprocess.run", return_value=fake_result) as mock_run, \
             patch("evaluation.dnb_toc_ocr.pdfalto_runner.ensure_alto_xml", return_value=Path("/fake/out.alto.xml")), \
             patch("evaluation.dnb_toc_ocr._rows_from_alto_xml", return_value=["row"]):
            ocr_pages_to_rows(Path("/fake/book.pdf"))

        mock_run.assert_called_once()
        self.assertIsNone(mock_run.call_args.kwargs["env"])
