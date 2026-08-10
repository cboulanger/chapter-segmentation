"""Unit tests for evaluation/scripts/pdfalto_runner.py's pure logic: binary
resolution and cache-hit/cache-miss behavior. Running the real pdfalto
binary against a real PDF is exercised manually -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary


class TestResolvePdfaltoBinary(unittest.TestCase):
    def test_explicit_arg_wins(self):
        self.assertEqual(resolve_pdfalto_binary("/custom/pdfalto"), "/custom/pdfalto")

    @patch.dict("os.environ", {"PDFALTO_BIN": "/env/pdfalto"}, clear=True)
    def test_env_var_used_when_no_arg(self):
        self.assertEqual(resolve_pdfalto_binary(None), "/env/pdfalto")

    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_to_bare_name(self):
        self.assertEqual(resolve_pdfalto_binary(None), "pdfalto")


class TestEnsureAltoXml(unittest.TestCase):
    def test_runs_pdfalto_on_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "book.pdf"
            pdf_path.write_bytes(b"%PDF-fake")
            cache_dir = Path(tmp) / "cache"

            def fake_run(cmd, capture_output, text):
                Path(cmd[-1]).write_text("<alto/>", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch(
                "evaluation.scripts.pdfalto_runner.subprocess.run", side_effect=fake_run
            ) as mock_run:
                output_path = ensure_alto_xml(pdf_path, cache_dir, "pdfalto")
                self.assertTrue(output_path.exists())
                self.assertEqual(mock_run.call_count, 1)

    def test_skips_pdfalto_on_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "book.pdf"
            pdf_path.write_bytes(b"%PDF-fake")
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            (cache_dir / "book.alto.xml").write_text("<alto/>", encoding="utf-8")

            with patch("evaluation.scripts.pdfalto_runner.subprocess.run") as mock_run:
                output_path = ensure_alto_xml(pdf_path, cache_dir, "pdfalto")
                self.assertTrue(output_path.exists())
                mock_run.assert_not_called()

    def test_raises_on_pdfalto_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "book.pdf"
            pdf_path.write_bytes(b"%PDF-fake")
            cache_dir = Path(tmp) / "cache"

            with patch(
                "evaluation.scripts.pdfalto_runner.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="boom"),
            ):
                with self.assertRaises(RuntimeError):
                    ensure_alto_xml(pdf_path, cache_dir, "pdfalto")


if __name__ == "__main__":
    unittest.main()
