import json
import subprocess
import sys
import unittest

import pytest
from pypdf import PdfWriter

# This test needs a blank PDF to be OCR'd via the tesseract backend, which
# requires the real `tesseract` binary -- not available by default (see
# ocr_backends tests' convention for the same marker/reasoning).
pytestmark = pytest.mark.integration


class TestCli(unittest.TestCase):
    def test_analyze_a_blank_pdf_runs_end_to_end(self):
        writer = PdfWriter()
        writer.add_blank_page(width=400, height=600)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer.write(f)
            pdf_path = f.name

        result = subprocess.run(
            [sys.executable, "-m", "chapter_segmentation.cli", "analyze", pdf_path],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertIn("chapters", parsed)
        self.assertIn("total_pdf_pages", parsed)


if __name__ == "__main__":
    unittest.main()
