import shutil
import unittest
from unittest.mock import patch

import pytest

from chapter_segmentation.ocr_backends.tesseract import TesseractOcrBackend


class TestTesseractOcrBackendConstructor(unittest.TestCase):
    def test_raises_actionable_error_when_binary_missing(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                TesseractOcrBackend()
        self.assertIn("apt-get install tesseract-ocr", str(ctx.exception))

    @unittest.skipUnless(shutil.which("tesseract"), "tesseract binary not installed")
    def test_constructs_when_binary_present(self):
        TesseractOcrBackend()  # must not raise


@pytest.mark.integration
class TestTesseractOcrBackendRealOcr(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(shutil.which("tesseract"), "tesseract binary not installed")
    async def test_ocrs_a_real_rendered_page(self):
        import fitz  # pymupdf

        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        page.insert_text((50, 100), "HELLO WORLD", fontsize=24)
        pdf_bytes = doc.tobytes()
        doc.close()

        backend = TesseractOcrBackend()
        pages = await backend.ocr_pdf_pages(pdf_bytes, language="eng")

        self.assertEqual(len(pages), 1)
        self.assertIn("HELLO", pages[0].upper())


if __name__ == "__main__":
    unittest.main()
