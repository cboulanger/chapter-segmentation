"""Unit tests for backend.services.chapter_ocr."""

import io
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from pypdf import PdfWriter

from chapter_segmentation.ocr import (
    detect_language,
    load_cached_ocr,
    ocr_pdf_pages,
    save_ocr_cache,
)


class TestDetectLanguage(unittest.TestCase):
    def test_uses_item_language_field_if_set(self):
        self.assertEqual(detect_language(item_language="de", title="Some Title"), "deu")

    def test_detects_from_title_when_no_item_language(self):
        result = detect_language(item_language=None, title="Einführung in die Zitierweise")
        self.assertEqual(result, "deu")

    def test_falls_back_to_combined_default_when_undetectable(self):
        result = detect_language(item_language=None, title="")
        self.assertEqual(result, "eng+deu+fra+spa")


class TestOcrCache(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            save_ocr_cache(cache_dir, "abc123", detected_language="deu", pages=["page one", "page two"])
            result = load_cached_ocr(cache_dir, "abc123")
            self.assertEqual(result["detected_language"], "deu")
            self.assertEqual(result["pages"], ["page one", "page two"])

    def test_returns_none_when_not_cached(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(load_cached_ocr(Path(tmp), "does-not-exist"))


def _two_page_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestOcrPdfPages(unittest.IsolatedAsyncioTestCase):
    async def test_delegates_to_backend_and_caches_by_content_hash(self):
        pdf_bytes = _two_page_pdf_bytes()
        backend = AsyncMock()
        backend.ocr_pdf_pages.return_value = ["page one text", "page two text"]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            pages = await ocr_pdf_pages(pdf_bytes, backend=backend, cache_dir=cache_dir, language="deu")
            self.assertEqual(pages, ["page one text", "page two text"])
            backend.ocr_pdf_pages.assert_awaited_once_with(pdf_bytes, language="deu")
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)

            # Second call with identical bytes: served from cache, backend untouched.
            pages_again = await ocr_pdf_pages(pdf_bytes, backend=backend, cache_dir=cache_dir, language="deu")
            self.assertEqual(pages_again, ["page one text", "page two text"])
            backend.ocr_pdf_pages.assert_awaited_once()  # still just the one call


if __name__ == "__main__":
    unittest.main()
