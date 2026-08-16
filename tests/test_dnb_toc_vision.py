"""Unit/integration tests for evaluation/dnb_toc_vision.py -- vision-LLM
TOC extraction for dnb-toc-only, see design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section 3.
render_pages_to_images is integration-tested against a real (synthetic,
blank) PDF via the real pdftoppm binary -- poppler is already a documented
project dependency (evaluation/README.md). vision_extract_toc_entries is
tested with a mocked OpenAI-shaped client, no real network call."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pypdf import PdfWriter

from evaluation.dnb_toc_vision import (
    _MAX_VISION_PAGES,
    render_pages_to_images,
    vision_extract_toc_entries,
)


def _make_pdf(path: Path, page_count: int) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


class TestRenderPagesToImages(unittest.TestCase):
    def test_renders_one_png_per_page_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 3)
            images = render_pages_to_images(pdf_path)
            self.assertEqual(len(images), 3)
            for image_bytes in images:
                self.assertTrue(image_bytes.startswith(b"\x89PNG"))

    def test_raises_on_a_nonexistent_pdf(self):
        with self.assertRaises(RuntimeError):
            render_pages_to_images(Path("/nonexistent/does-not-exist.pdf"))


def _fake_vision_client(response_text: str):
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


_VISION_RESPONSE = (
    '[{"title": "Einleitung", "authors": [], "printed_page_number": "9"}, '
    '{"title": "Zur Soziologie des Rechts", "authors": ["Jane Author"], "printed_page_number": "17"}]'
)


class TestVisionExtractTocEntries(unittest.IsolatedAsyncioTestCase):
    async def test_parses_response_into_toc_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_vision_client(_VISION_RESPONSE)
            entries = await vision_extract_toc_entries(pdf_path, "some-model", client)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].title, "Einleitung")
            self.assertEqual(entries[0].printed_page_number, 9)
            self.assertEqual(entries[1].authors, ("Jane Author",))

    async def test_sends_one_image_content_block_per_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 2)
            client = _fake_vision_client("[]")
            await vision_extract_toc_entries(pdf_path, "some-model", client)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            content = messages[0]["content"]
            image_blocks = [c for c in content if c["type"] == "image_url"]
            self.assertEqual(len(image_blocks), 2)

    async def test_raises_on_malformed_json_instead_of_swallowing(self):
        # Unlike llm_extract_toc_entries (which catches internally and
        # returns [], making its own _call_with_retry wrapper dead code --
        # see generate_dnb_toc_ground_truth.py), vision_extract_toc_entries
        # deliberately propagates so the caller's retry wrapper is
        # meaningful.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_vision_client("not json at all")
            with self.assertRaises(Exception):
                await vision_extract_toc_entries(pdf_path, "some-model", client)

    async def test_raises_before_any_network_call_when_page_count_exceeds_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", _MAX_VISION_PAGES + 1)
            client = _fake_vision_client("[]")
            with self.assertRaises(ValueError):
                await vision_extract_toc_entries(pdf_path, "some-model", client)
            client.chat.completions.create.assert_not_called()
