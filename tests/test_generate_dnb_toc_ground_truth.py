"""Unit tests for evaluation/scripts/generate_dnb_toc_ground_truth.py's
pure logic. The real PDF-reading/LLM-calling main() is exercised manually
against the real corpus with a real KISSKI_API_KEY -- see design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md."""

import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from chapter_segmentation.segmentation import TocEntry, find_toc_candidates
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _call_with_retry,
    _load_cached_llm_entries,
    _toc_entries_for_scan,
    _write_cached_llm_entries,
)

_TOC_PAGE = (
    "Inhaltsverzeichnis\n"
    "Einleitung ..... 9\n"
    "Zur Soziologie des Rechts ..... 17\n"
    "Schlussbetrachtung ..... 89\n"
)


class TestTocEntriesForScan(unittest.TestCase):
    def test_raw_find_toc_candidates_rejects_realistic_page_numbers_on_a_tiny_pdf(self):
        # Demonstrates the bug: on an unpadded 2-page dnb-toc-only-shaped
        # PDF, _TOC_MAX_PAGE_NUMBER_RATIO (2.0) caps plausible page numbers
        # at 2*2=4 -- every real entry above that (9, 17, 89) is rejected.
        pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
        self.assertEqual(find_toc_candidates(pages), [])

    def test_padded_wrapper_recovers_the_same_entries(self):
        pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
        entries = _toc_entries_for_scan(pages)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].title, "Einleitung")
        self.assertEqual(entries[0].printed_page_number, 9)
        self.assertEqual(entries[2].printed_page_number, 89)


class TestLlmCacheRoundTrip(unittest.TestCase):
    def test_round_trips_entries_through_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0, authors=("Jane Author",)),
                TocEntry(title="Bibliographie", printed_page_number=-1, source_page_index=1),
            ]
            self.assertIsNone(_load_cached_llm_entries(cache_dir, "book1"))
            _write_cached_llm_entries(cache_dir, "book1", entries)
            loaded = _load_cached_llm_entries(cache_dir, "book1")
            self.assertEqual(loaded, entries)

    def test_round_trip_preserves_printed_roman(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(
                    title="Vorwort", printed_page_number=7, source_page_index=0,
                    printed_roman=True,
                ),
            ]
            _write_cached_llm_entries(cache_dir, "book2", entries)
            loaded = _load_cached_llm_entries(cache_dir, "book2")
            self.assertEqual(loaded, entries)
            self.assertTrue(loaded[0].printed_roman)


class TestCallWithRetry(unittest.IsolatedAsyncioTestCase):
    async def test_returns_first_success(self):
        coro_fn = AsyncMock(return_value="ok")
        result = await _call_with_retry(coro_fn, sleep=AsyncMock())
        self.assertEqual(result, "ok")
        coro_fn.assert_awaited_once()

    async def test_retries_then_succeeds(self):
        coro_fn = AsyncMock(side_effect=[RuntimeError("boom"), "ok"])
        result = await _call_with_retry(coro_fn, attempts=3, sleep=AsyncMock())
        self.assertEqual(result, "ok")
        self.assertEqual(coro_fn.await_count, 2)

    async def test_raises_after_exhausting_attempts(self):
        coro_fn = AsyncMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            await _call_with_retry(coro_fn, attempts=2, sleep=AsyncMock())
        self.assertEqual(coro_fn.await_count, 2)
