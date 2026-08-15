"""Unit tests for evaluation/scripts/generate_dnb_toc_ground_truth.py's
pure logic. The real PDF-reading/LLM-calling main() is exercised manually
against the real corpus with a real KISSKI_API_KEY -- see design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from chapter_segmentation.llm import LLMClient
from chapter_segmentation.segmentation import TocEntry, find_toc_candidates
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _call_with_retry,
    _load_cached_llm_entries,
    _run_book_pages,
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


_LLM_TOC_RESPONSE = (
    '[{"title": "Einleitung", "authors": [], "printed_page_number": 9}, '
    '{"title": "Zur Soziologie des Rechts", "authors": [], "printed_page_number": 17}, '
    '{"title": "Schlussbetrachtung", "authors": [], "printed_page_number": 89}]'
)

# pages_need_ocr (src/chapter_segmentation/segmentation.py) treats a page as
# "substantial" only above 500 characters -- a threshold tuned for full-length
# books. A bare one-line stand-in for a dnb-toc-only scan's colophon page (as
# used by TestTocEntriesForScan above) falls well under that, so a two-page
# [_TOC_PAGE, colophon] book would spuriously trip pages_need_ocr's "almost no
# text layer" branch before _run_book_pages ever reaches the gate logic. This
# longer, multi-line-but-still-non-TOC-shaped stand-in clears the substantial-
# content and degenerate-newline checks (like a real digitized colophon page
# would) without containing anything find_toc_candidates could mistake for a
# TOC line.
_NOT_TOC_FILLER_PAGE = (
    "Digitalisiert durch die Deutsche Nationalbibliothek im Rahmen des Projekts "
    "zur retrospektiven Digitalisierung von Bibliotheksbestaenden.\n"
    "Diese Seite enthaelt keine inhaltlichen Angaben zum Werk, sondern dient "
    "ausschliesslich der technischen Dokumentation des Scanvorgangs sowie der "
    "Bereitstellung urheberrechtlicher Hinweise fuer die Nutzerinnen und Nutzer "
    "der digitalen Bibliothek.\n"
    "Weitere Informationen zum Digitalisierungsprojekt finden sich auf der "
    "Webseite der Deutschen Nationalbibliothek.\n"
    "Alle Rechte vorbehalten.\n"
)


def _fake_llm(response: str):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


class TestRunBookPages(unittest.IsolatedAsyncioTestCase):
    async def test_passing_book_writes_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            pages = [_TOC_PAGE, _NOT_TOC_FILLER_PAGE]
            llm = _fake_llm(_LLM_TOC_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "9783899718188", pages, llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertEqual(key, "9783899718188")
            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            gt_path = corpus_directory / "9783899718188.expected.json"
            self.assertTrue(gt_path.exists())
            data = json.loads(gt_path.read_text(encoding="utf-8"))
            self.assertFalse(data["verified"])
            self.assertEqual(len(data["entries"]), 3)
            self.assertEqual(data["entries"][0]["printed_page_number"], "9")

    async def test_needs_ocr_book_is_skipped_without_calling_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            llm = _fake_llm(_LLM_TOC_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "empty-book", ["", ""], llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertFalse(passed)
            self.assertEqual(reason, "needs_ocr")
            llm.generate.assert_not_called()
            self.assertFalse((corpus_directory / "empty-book.expected.json").exists())

    async def test_below_threshold_book_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            pages = [_TOC_PAGE, _NOT_TOC_FILLER_PAGE]
            # LLM disagrees with almost everything the heuristic found.
            llm = _fake_llm('[{"title": "Ganz andere Sache", "authors": [], "printed_page_number": 200}]')
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "disagreeing-book", pages, llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertFalse(passed)
            self.assertEqual(reason, "below_threshold")
            self.assertFalse((corpus_directory / "disagreeing-book.expected.json").exists())

    async def test_cached_llm_entries_are_reused_without_a_new_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            pages = [_TOC_PAGE, _NOT_TOC_FILLER_PAGE]
            heuristic_entries = _toc_entries_for_scan(pages)
            _write_cached_llm_entries(cache_directory, "cached-book", heuristic_entries)
            llm = _fake_llm(_LLM_TOC_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "cached-book", pages, llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertTrue(passed)
            llm.generate.assert_not_called()
