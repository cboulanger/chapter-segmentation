"""Unit tests for evaluation/scripts/generate_dnb_toc_ground_truth.py's
pure logic. The real PDF-reading/vision-LLM-calling main() is exercised
manually against the real corpus with a real KISSKI_API_KEY -- see design
spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pypdf import PdfWriter

from chapter_segmentation.segmentation import TocEntry
from evaluation.kisski import KisskiModel
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _call_with_retry,
    _load_cached_llm_entries,
    _run_book,
    _run_book_entries,
    _select_best_models,
    _write_cached_llm_entries,
)


def _entry(title: str, page: int, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=-1, authors=authors)


class TestLlmCacheRoundTrip(unittest.TestCase):
    def test_round_trips_entries_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0, authors=("Jane Author",)),
                TocEntry(title="Bibliographie", printed_page_number=-1, source_page_index=1),
            ]
            self.assertIsNone(_load_cached_llm_entries(cache_dir, "book1", "model-a"))
            _write_cached_llm_entries(cache_dir, "book1", "model-a", entries)
            loaded = _load_cached_llm_entries(cache_dir, "book1", "model-a")
            self.assertEqual(loaded, entries)

    def test_round_trip_preserves_printed_roman(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Vorwort", printed_page_number=7, source_page_index=0, printed_roman=True),
            ]
            _write_cached_llm_entries(cache_dir, "book2", "model-a", entries)
            loaded = _load_cached_llm_entries(cache_dir, "book2", "model-a")
            self.assertEqual(loaded, entries)
            self.assertTrue(loaded[0].printed_roman)

    def test_different_models_get_independent_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries_a = [TocEntry(title="From model A", printed_page_number=1, source_page_index=0)]
            entries_b = [TocEntry(title="From model B", printed_page_number=1, source_page_index=0)]
            _write_cached_llm_entries(cache_dir, "book3", "model-a", entries_a)
            _write_cached_llm_entries(cache_dir, "book3", "model-b", entries_b)
            self.assertEqual(_load_cached_llm_entries(cache_dir, "book3", "model-a"), entries_a)
            self.assertEqual(_load_cached_llm_entries(cache_dir, "book3", "model-b"), entries_b)


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


class TestRunBookEntries(unittest.TestCase):
    def test_passing_book_writes_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            corpus_directory.mkdir()
            a = [_entry("Einleitung", 9), _entry("Schluss", 40)]
            b = [_entry("Einleitung", 9), _entry("Schluss", 40)]

            key, passed, reason = _run_book_entries("book1", a, b, corpus_directory)

            self.assertEqual(key, "book1")
            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            gt_path = corpus_directory / "book1.expected.json"
            self.assertTrue(gt_path.exists())
            data = json.loads(gt_path.read_text(encoding="utf-8"))
            self.assertFalse(data["verified"])
            self.assertEqual(len(data["entries"]), 2)

    def test_below_threshold_book_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            corpus_directory.mkdir()
            a = [_entry("Einleitung", 9), _entry("A", 20), _entry("B", 30), _entry("C", 40)]
            b = [_entry("Einleitung", 9)]

            key, passed, reason = _run_book_entries("book2", a, b, corpus_directory)

            self.assertFalse(passed)
            self.assertEqual(reason, "below_threshold")
            self.assertFalse((corpus_directory / "book2.expected.json").exists())

    def test_no_entries_from_either_side_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            corpus_directory.mkdir()

            key, passed, reason = _run_book_entries("book3", [], [], corpus_directory)

            self.assertFalse(passed)
            self.assertEqual(reason, "no_entries")
            self.assertFalse((corpus_directory / "book3.expected.json").exists())


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


def _make_pdf(path: Path) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


_VISION_RESPONSE = (
    '[{"title": "Einleitung", "authors": [], "printed_page_number": "9"}, '
    '{"title": "Schluss", "authors": [], "printed_page_number": "40"}]'
)


class TestRunBook(unittest.IsolatedAsyncioTestCase):
    async def test_calls_each_model_once_and_writes_on_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book1", pdf_path, ("model-a", "model-b"), client, semaphore, corpus_directory, cache_directory,
            )

            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            self.assertEqual(client.chat.completions.create.await_count, 2)
            self.assertTrue((corpus_directory / "book1.expected.json").exists())

    async def test_cached_model_entries_are_reused_without_a_new_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            entries = [_entry("Einleitung", 9), _entry("Schluss", 40)]
            _write_cached_llm_entries(cache_directory, "book2", "model-a", entries)
            _write_cached_llm_entries(cache_directory, "book2", "model-b", entries)
            client = _fake_vision_client(_VISION_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book2", pdf_path, ("model-a", "model-b"), client, semaphore, corpus_directory, cache_directory,
            )

            self.assertTrue(passed)
            client.chat.completions.create.assert_not_called()

    async def test_a_corrupt_pdf_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            bad_pdf = tmp_path / "not-a-pdf.pdf"
            bad_pdf.write_text("this is not a pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book3", bad_pdf, ("model-a", "model-b"), client, semaphore, corpus_directory, cache_directory,
            )

            self.assertFalse(passed)
            self.assertTrue(reason.startswith("error:"))

    async def test_one_model_failing_preserves_the_others_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = MagicMock()
            good_message = MagicMock()
            good_message.content = _VISION_RESPONSE
            good_choice = MagicMock()
            good_choice.message = good_message
            good_response = MagicMock()
            good_response.choices = [good_choice]
            client.chat.completions.create = AsyncMock(
                side_effect=[good_response, RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")]
            )
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book4", pdf_path, ("model-a", "model-b"), client, semaphore, corpus_directory, cache_directory,
            )

            self.assertFalse(passed)
            self.assertTrue(reason.startswith("error:"))
            self.assertIsNotNone(_load_cached_llm_entries(cache_directory, "book4", "model-a"))
            self.assertIsNone(_load_cached_llm_entries(cache_directory, "book4", "model-b"))


class TestSelectBestModels(unittest.TestCase):
    def test_picks_one_from_each_pattern_in_order(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni", demand=0),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=1),
        ]
        self.assertEqual(_select_best_models(models), ["qwen3-omni-30b-a3b-instruct", "gemma-4-31b-it"])

    def test_matches_omni_family_regardless_of_version(self):
        models = [
            KisskiModel(id="qwen5-omni-99b-instruct", name="Qwen Omni next", demand=0),
            KisskiModel(id="gemma-7-40b-it", name="Gemma next", demand=0),
        ]
        self.assertEqual(_select_best_models(models), ["qwen5-omni-99b-instruct", "gemma-7-40b-it"])

    def test_skips_very_busy_candidate_within_a_pattern(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni busy", demand=10),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=0),
        ]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_raises_when_fewer_than_two_vision_models_available(self):
        models = [KisskiModel(id="glm-4.7", name="GLM (not vision)", demand=0)]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_picks_least_busy_among_multiple_matches_in_the_same_pattern(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni A", demand=2),
            KisskiModel(id="qwen4-omni-30b-a3b-instruct", name="Qwen Omni B", demand=0),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=0),
        ]
        self.assertEqual(_select_best_models(models), ["qwen4-omni-30b-a3b-instruct", "gemma-4-31b-it"])
