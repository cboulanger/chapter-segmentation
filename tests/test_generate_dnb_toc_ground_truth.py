"""Unit tests for evaluation/scripts/generate_dnb_toc_ground_truth.py's
pure logic. The real PDF-reading/vision-LLM-calling main() is exercised
manually against the real corpus with a real KISSKI_API_KEY -- see design
spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import RateLimitError
from pypdf import PdfWriter

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import load_cached_llm_entries, write_cached_llm_entries
from evaluation.inference_endpoints import ModelEndpoint
from evaluation.kisski import KisskiModel
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _acquire_lock,
    _binding_rate_limit_window,
    _call_with_retry,
    _is_stale_bulk_gate_entry,
    _lock_path,
    _release_lock,
    _resolve_vision_endpoints,
    _retry_after_seconds,
    _run_book,
    _run_book_entries,
    _select_best_models,
    _still_needs_a_decision,
)


def _rate_limit_error(headers: dict) -> RateLimitError:
    return RateLimitError(
        "rate limited",
        response=httpx.Response(
            429, headers=headers, request=httpx.Request("POST", "https://example.com"),
        ),
        body=None,
    )


def _entry(title: str, page: int, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=-1, authors=authors)


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

    async def test_rate_limit_error_gets_the_longer_linear_backoff(self):
        rate_limit_error = RateLimitError(
            "rate limited", response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")), body=None,
        )
        coro_fn = AsyncMock(side_effect=[rate_limit_error, rate_limit_error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_any_call(20.0)
        sleep.assert_any_call(40.0)

    async def test_non_rate_limit_error_keeps_the_short_exponential_backoff(self):
        coro_fn = AsyncMock(side_effect=[RuntimeError("boom"), "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=2, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(2.0)

    async def test_rate_limit_error_with_retry_after_header_sleeps_that_exact_value(self):
        # A "minute" window is inline-retryable -- must use the server's own
        # retry-after value, not the blind rate_limit_delay*attempt formula.
        error = _rate_limit_error({"retry-after": "37", "x-ratelimit-remaining-minute": "0"})
        coro_fn = AsyncMock(side_effect=[error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(37.0)

    async def test_rate_limit_error_with_hour_window_also_retries_inline(self):
        error = _rate_limit_error({"retry-after": "900", "x-ratelimit-remaining-hour": "0"})
        coro_fn = AsyncMock(side_effect=[error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(900.0)

    async def test_rate_limit_error_bound_by_day_window_gives_up_immediately(self):
        # The whole point: a day-scale quota won't reset within this run,
        # so no further attempts should even be made, regardless of
        # `attempts` -- see _call_with_retry's docstring.
        error = _rate_limit_error({"retry-after": "54179", "x-ratelimit-remaining-day": "0"})
        coro_fn = AsyncMock(side_effect=error)
        sleep = AsyncMock()

        with self.assertRaises(RateLimitError):
            await _call_with_retry(coro_fn, attempts=6, sleep=sleep)

        coro_fn.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_rate_limit_error_with_no_headers_falls_back_to_linear_backoff(self):
        error = _rate_limit_error({})
        coro_fn = AsyncMock(side_effect=[error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(20.0)


class TestBindingRateLimitWindow(unittest.TestCase):
    def test_none_when_no_headers(self):
        self.assertIsNone(_binding_rate_limit_window({}))
        self.assertIsNone(_binding_rate_limit_window(None))

    def test_none_when_no_remaining_header_is_zero(self):
        headers = {"x-ratelimit-remaining-day": "5", "x-ratelimit-remaining-minute": "1"}
        self.assertIsNone(_binding_rate_limit_window(headers))

    def test_picks_the_single_zeroed_window(self):
        self.assertEqual(_binding_rate_limit_window({"x-ratelimit-remaining-minute": "0"}), "minute")
        self.assertEqual(_binding_rate_limit_window({"x-ratelimit-remaining-hour": "0"}), "hour")
        self.assertEqual(_binding_rate_limit_window({"x-ratelimit-remaining-day": "0"}), "day")

    def test_prefers_the_longest_window_when_several_are_zeroed(self):
        headers = {
            "x-ratelimit-remaining-minute": "0",
            "x-ratelimit-remaining-hour": "0",
            "x-ratelimit-remaining-day": "0",
        }
        self.assertEqual(_binding_rate_limit_window(headers), "day")


class TestRetryAfterSeconds(unittest.TestCase):
    def test_none_when_absent(self):
        self.assertIsNone(_retry_after_seconds({}))
        self.assertIsNone(_retry_after_seconds(None))

    def test_parses_a_numeric_value(self):
        self.assertEqual(_retry_after_seconds({"retry-after": "54179"}), 54179.0)

    def test_none_for_an_unparseable_value(self):
        self.assertIsNone(_retry_after_seconds({"retry-after": "not-a-number"}))


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
            self.assertEqual(data["source"], "bulk_gate")
            self.assertEqual(len(data["entries"]), 2)
            self.assertIn("skip", data["entries"][0])

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


def _endpoint(model_id: str, client) -> ModelEndpoint:
    return ModelEndpoint(label="test", model_id=model_id, client=client)


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
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book1", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
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
            write_cached_llm_entries(cache_directory, "book2", "model-a", entries)
            write_cached_llm_entries(cache_directory, "book2", "model-b", entries)
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book2", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
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
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book3", bad_pdf, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
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
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book4", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertFalse(passed)
            self.assertTrue(reason.startswith("error:"))
            self.assertIsNotNone(load_cached_llm_entries(cache_directory, "book4", "model-a"))
            self.assertIsNone(load_cached_llm_entries(cache_directory, "book4", "model-b"))

    async def test_semaphore_is_released_during_backoff_sleep(self):
        # Regression test for a real 2026-08-17 batch stall: the semaphore
        # used to wrap the whole retry sequence, so a backoff sleep held a
        # concurrency slot hostage -- if enough books hit RateLimitError
        # around the same time, every slot ended up asleep simultaneously
        # and the batch stalled with zero throughput even though nothing
        # had crashed. It must be released before each sleep so other
        # books can make progress while this one backs off.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = MagicMock()
            client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)
            observed_lock_state_during_sleep = []

            async def spying_sleep(_delay):
                observed_lock_state_during_sleep.append(semaphore.locked())

            await _run_book(
                "book5", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=spying_sleep,
            )

            self.assertTrue(observed_lock_state_during_sleep, "sleep (backoff) was never invoked")
            self.assertTrue(
                all(not locked for locked in observed_lock_state_during_sleep),
                "semaphore was still held during a backoff sleep",
            )

    async def test_two_independent_endpoints_each_get_their_own_client_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client_a = _fake_vision_client(_VISION_RESPONSE)
            client_b = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client_a), _endpoint("model-b", client_b))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book6", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertTrue(passed)
            client_a.chat.completions.create.assert_awaited_once()
            client_b.chat.completions.create.assert_awaited_once()
            self.assertEqual(client_a.chat.completions.create.await_args.kwargs["model"], "model-a")
            self.assertEqual(client_b.chat.completions.create.await_args.kwargs["model"], "model-b")

    async def test_a_book_whose_lock_is_already_held_is_skipped_without_calling_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)
            _acquire_lock(corpus_directory, "book7")  # simulate another process already working on it

            key, passed, reason = await _run_book(
                "book7", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertFalse(passed)
            self.assertEqual(reason, "locked_by_another_process")
            client.chat.completions.create.assert_not_called()

    async def test_the_lock_is_released_after_a_book_finishes_successfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            await _run_book(
                "book8", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertFalse(_lock_path(corpus_directory, "book8").exists())

    async def test_the_lock_is_released_even_when_the_book_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            bad_pdf = tmp_path / "not-a-pdf.pdf"
            bad_pdf.write_text("this is not a pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            await _run_book(
                "book9", bad_pdf, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertFalse(_lock_path(corpus_directory, "book9").exists())


class TestAcquireLock(unittest.TestCase):
    def test_first_acquire_succeeds_and_creates_the_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            self.assertTrue(_acquire_lock(cdir, "book1"))
            self.assertTrue(_lock_path(cdir, "book1").exists())

    def test_second_acquire_of_a_fresh_lock_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            _acquire_lock(cdir, "book1")
            self.assertFalse(_acquire_lock(cdir, "book1"))

    def test_acquire_succeeds_again_after_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            _acquire_lock(cdir, "book1")
            _release_lock(cdir, "book1")
            self.assertTrue(_acquire_lock(cdir, "book1"))

    def test_a_stale_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            _acquire_lock(cdir, "book1", stale_after=10.0)
            old = 1_000_000_000  # arbitrary, far enough in the past to be stale under any stale_after
            os.utime(_lock_path(cdir, "book1"), (old, old))
            self.assertTrue(_acquire_lock(cdir, "book1", stale_after=10.0))

    def test_release_of_a_lock_that_was_never_acquired_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            _release_lock(cdir, "book1")  # must not raise


class TestStillNeedsADecision(unittest.TestCase):
    def test_true_for_a_fresh_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            book = {"filename": "book1.pdf"}
            self.assertTrue(_still_needs_a_decision(book, cdir, set(), set()))

    def test_false_when_held_out_for_eval_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, cdir, {"book1"}, set()))

    def test_false_when_permanently_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, cdir, set(), {"book1"}))

    def test_false_when_expected_json_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            (cdir / "book1.expected.json").write_text("{}", encoding="utf-8")
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, cdir, set(), set()))

    def test_true_for_a_stale_pre_skip_field_bulk_gate_file(self):
        # A bulk_gate file written before the 2026-08-17 extraction-standard
        # change has entries with no "skip" key at all -- it's missing
        # whatever lines the old prompt told the model to omit outright, so
        # it counts as undecided again rather than staying stuck forever.
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            (cdir / "book1.expected.json").write_text(
                json.dumps({"entries": [{"title": "Einleitung", "authors": [], "printed_page_number": "9"}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            book = {"filename": "book1.pdf"}
            self.assertTrue(_still_needs_a_decision(book, cdir, set(), set()))

    def test_false_for_a_current_schema_bulk_gate_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            (cdir / "book1.expected.json").write_text(
                json.dumps({"entries": [{"title": "Einleitung", "authors": [], "printed_page_number": "9",
                                          "skip": False}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, cdir, set(), set()))

    def test_false_for_a_stale_arbitration_file_never_auto_reprocessed(self):
        # Unlike a stale bulk_gate file, a claude_arbitration file went
        # through direct human/Claude review -- it must never be silently
        # overwritten by an automated, unreviewed re-run just because it
        # also predates the "skip" key. Retrofitting it is a deliberate
        # manual task, not this function's job.
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            (cdir / "book1.expected.json").write_text(
                json.dumps({"entries": [{"title": "Einleitung", "authors": [], "printed_page_number": "9"}],
                            "verified": True, "source": "claude_arbitration"}),
                encoding="utf-8",
            )
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, cdir, set(), set()))


class TestIsStaleBulkGateEntry(unittest.TestCase):
    def test_true_when_bulk_gate_entries_lack_skip_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [{"title": "X", "authors": [], "printed_page_number": "1"}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            self.assertTrue(_is_stale_bulk_gate_entry(path))

    def test_false_when_skip_key_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [{"title": "X", "authors": [], "printed_page_number": "1", "skip": False}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            self.assertFalse(_is_stale_bulk_gate_entry(path))

    def test_false_for_non_bulk_gate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [{"title": "X", "authors": [], "printed_page_number": "1"}],
                            "verified": True, "source": "claude_arbitration"}),
                encoding="utf-8",
            )
            self.assertFalse(_is_stale_bulk_gate_entry(path))

    def test_false_for_empty_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [], "verified": False, "source": "bulk_gate"}), encoding="utf-8",
            )
            self.assertFalse(_is_stale_bulk_gate_entry(path))


class TestSelectBestModels(unittest.TestCase):
    def test_picks_one_from_each_pattern_in_order(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni", demand=0),
            KisskiModel(id="qwen3.6-27b", name="Qwen 3.6", demand=1),
        ]
        self.assertEqual(_select_best_models(models), ["qwen3-omni-30b-a3b-instruct", "qwen3.6-27b"])

    def test_matches_omni_family_regardless_of_version(self):
        models = [
            KisskiModel(id="qwen5-omni-99b-instruct", name="Qwen Omni next", demand=0),
            KisskiModel(id="qwen3.6-40b", name="Qwen 3.6 next", demand=0),
        ]
        self.assertEqual(_select_best_models(models), ["qwen5-omni-99b-instruct", "qwen3.6-40b"])

    def test_skips_very_busy_candidate_within_a_pattern(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni busy", demand=10),
            KisskiModel(id="qwen3.6-27b", name="Qwen 3.6", demand=0),
        ]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_raises_when_fewer_than_two_vision_models_available(self):
        models = [KisskiModel(id="glm-4.7", name="GLM (not vision)", demand=0)]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_picks_both_models_from_one_pattern_when_it_alone_has_enough(self):
        # A single pattern with >= count available candidates satisfies
        # count entirely on its own (least-busy first) -- it does NOT fall
        # through to a later pattern just because one exists. Contrast
        # with test_falls_through_to_next_pattern_when_first_has_too_few
        # below, where the first pattern genuinely doesn't have enough.
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni A", demand=2),
            KisskiModel(id="qwen4-omni-30b-a3b-instruct", name="Qwen Omni B", demand=0),
            KisskiModel(id="qwen3.6-27b", name="Qwen 3.6", demand=0),
        ]
        self.assertEqual(
            _select_best_models(models),
            ["qwen4-omni-30b-a3b-instruct", "qwen3-omni-30b-a3b-instruct"],
        )

    def test_falls_through_to_next_pattern_when_first_has_too_few_candidates(self):
        # Only one qwen-omni candidate exists -- not enough to satisfy
        # count=2 alone, so the loop must fall through to the qwen3.6
        # pattern for the second pick, per design spec section 3.1.
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni", demand=0),
            KisskiModel(id="qwen3.6-27b", name="Qwen 3.6 A", demand=2),
            KisskiModel(id="qwen3.6-99b", name="Qwen 3.6 B", demand=0),
        ]
        self.assertEqual(
            _select_best_models(models),
            ["qwen3-omni-30b-a3b-instruct", "qwen3.6-99b"],
        )


class TestResolveVisionEndpoints(unittest.TestCase):
    def test_no_aliases_falls_back_to_kisski_auto_select(self):
        env = {"KISSKI_API_KEY": "kisski-key"}
        with patch.dict(os.environ, env, clear=False), patch(
            "evaluation.scripts.generate_dnb_toc_ground_truth._pick_models",
            return_value=["model-x", "model-y"],
        ) as mock_pick:
            endpoints = _resolve_vision_endpoints(None)

        mock_pick.assert_called_once()
        self.assertEqual(endpoints[0].label, "kisski")
        self.assertEqual(endpoints[0].model_id, "model-x")
        self.assertEqual(endpoints[1].label, "kisski")
        self.assertEqual(endpoints[1].model_id, "model-y")
        self.assertIs(endpoints[0].client, endpoints[1].client)

    def test_exactly_two_aliases_resolved_independently(self):
        env = {
            "MPCDF_A_BASE_URL": "https://a.invalid/v1", "MPCDF_A_API_KEY": "ka", "MPCDF_A_MODEL": "model-a",
            "MPCDF_B_BASE_URL": "https://b.invalid/v1", "MPCDF_B_API_KEY": "kb", "MPCDF_B_MODEL": "model-b",
        }
        with patch.dict(os.environ, env, clear=False):
            endpoints = _resolve_vision_endpoints(["MPCDF_A", "MPCDF_B"])

        self.assertEqual(endpoints[0].label, "MPCDF_A")
        self.assertEqual(endpoints[0].model_id, "model-a")
        self.assertEqual(endpoints[1].label, "MPCDF_B")
        self.assertEqual(endpoints[1].model_id, "model-b")
        self.assertIsNot(endpoints[0].client, endpoints[1].client)

    def test_one_alias_is_a_user_error(self):
        with self.assertRaises(SystemExit):
            _resolve_vision_endpoints(["MPCDF_A"])

    def test_three_aliases_is_a_user_error(self):
        with self.assertRaises(SystemExit):
            _resolve_vision_endpoints(["MPCDF_A", "MPCDF_B", "MPCDF_C"])

    def test_config_file_with_two_tables_resolved_independently(self):
        table_a = (
            "framework_args\t--model=mistralai/Pixtral-12B-2409\n"
            "key\tka\n"
            "url\thttps://a.invalid/v1\n"
        )
        table_b = (
            "framework_args\t--model=Qwen/Qwen3-Omni-30B-A3B-Instruct\n"
            "key\tkb\n"
            "url\thttps://b.invalid/v1\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.txt"
            path.write_text(table_a + "\n" + table_b)

            endpoints = _resolve_vision_endpoints(None, path)

        self.assertEqual(endpoints[0].label, "config:A")
        self.assertEqual(endpoints[0].model_id, "mistralai/Pixtral-12B-2409")
        self.assertEqual(endpoints[1].label, "config:B")
        self.assertEqual(endpoints[1].model_id, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertIsNot(endpoints[0].client, endpoints[1].client)

    def test_config_file_with_one_table_is_a_user_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.txt"
            path.write_text("framework_args\t--model=x\nkey\tk\nurl\thttps://a.invalid/v1\n")

            with self.assertRaises(SystemExit):
                _resolve_vision_endpoints(None, path)
