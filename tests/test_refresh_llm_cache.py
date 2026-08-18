"""Unit tests for evaluation/refresh_llm_cache.py's pure logic: coverage
computation and cache upserts. The network-calling _main() orchestration
is exercised manually (see evaluation/README.md), not here."""

import asyncio
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from evaluation.inference_endpoints import ModelEndpoint
from evaluation.refresh_llm_cache import (
    _OpenAICompatibleLLMClient,
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _model_and_client_for_endpoint,
    _process_model,
    _run_book_for_model,
    _upsert_cache,
)


class TestOpenAICompatibleLLMClient(unittest.IsolatedAsyncioTestCase):
    async def test_uses_the_given_client_not_a_new_one(self):
        fake_client = unittest.mock.MagicMock()
        message = unittest.mock.MagicMock()
        message.content = "hello"
        choice = unittest.mock.MagicMock()
        choice.message = message
        response = unittest.mock.MagicMock()
        response.choices = [choice]
        fake_client.chat.completions.create = unittest.mock.AsyncMock(return_value=response)

        llm_client = _OpenAICompatibleLLMClient(model="model-x", client=fake_client)
        result = await llm_client.generate("prompt", max_tokens=10, temperature=0.0)

        self.assertEqual(result, "hello")
        fake_client.chat.completions.create.assert_awaited_once_with(
            model="model-x", messages=[{"role": "user", "content": "prompt"}], max_tokens=10, temperature=0.0,
        )


class TestModelAndClientForEndpoint(unittest.TestCase):
    def test_wraps_endpoint_with_demand_zero_and_the_endpoints_own_client(self):
        fake_client = unittest.mock.MagicMock()
        endpoint = ModelEndpoint(label="MPCDF_A", model_id="Qwen/Qwen3-VL-30B", client=fake_client)

        model, llm_client = _model_and_client_for_endpoint(endpoint)

        self.assertEqual(model.id, "Qwen/Qwen3-VL-30B")
        self.assertEqual(model.demand, 0)
        self.assertIsInstance(llm_client, _OpenAICompatibleLLMClient)
        self.assertIs(llm_client._client, fake_client)


class TestFullyCoveredModelIds(unittest.TestCase):
    def test_no_cache_files_means_nothing_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set(),
            )

    def test_model_present_in_every_book_is_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for key in ("book-a", "book-b"):
                (cache_dir / f"{key}.json").write_text(
                    json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                    encoding="utf-8",
                )
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x"},
            )

    def test_model_missing_from_one_book_is_not_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set(),
            )

    def test_a_book_with_no_cache_file_at_all_means_zero_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set(),
            )

    def test_books_from_different_corpus_cache_dirs_are_covered_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_a_dir = Path(tmp) / "corpus-a"
            corpus_b_dir = Path(tmp) / "corpus-b"
            corpus_a_dir.mkdir()
            corpus_b_dir.mkdir()
            (corpus_a_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (corpus_b_dir / "book-b.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _fully_covered_model_ids([(corpus_a_dir, "book-a"), (corpus_b_dir, "book-b")]), {"model-x"},
            )


class TestAllCachedModelIds(unittest.TestCase):
    def test_no_cache_files_means_no_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertEqual(_all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set())

    def test_unions_model_ids_across_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(
                json.dumps({"models": {"model-y": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x", "model-y"},
            )

    def test_model_present_in_only_one_book_still_counts(self):
        # Unlike _fully_covered_model_ids (intersection), this is a union
        # -- a model doesn't need to be cached for EVERY book to count.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            self.assertEqual(
                _all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x"},
            )

    def test_a_book_with_no_cache_file_at_all_is_skipped_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x"},
            )


class TestHasCachedEntry(unittest.TestCase):
    def test_no_cache_file_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_has_cached_entry(Path(tmp), "book-a", "model-x"))

    def test_cache_file_without_model_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            self.assertFalse(_has_cached_entry(cache_dir, "book-a", "model-x"))

    def test_cache_file_with_model_returns_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertTrue(_has_cached_entry(cache_dir, "book-a", "model-x"))

    def test_malformed_cache_file_is_treated_as_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text("{not valid json", encoding="utf-8")
            self.assertFalse(_has_cached_entry(cache_dir, "book-a", "model-x"))


class TestCallWithRetry(unittest.IsolatedAsyncioTestCase):
    async def test_succeeds_on_first_try_without_sleeping(self):
        sleeps = []

        async def sleep(delay):
            sleeps.append(delay)

        async def fn():
            return "ok"

        result = await _call_with_retry(fn, sleep=sleep)
        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [])

    async def test_retries_then_succeeds(self):
        attempts = []
        sleeps = []

        async def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient")
            return "ok"

        async def sleep(delay):
            sleeps.append(delay)

        result = await _call_with_retry(fn, attempts=3, base_delay=1.0, sleep=sleep)
        self.assertEqual(result, "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    async def test_raises_after_exhausting_all_attempts(self):
        async def fn():
            raise RuntimeError("permanent")

        async def sleep(delay):
            pass

        with self.assertRaises(RuntimeError):
            await _call_with_retry(fn, attempts=3, base_delay=0.01, sleep=sleep)


class TestProcessModel(unittest.IsolatedAsyncioTestCase):
    async def test_calls_worker_once_per_book_entry(self):
        calls = []

        async def worker(corpus, manifest_key, cache_dir):
            calls.append((corpus, manifest_key, cache_dir))

        book_entries = [("corpus-a", "book-1", Path("/x")), ("corpus-a", "book-2", Path("/x"))]
        await _process_model(book_entries, concurrency=4, worker=worker)
        self.assertEqual(len(calls), 2)

    async def test_never_exceeds_concurrency_limit(self):
        in_flight = 0
        max_in_flight = 0

        async def worker(corpus, manifest_key, cache_dir):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

        book_entries = [("c", f"book-{i}", Path("/x")) for i in range(10)]
        await _process_model(book_entries, concurrency=3, worker=worker)
        self.assertLessEqual(max_in_flight, 3)
        self.assertGreater(max_in_flight, 1)

    async def test_a_raising_worker_does_not_cancel_sibling_tasks(self):
        completed = []

        async def worker(corpus, manifest_key, cache_dir):
            if manifest_key == "book-1":
                raise RuntimeError("boom")
            completed.append(manifest_key)

        book_entries = [("c", "book-1", Path("/x")), ("c", "book-2", Path("/x")), ("c", "book-3", Path("/x"))]
        await _process_model(book_entries, concurrency=4, worker=worker)
        self.assertEqual(sorted(completed), ["book-2", "book-3"])


class TestRunBookForModel(unittest.IsolatedAsyncioTestCase):
    def _model(self):
        return SimpleNamespace(id="model-x", demand=0)

    async def test_fill_gaps_skips_when_already_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for") as pages_mock,
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only") as analyze_mock,
            ):
                await _run_book_for_model("corpus-a", "book-a", cache_dir, self._model(), "fill-gaps", llm_client=None)
                pages_mock.assert_not_called()
                analyze_mock.assert_not_called()

    async def test_fill_gaps_processes_when_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            async def fake_analyze(pages, llm_client):
                return {"chapters": [{"title": "Intro"}]}

            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for", return_value=["page text"]),
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only", side_effect=fake_analyze),
            ):
                await _run_book_for_model("corpus-a", "book-a", cache_dir, self._model(), "fill-gaps", llm_client=None)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertIn("model-x", data["models"])

    async def test_top5_reprocesses_even_when_already_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [{"title": "Old"}], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )

            async def fake_analyze(pages, llm_client):
                return {"chapters": [{"title": "New"}]}

            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for", return_value=["page text"]),
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only", side_effect=fake_analyze),
            ):
                await _run_book_for_model("corpus-a", "book-a", cache_dir, self._model(), "top5", llm_client=None)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertEqual(data["models"]["model-x"]["chapters"], [{"title": "New"}])

    async def test_failure_after_retries_is_logged_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            async def always_fails(pages, llm_client):
                raise RuntimeError("boom")

            async def fast_sleep(delay):
                pass

            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for", return_value=["page text"]),
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only", side_effect=always_fails),
            ):
                await _run_book_for_model(
                    "corpus-a", "book-a", cache_dir, self._model(), "fill-gaps", llm_client=None, sleep=fast_sleep,
                )
            self.assertFalse((cache_dir / "book-a.json").exists())


class TestUpsertCache(unittest.TestCase):
    def test_creates_new_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _upsert_cache(cache_dir, "book-a", "model-x", [{"title": "Intro"}], 1.5, demand=0)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertEqual(data["models"]["model-x"]["elapsed_seconds"], 1.5)
            self.assertEqual(data["models"]["model-x"]["demand_at_run"], 0)

    def test_preserves_other_models_when_upserting(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-old": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            _upsert_cache(cache_dir, "book-a", "model-new", [], 2.0, demand=1)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertIn("model-old", data["models"])
            self.assertIn("model-new", data["models"])

    def test_records_a_per_model_generated_at_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _upsert_cache(cache_dir, "book-a", "model-x", [], 1.0, demand=0)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertRegex(data["models"]["model-x"]["generated_at"], r"^\d{4}-\d{2}-\d{2}T")

    def test_a_second_models_generated_at_does_not_overwrite_the_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _upsert_cache(cache_dir, "book-a", "model-old", [], 1.0, demand=0)
            first_data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            first_timestamp = first_data["models"]["model-old"]["generated_at"]

            _upsert_cache(cache_dir, "book-a", "model-new", [], 1.0, demand=0)
            second_data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))

            self.assertEqual(second_data["models"]["model-old"]["generated_at"], first_timestamp)
            self.assertIn("generated_at", second_data["models"]["model-new"])

    def test_no_leftover_tmp_file_after_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _upsert_cache(cache_dir, "book-a", "model-x", [], 1.0, demand=0)
            self.assertEqual(sorted(p.name for p in cache_dir.iterdir()), ["book-a.json"])


if __name__ == "__main__":
    unittest.main()
