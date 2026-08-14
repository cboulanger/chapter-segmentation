"""Unit tests for evaluation/refresh_llm_cache.py's pure logic: coverage
computation and cache upserts. The network-calling _main() orchestration
is exercised manually (see evaluation/README.md), not here."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.refresh_llm_cache import _all_cached_model_ids, _fully_covered_model_ids, _upsert_cache


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


if __name__ == "__main__":
    unittest.main()
