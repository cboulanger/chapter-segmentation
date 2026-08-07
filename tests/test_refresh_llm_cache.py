"""Unit tests for evaluation/refresh_llm_cache.py's pure logic: coverage
computation and cache upserts. The network-calling _main() orchestration
is exercised manually (see evaluation/README.md), not here."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.refresh_llm_cache import _fully_covered_model_ids, _upsert_cache


class TestFullyCoveredModelIds(unittest.TestCase):
    def test_no_cache_files_means_nothing_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", Path(tmp)):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), set())

    def test_model_present_in_every_book_is_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for key in ("book-a", "book-b"):
                (cache_dir / f"{key}.json").write_text(
                    json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                    encoding="utf-8",
                )
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), {"model-x"})

    def test_model_missing_from_one_book_is_not_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), set())

    def test_a_book_with_no_cache_file_at_all_means_zero_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), set())


class TestUpsertCache(unittest.TestCase):
    def test_creates_new_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                _upsert_cache("book-a", "model-x", [{"title": "Intro"}], 1.5, demand=0)
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
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                _upsert_cache("book-a", "model-new", [], 2.0, demand=1)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertIn("model-old", data["models"])
            self.assertIn("model-new", data["models"])


if __name__ == "__main__":
    unittest.main()
