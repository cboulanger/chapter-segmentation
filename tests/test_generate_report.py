"""Unit tests for evaluation/generate_report.py -- the auto-published,
zero-API-call report covering heuristic, outline, and (if cached) the
best-performing LLM model."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.generate_report import _best_llm_model, generate


def _expected_json(chapters: list[dict]) -> str:
    return json.dumps({"chapters": chapters})


class TestBestLlmModel(unittest.TestCase):
    def test_returns_none_with_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.generate_report.LLM_CACHE_DIR", Path(tmp)):
                self.assertIsNone(_best_llm_model([("book-a", [{"pdf_start_index": 0, "pdf_end_index": 5}])]))

    def test_picks_the_higher_scoring_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            expected = [{"pdf_start_index": 0, "pdf_end_index": 5}]
            (cache_dir / "book-a.json").write_text(json.dumps({
                "models": {
                    "good-model": {"chapters": expected, "elapsed_seconds": 1.0, "demand_at_run": 0},
                    "bad-model": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0},
                }
            }), encoding="utf-8")
            with patch("evaluation.generate_report.LLM_CACHE_DIR", cache_dir):
                best = _best_llm_model([("book-a", expected)])
            self.assertEqual(best, "good-model")


class TestGenerate(unittest.TestCase):
    def test_writes_main_report_and_llm_detail_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_dir = tmp_path / "evaluation"
            public_cache_dir = eval_dir / "public-cache"
            llm_cache_dir = eval_dir / "llm-cache"
            public_cache_dir.mkdir(parents=True)
            llm_cache_dir.mkdir(parents=True)
            out_dir = tmp_path / "public"

            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            (eval_dir / "book-a.expected.json").write_text(_expected_json(chapters), encoding="utf-8")
            (public_cache_dir / "book-a.pages.json").write_text(
                json.dumps({"pages": ["Introduction\nBody text.", "more", "more", "more"]}), encoding="utf-8",
            )
            book = {"filename": "book-a.pdf", "title": "Book A"}

            with patch("evaluation.harness.EVAL_DIR", eval_dir), \
                 patch("evaluation.harness.PUBLIC_CACHE_DIR", public_cache_dir), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]), \
                 patch("evaluation.generate_report.LLM_CACHE_DIR", llm_cache_dir), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            self.assertTrue((out_dir / "index.html").exists())
            self.assertTrue((out_dir / "llm" / "index.html").exists())
            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("book-a", main_html)
            self.assertIn("N/A", main_html)  # outline has no cache entry in this fixture
            llm_html = (out_dir / "llm" / "index.html").read_text(encoding="utf-8")
            self.assertIn("No cached LLM results yet", llm_html)

    def test_main_report_includes_citation_accuracy_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_dir = tmp_path / "evaluation"
            public_cache_dir = eval_dir / "public-cache"
            llm_cache_dir = eval_dir / "llm-cache"
            public_cache_dir.mkdir(parents=True)
            llm_cache_dir.mkdir(parents=True)
            out_dir = tmp_path / "public"

            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3, "citation_pages": "1-4"}]
            (eval_dir / "book-a.expected.json").write_text(_expected_json(chapters), encoding="utf-8")
            (public_cache_dir / "book-a.pages.json").write_text(
                json.dumps({"pages": ["Introduction\nBody text.\n\n1", "2", "3", "4"]}), encoding="utf-8",
            )
            book = {"filename": "book-a.pdf", "title": "Book A"}

            with patch("evaluation.harness.EVAL_DIR", eval_dir), \
                 patch("evaluation.harness.PUBLIC_CACHE_DIR", public_cache_dir), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]), \
                 patch("evaluation.generate_report.LLM_CACHE_DIR", llm_cache_dir), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Start accuracy", main_html)
            self.assertIn("End accuracy", main_html)


if __name__ == "__main__":
    unittest.main()
