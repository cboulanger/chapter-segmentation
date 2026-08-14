"""Unit tests for evaluation/generate_report.py -- the auto-published,
zero-API-call report covering heuristic, outline, and (if cached) the
best-performing LLM model, one page per evaluation corpus plus a landing
page."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.generate_report import _best_llm_model, generate, generate_corpus


def _expected_json(chapters: list[dict]) -> str:
    return json.dumps({"chapters": chapters})


def _write_corpus_fixture(root: Path, corpus: str, chapters: list[dict], pages: list[str]) -> None:
    cdir = root / corpus
    public_cache_dir = cdir / "public-cache"
    llm_cache_dir = cdir / "llm-cache"
    public_cache_dir.mkdir(parents=True)
    llm_cache_dir.mkdir(parents=True)
    (cdir / "manifest.json").write_text(
        json.dumps({"books": [{"filename": "book-a.pdf", "title": "Book A"}]}), encoding="utf-8",
    )
    (cdir / "book-a.expected.json").write_text(_expected_json(chapters), encoding="utf-8")
    (public_cache_dir / "book-a.pages.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")


class TestBestLlmModel(unittest.TestCase):
    def test_returns_none_with_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test-corpus" / "llm-cache").mkdir(parents=True)
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(
                    _best_llm_model("test-corpus", [("book-a", [{"pdf_start_index": 0, "pdf_end_index": 5}])]),
                )

    def test_picks_the_higher_scoring_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_cache_dir = root / "test-corpus" / "llm-cache"
            llm_cache_dir.mkdir(parents=True)
            expected = [{"pdf_start_index": 0, "pdf_end_index": 5}]
            (llm_cache_dir / "book-a.json").write_text(json.dumps({
                "models": {
                    "good-model": {"chapters": expected, "elapsed_seconds": 1.0, "demand_at_run": 0},
                    "bad-model": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0},
                }
            }), encoding="utf-8")
            with patch("evaluation.harness.CORPUS_ROOT", root):
                best = _best_llm_model("test-corpus", [("book-a", expected)])
            self.assertEqual(best, "good-model")


class TestGenerateCorpus(unittest.TestCase):
    def test_writes_main_report_and_llm_detail_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                wrote = generate_corpus("test-corpus", out_dir)

            self.assertTrue(wrote)
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
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3, "citation_pages": "1-4"}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.\n\n1", "2", "3", "4"])
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Start accuracy", main_html)
            self.assertIn("End accuracy", main_html)

    def test_main_report_footer_includes_generation_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertRegex(main_html, r"Generated on \d{4}-\d{2}-\d{2} from commit")

    def test_returns_false_and_writes_nothing_for_a_corpus_with_no_scorable_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            cdir = root / "empty-corpus"
            cdir.mkdir(parents=True)
            (cdir / "manifest.json").write_text(json.dumps({"books": []}), encoding="utf-8")
            out_dir = tmp_path / "public" / "empty-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root):
                wrote = generate_corpus("empty-corpus", out_dir)

            self.assertFalse(wrote)
            self.assertFalse(out_dir.exists())


class TestGenerate(unittest.TestCase):
    def test_landing_page_links_only_to_corpora_with_scorable_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "scored-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            empty_dir = root / "empty-corpus"
            empty_dir.mkdir(parents=True)
            (empty_dir / "manifest.json").write_text(json.dumps({"books": []}), encoding="utf-8")
            out_dir = tmp_path / "public"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            self.assertTrue((out_dir / "scored-corpus" / "index.html").exists())
            self.assertFalse((out_dir / "empty-corpus" / "index.html").exists())
            landing_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("scored-corpus", landing_html)
            self.assertNotIn("empty-corpus", landing_html)

    def test_landing_page_footer_includes_generation_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "scored-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            out_dir = tmp_path / "public"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            landing_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertRegex(landing_html, r"Generated on \d{4}-\d{2}-\d{2} from commit")


if __name__ == "__main__":
    unittest.main()
