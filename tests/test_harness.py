"""Unit tests for evaluation/harness.py -- the pages-loading logic
shared by the pytest accuracy harness and the evaluation scripts. The
extraction and cache primitives are patched; what's under test is the
routing: healthy pages pass through, OCR-shaped pages come from the eval
OCR cache, and a cache miss returns None (caller skips the book)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chapter_segmentation.evidence.types import ChapterCandidate
from evaluation.harness import (
    analysis_pages_for,
    available_public_books,
    chapter_bounds_errors,
    list_corpora,
    outline_candidate_from_dict,
    outline_candidate_to_dict,
    public_outline_candidates_for,
    public_pages_for,
)

_HEALTHY_PAGES = ["Zeile\n" * 200] * 40
_OCR_PAGES = ["ocr text\n" * 100] * 40


class TestListCorpora(unittest.TestCase):
    def test_lists_only_subfolders_with_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus-a").mkdir()
            (root / "corpus-a" / "manifest.json").write_text('{"books": []}', encoding="utf-8")
            (root / "corpus-b").mkdir()
            (root / "corpus-b" / "manifest.json").write_text('{"books": []}', encoding="utf-8")
            (root / "not-a-corpus").mkdir()  # no manifest.json
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertEqual(list_corpora(), ["corpus-a", "corpus-b"])

    def test_returns_empty_list_when_corpus_root_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.harness.CORPUS_ROOT", Path(tmp) / "does-not-exist"):
                self.assertEqual(list_corpora(), [])


class TestChapterBoundsErrors(unittest.TestCase):
    def test_no_errors_for_valid_non_overlapping_chapters(self):
        chapters = [
            {"pdf_start_index": 0, "pdf_end_index": 4},
            {"pdf_start_index": 5, "pdf_end_index": 9},
        ]
        self.assertEqual(chapter_bounds_errors(chapters, total_pages=10), [])

    def test_flags_start_after_end(self):
        chapters = [{"pdf_start_index": 5, "pdf_end_index": 2}]
        errors = chapter_bounds_errors(chapters)
        self.assertEqual(len(errors), 1)
        self.assertIn("start>end", errors[0])

    def test_flags_overlap_between_chapters(self):
        chapters = [
            {"pdf_start_index": 0, "pdf_end_index": 10},
            {"pdf_start_index": 8, "pdf_end_index": 15},
        ]
        errors = chapter_bounds_errors(chapters)
        self.assertEqual(len(errors), 1)
        self.assertIn("overlap", errors[0])

    def test_flags_end_at_or_past_total_pages_only_when_given(self):
        chapters = [{"pdf_start_index": 0, "pdf_end_index": 10}]
        self.assertEqual(
            chapter_bounds_errors(chapters, total_pages=10),
            ["end>=total_pages(10): (0, 10)"],
        )
        self.assertEqual(chapter_bounds_errors(chapters, total_pages=None), [])

    def test_reports_every_problem_not_just_the_first(self):
        chapters = [
            {"pdf_start_index": 5, "pdf_end_index": 2},   # start>end
            {"pdf_start_index": 3, "pdf_end_index": 20},  # end>=total_pages, and overlaps the next range
            {"pdf_start_index": 6, "pdf_end_index": 8},
        ]
        errors = chapter_bounds_errors(chapters, total_pages=10)
        self.assertEqual(len(errors), 3)


class TestAnalysisPagesFor(unittest.TestCase):
    def test_returns_extracted_pages_when_text_layer_is_usable(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=(_HEALTHY_PAGES, False)), \
             patch("evaluation.harness.load_cached_ocr") as mock_cache:
            pages = analysis_pages_for("test-corpus", b"%PDF-fake")
        self.assertEqual(pages, _HEALTHY_PAGES)
        mock_cache.assert_not_called()

    def test_returns_cached_ocr_pages_for_ocr_shaped_input(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=([""] * 300, False)), \
             patch("evaluation.harness.load_cached_ocr", return_value={"detected_language": "deu", "pages": _OCR_PAGES}):
            pages = analysis_pages_for("test-corpus", b"%PDF-fake")
        self.assertEqual(pages, _OCR_PAGES)

    def test_returns_none_on_ocr_cache_miss(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=([""] * 300, False)), \
             patch("evaluation.harness.load_cached_ocr", return_value=None):
            self.assertIsNone(analysis_pages_for("test-corpus", b"%PDF-fake"))

    def test_returns_none_when_cached_ocr_pages_are_still_degenerate(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=([""] * 300, False)), \
             patch("evaluation.harness.load_cached_ocr", return_value={"detected_language": "deu", "pages": [""] * 300}):
            self.assertIsNone(analysis_pages_for("test-corpus", b"%PDF-fake"))


class TestAvailablePublicBooks(unittest.TestCase):
    def test_yields_books_with_a_cache_entry_and_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "test-corpus"
            public_cache_dir = cdir / "public-cache"
            public_cache_dir.mkdir(parents=True)
            (cdir / "9999999.expected.json").write_text("{}", encoding="utf-8")
            (public_cache_dir / "9999999.pages.json").write_text(
                json.dumps({"pages": ["a"]}), encoding="utf-8",
            )
            book = {"filename": "9999999.pdf", "title": "Test Book"}
            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]):
                results = available_public_books("test-corpus")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], "9999999")

    def test_skips_books_with_no_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "test-corpus"
            public_cache_dir = cdir / "public-cache"
            public_cache_dir.mkdir(parents=True)
            (cdir / "9999999.expected.json").write_text("{}", encoding="utf-8")
            book = {"filename": "9999999.pdf", "title": "Test Book"}
            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]):
                results = available_public_books("test-corpus")
            self.assertEqual(results, [])


class TestPublicPagesFor(unittest.TestCase):
    def test_returns_pages_for_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_cache_dir = root / "test-corpus" / "public-cache"
            public_cache_dir.mkdir(parents=True)
            (public_cache_dir / "9999999.pages.json").write_text(
                json.dumps({"pages": ["redacted page text"]}), encoding="utf-8",
            )
            with patch("evaluation.harness.CORPUS_ROOT", root):
                pages = public_pages_for("test-corpus", "9999999")
            self.assertEqual(pages, ["redacted page text"])

    def test_returns_none_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test-corpus" / "public-cache").mkdir(parents=True)
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(public_pages_for("test-corpus", "9999999"))


class TestOutlineCandidateSerialization(unittest.TestCase):
    def test_round_trips_all_fields(self):
        candidate = ChapterCandidate(
            title="Introduction", authors=("Jane Author",), printed_page_number=1,
            pdf_page_index=5, chapter_doi="10.1/x", source="outline", metadata_confidence=0.9,
        )
        restored = outline_candidate_from_dict(outline_candidate_to_dict(candidate))
        self.assertEqual(restored, candidate)

    def test_round_trips_defaults(self):
        candidate = ChapterCandidate(title="Introduction", pdf_page_index=5)
        restored = outline_candidate_from_dict(outline_candidate_to_dict(candidate))
        self.assertEqual(restored, candidate)


class TestPublicOutlineCandidatesFor(unittest.TestCase):
    def test_returns_candidates_for_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_cache_dir = root / "test-corpus" / "public-cache"
            public_cache_dir.mkdir(parents=True)
            candidate = ChapterCandidate(title="Introduction", pdf_page_index=5, source="outline")
            (public_cache_dir / "9999999.outline.json").write_text(
                json.dumps({"candidates": [outline_candidate_to_dict(candidate)]}), encoding="utf-8",
            )
            with patch("evaluation.harness.CORPUS_ROOT", root):
                candidates = public_outline_candidates_for("test-corpus", "9999999")
            self.assertEqual(candidates, [candidate])

    def test_returns_none_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test-corpus" / "public-cache").mkdir(parents=True)
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(public_outline_candidates_for("test-corpus", "9999999"))


if __name__ == "__main__":
    unittest.main()
