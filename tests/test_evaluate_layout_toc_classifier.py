"""Unit tests for evaluation/scripts/evaluate_layout_toc_classifier.py's
pure logic. The real pdfalto-subprocess-driven, real-corpus leave-one-book-out
run is exercised manually -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.scripts.evaluate_layout_toc_classifier import (
    build_feature_table,
    evaluate_leave_one_book_out,
    load_book_corpus,
    select_threshold,
)
from evaluation.scripts.layout_features import FEATURE_NAMES


class TestSelectThreshold(unittest.TestCase):
    def test_no_positives_returns_one(self):
        self.assertEqual(select_threshold([0.1, 0.9], [False, False], recall_target=0.9), 1.0)

    def test_single_positive_needs_its_own_probability(self):
        probs = [0.2, 0.8, 0.5]
        labels = [False, True, False]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.9), 0.8)

    def test_target_below_full_recall_picks_higher_cutoff(self):
        # Four positives at probabilities 0.9, 0.8, 0.7, 0.2 -- targeting
        # 75% recall needs the top 3 (round(0.75*4)=3), so the cutoff is
        # the third-highest positive probability, 0.7.
        probs = [0.9, 0.8, 0.7, 0.2]
        labels = [True, True, True, True]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.75), 0.7)

    def test_rounds_up_not_to_nearest(self):
        # Six positives targeting 90% recall: 0.9*6=5.4, which must round UP
        # to 6 (all of them), not to the nearest integer (5) -- rounding to
        # nearest would only guarantee 5/6 = 83.3% recall, undershooting the
        # 90% target. The correct cutoff is therefore the lowest positive
        # probability, 0.4.
        probs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        labels = [True, True, True, True, True, True]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.90), 0.4)


class TestBuildFeatureTable(unittest.TestCase):
    def test_joins_features_with_labels_per_page(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "chapter_first", "other"]},
        ]

        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}, 2: {"line_count": 3.0}}

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {"book_key": "book-a", "features": {"line_count": 1.0}, "label": "toc"})
        self.assertEqual(
            rows[1], {"book_key": "book-a", "features": {"line_count": 2.0}, "label": "chapter_first"}
        )
        self.assertEqual(rows[2], {"book_key": "book-a", "features": {"line_count": 3.0}, "label": "other"})

    def test_skips_pages_pdfalto_did_not_extract(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "other"]},
        ]
        # pdfalto only produced a feature vector for page 0.
        fake_features = {0: {"line_count": 1.0}}

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(len(rows), 1)

    def test_warns_on_stderr_when_pages_are_skipped(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "other", "toc"]},
        ]
        # Only page 1 ("other") got a feature vector; both "toc" pages are dropped.
        fake_features = {1: {"line_count": 1.0}}

        stderr = io.StringIO()
        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ), contextlib.redirect_stderr(stderr):
            build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        warning = stderr.getvalue()
        self.assertIn("WARNING", warning)
        self.assertIn("toc=2", warning)

    def test_no_warning_when_nothing_is_skipped(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc"]},
        ]
        fake_features = {0: {"line_count": 1.0}}

        stderr = io.StringIO()
        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ), contextlib.redirect_stderr(stderr):
            build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(stderr.getvalue(), "")


class TestLoadBookCorpus(unittest.TestCase):
    def test_filters_books_by_toc_key_and_pdf_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            oa_dir = tmp_path / "open-access"
            cs_dir = tmp_path / "copyrighted-scans"
            oa_dir.mkdir()
            cs_dir.mkdir()

            # Book A: has a "toc" key and a matching PDF on disk -> included.
            (oa_dir / "book-a.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )
            (oa_dir / "book-a.pdf").write_bytes(b"%PDF-fake")

            # Book B: no "toc" key at all -> excluded (not yet retrofitted).
            (oa_dir / "book-b.expected.json").write_text(
                json.dumps({"chapters": []}), encoding="utf-8"
            )
            (oa_dir / "book-b.pdf").write_bytes(b"%PDF-fake")

            # Book C: has a "toc" key but no matching PDF on disk -> excluded.
            (cs_dir / "book-c.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )

            fake_reader = Mock()
            fake_reader.pages = [Mock()] * 5

            with patch(
                "evaluation.scripts.evaluate_layout_toc_classifier._CORPUS_DIR", tmp_path
            ), patch(
                "evaluation.scripts.evaluate_layout_toc_classifier.PdfReader",
                return_value=fake_reader,
            ):
                books = load_book_corpus()

        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["key"], "book-a")
        self.assertEqual(books[0]["corpus"], "open-access")
        self.assertEqual(books[0]["pdf_path"], oa_dir / "book-a.pdf")
        self.assertEqual(len(books[0]["labels"]), 5)


def _feature_row(book_key: str, label: str, value: float) -> dict:
    return {"book_key": book_key, "features": {name: value for name in FEATURE_NAMES}, "label": label}


class TestEvaluateLeaveOneBookOut(unittest.TestCase):
    def test_perfectly_separable_data_gets_full_recall(self):
        # Three synthetic books, each with 5 pages: page 0 is "toc"
        # (features all 5.0), page 1 is "chapter_first" (features all
        # -5.0), pages 2-4 are "other" (features all 0.0) -- identical,
        # trivially separable pattern across every book.
        rows = []
        for book_key in ("book-a", "book-b", "book-c"):
            rows.append(_feature_row(book_key, "toc", 5.0))
            rows.append(_feature_row(book_key, "chapter_first", -5.0))
            rows.extend(_feature_row(book_key, "other", 0.0) for _ in range(3))

        summary = evaluate_leave_one_book_out(rows)

        self.assertEqual(summary["full_recall_fraction"], 1.0)
        self.assertLessEqual(summary["avg_candidate_fraction"], 0.45)
        self.assertEqual(len(summary["per_book"]), 3)


if __name__ == "__main__":
    unittest.main()
