"""Unit tests for evaluation/scripts/evaluate_layout_toc_classifier.py's
pure logic. The real pdfalto-subprocess-driven, real-corpus leave-one-book-out
run is exercised manually -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.scripts.evaluate_layout_toc_classifier import build_feature_table, select_threshold


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


if __name__ == "__main__":
    unittest.main()
