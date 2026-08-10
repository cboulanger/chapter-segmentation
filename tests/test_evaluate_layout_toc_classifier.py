"""Unit tests for evaluation/scripts/evaluate_layout_toc_classifier.py's
pure logic. The real pdfalto-subprocess-driven, real-corpus leave-one-book-out
run is exercised manually -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import unittest

from evaluation.scripts.evaluate_layout_toc_classifier import select_threshold


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


if __name__ == "__main__":
    unittest.main()
