"""Unit tests for evaluation/scripts/ground_truth_helper.py's
toc_page_range()."""

import unittest

from evaluation.scripts.ground_truth_helper import toc_page_range


class TestTocPageRange(unittest.TestCase):
    def test_empty_set_returns_none(self):
        self.assertIsNone(toc_page_range(set()))

    def test_single_page(self):
        self.assertEqual(toc_page_range({5}), (5, 5))

    def test_contiguous_run(self):
        self.assertEqual(toc_page_range({7, 5, 6}), (5, 7))

    def test_two_separate_runs_returns_none(self):
        self.assertIsNone(toc_page_range({5, 6, 20}))

    def test_two_adjacent_singletons_with_gap_returns_none(self):
        self.assertIsNone(toc_page_range({5, 7}))


if __name__ == "__main__":
    unittest.main()
