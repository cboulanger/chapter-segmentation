"""Unit tests for evaluation/scripts/build_crossref_gt_ground_truth.py's
_toc_field_for() pure logic -- the branch that decides what (if anything)
to write for the "toc" key when migrating a CrossRef-sourced book. Must
match add_toc_ground_truth.py's retrofit_book() semantics exactly (see
tests/test_add_toc_ground_truth.py): empty toc_pages means "confirmed no
TOC" (write null), non-empty-but-non-contiguous means "ambiguous, needs a
human" (omit the key entirely, not a wrong null), and a single contiguous
run writes the resolved range. The rest of process_book() (PDF fetching,
chapter reconciliation) is exercised manually against the real corpus --
see docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md.
"""

import unittest

from evaluation.scripts.build_crossref_gt_ground_truth import (
    _is_novel,
    _nearest_neighbor_distance,
    _novelty_threshold,
    _toc_field_for,
)


class TestTocFieldFor(unittest.TestCase):
    def test_empty_toc_pages_writes_null(self):
        field, write_key, status = _toc_field_for(set())
        self.assertIsNone(field)
        self.assertTrue(write_key)
        self.assertIn("null", status)

    def test_contiguous_toc_pages_writes_range(self):
        field, write_key, status = _toc_field_for({7, 8, 9})
        self.assertEqual(field, {"toc_start_index": 7, "toc_end_index": 9})
        self.assertTrue(write_key)
        self.assertIn("toc=", status)

    def test_non_contiguous_toc_pages_omits_key(self):
        field, write_key, status = _toc_field_for({3, 5})
        self.assertIsNone(field)
        self.assertFalse(write_key)
        self.assertTrue(status.startswith("toc NEEDS REVIEW"))

    def test_single_toc_page_writes_range(self):
        field, write_key, status = _toc_field_for({4})
        self.assertEqual(field, {"toc_start_index": 4, "toc_end_index": 4})
        self.assertTrue(write_key)


class TestNearestNeighborDistance(unittest.TestCase):
    def test_returns_distance_to_closest_point(self):
        # Distance to [1.0, 0.0] is 1.0; to [3.0, 4.0] is 5.0 -- the
        # nearer one wins.
        distance = _nearest_neighbor_distance([0.0, 0.0], [[3.0, 4.0], [1.0, 0.0]])
        self.assertAlmostEqual(distance, 1.0)

    def test_single_other_point(self):
        distance = _nearest_neighbor_distance([0.0], [[5.0]])
        self.assertAlmostEqual(distance, 5.0)


class TestNoveltyThreshold(unittest.TestCase):
    def test_percentile_of_leave_one_out_distances(self):
        # 1-D vectors [0, 1, 2, 3, 10]. Leave-one-out nearest-neighbor
        # distance is 1.0 for every point except 10.0 (nearest is 3.0,
        # distance 7.0): [1, 1, 1, 1, 7]. numpy's default linear-
        # interpolation 90th percentile of that sorted list is 4.6.
        vectors = [[0.0], [1.0], [2.0], [3.0], [10.0]]
        threshold = _novelty_threshold(vectors, percentile=90)
        self.assertAlmostEqual(threshold, 4.6)


class TestIsNovel(unittest.TestCase):
    def setUp(self):
        self.existing = [[0.0], [1.0], [2.0]]

    def test_candidate_close_to_existing_is_not_novel(self):
        self.assertFalse(_is_novel([[0.5]], self.existing, threshold=1.5))

    def test_candidate_far_from_existing_is_novel(self):
        self.assertTrue(_is_novel([[10.0]], self.existing, threshold=1.5))

    def test_novel_if_any_candidate_page_qualifies(self):
        # First candidate page is close (not novel alone); second is far.
        # Keep-if-at-least-one-page-is-novel per the design spec.
        self.assertTrue(_is_novel([[0.5], [10.0]], self.existing, threshold=1.5))


if __name__ == "__main__":
    unittest.main()
