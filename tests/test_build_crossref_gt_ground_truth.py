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

from evaluation.scripts.build_crossref_gt_ground_truth import _toc_field_for


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


if __name__ == "__main__":
    unittest.main()
