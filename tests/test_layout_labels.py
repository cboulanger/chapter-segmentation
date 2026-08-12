"""Unit tests for evaluation/scripts/layout_labels.py's page_labels()."""

import unittest

from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST, LABEL_OTHER, LABEL_TOC, page_labels


class TestPageLabels(unittest.TestCase):
    def test_missing_toc_key_returns_none(self):
        expected = {"chapters": []}
        self.assertIsNone(page_labels(expected, total_pages=5))

    def test_null_toc_labels_only_chapters(self):
        expected = {"chapters": [{"pdf_start_index": 2, "pdf_end_index": 4}], "toc": None}
        labels = page_labels(expected, total_pages=5)
        self.assertEqual(
            labels,
            [LABEL_OTHER, LABEL_OTHER, LABEL_CHAPTER_FIRST, LABEL_OTHER, LABEL_OTHER],
        )

    def test_toc_range_and_chapters_labeled(self):
        expected = {
            "chapters": [
                {"pdf_start_index": 3, "pdf_end_index": 6},
                {"pdf_start_index": 7, "pdf_end_index": 9},
            ],
            "toc": {"toc_start_index": 1, "toc_end_index": 2},
        }
        labels = page_labels(expected, total_pages=10)
        self.assertEqual(
            labels,
            [
                LABEL_OTHER, LABEL_TOC, LABEL_TOC, LABEL_CHAPTER_FIRST,
                LABEL_OTHER, LABEL_OTHER, LABEL_OTHER, LABEL_CHAPTER_FIRST,
                LABEL_OTHER, LABEL_OTHER,
            ],
        )


if __name__ == "__main__":
    unittest.main()
