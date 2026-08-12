"""Unit tests for evaluation/scripts/add_toc_ground_truth.py's retrofit_book()
pure logic. The real file-walking main() is exercised manually against the
real evaluation corpus -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import unittest

from evaluation.scripts.add_toc_ground_truth import retrofit_book


def _toc_like_page(entries: list[tuple[str, int]]) -> str:
    """Builds page text with 3+ "title ... number" lines, matching
    ground_truth_helper._TOC_LINE_RE, so find_toc_pages structurally
    detects it as a TOC page."""
    return "\n".join(f"{title} {'.' * 10} {number}" for title, number in entries)


_TOC_TEXT = _toc_like_page([("Chapter One", 5), ("Chapter Two", 12), ("Chapter Three", 25)])


class TestRetrofitBook(unittest.TestCase):
    def test_skips_when_toc_key_already_present_and_not_forced(self):
        expected = {"chapters": [], "toc": None}
        updated, message = retrofit_book(["page text"], expected, force=False)
        self.assertIsNone(updated)
        self.assertTrue(message.startswith("SKIP"))

    def test_writes_toc_null_when_no_toc_page_found(self):
        pages = ["Chapter One\n\nSome body text with no listing lines at all."]
        expected = {"chapters": []}
        updated, message = retrofit_book(pages, expected, force=False)
        self.assertEqual(updated["toc"], None)
        self.assertTrue(message.startswith("OK"))

    def test_writes_toc_range_for_contiguous_toc_pages(self):
        pages = ["Front cover", _TOC_TEXT, "Chapter One\n\nBody text starts here."]
        expected = {"chapters": []}
        updated, message = retrofit_book(pages, expected, force=False)
        self.assertEqual(updated["toc"], {"toc_start_index": 1, "toc_end_index": 1})
        self.assertTrue(message.startswith("OK"))

    def test_flags_non_contiguous_toc_pages_for_manual_review(self):
        pages = [_TOC_TEXT, "unrelated page", _TOC_TEXT]
        expected = {"chapters": []}
        updated, message = retrofit_book(pages, expected, force=False)
        self.assertIsNone(updated)
        self.assertTrue(message.startswith("NEEDS REVIEW"))

    def test_force_recomputes_even_when_toc_key_present(self):
        pages = [_TOC_TEXT]
        expected = {"chapters": [], "toc": {"toc_start_index": 99, "toc_end_index": 99}}
        updated, message = retrofit_book(pages, expected, force=True)
        self.assertEqual(updated["toc"], {"toc_start_index": 0, "toc_end_index": 0})


if __name__ == "__main__":
    unittest.main()
