"""Unit tests for evaluation/scripts/ground_truth_helper.py's toc_page_range(),
find_toc_pages(), and extract_printed_number(), including language-agnosticism
regression coverage for the latter two."""

import unittest

from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range


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


class TestLanguageAgnosticPatternMatching(unittest.TestCase):
    """Guards find_toc_pages/extract_printed_number against a future change
    that silently introduces an English-only assumption (e.g. a
    keyword-based "Contents" search) -- both currently key off page-number
    *shape* (digits/roman numerals), not language-specific words, so they
    must work identically on German and French TOC/page-number text."""

    def test_finds_german_toc_page(self):
        pages = [
            "Vorwort\n\n\nSeite 3",
            "Inhaltsverzeichnis\n\nEinleitung .......... 7\nKapitel 1 .......... 15\nKapitel 2 .......... 42\n",
            "Einleitung\n\nDies ist der erste Absatz.",
        ]
        self.assertEqual(find_toc_pages(pages), {1})

    def test_finds_french_toc_page(self):
        pages = [
            "Préface\n\n\nPage 3",
            "Table des matières\n\nIntroduction .......... 7\nChapitre 1 .......... 15\nChapitre 2 .......... 42\n",
            "Introduction\n\nCeci est le premier paragraphe.",
        ]
        self.assertEqual(find_toc_pages(pages), {1})

    def test_extracts_german_footer_page_number(self):
        text = "Einleitung\n\nDies ist der erste Absatz der Einleitung.\n\n7"
        self.assertEqual(extract_printed_number(text), "7")

    def test_extracts_roman_numeral_footer_page_number_in_german_context(self):
        # Roman-numeral page numbers are common in front matter (Vorwort,
        # Préface, etc.) regardless of the book's language -- this guards
        # against a regex change that only recognizes arabic digits, or
        # that gets confused by non-English surrounding words.
        text = "Vorwort\n\nDies ist das Vorwort zum Buch.\n\nvii"
        self.assertEqual(extract_printed_number(text), "vii")


if __name__ == "__main__":
    unittest.main()
