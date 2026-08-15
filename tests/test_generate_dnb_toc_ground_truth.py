"""Unit tests for evaluation/scripts/generate_dnb_toc_ground_truth.py's
pure logic. The real PDF-reading/LLM-calling main() is exercised manually
against the real corpus with a real KISSKI_API_KEY -- see design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md."""

import unittest

from chapter_segmentation.segmentation import find_toc_candidates
from evaluation.scripts.generate_dnb_toc_ground_truth import _toc_entries_for_scan

_TOC_PAGE = (
    "Inhaltsverzeichnis\n"
    "Einleitung ..... 9\n"
    "Zur Soziologie des Rechts ..... 17\n"
    "Schlussbetrachtung ..... 89\n"
)


class TestTocEntriesForScan(unittest.TestCase):
    def test_raw_find_toc_candidates_rejects_realistic_page_numbers_on_a_tiny_pdf(self):
        # Demonstrates the bug: on an unpadded 2-page dnb-toc-only-shaped
        # PDF, _TOC_MAX_PAGE_NUMBER_RATIO (2.0) caps plausible page numbers
        # at 2*2=4 -- every real entry above that (9, 17, 89) is rejected.
        pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
        self.assertEqual(find_toc_candidates(pages), [])

    def test_padded_wrapper_recovers_the_same_entries(self):
        pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
        entries = _toc_entries_for_scan(pages)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].title, "Einleitung")
        self.assertEqual(entries[0].printed_page_number, 9)
        self.assertEqual(entries[2].printed_page_number, 89)
