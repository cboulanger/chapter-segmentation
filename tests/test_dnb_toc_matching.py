"""Unit tests for evaluation/dnb_toc_matching.py -- the whole-book
agreement gate that decides which dnb-toc-only books' extracted entries
are trustworthy enough for the bulk ground-truth tier (design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4). No PDFs, no network -- pure functions over synthetic
TocEntry lists."""

import unittest

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import align_toc_entries


def _entry(title: str, page: int, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0, authors=authors)


class TestAlignTocEntries(unittest.TestCase):
    def test_matches_same_page_and_similar_title(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Einleitung", 9)]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_no_match_on_page_mismatch(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Einleitung", 11)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_no_match_when_either_page_unknown(self):
        a = [_entry("Einleitung", -1)]
        b = [_entry("Einleitung", 9)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_no_match_on_dissimilar_title_same_page(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Bibliographie", 9)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_order_preserving_scan_misses_out_of_order_match(self):
        # Same "TOC order is book order" monotonicity tradeoff
        # src/chapter_segmentation/evidence/fusion.py's _align already
        # makes: a[0] ("Erster Teil", page 9) can only match b at or after
        # b-index 0. b[0] ("Zweiter Teil", page 40) doesn't match, so the
        # scan advances to b[1] ("Erster Teil", page 9), which does --
        # consuming last_j=1. a[1] ("Zweiter Teil", page 40) then has no
        # b-index left to scan (>= 2), even though a real match existed
        # earlier in b. This is expected behavior, not a bug.
        a = [_entry("Erster Teil", 9), _entry("Zweiter Teil", 40)]
        b = [_entry("Zweiter Teil", 40), _entry("Erster Teil", 9)]
        self.assertEqual(align_toc_entries(a, b), [(0, 1)])

    def test_empty_lists(self):
        self.assertEqual(align_toc_entries([], []), [])
        self.assertEqual(align_toc_entries([_entry("X", 1)], []), [])
        self.assertEqual(align_toc_entries([], [_entry("X", 1)]), [])
