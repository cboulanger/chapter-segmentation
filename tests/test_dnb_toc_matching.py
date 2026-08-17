"""Unit tests for evaluation/dnb_toc_matching.py -- the whole-book
agreement gate that decides which dnb-toc-only books' extracted entries
are trustworthy enough for the bulk ground-truth tier (design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4). No PDFs, no network -- pure functions over synthetic
TocEntry lists."""

import unittest

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import align_toc_entries, diff_toc_entries, gate_book, toc_entry_to_gt_dict


def _entry(title: str, page: str | int | None, authors: tuple[str, ...] = ()) -> TocEntry:
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

    def test_matches_via_title_variant_when_primary_title_is_a_wrapped_fragment(self):
        # Real case found in a 2026-08-15 smoke test (book 9783899718188):
        # the heuristic's regex only captures a wrapped title's last
        # line as .title, with the full title in title_variants. The
        # LLM reads the title whole. Without checking title_variants,
        # this real match was scored below the alignment threshold.
        a = [TocEntry(
            title="Systemtheorie für Recht und Rechtswissenschaft",
            printed_page_number=33, source_page_index=0,
            title_variants=(
                "Niklas Luhmann und das Recht - Über die Nutzlosigkeit der "
                "Systemtheorie für Recht und Rechtswissenschaft",
            ),
        )]
        b = [_entry(
            "Niklas Luhmann und das Recht - Über die Nutzlosigkeit der "
            "Systemtheorie für Recht und Rechtswissenschaft",
            33,
        )]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_despite_trailing_ocr_noise_via_partial_ratio(self):
        # Real garbled-dot-leader-OCR cases measured in the investigation
        # behind docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
        # section 1d: token_sort_ratio alone scores these well below the
        # 70.0 threshold (a handful of garbage tokens dominates a short
        # real title's token multiset) even though one title is exactly
        # the other's real content plus a trailing noise run.
        a = [_entry("Ein Interview ss m onen een ee eee eee ees", 81)]
        b = [_entry("Ein Interview", 81)]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_still_rejects_genuinely_different_titles_on_the_same_page(self):
        # Negative control: partial_ratio must not become so permissive
        # that two different real entries sharing a page number align.
        a = [_entry("Die Einheit der Vernunft in der Vielfalt ihrer Stimmen", 117)]
        b = [_entry("Metaphysik nach Kant", 117)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_matches_on_prefixed_page_marker(self):
        # The real reported bug this whole change exists for: two
        # independent extractions that both correctly read "R42" for the
        # same line must be able to align -- the old int-with--1-sentinel
        # representation collapsed both to the same "unknown" value and
        # skipped them before matching was even attempted.
        a = [_entry("Appendix", "R42")]
        b = [_entry("Appendix", "R42")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_prefixed_marker_case_insensitively(self):
        a = [_entry("Appendix", "R42")]
        b = [_entry("Appendix", "r42")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_roman_numeral_case_difference(self):
        a = [_entry("Foreword", "VII")]
        b = [_entry("Foreword", "vii")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_arabic_page_with_leading_zero(self):
        a = [_entry("Einleitung", "07")]
        b = [_entry("Einleitung", "7")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])


class TestDiffTocEntries(unittest.TestCase):
    def test_full_agreement_has_no_singletons(self):
        a = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        b = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 2)
        self.assertEqual(only_a, [])
        self.assertEqual(only_b, [])

    def test_partial_agreement_separates_singletons_per_side(self):
        a = [_entry("Einleitung", 9), _entry("Only in A", 20)]
        b = [_entry("Einleitung", 9), _entry("Only in B", 30)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 1)
        self.assertEqual([e.title for e in only_a], ["Only in A"])
        self.assertEqual([e.title for e in only_b], ["Only in B"])

    def test_complete_disagreement_puts_everything_in_singletons(self):
        a = [_entry("A", 1)]
        b = [_entry("B", 2)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(matched, [])
        self.assertEqual([e.title for e in only_a], ["A"])
        self.assertEqual([e.title for e in only_b], ["B"])

    def test_matched_pairs_hold_entry_objects_not_indices(self):
        a = [_entry("Einleitung", 9, authors=("A Author",))]
        b = [_entry("Einleitung", 9, authors=("B Author",))]
        matched, _, _ = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 1)
        entry_a, entry_b = matched[0]
        self.assertEqual(entry_a.authors, ("A Author",))
        self.assertEqual(entry_b.authors, ("B Author",))


class TestGateBook(unittest.TestCase):
    def test_perfect_agreement_passes_with_union_equal_to_either_list(self):
        h = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        l = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.title for e in entries], ["Einleitung", "Schluss"])

    def test_below_threshold_rejects_whole_book(self):
        h = [_entry("Einleitung", 9), _entry("A", 20), _entry("B", 30), _entry("C", 40)]
        l = [_entry("Einleitung", 9)]  # agreement_rate = 1/4 = 0.25
        passed, entries = gate_book(h, l)
        self.assertFalse(passed)
        self.assertEqual(entries, [])

    def test_above_threshold_unions_singleton_entries_rather_than_dropping_them(self):
        # 9 of 10 heuristic entries agree with the LLM list -- rate 0.90.
        # The heuristic's 10th, LLM-missed entry must survive in the
        # merged result rather than being silently trimmed: the design's
        # core "no incomplete training target" requirement (spec section
        # 4.2) -- once a book clears the trust bar, a line only one
        # extractor caught is more likely a real miss than a hallucination.
        h = [_entry(f"Chapter {i}", i * 10) for i in range(1, 11)]
        l = h[:9]
        passed, entries = gate_book(h, l, threshold=0.90)
        self.assertTrue(passed)
        self.assertEqual(len(entries), 10)
        self.assertIn("Chapter 10", [e.title for e in entries])

    def test_empty_both_lists_rejects(self):
        self.assertEqual(gate_book([], []), (False, []))

    def test_merged_entries_sorted_by_printed_page_number(self):
        h = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        l = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.printed_page_number for e in entries], ["9", "40"])

    def test_matched_pair_keeps_heuristic_title_but_falls_back_to_llm_authors(self):
        # The heuristic (find_toc_candidates) almost never populates
        # authors -- only in a narrow "by <Name>" marker-line case -- so
        # always preferring the heuristic's own (empty) authors on a
        # matched pair would silently discard real author info the LLM
        # extracted. Falling back to the LLM's authors when the
        # heuristic's own are empty avoids that.
        h = [_entry("Einleitung", 9, authors=())]
        l = [_entry("Einleitung", 9, authors=("Jane Author",))]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual(entries[0].title, "Einleitung")  # heuristic's title kept
        self.assertEqual(entries[0].authors, ("Jane Author",))  # LLM's authors used

    def test_matched_pair_keeps_heuristic_authors_when_heuristic_has_them(self):
        h = [_entry("Einleitung", 9, authors=("Regex Author",))]
        l = [_entry("Einleitung", 9, authors=("Different LLM Reading",))]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual(entries[0].authors, ("Regex Author",))  # heuristic's own authors preferred when present


class TestTocEntryToGtDict(unittest.TestCase):
    def test_known_page_number_becomes_string(self):
        entry = _entry("Einleitung", 9, authors=("Jane Author",))
        self.assertEqual(
            toc_entry_to_gt_dict(entry),
            {"title": "Einleitung", "authors": ["Jane Author"], "printed_page_number": "9"},
        )

    def test_unknown_page_number_becomes_none(self):
        entry = _entry("Bibliographie", -1)
        self.assertEqual(
            toc_entry_to_gt_dict(entry),
            {"title": "Bibliographie", "authors": [], "printed_page_number": None},
        )
