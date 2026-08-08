"""Unit tests for evaluation/metrics.py -- precision/recall/F1 scoring and
micro-aggregation shared by generate_report.py and refresh_llm_cache.py."""

import unittest

from evaluation.metrics import CitationPageAggregate, MicroAggregate, citation_pages_metrics, precision_recall_f1


def _chapter(start: int, end: int) -> dict:
    return {"pdf_start_index": start, "pdf_end_index": end}


def _chapter_with_citation(start: int, end: int, citation_pages: str | None) -> dict:
    return {"pdf_start_index": start, "pdf_end_index": end, "citation_pages": citation_pages}


class TestPrecisionRecallF1(unittest.TestCase):
    def test_perfect_match(self):
        expected = [_chapter(0, 5), _chapter(6, 10)]
        found = [_chapter(0, 5), _chapter(6, 10)]
        m = precision_recall_f1(expected, found)
        self.assertEqual((m.precision, m.recall, m.f1), (1.0, 1.0, 1.0))
        self.assertEqual((m.true_positives, m.found_count, m.expected_count), (2, 2, 2))

    def test_partial_overlap_no_partial_credit(self):
        expected = [_chapter(0, 5)]
        found = [_chapter(0, 6)]  # one page off -- not a match at all
        m = precision_recall_f1(expected, found)
        self.assertEqual(m.true_positives, 0)
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.f1, 0.0)

    def test_empty_found_gives_zero_precision_and_recall(self):
        m = precision_recall_f1([_chapter(0, 5)], [])
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.f1, 0.0)

    def test_empty_expected_and_found_gives_zero_not_division_error(self):
        m = precision_recall_f1([], [])
        self.assertEqual((m.precision, m.recall, m.f1), (0.0, 0.0, 0.0))

    def test_f1_is_harmonic_mean(self):
        # precision=1.0 (1/1 found correct), recall=0.5 (1/2 expected found)
        m = precision_recall_f1([_chapter(0, 5), _chapter(6, 10)], [_chapter(0, 5)])
        self.assertAlmostEqual(m.precision, 1.0)
        self.assertAlmostEqual(m.recall, 0.5)
        self.assertAlmostEqual(m.f1, 2 * 1.0 * 0.5 / (1.0 + 0.5))


class TestMicroAggregate(unittest.TestCase):
    def test_pools_counts_across_documents_before_scoring(self):
        agg = MicroAggregate()
        # Book A: 1 correct out of 1 found, 2 expected
        agg.add(precision_recall_f1([_chapter(0, 5), _chapter(6, 10)], [_chapter(0, 5)]), elapsed_seconds=1.0)
        # Book B: 1 correct out of 1 found, 1 expected
        agg.add(precision_recall_f1([_chapter(0, 5)], [_chapter(0, 5)]), elapsed_seconds=2.0)
        result = agg.compute()
        # Pooled: tp=2, found=2, expected=3 -> precision=1.0, recall=2/3
        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.found_count, 2)
        self.assertEqual(result.expected_count, 3)
        self.assertAlmostEqual(result.precision, 1.0)
        self.assertAlmostEqual(result.recall, 2 / 3)
        self.assertEqual(agg.total_elapsed_seconds, 3.0)

    def test_empty_aggregate_is_all_zero(self):
        result = MicroAggregate().compute()
        self.assertEqual((result.precision, result.recall, result.f1), (0.0, 0.0, 0.0))
        self.assertEqual(MicroAggregate().total_elapsed_seconds, 0.0)


class TestCitationPagesMetrics(unittest.TestCase):
    def test_correct_start_and_exact_end(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-20")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.checked_count, 1)
        self.assertEqual((m.start_coverage, m.start_accuracy), (1.0, 1.0))
        self.assertEqual((m.end_coverage, m.end_accuracy), (1.0, 1.0))

    def test_wrong_start_counts_as_incorrect(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "13-20")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.start_accuracy, 0.0)
        self.assertEqual(m.start_coverage, 1.0)  # a (wrong) value was still found

    def test_end_over_inclusive_within_tolerance_counts_as_correct(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-23")]  # +3, at the tolerance boundary
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 1.0)

    def test_end_over_inclusive_beyond_tolerance_counts_as_incorrect(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-24")]  # +4, past the tolerance
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 0.0)

    def test_end_under_inclusive_by_one_page_counts_as_incorrect(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-19")]  # -1, real content cut off
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 0.0)

    def test_null_found_citation_pages_counts_as_uncovered(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, None)]
        m = citation_pages_metrics(expected, found)
        self.assertEqual((m.start_coverage, m.start_accuracy), (0.0, 0.0))
        self.assertEqual((m.end_coverage, m.end_accuracy), (0.0, 0.0))

    def test_expected_chapter_with_null_citation_pages_excluded_from_denominator(self):
        expected = [_chapter_with_citation(0, 5, None), _chapter_with_citation(6, 10, "1-4")]
        found = [_chapter_with_citation(0, 5, None), _chapter_with_citation(6, 10, "1-4")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.checked_count, 1)

    def test_expected_chapter_with_no_matching_found_range_excluded_from_denominator(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 6, "12-20")]  # different range -- not a match at all
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.checked_count, 0)

    def test_handles_roman_numeral_pages(self):
        expected = [_chapter_with_citation(0, 5, "vii-x")]
        found = [_chapter_with_citation(0, 5, "vii-xi")]  # end +1 (x=10 -> xi=11), within tolerance
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.start_accuracy, 1.0)
        self.assertEqual(m.end_accuracy, 1.0)

    def test_malformed_roman_numeral_in_expected_end_is_treated_as_unscoreable(self):
        # "iiii" is not well-formed roman notation (should be "iv") --
        # the real production parser (_parse_toc_page_number) rejects it
        # rather than silently computing a plausible-looking value, and
        # this metric must not be more permissive than production is,
        # since expected["citation_pages"] here comes from hand-authored
        # ground truth that never passes through production validation.
        expected = [_chapter_with_citation(0, 5, "1-iiii")]
        found = [_chapter_with_citation(0, 5, "1-iv")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 0.0)

    def test_implausibly_large_roman_numeral_is_treated_as_unscoreable(self):
        # Over _ROMAN_PAGE_MAX_VALUE (50) -- must be rejected the same way
        # production rejects it, not silently parsed as a huge int.
        expected = [_chapter_with_citation(0, 5, "1-mmmm")]
        found = [_chapter_with_citation(0, 5, "1-mmmm")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 0.0)

    def test_no_checked_chapters_returns_all_zero(self):
        m = citation_pages_metrics([], [])
        self.assertEqual(
            (m.start_coverage, m.start_accuracy, m.end_coverage, m.end_accuracy, m.checked_count),
            (0.0, 0.0, 0.0, 0.0, 0),
        )


class TestCitationPageAggregate(unittest.TestCase):
    def test_pools_counts_across_documents_before_computing_rates(self):
        agg = CitationPageAggregate()
        # Book A: 1 of 1 checked, start correct, end wrong (under-inclusive)
        agg.add(citation_pages_metrics(
            [_chapter_with_citation(0, 5, "12-20")], [_chapter_with_citation(0, 5, "12-19")],
        ))
        # Book B: 1 of 1 checked, both correct
        agg.add(citation_pages_metrics(
            [_chapter_with_citation(0, 5, "1-4")], [_chapter_with_citation(0, 5, "1-4")],
        ))
        result = agg.compute()
        self.assertEqual(result.checked_count, 2)
        self.assertEqual(result.start_accuracy, 1.0)
        self.assertEqual(result.end_accuracy, 0.5)

    def test_empty_aggregate_is_all_zero(self):
        result = CitationPageAggregate().compute()
        self.assertEqual(result.checked_count, 0)
        self.assertEqual((result.start_accuracy, result.end_accuracy), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
