"""Unit tests for evaluation/metrics.py -- precision/recall/F1 scoring and
micro-aggregation shared by generate_report.py and refresh_llm_cache.py."""

import unittest

from evaluation.metrics import MicroAggregate, precision_recall_f1


def _chapter(start: int, end: int) -> dict:
    return {"pdf_start_index": start, "pdf_end_index": end}


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


if __name__ == "__main__":
    unittest.main()
