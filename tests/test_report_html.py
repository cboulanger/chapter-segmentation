"""Unit tests for evaluation/report_html.py -- the shared table renderer
used by both generate_report.py's main report and its LLM detail page."""

import unittest

from evaluation.metrics import Metrics
from evaluation.report_html import render_strategy_tables


def _metrics(precision: float, recall: float, f1: float, tp: int = 1, found: int = 1, expected: int = 1) -> Metrics:
    return Metrics(precision=precision, recall=recall, f1=f1, true_positives=tp, found_count=found, expected_count=expected)


class TestRenderStrategyTables(unittest.TestCase):
    def test_marks_highest_f1_cell_per_document_row(self):
        per_document = {
            "book-a": {
                "heuristic": (_metrics(0.5, 0.5, 0.5), 1.0),
                "outline": (_metrics(1.0, 1.0, 1.0), 2.0),
            },
        }
        html = render_strategy_tables(
            title="Test report", description_html="<p>desc</p>",
            strategy_names=["heuristic", "outline"],
            per_document=per_document,
            aggregates={"heuristic": _metrics(0.5, 0.5, 0.5), "outline": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0, "outline": 2.0},
        )
        # The outline cell (F1=1.0) should be marked; the heuristic cell (F1=0.5) should not.
        outline_cell_start = html.index("F1=1.00")
        heuristic_cell_start = html.index("F1=0.50")
        self.assertIn("font-weight:bold", html[max(0, outline_cell_start - 100):outline_cell_start])
        self.assertNotIn("font-weight:bold", html[max(0, heuristic_cell_start - 100):heuristic_cell_start])

    def test_does_not_highlight_a_shared_zero_f1_as_best(self):
        # Every strategy finding nothing for a book is a shared failure,
        # not a "win" for whichever one happens to be listed -- found on
        # the real corpus, where several books score F1=0.00 across every
        # strategy and both cells lit up green as "best in row".
        per_document = {
            "book-a": {
                "heuristic": (_metrics(0.0, 0.0, 0.0, tp=0, found=2, expected=5), 1.0),
                "outline": (_metrics(0.0, 0.0, 0.0, tp=0, found=0, expected=5), 0.0),
            },
        }
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic", "outline"],
            per_document=per_document,
            aggregates={
                "heuristic": _metrics(0.0, 0.0, 0.0, tp=0, found=2, expected=5),
                "outline": _metrics(0.0, 0.0, 0.0, tp=0, found=0, expected=5),
            },
            aggregate_times={"heuristic": 1.0, "outline": 0.0},
        )
        self.assertNotIn("font-weight:bold", html[: html.index("Per strategy")])

    def test_renders_na_for_missing_strategy_result(self):
        per_document = {"book-a": {"heuristic": (_metrics(1.0, 1.0, 1.0), 1.0), "outline": None}}
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic", "outline"],
            per_document=per_document,
            aggregates={"heuristic": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0},
        )
        self.assertIn("<td>N/A</td>", html)

    def test_orders_aggregate_rows_by_f1_descending(self):
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["low", "high"],
            per_document={},
            aggregates={"low": _metrics(0.3, 0.3, 0.3), "high": _metrics(0.9, 0.9, 0.9)},
            aggregate_times={"low": 1.0, "high": 1.0},
        )
        # Scoped to the aggregate section, not the whole page: the
        # per-document table's header row also renders "low"/"high" (from
        # strategy_names, in caller-given order) regardless of whether
        # per_document has any rows, which would otherwise contaminate a
        # whole-page substring search with an unrelated ordering.
        agg_section = html[html.index("Per strategy"):]
        self.assertLess(agg_section.index(">high<"), agg_section.index(">low<"))

    def test_includes_document_keys_as_row_labels(self):
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic"],
            per_document={"9783031466373": {"heuristic": (_metrics(1.0, 1.0, 1.0), 1.0)}},
            aggregates={"heuristic": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0},
        )
        self.assertIn("9783031466373", html)

    def test_title_appears_in_output(self):
        html = render_strategy_tables(
            title="My Special Report Title", description_html="",
            strategy_names=[], per_document={}, aggregates={}, aggregate_times={},
        )
        self.assertIn("My Special Report Title", html)


if __name__ == "__main__":
    unittest.main()
