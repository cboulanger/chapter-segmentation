"""Precision/recall/F1 scoring shared by generate_report.py and
refresh_llm_cache.py -- one implementation so every strategy/report is
scored identically. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    found_count: int
    expected_count: int


def precision_recall_f1(expected: list[dict], found: list[dict]) -> Metrics:
    """Exact (pdf_start_index, pdf_end_index) range match -- no partial
    credit for an overlapping-but-not-identical range."""
    expected_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in expected}
    found_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in found}
    true_positives = expected_ranges & found_ranges
    tp, found_count, expected_count = len(true_positives), len(found_ranges), len(expected_ranges)
    precision = tp / found_count if found_count else 0.0
    recall = tp / expected_count if expected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(
        precision=precision, recall=recall, f1=f1,
        true_positives=tp, found_count=found_count, expected_count=expected_count,
    )


class MicroAggregate:
    """Pools true-positive/found/expected counts across documents before
    computing precision/recall/F1 -- weights larger books more heavily,
    matching generate_report.py's aggregate style. Also sums elapsed time
    across every `add()` call, for the "total time spent" column."""

    def __init__(self) -> None:
        self._tp = 0
        self._found = 0
        self._expected = 0
        self._elapsed_seconds = 0.0

    def add(self, metrics: Metrics, elapsed_seconds: float = 0.0) -> None:
        self._tp += metrics.true_positives
        self._found += metrics.found_count
        self._expected += metrics.expected_count
        self._elapsed_seconds += elapsed_seconds

    def compute(self) -> Metrics:
        precision = self._tp / self._found if self._found else 0.0
        recall = self._tp / self._expected if self._expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return Metrics(
            precision=precision, recall=recall, f1=f1,
            true_positives=self._tp, found_count=self._found, expected_count=self._expected,
        )

    @property
    def total_elapsed_seconds(self) -> float:
        return self._elapsed_seconds
