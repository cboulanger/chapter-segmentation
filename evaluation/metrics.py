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


# Chapter-boundary correctness (pdf_start_index/pdf_end_index) is scored by
# precision_recall_f1 above with no tolerance. citation_pages is a SEPARATE,
# human-facing signal (the printed page range a consumer would cite/split
# on) -- see design spec docs/superpowers/specs/2026-08-08-citation-pages-
# mapping-design.md. Its own accuracy is scored independently per side: the
# start page is load-bearing (a consumer can't reconstruct it from
# context), the end page can reasonably be approximated (from the next
# chapter's start, or end-of-book) and only actually hurts usability when
# it's under-inclusive (cuts off real content).
_CITATION_END_OVER_INCLUSION_TOLERANCE = 3  # printed pages

# A local, minimal reimplementation of chapter_segmentation.segmentation's
# private _parse_toc_page_number -- evaluation/ only depends on
# segmentation's public surface elsewhere, and citation_pages strings here
# are always already-validated production output (never re-validated
# against segmentation's own plausibility guards), so a simple digit/roman
# parse is all this needs.
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _page_number_value(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    lowered = raw.lower()
    if not lowered or not all(ch in _ROMAN_VALUES for ch in lowered):
        return None
    total = 0
    for ch, nxt in zip(lowered, lowered[1:] + " "):
        value = _ROMAN_VALUES[ch]
        total += -value if nxt != " " and _ROMAN_VALUES.get(nxt, 0) > value else value
    return total


def _split_citation_pages(value: str | None) -> tuple[str, str] | None:
    if value is None or "-" not in value:
        return None
    start, _, end = value.partition("-")
    return start, end


@dataclass(frozen=True)
class CitationPageMetrics:
    start_coverage: float
    start_accuracy: float
    end_coverage: float
    end_accuracy: float
    checked_count: int
    start_covered_count: int = 0
    start_correct_count: int = 0
    end_covered_count: int = 0
    end_correct_count: int = 0


def citation_pages_metrics(expected: list[dict], found: list[dict]) -> CitationPageMetrics:
    """Among chapters whose (pdf_start_index, pdf_end_index) correctly
    matches expected (the same true-positive set precision_recall_f1
    scores), how well does citation_pages do -- restricted to expected
    chapters that themselves have a non-null citation_pages (an expected
    null means no printed number is visible anywhere on that chapter's real
    boundary pages, so there is nothing to score). End-page correctness
    tolerates a found value up to _CITATION_END_OVER_INCLUSION_TOLERANCE
    printed pages PAST the expected end (trailing blank/divider pages
    absorbed into the range) but never before it (that would mean real
    chapter content was cut off, a real defect, not an approximation)."""
    found_by_range = {(c["pdf_start_index"], c["pdf_end_index"]): c for c in found}
    checked = [
        e for e in expected
        if _split_citation_pages(e.get("citation_pages")) is not None
        and (e["pdf_start_index"], e["pdf_end_index"]) in found_by_range
    ]
    if not checked:
        return CitationPageMetrics(0.0, 0.0, 0.0, 0.0, 0)
    start_covered = start_correct = end_covered = end_correct = 0
    for e in checked:
        expected_start, expected_end = _split_citation_pages(e["citation_pages"])
        found_split = _split_citation_pages(found_by_range[(e["pdf_start_index"], e["pdf_end_index"])].get("citation_pages"))
        found_start, found_end = found_split if found_split else (None, None)
        start_covered += found_start is not None
        start_correct += found_start == expected_start
        end_covered += found_end is not None
        expected_end_value = _page_number_value(expected_end)
        found_end_value = _page_number_value(found_end) if found_end else None
        end_correct += (
            found_end_value is not None
            and expected_end_value is not None
            and expected_end_value <= found_end_value <= expected_end_value + _CITATION_END_OVER_INCLUSION_TOLERANCE
        )
    n = len(checked)
    return CitationPageMetrics(
        start_covered / n, start_correct / n, end_covered / n, end_correct / n, n,
        start_covered, start_correct, end_covered, end_correct,
    )


class CitationPageAggregate:
    """Pools citation_pages_metrics results across documents before
    computing final rates -- same weighted-pooling style as MicroAggregate,
    reading raw counts (never re-deriving them from rates, which would
    round-trip through lossy float rounding)."""

    def __init__(self) -> None:
        self._checked = 0
        self._start_covered = 0
        self._start_correct = 0
        self._end_covered = 0
        self._end_correct = 0

    def add(self, metrics: CitationPageMetrics) -> None:
        self._checked += metrics.checked_count
        self._start_covered += metrics.start_covered_count
        self._start_correct += metrics.start_correct_count
        self._end_covered += metrics.end_covered_count
        self._end_correct += metrics.end_correct_count

    def compute(self) -> CitationPageMetrics:
        n = self._checked
        if n == 0:
            return CitationPageMetrics(0.0, 0.0, 0.0, 0.0, 0)
        return CitationPageMetrics(
            self._start_covered / n, self._start_correct / n,
            self._end_covered / n, self._end_correct / n, n,
            self._start_covered, self._start_correct, self._end_covered, self._end_correct,
        )
