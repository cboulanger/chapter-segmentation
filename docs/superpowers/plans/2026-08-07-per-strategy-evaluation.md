# Per-strategy evaluation and reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each chapter-segmentation strategy (heuristic, outline, LLM) be measured independently against the evaluation corpus, with per-document and per-strategy (aggregate, timed) reporting, plus a manually/nightly-refreshed, cached LLM evaluation folded into the automated report.

**Architecture:** Two new standalone orchestration functions in `segmentation.py` let outline and LLM strategies run without the production pipeline's merge/fallback decisions. A committed `public-cache/*.outline.json` extends the PDF-free CI corpus to cover the outline strategy. A committed `llm-cache/*.json` holds raw per-model LLM results so the expensive part (calling KISSKI) is decoupled from the free part (rendering reports). Shared `metrics.py`/`report_html.py` modules back two generated pages: the main report (heuristic + outline + best-cached-LLM-model) and a full LLM detail page (every cached model).

**Tech Stack:** Python 3.12, pytest/unittest, httpx, openai SDK, GitHub Actions.

Full design: `docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md`.

---

### Task 1: `analyze_attachment_outline_only` in segmentation.py

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:1259` (insert after `_BOOK_TITLE_BOOKMARK_FUZZ_THRESHOLD`, before `def build_book_context`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_segmentation.py`. First add `analyze_attachment_outline_only` to the existing `from chapter_segmentation.segmentation import (...)` block at the top (the one starting `TocEntry, extract_page_texts_from_pdf_bytes, ...` at line 13), and add these two imports near the top of the file:

```python
from chapter_segmentation.evidence.outline_strategy import extract_outline_candidates
from chapter_segmentation.evidence.types import ChapterCandidate
```

Then append this test class to the end of the file:

```python
class TestAnalyzeAttachmentOutlineOnly(unittest.TestCase):
    _TWO_CHAPTER_PAGES = [
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 0
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 1
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 2
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 3
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 4
        "Introduction\nJane Author\n\nBody text opening the chapter.\n\n1",  # 5
        "...continued introduction text with real body content here.\n\n2",  # 6
        "...more continued introduction text with real body content.\n\n3",  # 7
        "...final continued introduction text with real body content.\n\n4",  # 8
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 9
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 10
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 11
        "Comparing Citation Styles\n\nJohn Smith\n\nBody text opening this chapter.\n\n5",  # 12
        "...continued citation styles text with real body content here.\n\n6",  # 13
        "...more continued citation styles text with real body content.\n\n7",  # 14
        "...final continued citation styles text with real body content.\n\n8",  # 15
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 16
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 17
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 18
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 19
    ]

    def test_builds_chapters_directly_from_resolved_candidates(self):
        candidates = [
            ChapterCandidate(title="Introduction", authors=("Jane Author",), pdf_page_index=5, source="outline"),
            ChapterCandidate(title="Comparing Citation Styles", authors=("John Smith",), pdf_page_index=12, source="outline"),
        ]
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, candidates)
        chapters = sorted(result["chapters"], key=lambda c: c["pdf_start_index"])
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "Introduction")
        self.assertEqual(chapters[0]["pdf_start_index"], 5)
        self.assertEqual(chapters[0]["source"], "outline")
        self.assertEqual(chapters[0]["confidence"], 0.98)
        self.assertEqual(chapters[1]["pdf_start_index"], 12)
        self.assertEqual(result["diagnostics"]["outline_candidates_found"], 2)

    def test_empty_candidates_yields_no_chapters(self):
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, [])
        self.assertEqual(result["chapters"], [])
        self.assertEqual(result["segmentation_confidence"], "low")

    def test_skips_candidate_missing_pdf_page_index(self):
        # Should never happen for a real extract_outline_candidates() result
        # (every entry it returns already has pdf_page_index resolved), but
        # must not crash if it does -- skip rather than guess.
        candidates = [
            ChapterCandidate(title="Introduction", pdf_page_index=5, source="outline"),
            ChapterCandidate(title="Undated", pdf_page_index=None, source="outline"),
        ]
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, candidates)
        titles = [c["title"] for c in result["chapters"]]
        self.assertEqual(titles, ["Introduction"])

    def test_matches_strategies_pipeline_outline_only_result(self):
        # Same fixture as
        # TestAnalyzeAttachmentWithStrategiesOutlineOnly.test_outline_only_uses_direct_localization_and_fixed_confidence
        # in tests/test_segmentation_strategies.py -- the standalone
        # function must agree with the pipeline's own outline-only branch.
        pdf_bytes = _pdf_with_outline(20, [("Introduction", 5), ("Comparing Citation Styles", 12)])
        candidates = extract_outline_candidates(pdf_bytes)
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, candidates)
        chapters = sorted(result["chapters"], key=lambda c: c["pdf_start_index"])
        self.assertEqual([c["pdf_start_index"] for c in chapters], [5, 12])
```

`_pdf_with_outline` is not yet imported in this file -- add it as a local helper at module scope (near the existing `_blank_pdf` helper):

```python
def _pdf_with_outline(num_pages: int, entries: list[tuple[str, int]]) -> bytes:
    writer = _PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    for title, page_number in entries:
        writer.add_outline_item(title, page_number)
    buf = _io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py -k TestAnalyzeAttachmentOutlineOnly -v`
Expected: FAIL with `ImportError: cannot import name 'analyze_attachment_outline_only'`

- [ ] **Step 3: Implement `analyze_attachment_outline_only`**

Insert into `src/chapter_segmentation/segmentation.py` right after the `_BOOK_TITLE_BOOKMARK_FUZZ_THRESHOLD = 90.0` line (line 1259) and before `def build_book_context`:

```python
def analyze_attachment_outline_only(pages: list[str], outline_candidates: list[ChapterCandidate]) -> dict:
    """Builds chapters directly from a PDF's embedded outline/bookmark
    candidates (see extract_outline_candidates), standalone -- no metadata
    merge, no LLM, no heuristic fallback. Every outline candidate already
    carries a resolved pdf_page_index (see ChapterCandidate's docstring),
    so this never needs content-search localization; a candidate that
    somehow lacks one is skipped rather than guessed at. Mirrors the
    "pre_located" branch analyze_attachment_with_strategies uses for
    outline candidates, factored out so it can be measured on its own
    (design spec 2026-08-07 "Production code changes").
    """
    pre_located = [c for c in outline_candidates if c.pdf_page_index is not None]
    located = [
        (
            _candidate_to_toc_entry(candidate),
            ChapterStartMatch(index=candidate.pdf_page_index, score=100.0, margin=_CONFIDENCE_MARGIN_SATURATION),
        )
        for candidate in pre_located
    ]
    located.sort(key=lambda pair: pair[1].index)
    entry_source = {entry: "outline" for entry, _match in located}
    chapters = _chapters_from_located(pages, located, entry_source=entry_source)
    for chapter in chapters:
        chapter["confidence"] = _OUTLINE_CONFIDENCE
    return {
        "total_pdf_pages": len(pages),
        "segmentation_confidence": "high" if chapters else "low",
        "chapters": chapters,
        "diagnostics": {"outline_candidates_found": len(outline_candidates)},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -k TestAnalyzeAttachmentOutlineOnly -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: add analyze_attachment_outline_only for standalone outline evaluation"
```

---

### Task 2: `analyze_attachment_llm_only` in segmentation.py

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:1247` (insert after `analyze_attachment_with_llm_fallback`, before `_OUTLINE_CONFIDENCE`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add `analyze_attachment_llm_only` to the same import block as Task 1's `analyze_attachment_outline_only`. Append to `tests/test_segmentation.py`:

```python
class TestAnalyzeAttachmentLlmOnly(unittest.IsolatedAsyncioTestCase):
    def _fake_llm(self, toc_response: str | None = None, disambiguation_response: str | None = None):
        llm = MagicMock()
        responses = [r for r in (toc_response, disambiguation_response) if r is not None]
        llm.generate = AsyncMock(side_effect=responses)
        return llm

    async def test_calls_llm_even_when_heuristic_would_succeed(self):
        # Same fixture as
        # TestAnalyzeAttachmentWithLlmFallback.test_does_not_call_llm_when_heuristic_already_succeeds
        # in this file -- a regex-parseable TOC exists, but
        # analyze_attachment_llm_only must call the LLM anyway, since it is
        # the standalone-LLM strategy, not the fallback pipeline.
        pages = [
            "CONTENTS\n"
            "Introduction ..... 1\n"
            "Comparing Citation Styles ..... 3\n"
            "Appendix ..... 5\n",
            "Introduction\nJane Author\n\nThis book explores reference management.\n\n1",
            "...continued text follows here, with enough body content on this "
            "page that it clearly reads as a real continuation of the "
            "chapter rather than a blank divider page between sections.\n\n2",
            "Comparing Citation Styles\n\nJohn Smith\n\nThis chapter examines APA and MLA.\n\n3",
            "...continued chapter text, with enough body content on this "
            "final page that it clearly reads as a real continuation of "
            "the chapter rather than a blank divider page.\n\n4",
        ]
        response = (
            '[{"title": "Introduction", "authors": ["Jane Author"], "printed_page_number": 1}, '
            '{"title": "Comparing Citation Styles", "authors": ["John Smith"], "printed_page_number": 3}]'
        )
        llm = self._fake_llm(toc_response=response)
        result = await analyze_attachment_llm_only(pages, llm)
        llm.generate.assert_called_once()
        self.assertEqual(len(result["chapters"]), 2)
        self.assertTrue(all(c["source"] == "llm" for c in result["chapters"]))

    async def test_empty_llm_response_yields_no_chapters(self):
        llm = self._fake_llm(toc_response="[]")
        pages = ["front matter"] * 20
        result = await analyze_attachment_llm_only(pages, llm)
        self.assertEqual(result["chapters"], [])
        self.assertEqual(result["diagnostics"]["toc_matches_found"], 0)

    async def test_swallows_llm_exception_and_returns_empty_result(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("network error"))
        pages = ["front matter"] * 20
        result = await analyze_attachment_llm_only(pages, llm)
        self.assertEqual(result["chapters"], [])

    async def test_llm_disambiguation_resolves_ambiguous_llm_entry(self):
        # Same fixture shape as
        # TestAnalyzeAttachmentWithLlmFallback.test_llm_disambiguation_resolves_ambiguous_chapter,
        # but the TOC entries themselves come from the LLM's own
        # extraction response (first generate() call) instead of the
        # regex heuristic, since this function never runs the heuristic
        # at all.
        filler = "Unrelated body filler text, nothing chapter-related here."
        pages = [
            "Front matter, no parseable TOC here at all.",
            filler,
            filler,
            "Alpha Overview\n\nBy Jane Author\n\nThis opening chapter surveys the field.",  # index 3
            "Omega Summary\n\nBy Jane Author\n\nThis closing chapter wraps everything up.",  # index 4
            filler,
            "Comparing Citation Styles\n\nBy Jane Doe\n\nThis chapter examines APA style only.",  # index 6
            *([filler] * 5),  # indices 7-11
            "Comparing Citation Style\n\nBy John Smith\n\nAnother chapter about MLA style.",  # index 12
            *([filler] * 7),  # indices 13-19
        ]
        self.assertEqual(len(pages), 20)
        toc_response = (
            '[{"title": "Alpha Overview", "authors": ["Jane Author"], "printed_page_number": 1}, '
            '{"title": "Comparing Citation Styles", "authors": [], "printed_page_number": 5}, '
            '{"title": "Omega Summary", "authors": ["Jane Author"], "printed_page_number": 9}]'
        )
        llm = self._fake_llm(toc_response=toc_response, disambiguation_response='{"chosen_candidate": 1}')
        result = await analyze_attachment_llm_only(pages, llm)
        self.assertEqual(result["diagnostics"]["llm_disambiguation_used"], 1)
        sources = {c["title"]: c["source"] for c in result["chapters"]}
        self.assertEqual(sources.get("Comparing Citation Styles"), "llm")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py -k TestAnalyzeAttachmentLlmOnly -v`
Expected: FAIL with `ImportError: cannot import name 'analyze_attachment_llm_only'`

- [ ] **Step 3: Implement `analyze_attachment_llm_only`**

Insert into `src/chapter_segmentation/segmentation.py` right after the end of `analyze_attachment_with_llm_fallback` (after its closing `}` / line 1247), before the `_OUTLINE_CONFIDENCE = 0.98` line:

```python
async def analyze_attachment_llm_only(pages: list[str], llm_client: LLMClient) -> dict:
    """Standalone LLM strategy: unconditionally extracts the TOC via
    llm_extract_toc_entries (never gated behind heuristic failure, unlike
    analyze_attachment_with_llm_fallback), locates each entry the same way
    the heuristic pipeline does, and uses llm_disambiguate_chapter_start
    for any entry left genuinely ambiguous. Measures the LLM strategy's
    true standalone accuracy -- the fallback pipeline only ever exercises
    the LLM after the heuristic has already failed, which hides this
    number (design spec 2026-08-07 "Production code changes"). Any LLM
    exception is swallowed and treated as "found nothing", same
    fail-safe convention as analyze_attachment_with_llm_fallback.
    """
    try:
        toc_entries = await llm_extract_toc_entries(pages, llm_client)
    except Exception:
        logger.warning("analyze_attachment_llm_only: TOC extraction failed", exc_info=True)
        toc_entries = []

    toc_page_indices = _toc_scan_indices(pages)
    located, unlocated, non_content_pages = _locate_toc_entries(pages, toc_entries, exclude_indices=toc_page_indices)
    entry_source: dict[TocEntry, str] = {e: "llm" for e in toc_entries}

    disambiguation_count = 0
    for entry in unlocated:
        candidates = locate_chapter_start_candidates(pages, entry.title, exclude_indices=toc_page_indices, authors=entry.authors)
        if len(candidates) <= 1:
            continue  # zero-candidate case is out of scope, same as the fallback pipeline
        try:
            resolved = await llm_disambiguate_chapter_start(pages, entry.title, entry.authors, candidates, llm_client)
        except Exception:
            logger.warning("analyze_attachment_llm_only: disambiguation failed for %r", entry.title, exc_info=True)
            continue
        if resolved is not None:
            located.append((entry, resolved))
            disambiguation_count += 1

    located.sort(key=lambda pair: pair[1].index)
    chapters = _chapters_from_located(pages, located, entry_source=entry_source, non_content_pages=non_content_pages)

    return {
        "total_pdf_pages": len(pages),
        "segmentation_confidence": "high" if chapters else "low",
        "chapters": chapters,
        "diagnostics": {
            "toc_matches_found": len(toc_entries),
            "toc_matches_located": len(located),
            "llm_disambiguation_used": disambiguation_count,
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -k TestAnalyzeAttachmentLlmOnly -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full segmentation test file to check for regressions**

Run: `uv run pytest tests/test_segmentation.py tests/test_segmentation_strategies.py -q`
Expected: PASS, no failures

- [ ] **Step 6: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: add analyze_attachment_llm_only for standalone LLM evaluation"
```

---

### Task 3: `evaluation/metrics.py` -- shared precision/recall/F1 scoring

**Files:**
- Create: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.metrics'`

- [ ] **Step 3: Implement `evaluation/metrics.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add shared precision/recall/F1 scoring for evaluation reports"
```

---

### Task 4: `evaluation/kisski.py` -- model listing and selection

**Files:**
- Create: `evaluation/kisski.py`
- Modify: `pyproject.toml:19` (add `httpx` to the `llm-eval` extra)
- Test: `tests/test_kisski.py`

- [ ] **Step 1: Add `httpx` to the `llm-eval` extra**

In `pyproject.toml`, change:

```toml
llm-eval = ["openai>=1.0.0"]
```

to:

```toml
llm-eval = ["openai>=1.0.0", "httpx>=0.27.0"]
```

`kisski.py` (below) imports `httpx` directly for the `/models` call, so it needs an explicit declared dependency even though `openai` already pulls httpx in transitively.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_kisski.py`:

```python
"""Unit tests for evaluation/kisski.py -- KISSKI model listing and
selection. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"LLM cache refresh"."""

import unittest
from unittest.mock import MagicMock, patch

from evaluation.kisski import KisskiModel, fetch_kisski_models, select_gap_fill, select_top5


class TestKisskiModelAvailability(unittest.TestCase):
    def test_zero_demand_is_available(self):
        self.assertEqual(KisskiModel(id="a", name="A", demand=0).availability, "available")

    def test_low_demand_is_busy(self):
        self.assertEqual(KisskiModel(id="a", name="A", demand=5).availability, "busy")

    def test_high_demand_is_very_busy(self):
        self.assertEqual(KisskiModel(id="a", name="A", demand=6).availability, "very busy")


class TestFetchKisskiModels(unittest.TestCase):
    def test_posts_to_models_endpoint_with_bearer_auth(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "model-a", "name": "Model A", "demand": 0}]}
        response.raise_for_status.return_value = None
        with patch("evaluation.kisski.httpx.post", return_value=response) as mock_post:
            models = fetch_kisski_models("https://example.test/v1", "secret-key")
        self.assertEqual(models, [KisskiModel(id="model-a", name="Model A", demand=0)])
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://example.test/v1/models")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-key")

    def test_strips_trailing_slash_on_base_url(self):
        response = MagicMock()
        response.json.return_value = {"data": []}
        response.raise_for_status.return_value = None
        with patch("evaluation.kisski.httpx.post", return_value=response) as mock_post:
            fetch_kisski_models("https://example.test/v1/", "secret-key")
        self.assertEqual(mock_post.call_args[0][0], "https://example.test/v1/models")

    def test_missing_name_falls_back_to_id(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "model-a", "demand": 0}]}
        response.raise_for_status.return_value = None
        with patch("evaluation.kisski.httpx.post", return_value=response):
            models = fetch_kisski_models("https://example.test/v1", "secret-key")
        self.assertEqual(models[0].name, "model-a")


class TestSelectTop5(unittest.TestCase):
    def test_prefers_available_over_busy(self):
        models = [
            KisskiModel(id="busy-1", name="Busy 1", demand=3),
            KisskiModel(id="avail-1", name="Avail 1", demand=0),
        ]
        selected = select_top5(models)
        self.assertEqual([m.id for m in selected], ["avail-1", "busy-1"])

    def test_excludes_very_busy(self):
        models = [KisskiModel(id="very-busy", name="Very Busy", demand=10)]
        self.assertEqual(select_top5(models), [])

    def test_caps_at_five(self):
        models = [KisskiModel(id=f"m{i}", name=f"M{i}", demand=0) for i in range(8)]
        self.assertEqual(len(select_top5(models)), 5)

    def test_ascending_demand_order_within_same_availability(self):
        models = [
            KisskiModel(id="m-demand-2", name="M2", demand=2),
            KisskiModel(id="m-demand-1", name="M1", demand=1),
        ]
        selected = select_top5(models)
        self.assertEqual([m.id for m in selected], ["m-demand-1", "m-demand-2"])


class TestSelectGapFill(unittest.TestCase):
    def test_skips_already_covered_models(self):
        models = [
            KisskiModel(id="covered", name="Covered", demand=0),
            KisskiModel(id="uncovered", name="Uncovered", demand=0),
        ]
        selected = select_gap_fill(models, covered_model_ids={"covered"})
        self.assertEqual([m.id for m in selected], ["uncovered"])

    def test_excludes_very_busy_even_if_uncovered(self):
        models = [KisskiModel(id="very-busy", name="Very Busy", demand=10)]
        self.assertEqual(select_gap_fill(models, covered_model_ids=set()), [])

    def test_respects_limit(self):
        models = [KisskiModel(id=f"m{i}", name=f"M{i}", demand=0) for i in range(8)]
        selected = select_gap_fill(models, covered_model_ids=set(), limit=3)
        self.assertEqual(len(selected), 3)

    def test_all_covered_returns_empty(self):
        models = [KisskiModel(id="a", name="A", demand=0)]
        self.assertEqual(select_gap_fill(models, covered_model_ids={"a"}), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_kisski.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.kisski'`

- [ ] **Step 4: Implement `evaluation/kisski.py`**

```python
"""KISSKI (Academic Cloud) model listing and selection -- mirrors
zotero-rag's backend/utils/kisski.py fetch_kisski_rag_models and
backend/dependencies.py _resolve_auto_select_candidates, reimplemented
standalone since this repo has no dependency on zotero-rag's backend
package. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"LLM cache refresh".
"""

from dataclasses import dataclass

import httpx

DEFAULT_KISSKI_BASE_URL = "https://chat-ai.academiccloud.de/v1"


@dataclass(frozen=True)
class KisskiModel:
    id: str
    name: str
    demand: int

    @property
    def availability(self) -> str:
        if self.demand == 0:
            return "available"
        if self.demand <= 5:
            return "busy"
        return "very busy"


def fetch_kisski_models(base_url: str, api_key: str, timeout: float = 5.0) -> list[KisskiModel]:
    """POST {base_url}/models -- same request shape as zotero-rag's
    fetch_kisski_rag_models. Raises on a network/HTTP error; the caller
    decides how to handle that."""
    url = base_url.rstrip("/") + "/models"
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    entries = response.json().get("data", [])
    return [
        KisskiModel(id=entry["id"], name=entry.get("name") or entry["id"], demand=int(entry.get("demand", 0)))
        for entry in entries
    ]


def select_top5(models: list[KisskiModel]) -> list[KisskiModel]:
    """On-demand-refresh selection: every non-'very busy' model, ascending
    by demand (so 'available' models sort before 'busy' ones), capped at
    5."""
    eligible = [m for m in models if m.availability != "very busy"]
    return sorted(eligible, key=lambda m: m.demand)[:5]


def select_gap_fill(models: list[KisskiModel], covered_model_ids: set[str], limit: int = 5) -> list[KisskiModel]:
    """Nightly-refresh selection: from non-'very busy' models not already
    in `covered_model_ids` (a model id counted as covered only once it has
    a cache entry for EVERY book in the current corpus -- see
    refresh_llm_cache.py's _fully_covered_model_ids), take up to `limit`
    in ascending-demand order."""
    eligible = [m for m in models if m.availability != "very busy" and m.id not in covered_model_ids]
    return sorted(eligible, key=lambda m: m.demand)[:limit]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_kisski.py -v`
Expected: PASS (12 tests)

- [ ] **Step 6: Commit**

```bash
git add evaluation/kisski.py tests/test_kisski.py pyproject.toml
git commit -m "feat: add KISSKI model listing and selection for LLM cache refresh"
```

---

### Task 5: `evaluation/harness.py` -- outline cache and LLM cache paths

**Files:**
- Modify: `evaluation/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_harness.py`. First add these imports near the top of the file:

```python
from chapter_segmentation.evidence.types import ChapterCandidate
from evaluation.harness import (
    outline_candidate_from_dict,
    outline_candidate_to_dict,
    public_outline_candidates_for,
)
```

Then append:

```python
class TestOutlineCandidateSerialization(unittest.TestCase):
    def test_round_trips_all_fields(self):
        candidate = ChapterCandidate(
            title="Introduction", authors=("Jane Author",), printed_page_number=1,
            pdf_page_index=5, chapter_doi="10.1/x", source="outline", metadata_confidence=0.9,
        )
        restored = outline_candidate_from_dict(outline_candidate_to_dict(candidate))
        self.assertEqual(restored, candidate)

    def test_round_trips_defaults(self):
        candidate = ChapterCandidate(title="Introduction", pdf_page_index=5)
        restored = outline_candidate_from_dict(outline_candidate_to_dict(candidate))
        self.assertEqual(restored, candidate)


class TestPublicOutlineCandidatesFor(unittest.TestCase):
    def test_returns_candidates_for_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_cache_dir = Path(tmp) / "public-cache"
            public_cache_dir.mkdir()
            candidate = ChapterCandidate(title="Introduction", pdf_page_index=5, source="outline")
            (public_cache_dir / "9999999.outline.json").write_text(
                json.dumps({"candidates": [outline_candidate_to_dict(candidate)]}), encoding="utf-8",
            )
            with patch("evaluation.harness.PUBLIC_CACHE_DIR", public_cache_dir):
                candidates = public_outline_candidates_for("9999999")
            self.assertEqual(candidates, [candidate])

    def test_returns_none_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_cache_dir = Path(tmp) / "public-cache"
            public_cache_dir.mkdir()
            with patch("evaluation.harness.PUBLIC_CACHE_DIR", public_cache_dir):
                self.assertIsNone(public_outline_candidates_for("9999999"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py -v`
Expected: FAIL with `ImportError: cannot import name 'outline_candidate_from_dict'`

- [ ] **Step 3: Implement the harness.py additions**

In `evaluation/harness.py`, add the import and constant near the top (alongside the existing `PUBLIC_CACHE_DIR`/`OCR_CACHE_DIR` definitions at lines 28-30):

```python
from chapter_segmentation.evidence.types import ChapterCandidate
```

```python
LLM_CACHE_DIR = EVAL_DIR / "llm-cache"
```

Then append these functions at the end of `evaluation/harness.py`:

```python
def outline_candidate_to_dict(candidate: ChapterCandidate) -> dict:
    return {
        "title": candidate.title,
        "authors": list(candidate.authors),
        "printed_page_number": candidate.printed_page_number,
        "pdf_page_index": candidate.pdf_page_index,
        "chapter_doi": candidate.chapter_doi,
        "source": candidate.source,
        "metadata_confidence": candidate.metadata_confidence,
    }


def outline_candidate_from_dict(data: dict) -> ChapterCandidate:
    return ChapterCandidate(
        title=data["title"],
        authors=tuple(data.get("authors", [])),
        printed_page_number=data.get("printed_page_number"),
        pdf_page_index=data.get("pdf_page_index"),
        chapter_doi=data.get("chapter_doi"),
        source=data.get("source", "outline"),
        metadata_confidence=data.get("metadata_confidence", 1.0),
    )


def public_outline_candidates_for(manifest_key: str) -> Optional[list[ChapterCandidate]]:
    """Cached outline-strategy candidates for one book from the committed
    public-cache, or None if no entry exists yet (either the book's PDF
    has no outline, or the cache hasn't been generated for it yet -- see
    evaluation/scripts/generate_public_evaluation_cache.py)."""
    cache_path = PUBLIC_CACHE_DIR / f"{manifest_key}.outline.json"
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [outline_candidate_from_dict(c) for c in data["candidates"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: PASS (all tests, including the pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add evaluation/harness.py tests/test_harness.py
git commit -m "feat: add outline-candidate cache loading to the evaluation harness"
```

---

### Task 6: Extend `generate_public_evaluation_cache.py` to write outline snapshots

**Files:**
- Modify: `evaluation/scripts/generate_public_evaluation_cache.py`
- Modify: `evaluation/CLAUDE.md` (documentation note)

- [ ] **Step 1: Add the outline-cache write**

In `evaluation/scripts/generate_public_evaluation_cache.py`, add this import near the top (alongside the existing imports):

```python
from chapter_segmentation.evidence.outline_strategy import extract_outline_candidates
from evaluation.harness import outline_candidate_to_dict
```

Then, inside the `for pdf_path, _expected_path, book in available_books():` loop, right after the existing pages-cache write (`cache_path.write_text(...)` / the `print(f"{manifest_key}: OK, wrote {cache_path}")` line), add:

```python
        outline_candidates = extract_outline_candidates(file_bytes)
        outline_cache_path = PUBLIC_CACHE_DIR / f"{manifest_key}.outline.json"
        outline_cache_path.write_text(
            json.dumps(
                {"candidates": [outline_candidate_to_dict(c) for c in outline_candidates]},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"{manifest_key}: wrote {len(outline_candidates)} outline candidate(s) to {outline_cache_path}")
```

`extract_outline_candidates` never raises (see its own docstring), so no extra try/except is needed around this block.

- [ ] **Step 2: Update the module docstring**

Update the top-of-file docstring's first paragraph to mention the new output:

```python
"""Generate evaluation/public-cache/ -- a
redacted, git-trackable corpus safe to commit and distribute (real
navigational/bibliographic text kept verbatim, chapter prose replaced with
random real words in the book's own language) plus, per book, a resolved
outline-strategy candidate snapshot (<key>.outline.json -- titles/authors/
page indices only, no prose) so the outline strategy is also testable
without the real PDF -- see evaluation/README.md for the redaction
rationale and workflow, and docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
for the outline-snapshot rationale.
```

- [ ] **Step 3: Update `evaluation/CLAUDE.md`'s public-cache bullet**

In `evaluation/CLAUDE.md`, find the bullet starting `**`public-cache/`** — a redacted, git-tracked snapshot...` and add one sentence at the end of its first paragraph (before the "Regenerate it with" sentence):

```
Also writes `<key>.outline.json` per book -- a resolved snapshot of
`extract_outline_candidates`' output (titles/authors/page indices only),
letting the outline strategy be evaluated in CI without the real PDF.
```

- [ ] **Step 4: Regenerate the cache (requires the real evaluation PDFs locally)**

This step needs the actual evaluation PDFs present at `evaluation/*.pdf` (fetched via `uv run python evaluation/scripts/fetch_evaluation_pdfs.py` for open-access books, or already present for `manifest.local.json` books). **If you don't have these PDFs, skip this step** -- the code from Steps 1-3 is already complete and correct; `public_outline_candidates_for` will just return `None` for every book (rendered as "N/A" in the report) until someone who has the PDFs runs this once.

If you do have the PDFs:

```bash
uv run python evaluation/scripts/generate_public_evaluation_cache.py
```

Expected: prints one `wrote N outline candidate(s)` line per book (N may be 0 for books with no embedded outline), no `VERIFY FAILED` or `FAILED` lines (aside from the pre-existing, documented `9783031466373` verify failure -- see `evaluation/CLAUDE.md`'s "Known failure modes").

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_public_evaluation_cache.py evaluation/CLAUDE.md
# If Step 4 was run, also: git add evaluation/public-cache/
git commit -m "feat: cache resolved outline-strategy candidates alongside redacted pages"
```

---

### Task 7: `evaluation/report_html.py` -- shared table renderer

**Files:**
- Create: `evaluation/report_html.py`
- Test: `tests/test_report_html.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report_html.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report_html.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.report_html'`

- [ ] **Step 3: Implement `evaluation/report_html.py`**

```python
"""Shared HTML table rendering for generate_report.py's main report and
its LLM detail page -- one renderer so both pages look and behave
identically. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"Metrics and rendering (shared code)".
"""

from evaluation.metrics import Metrics

_TableCell = tuple[Metrics, float] | None  # (metrics, elapsed_seconds), or None for "not run"


def _cell_html(cell: _TableCell, is_best: bool) -> str:
    if cell is None:
        return "<td>N/A</td>"
    metrics, elapsed_seconds = cell
    style = ' style="background:#e6ffe6; font-weight:bold;"' if is_best else ""
    return (
        f"<td{style}>P={metrics.precision:.2f} R={metrics.recall:.2f} F1={metrics.f1:.2f}<br>"
        f"{metrics.true_positives}/{metrics.found_count} found, "
        f"{metrics.true_positives}/{metrics.expected_count} expected<br>"
        f"{elapsed_seconds:.2f}s</td>"
    )


def render_strategy_tables(
    title: str,
    description_html: str,
    strategy_names: list[str],
    per_document: dict[str, dict[str, _TableCell]],
    aggregates: dict[str, Metrics],
    aggregate_times: dict[str, float],
) -> str:
    """
    strategy_names: column order for the per-document table.
    per_document: {document_key: {strategy_name: (Metrics, elapsed_seconds) or None}}.
        None (or a missing key) means the strategy produced no result for
        that document -- rendered as "N/A".
    aggregates: {strategy_name: Metrics} -- micro-aggregate across every
        document that strategy actually ran on.
    aggregate_times: {strategy_name: total_elapsed_seconds}.
    Returns a full <html> document string.
    """
    doc_rows = []
    for doc_key in sorted(per_document):
        cells = per_document[doc_key]
        best_f1 = max(
            (cell[0].f1 for cell in cells.values() if cell is not None),
            default=None,
        )
        row_cells = []
        for strategy in strategy_names:
            cell = cells.get(strategy)
            is_best = cell is not None and best_f1 is not None and cell[0].f1 == best_f1
            row_cells.append(_cell_html(cell, is_best))
        doc_rows.append(f"<tr><td>{doc_key}</td>{''.join(row_cells)}</tr>")

    ranked_strategies = sorted(aggregates, key=lambda s: aggregates[s].f1, reverse=True)
    agg_rows = []
    for strategy in ranked_strategies:
        m = aggregates[strategy]
        t = aggregate_times.get(strategy, 0.0)
        agg_rows.append(
            f"<tr><td>{strategy}</td><td>{m.precision:.2f}</td><td>{m.recall:.2f}</td>"
            f"<td>{m.f1:.2f}</td><td>{m.true_positives}/{m.found_count} found, "
            f"{m.true_positives}/{m.expected_count} expected</td><td>{t:.2f}s</td></tr>"
        )

    doc_header = "".join(f"<th>{s}</th>" for s in strategy_names)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }}</style>
</head><body>
<h1>{title}</h1>
{description_html}
<h2>Per document</h2>
<table>
<tr><th>Book</th>{doc_header}</tr>
{"".join(doc_rows)}
</table>
<h2>Per strategy (aggregate, ordered by F1)</h2>
<table>
<tr><th>Strategy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Found / Expected</th><th>Total time</th></tr>
{"".join(agg_rows)}
</table>
</body></html>
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report_html.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/report_html.py tests/test_report_html.py
git commit -m "feat: add shared strategy-comparison table renderer"
```

---

### Task 8: Rewrite `evaluation/generate_report.py`

**Files:**
- Modify: `evaluation/generate_report.py` (full rewrite)
- Test: `tests/test_generate_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_report.py`:

```python
"""Unit tests for evaluation/generate_report.py -- the auto-published,
zero-API-call report covering heuristic, outline, and (if cached) the
best-performing LLM model."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.generate_report import _best_llm_model, generate


def _expected_json(chapters: list[dict]) -> str:
    return json.dumps({"chapters": chapters})


class TestBestLlmModel(unittest.TestCase):
    def test_returns_none_with_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.generate_report.LLM_CACHE_DIR", Path(tmp)):
                self.assertIsNone(_best_llm_model([("book-a", [{"pdf_start_index": 0, "pdf_end_index": 5}])]))

    def test_picks_the_higher_scoring_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            expected = [{"pdf_start_index": 0, "pdf_end_index": 5}]
            (cache_dir / "book-a.json").write_text(json.dumps({
                "models": {
                    "good-model": {"chapters": expected, "elapsed_seconds": 1.0, "demand_at_run": 0},
                    "bad-model": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0},
                }
            }), encoding="utf-8")
            with patch("evaluation.generate_report.LLM_CACHE_DIR", cache_dir):
                best = _best_llm_model([("book-a", expected)])
            self.assertEqual(best, "good-model")


class TestGenerate(unittest.TestCase):
    def test_writes_main_report_and_llm_detail_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_dir = tmp_path / "evaluation"
            public_cache_dir = eval_dir / "public-cache"
            llm_cache_dir = eval_dir / "llm-cache"
            public_cache_dir.mkdir(parents=True)
            llm_cache_dir.mkdir(parents=True)
            out_dir = tmp_path / "public"

            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            (eval_dir / "book-a.expected.json").write_text(_expected_json(chapters), encoding="utf-8")
            (public_cache_dir / "book-a.pages.json").write_text(
                json.dumps({"pages": ["Introduction\nBody text.", "more", "more", "more"]}), encoding="utf-8",
            )
            book = {"filename": "book-a.pdf", "title": "Book A"}

            with patch("evaluation.harness.EVAL_DIR", eval_dir), \
                 patch("evaluation.harness.PUBLIC_CACHE_DIR", public_cache_dir), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]), \
                 patch("evaluation.generate_report.LLM_CACHE_DIR", llm_cache_dir), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            self.assertTrue((out_dir / "index.html").exists())
            self.assertTrue((out_dir / "llm" / "index.html").exists())
            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("book-a", main_html)
            self.assertIn("N/A", main_html)  # outline has no cache entry in this fixture
            llm_html = (out_dir / "llm" / "index.html").read_text(encoding="utf-8")
            self.assertIn("No cached LLM results yet", llm_html)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_report.py -v`
Expected: FAIL (`ImportError` or `AttributeError` -- `_best_llm_model`/`LLM_CACHE_DIR`/etc. don't exist yet in the current `generate_report.py`)

- [ ] **Step 3: Rewrite `evaluation/generate_report.py`**

Replace the entire file content with:

```python
#!/usr/bin/env python3
"""Generates a prose-free static results page from the committed
public-cache corpus -- see design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md.
Runs the heuristic and outline strategies live (no network/API calls); if
evaluation/llm-cache/ has cached LLM results, folds in the single
best-performing cached model too. Also regenerates public/llm/index.html,
a full breakdown of every cached LLM model. No LLM call anywhere in this
path; a plain f-string template, no templating-engine dependency.

    uv run python evaluation/generate_report.py --out public/
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapter_segmentation.segmentation import analyze_attachment, analyze_attachment_outline_only
from evaluation.harness import (
    LLM_CACHE_DIR,
    available_public_books,
    public_outline_candidates_for,
    public_pages_for,
)
from evaluation.metrics import MicroAggregate, precision_recall_f1
from evaluation.report_html import render_strategy_tables

HEURISTIC = "heuristic"
OUTLINE = "outline"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_llm_cache(manifest_key: str) -> dict:
    cache_path = LLM_CACHE_DIR / f"{manifest_key}.json"
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})


def _best_llm_model(books: list[tuple[str, list[dict]]]) -> str | None:
    """books: [(manifest_key, expected_chapters)]. Picks the cached model
    with the highest micro-F1 aggregated across every book that has a
    cache entry for it (a model with partial corpus coverage is still
    scored, on however many books it has -- see design spec's "LLM
    results cache"), ties broken by lower total time. Returns None if no
    book has any cached LLM result at all."""
    per_model_aggregate: dict[str, MicroAggregate] = {}
    for manifest_key, expected in books:
        for model_id, entry in _load_llm_cache(manifest_key).items():
            agg = per_model_aggregate.setdefault(model_id, MicroAggregate())
            agg.add(precision_recall_f1(expected, entry["chapters"]), entry["elapsed_seconds"])
    if not per_model_aggregate:
        return None
    return max(
        per_model_aggregate,
        key=lambda model_id: (
            per_model_aggregate[model_id].compute().f1,
            -per_model_aggregate[model_id].total_elapsed_seconds,
        ),
    )


def generate(out_dir: Path) -> None:
    books = available_public_books()
    expected_by_key = {
        key: json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
        for key, expected_path, _book in books
    }

    best_llm_model = _best_llm_model(list(expected_by_key.items()))
    llm_strategy_name = f"LLM ({best_llm_model})" if best_llm_model else None
    strategy_names = [HEURISTIC, OUTLINE] + ([llm_strategy_name] if llm_strategy_name else [])

    per_document: dict[str, dict] = {}
    heuristic_agg, outline_agg, llm_agg = MicroAggregate(), MicroAggregate(), MicroAggregate()

    for manifest_key, _expected_path, _book in books:
        pages = public_pages_for(manifest_key)
        expected = expected_by_key[manifest_key]
        cells: dict = {}

        start = time.perf_counter()
        heuristic_result = analyze_attachment(pages)
        heuristic_elapsed = time.perf_counter() - start
        heuristic_metrics = precision_recall_f1(expected, heuristic_result["chapters"])
        heuristic_agg.add(heuristic_metrics, heuristic_elapsed)
        cells[HEURISTIC] = (heuristic_metrics, heuristic_elapsed)

        outline_candidates = public_outline_candidates_for(manifest_key)
        if outline_candidates is not None:
            start = time.perf_counter()
            outline_result = analyze_attachment_outline_only(pages, outline_candidates)
            outline_elapsed = time.perf_counter() - start
            outline_metrics = precision_recall_f1(expected, outline_result["chapters"])
            outline_agg.add(outline_metrics, outline_elapsed)
            cells[OUTLINE] = (outline_metrics, outline_elapsed)
        else:
            cells[OUTLINE] = None

        if llm_strategy_name:
            llm_entry = _load_llm_cache(manifest_key).get(best_llm_model)
            if llm_entry:
                llm_metrics = precision_recall_f1(expected, llm_entry["chapters"])
                llm_agg.add(llm_metrics, llm_entry["elapsed_seconds"])
                cells[llm_strategy_name] = (llm_metrics, llm_entry["elapsed_seconds"])
            else:
                cells[llm_strategy_name] = None

        per_document[manifest_key] = cells

    aggregates = {HEURISTIC: heuristic_agg.compute(), OUTLINE: outline_agg.compute()}
    aggregate_times = {HEURISTIC: heuristic_agg.total_elapsed_seconds, OUTLINE: outline_agg.total_elapsed_seconds}
    if llm_strategy_name:
        aggregates[llm_strategy_name] = llm_agg.compute()
        aggregate_times[llm_strategy_name] = llm_agg.total_elapsed_seconds

    description = """<p>Each book has a hand-verified <code>*.expected.json</code> ground truth (real
chapter boundaries as exact PDF page ranges). Each strategy below is run
independently against the same pages -- no pipeline merge/fallback logic
is involved, so this reflects each strategy's own standalone accuracy, not
a production routing decision. A match requires the exact same page range
-- no partial credit. For per-book root-cause notes, see
<a href="https://github.com/cboulanger/chapter-segmentation/blob/main/evaluation/RESULTS.md">RESULTS.md</a>.
The full breakdown of every LLM model ever evaluated (not just the best)
is at <a href="llm/index.html">llm/index.html</a>.</p>"""

    html = render_strategy_tables(
        title="chapter-segmentation: public-cache corpus results",
        description_html=description,
        strategy_names=strategy_names,
        per_document=per_document,
        aggregates=aggregates,
        aggregate_times=aggregate_times,
    )
    html = html.replace(
        "</body></html>",
        f"<p>Generated from commit {_git_sha()}.</p></body></html>",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    _generate_llm_detail_page(out_dir, list(expected_by_key.items()))


def _generate_llm_detail_page(out_dir: Path, books: list[tuple[str, list[dict]]]) -> None:
    model_ids: set[str] = set()
    for manifest_key, _expected in books:
        model_ids.update(_load_llm_cache(manifest_key).keys())

    if not model_ids:
        html = (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<title>chapter-segmentation LLM results</title></head><body>"
            "<h1>chapter-segmentation: LLM strategy results</h1>"
            "<p>No cached LLM results yet -- run "
            "<code>evaluation/refresh_llm_cache.py</code>.</p></body></html>"
        )
    else:
        per_document: dict[str, dict] = {}
        aggregates_acc = {model_id: MicroAggregate() for model_id in model_ids}
        for manifest_key, expected in books:
            cache = _load_llm_cache(manifest_key)
            cells: dict = {}
            for model_id in model_ids:
                entry = cache.get(model_id)
                if entry is None:
                    cells[model_id] = None
                    continue
                metrics = precision_recall_f1(expected, entry["chapters"])
                aggregates_acc[model_id].add(metrics, entry["elapsed_seconds"])
                cells[model_id] = (metrics, entry["elapsed_seconds"])
            per_document[manifest_key] = cells

        aggregates = {model_id: acc.compute() for model_id, acc in aggregates_acc.items()}
        aggregate_times = {model_id: acc.total_elapsed_seconds for model_id, acc in aggregates_acc.items()}
        html = render_strategy_tables(
            title="chapter-segmentation: LLM strategy results (all cached models)",
            description_html=(
                "<p>Every KISSKI model ever evaluated by "
                "<code>evaluation/refresh_llm_cache.py</code>, run standalone via "
                "<code>analyze_attachment_llm_only</code> (no heuristic fallback). "
                'See <a href="../index.html">the main report</a> for how the single '
                "best-performing model compares against the heuristic and outline "
                "strategies.</p>"
            ),
            strategy_names=sorted(model_ids),
            per_document=per_document,
            aggregates=aggregates,
            aggregate_times=aggregate_times,
        )

    llm_dir = out_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    (llm_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="public")
    args = parser.parse_args()
    generate(Path(args.out))
```

Note: `generate_report.py` only imports `available_public_books`/`public_pages_for`/`public_outline_candidates_for`/`LLM_CACHE_DIR` from `evaluation.harness` -- it never references `EVAL_DIR`, `PUBLIC_CACHE_DIR`, or `load_manifest_books` by name, since those are internal to how `available_public_books()` locates books. That's why the test in Step 1 patches `evaluation.harness.EVAL_DIR`/`PUBLIC_CACHE_DIR`/`load_manifest_books` (where `available_public_books()` actually looks them up at call time) but patches `evaluation.generate_report.LLM_CACHE_DIR` and `evaluation.generate_report.public_outline_candidates_for` (read directly inside `generate_report.py`'s own functions).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_report.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Manual smoke test against the real committed corpus**

Run: `uv run python evaluation/generate_report.py --out /tmp/report-smoke-test`
Expected: exits 0; `/tmp/report-smoke-test/index.html` and `/tmp/report-smoke-test/llm/index.html` both exist. Open `/tmp/report-smoke-test/index.html` in a browser and confirm it shows a heuristic column for every book, an outline column (N/A for books with no outline cache yet), and no LLM column (no cache exists yet at this point in the plan).

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: PASS, no failures

- [ ] **Step 7: Commit**

```bash
git add evaluation/generate_report.py tests/test_generate_report.py
git commit -m "feat: rewrite generate_report.py to score heuristic/outline/best-LLM strategies independently"
```

---

### Task 9: `evaluation/refresh_llm_cache.py` and retiring the old LLM-fallback eval script

**Files:**
- Create: `evaluation/refresh_llm_cache.py`
- Delete: `evaluation/scripts/evaluate_chapter_segmentation_llm_fallback.py`
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_refresh_llm_cache.py` (covers the pure/file-based logic only -- no real KISSKI or LLM calls):

```python
"""Unit tests for evaluation/refresh_llm_cache.py's pure logic: coverage
computation and cache upserts. The network-calling _main() orchestration
is exercised manually (see evaluation/README.md), not here."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.refresh_llm_cache import _fully_covered_model_ids, _upsert_cache


class TestFullyCoveredModelIds(unittest.TestCase):
    def test_no_cache_files_means_nothing_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", Path(tmp)):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), set())

    def test_model_present_in_every_book_is_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for key in ("book-a", "book-b"):
                (cache_dir / f"{key}.json").write_text(
                    json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                    encoding="utf-8",
                )
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), {"model-x"})

    def test_model_missing_from_one_book_is_not_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), set())

    def test_a_book_with_no_cache_file_at_all_means_zero_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                self.assertEqual(_fully_covered_model_ids(["book-a", "book-b"]), set())


class TestUpsertCache(unittest.TestCase):
    def test_creates_new_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                _upsert_cache("book-a", "model-x", [{"title": "Intro"}], 1.5, demand=0)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertEqual(data["models"]["model-x"]["elapsed_seconds"], 1.5)
            self.assertEqual(data["models"]["model-x"]["demand_at_run"], 0)

    def test_preserves_other_models_when_upserting(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-old": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            with patch("evaluation.refresh_llm_cache.LLM_CACHE_DIR", cache_dir):
                _upsert_cache("book-a", "model-new", [], 2.0, demand=1)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertIn("model-old", data["models"])
            self.assertIn("model-new", data["models"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.refresh_llm_cache'`

- [ ] **Step 3: Delete the old LLM-fallback eval script**

```bash
git rm evaluation/scripts/evaluate_chapter_segmentation_llm_fallback.py
```

Its purpose (manually running the evaluation corpus through an LLM and printing precision/recall) is fully superseded by `refresh_llm_cache.py` (below) plus `generate_report.py`'s LLM columns -- the new pair adds caching and multi-model coverage the old script never had.

- [ ] **Step 4: Implement `evaluation/refresh_llm_cache.py`**

```python
#!/usr/bin/env python3
"""Refreshes evaluation/llm-cache/ -- the only script in this repo that
spends real KISSKI API budget. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"LLM cache refresh".

Reads KISSKI_API_KEY from the environment. Locally, source it from
zotero-rag's .env, e.g.:

    export KISSKI_API_KEY=$(grep '^KISSKI_API_KEY=' ../zotero-rag/.env | cut -d= -f2-)
    uv run python evaluation/refresh_llm_cache.py --mode top5

In CI it comes from a repository secret (see
.github/workflows/refresh-llm-cache.yml). Not a pytest test.

--mode top5 (default): refreshes the current 5 least-busy models,
unconditionally, even if already cached -- a quick manual sanity check.

--mode fill-gaps: finds non-"very busy" models not yet cached for EVERY
book in the current public corpus, and runs up to 5 of those -- how the
cache grows to cover every model over time (see the nightly schedule in
the workflow above).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapter_segmentation.segmentation import analyze_attachment_llm_only
from evaluation.harness import LLM_CACHE_DIR, available_public_books, public_pages_for
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models, select_gap_fill, select_top5


class _OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) backed by
    any OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float,
        is_valid: Optional[Callable[[str], bool]] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def _fully_covered_model_ids(manifest_keys: list[str]) -> set[str]:
    """A model id counts as covered only if EVERY given book's cache entry
    already has it. A book with no cache file at all has zero coverage --
    every model is still a gap for it."""
    per_book_model_ids = []
    for manifest_key in manifest_keys:
        cache_path = LLM_CACHE_DIR / f"{manifest_key}.json"
        if not cache_path.exists():
            return set()
        models = json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})
        per_book_model_ids.append(set(models))
    return set.intersection(*per_book_model_ids) if per_book_model_ids else set()


def _upsert_cache(manifest_key: str, model_id: str, chapters: list[dict], elapsed_seconds: float, demand: int) -> None:
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = LLM_CACHE_DIR / f"{manifest_key}.json"
    data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"models": {}}
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["models"][model_id] = {"chapters": chapters, "elapsed_seconds": elapsed_seconds, "demand_at_run": demand}
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def _main(mode: str, base_url: str) -> int:
    api_key = os.environ["KISSKI_API_KEY"]
    books = available_public_books()
    if not books:
        print("No public-cache evaluation books present.")
        return 1
    manifest_keys = [key for key, _expected_path, _book in books]

    all_models = fetch_kisski_models(base_url, api_key)
    if mode == "top5":
        selected = select_top5(all_models)
    else:
        selected = select_gap_fill(all_models, _fully_covered_model_ids(manifest_keys))

    if not selected:
        print("No models to run (fill-gaps: every non-busy model already fully covered).")
        return 0

    print(f"Selected models: {[m.id for m in selected]}")
    for model in selected:
        llm_client = _OpenAICompatibleLLMClient(model=model.id, base_url=base_url, api_key=api_key)
        for manifest_key, _expected_path, _book in books:
            pages = public_pages_for(manifest_key)
            start = time.perf_counter()
            result = await analyze_attachment_llm_only(pages, llm_client)
            elapsed = time.perf_counter() - start
            _upsert_cache(manifest_key, model.id, result["chapters"], elapsed, model.demand)
            print(f"{manifest_key} / {model.id}: {len(result['chapters'])} chapters, {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["top5", "fill-gaps"], default="top5")
    parser.add_argument("--base-url", default=DEFAULT_KISSKI_BASE_URL)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(mode=args.mode, base_url=args.base_url)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: PASS, no failures (confirms removing the old LLM-fallback script broke nothing else)

- [ ] **Step 7: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "feat: add refresh_llm_cache.py, replacing the old LLM-fallback eval script"
```

---

### Task 10: GitHub Actions workflow for the LLM cache refresh

**Files:**
- Create: `.github/workflows/refresh-llm-cache.yml`

- [ ] **Step 1: Create the workflow**

```yaml
name: Refresh LLM cache

on:
  workflow_dispatch: {}
  schedule:
    - cron: "0 3 * * *"

permissions:
  contents: write

concurrency:
  group: refresh-llm-cache
  cancel-in-progress: false

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra llm-eval
      - name: Select mode
        run: |
          if [ "${{ github.event_name }}" = "schedule" ]; then
            echo "MODE=fill-gaps" >> "$GITHUB_ENV"
          else
            echo "MODE=top5" >> "$GITHUB_ENV"
          fi
      - name: Refresh cache
        run: uv run python evaluation/refresh_llm_cache.py --mode "$MODE"
        env:
          KISSKI_API_KEY: ${{ secrets.KISSKI_API_KEY }}
      - name: Commit and push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add evaluation/llm-cache/
          if git diff --cached --quiet; then
            echo "No cache changes"
          else
            git commit -m "chore: refresh LLM evaluation cache ($MODE)"
            git push
          fi
```

- [ ] **Step 2: Validate the YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/refresh-llm-cache.yml'))"`
Expected: no output, exit code 0 (confirms valid YAML; this does not validate GitHub Actions semantics, only that the file parses)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/refresh-llm-cache.yml
git commit -m "ci: add manual/nightly LLM cache refresh workflow"
```

- [ ] **Step 4: Manual follow-up (not automatable from this plan)**

Tell the user directly (do not attempt this yourself): a repository admin needs to add a `KISSKI_API_KEY` secret under Settings > Secrets and variables > Actions before this workflow can succeed. Until that secret exists, `workflow_dispatch` runs and the nightly schedule will fail at the "Refresh cache" step with a `KeyError: 'KISSKI_API_KEY'` -- expected, and harmless (no cache files change, nothing gets committed).

---

### Task 11: Update `evaluation/README.md`

**Files:**
- Modify: `evaluation/README.md`

- [ ] **Step 1: Replace the "LLM-fallback evaluation" section**

Find this section (currently right after "Running an evaluation"'s main paragraph block):

```markdown
### LLM-fallback evaluation

`evaluation/scripts/evaluate_chapter_segmentation_llm_fallback.py` runs the same
evaluation set through `analyze_attachment_with_llm_fallback` instead of
the pure-heuristic `analyze_attachment` (see
`docs/superpowers/specs/2026-07-25-llm-chapter-segmentation-fallback-design.md`).
Unlike the harness above, this requires a real, working LLM (reads normal
app settings/API keys) and costs a paid API call per book, so it's a
manual script, not a pytest test:

```bash
uv run python evaluation/scripts/evaluate_chapter_segmentation_llm_fallback.py
```

It prints the same precision/recall table format as the harness above, plus
per-book counts of how often each fallback path (`llm_toc_extraction_used`,
`llm_disambiguation_used`) actually fired. Run it after any prompt or
heuristic change to check whether the fallback is still net-helpful on the
real evaluation set -- record what you find in `RESULTS.md`.
```

Replace it with:

```markdown
### Per-strategy evaluation report

`evaluation/generate_report.py` (published automatically to GitHub Pages
on every push to `main` -- see `.github/workflows/publish-results.yml`)
scores the heuristic and outline strategies independently against the
public-cache corpus -- no pipeline merge/fallback decision is involved,
so each strategy's own standalone accuracy is visible, not just which one
a production run happened to pick. It costs no API calls and needs no
PDFs; run it locally the same way CI does:

```bash
uv run python evaluation/generate_report.py --out public/
```

Produces two pages, both using the same table format: `public/index.html`
(one row per book x strategy, with precision/recall/F1/time, best-F1 cell
per row marked, plus a per-strategy summary ordered by aggregate F1) and
`public/llm/index.html` (see "LLM strategy evaluation" below).

### LLM strategy evaluation

Unlike the heuristic and outline strategies, evaluating the LLM strategy
costs real KISSKI API budget, so it is decoupled from report generation:
`evaluation/refresh_llm_cache.py` is the only script that calls an LLM,
and it writes its results into `evaluation/llm-cache/<book>.json` (raw
chapters found + timing per model, committed to git) rather than printing
a report directly. `evaluation/generate_report.py` then reads that cache
for free on every run -- folding the single best-performing cached model
into the main report as an "LLM (\<model\>)" column, and rendering every
cached model's full breakdown at `public/llm/index.html`.

Run it manually, with `KISSKI_API_KEY` in the environment (locally, source
it from `zotero-rag`'s `.env`):

```bash
export KISSKI_API_KEY=$(grep '^KISSKI_API_KEY=' ../zotero-rag/.env | cut -d= -f2-)
uv run python evaluation/refresh_llm_cache.py --mode top5
```

`--mode top5` (the default) refreshes the 5 currently-least-busy KISSKI
models. `--mode fill-gaps` instead finds non-busy models not yet cached
for every book in the corpus and runs up to 5 of those -- this is what
`.github/workflows/refresh-llm-cache.yml`'s nightly schedule uses, so the
cache grows to cover every available/busy model over time without paying
to re-run models it already has complete data for. The same workflow also
exposes a manual `workflow_dispatch` trigger (using `--mode top5`) for an
on-demand refresh, e.g. right after a prompt change, to sanity-check the
current best models. Either trigger commits the updated cache files
straight to `main`, which republishes the report automatically.
```

- [ ] **Step 2: Update the "Strategy-pipeline evaluation" section's cross-reference**

That section (unchanged in behavior) currently reads fine on its own; add one sentence at its end noting the relationship to the new per-strategy report, right after `-- record what you find in RESULTS.md.`:

```markdown
(This script evaluates the *merged pipeline's* Crossref/Zotero-catalog
behavior specifically -- for the outline and LLM strategies evaluated
independently of any pipeline decision, see "Per-strategy evaluation
report" and "LLM strategy evaluation" above.)
```

- [ ] **Step 3: Commit**

```bash
git add evaluation/README.md
git commit -m "docs: document the per-strategy report and LLM cache refresh workflow"
```

---

### Task 12: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS, no failures, no unexpected skips beyond the pre-existing `integration`-marked tests

- [ ] **Step 2: Run the integration accuracy suite if the real evaluation PDFs are present locally**

Run: `uv run pytest tests/test_segmentation_accuracy.py tests/test_public_evaluation_cache_parity.py -q -s -m integration`
Expected: PASS (or a clean SKIP per book without a local PDF/cache entry) -- confirms nothing in this plan changed `analyze_attachment`'s own behavior

- [ ] **Step 3: Regenerate the main report and eyeball it**

Run: `uv run python evaluation/generate_report.py --out /tmp/final-report-check`
Expected: exits 0. Open `/tmp/final-report-check/index.html` in a browser: confirm the per-document table shows heuristic and outline columns (outline shows real numbers for any book with a `.outline.json` cache entry from Task 6, "N/A" otherwise), the per-strategy summary table is ordered with the highest-F1 strategy first, and the page links to `llm/index.html`.

- [ ] **Step 4: Confirm the old script is gone and nothing still references it**

Run: `grep -rn "evaluate_chapter_segmentation_llm_fallback" --include="*.py" --include="*.md" --include="*.yml" .`
Expected: no output (the file itself is deleted; no remaining doc/code references)

- [ ] **Step 5: Review the full diff before considering this plan done**

Run: `git log --oneline docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md..HEAD` and `git diff docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md~1..HEAD --stat`
Expected: one commit per task above (12 commits), touching exactly the files each task named -- no stray/unexpected files changed
