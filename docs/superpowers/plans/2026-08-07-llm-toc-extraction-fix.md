# Fix llm_extract_toc_entries truncation and oversized input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `llm_extract_toc_entries` from silently returning zero chapters on large-TOC books (output truncated by a fixed `max_tokens=1024`) and from feeding oversized, noisy input to models with smaller context windows.

**Architecture:** Three small, independently-testable private helpers added to `src/chapter_segmentation/segmentation.py`, then wired into `llm_extract_toc_entries`'s existing body with no signature change: `_llm_scan_indices` (prefer the regex heuristic's detected TOC page range over the blind front/back-matter fraction), `_extract_with_retry` (retry once at a much higher `max_tokens` when the first response doesn't parse as JSON), `_classify_llm_failure` (bucket the final failure reason for logging). Both existing callers (`analyze_attachment_with_llm_fallback`, `analyze_attachment_llm_only`) pick up the fix automatically.

**Tech Stack:** Python 3.12, unittest (`unittest.IsolatedAsyncioTestCase` for async LLM-calling code).

Full design: `docs/superpowers/specs/2026-08-07-llm-toc-extraction-fix-design.md`.

---

### Task 1: `_llm_scan_indices`

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:501` (insert new function between the `_LLM_TOC_EXTRACTION_PROMPT` constant, which ends at line 500, and `async def llm_extract_toc_entries` at line 503)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_segmentation.py`, add `_llm_scan_indices` to the existing private-helper import tuple (the one already importing `_toc_scan_indices`, near line 13):

```python
from chapter_segmentation.segmentation import (
    TocEntry,
    extract_page_texts_from_pdf_bytes,
    extract_page_texts_for_analysis,
    find_toc_candidates,
    llm_extract_toc_entries,
    load_cached_analysis,
    pages_need_ocr,
    save_analysis_cache,
    _toc_scan_indices,
    _llm_scan_indices,
    analyze_attachment_with_llm_fallback,
    analyze_attachment_outline_only,
    analyze_attachment_llm_only,
)
```

Then add this new test class right after `TestFindTocCandidates` (i.e. immediately before `class TestLlmExtractTocEntries` at line 392):

```python
class TestLlmScanIndices(unittest.TestCase):
    _FILLER_PAGE = "Ordinary body filler text, nothing chapter-related here at all."

    def test_falls_back_to_blind_fraction_when_no_heuristic_toc_found(self):
        pages = [self._FILLER_PAGE] * 20
        self.assertEqual(_llm_scan_indices(pages), sorted(_toc_scan_indices(pages)))

    def test_narrows_to_padded_heuristic_toc_page(self):
        # 40 pages so the blind fraction (front 15% -> {0..5}, back 5% ->
        # {38,39}) is wide enough to prove real narrowing: the heuristic TOC
        # sits at index 3, inside the front zone, but _llm_scan_indices
        # should return only that page +-1, not the whole 8-page blind zone.
        pages = (
            [self._FILLER_PAGE] * 3
            + [
                "CONTENTS\n"
                "Introduction to Reference Management ..... 1\n"
                "Comparing Citation Styles ..... 45\n"
                "Zotero in Practice ..... 60\n"
            ]
            + [self._FILLER_PAGE] * 36
        )
        self.assertEqual(len(pages), 40)
        self.assertEqual(sorted(_toc_scan_indices(pages)), [0, 1, 2, 3, 4, 5, 38, 39])
        self.assertEqual(_llm_scan_indices(pages), [2, 3, 4])

    def test_clamps_padding_at_document_start(self):
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 60\n"
        ] + [self._FILLER_PAGE] * 39
        self.assertEqual(_llm_scan_indices(pages), [0, 1])

    def test_returns_empty_list_for_empty_pages(self):
        self.assertEqual(_llm_scan_indices([]), [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestLlmScanIndices -v`
Expected: FAIL with `ImportError: cannot import name '_llm_scan_indices'`

- [ ] **Step 3: Implement `_llm_scan_indices`**

In `src/chapter_segmentation/segmentation.py`, insert this new function at line 501 (right after `_LLM_TOC_EXTRACTION_PROMPT`'s closing `"""` and blank line, before `async def llm_extract_toc_entries`):

```python
def _llm_scan_indices(pages: list[str]) -> list[int]:
    """The page range llm_extract_toc_entries sends to the LLM. Prefers a
    narrow +-1-page-padded range around any TOC page find_toc_candidates
    (the regex heuristic) already located -- far less input text, and far
    less redacted/irrelevant body prose bleeding into the LLM's "authors"
    field, than the blind front/back-matter fraction. Falls back to the
    blind fraction only when the heuristic found nothing -- the case that
    matters most in practice: analyze_attachment_with_llm_fallback only
    ever calls llm_extract_toc_entries when the heuristic already found
    zero usable entries, so this narrowing is a no-op there; its full
    effect is felt by the standalone analyze_attachment_llm_only strategy,
    which always calls this function regardless of heuristic success.
    """
    heuristic_entries = find_toc_candidates(pages)
    if not heuristic_entries:
        return sorted(_toc_scan_indices(pages))
    total = len(pages)
    padded: set[int] = set()
    for entry in heuristic_entries:
        for i in range(entry.source_page_index - 1, entry.source_page_index + 2):
            if 0 <= i < total:
                padded.add(i)
    return sorted(padded)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestLlmScanIndices -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: add _llm_scan_indices to narrow LLM TOC-extraction input"
```

---

### Task 2: `_classify_llm_failure`

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py` (insert right after `_llm_scan_indices`, added in Task 1)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add `_classify_llm_failure` to the same private-helper import tuple used in Task 1 (alongside `_llm_scan_indices`).

Add this test class right after `TestLlmScanIndices`:

```python
class TestClassifyLlmFailure(unittest.TestCase):
    def test_classifies_context_length_message(self):
        exc = Exception("This model's maximum context length is 65536 tokens, however you requested 98213 tokens")
        self.assertEqual(_classify_llm_failure(exc), "context_length_exceeded")

    def test_classifies_no_json_array_found_message(self):
        exc = ValueError("No JSON array found in LLM response: '...'")
        self.assertEqual(_classify_llm_failure(exc), "invalid_or_truncated_json")

    def test_classifies_json_decode_error_message(self):
        exc = ValueError("Expecting ',' delimiter: line 1 column 50 (char 49)")
        self.assertEqual(_classify_llm_failure(exc), "invalid_or_truncated_json")

    def test_classifies_unrecognized_message_as_api_error(self):
        exc = RuntimeError("connection reset by peer")
        self.assertEqual(_classify_llm_failure(exc), "api_error")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestClassifyLlmFailure -v`
Expected: FAIL with `ImportError: cannot import name '_classify_llm_failure'`

- [ ] **Step 3: Implement `_classify_llm_failure`**

```python
def _classify_llm_failure(exc: Exception) -> str:
    """Buckets a final (post-retry) llm_extract_toc_entries failure for
    logging, without importing any provider-specific SDK -- LLMClient is a
    structural Protocol (see llm.py), so segmentation.py can't assume which
    concrete exception type a given implementation raises.
    """
    message = str(exc).lower()
    if "context length" in message or "maximum context" in message:
        return "context_length_exceeded"
    if "json array found" in message or "expecting" in message:
        return "invalid_or_truncated_json"
    return "api_error"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestClassifyLlmFailure -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: add _classify_llm_failure for LLM TOC-extraction failure logging"
```

---

### Task 3: `_extract_with_retry`

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py` (insert right after `_classify_llm_failure`, added in Task 2)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add `_extract_with_retry` to the same private-helper import tuple used in Tasks 1-2.

Add this test class right after `TestClassifyLlmFailure` (before `class TestLlmExtractTocEntries`):

```python
class TestExtractWithRetry(unittest.IsolatedAsyncioTestCase):
    def _fake_llm(self, *responses):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=list(responses))
        return llm

    async def test_returns_parsed_result_on_first_success_without_retry(self):
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": 1}]')
        items = await _extract_with_retry("prompt", llm)
        self.assertEqual(len(items), 1)
        llm.generate.assert_called_once()
        self.assertEqual(llm.generate.call_args.kwargs["max_tokens"], 1024)

    async def test_retries_with_higher_max_tokens_on_truncated_first_response(self):
        llm = self._fake_llm(
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}',  # truncated, no closing ]
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}]',  # valid
        )
        items = await _extract_with_retry("prompt", llm)
        self.assertEqual(len(items), 1)
        self.assertEqual(llm.generate.call_count, 2)
        self.assertEqual(llm.generate.call_args_list[0].kwargs["max_tokens"], 1024)
        self.assertEqual(llm.generate.call_args_list[1].kwargs["max_tokens"], 8192)

    async def test_raises_when_both_attempts_fail_to_parse(self):
        llm = self._fake_llm("not json at all", "still not json")
        with self.assertRaises(Exception):
            await _extract_with_retry("prompt", llm)
        self.assertEqual(llm.generate.call_count, 2)

    async def test_does_not_retry_when_generate_itself_raises(self):
        # A context-length error (or any other API-level failure) can't be
        # fixed by asking for more output tokens -- only truncated-but-
        # otherwise-successful responses should trigger the retry.
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("maximum context length is 65536 tokens"))
        with self.assertRaises(RuntimeError):
            await _extract_with_retry("prompt", llm)
        llm.generate.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestExtractWithRetry -v`
Expected: FAIL with `ImportError: cannot import name '_extract_with_retry'`

- [ ] **Step 3: Implement `_extract_with_retry`**

```python
_LLM_TOC_RETRY_MAX_TOKENS = 8192


async def _extract_with_retry(prompt: str, llm_client: LLMClient) -> list:
    """Calls llm_client.generate for the TOC-extraction prompt, retrying
    once with a much higher max_tokens if the first response doesn't parse
    as a JSON array. A truncated array reliably fails parse_json_array (no
    closing "]") regardless of the underlying cause, so JSON-parseability
    alone is a sufficient, client-agnostic retry trigger -- no LLMClient
    protocol changes needed. Does NOT retry when generate() itself raises
    (e.g. a context-length error): a bigger max_tokens can't fix an input
    that's already too large, so that exception propagates immediately.
    """
    last_error: Exception | None = None
    for max_tokens in (1024, _LLM_TOC_RETRY_MAX_TOKENS):
        raw = await llm_client.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=0.0, is_valid=_parses_as_json_array,
        )
        try:
            return parse_json_array(raw)
        except Exception as exc:
            last_error = exc
    raise last_error
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestExtractWithRetry -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: add _extract_with_retry for LLM TOC-extraction truncation recovery"
```

---

### Task 4: Wire the three helpers into `llm_extract_toc_entries`

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:514-525` (the body of `llm_extract_toc_entries`, between Tasks 1-3's new helpers and the rest of the unchanged function)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing test**

Add this test method to the existing `TestLlmExtractTocEntries` class (after `test_passes_is_valid_check_for_json_array_shape`, the last method in that class):

```python
    async def test_recovers_via_retry_when_first_response_is_truncated(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=[
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}',  # truncated
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}]',  # valid on retry
        ])
        entries = await llm_extract_toc_entries(["front matter"] * 20, llm)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].title, "Introduction")
        self.assertEqual(llm.generate.call_count, 2)
        self.assertEqual(llm.generate.call_args_list[0].kwargs["max_tokens"], 1024)
        self.assertEqual(llm.generate.call_args_list[1].kwargs["max_tokens"], 8192)

    async def test_logs_classified_reason_when_both_attempts_fail(self):
        llm = self._fake_llm("not json at all")
        with self.assertLogs("chapter_segmentation.segmentation", level="WARNING") as cm:
            entries = await llm_extract_toc_entries(["front matter"] * 20, llm)
        self.assertEqual(entries, [])
        self.assertTrue(any("invalid_or_truncated_json" in message for message in cm.output))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k "test_recovers_via_retry_when_first_response_is_truncated or test_logs_classified_reason_when_both_attempts_fail" -v`
Expected: FAIL -- `test_recovers_via_retry_when_first_response_is_truncated` fails because `llm.generate` is only called once today (no retry); `test_logs_classified_reason_when_both_attempts_fail` fails because today's log message is the generic `"LLM call or JSON parse failed"`, not the classified reason.

- [ ] **Step 3: Update `llm_extract_toc_entries`'s body**

In `src/chapter_segmentation/segmentation.py`, replace this block (currently lines 514-525):

```python
    scan_indices = sorted(_toc_scan_indices(pages))
    if not scan_indices:
        return []
    page_blocks = "\n\n".join(f"[PAGE {i}]\n{pages[i]}" for i in scan_indices)
    prompt = _LLM_TOC_EXTRACTION_PROMPT.format(page_blocks=page_blocks)

    try:
        raw = await llm_client.generate(prompt=prompt, max_tokens=1024, temperature=0.0, is_valid=_parses_as_json_array)
        items = parse_json_array(raw)
    except Exception:
        logger.warning("llm_extract_toc_entries: LLM call or JSON parse failed", exc_info=True)
        return []
```

with:

```python
    scan_indices = _llm_scan_indices(pages)
    if not scan_indices:
        return []
    page_blocks = "\n\n".join(f"[PAGE {i}]\n{pages[i]}" for i in scan_indices)
    prompt = _LLM_TOC_EXTRACTION_PROMPT.format(page_blocks=page_blocks)

    try:
        items = await _extract_with_retry(prompt, llm_client)
    except Exception as exc:
        logger.warning(
            "llm_extract_toc_entries: giving up (%s)", _classify_llm_failure(exc), exc_info=True,
        )
        return []
```

- [ ] **Step 4: Run the new tests, then the full file's test suite**

Run: `uv run python -m pytest tests/test_segmentation.py -k "test_recovers_via_retry_when_first_response_is_truncated or test_logs_classified_reason_when_both_attempts_fail" -v`
Expected: PASS (2 tests)

Run: `uv run python -m pytest tests/test_segmentation.py -v`
Expected: PASS, all tests (existing + new) -- in particular confirm these existing tests are unaffected: `TestLlmExtractTocEntries::test_returns_empty_list_on_malformed_response`, `TestAnalyzeAttachmentWithLlmFallback::test_llm_toc_extraction_fires_when_heuristic_finds_nothing`, `TestAnalyzeAttachmentWithLlmFallback::test_does_not_call_llm_when_heuristic_already_succeeds`, `TestAnalyzeAttachmentLlmOnly::test_calls_llm_even_when_heuristic_would_succeed`, `TestAnalyzeAttachmentLlmOnly::test_swallows_llm_exception_and_returns_empty_result`.

Then run the full project test suite to confirm no regressions anywhere else:

Run: `uv run python -m pytest -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "fix: retry truncated LLM TOC extraction and narrow its input range"
```

---

### Task 5: Validate against the real KISSKI API and update RESULTS.md

**Files:**
- Modify: `evaluation/RESULTS.md` (the "Per-strategy standalone results" section added in the prior per-strategy-evaluation work)
- No source changes in this task -- this is a real-API validation + documentation update.

- [ ] **Step 1: Re-run the LLM cache refresh against the real API**

```bash
set -a; source .env; set +a
uv run python evaluation/refresh_llm_cache.py --mode top5
```

This overwrites `evaluation/llm-cache/*.json` for the 5 currently-least-busy KISSKI models, now using the fixed `llm_extract_toc_entries`.

- [ ] **Step 2: Regenerate the report locally and inspect it**

```bash
uv run python evaluation/generate_report.py --out /tmp/llm-fix-check
```

Open `/tmp/llm-fix-check/index.html` and `/tmp/llm-fix-check/llm/index.html`. Confirm: the three previously-0/0 large-TOC books (`9783322969828`, `9783847432364`, `9783848736829`) now find a non-zero number of chapters on at least the models that aren't `apertus-70b-instruct-2509`, and the per-strategy aggregate table's LLM row F1 has moved up from 0.29.

- [ ] **Step 3: Update `evaluation/RESULTS.md`**

Update the "Per-strategy standalone results (heuristic / outline / LLM)" table and its surrounding bullet points with the new numbers from Step 2 (best model, its new aggregate P/R/F1/time, and updated commentary reflecting that the truncation/oversized-input root causes are now fixed -- keep noting `apertus-70b-instruct-2509`'s remaining 65536-token ceiling for any book whose even-narrowed prompt still exceeds it, if that's still the case).

- [ ] **Step 4: Commit**

```bash
git add evaluation/llm-cache/ evaluation/RESULTS.md
git commit -m "data: refresh LLM cache and RESULTS.md after truncation/input-size fix"
```

Do not push without separately confirming with the user first (per this project's established pattern of confirming before pushing to `main`).
