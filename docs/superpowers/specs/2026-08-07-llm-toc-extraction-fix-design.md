# Fix `llm_extract_toc_entries` truncation and oversized input

Status: approved for planning
Date: 2026-08-07

## Problem

Running the real per-strategy evaluation (see
`docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md`) against
the live KISSKI API surfaced that the standalone LLM strategy (F1 0.29, best
model) substantially underperforms the pure heuristic (F1 0.58), and one
model (`apertus-70b-instruct-2509`) scores 0.00 on every single book. Root
causes, confirmed from the actual cache data in `evaluation/llm-cache/*.json`
and `evaluation/RESULTS.md`:

1. **Output truncation.** `llm_extract_toc_entries`
   (`src/chapter_segmentation/segmentation.py:503`) calls
   `llm_client.generate(..., max_tokens=1024, ...)` unconditionally. Every
   book with 20+ expected chapters (`9783322969828`: 24, `9783847432364`: 21,
   `9783848736829`: 23) scores exactly 0/0 across every one of the 5 models
   tested -- the JSON chapter-listing array for that many entries exceeds
   1024 output tokens, truncates mid-object, fails `parse_json_array` (no
   closing `]`), and is swallowed by the function's broad `except Exception`
   as "found nothing." Books with 10-17 expected chapters get non-zero
   partial credit across multiple models, consistent with this being purely
   an output-budget problem, not a model-capability problem.

2. **Oversized input.** The same function always sends the full blind
   front-15%/back-5% page fraction (`_toc_scan_indices`) to the LLM, even
   when a much narrower real TOC region is already known. Measured prompt
   sizes for this corpus range 67905-133908 input tokens depending on book
   size. `apertus-70b-instruct-2509` has a 65536-token context window --
   smaller than every single book's prompt -- so it fails on 100% of the
   corpus in under a second every time, always via the same swallowed
   exception. The wide input also feeds the model a large amount of
   redacted/irrelevant body prose, which is why extracted `authors` fields
   pick up garbled placeholder words -- title/page-number extraction (real
   TOC text) stays cleaner.

3. **One generic failure log.** Both failure modes above (and any other API
   error) currently produce the same message: `"llm_extract_toc_entries: LLM
   call or JSON parse failed"`. There is no way to tell from logs alone
   whether a given failure was truncation, a context-length error, or
   something else -- this is why root-causing #1 and #2 required manually
   re-running the eval and inspecting raw cache data by hand.

## Fix

All changes are inside `llm_extract_toc_entries`
(`src/chapter_segmentation/segmentation.py:503`). No signature or call-site
changes -- both `analyze_attachment_with_llm_fallback` (production) and
`analyze_attachment_llm_only` (standalone eval strategy) benefit
automatically.

### 1. Narrow the scan range when a real TOC is already detectable

Before falling back to `_toc_scan_indices(pages)` (the blind fraction), call
the existing regex heuristic `find_toc_candidates(pages)`. If it returns any
entries, build the scan range from their `source_page_index` values, padded
&plusmn;1 page each side (to still catch a multi-page TOC listing the regex
only partially matched), clamped to `[0, len(pages))`. Use
`_toc_scan_indices(pages)` only when `find_toc_candidates` returns nothing.

```python
def _llm_scan_indices(pages: list[str]) -> list[int]:
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

This does not change `analyze_attachment_with_llm_fallback`'s behavior in
its primary call path: it only invokes `llm_extract_toc_entries` when
`find_toc_candidates` already returned nothing (`len(toc_entries) == 0`), in
which case `_llm_scan_indices` falls back to the same blind fraction as
today. The secondary path (entries exist but `heuristic_chapters == 0`) and
`analyze_attachment_llm_only` (which always calls this function) both get
the narrower, cleaner input.

### 2. Retry once on truncated/invalid output

First attempt keeps `max_tokens=1024` (already sufficient for the common
10-17-chapter case -- no need to pay a larger budget by default). If
`parse_json_array(raw)` raises, retry the whole call once with
`max_tokens=8192` before giving up:

```python
async def _extract_with_retry(prompt: str, llm_client: LLMClient) -> list:
    for max_tokens in (1024, 8192):
        raw = await llm_client.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=0.0,
            is_valid=_parses_as_json_array,
        )
        try:
            return parse_json_array(raw)
        except Exception:
            last_error = raw
            continue
    raise ValueError(f"LLM response did not parse as a JSON array after retry: {last_error!r}")
```

No `LLMClient` protocol changes needed -- a truncated array reliably fails
`parse_json_array` (no closing `]`) regardless of which underlying failure
caused the truncation, so JSON-parseability alone is a sufficient retry
trigger.

### 3. Classify the failure reason on final give-up

Replace the single generic warning with a classifier that inspects the
final exception without importing any provider-specific SDK (segmentation.py
must stay client-agnostic per the `LLMClient` protocol):

```python
def _classify_llm_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "context length" in message or "maximum context" in message:
        return "context_length_exceeded"
    if "json array found" in message or "expecting" in message:
        return "invalid_or_truncated_json"
    return "api_error"
```

```python
    try:
        items = await _extract_with_retry(prompt, llm_client)
    except Exception as exc:
        logger.warning(
            "llm_extract_toc_entries: giving up (%s)",
            _classify_llm_failure(exc), exc_info=True,
        )
        return []
```

## Testing

- Unit tests for `_llm_scan_indices`: heuristic entries present (narrowed +
  padded range, clamped at document boundaries) vs. heuristic entries absent
  (falls back to `_toc_scan_indices`).
- Unit tests for `_extract_with_retry` using a fake `LLMClient`: first call
  returns truncated/invalid JSON, second call (higher `max_tokens`) returns
  valid JSON -- verify both `max_tokens` values used in order and the parsed
  result comes from the second call. Also cover both-attempts-fail (raises).
- Unit tests for `_classify_llm_failure`: context-length message, JSON-parse
  message, and an unrelated message each map to the expected category.
- Existing `tests/test_segmentation_accuracy.py` /
  `tests/test_public_evaluation_cache_parity.py` are unaffected (heuristic
  pipeline only).

## Validation

After merging, re-run `KISSKI_API_KEY=... uv run python
evaluation/refresh_llm_cache.py --mode top5` against the real API to
regenerate `evaluation/llm-cache/*.json` with the fixed extraction logic,
regenerate the report, and update `evaluation/RESULTS.md`'s per-strategy
section with the before/after numbers. Expect: the three 20+-chapter books
to stop scoring 0/0, and `apertus-70b-instruct-2509`'s F1 to move off 0.00 on
the (now much smaller) narrowed-input books it can fit in its 65536-token
context.

## Out of scope

- Changing the `LLMClient` protocol (`src/chapter_segmentation/llm.py`) --
  the retry and classification logic both work entirely from the returned
  string, no protocol changes needed.
- Chunked/windowed multi-call TOC extraction -- rejected in favor of the
  simpler retry-with-bigger-budget approach; can be revisited later if a
  book's real TOC still overflows an 8192-token response.
- `llm_disambiguate_chapter_start` and the rest of the disambiguation path --
  this fix is scoped to TOC extraction only, the specific function
  responsible for every observed failure.
