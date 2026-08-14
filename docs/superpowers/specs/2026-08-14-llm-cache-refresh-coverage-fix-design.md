# LLM cache refresh: fix stalled nightly coverage growth

Status: approved for planning
Date: 2026-08-14

## Problem

The published report shows only a few LLM models with meaningful book
coverage despite `.github/workflows/refresh-llm-cache.yml` running nightly
in `--mode fill-gaps`. Investigation of `evaluation/refresh_llm_cache.py`
found two compounding causes:

1. **The actual bug:** the main loop in `_main` reprocesses *every* book in
   scope for a selected model, regardless of whether that book already has
   a cached entry for it. `_upsert_cache` unconditionally overwrites. A
   model is selected by `fill-gaps` (`select_gap_fill`) whenever it's
   missing from *at least one* book's cache
   (`_fully_covered_model_ids` uses set intersection across all books) --
   but once selected, the loop redoes the full ~89-book pass from book 1,
   even for books that already succeeded on a prior run. If the run is
   interrupted (job timeout, a transient failure) partway through, the
   next night's run picks the same still-not-fully-covered model and
   discards all of that progress by starting over. `top5` and `full` modes
   are *documented* to always rerun unconditionally (`top5`: "unconditionally,
   even if already cached"; `full`: "re-runs EVERY model that already has
   at least one cached entry") -- `fill-gaps` was inheriting that same
   unconditional behavior even though its entire purpose is incremental
   gap-filling.
2. **Compounding factor** (already known): fully sequential per-book
   processing (~31s/book x ~89 books ~= 46 minutes per model) leaves
   little slack inside the job's 60-minute timeout, so there's barely
   room for one model's full pass, let alone recovery from #1's wasted
   reruns.

## Goal

Make `--mode fill-gaps` runs genuinely incremental (coverage only grows,
never resets) and cut wall-clock time per model so more of the 60-minute
budget turns into completed coverage.

## Non-goals

- Changing `generate_report.py`'s display logic (which model(s) it shows)
  -- already addressed in a separate change (per-model "as of" dates).
- Changing `top5`/`full` modes' unconditional-rerun behavior -- that's
  their documented, intentional purpose (a deliberate full/sanity-check
  regeneration, not incremental gap-filling).
- Discovering/documenting KISSKI's actual rate limits -- not published
  anywhere reachable; the concurrency default is a conservative guess,
  overridable via a flag.

## Design

### 1. Skip already-cached (book, model) pairs in `fill-gaps` mode

In `_main`, when iterating `selected` models and `book_entries`, skip a
book for a given model if that book's cache file already has an entry
for `model.id` -- but **only when `mode == "fill-gaps"`**. `top5` and
`full` keep today's unconditional overwrite behavior.

```python
def _has_cached_entry(cache_dir: Path, manifest_key: str, model_id: str) -> bool:
    cache_path = cache_dir / f"{manifest_key}.json"
    if not cache_path.exists():
        return False
    return model_id in json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})
```

Called from the book loop; when `mode == "fill-gaps"` and
`_has_cached_entry(...)` is true, print a `SKIP (already cached)` line and
continue without spending an API call.

### 2. Bounded concurrency per model

Replace the sequential `for corpus, manifest_key, cache_dir in book_entries`
loop (per selected model) with concurrent tasks bounded by an
`asyncio.Semaphore`, default size 4, overridable via a new `--concurrency`
CLI flag. Each task performs the same try/except-log-continue work as
today (plus the new skip check and retry, below); results are gathered
with `asyncio.gather`. Order of completion doesn't matter -- each task
writes its own book's cache file independently and printing happens as
each task finishes.

### 3. Retry with backoff on transient failures

Wrap the `analyze_attachment_llm_only` call in a small retry helper: up to
3 attempts, exponential backoff (e.g. 1s, 2s, 4s) between attempts, on any
exception. Only after all 3 attempts fail does the book/model pair get
logged as `FAILED` and skipped (matching today's final behavior, just with
retries first). This matters more once requests run concurrently (higher
chance of hitting a transient 429/503) and it also means a single flaky
call no longer permanently blocks a model from ever reaching full coverage
on a clean pass.

## Testing

- `tests/test_refresh_llm_cache.py`: add cases for
  - `_has_cached_entry` returns true/false correctly.
  - `_main` with `mode="fill-gaps"` skips a book already cached for the
    selected model (no client call made for it) while still processing an
    uncached book for the same model.
  - `_main` with `mode="top5"` (and `mode="full"`) still unconditionally
    reprocesses an already-cached book (no behavior change for those
    modes).
  - Concurrency: multiple books for one model complete via
    `asyncio.gather`, respecting the semaphore bound (can be tested by
    asserting the max number of concurrent in-flight fake calls never
    exceeds the configured limit, using a counter + `asyncio.Event`/small
    delay in a fake client).
  - Retry: a fake client that fails twice then succeeds on the 3rd attempt
    results in a cached entry, not a `FAILED` log; a fake client that
    fails all 3 attempts results in `FAILED` (matching today's behavior).
