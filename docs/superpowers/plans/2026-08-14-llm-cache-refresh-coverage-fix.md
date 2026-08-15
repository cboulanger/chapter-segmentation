# LLM Cache Refresh Coverage Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `evaluation/refresh_llm_cache.py --mode fill-gaps` runs genuinely incremental (a book/model pair already cached is never silently redone) and cut per-model wall-clock time via bounded concurrency + retry, so the nightly job's coverage actually grows night over night instead of resetting on any interruption.

**Architecture:** Extract four small, independently testable units out of `_main`'s inline loop: a pure cache-lookup predicate (`_has_cached_entry`), a generic async retry-with-backoff wrapper (`_call_with_retry`), a per-book worker (`_run_book_for_model`) that composes skip-check + retry + upsert + logging, and a concurrency-bounded fan-out runner (`_process_model`) that's worker-agnostic (takes the worker as a parameter) so it's testable without any real network/LLM dependency. `_main` is left as thin composition of these pieces plus the new `--concurrency` CLI flag.

**Tech Stack:** Python 3.12, `asyncio` (stdlib), `unittest.IsolatedAsyncioTestCase` for async tests, no new dependencies.

---

### Task 1: `_has_cached_entry` -- pure cache-lookup predicate

**Files:**
- Modify: `evaluation/refresh_llm_cache.py` (add function after `_fully_covered_model_ids`/`_all_cached_model_ids`, before `_upsert_cache`, i.e. after line 125)
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_refresh_llm_cache.py`, after the `TestAllCachedModelIds` class (after line 119), before `TestUpsertCache`:

```python
class TestHasCachedEntry(unittest.TestCase):
    def test_no_cache_file_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_has_cached_entry(Path(tmp), "book-a", "model-x"))

    def test_cache_file_without_model_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            self.assertFalse(_has_cached_entry(cache_dir, "book-a", "model-x"))

    def test_cache_file_with_model_returns_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertTrue(_has_cached_entry(cache_dir, "book-a", "model-x"))
```

Update the import line (line 10) to add `_has_cached_entry`:

```python
from evaluation.refresh_llm_cache import (
    _all_cached_model_ids,
    _fully_covered_model_ids,
    _has_cached_entry,
    _upsert_cache,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k HasCachedEntry`
Expected: FAIL with `ImportError: cannot import name '_has_cached_entry'`

- [ ] **Step 3: Write the implementation**

In `evaluation/refresh_llm_cache.py`, add after `_all_cached_model_ids` (after line 125, before `def _upsert_cache`):

```python
def _has_cached_entry(cache_dir: Path, manifest_key: str, model_id: str) -> bool:
    """True if this book's cache file already has an entry for model_id --
    used by fill-gaps mode to skip work already done, so an interrupted run
    doesn't get redone from scratch the next time this model is selected."""
    cache_path = cache_dir / f"{manifest_key}.json"
    if not cache_path.exists():
        return False
    return model_id in json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k HasCachedEntry`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "feat: add _has_cached_entry predicate for skip-if-cached logic"
```

---

### Task 2: `_call_with_retry` -- generic async retry with backoff

**Files:**
- Modify: `evaluation/refresh_llm_cache.py`
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_refresh_llm_cache.py`, after the new `TestHasCachedEntry` class:

```python
class TestCallWithRetry(unittest.IsolatedAsyncioTestCase):
    async def test_succeeds_on_first_try_without_sleeping(self):
        sleeps = []

        async def sleep(delay):
            sleeps.append(delay)

        async def fn():
            return "ok"

        result = await _call_with_retry(fn, sleep=sleep)
        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [])

    async def test_retries_then_succeeds(self):
        attempts = []
        sleeps = []

        async def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient")
            return "ok"

        async def sleep(delay):
            sleeps.append(delay)

        result = await _call_with_retry(fn, attempts=3, base_delay=1.0, sleep=sleep)
        self.assertEqual(result, "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    async def test_raises_after_exhausting_all_attempts(self):
        async def fn():
            raise RuntimeError("permanent")

        async def sleep(delay):
            pass

        with self.assertRaises(RuntimeError):
            await _call_with_retry(fn, attempts=3, base_delay=0.01, sleep=sleep)
```

Add `import asyncio` near the top of `tests/test_refresh_llm_cache.py` (it currently has no `asyncio` import -- add it as the first import, before `json`).

Update the import from `evaluation.refresh_llm_cache` to add `_call_with_retry`:

```python
from evaluation.refresh_llm_cache import (
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _upsert_cache,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k CallWithRetry`
Expected: FAIL with `ImportError: cannot import name '_call_with_retry'`

- [ ] **Step 3: Write the implementation**

In `evaluation/refresh_llm_cache.py`, add after `_has_cached_entry` (which Task 1 placed before `_upsert_cache`):

```python
async def _call_with_retry(
    fn: Callable[[], Awaitable],
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
):
    """Awaits fn() up to `attempts` times with exponential backoff
    (base_delay, base_delay*2, base_delay*4, ...) between failures,
    re-raising the last exception once every attempt is exhausted. `sleep`
    is injectable so tests don't pay real wall-clock delay."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                await sleep(base_delay * (2**attempt))
    raise last_exc
```

Also update the `typing` import (current line 67, `from typing import Callable, Optional`) to add `Awaitable`:

```python
from typing import Awaitable, Callable, Optional
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k CallWithRetry`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "feat: add _call_with_retry generic async retry-with-backoff helper"
```

---

### Task 3: `_process_model` -- concurrency-bounded fan-out over book entries

**Files:**
- Modify: `evaluation/refresh_llm_cache.py`
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_refresh_llm_cache.py`, after `TestCallWithRetry`:

```python
class TestProcessModel(unittest.IsolatedAsyncioTestCase):
    async def test_calls_worker_once_per_book_entry(self):
        calls = []

        async def worker(corpus, manifest_key, cache_dir):
            calls.append((corpus, manifest_key, cache_dir))

        book_entries = [("corpus-a", "book-1", Path("/x")), ("corpus-a", "book-2", Path("/x"))]
        await _process_model(book_entries, concurrency=4, worker=worker)
        self.assertEqual(len(calls), 2)

    async def test_never_exceeds_concurrency_limit(self):
        in_flight = 0
        max_in_flight = 0

        async def worker(corpus, manifest_key, cache_dir):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

        book_entries = [("c", f"book-{i}", Path("/x")) for i in range(10)]
        await _process_model(book_entries, concurrency=3, worker=worker)
        self.assertLessEqual(max_in_flight, 3)
        self.assertGreater(max_in_flight, 1)
```

Update the import from `evaluation.refresh_llm_cache` to add `_process_model`:

```python
from evaluation.refresh_llm_cache import (
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _process_model,
    _upsert_cache,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k ProcessModel`
Expected: FAIL with `ImportError: cannot import name '_process_model'`

- [ ] **Step 3: Write the implementation**

In `evaluation/refresh_llm_cache.py`, add after `_call_with_retry`:

```python
async def _process_model(
    book_entries: list[tuple[str, str, Path]],
    concurrency: int,
    worker: Callable[[str, str, Path], Awaitable],
) -> None:
    """Runs worker(corpus, manifest_key, cache_dir) for every book_entries
    tuple concurrently, bounded by `concurrency` in-flight at once. worker
    is expected to handle its own errors (it must not raise) -- one book's
    failure must not cancel the others via asyncio.gather."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(corpus: str, manifest_key: str, cache_dir: Path) -> None:
        async with semaphore:
            await worker(corpus, manifest_key, cache_dir)

    await asyncio.gather(*(_bounded(corpus, manifest_key, cache_dir) for corpus, manifest_key, cache_dir in book_entries))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k ProcessModel`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "feat: add _process_model concurrency-bounded fan-out helper"
```

---

### Task 4: `_run_book_for_model` + wire into `_main` + `--concurrency` flag

**Files:**
- Modify: `evaluation/refresh_llm_cache.py`
- Test: `tests/test_refresh_llm_cache.py`

This task wires together Tasks 1-3. `_run_book_for_model` composes them, so it's tested directly by mocking its two real dependencies (`public_pages_for`, `analyze_attachment_llm_only`) at the module level -- this stays fully isolated (no network, no filesystem beyond the tmpdir cache) despite those two functions being real production code elsewhere. Only `_main` itself (the httpx call to `fetch_kisski_models`, real CLI argument wiring) stays outside unit tests, per the existing test-file docstring ("The network-calling `_main()` orchestration is exercised manually... not here") -- covered instead by Step 9's manual `--help` check.

- [ ] **Step 1: Write the failing tests for `_run_book_for_model`**

Add near the top of `tests/test_refresh_llm_cache.py`, alongside the other imports:

```python
import unittest.mock
from types import SimpleNamespace
```

Update the import from `evaluation.refresh_llm_cache` to add `_run_book_for_model`:

```python
from evaluation.refresh_llm_cache import (
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _process_model,
    _run_book_for_model,
    _upsert_cache,
)
```

Add a new class after `TestProcessModel`:

```python
class TestRunBookForModel(unittest.IsolatedAsyncioTestCase):
    def _model(self):
        return SimpleNamespace(id="model-x", demand=0)

    async def test_fill_gaps_skips_when_already_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for") as pages_mock,
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only") as analyze_mock,
            ):
                await _run_book_for_model("corpus-a", "book-a", cache_dir, self._model(), "fill-gaps", llm_client=None)
                pages_mock.assert_not_called()
                analyze_mock.assert_not_called()

    async def test_fill_gaps_processes_when_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            async def fake_analyze(pages, llm_client):
                return {"chapters": [{"title": "Intro"}]}

            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for", return_value=["page text"]),
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only", side_effect=fake_analyze),
            ):
                await _run_book_for_model("corpus-a", "book-a", cache_dir, self._model(), "fill-gaps", llm_client=None)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertIn("model-x", data["models"])

    async def test_top5_reprocesses_even_when_already_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [{"title": "Old"}], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )

            async def fake_analyze(pages, llm_client):
                return {"chapters": [{"title": "New"}]}

            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for", return_value=["page text"]),
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only", side_effect=fake_analyze),
            ):
                await _run_book_for_model("corpus-a", "book-a", cache_dir, self._model(), "top5", llm_client=None)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertEqual(data["models"]["model-x"]["chapters"], [{"title": "New"}])

    async def test_failure_after_retries_is_logged_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            async def always_fails(pages, llm_client):
                raise RuntimeError("boom")

            async def fast_sleep(delay):
                pass

            with (
                unittest.mock.patch("evaluation.refresh_llm_cache.public_pages_for", return_value=["page text"]),
                unittest.mock.patch("evaluation.refresh_llm_cache.analyze_attachment_llm_only", side_effect=always_fails),
            ):
                await _run_book_for_model(
                    "corpus-a", "book-a", cache_dir, self._model(), "fill-gaps", llm_client=None, sleep=fast_sleep,
                )
            self.assertFalse((cache_dir / "book-a.json").exists())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k RunBookForModel`
Expected: FAIL with `ImportError: cannot import name '_run_book_for_model'`

- [ ] **Step 3: Add `functools` import**

In `evaluation/refresh_llm_cache.py`, add to the import block (after line 61, `import asyncio`):

```python
import functools
```

- [ ] **Step 4: Add `_run_book_for_model`**

Add after `_process_model` (and before `async def _main`):

```python
async def _run_book_for_model(
    corpus: str,
    manifest_key: str,
    cache_dir: Path,
    model,
    mode: str,
    llm_client,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
) -> None:
    if mode == "fill-gaps" and _has_cached_entry(cache_dir, manifest_key, model.id):
        print(f"{corpus}/{manifest_key} / {model.id}: SKIP (already cached)")
        return
    try:
        pages = public_pages_for(corpus, manifest_key)
        start = time.perf_counter()
        result = await _call_with_retry(lambda: analyze_attachment_llm_only(pages, llm_client), sleep=sleep)
        elapsed = time.perf_counter() - start
        _upsert_cache(cache_dir, manifest_key, model.id, result["chapters"], elapsed, model.demand)
        print(f"{corpus}/{manifest_key} / {model.id}: {len(result['chapters'])} chapters, {elapsed:.1f}s")
    except Exception as exc:
        # One book/model failure (after retries) must not strand the whole
        # batch or discard cache entries already written for other books/
        # models in this same run -- same catch-log-continue convention as
        # generate_public_evaluation_cache.py.
        print(f"{corpus}/{manifest_key} / {model.id}: FAILED ({exc}) -- skipping")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v -k RunBookForModel`
Expected: PASS (4 tests)

- [ ] **Step 6: Replace `_main`'s book-processing loop**

In `evaluation/refresh_llm_cache.py`, replace this block (current lines 181-197):

```python
    print(f"Selected models: {[m.id for m in selected]}")
    for model in selected:
        llm_client = _OpenAICompatibleLLMClient(model=model.id, base_url=base_url, api_key=api_key)
        for corpus, manifest_key, cache_dir in book_entries:
            try:
                pages = public_pages_for(corpus, manifest_key)
                start = time.perf_counter()
                result = await analyze_attachment_llm_only(pages, llm_client)
                elapsed = time.perf_counter() - start
                _upsert_cache(cache_dir, manifest_key, model.id, result["chapters"], elapsed, model.demand)
                print(f"{corpus}/{manifest_key} / {model.id}: {len(result['chapters'])} chapters, {elapsed:.1f}s")
            except Exception as exc:
                # One book/model failure must not strand the whole batch or
                # discard cache entries already written for other books/
                # models in this same run -- same catch-log-continue
                # convention as generate_public_evaluation_cache.py.
                print(f"{corpus}/{manifest_key} / {model.id}: FAILED ({exc}) -- skipping")
    return 0
```

with:

```python
    print(f"Selected models: {[m.id for m in selected]}")
    for model in selected:
        llm_client = _OpenAICompatibleLLMClient(model=model.id, base_url=base_url, api_key=api_key)
        worker = functools.partial(_run_book_for_model, model=model, mode=mode, llm_client=llm_client)
        await _process_model(book_entries, concurrency, worker)
    return 0
```

- [ ] **Step 7: Thread `concurrency` through `_main`'s signature**

Change `_main`'s signature (current line 137):

```python
async def _main(mode: str, base_url: str, limit: int, corpus: Optional[str], clear: bool) -> int:
```

to:

```python
async def _main(mode: str, base_url: str, limit: int, corpus: Optional[str], clear: bool, concurrency: int) -> int:
```

- [ ] **Step 8: Add the `--concurrency` CLI flag and thread it through the `__main__` block**

In the `if __name__ == "__main__":` block, add after the `--clear` argument (current lines 210-213):

```python
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent book requests per model. KISSKI publishes no documented rate limit, "
             "so this is a conservative default -- raise it if you don't observe 429s. Default 4.",
    )
```

Change the `raise SystemExit(...)` call (current lines 215-217):

```python
    raise SystemExit(asyncio.run(_main(
        mode=args.mode, base_url=args.base_url, limit=args.limit, corpus=args.corpus, clear=args.clear,
    )))
```

to:

```python
    raise SystemExit(asyncio.run(_main(
        mode=args.mode, base_url=args.base_url, limit=args.limit, corpus=args.corpus, clear=args.clear,
        concurrency=args.concurrency,
    )))
```

- [ ] **Step 9: Update the module docstring**

In the module docstring (lines 1-57), the `--mode fill-gaps` paragraph (lines 21-24) currently reads:

```
--mode fill-gaps: finds non-"very busy" models not yet cached for EVERY
book across every corpus's current public books, and runs up to 5
(--limit) of those -- how the cache grows to cover every model over time
(see the nightly schedule in the workflow above).
```

Replace with:

```
--mode fill-gaps: finds non-"very busy" models not yet cached for EVERY
book across every corpus's current public books, and runs up to 5
(--limit) of those -- how the cache grows to cover every model over time
(see the nightly schedule in the workflow above). Within a fill-gaps run,
a (book, model) pair that's already cached is skipped rather than redone
-- so an interrupted run (job timeout, a transient failure) picks up
where it left off next time, instead of restarting the selected model's
whole book set from scratch. (--mode top5/full always rerun
unconditionally, regardless of what's already cached -- see below.)
```

Add a new paragraph after the `--clear` paragraph (after current line 48, before the `--mode full` paragraph):

```
--concurrency (default 4): how many books to process concurrently for
the currently-selected model. KISSKI publishes no documented rate limit,
so this default is a conservative guess -- raise it if you don't observe
429/503 errors. Each request also retries up to 3 times with exponential
backoff (1s, 2s, 4s) on any failure before being logged as FAILED, so
transient errors under concurrency don't permanently block a book/model
pair.
```

- [ ] **Step 10: Run the full test suite**

Run: `uv run pytest tests/ -q`
Expected: all tests pass (no regressions; new tests from Tasks 1-3 and this task's Step 1 included)

- [ ] **Step 11: Manual sanity check of the CLI (no network call)**

Run: `uv run python evaluation/refresh_llm_cache.py --help`
Expected: help text shows `--concurrency` with its default and description, and the updated `--mode`/module docstring text at the top.

- [ ] **Step 12: Commit**

```bash
git add evaluation/refresh_llm_cache.py
git commit -m "feat: skip already-cached (book, model) pairs in fill-gaps mode; add concurrency"
```

---

### Task 5: Document the behavior change in `evaluation/README.md`

**Files:**
- Modify: `evaluation/README.md` (the "LLM strategy evaluation" section, current lines 291-321)

- [ ] **Step 1: Update the section**

Replace this paragraph (current lines 312-321):

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

with:

```
`--mode top5` (the default) refreshes the 5 currently-least-busy KISSKI
models. `--mode fill-gaps` instead finds non-busy models not yet cached
for every book in the corpus and runs up to 5 of those -- this is what
`.github/workflows/refresh-llm-cache.yml`'s nightly schedule uses, so the
cache grows to cover every available/busy model over time without paying
to re-run models it already has complete data for. Within a fill-gaps
run, a (book, model) pair that's already cached is skipped, so an
interrupted run (job timeout, a transient failure) resumes from where it
left off next time rather than redoing the selected model's whole book
set from scratch. Books are processed concurrently per model (`--concurrency`,
default 4 -- KISSKI publishes no documented rate limit, so this is a
conservative default), with up to 3 retries (exponential backoff) per
request before a book/model pair is logged as failed and skipped. The
same workflow also exposes a manual `workflow_dispatch` trigger (using
`--mode top5`) for an on-demand refresh, e.g. right after a prompt
change, to sanity-check the current best models. Either trigger commits
the updated cache files straight to `main`, which republishes the report
automatically.
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/README.md
git commit -m "docs: document fill-gaps skip-if-cached, concurrency, and retry behavior"
```

---

### Task 6: Final regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite one more time**

Run: `uv run pytest tests/ -q`
Expected: all tests pass, 0 failures.

- [ ] **Step 2: Confirm no leftover references to the old inline loop**

Run: `grep -n "for corpus, manifest_key, cache_dir in book_entries" evaluation/refresh_llm_cache.py`
Expected: no matches inside `_main` (the loop now lives only in `_process_model`, which iterates `book_entries` under a different variable name via the generator expression in `asyncio.gather(...)` -- this grep should return nothing, confirming the old inline sequential loop was fully replaced).
