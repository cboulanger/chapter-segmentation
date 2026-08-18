#!/usr/bin/env python3
"""Refreshes each corpus's llm-cache/ -- the only script in this repo that
spends real KISSKI API budget. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"LLM cache refresh" and
docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md.

Reads KISSKI_API_KEY from the environment. Locally, source it from
zotero-rag's .env, e.g.:

    export KISSKI_API_KEY=$(grep '^KISSKI_API_KEY=' ../zotero-rag/.env | cut -d= -f2-)
    uv run python evaluation/refresh_llm_cache.py --mode top5

In CI it comes from a repository secret (see
.github/workflows/refresh-llm-cache.yml). Not a pytest test.

--mode top5 (default): refreshes the current 5 (--limit) least-busy
models, unconditionally, even if already cached -- a quick manual sanity
check.

--mode fill-gaps: finds non-"very busy" models not yet cached for EVERY
book across every corpus's current public books, and runs up to 5
(--limit) of those -- how the cache grows to cover every model over time
(see the nightly schedule in the workflow above). Within a fill-gaps run,
a (book, model) pair that's already cached is skipped rather than redone
-- so an interrupted run (job timeout, a transient failure) picks up
where it left off next time, instead of restarting the selected model's
whole book set from scratch. (--mode top5/full always rerun
unconditionally, regardless of what's already cached -- see below.)

--limit (default 5, applies to top5/fill-gaps only -- full is
deliberately uncapped): how many models to run this invocation. Lower it
(e.g. --limit 1) for a quick single-model pass right after a change that
invalidated cached entries (a redaction/extraction change, a manifest
promotion) -- rather than paying for a full 5-model refresh immediately,
run one model now to get *something* current, and let the nightly
fill-gaps job (which always uses the default) pick up the rest of the
models over subsequent runs.

--corpus restricts everything (coverage checks, clearing, execution) to
one corpus, e.g. --corpus copyrighted-scans -- default is every corpus
`evaluation.harness.list_corpora()` finds.

--clear deletes every book's cache file within the --corpus scope (or all
corpora, if --corpus is omitted) before selecting/running models --
because the cache holds an LLM's output for a specific input, invalidating
that input (public-cache text changed -- a redaction fix, a corpus
promotion) invalidates every model's entry, not just the ones this
invocation happens to re-run. Combine with --mode fill-gaps (which then
sees a completely uncovered corpus and picks --limit models from
scratch) rather than --mode top5, which would refresh only its 5 models
unconditionally and leave the rest of the now-cleared cache empty until
something else fills it back in.

--concurrency (default 4): how many books to process concurrently for
the currently-selected model. KISSKI publishes no documented rate limit,
so this default is a conservative guess -- raise it if you don't observe
429/503 errors. Each request also retries up to 3 times with exponential
backoff (1s, 2s, 4s) on any failure before being logged as FAILED, so
transient errors under concurrency don't permanently block a book/model
pair.

--mode full: re-runs EVERY model that already has at least one cached
entry (its full historical footprint), across all books in all corpora,
regardless of current busy/demand status -- use after a change to the
extraction logic itself (prompt, max_tokens, page selection, ...) makes
every existing cache entry potentially stale, not just the 5 models
--mode top5 happens to touch. A cached model no longer offered by KISSKI is
skipped with a warning (nothing to run it against).

--endpoint ALIAS (repeatable): bypasses KISSKI discovery entirely and
runs the corpus against the given OpenAI-compatible endpoint(s) instead
-- e.g. an MPCDF LLM Inference Service session
(https://llm.mpcdf.mpg.de). Each ALIAS must have <ALIAS>_BASE_URL,
<ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment (see
evaluation/inference_endpoints.py). Mutually exclusive with --mode --
there's no discovery/demand concept for a model you deployed yourself,
so top5/fill-gaps/full's sweep-a-shared-pool semantics don't apply; every
given endpoint just runs once, unconditionally, over the corpus.
"""

import argparse
import asyncio
import functools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI

from chapter_segmentation.segmentation import analyze_attachment_llm_only
from evaluation.harness import available_public_books, list_corpora, llm_cache_dir, public_pages_for
from evaluation.inference_endpoints import ModelEndpoint, resolve_endpoint_from_env
from evaluation.kisski import (
    DEFAULT_KISSKI_BASE_URL, KisskiModel, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5,
)


class _OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) wrapping
    an already-built AsyncOpenAI client -- callers construct the client
    themselves (KISSKI's own base_url/api_key, or a ModelEndpoint's
    client from evaluation.inference_endpoints), so this class has no
    provider-specific knowledge at all."""

    def __init__(self, model: str, client: AsyncOpenAI):
        self._client = client
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


def _fully_covered_model_ids(book_specs: list[tuple[Path, str]]) -> set[str]:
    """book_specs: [(cache_dir, manifest_key), ...] -- one entry per book,
    each carrying its own corpus's llm_cache_dir(). A model id counts as
    covered only if EVERY given book's cache entry already has it. A book
    with no cache file at all has zero coverage -- every model is still a
    gap for it."""
    per_book_model_ids = []
    for cache_dir, manifest_key in book_specs:
        cache_path = cache_dir / f"{manifest_key}.json"
        if not cache_path.exists():
            return set()
        models = json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})
        per_book_model_ids.append(set(models))
    return set.intersection(*per_book_model_ids) if per_book_model_ids else set()


def _all_cached_model_ids(book_specs: list[tuple[Path, str]]) -> set[str]:
    """Every model id with at least one cached entry across any book --
    the "full regeneration" scope (a model's full historical footprint),
    unlike _fully_covered_model_ids' intersection-based "covered
    everywhere" definition used for gap-filling."""
    ids: set[str] = set()
    for cache_dir, manifest_key in book_specs:
        cache_path = cache_dir / f"{manifest_key}.json"
        if not cache_path.exists():
            continue
        ids.update(json.loads(cache_path.read_text(encoding="utf-8")).get("models", {}))
    return ids


def _has_cached_entry(cache_dir: Path, manifest_key: str, model_id: str) -> bool:
    """True if this book's cache file already has an entry for model_id --
    used by fill-gaps mode to skip work already done, so an interrupted run
    doesn't get redone from scratch the next time this model is selected.
    A malformed/corrupt cache file (e.g. from a process killed mid-write,
    before _upsert_cache wrote atomically) is treated as not-cached, so
    fill-gaps mode reprocesses and overwrites (self-heals) it instead of
    silently and permanently losing that book/model pair."""
    cache_path = cache_dir / f"{manifest_key}.json"
    if not cache_path.exists():
        return False
    try:
        models = json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})
    except json.JSONDecodeError:
        return False
    return model_id in models


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


async def _process_model(
    book_entries: list[tuple[str, str, Path]],
    concurrency: int,
    worker: Callable[[str, str, Path], Awaitable],
) -> None:
    """Runs worker(corpus, manifest_key, cache_dir) for every book_entries
    tuple concurrently, bounded by `concurrency` in-flight at once. worker
    is expected to handle its own errors and not raise -- but
    return_exceptions=True is passed defensively anyway, so even a
    surprise exception from one book can't cancel the others still
    in-flight via asyncio.gather's default all-or-nothing behavior."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(corpus: str, manifest_key: str, cache_dir: Path) -> None:
        async with semaphore:
            await worker(corpus, manifest_key, cache_dir)

    await asyncio.gather(
        *(_bounded(corpus, manifest_key, cache_dir) for corpus, manifest_key, cache_dir in book_entries),
        return_exceptions=True,
    )


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


def _upsert_cache(cache_dir: Path, manifest_key: str, model_id: str, chapters: list[dict], elapsed_seconds: float, demand: int) -> None:
    """Writes/updates model_id's cache entry for one book. Each model entry
    carries its own generated_at timestamp (not just the file-level one,
    which reflects whichever model was upserted most recently across the
    whole file) -- generate_report.py needs a per-model date to show how
    fresh THAT model's specific numbers are, since different models in the
    same file can have been refreshed on different nights. Writes
    atomically (temp file + rename) so a process killed mid-write (e.g. a
    job timeout) can never leave a truncated/corrupt cache file behind."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{manifest_key}.json"
    data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"models": {}}
    now = datetime.now(timezone.utc).isoformat()
    data["generated_at"] = now
    data["models"][model_id] = {
        "chapters": chapters,
        "elapsed_seconds": elapsed_seconds,
        "demand_at_run": demand,
        "generated_at": now,
    }
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(cache_path)


def _model_and_client_for_endpoint(endpoint: ModelEndpoint) -> tuple[KisskiModel, _OpenAICompatibleLLMClient]:
    """Wraps a resolved ModelEndpoint into the (model, llm_client) shape
    _run_book_for_model/_upsert_cache expect. demand=0 -- KisskiModel's
    own `demand` field has no meaning for an --endpoint-selected model
    (no shared pool, nothing to be busy relative to); 0 is also what
    KisskiModel.availability reads as "available", the only sensible
    default for a model you deployed yourself and know is up. Reuses
    KisskiModel itself rather than inventing a second (id, name, demand)
    type -- despite the name, it's just a model-identity-plus-demand
    record, not KISSKI-specific in shape."""
    model = KisskiModel(id=endpoint.model_id, name=endpoint.model_id, demand=0)
    llm_client = _OpenAICompatibleLLMClient(model=endpoint.model_id, client=endpoint.client)
    return model, llm_client


async def _main(
    mode: Optional[str], endpoint_aliases: Optional[list[str]], base_url: str, limit: int,
    corpus: Optional[str], clear: bool, concurrency: int,
) -> int:
    corpora = [corpus] if corpus else list_corpora()
    # (corpus, manifest_key, cache_dir) for every scorable book across every in-scope corpus.
    book_entries: list[tuple[str, str, Path]] = [
        (c, manifest_key, llm_cache_dir(c))
        for c in corpora
        for manifest_key, _expected_path, _book in available_public_books(c)
    ]
    if not book_entries:
        print("No public-cache evaluation books present.")
        return 1
    book_specs = [(cache_dir, manifest_key) for _corpus, manifest_key, cache_dir in book_entries]

    if clear:
        cleared = 0
        for _corpus, manifest_key, cache_dir in book_entries:
            cache_path = cache_dir / f"{manifest_key}.json"
            if cache_path.exists():
                cache_path.unlink()
                cleared += 1
        print(f"--clear: removed {cleared} cache file(s) across {len(corpora)} corpus/corpora before regenerating.")

    if endpoint_aliases:
        endpoints = [resolve_endpoint_from_env(alias) for alias in endpoint_aliases]
        print(f"Selected endpoints: {endpoint_aliases}")
        for endpoint in endpoints:
            model, llm_client = _model_and_client_for_endpoint(endpoint)
            worker = functools.partial(_run_book_for_model, model=model, mode="endpoint", llm_client=llm_client)
            await _process_model(book_entries, concurrency, worker)
        return 0

    api_key = os.environ["KISSKI_API_KEY"]
    all_models = fetch_kisski_models(base_url, api_key)
    if mode == "top5":
        selected = select_top5(all_models, limit=limit)
    elif mode == "fill-gaps":
        selected = select_gap_fill(all_models, _fully_covered_model_ids(book_specs), limit=limit)
    else:
        cached_ids = _all_cached_model_ids(book_specs)
        selected = select_full_regen(all_models, cached_ids)
        retired = sorted(cached_ids - {m.id for m in all_models})
        if retired:
            print(f"Skipping cached models no longer offered by KISSKI: {retired}")

    if not selected:
        if mode == "fill-gaps":
            print("No models to run (fill-gaps: every non-busy model already fully covered).")
        elif mode == "full":
            print("No models to run (full: no models are cached yet).")
        else:
            print("No models to run.")
        return 0

    print(f"Selected models: {[m.id for m in selected]}")
    for model in selected:
        llm_client = _OpenAICompatibleLLMClient(model=model.id, client=AsyncOpenAI(base_url=base_url, api_key=api_key))
        worker = functools.partial(_run_book_for_model, model=model, mode=mode, llm_client=llm_client)
        await _process_model(book_entries, concurrency, worker)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mode", choices=["top5", "fill-gaps", "full"], default=None)
    mode_group.add_argument(
        "--endpoint", action="append", default=None, metavar="ALIAS",
        help="Use one or more explicit OpenAI-compatible endpoints instead of KISSKI discovery -- repeatable, "
             "e.g. --endpoint MPCDF_A --endpoint MPCDF_B. Each ALIAS must have <ALIAS>_BASE_URL, "
             "<ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment. Runs every given endpoint once over the "
             "corpus (unconditionally, like --mode full); --mode's top5/fill-gaps/full sweep-a-shared-pool "
             "semantics don't apply since there's no discovery involved -- you already know exactly which "
             "model(s) you deployed.",
    )
    parser.add_argument("--base-url", default=DEFAULT_KISSKI_BASE_URL)
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max models to run this invocation (top5/fill-gaps only; full is always uncapped). Default 5.",
    )
    parser.add_argument("--corpus", help="Only refresh this corpus (default: every corpus under evaluation/corpus/)")
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete every cache file in scope before regenerating (use when the underlying public-cache text changed).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent book requests per model. KISSKI publishes no documented rate limit, "
             "so this is a conservative default -- raise it if you don't observe 429s. Default 4.",
    )
    args = parser.parse_args()
    if args.mode is None and not args.endpoint:
        args.mode = "top5"
    raise SystemExit(asyncio.run(_main(
        mode=args.mode, endpoint_aliases=args.endpoint, base_url=args.base_url, limit=args.limit,
        corpus=args.corpus, clear=args.clear, concurrency=args.concurrency,
    )))
