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

--mode full: re-runs EVERY model that already has at least one cached
entry (its full historical footprint), across all books, regardless of
current busy/demand status -- use after a change to the extraction logic
itself (prompt, max_tokens, page selection, ...) makes every existing
cache entry potentially stale, not just the 5 models --mode top5 happens
to touch. A cached model no longer offered by KISSKI is skipped with a
warning (nothing to run it against).
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
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5


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


def _all_cached_model_ids(manifest_keys: list[str]) -> set[str]:
    """Every model id with at least one cached entry across any book --
    the "full regeneration" scope (a model's full historical footprint),
    unlike _fully_covered_model_ids' intersection-based "covered
    everywhere" definition used for gap-filling."""
    ids: set[str] = set()
    for manifest_key in manifest_keys:
        cache_path = LLM_CACHE_DIR / f"{manifest_key}.json"
        if not cache_path.exists():
            continue
        ids.update(json.loads(cache_path.read_text(encoding="utf-8")).get("models", {}))
    return ids


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
    elif mode == "fill-gaps":
        selected = select_gap_fill(all_models, _fully_covered_model_ids(manifest_keys))
    else:
        cached_ids = _all_cached_model_ids(manifest_keys)
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
        llm_client = _OpenAICompatibleLLMClient(model=model.id, base_url=base_url, api_key=api_key)
        for manifest_key, _expected_path, _book in books:
            try:
                pages = public_pages_for(manifest_key)
                start = time.perf_counter()
                result = await analyze_attachment_llm_only(pages, llm_client)
                elapsed = time.perf_counter() - start
                _upsert_cache(manifest_key, model.id, result["chapters"], elapsed, model.demand)
                print(f"{manifest_key} / {model.id}: {len(result['chapters'])} chapters, {elapsed:.1f}s")
            except Exception as exc:
                # One book/model failure must not strand the whole batch or
                # discard cache entries already written for other books/
                # models in this same run -- same catch-log-continue
                # convention as generate_public_evaluation_cache.py.
                print(f"{manifest_key} / {model.id}: FAILED ({exc}) -- skipping")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["top5", "fill-gaps", "full"], default="top5")
    parser.add_argument("--base-url", default=DEFAULT_KISSKI_BASE_URL)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(mode=args.mode, base_url=args.base_url)))
