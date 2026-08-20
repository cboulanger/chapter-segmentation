"""Generates bulk-tier structured ground truth for dnb-toc-only (design
spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md,
which supersedes the two-text-extractor design in
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md).
For every manifest book not held out in eval_tier_ids.json (see
select_dnb_toc_eval_sample.py and evaluation/README.md's "Building
dnb-toc-only ground truth"), not already carrying a `.expected.json`
(bulk-gated or arbitrated), and not permanently rejected
(arbitration-rejected.json), sends the book's page images to two
independent vision-capable KISSKI models
(evaluation.dnb_toc_vision.vision_extract_toc_entries) and writes
<id>.expected.json with "verified": false only when they agree well
enough (evaluation.dnb_toc_matching.gate_book, >=0.90 whole-book
agreement). Books that don't clear the gate are skipped and reported, not
partially written -- run evaluation/scripts/arbitrate_dnb_toc.py on them
next. Skipping already-decided and rejected books means `--limit N` always
means "the next N books that still need a decision," so repeated
invocations advance through the corpus in batches instead of reprocessing
the same prefix every time. A bulk-gate `.expected.json` written before
the 2026-08-17 extraction-standard change (verbatim per-line extraction
plus a "skip" flag, replacing outright omission of front/back matter and
dividers -- see TocEntry.skip's docstring) counts as undecided again and
gets regenerated; an arbitrated one never does (see
`_is_stale_bulk_gate_entry`).

Spends real KISSKI API budget (two calls per book, one per vision model --
see evaluation/refresh_llm_cache.py's docstring for the shared
KISSKI_API_KEY setup this script reuses). Not a pytest test.

Safe to run two invocations concurrently against the same checkout (e.g. a
KISSKI-backed run and an MPCDF-backed `--endpoint` run) -- each book is
claimed via a per-key lock file under `.locks/` before either process
touches its cache or spends an API call on it, see `_acquire_lock`'s
docstring.

    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 50   # next batch of 50
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py               # all remaining books
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --spot-check 30

`--spot-check N` does not generate anything -- instead it samples N books
that already passed the bulk-tier gate (i.e. have "verified": false;
already-human-verified eval-tier entries are excluded) and walks through a
manual, terminal-driven visual Accept/Reject check against the real PDF for
each, then prints the measured accept rate as an estimate of the gate's
real precision.
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI, RateLimitError

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_ocr import text_extract_toc_entries
from evaluation.dnb_toc_vision import load_cached_llm_entries, vision_extract_toc_entries, write_cached_llm_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.inference_endpoints import (
    DEFAULT_SESSIONS_FILENAME, DEFAULT_TIMEOUT, ModelEndpoint, resolve_endpoint_from_env,
    resolve_endpoints_from_config_file,
)
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key


_RATE_LIMIT_WINDOW_ORDER = ("day", "hour", "minute")

# Windows worth sleeping out inline and retrying within THIS run. "day" is
# deliberately excluded -- see _call_with_retry's docstring.
_INLINE_RETRY_WINDOWS = frozenset({"hour", "minute"})


def _binding_rate_limit_window(headers) -> Optional[str]:
    """Which of KISSKI's `x-ratelimit-remaining-<window>` response headers
    is actually at 0 -- i.e. which window is the real reason this request
    was rejected (confirmed header shape: evaluation/experiments/dnb-toc-ground-truth.md's
    "genuine daily quota" finding, headers.get() is case-insensitive on both openai's and
    httpx's Headers types). Returns the LONGEST zeroed window (day > hour >
    minute) when more than one is reported at 0, since that's the one
    whose reset actually gates recovery -- waiting out an exhausted
    per-minute window changes nothing if per-day is also at 0. None if no
    `remaining-*` header reports exactly "0" (e.g. a 429 for some other
    reason, or KISSKI changes its header shape without notice) -- callers
    must treat that the same as an unclassifiable rate limit."""
    if not headers:
        return None
    zeroed = {
        key.lower().rsplit("-", 1)[-1]
        for key, value in headers.items()
        if key.lower().startswith("x-ratelimit-remaining-") and value.strip() == "0"
    }
    for window in _RATE_LIMIT_WINDOW_ORDER:
        if window in zeroed:
            return window
    return None


def _retry_after_seconds(headers) -> Optional[float]:
    """Parses KISSKI's `retry-after` response header (seconds until the
    binding window resets) into a float, or None if absent/unparseable."""
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def _call_with_retry(
    coro_fn, attempts: int = 6, base_delay: float = 2.0, rate_limit_delay: float = 20.0, sleep=asyncio.sleep,
):
    """Same shape as evaluation/refresh_llm_cache.py's own retry helper
    (exponential backoff from base_delay) for a non-429 failure. A 429
    instead schedules its retry from the response's own rate-limit headers
    when present (`retry-after` for the exact delay, `x-ratelimit-remaining-
    <window>` to identify which window is actually binding -- see
    _binding_rate_limit_window/_retry_after_seconds), falling back to the
    old blind `rate_limit_delay * attempt_number` linear backoff only when
    those headers are missing.

    A 429 whose binding window is "day" gives up immediately instead of
    sleeping -- found empirically (2026-08-17/18 batch runs) that KISSKI's
    daily quota, once exhausted, does not reset within a single script
    invocation's realistic lifetime (`retry-after` observed as high as
    ~54179s, i.e. ~15h), so blind or even header-precise inline retrying
    for it just burns wall time one book at a time discovering the same
    fact the first 429 already established (a ~6.5h run once lost to
    exactly this). A "day"-bound 429 is instead reported as a failure right
    away; re-invoking the script once the daily quota actually resets
    already skips every book with a cached/decided result, so nothing
    extra is lost by not waiting inline. "hour"/"minute" windows (and an
    unclassifiable 429 with no headers at all) DO retry inline, since those
    can plausibly clear before `attempts` is exhausted. `sleep` is
    injectable so tests don't actually wait."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001 -- any failure here (network, parse) is retryable
            last_exc = exc
            if attempt >= attempts - 1:
                break
            if isinstance(exc, RateLimitError):
                response = getattr(exc, "response", None)
                headers = response.headers if response is not None else None
                window = _binding_rate_limit_window(headers)
                if window is not None and window not in _INLINE_RETRY_WINDOWS:
                    break
                retry_after = _retry_after_seconds(headers)
                delay = retry_after if retry_after is not None else rate_limit_delay * (attempt + 1)
            else:
                delay = base_delay * 2 ** attempt
            await sleep(delay)
    raise last_exc


_GATE_THRESHOLD = 0.90


def _run_book_entries(
    key: str, entries_a: list[TocEntry], entries_b: list[TocEntry], corpus_directory: Path,
) -> tuple[str, bool, str]:
    """Core per-book gating logic, given two already-extracted TocEntry
    lists -- kept separate from PDF/vision-call I/O so it's directly
    unit-testable with synthetic entries, no real PDF or network call
    needed. Returns (key, passed, reason); reason is "ok" on success, else
    why the book was skipped/rejected ("no_entries", "below_threshold")."""
    if not entries_a and not entries_b:
        return key, False, "no_entries"
    passed, entries = gate_book(entries_a, entries_b, threshold=_GATE_THRESHOLD)
    if not passed:
        return key, False, "below_threshold"
    gt_path = corpus_directory / f"{key}.expected.json"
    gt_path.write_text(
        json.dumps(
            {"entries": [toc_entry_to_gt_dict(e) for e in entries], "verified": False, "source": "bulk_gate"},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return key, True, "ok"


_CORPUS_NAME = "dnb-toc-only"

# How long a lock file is trusted before a later run treats it as
# abandoned (a crashed/killed process that skipped its `finally` release)
# and reclaims it -- generous relative to one book's worst-case retry
# sequence (a handful of vision calls plus backoff), short enough that a
# genuinely dead lock doesn't block a book forever.
_LOCK_STALE_AFTER_SECONDS = 1800.0


def _lock_path(corpus_directory: Path, key: str) -> Path:
    return corpus_directory / ".locks" / f"{key}.lock"


def _acquire_lock(corpus_directory: Path, key: str, *, stale_after: float = _LOCK_STALE_AFTER_SECONDS) -> bool:
    """Claims `key` for this process via an atomic exclusive file create --
    the standard cross-process mutex primitive, safe against two separate
    `generate_dnb_toc_ground_truth.py` invocations (e.g. a KISSKI-backed run
    and an MPCDF-backed run sharing the same checkout) racing on the same
    book. `_generate`'s `eligible` list is a snapshot taken once at
    startup, so it has no way to see a book another already-running process
    claimed after that snapshot -- the lock is what actually prevents both
    from spending API budget on it.

    A lock older than `stale_after` is assumed to belong to a process that
    crashed or was killed before reaching `_release_lock`'s `finally` (a
    normal exit, including a caught exception, always releases) -- reclaimed
    by deleting it and retrying the exclusive create once. If a third
    process wins that retry first, this returns False like any other
    lost race; the loser simply skips the book this run and picks it up
    (or finds it already decided) next time."""
    lock_path = _lock_path(corpus_directory, key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        pass
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        age = None  # released between our failed create and this stat -- fall through to retry
    if age is not None and age < stale_after:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def _release_lock(corpus_directory: Path, key: str) -> None:
    try:
        _lock_path(corpus_directory, key).unlink()
    except FileNotFoundError:
        pass


async def _run_book(
    key: str, pdf_path: Path, endpoints: tuple[ModelEndpoint, ModelEndpoint], semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path, sleep=asyncio.sleep,
) -> tuple[str, bool, str]:
    """Thin I/O wrapper around _run_book_entries -- calls
    vision_extract_toc_entries once per endpoint (through the cache, then
    _call_with_retry on a miss), and delegates the two resulting entry
    lists to _run_book_entries. `endpoints` carries each side's own
    client, not a single shared one -- the two independent vision reads
    can come from entirely different inference endpoints (e.g. two MPCDF
    sessions, or one MPCDF + one KISSKI model), not just two models
    behind KISSKI's single base URL. Catches any exception (a corrupt/
    unreadable PDF, a network error that survives _call_with_retry's own
    retries, etc.) and reports it as a failed-but-tuple-shaped result
    instead of letting it propagate -- same "catch-log-continue"
    convention evaluation/refresh_llm_cache.py already established for
    this kind of long, unattended, budget-spending batch job. One book's
    failure must never abort the rest of a ~1000-book run.

    `semaphore` is acquired only around each individual API call attempt
    (inside the closure passed to _call_with_retry), NOT around the whole
    retry sequence -- found the hard way (2026-08-17 batch run) that
    holding it for the full sequence lets a backoff sleep occupy a
    concurrency slot for up to minutes, and if enough books hit
    RateLimitError around the same time, every slot ends up asleep at once
    and the entire batch stalls with zero throughput even though nothing
    actually crashed. Releasing it between attempts lets other books make
    progress while one book backs off.

    Claims `key` via `_acquire_lock` before doing any cache lookup or API
    call, and always releases it in a `finally` -- guards against a
    SEPARATE process (not this run's own `semaphore`, which only bounds
    concurrency within one process) picking up the same book, see
    `_acquire_lock`'s own docstring. A lost race returns immediately with
    reason "locked_by_another_process" -- cheap, no cache/API touched."""
    if not _acquire_lock(corpus_directory, key):
        return key, False, "locked_by_another_process"
    try:
        entries_by_model = []
        for endpoint in endpoints:
            cached = load_cached_llm_entries(cache_directory, key, endpoint.model_id)
            if cached is not None:
                entries = cached
            else:
                async def _call(ep=endpoint):
                    async with semaphore:
                        return await vision_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                entries = await _call_with_retry(_call, sleep=sleep)
                # Only cache a non-empty result -- an empty list here
                # could be a genuine "no TOC content on these pages" or
                # a transient failure already exhausted by
                # _call_with_retry; caching it either way would make a
                # later re-run trust a possibly-transient empty result
                # forever instead of retrying.
                if entries:
                    write_cached_llm_entries(cache_directory, key, endpoint.model_id, entries)
            entries_by_model.append(entries)
        return _run_book_entries(key, entries_by_model[0], entries_by_model[1], corpus_directory)
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}{_rate_limit_headers_suffix(exc)}")
        return key, False, f"error: {type(exc).__name__}"
    finally:
        _release_lock(corpus_directory, key)


def _rate_limit_headers_suffix(exc: Exception) -> str:
    """For a RateLimitError, appends whichever of KISSKI's per-minute/
    per-hour/per-day limit+remaining response headers are present, so a
    batch log directly shows which window is actually binding instead of
    requiring a separate one-off probe script -- see the "genuine daily
    quota" investigation in evaluation/experiments/dnb-toc-ground-truth.md, which had to inspect
    e.response.headers by hand to establish this. Empty string for any
    other exception type or a response with no such headers."""
    response = getattr(exc, "response", None)
    if not isinstance(exc, RateLimitError) or response is None:
        return ""
    relevant = {k: v for k, v in response.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
    if not relevant:
        return ""
    return " [" + ", ".join(f"{k}={v}" for k, v in sorted(relevant.items())) + "]"


# Vision-capable KISSKI model families, confirmed by direct experiment
# (design spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
# section 2.1) -- KISSKI's /models endpoint has no "supports vision" flag,
# so this is a curated allowlist, not something discoverable from the API
# response.
#
# gemma-4-31b-it was the original second pattern but was dropped after a
# real 15-book smoke test (evaluation/experiments/dnb-toc-ground-truth.md, 2026-08-16) found it
# doesn't just extract at a coarser granularity than qwen-omni -- on 5 of
# 8 below-threshold books it silently DROPPED the entire early portion of
# the TOC (e.g. one clean 8-entry numbered list came back with only the
# last 2 entries), even on short, simple 2-page scans. qwen3.6-27b was
# spot-checked against the same books and correctly covered the full
# page range every time (matching qwen-omni's own range), so it replaced
# gemma as the second family -- slower per call, but reliable, which
# matters far more for an agreement gate than speed.
#
# Pinned to qwen3.6 specifically, NOT a version-agnostic "any qwenX.Y"
# pattern: a broader pattern once picked qwen3.5-122b-a10b instead (lower
# demand at request time), which turned out measurably less reliable on
# this exact task -- it returned an empty response or malformed JSON on
# 2 of 3 books tested with the (more verbose, nested-sub-point-inclusive)
# prompt below, where qwen3.6-27b succeeded on all of them. Sibling
# versions of the same model family are not interchangeable in practice;
# re-validate before widening this again.
_VISION_MODEL_PATTERNS = (
    re.compile(r"^qwen\d+-omni"),
    re.compile(r"^qwen3\.6-"),
)


def _select_best_models(models: list, patterns=_VISION_MODEL_PATTERNS, count: int = 2) -> list[str]:
    """Picks `count` DISTINCT vision-capable model ids. Walks `patterns` in
    preference order, taking as many least-busy candidates as a pattern
    has available (falling through to the next pattern only once the
    current one is exhausted) until `count` is reached -- so a busy/absent
    family doesn't abort the run when an earlier family alone has enough
    distinct, available candidates to satisfy `count`. Deliberately does
    NOT fall back to an arbitrary global least-busy model: a non-vision-
    capable model given image content would either error or silently
    ignore the images, and the whole point of the agreement gate is two
    INDEPENDENT reads -- gating a single model against itself (or against
    a model that never saw the images at all) would measure something
    other than what it claims to. Raises loudly rather than silently
    degrading to fewer models."""
    selected: list[str] = []
    for pattern in patterns:
        candidates = sorted(
            (m for m in models if pattern.match(m.id) and m.availability != "very busy" and m.id not in selected),
            key=lambda m: m.demand,
        )
        for candidate in candidates:
            selected.append(candidate.id)
            if len(selected) >= count:
                break
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Need {count} distinct vision-capable models, found {len(selected)}: {selected}")
    return selected


def _pick_models(base_url: str, api_key: str) -> list[str]:
    return _select_best_models(fetch_kisski_models(base_url, api_key))


def _resolve_vision_endpoints(
    endpoint_aliases: Optional[list[str]], config_file: Optional[Path] = None,
) -> tuple[ModelEndpoint, ModelEndpoint]:
    """Resolves the two ModelEndpoints the two-independent-vision-model
    gate calls. Neither --endpoint nor --config-file given -> today's
    default: KISSKI discovery picks two distinct vision-capable models,
    sharing one client (both live behind the same KISSKI base URL,
    unchanged from before this endpoint abstraction existed). --config-file
    PATH -> both endpoints come from PATH's pasted session tables, in file
    order (must contain exactly two). Exactly two --endpoint aliases ->
    each resolved independently via resolve_endpoint_from_env, letting the
    two reads come from different endpoints/providers (e.g. two MPCDF
    sessions, or one MPCDF session + one manually-picked KISSKI model).
    Any other alias count, or a --config-file with a table count != 2, is
    a user error -- the gate's independence guarantee requires exactly two
    reads. --endpoint and --config-file are mutually exclusive (enforced
    by the CLI parser's mutually exclusive group)."""
    if config_file:
        endpoints = resolve_endpoints_from_config_file(config_file)
        if len(endpoints) != 2:
            raise SystemExit(
                f"--config-file requires exactly 2 pasted session tables for the two-independent-model gate, "
                f"got {len(endpoints)} in {config_file}"
            )
        return (endpoints[0], endpoints[1])
    if not endpoint_aliases:
        api_key = os.environ["KISSKI_API_KEY"]
        model_ids = tuple(_pick_models(DEFAULT_KISSKI_BASE_URL, api_key))
        # Explicit per-request timeout -- the openai SDK's own default
        # (600s read timeout) let one slow/hung KISSKI response occupy a
        # concurrency slot for up to 10 minutes PER ATTEMPT, times up to 6
        # retry attempts (_call_with_retry's default), a worst case over
        # an hour for a single book (found live, 2026-08-17: a batch
        # stalled with 4 connections to KISSKI stuck ESTABLISHED for 20+
        # minutes, well past this script's typical successful per-call
        # latency). 90s is generous for a 1-4 page TOC scan's vision call
        # while still bounding the worst case.
        client = AsyncOpenAI(base_url=DEFAULT_KISSKI_BASE_URL, api_key=api_key, timeout=DEFAULT_TIMEOUT)
        return (
            ModelEndpoint(label="kisski", model_id=model_ids[0], client=client),
            ModelEndpoint(label="kisski", model_id=model_ids[1], client=client),
        )
    if len(endpoint_aliases) != 2:
        raise SystemExit(
            f"--endpoint requires exactly 2 aliases for the two-independent-model gate, "
            f"got {len(endpoint_aliases)}: {endpoint_aliases}"
        )
    return tuple(resolve_endpoint_from_env(alias) for alias in endpoint_aliases)


def _resolve_endpoints(
    endpoint_aliases: Optional[list[str]],
    config_file: Optional[Path],
    text_endpoint_alias: Optional[str],
    text_config_file: Optional[Path],
) -> tuple[ModelEndpoint, ModelEndpoint, str]:
    """Resolves the gate's two endpoints plus which extraction path the
    second one needs ("vision" for vision_extract_toc_entries, "text" for
    text_extract_toc_entries) -- see design spec
    docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
    section 1's combination table. Neither --text-endpoint nor
    --text-config-file given delegates entirely to _resolve_vision_endpoints
    (today's two-vision-model behavior, completely unchanged), second_kind
    "vision". Either text flag given pairs exactly one vision-side endpoint
    (--endpoint with exactly 1 alias, or --config-file with exactly 1
    pasted session table) with exactly one text-side endpoint
    (--text-endpoint, or --text-config-file with exactly 1 table),
    second_kind "text" -- the vision and text sides may use different
    sourcing mechanisms freely (e.g. vision via --endpoint, text via
    --text-config-file). Any other shape (e.g. 2 vision endpoints ALSO
    given a text endpoint, or a text flag with no vision side at all) is a
    user error, raising SystemExit naming exactly what's wrong -- same
    style as _resolve_vision_endpoints' own existing errors."""
    if not text_endpoint_alias and not text_config_file:
        vision_a, vision_b = _resolve_vision_endpoints(endpoint_aliases, config_file)
        return vision_a, vision_b, "vision"

    if config_file:
        vision_endpoints = resolve_endpoints_from_config_file(config_file)
        if len(vision_endpoints) != 1:
            raise SystemExit(
                f"--config-file paired with --text-endpoint/--text-config-file requires exactly 1 pasted "
                f"session table for the vision side, got {len(vision_endpoints)} in {config_file}"
            )
        vision_endpoint = vision_endpoints[0]
    elif endpoint_aliases:
        if len(endpoint_aliases) != 1:
            raise SystemExit(
                f"--endpoint paired with --text-endpoint/--text-config-file requires exactly 1 alias for the "
                f"vision side, got {len(endpoint_aliases)}: {endpoint_aliases}"
            )
        vision_endpoint = resolve_endpoint_from_env(endpoint_aliases[0])
    else:
        raise SystemExit(
            "--text-endpoint/--text-config-file requires a vision-side --endpoint or --config-file too -- "
            "the gate needs one vision read and one text read"
        )

    if text_config_file:
        text_endpoints = resolve_endpoints_from_config_file(text_config_file)
        if len(text_endpoints) != 1:
            raise SystemExit(
                f"--text-config-file requires exactly 1 pasted session table, got {len(text_endpoints)} in "
                f"{text_config_file}"
            )
        text_endpoint = text_endpoints[0]
    else:
        text_endpoint = resolve_endpoint_from_env(text_endpoint_alias)

    return vision_endpoint, text_endpoint, "text"


async def _run_all(
    keys_and_paths: list[tuple[str, Path]], endpoints: tuple[ModelEndpoint, ModelEndpoint], concurrency: int,
    corpus_directory: Path, cache_directory: Path,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, endpoints, semaphore, corpus_directory, cache_directory)
        for key, path in keys_and_paths
    ]))


def _is_stale_bulk_gate_entry(gt_path: Path) -> bool:
    """True if gt_path is a BULK-tier (`"source": "bulk_gate"`) file
    written before the 2026-08-17 extraction-standard change (verbatim
    per-line extraction with a `"skip"` flag, replacing the vision
    prompt's own outright omission of front/back matter and dividers --
    see TocEntry.skip's docstring) -- recognized by at least one entry
    missing the "skip" key, since that key didn't exist before the
    change. Deliberately restricted to `bulk_gate`: a `claude_arbitration`
    file went through direct human/Claude review and must never be
    silently overwritten by a fresh, unreviewed automated gate result just
    because it also predates this key -- those need a deliberate manual
    retrofit (reopen the PDF, add the previously-omitted lines with
    "skip": true) instead, tracked separately rather than reprocessed by
    this script."""
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if data.get("source") != "bulk_gate":
        return False
    entries = data.get("entries", [])
    return bool(entries) and not all("skip" in e for e in entries)


def _still_needs_a_decision(book: dict, cdir: Path, eval_tier_ids: set[str], rejected_ids: set[str]) -> bool:
    """True if `book` hasn't already been settled one way or another --
    held out for the eval tier, already carrying a current-schema
    `.expected.json` (bulk-gated or arbitrated), or permanently rejected.
    A pre-2026-08-17 bulk-gate file (see `_is_stale_bulk_gate_entry`)
    counts as NOT yet decided, so it's regenerated under the current
    verbatim-extraction standard rather than left stale forever; a
    pre-2026-08-17 arbitration file is left alone (not this script's job
    to touch). Factored out of `_generate` so repeated invocations
    naturally advance to the next undecided books instead of reprocessing
    the same prefix, and so this filtering is unit-testable without a
    real corpus/API key."""
    key = manifest_key(book)
    if key in eval_tier_ids or key in rejected_ids:
        return False
    gt_path = cdir / f"{key}.expected.json"
    if not gt_path.exists():
        return True
    return _is_stale_bulk_gate_entry(gt_path)


def _generate(args: argparse.Namespace) -> int:
    cdir = corpus_dir(_CORPUS_NAME)
    eval_tier_path = cdir / "eval_tier_ids.json"
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()
    rejected_path = cdir / "arbitration-rejected.json"
    rejected_ids = (
        {entry["key"] for entry in json.loads(rejected_path.read_text(encoding="utf-8"))["rejected"]}
        if rejected_path.exists() else set()
    )

    books = load_manifest_books(_CORPUS_NAME)
    eligible = [b for b in books if _still_needs_a_decision(b, cdir, eval_tier_ids, rejected_ids)]
    if args.limit is not None:
        eligible = eligible[: args.limit]
    candidates = [(manifest_key(b), cdir / b["filename"]) for b in eligible if (cdir / b["filename"]).exists()]
    missing_pdf_count = len(eligible) - len(candidates)

    endpoints = _resolve_vision_endpoints(args.endpoint, args.config_file)

    results = asyncio.run(_run_all(candidates, endpoints, args.concurrency, cdir, llm_cache_dir(_CORPUS_NAME)))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print(
        f"Vision models used: {endpoints[0].label}:{endpoints[0].model_id}, "
        f"{endpoints[1].label}:{endpoints[1].model_id}"
    )
    print(f"{len(passed)}/{len(results)} books passed the gate and got .expected.json written.")
    for reason, count in sorted(by_reason.items()):
        print(f"  {count} skipped: {reason}")
    if missing_pdf_count:
        print(f"  {missing_pdf_count} skipped: missing_pdf (not downloaded locally)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many books (smoke-test convenience)")
    parser.add_argument("--concurrency", type=int, default=4, help="How many books to process concurrently (default: 4)")
    parser.add_argument(
        "--spot-check", type=int, default=None, metavar="N",
        help="Instead of generating, sample N passing bulk-tier books and walk through a visual Accept/Reject check",
    )
    endpoint_group = parser.add_mutually_exclusive_group()
    endpoint_group.add_argument(
        "--endpoint", action="append", default=None, metavar="ALIAS",
        help="Use an explicit OpenAI-compatible endpoint instead of KISSKI auto-discovery for the VISION side -- "
             "pass exactly twice for two independent vision reads (e.g. --endpoint MPCDF_A --endpoint MPCDF_B), "
             "or exactly once when paired with --text-endpoint/--text-config-file. Each ALIAS must have "
             "<ALIAS>_BASE_URL, <ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment.",
    )
    endpoint_group.add_argument(
        "--config-file", nargs="?", const=DEFAULT_SESSIONS_FILENAME, default=None, metavar="PATH",
        help="Same as --endpoint, but sources the vision endpoint(s) from a pasted-session-table file instead of "
             f"env vars -- PATH defaults to {DEFAULT_SESSIONS_FILENAME} when omitted; must contain exactly 2 "
             "pasted session tables (two vision reads), or exactly 1 when paired with "
             "--text-endpoint/--text-config-file. See evaluation/hpc/llm-mpcdf.md.",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--text-endpoint", default=None, metavar="ALIAS",
        help="Pair the vision endpoint (--endpoint or --config-file, exactly 1 either way) with a text-only "
             "endpoint fed freshly-OCR'd page text instead of a second vision read -- ALIAS must have "
             "<ALIAS>_BASE_URL, <ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment. See design spec "
             "docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md.",
    )
    text_group.add_argument(
        "--text-config-file", nargs="?", const=DEFAULT_SESSIONS_FILENAME, default=None, metavar="PATH",
        help="Same as --text-endpoint, but sources the text endpoint from a pasted-session-table file -- PATH "
             f"defaults to {DEFAULT_SESSIONS_FILENAME} when omitted; must contain exactly 1 pasted session table.",
    )
    args = parser.parse_args()
    if args.config_file:
        args.config_file = Path(args.config_file)
    if args.text_config_file:
        args.text_config_file = Path(args.text_config_file)
    if args.spot_check is not None:
        return _spot_check(corpus_dir(_CORPUS_NAME), args.spot_check)
    return _generate(args)


def _spot_check(cdir: Path, n: int) -> int:
    """Terminal-driven precision check (design spec section 7): sample n
    books that passed the bulk-tier gate, print each one's PDF path and
    generated entries, and prompt for a manual Accept/Reject after
    visually opening the PDF (e.g. via the Read tool's pages param, same
    as the manual eval-tier transcription workflow in evaluation/README.md)
    -- then report measured precision for the >=0.90 gate threshold.
    Only samples from books whose "verified" field is False (bulk-tier,
    machine-gated) -- excludes eval-tier books, which carry
    "verified": true once independently human-verified and would
    trivially pass the Accept prompt, inflating the measured precision."""
    passing = []
    for p in sorted(cdir.glob("*.expected.json")):
        gt = json.loads(p.read_text(encoding="utf-8"))
        if gt.get("verified") is False:
            passing.append(p.name.removesuffix(".expected.json"))
    sample = random.sample(passing, min(max(n, 0), len(passing)))
    accepted = 0
    for key in sample:
        gt = json.loads((cdir / f"{key}.expected.json").read_text(encoding="utf-8"))
        print(f"\n=== {key} ===\nPDF: {cdir / f'{key}.pdf'}")
        print(json.dumps(gt["entries"], indent=2, ensure_ascii=False))
        answer = input("Matches the scan? [y/N] ").strip().lower()
        if answer == "y":
            accepted += 1
    if sample:
        print(f"\nSpot-check precision: {accepted}/{len(sample)} = {accepted / len(sample):.0%}")
    else:
        print("No passing books found to spot-check yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
