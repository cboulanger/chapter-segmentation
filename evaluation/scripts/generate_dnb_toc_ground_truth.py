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
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI, RateLimitError

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_vision import load_cached_llm_entries, vision_extract_toc_entries, write_cached_llm_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
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


async def _run_book(
    key: str, pdf_path: Path, models: tuple[str, str], client, semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path, sleep=asyncio.sleep,
) -> tuple[str, bool, str]:
    """Thin I/O wrapper around _run_book_entries -- calls
    vision_extract_toc_entries once per model (through the cache, then
    _call_with_retry on a miss), and delegates the two resulting entry
    lists to _run_book_entries. Catches any exception (a corrupt/unreadable
    PDF, a network error that survives _call_with_retry's own retries,
    etc.) and reports it as a failed-but-tuple-shaped result instead of
    letting it propagate -- same "catch-log-continue" convention
    evaluation/refresh_llm_cache.py already established for this kind of
    long, unattended, budget-spending batch job. One book's failure must
    never abort the rest of a ~1000-book run.

    `semaphore` is acquired only around each individual API call attempt
    (inside the closure passed to _call_with_retry), NOT around the whole
    retry sequence -- found the hard way (2026-08-17 batch run) that
    holding it for the full sequence lets a backoff sleep occupy a
    concurrency slot for up to minutes, and if enough books hit
    RateLimitError around the same time, every slot ends up asleep at once
    and the entire batch stalls with zero throughput even though nothing
    actually crashed. Releasing it between attempts lets other books make
    progress while one book backs off."""
    try:
        entries_by_model = []
        for model in models:
            cached = load_cached_llm_entries(cache_directory, key, model)
            if cached is not None:
                entries = cached
            else:
                async def _call(m=model):
                    async with semaphore:
                        return await vision_extract_toc_entries(pdf_path, m, client)
                entries = await _call_with_retry(_call, sleep=sleep)
                # Only cache a non-empty result -- an empty list here
                # could be a genuine "no TOC content on these pages" or
                # a transient failure already exhausted by
                # _call_with_retry; caching it either way would make a
                # later re-run trust a possibly-transient empty result
                # forever instead of retrying.
                if entries:
                    write_cached_llm_entries(cache_directory, key, model, entries)
            entries_by_model.append(entries)
        return _run_book_entries(key, entries_by_model[0], entries_by_model[1], corpus_directory)
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}{_rate_limit_headers_suffix(exc)}")
        return key, False, f"error: {type(exc).__name__}"


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


async def _run_all(
    keys_and_paths: list[tuple[str, Path]], models: tuple[str, str], client, concurrency: int,
    corpus_directory: Path, cache_directory: Path,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, models, client, semaphore, corpus_directory, cache_directory)
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

    api_key = os.environ["KISSKI_API_KEY"]
    models = tuple(_pick_models(DEFAULT_KISSKI_BASE_URL, api_key))
    # Explicit per-request timeout -- the openai SDK's own default (600s
    # read timeout) let one slow/hung KISSKI response occupy a concurrency
    # slot for up to 10 minutes PER ATTEMPT, times up to 6 retry attempts
    # (_call_with_retry's default), a worst case over an hour for a single
    # book (found live, 2026-08-17: a batch stalled with 4 connections to
    # KISSKI stuck ESTABLISHED for 20+ minutes, well past this script's
    # typical successful per-call latency). 90s is generous for a 1-4 page
    # TOC scan's vision call while still bounding the worst case.
    client = AsyncOpenAI(base_url=DEFAULT_KISSKI_BASE_URL, api_key=api_key, timeout=90.0)

    results = asyncio.run(_run_all(candidates, models, client, args.concurrency, cdir, llm_cache_dir(_CORPUS_NAME)))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print(f"Vision models used: {models[0]}, {models[1]}")
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
    args = parser.parse_args()
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
