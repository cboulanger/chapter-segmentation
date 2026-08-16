"""Generates bulk-tier structured ground truth for dnb-toc-only (design
spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md,
which supersedes the two-text-extractor design in
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md).
For every manifest book not held out in eval_tier_ids.json (see
select_dnb_toc_eval_sample.py and evaluation/README.md's "Building
dnb-toc-only ground truth"), sends the book's page images to two
independent vision-capable KISSKI models
(evaluation.dnb_toc_vision.vision_extract_toc_entries) and writes
<id>.expected.json with "verified": false only when they agree well
enough (evaluation.dnb_toc_matching.gate_book, >=0.90 whole-book
agreement). Books that don't clear the gate are skipped and reported, not
partially written.

Spends real KISSKI API budget (two calls per book, one per vision model --
see evaluation/refresh_llm_cache.py's docstring for the shared
KISSKI_API_KEY setup this script reuses). Not a pytest test.

    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 50   # smoke test
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py               # full corpus
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

from openai import AsyncOpenAI

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_vision import vision_extract_toc_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key


def _cache_path(cache_directory: Path, key: str, model: str) -> Path:
    return cache_directory / f"{key}.{model}.json"


def _load_cached_llm_entries(cache_directory: Path, key: str, model: str) -> Optional[list[TocEntry]]:
    path = _cache_path(cache_directory, key, model)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TocEntry(
            title=e["title"], printed_page_number=e["printed_page_number"],
            source_page_index=e["source_page_index"], authors=tuple(e["authors"]),
            printed_roman=e["printed_roman"],
        )
        for e in data["entries"]
    ]


def _write_cached_llm_entries(cache_directory: Path, key: str, model: str, entries: list[TocEntry]) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_directory, key, model)
    data = {
        "generated_at": time.time(),
        "entries": [
            {
                "title": e.title, "printed_page_number": e.printed_page_number,
                "source_page_index": e.source_page_index, "authors": list(e.authors),
                "printed_roman": e.printed_roman,
            }
            for e in entries
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


async def _call_with_retry(coro_fn, attempts: int = 3, base_delay: float = 1.0, sleep=asyncio.sleep):
    """Same shape as evaluation/refresh_llm_cache.py's own retry helper
    (3 attempts, exponential backoff from base_delay) -- `sleep` is
    injectable so tests don't actually wait."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001 -- any failure here (network, parse) is retryable
            last_exc = exc
            if attempt < attempts - 1:
                await sleep(base_delay * 2 ** attempt)
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
        json.dumps({"entries": [toc_entry_to_gt_dict(e) for e in entries], "verified": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return key, True, "ok"


_CORPUS_NAME = "dnb-toc-only"


async def _run_book(
    key: str, pdf_path: Path, models: tuple[str, str], client, semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path,
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
    never abort the rest of a ~1000-book run."""
    try:
        entries_by_model = []
        for model in models:
            cached = _load_cached_llm_entries(cache_directory, key, model)
            async with semaphore:
                if cached is not None:
                    entries = cached
                else:
                    entries = await _call_with_retry(
                        lambda m=model: vision_extract_toc_entries(pdf_path, m, client)
                    )
                    # Only cache a non-empty result -- an empty list here
                    # could be a genuine "no TOC content on these pages" or
                    # a transient failure already exhausted by
                    # _call_with_retry; caching it either way would make a
                    # later re-run trust a possibly-transient empty result
                    # forever instead of retrying.
                    if entries:
                        _write_cached_llm_entries(cache_directory, key, model, entries)
            entries_by_model.append(entries)
        return _run_book_entries(key, entries_by_model[0], entries_by_model[1], corpus_directory)
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}")
        return key, False, f"error: {type(exc).__name__}"


# Vision-capable KISSKI model families, confirmed by direct experiment
# (design spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
# section 2.1) -- KISSKI's /models endpoint has no "supports vision" flag,
# so this is a curated allowlist, not something discoverable from the API
# response. Tried in this order: qwen-omni was faster and more accurate
# than gemma in the tested cases.
_VISION_MODEL_PATTERNS = (
    re.compile(r"^qwen\d+-omni"),
    re.compile(r"^gemma-\d+-"),
)


def _select_best_models(models: list, patterns=_VISION_MODEL_PATTERNS, count: int = 2) -> list[str]:
    """Picks `count` DISTINCT vision-capable model ids, one per pattern in
    preference order. Deliberately does NOT fall back to an arbitrary
    global least-busy model: a non-vision-capable model given image
    content would either error or silently ignore the images, and the
    whole point of the agreement gate is two INDEPENDENT reads -- gating a
    single model against itself (or against a model that never saw the
    images at all) would measure something other than what it claims to.
    Raises loudly rather than silently degrading to fewer models."""
    selected: list[str] = []
    for pattern in patterns:
        candidates = [
            m for m in models
            if pattern.match(m.id) and m.availability != "very busy" and m.id not in selected
        ]
        if candidates:
            selected.append(min(candidates, key=lambda m: m.demand).id)
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


def _generate(args: argparse.Namespace) -> int:
    cdir = corpus_dir(_CORPUS_NAME)
    eval_tier_path = cdir / "eval_tier_ids.json"
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()

    books = load_manifest_books(_CORPUS_NAME)
    eligible = [b for b in books if manifest_key(b) not in eval_tier_ids]
    if args.limit is not None:
        eligible = eligible[: args.limit]
    candidates = [(manifest_key(b), cdir / b["filename"]) for b in eligible if (cdir / b["filename"]).exists()]
    missing_pdf_count = len(eligible) - len(candidates)

    api_key = os.environ["KISSKI_API_KEY"]
    models = tuple(_pick_models(DEFAULT_KISSKI_BASE_URL, api_key))
    client = AsyncOpenAI(base_url=DEFAULT_KISSKI_BASE_URL, api_key=api_key)

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
