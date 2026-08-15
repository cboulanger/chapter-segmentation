"""Generates bulk-tier structured ground truth for dnb-toc-only (design
spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md).
For every manifest book not held out in eval_tier_ids.json (see
select_dnb_toc_eval_sample.py and evaluation/README.md's "Building
dnb-toc-only ground truth"), runs two independent extractors -- the regex
heuristic (find_toc_candidates) and a KISSKI LLM pass
(llm_extract_toc_entries) -- and writes <id>.expected.json with
"verified": false only when they agree well enough
(evaluation.dnb_toc_matching.gate_book, >=0.90 whole-book agreement).
Books that don't clear the gate are skipped and reported, not partially
written.

Spends real KISSKI API budget (one call per book, not per-model -- see
evaluation/refresh_llm_cache.py's docstring for the shared KISSKI_API_KEY
setup this script reuses). Not a pytest test.

    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 50   # smoke test
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py               # full corpus
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --spot-check 30
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from chapter_segmentation.llm import LLMClient
from chapter_segmentation.segmentation import (
    TocEntry,
    extract_page_texts_for_analysis,
    find_toc_candidates,
    llm_extract_toc_entries,
    pages_need_ocr,
)
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models
from evaluation.refresh_llm_cache import _OpenAICompatibleLLMClient
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key

# find_toc_candidates rejects any printed page number above
# len(pages) * _TOC_MAX_PAGE_NUMBER_RATIO (2.0, segmentation.py) -- a
# guard against mistaking book-internal noise for a real TOC elsewhere in
# a full book. A dnb-toc-only PDF *is* the TOC (1-3 pages) but prints page
# numbers from the ORIGINAL BOOK (which can run into the hundreds), so
# calling find_toc_candidates on it unpadded silently rejects nearly every
# real entry. Padding with harmless filler pages before the call -- same
# technique tests/test_segmentation.py's own _FILLER_PAGES fixture already
# uses -- raises the ratio guard's ceiling comfortably above any real
# book's page count without touching segmentation.py. The real content is
# always at the front of the padded list, well within
# find_toc_candidates' default 15% front-scan window regardless of how
# much padding is appended.
_PAGE_NUMBER_GUARD_PADDING = 1000


def _toc_entries_for_scan(pages: list[str]) -> list[TocEntry]:
    """Runs the heuristic regex extractor on a dnb-toc-only book's own
    page texts, working around _TOC_MAX_PAGE_NUMBER_RATIO's tiny-PDF
    false-rejection (see module docstring)."""
    padded = pages + ["Filler page, not part of the digitized TOC scan."] * _PAGE_NUMBER_GUARD_PADDING
    return find_toc_candidates(padded)


def _cache_path(cache_directory: Path, key: str) -> Path:
    return cache_directory / f"{key}.json"


def _load_cached_llm_entries(cache_directory: Path, key: str) -> Optional[list[TocEntry]]:
    path = _cache_path(cache_directory, key)
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


def _write_cached_llm_entries(cache_directory: Path, key: str, entries: list[TocEntry]) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_directory, key)
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


async def _run_book_pages(
    key: str, pages: list[str], llm_client: LLMClient, semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path,
) -> tuple[str, bool, str]:
    """Core per-book logic, given already-extracted page texts -- kept
    separate from PDF/file reading so it's directly unit-testable with
    synthetic pages and a fake LLMClient, no real PDF needed. Returns
    (key, passed, reason); reason is "ok" on success, else why the book
    was skipped/rejected ("needs_ocr", "no_entries", "below_threshold")."""
    if pages_need_ocr(pages):
        return key, False, "needs_ocr"
    heuristic_entries = _toc_entries_for_scan(pages)
    # Read outside the semaphore (cheap, no network); relies on the
    # caller never invoking this twice concurrently for the same `key`
    # (true today -- Task 8's driver builds one call per unique manifest
    # key) -- if that ever changes, two concurrent calls for the same
    # uncached book could both miss the cache and both call the LLM.
    cached = _load_cached_llm_entries(cache_directory, key)
    async with semaphore:
        if cached is not None:
            llm_entries = cached
        else:
            llm_entries = await _call_with_retry(lambda: llm_extract_toc_entries(pages, llm_client))
            # llm_extract_toc_entries never raises to us -- any real failure
            # (network, rate limit, timeout) is caught internally and
            # returns [], indistinguishable from a genuine "no TOC found"
            # result. Caching that [] would be permanent: a later re-run
            # would trust the stale empty cache entry forever instead of
            # retrying. Only caching a non-empty result means a future run
            # simply re-queries the LLM for this book instead.
            if llm_entries:
                _write_cached_llm_entries(cache_directory, key, llm_entries)
    if not heuristic_entries and not llm_entries:
        return key, False, "no_entries"
    passed, entries = gate_book(heuristic_entries, llm_entries, threshold=_GATE_THRESHOLD)
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
    key: str, pdf_path: Path, llm_client: LLMClient, semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path,
) -> tuple[str, bool, str]:
    """Thin I/O wrapper around _run_book_pages -- reads the real PDF,
    extracts page text the same way production does, and delegates.
    Catches any exception (a corrupt/unreadable PDF, a network error not
    already absorbed by llm_extract_toc_entries' own handling, etc.) and
    reports it as a failed-but-tuple-shaped result instead of letting it
    propagate -- same "catch-log-continue" convention
    evaluation/refresh_llm_cache.py already established for this same
    kind of long, unattended, budget-spending batch job. One book's
    failure must never abort the rest of a ~1000-book run."""
    try:
        pages, _ = extract_page_texts_for_analysis(pdf_path.read_bytes())
        return await _run_book_pages(key, pages, llm_client, semaphore, corpus_directory, cache_directory)
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}")
        return key, False, f"error: {exc}"


def _pick_model(base_url: str, api_key: str) -> str:
    models = fetch_kisski_models(base_url, api_key)
    return min(models, key=lambda m: m.demand).id


async def _run_all(
    keys_and_paths: list[tuple[str, Path]], llm_client: LLMClient, concurrency: int,
    corpus_directory: Path, cache_directory: Path,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, llm_client, semaphore, corpus_directory, cache_directory)
        for key, path in keys_and_paths
    ]))


def _generate(args: argparse.Namespace) -> int:
    cdir = corpus_dir(_CORPUS_NAME)
    eval_tier_path = cdir / "eval_tier_ids.json"
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()

    books = load_manifest_books(_CORPUS_NAME)
    eligible = [b for b in books if manifest_key(b) not in eval_tier_ids]
    candidates = [(manifest_key(b), cdir / b["filename"]) for b in eligible if (cdir / b["filename"]).exists()]
    missing_pdf_count = len(eligible) - len(candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    api_key = os.environ["KISSKI_API_KEY"]
    model = _pick_model(DEFAULT_KISSKI_BASE_URL, api_key)
    llm_client = _OpenAICompatibleLLMClient(model, DEFAULT_KISSKI_BASE_URL, api_key)

    results = asyncio.run(_run_all(candidates, llm_client, args.concurrency, cdir, llm_cache_dir(_CORPUS_NAME)))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
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
    -- then report measured precision for the >=0.90 gate threshold."""
    passing = sorted(p.name.removesuffix(".expected.json") for p in cdir.glob("*.expected.json"))
    sample = random.sample(passing, min(n, len(passing)))
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
