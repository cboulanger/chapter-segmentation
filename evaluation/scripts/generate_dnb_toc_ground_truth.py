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

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from chapter_segmentation.llm import LLMClient
from chapter_segmentation.segmentation import TocEntry, find_toc_candidates, llm_extract_toc_entries, pages_need_ocr
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict

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
