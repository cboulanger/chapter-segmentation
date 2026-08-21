"""Shared loading helpers for the chapter-segmentation evaluation set.

Single home for the manifest-merging, PDF-availability, and page-loading
logic that tests/test_segmentation_accuracy.py and the
evaluation/scripts/evaluate_chapter_segmentation_*.py scripts previously each
carried their own copy of. Lives under evaluation/ (not tests/) because
evaluation/scripts/ must not depend on the test tree.

Every evaluation book lives under evaluation/corpus/<corpus>/ -- a
self-contained subfolder (manifest.json, optional manifest.local.json,
PDFs, ground truth, and caches) per corpus -- see
docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md.
list_corpora() is the single source of truth for "what corpora exist";
every function below takes a corpus name so callers can target one corpus
or loop list_corpora() to cover all of them.

Page loading mirrors production's chapter_segmentation.run(): default
extraction with the layout-mode fallback, then -- for books whose text
layer is absent or degenerate (pages_need_ocr) -- the content-hash-keyed
OCR cache populated by evaluation/scripts/ocr_evaluation_pdfs.py. A book
whose OCR cache entry is missing loads as None and should be skipped by the
caller with a pointer to that script.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from chapter_segmentation.evidence.types import ChapterCandidate
from chapter_segmentation.ocr import load_cached_ocr
from chapter_segmentation.segmentation import (
    extract_page_texts_for_analysis,
    pages_need_ocr,
)

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = EVAL_DIR / "corpus"

DNB_TOC_CORPUS_NAME = "dnb-toc-only"
_DNB_TOC_CORPUS_DIR_ENV_VAR = "DNB_TOC_CORPUS_DIR"
_DNB_TOC_CORPUS_DIR_DEFAULT = EVAL_DIR.parent.parent / "dnb-toc-ground-truth" / "data" / "corpus" / "pilot"


def list_corpora(include_toc_only: bool = False) -> list[str]:
    """Sorted names of every subfolder under evaluation/corpus/ that has a
    manifest.json -- the single source of truth for "what corpora exist"
    that every runner iterates over. A corpus whose manifest.json sets
    "toc_only": true (see dnb-toc-only/) is excluded unless
    include_toc_only=True: every existing caller assumes a corpus has
    fetchable PDFs and full .expected.json chapter fields, neither of
    which a toc-only corpus has.

    dnb-toc-only itself moved to the standalone dnb-toc-ground-truth repo
    (2026-08-21 migration; see corpus_dir()'s redirect) and its local
    evaluation/corpus/dnb-toc-only/ folder is slated for deletion, so once
    that happens the local scan above will no longer find it -- the check
    below falls back to the redirected location so callers keep seeing it
    listed as long as include_toc_only=True."""
    if not CORPUS_ROOT.is_dir():
        return []
    names = []
    for p in CORPUS_ROOT.iterdir():
        manifest_path = p / "manifest.json"
        if not (p.is_dir() and manifest_path.exists()):
            continue
        if not include_toc_only:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("toc_only"):
                continue
        names.append(p.name)
    if include_toc_only and DNB_TOC_CORPUS_NAME not in names:
        if (corpus_dir(DNB_TOC_CORPUS_NAME) / "manifest.json").exists():
            names.append(DNB_TOC_CORPUS_NAME)
    return sorted(names)


def corpus_dir(corpus: str) -> Path:
    # dnb-toc-only moved to the standalone dnb-toc-ground-truth repo
    # (2026-08-21 migration) -- everything under evaluation/corpus/ still
    # resolves normally; only this one corpus name is redirected to the
    # sibling checkout's data/corpus/pilot/, via DNB_TOC_CORPUS_DIR (same
    # override convention as PDFALTO_BIN) or the default sibling path.
    if corpus == DNB_TOC_CORPUS_NAME:
        override = os.environ.get(_DNB_TOC_CORPUS_DIR_ENV_VAR)
        return Path(override) if override else _DNB_TOC_CORPUS_DIR_DEFAULT
    return CORPUS_ROOT / corpus


def public_cache_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / "public-cache"


def ocr_cache_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / ".ocr-cache"


def llm_cache_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / "llm-cache"


def load_manifest_books(corpus: str) -> list[dict]:
    """Merge a corpus's committed manifest.json with its gitignored,
    optional manifest.local.json (see evaluation/CLAUDE.md) -- the latter
    holds books that have no DOI or otherwise can't be shared, still
    exercised in local runs on the machine that added them."""
    cdir = corpus_dir(corpus)
    books = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))["books"]
    local_manifest_path = cdir / "manifest.local.json"
    if local_manifest_path.exists():
        books = books + json.loads(local_manifest_path.read_text(encoding="utf-8"))["books"]
    return books


def available_books(corpus: str) -> list[tuple[Path, Path, dict]]:
    """(pdf_path, expected_json_path, manifest_entry) for every book in
    this corpus whose PDF and ground truth are both present locally right
    now."""
    cdir = corpus_dir(corpus)
    triples = []
    for book in load_manifest_books(corpus):
        pdf_path = cdir / book["filename"]
        expected_path = cdir / (Path(book["filename"]).stem + ".expected.json")
        if pdf_path.exists() and expected_path.exists():
            triples.append((pdf_path, expected_path, book))
    return triples


def available_public_books(corpus: str) -> list[tuple[str, Path, dict]]:
    """(manifest_key, expected_json_path, manifest_entry) for every book in
    this corpus with a public-cache entry -- no PDF or .ocr-cache
    required."""
    cdir = corpus_dir(corpus)
    cache_dir = public_cache_dir(corpus)
    triples = []
    for book in load_manifest_books(corpus):
        manifest_key = Path(book["filename"]).stem
        expected_path = cdir / f"{manifest_key}.expected.json"
        cache_path = cache_dir / f"{manifest_key}.pages.json"
        if cache_path.exists() and expected_path.exists():
            triples.append((manifest_key, expected_path, book))
    return triples


def public_pages_for(corpus: str, manifest_key: str) -> Optional[list[str]]:
    """Redacted pages for one book from this corpus's committed
    public-cache, or None if no entry exists yet for this key."""
    cache_path = public_cache_dir(corpus) / f"{manifest_key}.pages.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))["pages"]


def analysis_pages_for(corpus: str, file_bytes: bytes) -> Optional[list[str]]:
    """Page texts for this PDF the same way production run() would see
    them, or None when the book needs OCR and this corpus's eval OCR cache
    has no usable entry yet (run evaluation/scripts/ocr_evaluation_pdfs.py
    to populate it)."""
    pages, _layout_used = extract_page_texts_for_analysis(file_bytes)
    if not pages_need_ocr(pages):
        return pages
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    cached = load_cached_ocr(ocr_cache_dir(corpus), content_hash)
    if cached is not None and not pages_need_ocr(cached["pages"]):
        return cached["pages"]
    return None


def outline_candidate_to_dict(candidate: ChapterCandidate) -> dict:
    return {
        "title": candidate.title,
        "authors": list(candidate.authors),
        "printed_page_number": candidate.printed_page_number,
        "pdf_page_index": candidate.pdf_page_index,
        "chapter_doi": candidate.chapter_doi,
        "source": candidate.source,
        "metadata_confidence": candidate.metadata_confidence,
    }


def outline_candidate_from_dict(data: dict) -> ChapterCandidate:
    return ChapterCandidate(
        title=data["title"],
        authors=tuple(data.get("authors", [])),
        printed_page_number=data.get("printed_page_number"),
        pdf_page_index=data.get("pdf_page_index"),
        chapter_doi=data.get("chapter_doi"),
        source=data.get("source", "outline"),
        metadata_confidence=data.get("metadata_confidence", 1.0),
    )


def public_outline_candidates_for(corpus: str, manifest_key: str) -> Optional[list[ChapterCandidate]]:
    """Cached outline-strategy candidates for one book from this corpus's
    committed public-cache, or None if no entry exists yet (either the
    book's PDF has no outline, or the cache hasn't been generated for it
    yet -- see evaluation/scripts/generate_public_evaluation_cache.py)."""
    cache_path = public_cache_dir(corpus) / f"{manifest_key}.outline.json"
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [outline_candidate_from_dict(c) for c in data["candidates"]]


def chapter_bounds_errors(chapters: list[dict], total_pages: Optional[int] = None) -> list[str]:
    """Structural sanity check on one book's ground-truth chapter ranges:
    every pdf_start_index <= pdf_end_index, no two chapters' ranges
    overlap, and -- only when total_pages is given -- every
    pdf_end_index < total_pages. Returns every problem found (empty list
    if none), not just the first -- needs no PDF unless total_pages is
    passed, so it can run against every corpus even before any PDF has
    been fetched locally."""
    ranges = sorted((c["pdf_start_index"], c["pdf_end_index"]) for c in chapters)
    errors = []
    for start, end in ranges:
        if start > end:
            errors.append(f"start>end: {(start, end)}")
        if total_pages is not None and end >= total_pages:
            errors.append(f"end>=total_pages({total_pages}): {(start, end)}")
    for (_, end1), (start2, _) in zip(ranges, ranges[1:]):
        if start2 <= end1:
            errors.append(f"overlap: end {end1} vs next start {start2}")
    return errors
