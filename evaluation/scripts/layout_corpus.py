"""Builds a per-page layout-feature table from the evaluation corpus --
evaluation-specific glue on top of chapter_segmentation's pdfalto/
layout_features/layout_labels primitives. Used by
evaluation/scripts/build_crossref_gt_ground_truth.py's layout-feature
novelty check."""

import json
import sys
from pathlib import Path

from pypdf import PdfReader

from chapter_segmentation.layout_features import add_book_context_features, extract_page_features
from chapter_segmentation.layout_labels import page_labels
from chapter_segmentation.pdfalto_runner import ensure_alto_xml

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
_CORPORA = ["open-access", "copyrighted-scans"]


def load_book_corpus(corpora: list[str] | None = None) -> list[dict]:
    """Returns one entry per book with a usable "toc" field: {"key",
    "corpus", "pdf_path", "labels"} -- books whose .expected.json has no
    "toc" key at all are excluded entirely (not yet retrofitted, or
    flagged for manual review). `corpora` defaults to open-access and
    copyrighted-scans (_CORPORA); pass an explicit subset (e.g.
    ["open-access"]) to restrict it."""
    books = []
    for corpus in corpora if corpora is not None else _CORPORA:
        corpus_dir = _CORPUS_DIR / corpus
        for expected_path in sorted(corpus_dir.glob("*.expected.json")):
            key = expected_path.name.removesuffix(".expected.json")
            pdf_path = corpus_dir / f"{key}.pdf"
            if not pdf_path.exists():
                continue
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            if "toc" not in expected:
                continue
            total_pages = len(PdfReader(str(pdf_path)).pages)
            labels = page_labels(expected, total_pages)
            books.append({"key": key, "corpus": corpus, "pdf_path": pdf_path, "labels": labels})
    return books


def build_feature_table(books: list[dict], cache_dir_for, pdfalto_bin: str) -> list[dict]:
    """Runs pdfalto (cached via cache_dir_for(corpus) -> Path) over every
    book and returns one row per page with an extracted feature vector:
    {"book_key", "features": {...}, "label": "toc"|"chapter_first"|"other"}.
    Pages pdfalto didn't produce a feature vector for (should not normally
    happen, but has been observed for malformed/oversized scanned pages)
    are skipped rather than crashing the whole run -- skip counts are
    tallied per label and printed as a warning once the whole table has
    been built."""
    rows = []
    skipped_by_label: dict[str, int] = {}
    for book in books:
        cache_dir = cache_dir_for(book["corpus"])
        alto_path = ensure_alto_xml(book["pdf_path"], cache_dir, pdfalto_bin)
        page_features = add_book_context_features(
            extract_page_features(str(alto_path)), len(book["labels"])
        )
        for page_index, label in enumerate(book["labels"]):
            features = page_features.get(page_index)
            if features is None:
                skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                continue
            rows.append({"book_key": book["key"], "features": features, "label": label})
    if skipped_by_label:
        total_skipped = sum(skipped_by_label.values())
        breakdown = ", ".join(f"{label}={count}" for label, count in sorted(skipped_by_label.items()))
        print(
            f"WARNING: build_feature_table skipped {total_skipped} page(s) with no "
            f"pdfalto feature vector ({breakdown})",
            file=sys.stderr,
        )
    return rows
