#!/usr/bin/env python3
"""Pilot: leave-one-book-out evaluation of a layout-geometry TOC/
chapter-first-page classifier. See
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md.

Manual run, not part of `uv run pytest` -- same convention as
evaluation/scripts/fetch_evaluation_pdfs.py.

Usage:
    uv run python evaluation/scripts/evaluate_layout_toc_classifier.py
    uv run python evaluation/scripts/evaluate_layout_toc_classifier.py --pdfalto-bin /path/to/pdfalto
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json

from pypdf import PdfReader

from evaluation.scripts.layout_features import extract_page_features
from evaluation.scripts.layout_labels import page_labels
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
_CORPORA = ["open-access", "copyrighted-scans"]

_RECALL_TARGET = 0.90  # threshold picked per fold to hit this recall on training pages


def select_threshold(
    train_probs: list[float], train_labels: list[bool], recall_target: float
) -> float:
    """Picks the highest probability threshold that still achieves at least
    recall_target on the training positives -- a lower threshold always
    yields recall >= a higher one, so the highest satisfying threshold is
    the most precise choice that still clears the bar. Returns 1.0 (accept
    nothing) if there are no positives to calibrate against."""
    if not 0.0 < recall_target <= 1.0:
        raise ValueError(f"recall_target must be in (0.0, 1.0], got {recall_target!r}")
    if len(train_probs) != len(train_labels):
        raise ValueError(
            f"train_probs and train_labels must be the same length, "
            f"got {len(train_probs)} and {len(train_labels)}"
        )
    positive_probs = sorted(
        (p for p, is_positive in zip(train_probs, train_labels) if is_positive), reverse=True
    )
    if not positive_probs:
        return 1.0
    # ceil, not round: "at least recall_target" must never be undershot,
    # e.g. 6 positives at 0.90 -> ceil(5.4) = 6, not round(5.4) = 5.
    n_needed = max(1, math.ceil(recall_target * len(positive_probs)))
    return positive_probs[n_needed - 1]


def load_book_corpus() -> list[dict]:
    """Returns one entry per book with a usable "toc" field: {"key",
    "corpus", "pdf_path", "labels"} -- books whose .expected.json has no
    "toc" key at all are excluded entirely (not yet retrofitted, or
    flagged for manual review), per the design spec."""
    books = []
    for corpus in _CORPORA:
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
    tallied per label and, since a dropped "toc" or "chapter_first" row
    silently erodes the very recall this pilot exists to measure, printed
    as a warning once the whole table has been built."""
    rows = []
    skipped_by_label: dict[str, int] = {}
    for book in books:
        cache_dir = cache_dir_for(book["corpus"])
        alto_path = ensure_alto_xml(book["pdf_path"], cache_dir, pdfalto_bin)
        page_features = extract_page_features(str(alto_path))
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
