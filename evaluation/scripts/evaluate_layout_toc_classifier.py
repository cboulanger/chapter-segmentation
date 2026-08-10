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
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluation.scripts.layout_features import FEATURE_NAMES, extract_page_features
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST, LABEL_TOC, page_labels
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


def _evaluate_label(
    label: str,
    train_rows: list[dict],
    test_rows: list[dict],
    X_train: list[list[float]],
    X_test: list[list[float]],
    ground_truth_count: int,
    held_out: str,
) -> tuple[float | None, bool, set[int]]:
    """Trains a one-vs-rest classifier for `label` on `train_rows` (skipped
    if there are no positive training examples), scores `held_out`'s
    X_test pages against a recall-target-calibrated threshold, and returns
    (recall, passed, predicted_indices).

    recall is computed against `ground_truth_count` -- the book's TRUE
    ground-truth page count for this label, taken from its
    .expected.json-derived labels list -- never against however many of
    those pages happen to have survived into `test_rows`.
    build_feature_table's own pdfalto-extraction-skip path can silently
    drop some or all of a book's pages for a label; scoring recall against
    that reduced count instead of the true one would let a dropped page
    inflate recall right past the exact failure this metric exists to
    catch. A stderr warning fires whenever `test_rows` undercounts
    `ground_truth_count` for this book/label -- whether partially (some
    pages dropped) or completely (all of them).

    Returns (None, True, predicted_indices) -- a vacuous pass, nothing to
    recall -- when `ground_truth_count` is 0: the book genuinely has no
    ground-truth pages of this label at all, confirmed by ground truth
    rather than merely inferred from an empty test set.

    `passed` applies the design spec's asymmetric per-label bar: exactly
    1.0 recall for chapter_first (every page must be caught), but merely
    catching at least one true positive for every other label (e.g. toc,
    where locating the section is enough)."""
    y_train = [r["label"] == label for r in train_rows]
    true_positive_indices = {i for i, r in enumerate(test_rows) if r["label"] == label}

    predicted_indices: set[int] = set()
    if sum(y_train) > 0:
        # min_samples_leaf's default of 20 can't split at all on a handful
        # of training rows (as in this module's own unit tests, or an
        # early-development corpus) -- every prediction collapses to one
        # constant, which then flags 100% of pages as candidates. 1 keeps
        # splits available at any corpus size; with thousands of real
        # training rows this trades a little overfitting resistance for
        # that -- acceptable since select_threshold's recall calibration
        # (and the held-out-book generalization check that *is* the
        # leave-one-book-out loop) is what actually guards against a
        # useless model, not this hyperparameter.
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0, min_samples_leaf=1)
        clf.fit(X_train, y_train)
        train_probs = [p[1] for p in clf.predict_proba(X_train)]
        threshold = select_threshold(train_probs, y_train, _RECALL_TARGET)
        test_probs = [p[1] for p in clf.predict_proba(X_test)]
        predicted_indices = {i for i, p in enumerate(test_probs) if p >= threshold}

    if ground_truth_count == 0:
        return None, True, predicted_indices

    if len(true_positive_indices) < ground_truth_count:
        lost = ground_truth_count - len(true_positive_indices)
        print(
            f"WARNING: evaluate_leave_one_book_out: {held_out} lost {lost} of "
            f"{ground_truth_count} ground-truth {label!r} page(s) upstream -- "
            f"recall measured against the true ground-truth count, not the reduced "
            f"set that survived",
            file=sys.stderr,
        )

    hit_indices = true_positive_indices & predicted_indices
    recall = len(hit_indices) / ground_truth_count
    passed = recall == 1.0 if label == LABEL_CHAPTER_FIRST else bool(hit_indices)
    return recall, passed, predicted_indices


def evaluate_leave_one_book_out(rows: list[dict], books: list[dict]) -> dict:
    """Runs leave-one-book-out cross-validation, returns per-book results
    and an aggregate summary matching the design spec's decision criteria.

    `books` (the same list `build_feature_table` was given -- each entry
    has "key" and a "labels" list of ground-truth labels, independent of
    whatever pdfalto did or didn't manage to extract) is what lets
    `_evaluate_label` tell a book with genuinely no pages of a label apart
    from one whose pages were dropped upstream -- see its docstring."""
    books_by_key = {book["key"]: book for book in books}
    book_keys = sorted({row["book_key"] for row in rows})
    per_book_results = []

    for held_out in book_keys:
        assert held_out in books_by_key, (
            f"evaluate_leave_one_book_out: {held_out!r} appears in rows but has no "
            f"matching entry in books -- rows and books must reference the same "
            f"book_key set"
        )
        train_rows = [r for r in rows if r["book_key"] != held_out]
        test_rows = [r for r in rows if r["book_key"] == held_out]
        ground_truth_labels = books_by_key[held_out]["labels"]

        X_train = [[r["features"][name] for name in FEATURE_NAMES] for r in train_rows]
        X_test = [[r["features"][name] for name in FEATURE_NAMES] for r in test_rows]

        result: dict = {"book_key": held_out, "total_pages": len(test_rows)}
        candidate_pages: set[int] = set()
        label_pass: dict[str, bool] = {}

        for label in (LABEL_TOC, LABEL_CHAPTER_FIRST):
            ground_truth_count = ground_truth_labels.count(label)
            recall, passed, predicted_indices = _evaluate_label(
                label, train_rows, test_rows, X_train, X_test, ground_truth_count, held_out
            )
            result[f"{label}_recall"] = recall
            label_pass[label] = passed
            candidate_pages |= predicted_indices

        result["candidate_fraction"] = len(candidate_pages) / result["total_pages"]
        result["full_recall"] = label_pass[LABEL_TOC] and label_pass[LABEL_CHAPTER_FIRST]
        per_book_results.append(result)

    n_books = len(per_book_results)
    n_full_recall = sum(1 for r in per_book_results if r["full_recall"])
    avg_candidate_fraction = sum(r["candidate_fraction"] for r in per_book_results) / n_books

    return {
        "per_book": per_book_results,
        "full_recall_fraction": n_full_recall / n_books,
        "avg_candidate_fraction": avg_candidate_fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdfalto-bin", default=None)
    args = parser.parse_args()
    pdfalto_bin = resolve_pdfalto_binary(args.pdfalto_bin)

    books = load_book_corpus()
    if not books:
        print("No books with a 'toc' field found -- run add_toc_ground_truth.py first.")
        return 1

    def cache_dir_for(corpus: str) -> Path:
        return _CORPUS_DIR / corpus / ".layout-cache"

    rows = build_feature_table(books, cache_dir_for, pdfalto_bin)
    summary = evaluate_leave_one_book_out(rows, books)

    print(f"Books evaluated: {len(books)}")
    print(
        f"Books with full recall (>=1 toc page + all chapter-first pages retained): "
        f"{summary['full_recall_fraction']:.0%}"
    )
    print(f"Average candidate-page fraction: {summary['avg_candidate_fraction']:.1%}")
    print()

    def fmt_recall(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    for r in summary["per_book"]:
        print(
            f"  {r['book_key']}: toc_recall={fmt_recall(r.get('toc_recall'))}, "
            f"chapter_first_recall={fmt_recall(r.get('chapter_first_recall'))}, "
            f"candidate_fraction={r['candidate_fraction']:.1%}"
        )

    meets_bar = summary["full_recall_fraction"] >= 0.90 and summary["avg_candidate_fraction"] <= 0.15
    print(
        f"\nDecision bar (>=90% full recall, <=15% avg candidate fraction): "
        f"{'MET' if meets_bar else 'NOT MET'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
