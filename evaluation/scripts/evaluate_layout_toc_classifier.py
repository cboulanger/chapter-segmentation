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


def evaluate_leave_one_book_out(rows: list[dict], books: list[dict]) -> dict:
    """Runs leave-one-book-out cross-validation, returns per-book results
    and an aggregate summary matching the design spec's decision criteria.

    The per-book "full recall" bar is asymmetric, per the design spec: a
    book passes only if the candidate set contains *every* true
    chapter_first page (missing even one loses a whole chapter) but only
    *at least one* true toc-range page (locating the section is enough --
    the point of catching a toc page at all is to find that section of
    the book, not to reproduce every one of its physical pages). `books`
    (the same list `build_feature_table` was given -- each entry has
    "key" and a "labels" list of ground-truth labels, independent of
    whatever pdfalto did or didn't manage to extract) is what lets this
    tell "this book genuinely has no pages of this label" (both labels'
    recall is vacuously satisfied) apart from "ground truth has such
    pages but every single one of them was dropped before reaching
    `rows`" (a real data-loss bug, not something to reward -- see
    build_feature_table's own skip-warning) -- relying on `rows` alone,
    post-filtering, can't distinguish the two."""
    books_by_key = {book["key"]: book for book in books}
    book_keys = sorted({row["book_key"] for row in rows})
    per_book_results = []

    for held_out in book_keys:
        train_rows = [r for r in rows if r["book_key"] != held_out]
        test_rows = [r for r in rows if r["book_key"] == held_out]
        ground_truth_labels = books_by_key[held_out]["labels"]

        X_train = [[r["features"][name] for name in FEATURE_NAMES] for r in train_rows]
        X_test = [[r["features"][name] for name in FEATURE_NAMES] for r in test_rows]

        result: dict = {"book_key": held_out, "total_pages": len(test_rows)}
        candidate_pages: set[int] = set()
        label_pass: dict[str, bool] = {}

        for label in (LABEL_TOC, LABEL_CHAPTER_FIRST):
            y_train = [r["label"] == label for r in train_rows]
            true_positive_indices = {i for i, r in enumerate(test_rows) if r["label"] == label}
            ground_truth_count = ground_truth_labels.count(label)

            predicted_indices: set[int] = set()
            if sum(y_train) > 0:
                # min_samples_leaf's default of 20 can't split at all on a
                # handful of training rows (as in this module's own unit
                # tests, or an early-development corpus) -- every
                # prediction collapses to one constant, which then flags
                # 100% of pages as candidates. 1 keeps splits available at
                # any corpus size; with thousands of real training rows
                # this trades a little overfitting resistance for that --
                # acceptable since select_threshold's recall calibration
                # (and the held-out-book generalization check that *is*
                # the leave-one-book-out loop) is what actually guards
                # against a useless model, not this hyperparameter.
                clf = HistGradientBoostingClassifier(
                    class_weight="balanced", random_state=0, min_samples_leaf=1
                )
                clf.fit(X_train, y_train)
                train_probs = [p[1] for p in clf.predict_proba(X_train)]
                threshold = select_threshold(train_probs, y_train, _RECALL_TARGET)
                test_probs = [p[1] for p in clf.predict_proba(X_test)]
                predicted_indices = {i for i, p in enumerate(test_probs) if p >= threshold}
                candidate_pages |= predicted_indices

            if ground_truth_count == 0:
                # Confirmed by ground truth, not just by an empty test
                # set: this book has no pages of this label at all.
                # Nothing to recall -- vacuously satisfied.
                result[f"{label}_recall"] = None
                label_pass[label] = True
            elif not true_positive_indices:
                # Ground truth says this book HAS ground_truth_count
                # page(s) of this label, but none of them made it into
                # `rows` -- e.g. dropped by build_feature_table's
                # pdfalto-extraction-skip path. Recall is unmeasurable,
                # and silently treating that as a vacuous pass would mask
                # real data loss, so surface it and count it as a miss.
                print(
                    f"WARNING: evaluate_leave_one_book_out: {held_out} has "
                    f"{ground_truth_count} ground-truth {label!r} page(s) but none "
                    f"survived to the feature table -- counting as a recall miss, "
                    f"not a vacuous pass",
                    file=sys.stderr,
                )
                result[f"{label}_recall"] = 0.0
                label_pass[label] = False
            else:
                hit_indices = true_positive_indices & predicted_indices
                recall = len(hit_indices) / len(true_positive_indices)
                result[f"{label}_recall"] = recall
                # chapter_first needs every page; toc only needs one hit
                # to have located the section -- see the design spec's
                # decision-criteria section.
                label_pass[label] = recall == 1.0 if label == LABEL_CHAPTER_FIRST else bool(hit_indices)

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
