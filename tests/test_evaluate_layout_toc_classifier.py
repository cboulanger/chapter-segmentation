"""Unit tests for evaluation/scripts/evaluate_layout_toc_classifier.py's
pure logic. The real pdfalto-subprocess-driven, real-corpus leave-one-book-out
run is exercised manually -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.scripts.evaluate_layout_toc_classifier import (
    build_feature_table,
    evaluate_leave_one_book_out,
    load_book_corpus,
    main,
    select_threshold,
)
from evaluation.scripts.layout_features import FEATURE_NAMES


class TestSelectThreshold(unittest.TestCase):
    def test_no_positives_returns_one(self):
        self.assertEqual(select_threshold([0.1, 0.9], [False, False], recall_target=0.9), 1.0)

    def test_single_positive_needs_its_own_probability(self):
        probs = [0.2, 0.8, 0.5]
        labels = [False, True, False]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.9), 0.8)

    def test_target_below_full_recall_picks_higher_cutoff(self):
        # Four positives at probabilities 0.9, 0.8, 0.7, 0.2 -- targeting
        # 75% recall needs the top 3 (round(0.75*4)=3), so the cutoff is
        # the third-highest positive probability, 0.7.
        probs = [0.9, 0.8, 0.7, 0.2]
        labels = [True, True, True, True]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.75), 0.7)

    def test_rounds_up_not_to_nearest(self):
        # Six positives targeting 90% recall: 0.9*6=5.4, which must round UP
        # to 6 (all of them), not to the nearest integer (5) -- rounding to
        # nearest would only guarantee 5/6 = 83.3% recall, undershooting the
        # 90% target. The correct cutoff is therefore the lowest positive
        # probability, 0.4.
        probs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        labels = [True, True, True, True, True, True]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.90), 0.4)


class TestBuildFeatureTable(unittest.TestCase):
    def test_joins_features_with_labels_per_page(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "chapter_first", "other"]},
        ]

        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}, 2: {"line_count": 3.0}}

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {"book_key": "book-a", "features": {"line_count": 1.0}, "label": "toc"})
        self.assertEqual(
            rows[1], {"book_key": "book-a", "features": {"line_count": 2.0}, "label": "chapter_first"}
        )
        self.assertEqual(rows[2], {"book_key": "book-a", "features": {"line_count": 3.0}, "label": "other"})

    def test_skips_pages_pdfalto_did_not_extract(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "other"]},
        ]
        # pdfalto only produced a feature vector for page 0.
        fake_features = {0: {"line_count": 1.0}}

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(len(rows), 1)

    def test_warns_on_stderr_when_pages_are_skipped(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "other", "toc"]},
        ]
        # Only page 1 ("other") got a feature vector; both "toc" pages are dropped.
        fake_features = {1: {"line_count": 1.0}}

        stderr = io.StringIO()
        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ), contextlib.redirect_stderr(stderr):
            build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        warning = stderr.getvalue()
        self.assertIn("WARNING", warning)
        self.assertIn("toc=2", warning)

    def test_no_warning_when_nothing_is_skipped(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc"]},
        ]
        fake_features = {0: {"line_count": 1.0}}

        stderr = io.StringIO()
        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ), contextlib.redirect_stderr(stderr):
            build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(stderr.getvalue(), "")


class TestLoadBookCorpus(unittest.TestCase):
    def test_filters_books_by_toc_key_and_pdf_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            oa_dir = tmp_path / "open-access"
            cs_dir = tmp_path / "copyrighted-scans"
            oa_dir.mkdir()
            cs_dir.mkdir()

            # Book A: has a "toc" key and a matching PDF on disk -> included.
            (oa_dir / "book-a.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )
            (oa_dir / "book-a.pdf").write_bytes(b"%PDF-fake")

            # Book B: no "toc" key at all -> excluded (not yet retrofitted).
            (oa_dir / "book-b.expected.json").write_text(
                json.dumps({"chapters": []}), encoding="utf-8"
            )
            (oa_dir / "book-b.pdf").write_bytes(b"%PDF-fake")

            # Book C: has a "toc" key but no matching PDF on disk -> excluded.
            (cs_dir / "book-c.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )

            fake_reader = Mock()
            fake_reader.pages = [Mock()] * 5

            with patch(
                "evaluation.scripts.evaluate_layout_toc_classifier._CORPUS_DIR", tmp_path
            ), patch(
                "evaluation.scripts.evaluate_layout_toc_classifier.PdfReader",
                return_value=fake_reader,
            ):
                books = load_book_corpus()

        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["key"], "book-a")
        self.assertEqual(books[0]["corpus"], "open-access")
        self.assertEqual(books[0]["pdf_path"], oa_dir / "book-a.pdf")
        self.assertEqual(len(books[0]["labels"]), 5)

    def test_corpora_param_restricts_which_directories_are_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            oa_dir = tmp_path / "open-access"
            cs_dir = tmp_path / "copyrighted-scans"
            oa_dir.mkdir()
            cs_dir.mkdir()

            (oa_dir / "book-a.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )
            (oa_dir / "book-a.pdf").write_bytes(b"%PDF-fake")

            (cs_dir / "book-c.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )
            (cs_dir / "book-c.pdf").write_bytes(b"%PDF-fake")

            fake_reader = Mock()
            fake_reader.pages = [Mock()] * 5

            with patch(
                "evaluation.scripts.evaluate_layout_toc_classifier._CORPUS_DIR", tmp_path
            ), patch(
                "evaluation.scripts.evaluate_layout_toc_classifier.PdfReader",
                return_value=fake_reader,
            ):
                books = load_book_corpus(corpora=["open-access"])

        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["key"], "book-a")


def _feature_row(book_key: str, label: str, value: float) -> dict:
    return {"book_key": book_key, "features": {name: value for name in FEATURE_NAMES}, "label": label}


def _clean_book(key: str) -> tuple[list[dict], dict]:
    """A background/training book with one obviously-separable page of
    each kind: toc pages score 5.0, chapter_first pages score -5.0, other
    pages score 0.0. Used as the well-behaved training population that
    the scenario-under-test's held-out book gets evaluated against."""
    rows = [
        _feature_row(key, "toc", 5.0),
        _feature_row(key, "chapter_first", -5.0),
    ] + [_feature_row(key, "other", 0.0) for _ in range(3)]
    labels = ["toc", "chapter_first", "other", "other", "other"]
    return rows, {"key": key, "labels": labels}


class TestEvaluateLeaveOneBookOut(unittest.TestCase):
    def test_perfectly_separable_data_gets_full_recall(self):
        # Three synthetic books, each with 5 pages: page 0 is "toc"
        # (features all 5.0), page 1 is "chapter_first" (features all
        # -5.0), pages 2-4 are "other" (features all 0.0) -- identical,
        # trivially separable pattern across every book.
        rows = []
        books = []
        for book_key in ("book-a", "book-b", "book-c"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)

        summary = evaluate_leave_one_book_out(rows, books)

        self.assertEqual(summary["full_recall_fraction"], 1.0)
        self.assertLessEqual(summary["avg_candidate_fraction"], 0.45)
        self.assertEqual(len(summary["per_book"]), 3)

    def test_catching_only_some_of_a_multi_page_toc_range_still_counts_as_full_recall(self):
        # The design spec's bar is asymmetric: chapter_first needs every
        # page caught, toc only needs one. Held-out book "partial" has a
        # 4-page toc range where 3 pages score strongly (5.0, same as
        # what the two clean training books' single toc page look like)
        # and one page scores weakly (0.5, much closer to "other"'s 0.0)
        # -- calibrated (empirically, not just in theory) to reliably get
        # caught on the 3 strong pages and missed on the weak one, while
        # its single chapter_first page (a strong -5.0, matching the
        # clean books' pattern) is still fully recalled.
        rows = []
        books = []
        for book_key in ("clean-a", "clean-b"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)

        rows.extend(
            [
                _feature_row("partial", "toc", 5.0),
                _feature_row("partial", "toc", 5.0),
                _feature_row("partial", "toc", 5.0),
                _feature_row("partial", "toc", 0.5),
                _feature_row("partial", "chapter_first", -5.0),
                _feature_row("partial", "other", 0.0),
                _feature_row("partial", "other", 0.0),
            ]
        )
        books.append(
            {
                "key": "partial",
                "labels": ["toc", "toc", "toc", "toc", "chapter_first", "other", "other"],
            }
        )

        summary = evaluate_leave_one_book_out(rows, books)

        partial_result = next(r for r in summary["per_book"] if r["book_key"] == "partial")
        self.assertLess(partial_result["toc_recall"], 1.0)
        self.assertGreater(partial_result["toc_recall"], 0.0)
        self.assertEqual(partial_result["chapter_first_recall"], 1.0)
        self.assertTrue(partial_result["full_recall"])

    def test_missing_every_toc_page_does_not_count_as_full_recall(self):
        # Held-out book "missall"'s toc pages score -2.0 -- nowhere near
        # the clean training books' toc pattern (5.0) -- so the
        # classifier should catch none of them, even though its
        # chapter_first page (a strong -5.0) is still fully recalled.
        # Catching zero toc pages must NOT satisfy the "at least one" bar.
        rows = []
        books = []
        for book_key in ("clean-a", "clean-b"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)

        rows.extend(
            [
                _feature_row("missall", "toc", -2.0),
                _feature_row("missall", "toc", -2.0),
                _feature_row("missall", "chapter_first", -5.0),
                _feature_row("missall", "other", 0.0),
                _feature_row("missall", "other", 0.0),
            ]
        )
        books.append({"key": "missall", "labels": ["toc", "toc", "chapter_first", "other", "other"]})

        summary = evaluate_leave_one_book_out(rows, books)

        missall_result = next(r for r in summary["per_book"] if r["book_key"] == "missall")
        self.assertEqual(missall_result["toc_recall"], 0.0)
        self.assertEqual(missall_result["chapter_first_recall"], 1.0)
        self.assertFalse(missall_result["full_recall"])

    def test_missing_any_chapter_first_page_does_not_count_as_full_recall(self):
        # Held-out book "misschap"'s chapter_first page scores 0.4 --
        # nowhere near the clean training books' chapter_first pattern
        # (-5.0) -- so the classifier should miss it, even though its
        # single toc page (a strong 5.0) is fully recalled. Unlike toc,
        # chapter_first has no "at least one is enough" exception, so
        # this must NOT count as full recall.
        rows = []
        books = []
        for book_key in ("clean-a", "clean-b"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)

        rows.extend(
            [
                _feature_row("misschap", "toc", 5.0),
                _feature_row("misschap", "chapter_first", 0.4),
                _feature_row("misschap", "other", 0.0),
                _feature_row("misschap", "other", 0.0),
            ]
        )
        books.append({"key": "misschap", "labels": ["toc", "chapter_first", "other", "other"]})

        summary = evaluate_leave_one_book_out(rows, books)

        misschap_result = next(r for r in summary["per_book"] if r["book_key"] == "misschap")
        self.assertEqual(misschap_result["toc_recall"], 1.0)
        self.assertEqual(misschap_result["chapter_first_recall"], 0.0)
        self.assertFalse(misschap_result["full_recall"])

    def test_toc_pages_dropped_before_the_feature_table_are_a_miss_not_a_vacuous_pass(self):
        # Held-out book "dropped" has 2 toc pages in its ground-truth
        # labels, but neither made it into `rows` -- simulating
        # build_feature_table's pdfalto-extraction-skip path silently
        # dropping every toc-labeled page for a book. This must be
        # distinguished from a book that genuinely has no toc pages: it
        # should count as a recall miss (not a vacuous pass) and should
        # print a warning rather than failing silently.
        rows = []
        books = []
        for book_key in ("clean-a", "clean-b"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)

        rows.extend(
            [
                _feature_row("dropped", "chapter_first", -5.0),
                _feature_row("dropped", "other", 0.0),
                _feature_row("dropped", "other", 0.0),
            ]
        )
        books.append({"key": "dropped", "labels": ["toc", "toc", "chapter_first", "other", "other"]})

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            summary = evaluate_leave_one_book_out(rows, books)

        dropped_result = next(r for r in summary["per_book"] if r["book_key"] == "dropped")
        self.assertEqual(dropped_result["toc_recall"], 0.0)
        self.assertEqual(dropped_result["chapter_first_recall"], 1.0)
        self.assertFalse(dropped_result["full_recall"])
        self.assertIn("dropped", stderr.getvalue())
        self.assertIn("toc", stderr.getvalue())

    def test_partially_dropped_chapter_first_pages_lower_recall_not_hide_it(self):
        # Held-out book "partial_chapter" has 3 ground-truth chapter_first
        # pages, but only 2 of them made it into `rows` -- e.g. one
        # chapter-opening page got dropped by build_feature_table's
        # pdfalto-extraction-skip path while the other two survived. The
        # classifier catches both surviving pages perfectly. Scoring
        # recall against the survived count (2) rather than the true
        # ground-truth count (3) would wrongly report 2/2 == 1.0 full
        # recall; it must instead be 2/3, which fails the strict
        # chapter_first bar (unlike toc, chapter_first has no "at least
        # one is enough" exception) and prints a warning about the loss.
        rows = []
        books = []
        for book_key in ("clean-a", "clean-b"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)

        rows.extend(
            [
                _feature_row("partial_chapter", "toc", 5.0),
                _feature_row("partial_chapter", "chapter_first", -5.0),
                _feature_row("partial_chapter", "chapter_first", -5.0),
                _feature_row("partial_chapter", "other", 0.0),
                _feature_row("partial_chapter", "other", 0.0),
            ]
        )
        books.append(
            {
                "key": "partial_chapter",
                "labels": ["toc", "chapter_first", "chapter_first", "chapter_first", "other", "other"],
            }
        )

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            summary = evaluate_leave_one_book_out(rows, books)

        partial_chapter_result = next(
            r for r in summary["per_book"] if r["book_key"] == "partial_chapter"
        )
        self.assertAlmostEqual(partial_chapter_result["chapter_first_recall"], 2 / 3)
        self.assertFalse(partial_chapter_result["full_recall"])
        self.assertIn("partial_chapter", stderr.getvalue())
        self.assertIn("chapter_first", stderr.getvalue())

    def test_book_with_no_toc_at_all_is_a_vacuous_pass(self):
        # Held-out book "notoc" genuinely has no toc pages at all
        # ("toc": null in the source .expected.json, i.e. ground truth
        # confirms there's nothing to recall) -- this is the legitimate
        # vacuous-pass case and must NOT print a warning or count as a
        # miss, unlike the "dropped" scenario above.
        rows = []
        books = []
        for book_key in ("clean-a", "clean-b"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)

        rows.extend(
            [
                _feature_row("notoc", "chapter_first", -5.0),
                _feature_row("notoc", "other", 0.0),
                _feature_row("notoc", "other", 0.0),
            ]
        )
        books.append({"key": "notoc", "labels": ["chapter_first", "other", "other"]})

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            summary = evaluate_leave_one_book_out(rows, books)

        notoc_result = next(r for r in summary["per_book"] if r["book_key"] == "notoc")
        self.assertIsNone(notoc_result["toc_recall"])
        self.assertEqual(notoc_result["chapter_first_recall"], 1.0)
        self.assertTrue(notoc_result["full_recall"])
        self.assertEqual(stderr.getvalue(), "")


class TestMainCorporaValidation(unittest.TestCase):
    def test_unknown_corpus_name_errors_clearly_instead_of_silent_fallthrough(self):
        # A typo'd --corpora value must fail loudly, before ever reaching
        # load_book_corpus -- not silently glob zero books and fall through
        # to the unrelated "No books with a 'toc' field found" message.
        stderr = io.StringIO()
        with patch(
            "sys.argv",
            ["evaluate_layout_toc_classifier.py", "--corpora", "nonexistent-corpus"],
        ), contextlib.redirect_stderr(stderr):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        message = stderr.getvalue()
        self.assertIn("Unknown corpus", message)
        self.assertIn("nonexistent-corpus", message)
        self.assertNotIn("No books with a 'toc' field found", message)


if __name__ == "__main__":
    unittest.main()
