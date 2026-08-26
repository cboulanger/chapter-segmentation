"""Unit tests for evaluation/scripts/layout_corpus.py's pure logic."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.scripts.layout_corpus import build_feature_table, load_book_corpus


class TestBuildFeatureTable(unittest.TestCase):
    def test_joins_features_with_labels_per_page(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "chapter_first", "other"]},
        ]

        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}, 2: {"line_count": 3.0}}

        with patch(
            "evaluation.scripts.layout_corpus.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.layout_corpus.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.layout_corpus.add_book_context_features",
            side_effect=lambda page_features, total_pages: page_features,
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
            "evaluation.scripts.layout_corpus.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.layout_corpus.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.layout_corpus.add_book_context_features",
            side_effect=lambda page_features, total_pages: page_features,
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
            "evaluation.scripts.layout_corpus.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.layout_corpus.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.layout_corpus.add_book_context_features",
            side_effect=lambda page_features, total_pages: page_features,
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
            "evaluation.scripts.layout_corpus.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.layout_corpus.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.layout_corpus.add_book_context_features",
            side_effect=lambda page_features, total_pages: page_features,
        ), contextlib.redirect_stderr(stderr):
            build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(stderr.getvalue(), "")

    def test_pipes_extracted_features_through_book_context_pass(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "other"]},
        ]
        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}}
        context_pass = Mock(side_effect=lambda page_features, total_pages: {
            index: {**page, "context_added": 1.0} for index, page in page_features.items()
        })

        with patch(
            "evaluation.scripts.layout_corpus.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.layout_corpus.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.layout_corpus.add_book_context_features",
            context_pass,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        context_pass.assert_called_once_with(fake_features, 2)
        self.assertEqual(rows[0]["features"]["context_added"], 1.0)


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
                "evaluation.scripts.layout_corpus._CORPUS_DIR", tmp_path
            ), patch(
                "evaluation.scripts.layout_corpus.PdfReader",
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
                "evaluation.scripts.layout_corpus._CORPUS_DIR", tmp_path
            ), patch(
                "evaluation.scripts.layout_corpus.PdfReader",
                return_value=fake_reader,
            ):
                books = load_book_corpus(corpora=["open-access"])

        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["key"], "book-a")
