"""Structural sanity check for every corpus's ground truth --
evaluation/harness.py's chapter_bounds_errors() run against every
<isbn>.expected.json present, for every corpus. NOT marked integration:
the overlap/ordering checks need no PDF at all, so this runs in the
default `uv run pytest` even with zero evaluation PDFs downloaded; the
pdf_end_index<total_pages check only applies to books whose PDF happens
to be present locally too. See
docs/superpowers/specs/2026-08-12-ground-truth-pipeline-hardening-design.md."""

import json
import unittest

from pypdf import PdfReader

from evaluation.harness import chapter_bounds_errors, corpus_dir, list_corpora, load_manifest_books


class TestGroundTruthIntegrity(unittest.TestCase):
    def test_no_bounds_or_overlap_errors(self):
        for corpus in list_corpora():
            cdir = corpus_dir(corpus)
            for book in load_manifest_books(corpus):
                isbn = book["filename"].removesuffix(".pdf")
                expected_path = cdir / f"{isbn}.expected.json"
                if not expected_path.exists():
                    continue
                with self.subTest(book=f"{corpus}/{isbn}"):
                    chapters = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
                    pdf_path = cdir / book["filename"]
                    total_pages = len(PdfReader(str(pdf_path)).pages) if pdf_path.exists() else None
                    errors = chapter_bounds_errors(chapters, total_pages)
                    self.assertEqual(errors, [], f"{corpus}/{isbn}: {errors}")


if __name__ == "__main__":
    unittest.main()
