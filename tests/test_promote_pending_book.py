"""Unit tests for evaluation/scripts/promote_pending_book.py's
promote_book() against temp-directory fake corpora -- no real
evaluation/corpus/ data touched. The bounds/overlap gate and the
missing-manifest-entry/missing-ground-truth gates need no network; the
open-access license-resolution path is exercised with a mocked httpx
client (same convention as tests/test_discover_crossref_candidates.py)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from pypdf import PdfWriter

from evaluation.scripts.promote_pending_book import promote_book


def _write_blank_pdf(path: Path, num_pages: int) -> None:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)


def _write_manifest(path: Path, books: list[dict]) -> None:
    path.write_text(json.dumps({"books": books}), encoding="utf-8")


def _write_expected(path: Path, chapters: list[dict]) -> None:
    path.write_text(json.dumps({"chapters": chapters}), encoding="utf-8")


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


_BOOK = {
    "filename": "9781234567897.pdf",
    "title": "Test Book",
    "language": "en",
    "extraction_type": "native",
    "embedded_toc": True,
    "oa": True,
    "doi": "10.1/test",
    "download_url": "https://example.org/test.pdf",
}

_VALID_CHAPTERS = [
    {"title": "Introduction", "authors": [], "pdf_start_index": 0, "pdf_end_index": 4, "citation_pages": "1-5"},
    {"title": "Chapter One", "authors": [], "pdf_start_index": 5, "pdf_end_index": 9, "citation_pages": "6-10"},
]

_OVERLAPPING_CHAPTERS = [
    {"title": "Introduction", "authors": [], "pdf_start_index": 0, "pdf_end_index": 6, "citation_pages": "1-7"},
    {"title": "Chapter One", "authors": [], "pdf_start_index": 5, "pdf_end_index": 9, "citation_pages": "6-10"},
]


class _CorpusFixture:
    """Builds a temp pending_dir + target_dir pair, both starting with an
    empty manifest.json, seeded by with_pending_book()."""

    def __init__(self, tmp: Path):
        self.pending_dir = tmp / "pending"
        self.target_dir = tmp / "open-access"
        self.pending_dir.mkdir()
        self.target_dir.mkdir()
        _write_manifest(self.pending_dir / "manifest.json", [])
        _write_manifest(self.target_dir / "manifest.json", [])

    def with_pending_book(self, isbn: str, num_pages: int, chapters: list[dict]) -> "_CorpusFixture":
        manifest = json.loads((self.pending_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["books"].append({**_BOOK, "filename": f"{isbn}.pdf"})
        _write_manifest(self.pending_dir / "manifest.json", manifest["books"])
        _write_blank_pdf(self.pending_dir / f"{isbn}.pdf", num_pages)
        _write_expected(self.pending_dir / f"{isbn}.expected.json", chapters)
        return self


class TestPromoteBookGates(unittest.TestCase):
    def test_skips_isbn_not_in_pending_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp))
            _isbn, outcome = promote_book(
                "0000000000000", fixture.pending_dir, fixture.target_dir, "open-access", Mock(), None, dry_run=True
            )
        self.assertTrue(outcome.startswith("SKIP: not in pending/manifest.json"))

    def test_skips_when_no_expected_json_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp))
            manifest = json.loads((fixture.pending_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["books"].append({**_BOOK, "filename": "9781234567897.pdf"})
            _write_manifest(fixture.pending_dir / "manifest.json", manifest["books"])
            _write_blank_pdf(fixture.pending_dir / "9781234567897.pdf", 10)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir, "open-access", Mock(), None, dry_run=True
            )
        self.assertTrue(outcome.startswith("SKIP: no ground truth yet"))

    def test_skips_when_bounds_overlap_check_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _OVERLAPPING_CHAPTERS)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir, "open-access", Mock(), None, dry_run=True
            )
        self.assertTrue(outcome.startswith("SKIP: bounds/overlap check failed"))
        self.assertIn("overlap", outcome)


class TestPromoteBookDryRun(unittest.TestCase):
    def test_dry_run_moves_nothing_and_reports_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _VALID_CHAPTERS)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir,
                "copyrighted-scans", Mock(), None, dry_run=True,
            )
            self.assertTrue(outcome.startswith("OK (dry-run): would move to copyrighted-scans/"))
            self.assertTrue((fixture.pending_dir / "9781234567897.pdf").exists())
            self.assertTrue((fixture.pending_dir / "9781234567897.expected.json").exists())
            self.assertFalse((fixture.target_dir / "9781234567897.pdf").exists())
            target_manifest = json.loads((fixture.target_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(target_manifest["books"], [])


class TestPromoteBookRealMove(unittest.TestCase):
    def test_moves_files_and_updates_both_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _VALID_CHAPTERS)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir,
                "copyrighted-scans", Mock(), None, dry_run=False,
            )
            self.assertTrue(outcome.startswith("OK: moved to copyrighted-scans/"))
            self.assertFalse((fixture.pending_dir / "9781234567897.pdf").exists())
            self.assertFalse((fixture.pending_dir / "9781234567897.expected.json").exists())
            self.assertTrue((fixture.target_dir / "9781234567897.pdf").exists())
            self.assertTrue((fixture.target_dir / "9781234567897.expected.json").exists())

            pending_manifest = json.loads((fixture.pending_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(pending_manifest["books"], [])
            target_manifest = json.loads((fixture.target_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(target_manifest["books"]), 1)
            self.assertEqual(target_manifest["books"][0]["filename"], "9781234567897.pdf")
            self.assertNotIn("license", target_manifest["books"][0])

    def test_open_access_target_resolves_and_writes_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _VALID_CHAPTERS)
            client = Mock()
            client.get.return_value = _json_response({
                "message": {"items": [{
                    "type": "book-chapter",
                    "license": [{
                        "URL": "https://creativecommons.org/licenses/by/4.0/",
                        "content-version": "vor",
                        "delay-in-days": 0,
                    }],
                }]}
            })
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir,
                "open-access", client, None, dry_run=False,
            )
            self.assertTrue(outcome.startswith("OK: moved to open-access/"))
            target_manifest = json.loads((fixture.target_dir / "manifest.json").read_text(encoding="utf-8"))
            book = target_manifest["books"][0]
            self.assertEqual(book["license"], "https://creativecommons.org/licenses/by/4.0/")
            self.assertEqual(book["license_source"], "crossref")


if __name__ == "__main__":
    unittest.main()
