"""Unit tests for evaluation/scripts/fetch_dnb_toc_corpus.py's pure logic
(record filtering, field extraction, streaming JSON-Lines decode) against
mocked httpx responses and an in-memory gzip stream -- no live network,
matching tests/test_discover_crossref_candidates.py's convention. The real
network-calling main()/_run_isbns_file()/_run_from_dump() orchestration is
exercised manually (see docs/superpowers/plans/2026-08-14-dnb-toc-corpus-acquisition-plan.md
Task 4's smoke test), matching fetch_crossref_gt_corpus.py's existing
convention of no pytest coverage for its own network-calling entry point."""

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from evaluation.scripts.fetch_dnb_toc_corpus import (
    _ChunkStreamReader,
    _acquire_record,
    _append_book,
    _ensure_manifest_shell,
    _iter_dump_records_from_chunks,
    _load_existing_keys,
    _read_isbns_file,
    _record_key,
    _record_language,
    _record_matches,
    _search_by_isbn,
    _toc_download_url,
    manifest_entry_from_record,
)

# Trimmed to the fields this script actually reads, sourced from a real
# lobid-resources record (isbn:9783899718188, confirmed live 2026-08-14 --
# see the plan's header).
_SAMPLE_RECORD = {
    "id": "http://lobid.org/resources/990183806670206441#!",
    "type": ["BibliographicResource", "EditedVolume", "Book"],
    "title": "Systemtheorie in den Fachwissenschaften",
    "isbn": ["9783899718188", "3899718186"],
    "language": [{"id": "http://id.loc.gov/vocabulary/iso639-2/ger", "label": "Deutsch"}],
    "tableOfContents": [
        {
            "label": "Inhaltsverzeichnis",
            "id": "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf",
        }
    ],
}


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


class TestRecordMatches(unittest.TestCase):
    def test_matches_book_with_toc(self):
        self.assertTrue(_record_matches(_SAMPLE_RECORD))

    def test_rejects_wrong_type(self):
        record = {**_SAMPLE_RECORD, "type": ["BibliographicResource", "Article"]}
        self.assertFalse(_record_matches(record))

    def test_rejects_missing_toc(self):
        record = {**_SAMPLE_RECORD, "tableOfContents": []}
        self.assertFalse(_record_matches(record))

    def test_rejects_absent_toc_key(self):
        record = {k: v for k, v in _SAMPLE_RECORD.items() if k != "tableOfContents"}
        self.assertFalse(_record_matches(record))


class TestTocDownloadUrl(unittest.TestCase):
    def test_returns_first_entry_id(self):
        self.assertEqual(
            _toc_download_url(_SAMPLE_RECORD),
            "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf",
        )

    def test_returns_none_when_absent(self):
        self.assertIsNone(_toc_download_url({}))


class TestRecordKey(unittest.TestCase):
    def test_prefers_isbn(self):
        self.assertEqual(_record_key(_SAMPLE_RECORD), "9783899718188")

    def test_falls_back_to_record_id(self):
        record = {k: v for k, v in _SAMPLE_RECORD.items() if k != "isbn"}
        self.assertEqual(_record_key(record), "990183806670206441")


class TestRecordLanguage(unittest.TestCase):
    def test_maps_iso639_2_to_iso639_1(self):
        self.assertEqual(_record_language(_SAMPLE_RECORD), "de")

    def test_falls_back_to_raw_code_when_unmapped(self):
        record = {**_SAMPLE_RECORD, "language": [{"id": ".../vocabulary/iso639-2/wen"}]}
        self.assertEqual(_record_language(record), "wen")

    def test_none_when_absent(self):
        self.assertIsNone(_record_language({}))


class TestManifestEntryFromRecord(unittest.TestCase):
    def test_builds_expected_shape(self):
        entry = manifest_entry_from_record(_SAMPLE_RECORD, "9783899718188.pdf")
        self.assertEqual(entry["filename"], "9783899718188.pdf")
        self.assertEqual(entry["title"], "Systemtheorie in den Fachwissenschaften")
        self.assertEqual(entry["language"], "de")
        self.assertIsNone(entry["doi"])
        self.assertEqual(
            entry["toc_download_url"],
            "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf",
        )
        self.assertEqual(entry["license"], "CC0-1.0")
        self.assertEqual(entry["license_source"], "dnb")
        self.assertEqual(entry["lobid_record"], _SAMPLE_RECORD)


class TestSearchByIsbn(unittest.TestCase):
    def test_returns_first_member(self):
        client = Mock()
        client.get.return_value = _json_response({"member": [_SAMPLE_RECORD]})
        self.assertEqual(_search_by_isbn("9783899718188", client), _SAMPLE_RECORD)

    def test_returns_none_when_no_member(self):
        client = Mock()
        client.get.return_value = _json_response({"member": []})
        self.assertIsNone(_search_by_isbn("0000000000000", client))


class TestAcquireRecord(unittest.TestCase):
    """rate_limit_seconds=0 everywhere below to avoid a real time.sleep()
    per test -- simplest option since it's already a parameter, per the
    fix task's own guidance."""

    def _setup(self, tmp):
        cdir = Path(tmp)
        manifest_path = cdir / "manifest.json"
        _ensure_manifest_shell(manifest_path)
        return cdir, manifest_path

    def test_acquires_new_matching_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir, manifest_path = self._setup(tmp)
            client = Mock()
            response = Mock()
            response.raise_for_status = Mock()
            response.content = b"%PDF-1.4 fake toc bytes"
            client.get.return_value = response
            seen_keys = set()

            result = _acquire_record(_SAMPLE_RECORD, cdir, manifest_path, client, 0, seen_keys)

            self.assertIsNone(result)
            pdf_path = cdir / "9783899718188.pdf"
            self.assertTrue(pdf_path.exists())
            self.assertEqual(pdf_path.read_bytes(), b"%PDF-1.4 fake toc bytes")
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["books"]), 1)
            self.assertEqual(data["books"][0]["filename"], "9783899718188.pdf")
            self.assertIn("9783899718188", seen_keys)

    def test_skips_already_acquired_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir, manifest_path = self._setup(tmp)
            client = Mock()
            seen_keys = {"9783899718188"}

            result = _acquire_record(_SAMPLE_RECORD, cdir, manifest_path, client, 0, seen_keys)

            self.assertIsNotNone(result)
            client.get.assert_not_called()
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"], [])

    def test_skips_non_matching_record_without_any_network_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir, manifest_path = self._setup(tmp)
            client = Mock()
            record = {**_SAMPLE_RECORD, "type": ["BibliographicResource", "Article"]}
            seen_keys = set()

            result = _acquire_record(record, cdir, manifest_path, client, 0, seen_keys)

            self.assertIsNotNone(result)
            client.get.assert_not_called()
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"], [])
            self.assertEqual(seen_keys, set())

    def test_download_http_error_is_caught_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir, manifest_path = self._setup(tmp)
            client = Mock()
            response = Mock()
            response.raise_for_status = Mock(side_effect=httpx.HTTPError("boom"))
            client.get.return_value = response
            seen_keys = set()

            result = _acquire_record(_SAMPLE_RECORD, cdir, manifest_path, client, 0, seen_keys)

            self.assertIsNotNone(result)
            self.assertIn("boom", result)
            self.assertFalse((cdir / "9783899718188.pdf").exists())
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"], [])
            self.assertEqual(seen_keys, set())


class TestChunkStreamReader(unittest.TestCase):
    def test_read_reassembles_chunks_of_any_requested_size(self):
        reader = _ChunkStreamReader(iter([b"ab", b"cde", b"f"]))
        self.assertEqual(reader.read(4), b"abcd")
        self.assertEqual(reader.read(2), b"ef")
        self.assertEqual(reader.read(10), b"")

    def test_supports_gzip_decompression_through_small_reads(self):
        original = b"line one\nline two\nline three\n"
        compressed = gzip.compress(original)
        # Force many small reads to exercise the buffering logic.
        chunks = [compressed[i:i + 3] for i in range(0, len(compressed), 3)]
        reader = _ChunkStreamReader(iter(chunks))
        with gzip.GzipFile(fileobj=reader) as gz:
            self.assertEqual(gz.read(), original)


class TestIterDumpRecordsFromChunks(unittest.TestCase):
    def test_decodes_gzipped_jsonl_stream(self):
        lines = [json.dumps({"n": 1}), json.dumps({"n": 2}), ""]
        compressed = gzip.compress("\n".join(lines).encode("utf-8"))
        chunks = [compressed[i:i + 5] for i in range(0, len(compressed), 5)]
        records = list(_iter_dump_records_from_chunks(iter(chunks)))
        self.assertEqual(records, [{"n": 1}, {"n": 2}])


class TestReadIsbnsFile(unittest.TestCase):
    def test_ignores_blank_lines_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "isbns.txt"
            path.write_text("9783899718188\n\n# a comment\n9781234567897\n", encoding="utf-8")
            self.assertEqual(_read_isbns_file(path), ["9783899718188", "9781234567897"])


class TestManifestFileHelpers(unittest.TestCase):
    def test_ensure_manifest_shell_creates_toc_only_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "sub" / "manifest.json"
            _ensure_manifest_shell(manifest_path)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data, {"toc_only": True, "books": []})

    def test_ensure_manifest_shell_is_a_noop_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text('{"toc_only": true, "books": [{"filename": "x.pdf"}]}', encoding="utf-8")
            _ensure_manifest_shell(manifest_path)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["books"]), 1)

    def test_append_book_adds_to_existing_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text('{"toc_only": true, "books": []}', encoding="utf-8")
            _append_book(manifest_path, {"filename": "a.pdf"})
            _append_book(manifest_path, {"filename": "b.pdf"})
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([b["filename"] for b in data["books"]], ["a.pdf", "b.pdf"])

    def test_load_existing_keys_returns_filename_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf"}]}),
                encoding="utf-8",
            )
            self.assertEqual(_load_existing_keys(manifest_path), {"9783899718188"})

    def test_load_existing_keys_empty_when_manifest_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_load_existing_keys(Path(tmp) / "manifest.json"), set())


if __name__ == "__main__":
    unittest.main()
