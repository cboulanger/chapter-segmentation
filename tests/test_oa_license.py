"""Unit tests for evaluation/oa_license.py's pure logic (item_license_url,
book_license_url) against literal Crossref-shaped dicts -- no network.
unpaywall_license_url/resolve_license are exercised indirectly by the
scripts that call them against the real network (fetch_crossref_gt_corpus.py,
discover_crossref_candidates.py, promote_pending_book.py); this file only
covers what needs no mocking to test meaningfully."""

import unittest

from evaluation.oa_license import book_license_url, item_license_url


class TestItemLicenseUrl(unittest.TestCase):
    def test_prefers_version_of_record_with_no_delay(self):
        item = {
            "license": [
                {"URL": "https://embargoed", "content-version": "am", "delay-in-days": 365},
                {"URL": "https://vor", "content-version": "vor", "delay-in-days": 0},
            ]
        }
        self.assertEqual(item_license_url(item), "https://vor")

    def test_falls_back_to_first_entry_when_no_vor_matches(self):
        item = {"license": [{"URL": "https://only-one", "content-version": "am", "delay-in-days": 365}]}
        self.assertEqual(item_license_url(item), "https://only-one")

    def test_none_when_no_license_key(self):
        self.assertIsNone(item_license_url({}))


class TestBookLicenseUrl(unittest.TestCase):
    def test_majority_vote_across_chapters(self):
        items = [
            {"license": [{"URL": "https://a", "content-version": "vor", "delay-in-days": 0}]},
            {"license": [{"URL": "https://a", "content-version": "vor", "delay-in-days": 0}]},
            {"license": [{"URL": "https://b", "content-version": "vor", "delay-in-days": 0}]},
        ]
        self.assertEqual(book_license_url(items), "https://a")

    def test_none_when_no_chapter_has_a_license(self):
        self.assertIsNone(book_license_url([{}, {}]))


if __name__ == "__main__":
    unittest.main()
