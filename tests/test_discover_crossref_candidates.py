"""Unit tests for evaluation/scripts/discover_crossref_candidates.py's pure
logic (URL/license resolution, dedup, language-priority ranking) against
mocked httpx responses -- no live network. The real Crossref-search
orchestration (discover()/main()) is exercised manually, matching
fetch_crossref_gt_corpus.py's existing convention of no pytest coverage for
its own network-calling main() entry point."""

import unittest
from collections import Counter
from unittest.mock import Mock

from evaluation.scripts.discover_crossref_candidates import (
    _crossref_link_pdf_url,
    _crossref_publisher_works,
    _is_new_candidate,
    _item_isbn,
    _item_title,
    _language_priority,
    _openalex_pdf_url,
    _select_candidates,
    _unpaywall_pdf_url,
    resolve_download_url,
)


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


class TestItemIsbnAndTitle(unittest.TestCase):
    def test_item_isbn_takes_first_entry(self):
        self.assertEqual(_item_isbn({"ISBN": ["9781234567897", "9781234567904"]}), "9781234567897")

    def test_item_isbn_none_when_absent(self):
        self.assertIsNone(_item_isbn({}))

    def test_item_title_takes_first_entry(self):
        self.assertEqual(_item_title({"title": ["A Book Title"]}), "A Book Title")

    def test_item_title_none_when_absent(self):
        self.assertIsNone(_item_title({}))


class TestCrossrefLinkPdfUrl(unittest.TestCase):
    def test_finds_application_pdf_content_type(self):
        item = {"link": [{"URL": "https://x/y.pdf", "content-type": "application/pdf"}]}
        self.assertEqual(_crossref_link_pdf_url(item), "https://x/y.pdf")

    def test_finds_unspecified_content_type(self):
        # Some publishers (e.g. Open Book Publishers) register their real
        # PDF link with content-type "unspecified" rather than the correct
        # MIME type -- confirmed against a live Crossref record.
        item = {"link": [{"URL": "https://obp/z.pdf", "content-type": "unspecified"}]}
        self.assertEqual(_crossref_link_pdf_url(item), "https://obp/z.pdf")

    def test_skips_non_pdf_content_type(self):
        item = {"link": [{"URL": "https://x/y.html", "content-type": "text/html"}]}
        self.assertIsNone(_crossref_link_pdf_url(item))

    def test_returns_none_when_no_link(self):
        self.assertIsNone(_crossref_link_pdf_url({}))


class TestUnpaywallPdfUrl(unittest.TestCase):
    def test_returns_url_for_pdf(self):
        client = Mock()
        client.get.return_value = _json_response(
            {"best_oa_location": {"url_for_pdf": "https://repo/paper.pdf"}}
        )
        self.assertEqual(_unpaywall_pdf_url("10.1/x", client, None), "https://repo/paper.pdf")

    def test_none_when_no_doi(self):
        client = Mock()
        self.assertIsNone(_unpaywall_pdf_url(None, client, None))
        client.get.assert_not_called()

    def test_none_when_request_fails(self):
        client = Mock()
        client.get.side_effect = Exception("network error")
        self.assertIsNone(_unpaywall_pdf_url("10.1/x", client, None))


class TestOpenAlexPdfUrl(unittest.TestCase):
    def test_returns_pdf_url(self):
        client = Mock()
        client.get.return_value = _json_response({"best_oa_location": {"pdf_url": "https://repo/paper.pdf"}})
        self.assertEqual(_openalex_pdf_url("10.1/x", client), "https://repo/paper.pdf")

    def test_none_when_no_doi(self):
        client = Mock()
        self.assertIsNone(_openalex_pdf_url(None, client))
        client.get.assert_not_called()


class TestResolveDownloadUrl(unittest.TestCase):
    def test_prefers_crossref_link(self):
        item = {"link": [{"URL": "https://crossref/x.pdf", "content-type": "application/pdf"}]}
        client = Mock()
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), ("https://crossref/x.pdf", "crossref"))
        client.get.assert_not_called()

    def test_falls_back_to_unpaywall(self):
        item = {}
        client = Mock()
        client.get.return_value = _json_response(
            {"best_oa_location": {"url_for_pdf": "https://unpaywall/x.pdf"}}
        )
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), ("https://unpaywall/x.pdf", "unpaywall"))

    def test_falls_back_to_openalex_when_unpaywall_empty(self):
        item = {}
        client = Mock()
        client.get.side_effect = [
            _json_response({"best_oa_location": None}),
            _json_response({"best_oa_location": {"pdf_url": "https://openalex/x.pdf"}}),
        ]
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), ("https://openalex/x.pdf", "openalex"))

    def test_none_when_all_three_fail(self):
        item = {}
        client = Mock()
        client.get.return_value = _json_response({"best_oa_location": None})
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), (None, None))


class TestIsNewCandidate(unittest.TestCase):
    def test_new_isbn_is_new(self):
        self.assertTrue(_is_new_candidate("111", "10.1/a", {"999"}, {"10.1/z"}))

    def test_known_isbn_is_not_new(self):
        self.assertFalse(_is_new_candidate("111", "10.1/a", {"111"}, set()))

    def test_known_doi_is_not_new_even_with_new_isbn(self):
        self.assertFalse(_is_new_candidate("111", "10.1/a", set(), {"10.1/a"}))

    def test_no_isbn_is_not_new(self):
        self.assertFalse(_is_new_candidate(None, "10.1/a", set(), set()))


class TestLanguagePriority(unittest.TestCase):
    def test_ranks_least_represented_first(self):
        counts = Counter({"en": 30, "de": 9, "fr": 2})
        self.assertEqual(_language_priority(counts, {"en", "de", "fr"}), ["fr", "de", "en"])

    def test_unseen_language_ranks_first(self):
        counts = Counter({"en": 30})
        self.assertEqual(_language_priority(counts, {"en", "es"}), ["es", "en"])


class TestSelectCandidates(unittest.TestCase):
    def test_caps_per_language_in_priority_order(self):
        by_language = {
            "en": [{"isbn": "1"}, {"isbn": "2"}, {"isbn": "3"}],
            "fr": [{"isbn": "10"}, {"isbn": "11"}],
        }
        selected = _select_candidates(by_language, priority=["fr", "en"], max_per_language=1)
        self.assertEqual(selected, [{"isbn": "10"}, {"isbn": "1"}])


class TestCrossrefPublisherWorks(unittest.TestCase):
    def test_stops_when_page_smaller_than_rows(self):
        client = Mock()
        client.get.return_value = _json_response(
            {"message": {"items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}]}}
        )
        result = _crossref_publisher_works("4923", "monograph", client, None, rows=100)
        self.assertEqual(len(result), 2)
        self.assertEqual(client.get.call_count, 1)

    def test_paginates_across_full_pages(self):
        client = Mock()
        page1 = _json_response({"message": {"items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}]}})
        page2 = _json_response({"message": {"items": [{"DOI": "10.1/c"}]}})
        client.get.side_effect = [page1, page2]
        result = _crossref_publisher_works("4923", "monograph", client, None, rows=2)
        self.assertEqual(len(result), 3)
        self.assertEqual(client.get.call_count, 2)

    def test_network_error_returns_partial_results_without_raising(self):
        import httpx

        client = Mock()
        page1 = _json_response({"message": {"items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}]}})
        client.get.side_effect = [page1, httpx.ConnectError("boom")]
        result = _crossref_publisher_works("4923", "monograph", client, None, rows=2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
