"""Unit tests for evaluation/nuextract_baseline.py -- NuExtract-1.5-tiny
zero-shot TOC-extraction baseline spike. See design spec
docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md."""

import unittest
from unittest.mock import MagicMock, patch

import httpx

from evaluation.nuextract_baseline import (
    build_prompt, call_ollama, match_toc_entries, parse_response, score_book,
)


class TestBuildPrompt(unittest.TestCase):
    def test_includes_template_and_page_text(self):
        pages = ["front matter", "Contents\nIntro ... 1", "back matter"]
        prompt = build_prompt(pages, [1])
        self.assertIn("### Template:", prompt)
        self.assertIn('"chapters"', prompt)
        self.assertIn("### Text:", prompt)
        self.assertIn("Contents\nIntro ... 1", prompt)
        self.assertTrue(prompt.startswith("<|input|>"))
        self.assertTrue(prompt.endswith("<|output|>"))

    def test_joins_multiple_scan_indices_in_order_and_skips_others(self):
        pages = ["p0", "p1", "p2"]
        prompt = build_prompt(pages, [0, 2])
        text_section = prompt.split("### Text:\n")[1]
        self.assertTrue(text_section.startswith("p0"))
        self.assertIn("p2", text_section)
        self.assertNotIn("p1", text_section)


class TestParseResponse(unittest.TestCase):
    def test_extracts_chapters_list(self):
        raw = '{"chapters": [{"title": "Intro", "authors": ["A"], "printed_page_number": "1"}]}'
        self.assertEqual(
            parse_response(raw),
            [{"title": "Intro", "authors": ["A"], "printed_page_number": "1"}],
        )

    def test_strips_code_fence(self):
        raw = '```json\n{"chapters": []}\n```'
        self.assertEqual(parse_response(raw), [])

    def test_returns_empty_on_malformed_json(self):
        self.assertEqual(parse_response("not json at all"), [])

    def test_returns_empty_when_chapters_not_a_list(self):
        self.assertEqual(parse_response('{"chapters": "oops"}'), [])


class TestMatchTocEntries(unittest.TestCase):
    def test_matches_on_title_and_page(self):
        predicted = [{"title": "Introduction", "printed_page_number": "1"}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 1)

    def test_rejects_page_mismatch(self):
        predicted = [{"title": "Introduction", "printed_page_number": "2"}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)

    def test_rejects_title_mismatch(self):
        predicted = [{"title": "Completely Different", "printed_page_number": "1"}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)

    def test_handles_roman_numerals(self):
        predicted = [{"title": "Foreword", "printed_page_number": "vii"}]
        expected = [{"title": "Foreword", "citation_pages": "vii-ix"}]
        self.assertEqual(match_toc_entries(predicted, expected), 1)

    def test_is_monotonic_like_fusion_align(self):
        # "Chapter Two" is predicted first and consumes expected[1]; the
        # later "Chapter One" prediction can then only search from
        # expected[2:] onward (mirrors fusion._align's "TOC order is book
        # order" assumption), so it finds nothing even though a textual
        # match for it exists earlier in the expected list.
        predicted = [
            {"title": "Chapter Two", "printed_page_number": "20"},
            {"title": "Chapter One", "printed_page_number": "1"},
        ]
        expected = [
            {"title": "Chapter One", "citation_pages": "1-19"},
            {"title": "Chapter Two", "citation_pages": "20-39"},
        ]
        self.assertEqual(match_toc_entries(predicted, expected), 1)

    def test_null_printed_page_number_never_matches(self):
        predicted = [{"title": "Introduction", "printed_page_number": None}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)

    def test_null_citation_pages_never_matches(self):
        predicted = [{"title": "Introduction", "printed_page_number": "1"}]
        expected = [{"title": "Introduction", "citation_pages": None}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)

    def test_non_dict_predicted_item_is_skipped_not_raised(self):
        predicted = ["not a dict", {"title": "Introduction", "printed_page_number": "1"}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 1)


class TestScoreBook(unittest.TestCase):
    def test_computes_precision_and_recall(self):
        predicted = [
            {"title": "Introduction", "printed_page_number": "1"},
            {"title": "Spurious Entry", "printed_page_number": "99"},
        ]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        metrics = score_book(predicted, expected)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.found_count, 2)
        self.assertEqual(metrics.expected_count, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 1.0)

    def test_empty_predicted_scores_zero_precision_and_recall(self):
        metrics = score_book([], [{"title": "Introduction", "citation_pages": "1-31"}])
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)


class TestCallOllama(unittest.TestCase):
    def test_posts_raw_mode_request_to_generate_endpoint(self):
        response = MagicMock()
        response.json.return_value = {"response": '{"chapters": []}'}
        response.raise_for_status.return_value = None
        with patch("evaluation.nuextract_baseline.httpx.post", return_value=response) as mock_post:
            result = call_ollama("http://localhost:11434", "hf.co/example:Q8_0", "prompt-text")
        self.assertEqual(result, '{"chapters": []}')
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertIs(kwargs["json"]["raw"], True)
        self.assertEqual(kwargs["json"]["prompt"], "prompt-text")
        self.assertEqual(kwargs["json"]["model"], "hf.co/example:Q8_0")
        self.assertIs(kwargs["json"]["stream"], False)
        self.assertEqual(kwargs["json"]["options"], {"temperature": 0, "num_predict": 4000, "num_ctx": 16384})
        self.assertEqual(kwargs["timeout"], 300.0)

    def test_strips_trailing_slash_on_base_url(self):
        response = MagicMock()
        response.json.return_value = {"response": ""}
        response.raise_for_status.return_value = None
        with patch("evaluation.nuextract_baseline.httpx.post", return_value=response) as mock_post:
            call_ollama("http://localhost:11434/", "model", "prompt")
        self.assertEqual(mock_post.call_args[0][0], "http://localhost:11434/api/generate")

    def test_raises_on_http_error(self):
        response = MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=MagicMock()
        )
        with patch("evaluation.nuextract_baseline.httpx.post", return_value=response):
            with self.assertRaises(httpx.HTTPStatusError):
                call_ollama("http://localhost:11434", "model", "prompt")


if __name__ == "__main__":
    unittest.main()
