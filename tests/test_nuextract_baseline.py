"""Unit tests for evaluation/nuextract_baseline.py -- NuExtract-1.5-tiny
zero-shot TOC-extraction baseline spike. See design spec
docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md."""

import unittest
from unittest.mock import MagicMock, patch

from evaluation.nuextract_baseline import build_prompt, parse_response


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


if __name__ == "__main__":
    unittest.main()
