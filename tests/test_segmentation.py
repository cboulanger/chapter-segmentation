"""Unit tests for chapter_segmentation.segmentation."""

import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

from pathlib import Path
from tempfile import TemporaryDirectory

import io as _io
from pypdf import PdfWriter as _PdfWriter

from chapter_segmentation.segmentation import (
    TocEntry,
    extract_page_texts_from_pdf_bytes,
    extract_page_texts_for_analysis,
    find_toc_candidates,
    llm_extract_toc_entries,
    load_cached_analysis,
    pages_need_ocr,
    save_analysis_cache,
    _toc_scan_indices,
    _llm_scan_indices,
    _classify_llm_failure,
    _extract_with_retry,
    _page_number_anchors,
    _infer_printed_page,
    _to_roman,
    _toc_declared_page,
    _fallback_end_printed,
    analyze_attachment_with_llm_fallback,
    analyze_attachment_outline_only,
    analyze_attachment_llm_only,
)
from chapter_segmentation.segmentation import (
    ChapterStartCandidate,
    ChapterStartMatch,
    extract_printed_page_number,
    llm_disambiguate_chapter_start,
    locate_chapter_start,
    locate_chapter_start_candidates,
    match_confidence,
    _LOCATE_MARGIN_REQUIRED,
    _parse_toc_page_number,
)
from chapter_segmentation.segmentation import _chapters_from_located
from chapter_segmentation.segmentation import extract_authors_near
from chapter_segmentation.segmentation import analyze_attachment
from chapter_segmentation.evidence.outline_strategy import extract_outline_candidates
from chapter_segmentation.evidence.types import ChapterCandidate


def _blank_pdf(num_pages: int) -> bytes:
    writer = _PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = _io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _pdf_with_outline(num_pages: int, entries: list[tuple[str, int]]) -> bytes:
    writer = _PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    for title, page_number in entries:
        writer.add_outline_item(title, page_number)
    buf = _io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# Repeated (not a single short line) so this filler page clears
# pages_need_ocr's per-page "substantial" (>500 chars) and "not degenerate"
# (>=3 newlines) thresholds -- a single short line reads as OCR-shaped input
# and short-circuits run() into the needs_ocr branch before any chapter
# segmentation strategy runs.
_FILLER = "Unrelated body filler text, nothing chapter-related in this passage at all.\n" * 8

# Same 20-page, indices-5-and-12 fixture as
# test_chapter_segmentation_strategies.py's _TWO_CHAPTER_PAGES (Task 8) --
# kept as a separate copy here since these two test files don't import from
# each other. See that constant's comment for why the chapters must sit
# outside _toc_scan_indices's front/back exclusion zone.
_TWO_CHAPTER_PAGES = [
    _FILLER,  # 0
    _FILLER,  # 1
    _FILLER,  # 2
    _FILLER,  # 3
    _FILLER,  # 4
    "Introduction\nJane Author\n\nBody text opening the chapter.\n\n1",  # 5
    "...continued introduction text with real body content here.\n\n2",  # 6
    "...more continued introduction text with real body content.\n\n3",  # 7
    "...final continued introduction text with real body content.\n\n4",  # 8
    _FILLER,  # 9
    _FILLER,  # 10
    _FILLER,  # 11
    "Comparing Citation Styles\n\nJohn Smith\n\nBody text opening this chapter.\n\n5",  # 12
    "...continued citation styles text with real body content here.\n\n6",  # 13
    "...more continued citation styles text with real body content.\n\n7",  # 14
    "...final continued citation styles text with real body content.\n\n8",  # 15
    _FILLER,  # 16
    _FILLER,  # 17
    _FILLER,  # 18
    _FILLER,  # 19
]


class TestFindTocCandidates(unittest.TestCase):
    # Padded well past any page number used in these fixtures' TOC lines --
    # find_toc_candidates rejects a printed page number that looks
    # implausible relative to the PDF's actual total page count (see
    # _TOC_MAX_PAGE_NUMBER_RATIO), so tiny fixtures need enough filler pages
    # for their own realistic-looking page numbers (e.g. 89) not to trip
    # that guard by accident.
    _FILLER_PAGES = ["Just filler body text, nothing TOC-like here."] * 100

    def test_finds_dotted_leader_entries(self):
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 89\n",
            "Some front-matter page with no TOC pattern at all.",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].title, "Introduction to Reference Management")
        self.assertEqual(entries[0].printed_page_number, 1)
        self.assertEqual(entries[0].source_page_index, 0)
        # The listing's own "CONTENTS" heading is never merged into a
        # wrapped-title variant (see _TOC_MAX_CONTINUATION_LINES walk).
        self.assertEqual(entries[0].title_variants, ())
        self.assertEqual(entries[2].title, "Zotero in Practice")
        self.assertEqual(entries[2].printed_page_number, 89)

    def test_entries_default_to_empty_authors(self):
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual(entries[0].authors, ())

    def test_ignores_non_toc_lines(self):
        pages = ["Just some ordinary prose with numbers like 1999 in it, no leaders here."]
        entries = find_toc_candidates(pages)
        self.assertEqual(entries, [])

    def test_matches_whitespace_leaders_too(self):
        # Three dotted/whitespace-leader lines on the same page -- meets
        # _TOC_MIN_LINES_PER_PAGE, so all three are trusted as real entries
        # (an isolated single line on a page would now be discarded as
        # noise -- see test_ignores_isolated_lines_on_ordinary_pages below).
        pages = [
            "Bibliographic Software Overview          12\n"
            "Comparing Citation Managers          30\n"
            "Zotero in Practice          55\n"
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].title, "Bibliographic Software Overview")
        self.assertEqual(entries[0].printed_page_number, 12)

    def test_ignores_isolated_lines_on_ordinary_pages(self):
        # Only 1-2 matching lines on a page (below _TOC_MIN_LINES_PER_PAGE)
        # -- a citation, footnote reference, or running header that happens
        # to fit the dotted/whitespace-leader pattern, not a real chapter
        # listing. These must be discarded, not returned as entries.
        pages = [
            "Some ordinary content page mentioning a reference.          12",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual(entries, [])

    def test_ignores_implausible_page_numbers(self):
        # A publication year embedded in copyright/imprint text ("Opladen
        # (c) Publisher          2025") looks like a TOC line but its
        # "page number" is wildly larger than the PDF's actual length --
        # found empirically in a real evaluation book. Only lines with
        # plausible page numbers count toward _TOC_MIN_LINES_PER_PAGE (an
        # imprint page full of filtered lines must not qualify as the TOC,
        # see test_metadata_page_of_filtered_lines_does_not_shadow_real_toc)
        # -- three valid lines keep this page qualifying, and the imprint
        # line is still dropped from the result.
        pages = [
            "Publisher Imprint Line          2025\n"
            "Introduction to the Subject          1\n"
            "Comparing Citation Styles          45\n"
            "Zotero in Practice          89\n"
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        titles = [e.title for e in entries]
        self.assertNotIn("Publisher Imprint Line", titles)
        self.assertIn("Introduction to the Subject", titles)
        self.assertIn("Comparing Citation Styles", titles)

    def test_ignores_url_and_doi_lines(self):
        # A repeated DOI/URL line in front matter can otherwise fit the
        # dotted/whitespace-leader pattern (found empirically in a real
        # evaluation book, repeated 5 times) -- never a real chapter title.
        pages = [
            "https://doi.org/10.1234/example.5678\n"
            "Introduction to the Subject          1\n"
            "Comparing Citation Styles          45\n"
            "Zotero in Practice          89\n"
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        titles = [e.title for e in entries]
        self.assertFalse(any("doi.org" in t for t in titles))
        self.assertIn("Introduction to the Subject", titles)

    def test_metadata_page_of_filtered_lines_does_not_shadow_real_toc(self):
        # An imprint/metadata page whose lines ALL fail the content filters
        # (years, DOI numbers) must not qualify as the "first TOC cluster"
        # and shadow the real table of contents further in -- the per-line
        # filters run BEFORE the per-page density count (found empirically:
        # a French evaluation book's page-1 metadata block hid its real TOC
        # on pages 5-8).
        pages = [
            "Fancy Book Title Page",
            "DOI : 10.4000/books.example          7527\n"
            "Année d'édition :          2017\n"
            "Date de mise en ligne : 21 mai          2019\n",
            "Some other front matter.",
            "CONTENTS\n"
            "Introduction to Reference Management ..... 12\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual([e.source_page_index for e in entries], [3, 3, 3])
        self.assertIn("Comparing Citation Styles", [e.title for e in entries])

    def test_imprint_page_of_copyright_lines_does_not_shadow_real_toc(self):
        # A copyright/imprint page whose lines are plain boilerplate ("©
        # 1978 Publisher, City", "Gedruckt 1978 bei ...", "ISBN ...") isn't
        # a URL/DOI, so _looks_like_url_or_doi doesn't catch it -- but each
        # line still ends in a trailing number (a street/city code, a
        # printing detail, an ISBN check digit) and so matches the TOC-line
        # pattern just as easily as the DOI-block case above (found
        # empirically: a 1978 German Festschrift's copyright page -- three
        # such lines -- shadowed its real table of contents three pages
        # later, since _TOC_MIN_LINES_PER_PAGE is 3). The gap to the real
        # TOC below is kept > _TOC_PAGE_CLUSTER_GAP (2) so the two pages
        # form separate clusters rather than merging into one -- otherwise
        # this reproduces a different (milder) bug than the one observed.
        pages = [
            "Fancy Book Title Page",
            "© 1978 Some Publisher, Berlin 41\n"
            "Gedruckt 1978 bei Some Printing House, Berlin 61\n"
            "ISBN 3 428 04224 1\n",
            "Some other front matter.",
            "Some more front matter, still no TOC pattern here at all.",
            "Yet more front matter before the real listing begins.",
            "CONTENTS\n"
            "Introduction to Reference Management ..... 12\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual([e.source_page_index for e in entries], [5, 5, 5])
        self.assertIn("Comparing Citation Styles", [e.title for e in entries])

    def test_wrapped_title_builds_variants_from_preceding_lines(self):
        # A long title wrapped over several lines carries its page number on
        # the LAST line only -- the preceding lines are offered as
        # progressively longer variant readings for the locate step to
        # arbitrate. The walk stops at another entry's line.
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Making Terrorism: Security Practices and the Production of Terror\n"
            "Activities in Canada ..... 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        wrapped = next(e for e in entries if e.title == "Activities in Canada")
        self.assertIn(
            "Making Terrorism: Security Practices and the Production of Terror Activities in Canada",
            wrapped.title_variants,
        )
        # The walk stopped at the previous entry's own line -- it is never
        # part of a variant.
        self.assertFalse(any("Reference Management" in v for v in wrapped.title_variants))

    def test_wrapped_title_walk_stops_at_part_headers(self):
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Part II Gendered Violence and Racial Subjugation\n"
            "Activities in Canada ..... 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        wrapped = next(e for e in entries if e.title == "Activities in Canada")
        self.assertFalse(any("Part II" in v for v in wrapped.title_variants))

    def test_roman_numeral_page_numbers_accepted_for_front_matter(self):
        # "Foreword vii" is a real front-matter TOC entry; the entry is
        # flagged printed_roman so localization may search pre-TOC pages.
        pages = [
            "CONTENTS\n"
            "Foreword vii\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        foreword = next(e for e in entries if e.title == "Foreword")
        self.assertEqual(foreword.printed_page_number, 7)
        self.assertTrue(foreword.printed_roman)
        self.assertFalse(next(e for e in entries if e.title == "Zotero in Practice").printed_roman)

    def test_ordinary_words_of_roman_letters_are_not_page_numbers(self):
        # "civil" is c-i-v-i-l -- every letter a roman digit, but not a
        # valid roman numeral. A TOC line ending in such a word must not
        # become an entry with a hallucinated page number.
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "A Treatise on Matters          civil\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertFalse(any("Treatise" in e.title for e in entries))

    def test_bare_page_number_line_adopts_preceding_title_line(self):
        # Some TOC layouts put the dot leader + page number on a line of its
        # own, with the title (and author) lines above it.
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles\n"
            "................................. 45\n"
            "Zotero in Practice ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        adopted = next((e for e in entries if e.title == "Comparing Citation Styles"), None)
        self.assertIsNotNone(adopted)
        self.assertEqual(adopted.printed_page_number, 45)

    def test_author_marker_toc_keeps_only_chapter_level_entries(self):
        # French/OpenEdition-style TOC: each chapter's page number sits on a
        # "par <Author>" line under the wrapped title, while sub-headings
        # also carry page numbers. With 3+ marker entries, only they are
        # chapters -- and the author names are read off the marker line.
        pages = [
            "SOMMAIRE\n"
            "MODE D'EMPLOI\n"
            "par Lucie Daudin .................. 9\n"
            "UNE SOUS-PARTIE QUELCONQUE .................. 11\n"
            "FRANCE, SOCIÉTÉ MULTICULTURELLE\n"
            "par Patrick Simon .................. 29\n"
            "AUTRE SOUS-PARTIE .................. 31\n"
            "LANGUES ET POLITIQUES PUBLIQUES\n"
            "par Alexandra Filhon .................. 38\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(e.title.startswith("par ") for e in entries))
        simon = next(e for e in entries if "Patrick Simon" in e.title)
        self.assertEqual(simon.authors, ("Patrick Simon",))
        self.assertTrue(any("FRANCE, SOCIÉTÉ MULTICULTURELLE" in v for v in simon.title_variants))

    def test_toc_continuation_page_with_few_entries_joins_cluster(self):
        # The listing's last page may hold only its final two entries --
        # below the density threshold on its own, but a genuine continuation
        # of the already-trusted cluster.
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 60\n",
            "Final Thoughts on Reference Management ..... 75\n"
            "Closing Remarks and Outlook ..... 89\n",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertIn("Closing Remarks and Outlook", [e.title for e in entries])

    def test_single_stray_line_does_not_extend_toc_cluster(self):
        # One lone matching line on the page after the TOC is
        # indistinguishable from an ordinary body page's incidental
        # "text ... number" line -- swallowing it as "TOC" would cost the
        # first chapter (its opening page becomes excluded from location).
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 60\n",
            "Ordinary body text that happens to end with a number          12\n"
            "and then continues with completely ordinary prose afterwards.",
        ] + self._FILLER_PAGES
        entries = find_toc_candidates(pages)
        self.assertEqual({e.source_page_index for e in entries}, {0})


class TestLlmScanIndices(unittest.TestCase):
    _FILLER_PAGE = "Ordinary body filler text, nothing chapter-related here at all."

    def test_falls_back_to_blind_fraction_when_no_heuristic_toc_found(self):
        pages = [self._FILLER_PAGE] * 20
        self.assertEqual(_llm_scan_indices(pages), sorted(_toc_scan_indices(pages)))

    def test_narrows_to_padded_heuristic_toc_page(self):
        # 40 pages so the blind fraction (front 15% -> {0..5}, back 5% ->
        # {38,39}) is wide enough to prove real narrowing: the heuristic TOC
        # sits at index 3, inside the front zone, but _llm_scan_indices
        # should return only that page +-1, not the whole 8-page blind zone.
        pages = (
            [self._FILLER_PAGE] * 3
            + [
                "CONTENTS\n"
                "Introduction to Reference Management ..... 1\n"
                "Comparing Citation Styles ..... 45\n"
                "Zotero in Practice ..... 60\n"
            ]
            + [self._FILLER_PAGE] * 36
        )
        self.assertEqual(len(pages), 40)
        self.assertEqual(sorted(_toc_scan_indices(pages)), [0, 1, 2, 3, 4, 5, 38, 39])
        self.assertEqual(_llm_scan_indices(pages), [2, 3, 4])

    def test_clamps_padding_at_document_start(self):
        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Comparing Citation Styles ..... 45\n"
            "Zotero in Practice ..... 60\n"
        ] + [self._FILLER_PAGE] * 39
        self.assertEqual(_llm_scan_indices(pages), [0, 1])

    def test_returns_empty_list_for_empty_pages(self):
        self.assertEqual(_llm_scan_indices([]), [])


class TestClassifyLlmFailure(unittest.TestCase):
    def test_classifies_context_length_message(self):
        exc = Exception("This model's maximum context length is 65536 tokens, however you requested 98213 tokens")
        self.assertEqual(_classify_llm_failure(exc), "context_length_exceeded")

    def test_classifies_no_json_array_found_message(self):
        exc = ValueError("No JSON array found in LLM response: '...'")
        self.assertEqual(_classify_llm_failure(exc), "invalid_or_truncated_json")

    def test_classifies_json_decode_error_message(self):
        exc = ValueError("Expecting ',' delimiter: line 1 column 50 (char 49)")
        self.assertEqual(_classify_llm_failure(exc), "invalid_or_truncated_json")

    def test_classifies_unrecognized_message_as_api_error(self):
        exc = RuntimeError("connection reset by peer")
        self.assertEqual(_classify_llm_failure(exc), "api_error")


class TestExtractWithRetry(unittest.IsolatedAsyncioTestCase):
    def _fake_llm(self, *responses):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=list(responses))
        return llm

    async def test_returns_parsed_result_on_first_success_without_retry(self):
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": 1}]')
        items = await _extract_with_retry("prompt", llm)
        self.assertEqual(len(items), 1)
        llm.generate.assert_called_once()
        self.assertEqual(llm.generate.call_args.kwargs["max_tokens"], 1024)

    async def test_retries_with_higher_max_tokens_on_truncated_first_response(self):
        llm = self._fake_llm(
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}',  # truncated, no closing ]
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}]',  # valid
        )
        items = await _extract_with_retry("prompt", llm)
        self.assertEqual(len(items), 1)
        self.assertEqual(llm.generate.call_count, 2)
        self.assertEqual(llm.generate.call_args_list[0].kwargs["max_tokens"], 1024)
        self.assertEqual(llm.generate.call_args_list[1].kwargs["max_tokens"], 8192)

    async def test_raises_when_both_attempts_fail_to_parse(self):
        llm = self._fake_llm("not json at all", "still not json")
        with self.assertRaises(Exception):
            await _extract_with_retry("prompt", llm)
        self.assertEqual(llm.generate.call_count, 2)

    async def test_does_not_retry_when_generate_itself_raises(self):
        # A context-length error (or any other API-level failure) can't be
        # fixed by asking for more output tokens -- only truncated-but-
        # otherwise-successful responses should trigger the retry.
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("maximum context length is 65536 tokens"))
        with self.assertRaises(RuntimeError):
            await _extract_with_retry("prompt", llm)
        llm.generate.assert_called_once()


class TestLlmExtractTocEntries(unittest.IsolatedAsyncioTestCase):
    def _fake_llm(self, response: str):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=response)
        return llm

    async def test_parses_chapter_list_from_llm_response(self):
        response = (
            '[{"title": "Introduction", "authors": ["Jane Author"], "printed_page_number": 1}, '
            '{"title": "Comparing Citation Styles", "authors": [], "printed_page_number": null}]'
        )
        llm = self._fake_llm(response)
        pages = ["Front matter page with an irregularly formatted chapter listing."] * 5
        entries = await llm_extract_toc_entries(pages, llm)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].title, "Introduction")
        self.assertEqual(entries[0].authors, ("Jane Author",))
        self.assertEqual(entries[0].printed_page_number, 1)
        self.assertEqual(entries[1].printed_page_number, -1)  # null -> sentinel, unused downstream

    async def test_returns_empty_list_on_malformed_response(self):
        llm = self._fake_llm("not json at all")
        entries = await llm_extract_toc_entries(["some front matter"] * 5, llm)
        self.assertEqual(entries, [])

    async def test_skips_entries_with_too_short_title(self):
        llm = self._fake_llm('[{"title": "Hi", "authors": [], "printed_page_number": 1}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries, [])

    async def test_returns_empty_list_for_empty_pages(self):
        llm = self._fake_llm("[]")
        entries = await llm_extract_toc_entries([], llm)
        self.assertEqual(entries, [])
        llm.generate.assert_not_called()

    async def test_skip_defaults_to_false_when_absent(self):
        # The production text prompt never asks for "skip" (that's a
        # dnb-toc-only vision-extraction-specific concern -- see
        # TocEntry.skip's docstring) -- confirm the shared parser doesn't
        # require it.
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": 1}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertFalse(entries[0].skip)

    async def test_skip_is_parsed_when_present(self):
        llm = self._fake_llm(
            '[{"title": "Bibliographie", "authors": [], "printed_page_number": 1, "skip": true}]'
        )
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertTrue(entries[0].skip)

    async def test_ignores_non_list_authors_instead_of_corrupting(self):
        # A malformed LLM response giving a plain string instead of a list
        # (e.g. "authors": "Jane Doe") must not be iterated character-by-
        # character -- it should be treated as no author info at all.
        llm = self._fake_llm('[{"title": "Introduction", "authors": "Jane Doe", "printed_page_number": 1}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].authors, ())

    async def test_passes_is_valid_check_for_json_array_shape(self):
        # Lets an AutoSelectLLMService rotate to a different model when one
        # hallucinates a non-JSON response instead of raising an exception.
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": 1}]')
        await llm_extract_toc_entries(["front matter"] * 5, llm)
        is_valid = llm.generate.call_args.kwargs["is_valid"]
        self.assertTrue(is_valid('[{"title": "x"}]'))
        self.assertFalse(is_valid("I'll call a tool instead"))

    async def test_recovers_via_retry_when_first_response_is_truncated(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=[
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}',  # truncated
            '[{"title": "Introduction", "authors": [], "printed_page_number": 1}]',  # valid on retry
        ])
        entries = await llm_extract_toc_entries(["front matter"] * 20, llm)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].title, "Introduction")
        self.assertEqual(llm.generate.call_count, 2)
        self.assertEqual(llm.generate.call_args_list[0].kwargs["max_tokens"], 1024)
        self.assertEqual(llm.generate.call_args_list[1].kwargs["max_tokens"], 8192)

    async def test_logs_classified_reason_when_both_attempts_fail(self):
        llm = self._fake_llm("not json at all")
        with self.assertLogs("chapter_segmentation.segmentation", level="WARNING") as cm:
            entries = await llm_extract_toc_entries(["front matter"] * 20, llm)
        self.assertEqual(entries, [])
        self.assertTrue(any("invalid_or_truncated_json" in message for message in cm.output))

    async def test_parses_roman_numeral_page_string(self):
        llm = self._fake_llm('[{"title": "Foreword", "authors": [], "printed_page_number": "vii"}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, 7)
        self.assertTrue(entries[0].printed_roman)

    async def test_parses_arabic_page_string(self):
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": "12"}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, 12)
        self.assertFalse(entries[0].printed_roman)

    async def test_tolerates_legacy_bare_int_page_number(self):
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": 12}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, 12)
        self.assertFalse(entries[0].printed_roman)

    async def test_null_page_number_still_uses_sentinel(self):
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": null}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, -1)
        self.assertFalse(entries[0].printed_roman)

    async def test_implausible_roman_string_uses_sentinel(self):
        # Over _ROMAN_PAGE_MAX_VALUE (50) -- _parse_toc_page_number rejects
        # it as an implausible roman numeral, same as a heuristic-found one.
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": "mmmm"}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, -1)
        self.assertFalse(entries[0].printed_roman)


class TestLlmDisambiguateChapterStart(unittest.IsolatedAsyncioTestCase):
    def _fake_llm(self, response: str):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=response)
        return llm

    async def test_picks_chosen_candidate(self):
        candidates = [
            ChapterStartCandidate(index=0, score=95.0, author_confirmed=False),
            ChapterStartCandidate(index=5, score=93.0, author_confirmed=False),
        ]
        pages = ["Comparing Citation Styles by Jane Doe..."] * 6
        llm = self._fake_llm('{"chosen_candidate": 2}')
        match = await llm_disambiguate_chapter_start(pages, "Comparing Citation Styles", (), candidates, llm)
        self.assertIsNotNone(match)
        self.assertEqual(match.index, 5)
        self.assertEqual(match.score, 93.0)
        self.assertEqual(match.margin, _LOCATE_MARGIN_REQUIRED)

    async def test_returns_none_when_llm_picks_none(self):
        candidates = [ChapterStartCandidate(index=0, score=95.0, author_confirmed=False)]
        pages = ["some text"] * 2
        llm = self._fake_llm('{"chosen_candidate": null}')
        match = await llm_disambiguate_chapter_start(pages, "Title", (), candidates, llm)
        self.assertIsNone(match)

    async def test_returns_none_on_out_of_range_choice(self):
        candidates = [ChapterStartCandidate(index=0, score=95.0, author_confirmed=False)]
        pages = ["some text"] * 2
        llm = self._fake_llm('{"chosen_candidate": 5}')
        match = await llm_disambiguate_chapter_start(pages, "Title", (), candidates, llm)
        self.assertIsNone(match)

    async def test_returns_none_on_zero_choice(self):
        # 0 is a distinct boundary from "too high" -- a plausible LLM
        # off-by-one response if it 0-indexes instead of 1-indexing.
        candidates = [ChapterStartCandidate(index=0, score=95.0, author_confirmed=False)]
        pages = ["some text"] * 2
        llm = self._fake_llm('{"chosen_candidate": 0}')
        match = await llm_disambiguate_chapter_start(pages, "Title", (), candidates, llm)
        self.assertIsNone(match)

    async def test_returns_none_on_malformed_response(self):
        candidates = [ChapterStartCandidate(index=0, score=95.0, author_confirmed=False)]
        pages = ["some text"] * 2
        llm = self._fake_llm("not json")
        match = await llm_disambiguate_chapter_start(pages, "Title", (), candidates, llm)
        self.assertIsNone(match)

    async def test_passes_is_valid_check_for_json_object_shape(self):
        candidates = [ChapterStartCandidate(index=0, score=95.0, author_confirmed=False)]
        pages = ["some text"] * 2
        llm = self._fake_llm('{"chosen_candidate": 1}')
        await llm_disambiguate_chapter_start(pages, "Title", (), candidates, llm)
        is_valid = llm.generate.call_args.kwargs["is_valid"]
        self.assertTrue(is_valid('{"chosen_candidate": 1}'))
        self.assertFalse(is_valid("I'll call a tool instead"))


class TestTocScanIndices(unittest.TestCase):
    def test_scans_front_and_back_fractions(self):
        pages = ["x"] * 100
        indices = _toc_scan_indices(pages, max_front_fraction=0.1, max_back_fraction=0.05)
        self.assertIn(0, indices)
        self.assertIn(9, indices)  # last front-matter page (10% of 100)
        self.assertNotIn(10, indices)
        self.assertIn(99, indices)  # last back-matter page is always included
        self.assertNotIn(50, indices)  # a middle page is never scanned

    def test_empty_pages_returns_empty_set(self):
        self.assertEqual(_toc_scan_indices([]), set())


class TestLocateChapterStart(unittest.TestCase):
    def test_finds_matching_page_by_content(self):
        pages = [
            "CONTENTS\nComparing Citation Styles ..... 3\n",  # TOC page itself
            "Some unrelated front matter.",
            "Comparing Citation Styles\nBy Jane Author\n\nThis chapter examines...",
        ]
        match = locate_chapter_start(pages, "Comparing Citation Styles", exclude_indices={0})
        self.assertEqual(match.index, 2)
        self.assertEqual(match.score, 100.0)
        # Uncontested (only one qualifying cluster) -- margin equals score.
        self.assertEqual(match.margin, 100.0)

    def test_returns_none_when_no_good_match(self):
        pages = ["Nothing related to the query here at all, just filler prose."]
        match = locate_chapter_start(pages, "Comparing Citation Styles", exclude_indices=set())
        self.assertIsNone(match)

    def test_rejects_sparse_page_despite_high_raw_score(self):
        # Empirically (see rapidfuzz.fuzz.partial_ratio), a near-blank head
        # like "Cit" scores 100 against "Comparing Citation Styles" because
        # partial_ratio finds a perfect alignment for the short substring --
        # the same degenerate score as the genuine, well-formed chapter
        # opening on page 2. Without a minimum-length gate, the first-seen
        # (blank) page wins the tie. With the gate, the blank page is
        # excluded and the genuine page is the only remaining candidate.
        pages = [
            "Cit",  # stray near-blank page; partial_ratio scores this 100
            "Some unrelated front matter.",
            "Comparing Citation Styles\n\nBy Jane Author\n\nThis chapter examines...",
        ]
        match = locate_chapter_start(pages, "Comparing Citation Styles", exclude_indices=set())
        self.assertEqual(match.index, 2)

    def test_rejects_ambiguous_tie(self):
        # Empirically, both pages score >= _LOCATE_SCORE_THRESHOLD (80) for
        # this title -- page 0 scores 100.0, page 5 scores ~97.96 -- a margin
        # of ~2 that is well under _LOCATE_MARGIN_REQUIRED. They're placed
        # farther apart than _LOCATE_CLUSTER_GAP so they form two separate
        # clusters (genuine competing candidates), not one merged location.
        # Neither candidate can be trusted, so the function must return None.
        pages = [
            "Comparing Citation Styles\n\nBy Jane Author\n\nThis chapter examines APA style only.",
            "Unrelated filler page one.",
            "Unrelated filler page two.",
            "Unrelated filler page three.",
            "Unrelated filler page four.",
            "Comparing Citation Style\n\nBy John Smith\n\nAnother chapter about MLA style.",
        ]
        match = locate_chapter_start(pages, "Comparing Citation Styles", exclude_indices=set())
        self.assertIsNone(match)

    def test_treats_nearby_repeated_header_as_one_location(self):
        # Real books often repeat a chapter's title in the running header on
        # several of the chapter's OWN pages (found empirically in an
        # evaluation book: the same title scored 100.0 on both the opening
        # page and a later page of the same short chapter, with a gap in
        # between where an intervening page didn't score highly). These
        # nearby repeats must be treated as ONE candidate location -- not
        # rejected as an ambiguous tie between two different chapters -- so
        # the earliest (opening) page should still be returned.
        pages = [
            "Comparing Citation Styles\n\nBy Jane Author\n\nThis chapter examines APA style.",
            "Some body text continues the discussion of citation formats here.",
            "Comparing Citation Styles16\n\nMore body text about citation formats continues.",
        ]
        match = locate_chapter_start(pages, "Comparing Citation Styles", exclude_indices=set())
        self.assertEqual(match.index, 0)
        # Uncontested (the repeated-header page merged into the same
        # cluster, not a rival) -- margin equals score, same as any other
        # single-cluster match.
        self.assertEqual(match.margin, match.score)

    def test_locate_chapter_start_candidates_exposes_competing_clusters(self):
        pages = [
            "Comparing Citation Styles\n\nBy Jane Author\n\nThis chapter examines APA style only.",
            "Unrelated filler page one.",
            "Unrelated filler page two.",
            "Unrelated filler page three.",
            "Unrelated filler page four.",
            "Comparing Citation Style\n\nBy John Smith\n\nAnother chapter about MLA style.",
        ]
        candidates = locate_chapter_start_candidates(pages, "Comparing Citation Styles", exclude_indices=set())
        self.assertEqual(len(candidates), 2)
        self.assertEqual({c.index for c in candidates}, {0, 5})
        # Sorted best-first.
        self.assertGreaterEqual(candidates[0].score, candidates[1].score)

    def test_running_headers_are_stripped_before_matching(self):
        # A book that stamps every page with a long running header (page
        # number + publisher + book title) would otherwise fill the whole
        # head window locate_chapter_start scores against, hiding the actual
        # chapter title (found empirically on a real evaluation book whose
        # ~150-character header made every chapter unlocatable). The header
        # varies only in its page number, so digit-insensitive detection
        # recognizes it on every page.
        def page(n: int, body: str) -> str:
            return (
                f"{n} | Presses de l'exemple, 2017. <http://www.example.fr/presses/>\n"
                "Accueillir des publics divers. Une perspective de bibliothèque\n"
                + body
            )

        pages = [page(i + 1, "Ordinary body text for this page, nothing special here at all.") for i in range(10)]
        pages[6] = page(7, "Comparing Citation Styles\npar Jane Doe\n\nThis chapter examines citation styles.")
        match = locate_chapter_start(pages, "Comparing Citation Styles", exclude_indices=set())
        self.assertIsNotNone(match)
        self.assertEqual(match.index, 6)

    def test_author_confirmation_resolves_ambiguous_tie(self):
        # Same near-tie shape as test_rejects_ambiguous_tie, but the true
        # chapter's page also has its author's last name near the top --
        # locate_chapter_start_candidates flags that cluster author_confirmed,
        # and the bonus lets it beat an equally-scoring, unconfirmed rival
        # that would otherwise make the match ambiguous.
        pages = [
            "Comparing Citation Styles\n\nBy Jane Doe\n\nThis chapter examines APA style only.",
            "Unrelated filler page one.",
            "Unrelated filler page two.",
            "Unrelated filler page three.",
            "Unrelated filler page four.",
            "Comparing Citation Style\n\nBy John Smith\n\nAnother chapter about MLA style.",
        ]
        match = locate_chapter_start(pages, "Comparing Citation Styles", exclude_indices=set(), authors=("Jane Doe",))
        self.assertIsNotNone(match)
        self.assertEqual(match.index, 0)
        self.assertTrue(match.author_confirmed)


class TestMatchConfidence(unittest.TestCase):
    def test_uncontested_high_score_is_near_ceiling(self):
        # Uncontested match (margin == score) at a perfect score should be
        # the most trustworthy case there is.
        self.assertEqual(match_confidence(score=100.0, margin=100.0), 1.0)

    def test_bare_minimum_match_sits_near_the_low_end(self):
        # Just clearing the score threshold (80) with the bare minimum
        # margin (8, the smallest a contested match can have) is the lowest
        # confidence value actually reachable in practice (~0.62 -- the
        # nominal 0.5 floor is a hard bound, never actually hit), reflecting
        # a real rival almost as strong as the winner.
        confidence = match_confidence(score=80.0, margin=8.0)
        self.assertGreaterEqual(confidence, 0.5)
        self.assertLess(confidence, 0.65)

    def test_higher_margin_increases_confidence_at_same_score(self):
        low_margin = match_confidence(score=90.0, margin=8.0)
        high_margin = match_confidence(score=90.0, margin=20.0)
        self.assertLess(low_margin, high_margin)

    def test_higher_score_increases_confidence_at_same_margin(self):
        low_score = match_confidence(score=80.0, margin=15.0)
        high_score = match_confidence(score=100.0, margin=15.0)
        self.assertLess(low_score, high_score)


class TestExtractPrintedPageNumber(unittest.TestCase):
    def test_finds_arabic_footer_number(self):
        text = "Comparing Citation Styles\nBy Jane Author\n\nBody text here.\n\n45"
        self.assertEqual(extract_printed_page_number(text), "45")

    def test_finds_roman_numeral_header(self):
        text = "xii\nPreface\n\nBody text of the preface."
        self.assertEqual(extract_printed_page_number(text), "xii")

    def test_returns_none_when_no_number_present(self):
        text = "Just a page of prose with no isolated numeral line at all here."
        self.assertIsNone(extract_printed_page_number(text))

    def test_finds_embedded_trailing_number_on_first_line(self):
        text = "Comparing Citation Styles 12\nBody text of this page follows here."
        self.assertEqual(extract_printed_page_number(text), "12")

    def test_finds_embedded_leading_number_on_first_line(self):
        text = "12 Comparing Citation Styles\nBody text of this page follows here."
        self.assertEqual(extract_printed_page_number(text), "12")

    def test_does_not_misread_trailing_letter_of_ordinary_word_as_roman_numeral(self):
        text = "Afterword\nBody text of this page follows here, with no real page number."
        self.assertIsNone(extract_printed_page_number(text))

    def test_skips_url_line_when_looking_for_embedded_number(self):
        text = "https://doi.org/10.1007/978-3-030-12345-6\nComparing Citation Styles 12\nBody text follows."
        self.assertEqual(extract_printed_page_number(text), "12")

    def test_does_not_match_embedded_number_on_an_overly_long_first_line(self):
        long_line = "A very long running header line that goes on and on and on and on and on and on and on and on and on and continues further 12"
        self.assertTrue(len(long_line) >= 120)
        text = long_line + "\nBody text follows."
        self.assertIsNone(extract_printed_page_number(text))

    def test_does_not_match_number_glued_directly_to_adjacent_text(self):
        self.assertIsNone(extract_printed_page_number("Section12\nBody text follows."))
        self.assertIsNone(extract_printed_page_number("12Comparing Citation Styles\nBody text follows."))


class TestPageNumberAnchors(unittest.TestCase):
    def test_empty_pages_returns_no_anchors(self):
        self.assertEqual(_page_number_anchors([]), [])

    def test_finds_arabic_and_roman_anchors(self):
        pages = ["No number here at all.", "45", "xii"]
        self.assertEqual(_page_number_anchors(pages), [(1, 45, False), (2, 12, True)])

    def test_pages_with_no_readable_number_contribute_no_anchor(self):
        pages = ["Ordinary prose with no page number visible anywhere on it."]
        self.assertEqual(_page_number_anchors(pages), [])


class TestInferPrintedPage(unittest.TestCase):
    def test_interpolates_when_anchors_agree(self):
        anchors = [(5, 10, False), (9, 14, False)]  # offset +5 on both sides
        self.assertEqual(_infer_printed_page(7, anchors), "12")

    def test_returns_none_when_anchors_disagree(self):
        anchors = [(5, 10, False), (9, 20, False)]  # offsets +5 and +11
        self.assertIsNone(_infer_printed_page(7, anchors))

    def test_returns_none_when_gap_exceeds_max_on_one_side(self):
        anchors = [(18, 23, False), (35, 40, False)]
        self.assertIsNone(_infer_printed_page(20, anchors))  # "after" anchor is 15 pages away

    def test_returns_none_with_only_one_side_present(self):
        anchors = [(5, 10, False)]
        self.assertIsNone(_infer_printed_page(7, anchors))

    def test_rejects_roman_before_arabic_after_scheme_change(self):
        anchors = [(5, 8, True), (9, 3, False)]  # roman "viii" then arabic "3" -- offsets don't align
        self.assertIsNone(_infer_printed_page(7, anchors))

    def test_infers_roman_value_when_anchors_are_roman(self):
        anchors = [(3, 5, True), (7, 9, True)]  # offset +2 on both sides, roman zone
        self.assertEqual(_infer_printed_page(5, anchors), "vii")


class TestToRoman(unittest.TestCase):
    def test_renders_known_values(self):
        self.assertEqual(_to_roman(1), "i")
        self.assertEqual(_to_roman(4), "iv")
        self.assertEqual(_to_roman(9), "ix")
        self.assertEqual(_to_roman(14), "xiv")
        self.assertEqual(_to_roman(49), "xlix")

    def test_round_trips_through_parse_toc_page_number(self):
        for n in range(1, 50):
            self.assertEqual(_parse_toc_page_number(_to_roman(n)), n)


class TestTocDeclaredPage(unittest.TestCase):
    def test_valid_heuristic_value_formats_as_arabic(self):
        entry = TocEntry(title="Introduction", printed_page_number=12, source_page_index=0)
        self.assertEqual(_toc_declared_page(entry, total_pages=200), "12")

    def test_valid_roman_value_formats_as_roman(self):
        entry = TocEntry(title="Foreword", printed_page_number=7, source_page_index=0, printed_roman=True)
        self.assertEqual(_toc_declared_page(entry, total_pages=200), "vii")

    def test_llm_sentinel_returns_none(self):
        entry = TocEntry(title="Introduction", printed_page_number=-1, source_page_index=-1)
        self.assertIsNone(_toc_declared_page(entry, total_pages=200))

    def test_implausibly_large_value_returns_none(self):
        # Simulates an LLM hallucination -- a positive int the heuristic
        # regex parser could never produce (find_toc_candidates enforces
        # this same ceiling at parse time), so _toc_declared_page must
        # enforce it independently for LLM-sourced entries.
        entry = TocEntry(title="Introduction", printed_page_number=5000, source_page_index=-1)
        self.assertIsNone(_toc_declared_page(entry, total_pages=200))

    def test_roman_value_exceeding_roman_ceiling_returns_none(self):
        # Clears the ratio ceiling (1000 <= 600*2.0=1200) but exceeds
        # _ROMAN_PAGE_MAX_VALUE (50) -- a roman numeral is never
        # realistically this large regardless of book length, and letting
        # it through would produce a string _parse_toc_page_number itself
        # rejects (a round-trip break later callers rely on not happening).
        entry = TocEntry(title="Foreword", printed_page_number=1000, source_page_index=-1, printed_roman=True)
        self.assertIsNone(_toc_declared_page(entry, total_pages=600))


class TestFallbackEndPrinted(unittest.TestCase):
    def _located(self, indices: list[int]) -> list[tuple[TocEntry, ChapterStartMatch]]:
        return [
            (TocEntry(title=f"Chapter {n}", printed_page_number=-1, source_page_index=-1),
             ChapterStartMatch(index=idx, score=100.0, margin=20.0))
            for n, idx in enumerate(indices)
        ]

    def test_uses_page_before_next_chapters_start(self):
        pages = ["chapter one body", "chapter one body", "45", "chapter two body"]
        located = self._located([0, 3])
        anchors = _page_number_anchors(pages)
        result = _fallback_end_printed(0, located, total_pages=4, pages=pages, anchors=anchors, start_printed="43")
        self.assertEqual(result, "45")

    def test_falls_back_to_last_page_for_final_chapter(self):
        pages = ["chapter one body", "chapter one body", "chapter one body", "50"]
        located = self._located([0])
        anchors = _page_number_anchors(pages)
        result = _fallback_end_printed(0, located, total_pages=4, pages=pages, anchors=anchors, start_printed="47")
        self.assertEqual(result, "50")

    def test_rejects_fallback_smaller_than_start(self):
        pages = ["chapter one body", "chapter one body", "3", "chapter two body"]
        located = self._located([0, 3])
        anchors = _page_number_anchors(pages)
        result = _fallback_end_printed(0, located, total_pages=4, pages=pages, anchors=anchors, start_printed="43")
        self.assertIsNone(result)

    def test_rejects_fallback_with_different_numbering_scheme(self):
        pages = ["chapter one body", "chapter one body", "vii", "chapter two body"]
        located = self._located([0, 3])
        anchors = _page_number_anchors(pages)
        # start_printed="5" (arabic, value 5) vs. fallback raw "vii" (roman,
        # value 7): 7 >= 5 so the ordering check alone would pass -- only
        # the isdigit() scheme-mismatch guard rejects this one.
        result = _fallback_end_printed(0, located, total_pages=4, pages=pages, anchors=anchors, start_printed="5")
        self.assertIsNone(result)

    def test_returns_none_when_fallback_page_unresolvable(self):
        pages = ["chapter one body", "chapter one body", "chapter one body, no number here", "chapter two body"]
        located = self._located([0, 3])
        anchors = _page_number_anchors(pages)
        result = _fallback_end_printed(0, located, total_pages=4, pages=pages, anchors=anchors, start_printed="43")
        self.assertIsNone(result)


class TestExtractAuthorsNear(unittest.TestCase):
    def test_finds_person_entities_at_chapter_start(self):
        # "By" is needed as an authorial cue -- spaCy's small model doesn't
        # reliably tag bare names without it
        text = "Comparing Citation Styles\n\nBy Jane Author and John Smith\n\nThis chapter examines APA and MLA styles."
        authors = extract_authors_near(text)
        self.assertIn("Jane Author", authors)
        self.assertIn("John Smith", authors)

    def test_returns_empty_list_when_no_names_found(self):
        # Full names avoid a known false positive: spaCy's small model can
        # misclassify bare acronyms like "APA"/"MLA" as PERSON entities.
        text = "This chapter examines American Psychological Association and Modern Language Association citation styles in detail."
        authors = extract_authors_near(text)
        self.assertEqual(authors, [])


class TestAnalyzeAttachment(unittest.TestCase):
    def _fake_book_pages(self) -> list[str]:
        return [
            # page 0: TOC. A third entry ("Appendix") is included solely so
            # this page meets find_toc_candidates' _TOC_MIN_LINES_PER_PAGE
            # (a real TOC page has several entries; below that count, the
            # page's lines are now discarded as noise) -- no page below
            # actually contains "Appendix" text, so locate_chapter_start
            # simply never finds it and it's silently dropped, same as any
            # other unlocatable TOC entry.
            "CONTENTS\n"
            "Introduction ..... 1\n"
            "Comparing Citation Styles ..... 3\n"
            "Appendix ..... 5\n",
            # page 1: printed page "1" — Introduction body
            "Introduction\nJane Author\n\nThis book explores reference management.\n\n1",
            # page 2: continuation of Introduction, printed page "2"
            # NOTE: deliberately avoids repeating the word "introduction"
            # here -- with locate_chapter_start's ambiguity-margin guard, a
            # continuation page that happens to literally contain the
            # chapter title word ties with the real chapter-start page
            # (both score 100 via rapidfuzz partial_ratio) and neither can
            # be trusted, which is correct behavior but not what this
            # fixture is testing. Padded past 150 stripped characters so
            # analyze_attachment's trailing-blank-page trim (meant for real
            # divider pages) doesn't mistake this genuine body-text
            # continuation page for one and clip the chapter short.
            "...continued text follows here, with enough body content on this "
            "page that it clearly reads as a real continuation of the "
            "chapter rather than a blank divider page between sections.\n\n2",
            # page 3: printed page "3" — chapter start
            # NOTE: a blank line separates the title from "John Smith" here
            # (unlike the spec's literal single "\n"). Verified against the
            # real en_core_web_sm model: with only a single newline, spaCy's
            # NER merges the preceding title words and the name into one
            # PERSON span ("Citation Styles\nJohn Smith") because there is no
            # sentence boundary between them, so "John Smith" alone is never
            # produced and the assertIn assertion in
            # test_authors_attached_to_chapters fails. A blank line (already
            # the pattern used in TestExtractAuthorsNear's passing test)
            # gives spaCy a clear boundary and "John Smith" is extracted as
            # its own entity. This is the smallest change that makes the
            # literal spec assertion pass against the real model.
            "Comparing Citation Styles\n\nJohn Smith\n\nThis chapter examines APA and MLA.\n\n3",
            # page 4: continuation, printed page "4". Padded past 150
            # stripped characters for the same reason as page 2 above.
            "...continued chapter text, with enough body content on this "
            "final page that it clearly reads as a real continuation of "
            "the chapter rather than a blank divider page.\n\n4",
        ]

    def test_detects_two_chapters_with_pdf_indices(self):
        result = analyze_attachment(self._fake_book_pages())
        self.assertEqual(len(result["chapters"]), 2)
        first, second = result["chapters"]
        self.assertEqual(first["pdf_start_index"], 1)
        self.assertEqual(first["pdf_end_index"], 2)
        self.assertEqual(second["pdf_start_index"], 3)
        self.assertEqual(second["pdf_end_index"], 4)

    def test_citation_pages_extracted_when_present(self):
        result = analyze_attachment(self._fake_book_pages())
        self.assertEqual(result["chapters"][0]["citation_pages"], "1-2")
        self.assertEqual(result["chapters"][1]["citation_pages"], "3-4")

    def test_low_confidence_when_no_toc_found(self):
        result = analyze_attachment(["Just some prose with no discernible TOC or chapter structure."])
        self.assertEqual(result["segmentation_confidence"], "low")
        self.assertEqual(result["chapters"], [])

    def test_authors_attached_to_chapters(self):
        result = analyze_attachment(self._fake_book_pages())
        self.assertIn("John Smith", result["chapters"][1]["authors"])

    def test_chapters_default_to_heuristic_source(self):
        result = analyze_attachment(self._fake_book_pages())
        self.assertTrue(all(c["source"] == "heuristic" for c in result["chapters"]))

    def test_part_dividers_and_back_matter_bound_but_are_not_chapters(self):
        # "Part I ..." divider pages and standard back-matter sections
        # (Index, Contributors) are located -- so they bound their
        # neighbors' page ranges -- but never emitted as chapters.
        # Filler is varied per page (by words, not digits -- running-header
        # detection is digit-insensitive): byte-identical opening lines
        # across many pages would (correctly) be detected as a running
        # header and stripped, which is not what this test is about.
        def filler(n: int) -> str:
            topic = ["archives", "borrowing", "cataloguing", "digitisation", "editions",
                     "facsimiles", "gazettes", "holdings", "incunabula", "journals"][n]
            return (f"This page carries plenty of ordinary body text about {topic}, continuing "
                    f"the chapter well past the blank-page threshold and discussing {topic} "
                    "in enough detail that nothing gets trimmed by accident. ") * 2

        pages = [
            "CONTENTS\n"
            "Introduction to Reference Management ..... 1\n"
            "Part I Foundations of the Field ..... 3\n"
            "Comparing Citation Styles ..... 5\n"
            "Index ..... 9\n",
            "Introduction to Reference Management\nBy Jane Author\n\n" + filler(1),  # 1
            filler(2) + "\n2",  # 2
            "Part I Foundations of the Field",  # 3 -- divider page
            filler(4) + "\n4",  # 4 (padding)
            "Comparing Citation Styles\n\nBy John Smith\n\n" + filler(5),  # 5
            filler(6) + "\n6",  # 6
            filler(7) + "\n7",  # 7
            filler(8) + "\n8",  # 8
            "Index\n\naardvark, 12\nzotero, 45\n" + filler(9),  # 9
        ]
        result = analyze_attachment(pages)
        titles = [c["title"] for c in result["chapters"]]
        self.assertNotIn("Part I Foundations of the Field", titles)
        self.assertNotIn("Index", titles)
        ranges = {c["title"]: (c["pdf_start_index"], c["pdf_end_index"]) for c in result["chapters"]}
        # The divider bounds the Introduction's end; the Index bounds the
        # second chapter's end.
        self.assertEqual(ranges["Introduction to Reference Management"], (1, 2))
        self.assertEqual(ranges["Comparing Citation Styles"], (5, 8))

    def test_toc_order_resolves_shared_title_suffix_ambiguity(self):
        # Introduction and Conclusion sharing one distinctive suffix (the
        # book's own title) are individually ambiguous -- each matches both
        # pages -- but TOC order pins the Introduction to the earlier page
        # and the Conclusion to the later one.
        filler = ("Plenty of ordinary body text continues the chapter here, "
                  "well past the blank-page threshold so nothing is trimmed. ") * 3
        pages = [
            "CONTENTS\n"
            "Introduction: Transformations of Reference Management ..... 1\n"
            "A Middle Chapter About Citation Tools ..... 5\n"
            "Conclusion: Transformations of Reference Management ..... 9\n",
            "Introduction: Transformations of Reference Management\n\nBy Jane Author\n\n" + filler,  # 1
            filler + "\n2",
            filler + "\n3",
            filler + "\n4",
            "A Middle Chapter About Citation Tools\n\nBy John Smith\n\n" + filler,  # 5
            filler + "\n6",
            filler + "\n7",
            filler + "\n8",
            "Conclusion: Transformations of Reference Management\n\nBy Jane Author\n\n" + filler,  # 9
            filler + "\n10",
        ]
        result = analyze_attachment(pages)
        ranges = {c["title"]: c["pdf_start_index"] for c in result["chapters"]}
        self.assertEqual(ranges.get("Introduction: Transformations of Reference Management"), 1)
        self.assertEqual(ranges.get("Conclusion: Transformations of Reference Management"), 9)


class TestAnalyzeAttachmentWithLlmFallback(unittest.IsolatedAsyncioTestCase):
    def _fake_llm(self, toc_response: str | None = None, disambiguation_response: str | None = None):
        llm = MagicMock()
        responses = [r for r in (toc_response, disambiguation_response) if r is not None]
        llm.generate = AsyncMock(side_effect=responses)
        return llm

    async def test_llm_toc_extraction_fires_when_heuristic_finds_nothing(self):
        # 20 pages total so the front/back scan zones (15%/5% of the page
        # count, same fractions find_toc_candidates and llm_extract_toc_entries
        # both use) don't accidentally overlap the two real chapters' opening
        # pages, which are placed well inside the body.
        filler = "Unrelated body filler text, nothing chapter-related in this passage at all."
        pages = [
            "Front matter with an irregular listing the regex can't parse: "
            "Introduction (Jane Author) ... Comparing Citation Styles (John Smith)",
            "Filler front-matter page, nothing chapter-like here at all.",
            "Filler front-matter page, nothing chapter-like here at all.",
            *([filler] * 7),  # indices 3-9
            "Introduction\n\nJane Author\n\nThis book explores reference management in depth.",  # index 10
            *([filler] * 4),  # indices 11-14
            "Comparing Citation Styles\n\nJohn Smith\n\nThis chapter examines APA and MLA styles.",  # index 15
            *([filler] * 3),  # indices 16-18
            "Back matter index page, nothing chapter-related here.",  # index 19
        ]
        self.assertEqual(len(pages), 20)
        response = (
            '[{"title": "Introduction", "authors": ["Jane Author"], "printed_page_number": 1}, '
            '{"title": "Comparing Citation Styles", "authors": ["John Smith"], "printed_page_number": 15}]'
        )
        llm = self._fake_llm(toc_response=response)
        result = await analyze_attachment_with_llm_fallback(pages, llm)
        self.assertTrue(result["diagnostics"]["llm_toc_extraction_used"])
        self.assertEqual(len(result["chapters"]), 2)
        self.assertTrue(all(c["source"] == "llm" for c in result["chapters"]))
        self.assertEqual(result["chapters"][0]["pdf_start_index"], 10)
        self.assertEqual(result["chapters"][1]["pdf_start_index"], 15)

    async def test_does_not_call_llm_when_heuristic_already_succeeds(self):
        pages = [
            "CONTENTS\n"
            "Introduction ..... 1\n"
            "Comparing Citation Styles ..... 3\n"
            "Appendix ..... 5\n",
            "Introduction\nJane Author\n\nThis book explores reference management.\n\n1",
            "...continued text follows here, with enough body content on this "
            "page that it clearly reads as a real continuation of the "
            "chapter rather than a blank divider page between sections.\n\n2",
            "Comparing Citation Styles\n\nJohn Smith\n\nThis chapter examines APA and MLA.\n\n3",
            "...continued chapter text, with enough body content on this "
            "final page that it clearly reads as a real continuation of "
            "the chapter rather than a blank divider page.\n\n4",
        ]
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        result = await analyze_attachment_with_llm_fallback(pages, llm)
        self.assertFalse(result["diagnostics"]["llm_toc_extraction_used"])
        self.assertEqual(result["diagnostics"]["llm_disambiguation_used"], 0)
        llm.generate.assert_not_called()

    async def test_llm_disambiguation_resolves_ambiguous_chapter(self):
        # An entry ambiguous between two locations is normally resolved
        # heuristically by TOC-order constraints (see _locate_toc_entries's
        # second pass), so the LLM path only fires when ordering CANNOT
        # help: here the TOC lists "Comparing Citation Styles" between two
        # entries whose located pages (3 and 4) leave no room, while its
        # own two candidate pages (6 and 12) both sit outside that interval
        # -- a disordered/incorrect TOC only the LLM can arbitrate.
        filler = "Unrelated body filler text, nothing chapter-related here."
        pages = [
            "CONTENTS\n"
            "Alpha Overview ..... 1\n"
            "Comparing Citation Styles ..... 5\n"
            "Omega Summary ..... 9\n",
            filler,
            filler,
            "Alpha Overview\n\nBy Jane Author\n\nThis opening chapter surveys the field.",  # index 3
            "Omega Summary\n\nBy Jane Author\n\nThis closing chapter wraps everything up.",  # index 4
            filler,
            "Comparing Citation Styles\n\nBy Jane Doe\n\nThis chapter examines APA style only.",  # index 6
            *([filler] * 5),  # indices 7-11
            "Comparing Citation Style\n\nBy John Smith\n\nAnother chapter about MLA style.",  # index 12
            *([filler] * 7),  # indices 13-19
        ]
        self.assertEqual(len(pages), 20)
        llm = self._fake_llm(disambiguation_response='{"chosen_candidate": 1}')
        result = await analyze_attachment_with_llm_fallback(pages, llm)
        self.assertFalse(result["diagnostics"]["llm_toc_extraction_used"])
        self.assertEqual(result["diagnostics"]["llm_disambiguation_used"], 1)
        sources = {c["title"]: c["source"] for c in result["chapters"]}
        self.assertEqual(sources.get("Comparing Citation Styles"), "llm")


class TestChaptersFromLocated(unittest.TestCase):
    def test_entry_source_maps_llm_entries_and_defaults_others_to_heuristic(self):
        pages = [
            "Introduction\nJane Author\n\nBody text.\n\n1",
            "Comparing Citation Styles\n\nJohn Smith\n\nBody text.\n\n2",
        ]
        llm_entry = TocEntry(title="Introduction", printed_page_number=1, source_page_index=-1)
        heuristic_entry = TocEntry(title="Comparing Citation Styles", printed_page_number=2, source_page_index=0)
        located = [
            (llm_entry, ChapterStartMatch(index=0, score=100.0, margin=20.0)),
            (heuristic_entry, ChapterStartMatch(index=1, score=100.0, margin=20.0)),
        ]
        chapters = _chapters_from_located(pages, located, entry_source={llm_entry: "llm"})
        sources = {c["title"]: c["source"] for c in chapters}
        self.assertEqual(sources["Introduction"], "llm")
        self.assertEqual(sources["Comparing Citation Styles"], "heuristic")


class TestChaptersFromLocatedPageNumberPriority(unittest.TestCase):
    _FILLER = (
        "This page carries plenty of ordinary body text so that it is not "
        "mistaken for a blank or divider page during trimming, comfortably "
        "exceeding the minimum character threshold used by the heuristic. "
    )

    def test_toc_declared_start_wins_over_on_page_scanning(self):
        # Page 0's own on-page text would extract "99" if scanned directly
        # -- but the TOC-declared printed_page_number (3) must win, since
        # it's checked first and is the authoritative source. (Kept small,
        # not 12: _toc_declared_page's plausibility ceiling is total_pages
        # * _TOC_MAX_PAGE_NUMBER_RATIO(2.0) -- with only 2 pages in this
        # fixture, a value above 4 would be (correctly) rejected as
        # implausible before this test could even exercise the priority
        # chain it's meant to check.)
        pages = [self._FILLER + "\n\n99", self._FILLER + "\n\n4"]
        entry = TocEntry(title="Introduction", printed_page_number=3, source_page_index=0)
        located = [(entry, ChapterStartMatch(index=0, score=100.0, margin=20.0))]
        chapters = _chapters_from_located(pages, located)
        self.assertEqual(chapters[0]["citation_pages"], "3-4")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "high")

    def test_llm_sentinel_falls_through_to_on_page_extraction(self):
        pages = [self._FILLER + "\n\n12", self._FILLER + "\n\n13"]
        entry = TocEntry(title="Introduction", printed_page_number=-1, source_page_index=-1)
        located = [(entry, ChapterStartMatch(index=0, score=100.0, margin=20.0))]
        chapters = _chapters_from_located(pages, located)
        self.assertEqual(chapters[0]["citation_pages"], "12-13")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "high")

    def test_llm_sentinel_falls_through_to_anchor_interpolation(self):
        # The start page (index 1) has no printed number of its own;
        # neighboring pages bracket it with a consistent +11 offset.
        pages = [
            self._FILLER + "\n\n11",
            self._FILLER,  # no number -- must be inferred
            self._FILLER + "\n\n13",
        ]
        entry = TocEntry(title="Introduction", printed_page_number=-1, source_page_index=-1)
        located = [(entry, ChapterStartMatch(index=1, score=100.0, margin=20.0))]
        chapters = _chapters_from_located(pages, located)
        self.assertEqual(chapters[0]["citation_pages"], "12-13")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "inferred")

    def test_end_derived_from_next_entrys_toc_declared_value(self):
        pages = [self._FILLER, self._FILLER, self._FILLER]
        first = TocEntry(title="Introduction", printed_page_number=1, source_page_index=0)
        second = TocEntry(title="Comparing Citation Styles", printed_page_number=5, source_page_index=1)
        located = [
            (first, ChapterStartMatch(index=0, score=100.0, margin=20.0)),
            (second, ChapterStartMatch(index=2, score=100.0, margin=20.0)),
        ]
        chapters = _chapters_from_located(pages, located)
        self.assertEqual(chapters[0]["citation_pages"], "1-4")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "inferred")

    def test_end_derivation_skipped_when_next_entry_zone_differs(self):
        # The last page's filler text is deliberately NOT byte-identical to
        # the other four's: five pages sharing one verbatim leading line
        # would otherwise trip the unrelated, pre-existing running-header
        # heuristic (_RUNNING_HEADER_MIN_PAGES=5), which strips that line
        # (and the bare page-number line right after it) before the
        # trailing-blank-page trim check -- collapsing page index 1's
        # trimmed length to zero and wrongly trimming it away, which is not
        # what this test means to exercise.
        pages = [
            self._FILLER + "\n\nvii",
            self._FILLER + "\n\n6",
            self._FILLER + "\n\n1",
            self._FILLER,
            self._FILLER + " This closing page has slightly different wording.",
        ]
        first = TocEntry(title="Foreword", printed_page_number=7, source_page_index=0, printed_roman=True)
        second = TocEntry(title="Introduction", printed_page_number=1, source_page_index=2)
        located = [
            (first, ChapterStartMatch(index=0, score=100.0, margin=20.0)),
            (second, ChapterStartMatch(index=2, score=100.0, margin=20.0)),
        ]
        chapters = _chapters_from_located(pages, located)
        # second.printed_roman (False) != first.printed_roman (True) -- the
        # fast "next.printed_page_number - 1" path is skipped entirely (it
        # would otherwise wrongly compute "0", 1 - 1, as if still roman).
        # Falls through to on-page extraction of the chapter's own
        # (trimmed) end page instead, which has "6" printed directly on it.
        self.assertEqual(chapters[0]["citation_pages"], "vii-6")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "high")

    def test_end_derivation_skipped_when_next_entry_is_page_one(self):
        # next_entry.printed_page_number - 1 == 0 must not produce an empty
        # roman string ("") that then silently slips past every "is None"
        # fallback check downstream. The fixture needs a real gap between
        # the two located entries (second's raw start at index 2, not 1) so
        # that the first chapter's trimmed end_index (1) lands on a
        # distinct page from its own start_index (0) -- otherwise the
        # fallthrough would just re-read the start page's own number and
        # the test couldn't tell a correct fallthrough from the bug
        # producing the same start/end value by coincidence.
        pages = [
            self._FILLER + "\n\niii",
            self._FILLER + "\n\nii",
            self._FILLER + "\n\ni",
        ]
        first = TocEntry(title="Foreword", printed_page_number=3, source_page_index=0, printed_roman=True)
        second = TocEntry(title="Preface", printed_page_number=1, source_page_index=1, printed_roman=True)
        located = [
            (first, ChapterStartMatch(index=0, score=100.0, margin=20.0)),
            (second, ChapterStartMatch(index=2, score=100.0, margin=20.0)),
        ]
        chapters = _chapters_from_located(pages, located)
        self.assertEqual(chapters[0]["citation_pages"], "iii-ii")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "high")

    def test_fallback_end_used_when_direct_extraction_and_interpolation_both_fail(self):
        pages = [
            self._FILLER + "\n\n12",  # chapter 1 start, own number readable directly
            self._FILLER,              # chapter 1's real (post-trim) end page -- no number
            "Part II\n\n20",            # short divider page, trimmed off the range but still
                                        # readable -- this is what _fallback_end_printed uses
            self._FILLER + "\n\n21",   # chapter 2's raw start
        ]
        first = TocEntry(title="Introduction", printed_page_number=-1, source_page_index=-1)
        second = TocEntry(title="Comparing Citation Styles", printed_page_number=-1, source_page_index=-1)
        located = [
            (first, ChapterStartMatch(index=0, score=100.0, margin=20.0)),
            (second, ChapterStartMatch(index=3, score=100.0, margin=20.0)),
        ]
        chapters = _chapters_from_located(pages, located)
        self.assertEqual(chapters[0]["pdf_end_index"], 1)  # trimmed past the divider page
        self.assertEqual(chapters[0]["citation_pages"], "12-20")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "inferred")

    def test_unmappable_when_nothing_resolves(self):
        pages = [self._FILLER, self._FILLER]
        entry = TocEntry(title="Introduction", printed_page_number=-1, source_page_index=-1)
        located = [(entry, ChapterStartMatch(index=0, score=100.0, margin=20.0))]
        chapters = _chapters_from_located(pages, located)
        self.assertIsNone(chapters[0]["citation_pages"])
        self.assertEqual(chapters[0]["page_mapping_confidence"], "unmappable")


class TestAnalysisCache(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entry = {"item_key": "B1", "attachment_key": "A1", "chapters": []}
            save_analysis_cache(cache_dir, "B1", "A1", 5, "heuristic", entry)
            result = load_cached_analysis(cache_dir, "B1", "A1", 5, "heuristic")
            self.assertEqual(result, entry)

    def test_returns_none_when_not_cached(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(load_cached_analysis(Path(tmp), "B1", "A1", 5, "heuristic"))

    def test_different_version_is_a_cache_miss(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            save_analysis_cache(cache_dir, "B1", "A1", 5, "heuristic", {"chapters": []})
            self.assertIsNone(load_cached_analysis(cache_dir, "B1", "A1", 6, "heuristic"))

    def test_different_mode_is_a_cache_miss(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            save_analysis_cache(cache_dir, "B1", "A1", 5, "heuristic", {"chapters": []})
            self.assertIsNone(load_cached_analysis(cache_dir, "B1", "A1", 5, "llm_fallback"))


# Fake page sets for extract_page_texts_for_analysis. 10 pages total, so
# find_toc_candidates' front window (15%) is exactly page 0 and printed page
# numbers up to 2*10=20 are plausible. The "default mode" pages are healthy
# multi-line text with NO TOC-shaped lines; the "layout mode" pages carry a
# classic 3-entry dot-leader TOC on page 0.
_BODY_PAGE = "Dies ist eine gewoehnliche Textzeile ohne Nummer\n" * 35
_DEFAULT_MODE_PAGES = [_BODY_PAGE] * 10
_LAYOUT_MODE_PAGES = [
    "Inhalt\n"
    "Erstes Kapitel .......... 5\n"
    "Zweites Kapitel .......... 9\n"
    "Drittes Kapitel .......... 15\n"
] + [_BODY_PAGE] * 9
_TOC_DEFAULT_PAGES = [_LAYOUT_MODE_PAGES[0]] + [_BODY_PAGE] * 9


class TestExtractPageTextsForAnalysis(unittest.TestCase):
    """The layout fallback must only fire when default-mode extraction hides
    the TOC: default-found TOC -> default pages untouched (protects the
    existing 7-book baseline); no TOC either way -> default pages; OCR-shaped
    input -> default pages without even attempting the (slow) layout pass."""

    def test_keeps_default_pages_when_default_mode_finds_toc(self):
        def fake_extract(content, layout=False):
            if layout:
                raise AssertionError("layout extraction must not run when default mode already finds a TOC")
            return _TOC_DEFAULT_PAGES

        with unittest.mock.patch("chapter_segmentation.segmentation.extract_page_texts_from_pdf_bytes", side_effect=fake_extract):
            pages, layout_used = extract_page_texts_for_analysis(b"%PDF-fake")
        self.assertEqual(pages, _TOC_DEFAULT_PAGES)
        self.assertFalse(layout_used)

    def test_falls_back_to_layout_pages_when_only_layout_mode_finds_toc(self):
        def fake_extract(content, layout=False):
            return _LAYOUT_MODE_PAGES if layout else _DEFAULT_MODE_PAGES

        with unittest.mock.patch("chapter_segmentation.segmentation.extract_page_texts_from_pdf_bytes", side_effect=fake_extract):
            pages, layout_used = extract_page_texts_for_analysis(b"%PDF-fake")
        self.assertEqual(pages, _LAYOUT_MODE_PAGES)
        self.assertTrue(layout_used)

    def test_keeps_default_pages_when_neither_mode_finds_toc(self):
        def fake_extract(content, layout=False):
            return _DEFAULT_MODE_PAGES

        with unittest.mock.patch("chapter_segmentation.segmentation.extract_page_texts_from_pdf_bytes", side_effect=fake_extract):
            pages, layout_used = extract_page_texts_for_analysis(b"%PDF-fake")
        self.assertEqual(pages, _DEFAULT_MODE_PAGES)
        self.assertFalse(layout_used)

    def test_skips_layout_attempt_entirely_for_ocr_shaped_input(self):
        def fake_extract(content, layout=False):
            if layout:
                raise AssertionError("layout extraction must not run for pages that need OCR")
            return [""] * 300

        with unittest.mock.patch("chapter_segmentation.segmentation.extract_page_texts_from_pdf_bytes", side_effect=fake_extract):
            pages, layout_used = extract_page_texts_for_analysis(b"%PDF-fake")
        self.assertEqual(pages, [""] * 300)
        self.assertFalse(layout_used)


class TestPagesNeedOcr(unittest.TestCase):
    """pages_need_ocr must catch three real failure shapes found in the
    evaluation set (see evaluation/README.md):
    scans with no text layer, scans with a trivial amount of stray text,
    and PDFs whose text layer extracts as one giant line per page."""

    def _healthy_page(self) -> str:
        # ~35 lines of ~40 chars: realistic body page, plenty of newlines.
        return ("Dies ist eine gewoehnliche Textzeile ohne Nummer\n" * 35)

    def test_empty_page_list_needs_ocr(self):
        self.assertTrue(pages_need_ocr([]))

    def test_scan_without_text_layer_needs_ocr(self):
        self.assertTrue(pages_need_ocr([""] * 300))

    def test_scan_with_trivial_stray_text_needs_ocr(self):
        # Mirrors dnb-36942798X.pdf: 411 pages, only 2 carry any text
        # (2,366 chars total) -- more than the old >100-chars check allowed,
        # but obviously still an un-OCR'd scan.
        pages = [""] * 409 + [self._healthy_page()] * 2
        self.assertTrue(pages_need_ocr(pages))

    def test_normal_book_does_not_need_ocr(self):
        self.assertFalse(pages_need_ocr([self._healthy_page()] * 40))

    def test_one_giant_line_pages_need_ocr(self):
        # Mirrors 9783789057366.pdf / 9780367439712.pdf: pages have plenty
        # of text but essentially no newlines (one absolutely-positioned
        # run per page), so no line-oriented parsing can work.
        giant = "Wort " * 400  # ~2000 chars, zero newlines
        self.assertTrue(pages_need_ocr([giant] * 40))

    def test_few_degenerate_pages_among_healthy_ones_is_fine(self):
        # A handful of single-line pages (e.g. a part-divider printed
        # sideways) must not condemn a healthy book to OCR.
        pages = [self._healthy_page()] * 36 + ["Wort " * 400] * 4
        self.assertFalse(pages_need_ocr(pages))


class TestAnalyzeAttachmentOutlineOnly(unittest.TestCase):
    _TWO_CHAPTER_PAGES = [
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 0
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 1
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 2
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 3
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 4
        "Introduction\nJane Author\n\nBody text opening the chapter.\n\n1",  # 5
        "...continued introduction text with real body content here.\n\n2",  # 6
        "...more continued introduction text with real body content.\n\n3",  # 7
        "...final continued introduction text with real body content.\n\n4",  # 8
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 9
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 10
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 11
        "Comparing Citation Styles\n\nJohn Smith\n\nBody text opening this chapter.\n\n5",  # 12
        "...continued citation styles text with real body content here.\n\n6",  # 13
        "...more continued citation styles text with real body content.\n\n7",  # 14
        "...final continued citation styles text with real body content.\n\n8",  # 15
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 16
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 17
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 18
        "Unrelated body filler text, nothing chapter-related in this passage at all.",  # 19
    ]

    def test_builds_chapters_directly_from_resolved_candidates(self):
        candidates = [
            ChapterCandidate(title="Introduction", authors=("Jane Author",), pdf_page_index=5, source="outline"),
            ChapterCandidate(title="Comparing Citation Styles", authors=("John Smith",), pdf_page_index=12, source="outline"),
        ]
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, candidates)
        chapters = sorted(result["chapters"], key=lambda c: c["pdf_start_index"])
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "Introduction")
        self.assertEqual(chapters[0]["pdf_start_index"], 5)
        self.assertEqual(chapters[0]["source"], "outline")
        self.assertEqual(chapters[0]["confidence"], 0.98)
        self.assertEqual(chapters[1]["pdf_start_index"], 12)
        self.assertEqual(result["diagnostics"]["outline_candidates_found"], 2)

    def test_empty_candidates_yields_no_chapters(self):
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, [])
        self.assertEqual(result["chapters"], [])
        self.assertEqual(result["segmentation_confidence"], "low")

    def test_skips_candidate_missing_pdf_page_index(self):
        # Should never happen for a real extract_outline_candidates() result
        # (every entry it returns already has pdf_page_index resolved), but
        # must not crash if it does -- skip rather than guess.
        candidates = [
            ChapterCandidate(title="Introduction", pdf_page_index=5, source="outline"),
            ChapterCandidate(title="Undated", pdf_page_index=None, source="outline"),
        ]
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, candidates)
        titles = [c["title"] for c in result["chapters"]]
        self.assertEqual(titles, ["Introduction"])

    def test_matches_strategies_pipeline_outline_only_result(self):
        # Same fixture as
        # TestAnalyzeAttachmentWithStrategiesOutlineOnly.test_outline_only_uses_direct_localization_and_fixed_confidence
        # in tests/test_segmentation_strategies.py -- the standalone
        # function must agree with the pipeline's own outline-only branch.
        pdf_bytes = _pdf_with_outline(20, [("Introduction", 5), ("Comparing Citation Styles", 12)])
        candidates = extract_outline_candidates(pdf_bytes)
        result = analyze_attachment_outline_only(self._TWO_CHAPTER_PAGES, candidates)
        chapters = sorted(result["chapters"], key=lambda c: c["pdf_start_index"])
        self.assertEqual([c["pdf_start_index"] for c in chapters], [5, 12])


class TestAnalyzeAttachmentLlmOnly(unittest.IsolatedAsyncioTestCase):
    def _fake_llm(self, toc_response: str | None = None, disambiguation_response: str | None = None):
        llm = MagicMock()
        responses = [r for r in (toc_response, disambiguation_response) if r is not None]
        llm.generate = AsyncMock(side_effect=responses)
        return llm

    async def test_calls_llm_even_when_heuristic_would_succeed(self):
        # Same fixture as
        # TestAnalyzeAttachmentWithLlmFallback.test_does_not_call_llm_when_heuristic_already_succeeds
        # in this file -- a regex-parseable TOC exists, but
        # analyze_attachment_llm_only must call the LLM anyway, since it is
        # the standalone-LLM strategy, not the fallback pipeline.
        pages = [
            "CONTENTS\n"
            "Introduction ..... 1\n"
            "Comparing Citation Styles ..... 3\n"
            "Appendix ..... 5\n",
            "Introduction\nJane Author\n\nThis book explores reference management.\n\n1",
            "...continued text follows here, with enough body content on this "
            "page that it clearly reads as a real continuation of the "
            "chapter rather than a blank divider page between sections.\n\n2",
            "Comparing Citation Styles\n\nJohn Smith\n\nThis chapter examines APA and MLA.\n\n3",
            "...continued chapter text, with enough body content on this "
            "final page that it clearly reads as a real continuation of "
            "the chapter rather than a blank divider page.\n\n4",
        ]
        response = (
            '[{"title": "Introduction", "authors": ["Jane Author"], "printed_page_number": 1}, '
            '{"title": "Comparing Citation Styles", "authors": ["John Smith"], "printed_page_number": 3}]'
        )
        llm = self._fake_llm(toc_response=response)
        result = await analyze_attachment_llm_only(pages, llm)
        llm.generate.assert_called_once()
        self.assertEqual(len(result["chapters"]), 2)
        self.assertTrue(all(c["source"] == "llm" for c in result["chapters"]))

    async def test_empty_llm_response_yields_no_chapters(self):
        llm = self._fake_llm(toc_response="[]")
        pages = ["front matter"] * 20
        result = await analyze_attachment_llm_only(pages, llm)
        self.assertEqual(result["chapters"], [])
        self.assertEqual(result["diagnostics"]["toc_matches_found"], 0)

    async def test_swallows_llm_exception_and_returns_empty_result(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("network error"))
        pages = ["front matter"] * 20
        result = await analyze_attachment_llm_only(pages, llm)
        self.assertEqual(result["chapters"], [])

    async def test_llm_disambiguation_resolves_ambiguous_llm_entry(self):
        # Same fixture shape as
        # TestAnalyzeAttachmentWithLlmFallback.test_llm_disambiguation_resolves_ambiguous_chapter,
        # but the TOC entries themselves come from the LLM's own
        # extraction response (first generate() call) instead of the
        # regex heuristic, since this function never runs the heuristic
        # at all.
        filler = "Unrelated body filler text, nothing chapter-related here."
        pages = [
            "Front matter, no parseable TOC here at all.",
            filler,
            filler,
            "Alpha Overview\n\nBy Jane Author\n\nThis opening chapter surveys the field.",  # index 3
            "Omega Summary\n\nBy Jane Author\n\nThis closing chapter wraps everything up.",  # index 4
            filler,
            "Comparing Citation Styles\n\nBy Jane Doe\n\nThis chapter examines APA style only.",  # index 6
            *([filler] * 5),  # indices 7-11
            "Comparing Citation Style\n\nBy John Smith\n\nAnother chapter about MLA style.",  # index 12
            *([filler] * 7),  # indices 13-19
        ]
        self.assertEqual(len(pages), 20)
        toc_response = (
            '[{"title": "Alpha Overview", "authors": ["Jane Author"], "printed_page_number": 1}, '
            '{"title": "Comparing Citation Styles", "authors": [], "printed_page_number": 5}, '
            '{"title": "Omega Summary", "authors": ["Jane Author"], "printed_page_number": 9}]'
        )
        llm = self._fake_llm(toc_response=toc_response, disambiguation_response='{"chosen_candidate": 1}')
        result = await analyze_attachment_llm_only(pages, llm)
        self.assertEqual(result["diagnostics"]["llm_disambiguation_used"], 1)
        sources = {c["title"]: c["source"] for c in result["chapters"]}
        self.assertEqual(sources.get("Comparing Citation Styles"), "llm")


if __name__ == "__main__":
    unittest.main()
