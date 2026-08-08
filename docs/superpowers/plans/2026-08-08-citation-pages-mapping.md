# Improve citation_pages / page_mapping_confidence accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `citation_pages`/`page_mapping_confidence` (the printed-page-number metadata attached to each located chapter) actually usable, by preferring the already-parsed TOC-declared page number over fragile body-page text scanning, adding cross-page inference as a fallback, and fixing the LLM extraction schema's inability to represent roman numerals.

**Architecture:** All production changes live in `src/chapter_segmentation/segmentation.py`, layered as a priority chain feeding `_chapters_from_located`: (1) widen `extract_printed_page_number` to catch embedded (not just isolated) page numbers, (2) a new document-level anchor-interpolation helper (`_page_number_anchors`/`_infer_printed_page`) for pages with no number of their own, (3) a new `_toc_declared_page` helper that prefers `TocEntry.printed_page_number` (already parsed from the TOC by both the heuristic and the LLM) over all of the above, (4) a `_fallback_end_printed` last resort for an unresolvable end page, (5) wiring all four into `_chapters_from_located`'s existing loop with no signature changes. Separately, `llm_extract_toc_entries`'s prompt/parsing is fixed to preserve roman numerals instead of forcing an int. Finally, `evaluation/metrics.py` gains a `citation_pages_metrics` function (scored independently for start vs. end, with an over-inclusion tolerance on the end) so this stops being an untracked blind spot in `RESULTS.md`.

**Tech Stack:** Python 3.12, unittest.

Full design: `docs/superpowers/specs/2026-08-08-citation-pages-mapping-design.md`.

---

### Task 1: Widen `extract_printed_page_number`

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:937-950` (the current `extract_printed_page_number` function)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add these test methods to the existing `TestExtractPrintedPageNumber` class (`tests/test_segmentation.py:797`), right after `test_returns_none_when_no_number_present`:

```python
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
        long_line = "A very long running header line that goes on and on and on and on and on and on and on and on 12"
        self.assertTrue(len(long_line) >= 120)
        text = long_line + "\nBody text follows."
        self.assertIsNone(extract_printed_page_number(text))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k "test_finds_embedded_trailing_number_on_first_line or test_finds_embedded_leading_number_on_first_line or test_does_not_misread_trailing_letter_of_ordinary_word_as_roman_numeral or test_skips_url_line_when_looking_for_embedded_number or test_does_not_match_embedded_number_on_an_overly_long_first_line" -v`
Expected: FAIL — the three number-finding tests fail because today's function only checks isolated lines; the "Afterword" test currently passes already (no regression risk there, but it's included to lock in the guard once the new code path exists); the long-line test currently passes too for the same reason.

- [ ] **Step 3: Widen `extract_printed_page_number`**

Replace the current function (`src/chapter_segmentation/segmentation.py:937-950`):

```python
def extract_printed_page_number(page_text: str) -> str | None:
    """Read the printed page number actually shown on a page, by looking for
    an isolated numeral/roman-numeral line near the top or bottom of the
    page's text (a running header/footer). Returns None if no such line is
    found — callers must treat this as "unknown", never guess.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return None
    candidates = lines[:2] + lines[-2:]
    for line in candidates:
        if _PAGE_NUMBER_TOKEN_RE.match(line):
            return line
    return None
```

with:

```python
# Boundary-guarded so a bare trailing/leading letter of an ordinary word
# (e.g. "Afterword", "Index") never false-positives as a roman numeral --
# see evaluation/CLAUDE.md's "Known failure modes", this ports the fix
# already proven there (evaluation/scripts/ground_truth_helper.py).
_TRAILING_PAGE_NUM_RE = re.compile(r"(?<![A-Za-z])(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)
_LEADING_PAGE_NUM_RE = re.compile(r"^(\d{1,4}|[ivxlcdm]{1,7})(?![A-Za-z])", re.IGNORECASE)


def extract_printed_page_number(page_text: str) -> str | None:
    """Read the printed page number actually shown on a page: first an
    isolated numeral/roman-numeral header/footer line near the top or
    bottom of the page, then (if that finds nothing) a number embedded at
    either end of the page's first non-URL/DOI line -- running headers
    alternate between "<num> <author>" and "<title> ... <num>" depending on
    recto/verso convention (reuses the existing _looks_like_url_or_doi
    filter, the same one find_toc_candidates uses, rather than duplicating
    a URL check). Returns None if neither check finds anything -- callers
    must treat this as "unknown", never guess.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return None
    candidates = lines[:2] + lines[-2:]
    for line in candidates:
        if _PAGE_NUMBER_TOKEN_RE.match(line):
            return line
    first_line = next((ln for ln in lines if not _looks_like_url_or_doi(ln)), None)
    if first_line is not None and len(first_line) < 120:
        match = _TRAILING_PAGE_NUM_RE.search(first_line)
        if match:
            return match.group(1)
        match = _LEADING_PAGE_NUM_RE.match(first_line)
        if match:
            return match.group(1)
    return None
```

(Note: this reuses the existing `_looks_like_url_or_doi` helper (`segmentation.py:171`) instead of introducing a near-duplicate URL check — it's already proven and already imported/available at module scope.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestExtractPrintedPageNumber -v`
Expected: PASS (8 tests — 3 existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: widen extract_printed_page_number to catch embedded page numbers"
```

---

### Task 2: Document-level offset inference (`_page_number_anchors`, `_infer_printed_page`, `_to_roman`)

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py` (insert right after `extract_printed_page_number`, widened in Task 1, before `_NLP = None`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add `_page_number_anchors`, `_infer_printed_page`, `_to_roman`, and `_parse_toc_page_number` to the private-helper import tuple in `tests/test_segmentation.py` (the one already importing `_llm_scan_indices`, near line 13):

```python
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
    _parse_toc_page_number,
    analyze_attachment_with_llm_fallback,
    analyze_attachment_outline_only,
    analyze_attachment_llm_only,
)
```

Add these test classes right after `TestExtractPrintedPageNumber` (before `TestExtractAuthorsNear`, `tests/test_segmentation.py:811`):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k "TestPageNumberAnchors or TestInferPrintedPage or TestToRoman" -v`
Expected: FAIL with `ImportError: cannot import name '_page_number_anchors'`

- [ ] **Step 3: Implement `_page_number_anchors`, `_infer_printed_page`, `_to_roman`**

Insert this block immediately after `extract_printed_page_number` (widened in Task 1), before `_NLP = None`:

```python
_ROMAN_NUMERALS = [
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
    (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
]


def _to_roman(value: int) -> str:
    """Lowercase roman-numeral rendering of a positive int -- the inverse
    of _parse_toc_page_number's roman branch. No upper bound enforced here;
    callers that need one (e.g. a plausible page number) apply it
    themselves via _ROMAN_PAGE_MAX_VALUE / _TOC_MAX_PAGE_NUMBER_RATIO.
    """
    result = []
    for amount, numeral in _ROMAN_NUMERALS:
        count, value = divmod(value, amount)
        result.append(numeral * count)
    return "".join(result)


def _page_number_anchors(pages: list[str]) -> list[tuple[int, int, bool]]:
    """Every page whose printed number extract_printed_page_number can read
    directly, parsed to an int. Returns (page_index, value, is_roman)
    triples in page order -- the raw material _infer_printed_page uses to
    recover a page's printed number when its own text has no directly
    readable one.
    """
    anchors: list[tuple[int, int, bool]] = []
    for index, text in enumerate(pages):
        raw = extract_printed_page_number(text)
        if raw is None:
            continue
        value = _parse_toc_page_number(raw)
        if value is not None:
            anchors.append((index, value, not raw.isdigit()))
    return anchors


# A chapter boundary more than this many pages from the nearest directly-read
# page number is not trusted to share that anchor's offset -- long gaps are
# where an unpaginated insert or numbering-scheme change (roman front matter
# -> arabic body) is most likely to sit undetected between the two.
_PAGE_NUMBER_INFERENCE_MAX_GAP = 10


def _infer_printed_page(index: int, anchors: list[tuple[int, int, bool]]) -> str | None:
    """The printed page number for `index`, extrapolated from the nearest
    directly-read anchors on either side, when they agree. Deliberately
    interpolation, not extrapolation past both ends: an index with no
    anchor within _PAGE_NUMBER_INFERENCE_MAX_GAP on a given side is treated
    as if that side has no anchor at all, so a chapter near either edge of
    the document (or past the last/before the first directly-readable page
    number anywhere in it) stays unmappable rather than guessing.

    A roman-numbered anchor and an arabic-numbered anchor bracketing the
    same index will not coincidentally agree on offset (roman "vi"
    continuing arithmetically into arabic numbering doesn't land on the
    arabic anchor's actual value) -- this naturally rejects inference
    across a numbering-scheme change without needing to special-case roman
    vs. arabic.
    """
    before = [a for a in anchors if a[0] <= index and index - a[0] <= _PAGE_NUMBER_INFERENCE_MAX_GAP]
    after = [a for a in anchors if a[0] >= index and a[0] - index <= _PAGE_NUMBER_INFERENCE_MAX_GAP]
    if not before or not after:
        return None
    before_index, before_value, before_roman = min(before, key=lambda a: index - a[0])
    after_index, after_value, after_roman = min(after, key=lambda a: a[0] - index)
    offset_before = before_value - before_index
    offset_after = after_value - after_index
    if offset_before != offset_after or before_roman != after_roman:
        return None
    value = index + offset_before
    return _to_roman(value) if before_roman else str(value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k "TestPageNumberAnchors or TestInferPrintedPage or TestToRoman" -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: add cross-page anchor interpolation for printed page numbers"
```

---

### Task 3: Prefer the TOC-declared page number (`_format_page_number`, `_toc_declared_page`)

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py` (insert right after `_infer_printed_page`, added in Task 2, before `_NLP = None`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add `_toc_declared_page` to the same private-helper import tuple used in Task 2.

Add this test class right after `TestToRoman` (added in Task 2):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestTocDeclaredPage -v`
Expected: FAIL with `ImportError: cannot import name '_toc_declared_page'`

- [ ] **Step 3: Implement `_format_page_number` and `_toc_declared_page`**

Insert this block immediately after `_infer_printed_page` (added in Task 2), before `_NLP = None`:

```python
def _format_page_number(value: int, is_roman: bool) -> str:
    return _to_roman(value) if is_roman else str(value)


def _toc_declared_page(entry: TocEntry, total_pages: int) -> str | None:
    """entry's own printed_page_number, formatted, when the TOC (heuristic
    or LLM) supplied a plausible one. -1 is the sentinel both sources use
    for "not identified" (a regex-found entry always has a real, valid
    value -- find_toc_candidates never constructs one otherwise; an
    LLM-found entry uses -1 when it couldn't read one, see
    llm_extract_toc_entries). The plausibility ceiling mirrors
    find_toc_candidates' own guard (_TOC_MAX_PAGE_NUMBER_RATIO) -- the LLM
    path has no equivalent check of its own, and an LLM could hallucinate
    an implausible value where the heuristic regex parser structurally
    cannot.
    """
    value = entry.printed_page_number
    if value <= 0 or value > total_pages * _TOC_MAX_PAGE_NUMBER_RATIO:
        return None
    return _format_page_number(value, entry.printed_roman)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestTocDeclaredPage -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: prefer TOC-declared printed_page_number over body-page scanning"
```

---

### Task 4: Fallback end page (`_fallback_end_printed`)

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py` (insert right after `_toc_declared_page`, added in Task 3, before `_NLP = None`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add `_fallback_end_printed` to the same private-helper import tuple used in Tasks 2-3.

Add this test class right after `TestTocDeclaredPage` (added in Task 3):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestFallbackEndPrinted -v`
Expected: FAIL with `ImportError: cannot import name '_fallback_end_printed'`

- [ ] **Step 3: Implement `_fallback_end_printed`**

Insert this block immediately after `_toc_declared_page` (added in Task 3), before `_NLP = None`:

```python
def _fallback_end_printed(
    i: int,
    located: list[tuple[TocEntry, ChapterStartMatch]],
    total_pages: int,
    pages: list[str],
    anchors: list[tuple[int, int, bool]],
    start_printed: str,
) -> str | None:
    """When a chapter's own end page has no resolvable printed number, use
    the page immediately before the next chapter's raw start (or the book's
    last page, for the final chapter) as a stand-in -- typically at or past
    the true end (trailing blank/divider pages sit in that gap), which
    matches the philosophy that an over-inclusive end is still usable while
    an under-inclusive one is not. Rejects the fallback if it can't be
    resolved to a printed number at all, or if it parses to a different
    numbering scheme (roman/arabic) or a smaller value than start_printed --
    either signals something is wrong rather than merely imprecise, and a
    wrong-looking citation is worse than none.
    """
    fallback_index = (located[i + 1][1].index - 1) if i + 1 < len(located) else (total_pages - 1)
    if fallback_index < 0:
        return None
    raw = extract_printed_page_number(pages[fallback_index]) or _infer_printed_page(fallback_index, anchors)
    if raw is None:
        return None
    start_value = _parse_toc_page_number(start_printed)
    end_value = _parse_toc_page_number(raw)
    if start_value is None or end_value is None or end_value < start_value:
        return None
    if start_printed.isdigit() != raw.isdigit():
        return None
    return raw
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestFallbackEndPrinted -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: add _fallback_end_printed for unresolvable chapter end pages"
```

---

### Task 5: Wire it all together in `_chapters_from_located`

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:1158` (hoist `anchors`) and `:1209-1216` (the priority-chain block, line numbers as of the pre-Task-1-4 file; see Step 3 for exact before/after text instead of relying on these, since Tasks 1-4 shifted everything below them down)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add this new test class right after `TestChaptersFromLocated` (`tests/test_segmentation.py`, currently ending around what was line 1084 before Tasks 1-4 shifted it down — search for `class TestChaptersFromLocated` and add after its one existing test method, before `class TestAnalysisCache`):

```python
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
        pages = [
            self._FILLER + "\n\nvii",
            self._FILLER + "\n\n6",
            self._FILLER + "\n\n1",
            self._FILLER,
            self._FILLER,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestChaptersFromLocatedPageNumberPriority -v`
Expected: FAIL — every test's `citation_pages`/`page_mapping_confidence` assertion fails against today's simpler (non-priority-chain) implementation.

- [ ] **Step 3: Wire the priority chain into `_chapters_from_located`**

First, hoist the anchors computation. Find this line inside `_chapters_from_located` (right after `total_pages = len(pages)`):

```python
    header_lines = _running_header_lines(tuple(pages))
```

and add a line right after it:

```python
    header_lines = _running_header_lines(tuple(pages))
    anchors = _page_number_anchors(pages)
```

Then, inside the `for i, (entry, match) in enumerate(located):` loop, replace this block:

```python
        start_printed = extract_printed_page_number(pages[start_index])
        end_printed = extract_printed_page_number(pages[end_index])
        if start_printed is not None and end_printed is not None:
            citation_pages = f"{start_printed}-{end_printed}"
            page_mapping_confidence = "high"
        else:
            citation_pages = None
            page_mapping_confidence = "unmappable"
```

with:

```python
        start_printed = _toc_declared_page(entry, total_pages)
        start_is_high = start_printed is not None
        if start_printed is None:
            start_printed = extract_printed_page_number(pages[start_index])
            start_is_high = start_printed is not None
        if start_printed is None:
            start_printed = _infer_printed_page(start_index, anchors)

        end_printed = None
        next_entry = located[i + 1][0] if i + 1 < len(located) else None
        if (
            next_entry is not None
            and next_entry.printed_roman == entry.printed_roman
            and _toc_declared_page(next_entry, total_pages) is not None
            and next_entry.printed_page_number - 1 >= 0
        ):
            end_printed = _format_page_number(next_entry.printed_page_number - 1, entry.printed_roman)
        end_is_high = False
        if end_printed is None:
            end_printed = extract_printed_page_number(pages[end_index])
            end_is_high = end_printed is not None
        if end_printed is None:
            end_printed = _infer_printed_page(end_index, anchors)
        if end_printed is None and start_printed is not None:
            end_printed = _fallback_end_printed(i, located, total_pages, pages, anchors, start_printed)

        if start_printed is not None and end_printed is not None:
            citation_pages = f"{start_printed}-{end_printed}"
            page_mapping_confidence = "high" if (start_is_high and end_is_high) else "inferred"
        else:
            citation_pages = None
            page_mapping_confidence = "unmappable"
```

- [ ] **Step 4: Run the new tests, then the full file's test suite**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestChaptersFromLocatedPageNumberPriority -v`
Expected: PASS (7 tests)

Run: `uv run python -m pytest tests/test_segmentation.py -v`
Expected: PASS, all tests — in particular confirm `TestChaptersFromLocated::test_entry_source_maps_llm_entries_and_defaults_others_to_heuristic` and `TestAnalyzeAttachment::test_citation_pages_extracted_when_present` still pass unchanged (both books in those fixtures have plain, directly-readable page numbers, so they should still resolve to `"high"`).

Then run the full project test suite to confirm no regressions anywhere else:

Run: `uv run python -m pytest -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "feat: wire TOC-declared/interpolated/fallback priority chain into _chapters_from_located"
```

---

### Task 6: Fix the LLM's roman-numeral schema blind spot

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:488-500` (`_LLM_TOC_EXTRACTION_PROMPT`) and `:615-630` (the parsing loop in `llm_extract_toc_entries`) — these line numbers are unaffected by Tasks 1-5, which only touch code after line 937.
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add these test methods to the existing `TestLlmExtractTocEntries` class (`tests/test_segmentation.py`), right after `test_logs_classified_reason_when_both_attempts_fail`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_segmentation.py -k "test_parses_roman_numeral_page_string or test_implausible_roman_string_uses_sentinel" -v`
Expected: FAIL — `test_parses_roman_numeral_page_string` fails because today's code does `int("vii")`-equivalent logic that only recognizes `int`/`float`, so a string like `"vii"` falls to the `-1` sentinel instead of `7`; `test_implausible_roman_string_uses_sentinel` currently passes already (an unrelated string also falls to `-1` today), but is included to lock in the behavior once the new parsing path exists. (`test_parses_arabic_page_string`, `test_tolerates_legacy_bare_int_page_number`, `test_null_page_number_still_uses_sentinel` also currently pass — same reasoning.)

- [ ] **Step 3: Update the prompt and parsing**

Replace the prompt constant (`src/chapter_segmentation/segmentation.py:488-500`):

```python
_LLM_TOC_EXTRACTION_PROMPT = """\
You are reading the front and back matter of a scanned/extracted book to \
find its table of contents. Some layouts don't use simple dotted leaders \
(e.g. "Title ..... 12") -- read the text directly rather than pattern-matching.

{page_blocks}

Return ONLY a JSON array, one entry per real chapter -- skip \
acknowledgements, bibliography, index, and part-divider pages:
[{{"title": "...", "authors": ["First Last", ...], "printed_page_number": 12}}]

If a chapter's printed page number is not visible in this text, use null \
for printed_page_number. If authors are not identifiable, use an empty list."""
```

with:

```python
_LLM_TOC_EXTRACTION_PROMPT = """\
You are reading the front and back matter of a scanned/extracted book to \
find its table of contents. Some layouts don't use simple dotted leaders \
(e.g. "Title ..... 12") -- read the text directly rather than pattern-matching.

{page_blocks}

Return ONLY a JSON array, one entry per real chapter -- skip \
acknowledgements, bibliography, index, and part-divider pages:
[{{"title": "...", "authors": ["First Last", ...], "printed_page_number": "12"}}]

printed_page_number is the page number exactly AS PRINTED on the page -- \
copy it verbatim, including roman numerals for front-matter chapters \
(e.g. "vii", not 7). If a chapter's printed page number is not visible in \
this text, use null for printed_page_number. If authors are not \
identifiable, use an empty list."""
```

Then find this block inside `llm_extract_toc_entries`'s parsing loop:

```python
        raw_authors = item.get("authors")
        # Guard against a malformed LLM response giving a plain string
        # instead of a list (e.g. "authors": "Jane Doe") -- iterating a
        # string yields one entry per character, silently corrupting
        # author-aware disambiguation downstream.
        authors = tuple(str(a).strip() for a in raw_authors if str(a).strip()) if isinstance(raw_authors, list) else ()
        printed = item.get("printed_page_number")
        # -1 is a sentinel for "unknown" (LLM returned null or an
        # unparseable value) -- never a real printed page number, and
        # currently unread by any downstream consumer (see TocEntry).
        printed_page_number = int(printed) if isinstance(printed, (int, float)) else -1
        # source_page_index is a sentinel here -- unlike a regex-found entry,
        # an LLM-extracted entry has no single "the TOC line was on this
        # page" origin; the orchestration layer excludes the whole scanned
        # front/back-matter range instead (see _toc_scan_indices).
        entries.append(TocEntry(title=title, printed_page_number=printed_page_number, source_page_index=-1, authors=authors))
    return entries
```

and replace it with:

```python
        raw_authors = item.get("authors")
        # Guard against a malformed LLM response giving a plain string
        # instead of a list (e.g. "authors": "Jane Doe") -- iterating a
        # string yields one entry per character, silently corrupting
        # author-aware disambiguation downstream.
        authors = tuple(str(a).strip() for a in raw_authors if str(a).strip()) if isinstance(raw_authors, list) else ()
        printed = item.get("printed_page_number")
        if isinstance(printed, (int, float)):
            # Tolerate a model that ignores the string instruction and
            # returns a bare number anyway -- still unambiguous for the
            # arabic case.
            printed = str(int(printed))
        parsed_value = _parse_toc_page_number(printed.strip()) if isinstance(printed, str) else None
        # -1 is a sentinel for "unknown" (LLM returned null, an unparseable
        # value, or an implausible one, e.g. a roman numeral over
        # _ROMAN_PAGE_MAX_VALUE) -- never a real printed page number.
        printed_page_number = parsed_value if parsed_value is not None else -1
        printed_roman = parsed_value is not None and not printed.strip().isdigit()
        # source_page_index is a sentinel here -- unlike a regex-found entry,
        # an LLM-extracted entry has no single "the TOC line was on this
        # page" origin; the orchestration layer excludes the whole scanned
        # front/back-matter range instead (see _toc_scan_indices).
        entries.append(TocEntry(
            title=title, printed_page_number=printed_page_number, source_page_index=-1,
            authors=authors, printed_roman=printed_roman,
        ))
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_segmentation.py -k TestLlmExtractTocEntries -v`
Expected: PASS, all tests (existing + 5 new)

Run: `uv run python -m pytest tests/test_segmentation.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "fix: preserve roman numerals in LLM-extracted printed_page_number"
```

---

### Task 7: Evaluation metric for citation-page accuracy

**Files:**
- Modify: `evaluation/metrics.py` (append new code)
- Modify: `evaluation/report_html.py` (add optional citation-accuracy columns)
- Modify: `evaluation/generate_report.py` (compute and pass citation aggregates)
- Test: `tests/test_metrics.py`, `tests/test_report_html.py`, `tests/test_generate_report.py`

- [ ] **Step 1: Write the failing tests for `evaluation/metrics.py`**

In `tests/test_metrics.py`, update the import line and add a helper:

```python
from evaluation.metrics import CitationPageAggregate, MicroAggregate, citation_pages_metrics, precision_recall_f1


def _chapter_with_citation(start: int, end: int, citation_pages: str | None) -> dict:
    return {"pdf_start_index": start, "pdf_end_index": end, "citation_pages": citation_pages}
```

Add these test classes at the end of the file, before `if __name__ == "__main__":`:

```python
class TestCitationPagesMetrics(unittest.TestCase):
    def test_correct_start_and_exact_end(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-20")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.checked_count, 1)
        self.assertEqual((m.start_coverage, m.start_accuracy), (1.0, 1.0))
        self.assertEqual((m.end_coverage, m.end_accuracy), (1.0, 1.0))

    def test_wrong_start_counts_as_incorrect(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "13-20")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.start_accuracy, 0.0)
        self.assertEqual(m.start_coverage, 1.0)  # a (wrong) value was still found

    def test_end_over_inclusive_within_tolerance_counts_as_correct(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-23")]  # +3, at the tolerance boundary
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 1.0)

    def test_end_over_inclusive_beyond_tolerance_counts_as_incorrect(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-24")]  # +4, past the tolerance
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 0.0)

    def test_end_under_inclusive_by_one_page_counts_as_incorrect(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, "12-19")]  # -1, real content cut off
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.end_accuracy, 0.0)

    def test_null_found_citation_pages_counts_as_uncovered(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 5, None)]
        m = citation_pages_metrics(expected, found)
        self.assertEqual((m.start_coverage, m.start_accuracy), (0.0, 0.0))
        self.assertEqual((m.end_coverage, m.end_accuracy), (0.0, 0.0))

    def test_expected_chapter_with_null_citation_pages_excluded_from_denominator(self):
        expected = [_chapter_with_citation(0, 5, None), _chapter_with_citation(6, 10, "1-4")]
        found = [_chapter_with_citation(0, 5, None), _chapter_with_citation(6, 10, "1-4")]
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.checked_count, 1)

    def test_expected_chapter_with_no_matching_found_range_excluded_from_denominator(self):
        expected = [_chapter_with_citation(0, 5, "12-20")]
        found = [_chapter_with_citation(0, 6, "12-20")]  # different range -- not a match at all
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.checked_count, 0)

    def test_handles_roman_numeral_pages(self):
        expected = [_chapter_with_citation(0, 5, "vii-x")]
        found = [_chapter_with_citation(0, 5, "vii-xi")]  # end +1 (x=10 -> xi=11), within tolerance
        m = citation_pages_metrics(expected, found)
        self.assertEqual(m.start_accuracy, 1.0)
        self.assertEqual(m.end_accuracy, 1.0)

    def test_no_checked_chapters_returns_all_zero(self):
        m = citation_pages_metrics([], [])
        self.assertEqual(
            (m.start_coverage, m.start_accuracy, m.end_coverage, m.end_accuracy, m.checked_count),
            (0.0, 0.0, 0.0, 0.0, 0),
        )


class TestCitationPageAggregate(unittest.TestCase):
    def test_pools_counts_across_documents_before_computing_rates(self):
        agg = CitationPageAggregate()
        # Book A: 1 of 1 checked, start correct, end wrong (under-inclusive)
        agg.add(citation_pages_metrics(
            [_chapter_with_citation(0, 5, "12-20")], [_chapter_with_citation(0, 5, "12-19")],
        ))
        # Book B: 1 of 1 checked, both correct
        agg.add(citation_pages_metrics(
            [_chapter_with_citation(0, 5, "1-4")], [_chapter_with_citation(0, 5, "1-4")],
        ))
        result = agg.compute()
        self.assertEqual(result.checked_count, 2)
        self.assertEqual(result.start_accuracy, 1.0)
        self.assertEqual(result.end_accuracy, 0.5)

    def test_empty_aggregate_is_all_zero(self):
        result = CitationPageAggregate().compute()
        self.assertEqual(result.checked_count, 0)
        self.assertEqual((result.start_accuracy, result.end_accuracy), (0.0, 0.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'CitationPageAggregate'`

- [ ] **Step 3: Implement `citation_pages_metrics` and `CitationPageAggregate`**

Append to `evaluation/metrics.py`:

```python
# Chapter-boundary correctness (pdf_start_index/pdf_end_index) is scored by
# precision_recall_f1 above with no tolerance. citation_pages is a SEPARATE,
# human-facing signal (the printed page range a consumer would cite/split
# on) -- see design spec docs/superpowers/specs/2026-08-08-citation-pages-
# mapping-design.md. Its own accuracy is scored independently per side: the
# start page is load-bearing (a consumer can't reconstruct it from
# context), the end page can reasonably be approximated (from the next
# chapter's start, or end-of-book) and only actually hurts usability when
# it's under-inclusive (cuts off real content).
_CITATION_END_OVER_INCLUSION_TOLERANCE = 3  # printed pages

# A local, minimal reimplementation of chapter_segmentation.segmentation's
# private _parse_toc_page_number -- evaluation/ only depends on
# segmentation's public surface elsewhere, and citation_pages strings here
# are always already-validated production output (never re-validated
# against segmentation's own plausibility guards), so a simple digit/roman
# parse is all this needs.
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _page_number_value(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    lowered = raw.lower()
    if not lowered or not all(ch in _ROMAN_VALUES for ch in lowered):
        return None
    total = 0
    for ch, nxt in zip(lowered, lowered[1:] + " "):
        value = _ROMAN_VALUES[ch]
        total += -value if nxt != " " and _ROMAN_VALUES.get(nxt, 0) > value else value
    return total


def _split_citation_pages(value: str | None) -> tuple[str, str] | None:
    if value is None or "-" not in value:
        return None
    start, _, end = value.partition("-")
    return start, end


@dataclass(frozen=True)
class CitationPageMetrics:
    start_coverage: float
    start_accuracy: float
    end_coverage: float
    end_accuracy: float
    checked_count: int
    start_covered_count: int = 0
    start_correct_count: int = 0
    end_covered_count: int = 0
    end_correct_count: int = 0


def citation_pages_metrics(expected: list[dict], found: list[dict]) -> CitationPageMetrics:
    """Among chapters whose (pdf_start_index, pdf_end_index) correctly
    matches expected (the same true-positive set precision_recall_f1
    scores), how well does citation_pages do -- restricted to expected
    chapters that themselves have a non-null citation_pages (an expected
    null means no printed number is visible anywhere on that chapter's real
    boundary pages, so there is nothing to score). End-page correctness
    tolerates a found value up to _CITATION_END_OVER_INCLUSION_TOLERANCE
    printed pages PAST the expected end (trailing blank/divider pages
    absorbed into the range) but never before it (that would mean real
    chapter content was cut off, a real defect, not an approximation)."""
    found_by_range = {(c["pdf_start_index"], c["pdf_end_index"]): c for c in found}
    checked = [
        e for e in expected
        if _split_citation_pages(e.get("citation_pages")) is not None
        and (e["pdf_start_index"], e["pdf_end_index"]) in found_by_range
    ]
    if not checked:
        return CitationPageMetrics(0.0, 0.0, 0.0, 0.0, 0)
    start_covered = start_correct = end_covered = end_correct = 0
    for e in checked:
        expected_start, expected_end = _split_citation_pages(e["citation_pages"])
        found_split = _split_citation_pages(found_by_range[(e["pdf_start_index"], e["pdf_end_index"])].get("citation_pages"))
        found_start, found_end = found_split if found_split else (None, None)
        start_covered += found_start is not None
        start_correct += found_start == expected_start
        end_covered += found_end is not None
        expected_end_value = _page_number_value(expected_end)
        found_end_value = _page_number_value(found_end) if found_end else None
        end_correct += (
            found_end_value is not None
            and expected_end_value is not None
            and expected_end_value <= found_end_value <= expected_end_value + _CITATION_END_OVER_INCLUSION_TOLERANCE
        )
    n = len(checked)
    return CitationPageMetrics(
        start_covered / n, start_correct / n, end_covered / n, end_correct / n, n,
        start_covered, start_correct, end_covered, end_correct,
    )


class CitationPageAggregate:
    """Pools citation_pages_metrics results across documents before
    computing final rates -- same weighted-pooling style as MicroAggregate,
    reading raw counts (never re-deriving them from rates, which would
    round-trip through lossy float rounding)."""

    def __init__(self) -> None:
        self._checked = 0
        self._start_covered = 0
        self._start_correct = 0
        self._end_covered = 0
        self._end_correct = 0

    def add(self, metrics: CitationPageMetrics) -> None:
        self._checked += metrics.checked_count
        self._start_covered += metrics.start_covered_count
        self._start_correct += metrics.start_correct_count
        self._end_covered += metrics.end_covered_count
        self._end_correct += metrics.end_correct_count

    def compute(self) -> CitationPageMetrics:
        n = self._checked
        if n == 0:
            return CitationPageMetrics(0.0, 0.0, 0.0, 0.0, 0)
        return CitationPageMetrics(
            self._start_covered / n, self._start_correct / n,
            self._end_covered / n, self._end_correct / n, n,
            self._start_covered, self._start_correct, self._end_covered, self._end_correct,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_metrics.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add citation_pages_metrics and CitationPageAggregate to evaluation/metrics.py"
```

- [ ] **Step 6: Write the failing test for `evaluation/report_html.py`**

Add this test method to `tests/test_report_html.py`'s `TestRenderStrategyTables` class, and update the import line:

```python
from evaluation.metrics import CitationPageMetrics, Metrics
```

```python
    def test_renders_citation_accuracy_columns_when_provided(self):
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic"],
            per_document={},
            aggregates={"heuristic": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0},
            citation_aggregates={"heuristic": CitationPageMetrics(0.9, 0.8, 0.7, 0.6, 10)},
        )
        agg_section = html[html.index("Per strategy"):]
        self.assertIn("Start accuracy", agg_section)
        self.assertIn("End accuracy", agg_section)
        self.assertIn("0.80", agg_section)  # start_accuracy
        self.assertIn("0.60", agg_section)  # end_accuracy

    def test_omits_citation_accuracy_columns_when_not_provided(self):
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic"],
            per_document={},
            aggregates={"heuristic": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0},
        )
        self.assertNotIn("Start accuracy", html)
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_report_html.py -v`
Expected: FAIL — `TypeError: render_strategy_tables() got an unexpected keyword argument 'citation_aggregates'`

- [ ] **Step 8: Add the optional citation-accuracy columns**

In `evaluation/report_html.py`, update the import line:

```python
from evaluation.metrics import Metrics
```

to:

```python
from evaluation.metrics import CitationPageMetrics, Metrics
```

Then replace the `render_strategy_tables` function:

```python
def render_strategy_tables(
    title: str,
    description_html: str,
    strategy_names: list[str],
    per_document: dict[str, dict[str, _TableCell]],
    aggregates: dict[str, Metrics],
    aggregate_times: dict[str, float],
) -> str:
    """
    strategy_names: column order for the per-document table.
    per_document: {document_key: {strategy_name: (Metrics, elapsed_seconds) or None}}.
        None (or a missing key) means the strategy produced no result for
        that document -- rendered as "N/A".
    aggregates: {strategy_name: Metrics} -- micro-aggregate across every
        document that strategy actually ran on.
    aggregate_times: {strategy_name: total_elapsed_seconds}.
    Returns a full <html> document string.
    """
    doc_rows = []
    for doc_key in sorted(per_document):
        cells = per_document[doc_key]
        best_f1 = max(
            (cell[0].f1 for cell in cells.values() if cell is not None),
            default=None,
        )
        row_cells = []
        for strategy in strategy_names:
            cell = cells.get(strategy)
            # best_f1 == 0.0 means every strategy found nothing for this
            # book -- that's a shared failure, not a "win" for whichever
            # strategy happens to be listed, so nothing gets highlighted.
            is_best = (
                cell is not None and best_f1 is not None and best_f1 > 0.0 and cell[0].f1 == best_f1
            )
            row_cells.append(_cell_html(cell, is_best))
        doc_rows.append(f"<tr><td>{doc_key}</td>{''.join(row_cells)}</tr>")

    ranked_strategies = sorted(aggregates, key=lambda s: aggregates[s].f1, reverse=True)
    agg_rows = []
    for strategy in ranked_strategies:
        m = aggregates[strategy]
        t = aggregate_times.get(strategy, 0.0)
        agg_rows.append(
            f"<tr><td>{strategy}</td><td>{m.precision:.2f}</td><td>{m.recall:.2f}</td>"
            f"<td>{m.f1:.2f}</td><td>{m.true_positives}/{m.found_count} found, "
            f"{m.true_positives}/{m.expected_count} expected</td><td>{t:.2f}s</td></tr>"
        )

    doc_header = "".join(f"<th>{s}</th>" for s in strategy_names)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }}</style>
</head><body>
<h1>{title}</h1>
{description_html}
<h2>Per document</h2>
<table>
<tr><th>Book</th>{doc_header}</tr>
{"".join(doc_rows)}
</table>
<h2>Per strategy (aggregate, ordered by F1)</h2>
<table>
<tr><th>Strategy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Found / Expected</th><th>Total time</th></tr>
{"".join(agg_rows)}
</table>
</body></html>
"""
```

with:

```python
def render_strategy_tables(
    title: str,
    description_html: str,
    strategy_names: list[str],
    per_document: dict[str, dict[str, _TableCell]],
    aggregates: dict[str, Metrics],
    aggregate_times: dict[str, float],
    citation_aggregates: dict[str, CitationPageMetrics] | None = None,
) -> str:
    """
    strategy_names: column order for the per-document table.
    per_document: {document_key: {strategy_name: (Metrics, elapsed_seconds) or None}}.
        None (or a missing key) means the strategy produced no result for
        that document -- rendered as "N/A".
    aggregates: {strategy_name: Metrics} -- micro-aggregate across every
        document that strategy actually ran on.
    aggregate_times: {strategy_name: total_elapsed_seconds}.
    citation_aggregates: optional {strategy_name: CitationPageMetrics} --
        when given, adds "Start accuracy"/"End accuracy" columns to the
        aggregate table (see design spec 2026-08-08). Omitted entirely
        (no extra columns) when not given, so existing callers/tests are
        unaffected.
    Returns a full <html> document string.
    """
    doc_rows = []
    for doc_key in sorted(per_document):
        cells = per_document[doc_key]
        best_f1 = max(
            (cell[0].f1 for cell in cells.values() if cell is not None),
            default=None,
        )
        row_cells = []
        for strategy in strategy_names:
            cell = cells.get(strategy)
            # best_f1 == 0.0 means every strategy found nothing for this
            # book -- that's a shared failure, not a "win" for whichever
            # strategy happens to be listed, so nothing gets highlighted.
            is_best = (
                cell is not None and best_f1 is not None and best_f1 > 0.0 and cell[0].f1 == best_f1
            )
            row_cells.append(_cell_html(cell, is_best))
        doc_rows.append(f"<tr><td>{doc_key}</td>{''.join(row_cells)}</tr>")

    ranked_strategies = sorted(aggregates, key=lambda s: aggregates[s].f1, reverse=True)
    agg_rows = []
    for strategy in ranked_strategies:
        m = aggregates[strategy]
        t = aggregate_times.get(strategy, 0.0)
        citation_cells = ""
        if citation_aggregates is not None:
            c = citation_aggregates.get(strategy)
            citation_cells = (
                f"<td>{c.start_accuracy:.2f}</td><td>{c.end_accuracy:.2f}</td>" if c else "<td>N/A</td><td>N/A</td>"
            )
        agg_rows.append(
            f"<tr><td>{strategy}</td><td>{m.precision:.2f}</td><td>{m.recall:.2f}</td>"
            f"<td>{m.f1:.2f}</td><td>{m.true_positives}/{m.found_count} found, "
            f"{m.true_positives}/{m.expected_count} expected</td><td>{t:.2f}s</td>{citation_cells}</tr>"
        )

    doc_header = "".join(f"<th>{s}</th>" for s in strategy_names)
    citation_header = "<th>Start accuracy</th><th>End accuracy</th>" if citation_aggregates is not None else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }}</style>
</head><body>
<h1>{title}</h1>
{description_html}
<h2>Per document</h2>
<table>
<tr><th>Book</th>{doc_header}</tr>
{"".join(doc_rows)}
</table>
<h2>Per strategy (aggregate, ordered by F1)</h2>
<table>
<tr><th>Strategy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Found / Expected</th><th>Total time</th>{citation_header}</tr>
{"".join(agg_rows)}
</table>
</body></html>
"""
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_report_html.py -v`
Expected: PASS, all tests

- [ ] **Step 10: Commit**

```bash
git add evaluation/report_html.py tests/test_report_html.py
git commit -m "feat: add optional citation-accuracy columns to render_strategy_tables"
```

- [ ] **Step 11: Write the failing test for `evaluation/generate_report.py`**

Add this test method to `tests/test_generate_report.py`'s `TestGenerate` class, right after `test_writes_main_report_and_llm_detail_page`:

```python
    def test_main_report_includes_citation_accuracy_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_dir = tmp_path / "evaluation"
            public_cache_dir = eval_dir / "public-cache"
            llm_cache_dir = eval_dir / "llm-cache"
            public_cache_dir.mkdir(parents=True)
            llm_cache_dir.mkdir(parents=True)
            out_dir = tmp_path / "public"

            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3, "citation_pages": "1-4"}]
            (eval_dir / "book-a.expected.json").write_text(_expected_json(chapters), encoding="utf-8")
            (public_cache_dir / "book-a.pages.json").write_text(
                json.dumps({"pages": ["Introduction\nBody text.\n\n1", "2", "3", "4"]}), encoding="utf-8",
            )
            book = {"filename": "book-a.pdf", "title": "Book A"}

            with patch("evaluation.harness.EVAL_DIR", eval_dir), \
                 patch("evaluation.harness.PUBLIC_CACHE_DIR", public_cache_dir), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]), \
                 patch("evaluation.generate_report.LLM_CACHE_DIR", llm_cache_dir), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Start accuracy", main_html)
            self.assertIn("End accuracy", main_html)
```

- [ ] **Step 12: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_generate_report.py -k test_main_report_includes_citation_accuracy_columns -v`
Expected: FAIL — `AssertionError: 'Start accuracy' not found in ...`

- [ ] **Step 13: Wire `citation_pages_metrics`/`CitationPageAggregate` into `generate_report.py`**

Update the import line in `evaluation/generate_report.py`:

```python
from evaluation.metrics import MicroAggregate, precision_recall_f1
```

to:

```python
from evaluation.metrics import CitationPageAggregate, MicroAggregate, citation_pages_metrics, precision_recall_f1
```

In `generate()`, replace:

```python
    per_document: dict[str, dict] = {}
    heuristic_agg, outline_agg, llm_agg = MicroAggregate(), MicroAggregate(), MicroAggregate()
```

with:

```python
    per_document: dict[str, dict] = {}
    heuristic_agg, outline_agg, llm_agg = MicroAggregate(), MicroAggregate(), MicroAggregate()
    heuristic_citation_agg, outline_citation_agg, llm_citation_agg = (
        CitationPageAggregate(), CitationPageAggregate(), CitationPageAggregate(),
    )
```

Then, right after `heuristic_agg.add(heuristic_metrics, heuristic_elapsed)`, add:

```python
        heuristic_citation_agg.add(citation_pages_metrics(expected, heuristic_result["chapters"]))
```

Right after `outline_agg.add(outline_metrics, outline_elapsed)`, add:

```python
            outline_citation_agg.add(citation_pages_metrics(expected, outline_result["chapters"]))
```

Right after `llm_agg.add(llm_metrics, llm_entry["elapsed_seconds"])`, add:

```python
                llm_citation_agg.add(citation_pages_metrics(expected, llm_entry["chapters"]))
```

Then replace:

```python
    aggregates = {HEURISTIC: heuristic_agg.compute(), OUTLINE: outline_agg.compute()}
    aggregate_times = {HEURISTIC: heuristic_agg.total_elapsed_seconds, OUTLINE: outline_agg.total_elapsed_seconds}
    if llm_strategy_name:
        aggregates[llm_strategy_name] = llm_agg.compute()
        aggregate_times[llm_strategy_name] = llm_agg.total_elapsed_seconds
```

with:

```python
    aggregates = {HEURISTIC: heuristic_agg.compute(), OUTLINE: outline_agg.compute()}
    aggregate_times = {HEURISTIC: heuristic_agg.total_elapsed_seconds, OUTLINE: outline_agg.total_elapsed_seconds}
    citation_aggregates = {HEURISTIC: heuristic_citation_agg.compute(), OUTLINE: outline_citation_agg.compute()}
    if llm_strategy_name:
        aggregates[llm_strategy_name] = llm_agg.compute()
        aggregate_times[llm_strategy_name] = llm_agg.total_elapsed_seconds
        citation_aggregates[llm_strategy_name] = llm_citation_agg.compute()
```

And in the `render_strategy_tables(...)` call inside `generate()`, add the new keyword argument:

```python
    html = render_strategy_tables(
        title="chapter-segmentation: public-cache corpus results",
        description_html=description,
        strategy_names=strategy_names,
        per_document=per_document,
        aggregates=aggregates,
        aggregate_times=aggregate_times,
        citation_aggregates=citation_aggregates,
    )
```

Finally, in `_generate_llm_detail_page`, replace:

```python
        per_document: dict[str, dict] = {}
        aggregates_acc = {model_id: MicroAggregate() for model_id in model_ids}
        for manifest_key, expected in books:
            cache = _load_llm_cache(manifest_key)
            cells: dict = {}
            for model_id in model_ids:
                entry = cache.get(model_id)
                if entry is None:
                    cells[model_id] = None
                    continue
                metrics = precision_recall_f1(expected, entry["chapters"])
                aggregates_acc[model_id].add(metrics, entry["elapsed_seconds"])
                cells[model_id] = (metrics, entry["elapsed_seconds"])
            per_document[manifest_key] = cells

        aggregates = {model_id: acc.compute() for model_id, acc in aggregates_acc.items()}
        aggregate_times = {model_id: acc.total_elapsed_seconds for model_id, acc in aggregates_acc.items()}
        html = render_strategy_tables(
            title="chapter-segmentation: LLM strategy results (all cached models)",
            description_html=(
                "<p>Every KISSKI model ever evaluated by "
                "<code>evaluation/refresh_llm_cache.py</code>, run standalone via "
                "<code>analyze_attachment_llm_only</code> (no heuristic fallback). "
                'See <a href="../index.html">the main report</a> for how the single '
                "best-performing model compares against the heuristic and outline "
                "strategies.</p>"
            ),
            strategy_names=sorted(model_ids),
            per_document=per_document,
            aggregates=aggregates,
            aggregate_times=aggregate_times,
        )
```

with:

```python
        per_document: dict[str, dict] = {}
        aggregates_acc = {model_id: MicroAggregate() for model_id in model_ids}
        citation_aggregates_acc = {model_id: CitationPageAggregate() for model_id in model_ids}
        for manifest_key, expected in books:
            cache = _load_llm_cache(manifest_key)
            cells: dict = {}
            for model_id in model_ids:
                entry = cache.get(model_id)
                if entry is None:
                    cells[model_id] = None
                    continue
                metrics = precision_recall_f1(expected, entry["chapters"])
                aggregates_acc[model_id].add(metrics, entry["elapsed_seconds"])
                citation_aggregates_acc[model_id].add(citation_pages_metrics(expected, entry["chapters"]))
                cells[model_id] = (metrics, entry["elapsed_seconds"])
            per_document[manifest_key] = cells

        aggregates = {model_id: acc.compute() for model_id, acc in aggregates_acc.items()}
        aggregate_times = {model_id: acc.total_elapsed_seconds for model_id, acc in aggregates_acc.items()}
        citation_aggregates = {model_id: acc.compute() for model_id, acc in citation_aggregates_acc.items()}
        html = render_strategy_tables(
            title="chapter-segmentation: LLM strategy results (all cached models)",
            description_html=(
                "<p>Every KISSKI model ever evaluated by "
                "<code>evaluation/refresh_llm_cache.py</code>, run standalone via "
                "<code>analyze_attachment_llm_only</code> (no heuristic fallback). "
                'See <a href="../index.html">the main report</a> for how the single '
                "best-performing model compares against the heuristic and outline "
                "strategies.</p>"
            ),
            strategy_names=sorted(model_ids),
            per_document=per_document,
            aggregates=aggregates,
            aggregate_times=aggregate_times,
            citation_aggregates=citation_aggregates,
        )
```

- [ ] **Step 14: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_generate_report.py -v`
Expected: PASS, all tests

Then run the full project test suite to confirm no regressions anywhere else:

Run: `uv run python -m pytest -v`
Expected: PASS, all tests.

- [ ] **Step 15: Commit**

```bash
git add evaluation/generate_report.py tests/test_generate_report.py
git commit -m "feat: report citation-page start/end accuracy per strategy"
```

---

### Task 8: Validate against the real KISSKI API and update RESULTS.md

**Files:**
- Modify: `evaluation/RESULTS.md` (the "Per-strategy standalone results" section)
- No further source changes in this task — this is a real-API validation + documentation update. Task 6 changed the prompt `llm_extract_toc_entries` sends, so every cached LLM result is stale and needs regenerating.

- [ ] **Step 1: Re-run the LLM cache refresh against the real API**

```bash
set -a; source .env; set +a
uv run python evaluation/refresh_llm_cache.py --mode full
```

This regenerates `evaluation/llm-cache/*.json` for every model currently cached, now using the fixed `llm_extract_toc_entries` (roman-numeral-preserving prompt) and the new `_chapters_from_located` priority chain.

- [ ] **Step 2: Regenerate the report locally and inspect it**

```bash
uv run python evaluation/generate_report.py --out /tmp/citation-pages-check
```

Open `/tmp/citation-pages-check/index.html` and `/tmp/citation-pages-check/llm/index.html`. Confirm the new "Start accuracy"/"End accuracy" columns appear on the aggregate table for heuristic, outline, and every LLM model, with plausible (non-zero, since most chapters do have a resolvable citation now) values.

- [ ] **Step 3: Update `evaluation/RESULTS.md`**

Update the "Per-strategy standalone results (heuristic / outline / LLM)" section with the new start/end accuracy numbers from Step 2, and add a short bullet summarizing the headline change: citation_pages coverage/accuracy moved from an untracked, mostly-`"unmappable"` blind spot to a scored signal, driven by preferring the TOC-declared page number over body-page text scanning.

- [ ] **Step 4: Run the full test suite one more time**

Run: `uv run python -m pytest -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add evaluation/llm-cache/ evaluation/RESULTS.md
git commit -m "data: regenerate LLM cache and RESULTS.md after citation_pages accuracy fix"
```

Do not push without separately confirming with the user first (per this project's established pattern of confirming before pushing to `main`).
