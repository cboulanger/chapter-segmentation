# Printed Page Number String Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `TocEntry.printed_page_number` from a lossy `int` (with a `-1` "unknown" sentinel) to `str | None`, storing whatever page marker text an extractor actually read -- fixing the silent data loss on section-prefixed markers like `"R42"` -- while every existing call site that constructs a `TocEntry` with a bare int (tests, on-disk cache files) keeps working unmodified.

**Architecture:** A single `_normalize_printed_page_number` function, run from `TocEntry.__post_init__`, is the one place every legal input shape (str, int, float, None, the old `-1` sentinel) gets coerced into the canonical `str | None` form. Every producer (`find_toc_candidates`'s regex path, `_toc_items_to_entries` shared by both LLM paths) stores the raw captured/returned text verbatim instead of a parsed int. Everything that used to do integer arithmetic or `== -1` comparisons directly against the field now parses on demand via the pre-existing `_parse_toc_page_number(str) -> int | None`, falling back to today's existing "can't resolve" behavior when parsing fails.

**Tech Stack:** Python 3.12, `unittest` (this repo's test framework -- see any `tests/test_*.py` for the convention: `unittest.TestCase`/`unittest.IsolatedAsyncioTestCase`, run via `pytest`).

**Design spec:** `docs/superpowers/specs/2026-08-17-printed-page-number-string-design.md` -- read it first; this plan implements it exactly, including the digit-based heuristic in `_toc_declared_page` that keeps an implausible roman numeral (`"mmmm"`) mapping to "unknown" while a real prefixed marker (`"R42"`) is preserved.

**Verified against real data before writing this plan:** `evaluation/corpus/dnb-toc-only/llm-cache/*.json` has 629 real cached entries, 5 of which use the old `-1` int sentinel today; `.expected.json` files already store `printed_page_number` as a string or `null`. Both round-trip correctly through the new `__post_init__` normalization with no data migration needed.

---

### Task 1: `_normalize_printed_page_number` + `TocEntry.__post_init__` (the BC core)

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:82-88` (add helper near `_parse_toc_page_number`), `src/chapter_segmentation/segmentation.py:208-244` (`TocEntry`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_segmentation.py`, right after the existing `TestParseTocPageNumber`-style tests (find that class, or add a new one near the top-level helper tests -- e.g. right before `class TestTocEntry` if one doesn't exist yet, create it). First add the import: in the existing `from chapter_segmentation.segmentation import (... _parse_toc_page_number,)` block (lines 35-45), add `_normalize_printed_page_number` to the list.

```python
class TestNormalizePrintedPageNumber(unittest.TestCase):
    def test_passes_through_a_plain_string_verbatim(self):
        self.assertEqual(_normalize_printed_page_number("R42"), "R42")

    def test_strips_whitespace(self):
        self.assertEqual(_normalize_printed_page_number("  42  "), "42")

    def test_empty_string_becomes_none(self):
        self.assertIsNone(_normalize_printed_page_number(""))

    def test_string_null_becomes_none(self):
        self.assertIsNone(_normalize_printed_page_number("null"))

    def test_none_stays_none(self):
        self.assertIsNone(_normalize_printed_page_number(None))

    def test_legacy_negative_one_sentinel_becomes_none(self):
        self.assertIsNone(_normalize_printed_page_number(-1))

    def test_legacy_bare_int_becomes_str(self):
        self.assertEqual(_normalize_printed_page_number(12), "12")

    def test_legacy_bare_float_becomes_str(self):
        self.assertEqual(_normalize_printed_page_number(12.0), "12")


class TestTocEntryConstructionBc(unittest.TestCase):
    def test_legacy_bare_int_construction_normalizes_to_str(self):
        entry = TocEntry(title="Introduction", printed_page_number=12, source_page_index=0)
        self.assertEqual(entry.printed_page_number, "12")

    def test_legacy_negative_one_sentinel_construction_normalizes_to_none(self):
        entry = TocEntry(title="Introduction", printed_page_number=-1, source_page_index=-1)
        self.assertIsNone(entry.printed_page_number)

    def test_string_construction_passes_through_verbatim(self):
        entry = TocEntry(title="Appendix", printed_page_number="R42", source_page_index=-1)
        self.assertEqual(entry.printed_page_number, "R42")

    def test_none_construction_stays_none(self):
        entry = TocEntry(title="Introduction", printed_page_number=None, source_page_index=-1)
        self.assertIsNone(entry.printed_page_number)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py -k "NormalizePrintedPageNumber or TocEntryConstructionBc" -v`
Expected: FAIL -- `ImportError: cannot import name '_normalize_printed_page_number'` (it doesn't exist yet), and/or `AssertionError` once the import is stubbed out, since `TocEntry.printed_page_number` is still a bare `int` field with no normalization.

- [ ] **Step 3: Implement `_normalize_printed_page_number` and `TocEntry.__post_init__`**

In `src/chapter_segmentation/segmentation.py`, add this function right after `_parse_toc_page_number` (currently ending at line 103, right before the `_TOC_MIN_LINES_PER_PAGE` comment block at line 105):

```python
def _normalize_printed_page_number(value: object) -> str | None:
    """Coerces any of TocEntry.printed_page_number's legal input shapes
    into the canonical str | None form -- the single place every producer
    (and every historical caller/on-disk cache file) gets normalized, via
    TocEntry.__post_init__. A str is returned verbatim (just stripped) --
    preserving whatever text an extractor actually read, including a
    section-prefixed marker like "R42" -- which is the entire point of
    this type. int/float only ever arrives from a legacy caller: the old
    -1 "unknown" sentinel (now None), or a bare numeric JSON value some
    LLM response used instead of the requested string.
    """
    if isinstance(value, str):
        text = value.strip()
        return None if not text or text.lower() == "null" else text
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = int(value)
        return None if value == -1 else str(value)
    return None
```

Then change `TocEntry` (lines 208-244) so the `printed_page_number` field is `str | None` and gets normalized on construction. The dataclass is `frozen=True`, so `__post_init__` must use `object.__setattr__`:

```python
@dataclass(frozen=True)
class TocEntry:
    title: str
    printed_page_number: str | None
    source_page_index: int  # which page (0-based) the TOC entry itself was found on
    authors: tuple[str, ...] = ()  # populated only by llm_extract_toc_entries (see
    # docs/superpowers/specs/2026-07-25-llm-chapter-segmentation-fallback-design.md §4)
    # -- a regex-found TOC line has no author info, so heuristic-found entries always
    # leave this empty. Feeds locate_chapter_start's author-aware disambiguation (§5).
    printed_roman: bool = False  # True when the TOC listed this entry with a
    # roman-numeral page ("Foreword vii") -- i.e. it lives in the book's
    # roman-paginated FRONT matter, so the usual "chapters start after the
    # table of contents" exclusion must not apply to it.
    title_variants: tuple[str, ...] = ()  # longer alternative readings of `title`,
    # built by prepending the preceding non-matching TOC-page line(s): a long
    # title wrapped over several lines puts its page number on the LAST line
    # only, so the regex captures just that tail fragment ("Gaze", "in der
    # Krise") -- far too generic to locate reliably. Which reading is correct
    # can't be known from the TOC page alone (the preceding line may be a
    # previous entry's author line or a part header), so ALL of them are
    # carried and _locate_toc_entries picks whichever variant actually
    # locates best in the book body. Empty for LLM-extracted entries, whose
    # titles are already read whole.
    skip: bool = False  # True when this entry is not itself a real chapter --
    # a part/section divider, or front/back matter (preface, bibliography,
    # index, ...). Only ever set by evaluation/dnb_toc_vision.py's vision
    # extraction (see its prompt): that path deliberately extracts EVERY
    # printed TOC line verbatim rather than omitting non-chapter lines
    # outright, so a two-model disagreement over what to extract can never
    # cause an editorial-judgment mismatch to fail the whole-book agreement
    # gate (evaluation/dnb_toc_matching.py) -- only a genuine reading
    # mismatch can. Which lines count as "real chapters" is then a
    # separate, revisable downstream decision instead of being baked
    # irreversibly into the extraction step. llm_extract_toc_entries'
    # production text prompt never asks for this field, so it's always
    # False there -- unrelated to this dnb-toc-only-specific concern.

    def __post_init__(self) -> None:
        # Normalizes every legal input shape (str, legacy bare int/float,
        # the old -1 sentinel, None) into the canonical str | None form --
        # see _normalize_printed_page_number's own docstring. object.
        # __setattr__ is required: this dataclass is frozen.
        object.__setattr__(self, "printed_page_number", _normalize_printed_page_number(self.printed_page_number))
```

Also add `_normalize_printed_page_number` to the `tests/test_segmentation.py` import block (lines 35-45): append `_normalize_printed_page_number,` to that `from chapter_segmentation.segmentation import (...)` tuple.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -k "NormalizePrintedPageNumber or TocEntryConstructionBc" -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Run the full existing test_segmentation.py suite to check for breakage**

Run: `uv run pytest tests/test_segmentation.py -v 2>&1 | tail -80`
Expected: Many failures -- every place that still asserts `entry.printed_page_number == <bare int>` or `== -1` now compares a string/`None` against an int and fails. This is expected at this point in the plan; Tasks 2-6 fix each of these call sites and their corresponding test assertions one at a time. Confirm the failures are ONLY `AssertionError`s comparing `printed_page_number`/`-1` (not `TypeError`s or import errors) before moving on -- that confirms `__post_init__` itself is working correctly and the remaining failures are exactly the expected fallout.

- [ ] **Step 6: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "$(cat <<'EOF'
feat: normalize TocEntry.printed_page_number to str | None

Adds _normalize_printed_page_number, run from TocEntry.__post_init__,
so every legal input shape (str, legacy bare int, the old -1 sentinel,
None) coerces into the canonical str | None form. Existing call sites
constructing TocEntry with a bare int keep working unmodified.
EOF
)"
```

---

### Task 2: `_toc_items_to_entries` -- verbatim storage for both LLM extraction paths

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:599-642`
- Test: `tests/test_segmentation.py` (LLM extraction tests), `tests/test_dnb_toc_vision.py`

- [ ] **Step 1: Update the existing test expectations that will change**

In `tests/test_segmentation.py`, class `TestLlmExtractTocEntries` (around line 505-603):

Change line 516-517 (`test_parses_chapter_list_from_llm_response`):
```python
        self.assertEqual(entries[0].printed_page_number, "1")
        self.assertIsNone(entries[1].printed_page_number)  # null -> None
```

Change line 576 (`test_parses_roman_numeral_page_string`) -- the LLM returned `"vii"` verbatim, so that's exactly what's now stored (not the parsed int `7`):
```python
    async def test_parses_roman_numeral_page_string(self):
        llm = self._fake_llm('[{"title": "Foreword", "authors": [], "printed_page_number": "vii"}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, "vii")
        self.assertTrue(entries[0].printed_roman)
```

Change line 582 (`test_parses_arabic_page_string`):
```python
        self.assertEqual(entries[0].printed_page_number, "12")
```

Change line 588 (`test_tolerates_legacy_bare_int_page_number`):
```python
        self.assertEqual(entries[0].printed_page_number, "12")
```

Change line 594 (`test_null_page_number_still_uses_sentinel`) -- rename it too, since there's no more sentinel:
```python
    async def test_null_page_number_becomes_none(self):
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": null}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertIsNone(entries[0].printed_page_number)
        self.assertFalse(entries[0].printed_roman)
```

Change `test_implausible_roman_string_uses_sentinel` (around line 597-603) -- rename and update: storage is now unconditionally verbatim; whether this counts as a plausible page for citation purposes is `_toc_declared_page`'s job (Task 4), not construction's:
```python
    async def test_implausible_roman_string_is_stored_verbatim(self):
        # Over _ROMAN_PAGE_MAX_VALUE (50), and not even a validly-shaped
        # roman numeral either (4 "m"s is not valid roman notation) --
        # stored verbatim regardless, matching the "copy it verbatim"
        # prompt instruction. See TestTocDeclaredPage for where this
        # gets treated as unknown for citation-page purposes.
        llm = self._fake_llm('[{"title": "Introduction", "authors": [], "printed_page_number": "mmmm"}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, "mmmm")
        self.assertFalse(entries[0].printed_roman)
```

- [ ] **Step 2: Add a new failing test for the reported bug -- a prefixed marker surviving verbatim**

Add to `TestLlmExtractTocEntries`:
```python
    async def test_section_prefixed_page_marker_survives_verbatim(self):
        # The real-world case this whole change exists for: a DNB-digitized
        # TOC scan's page marker like "R42" is neither a pure digit run nor
        # a valid roman numeral, so _parse_toc_page_number can't interpret
        # it -- but it must not be discarded the way the old int-with--1-
        # sentinel representation discarded it.
        llm = self._fake_llm('[{"title": "Appendix", "authors": [], "printed_page_number": "R42"}]')
        entries = await llm_extract_toc_entries(["front matter"] * 5, llm)
        self.assertEqual(entries[0].printed_page_number, "R42")
        self.assertFalse(entries[0].printed_roman)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py -k TestLlmExtractTocEntries -v`
Expected: FAIL -- the updated/new assertions compare against strings, but `_toc_items_to_entries` still parses to int and applies the old `-1` sentinel.

- [ ] **Step 4: Implement -- update `_toc_items_to_entries`**

Replace lines 622-633 of `src/chapter_segmentation/segmentation.py` (inside `_toc_items_to_entries`):

Old:
```python
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
```

New:
```python
        printed_page_number = _normalize_printed_page_number(item.get("printed_page_number"))
        # Roman iff it parses via _parse_toc_page_number AND isn't a plain
        # digit string -- same semantics as before, just checked against
        # the normalized (verbatim) string instead of a pre-parsed int.
        # A section-prefixed marker like "R42" correctly comes out False:
        # it isn't a digit run, but it also doesn't parse as roman either.
        printed_roman = (
            printed_page_number is not None
            and not printed_page_number.isdigit()
            and _parse_toc_page_number(printed_page_number) is not None
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -k "TestLlmExtractTocEntries" -v`
Expected: PASS (all tests in this class, including the two new ones)

- [ ] **Step 6: Fix the one affected assertion in test_dnb_toc_vision.py**

In `tests/test_dnb_toc_vision.py`, line 120 (`test_parses_response_into_toc_entries` -- the mocked vision response uses `"printed_page_number": "9"`, already a string, now stored verbatim):
```python
            self.assertEqual(entries[0].printed_page_number, "9")
```

Run: `uv run pytest tests/test_dnb_toc_vision.py -v`
Expected: PASS. (The cache round-trip tests in this file need no changes -- both sides of each `==` comparison go through the same normalization, so equality still holds regardless of the underlying representation.)

- [ ] **Step 7: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py tests/test_dnb_toc_vision.py
git commit -m "$(cat <<'EOF'
feat: store LLM-extracted printed_page_number verbatim, not parsed

_toc_items_to_entries (shared by llm_extract_toc_entries and
evaluation/dnb_toc_vision.py's vision extraction) now stores whatever
text the model returned, instead of collapsing anything
_parse_toc_page_number can't interpret (e.g. a section-prefixed
marker like "R42") to the same -1 sentinel used for "no value at
all". Plausibility filtering for citation-page purposes moves to
_toc_declared_page (a later commit), not construction.
EOF
)"
```

---

### Task 3: `find_toc_candidates`'s regex path -- verbatim storage

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:380-438` (`_valid_entries` inside `find_toc_candidates`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Update the existing test expectations that will change**

In `tests/test_segmentation.py`:

Line 129 (`test_parses_a_simple_dotted_toc`, or whichever test this line lives in -- the one building a 3-entry CONTENTS page):
```python
        self.assertEqual(entries[0].printed_page_number, "1")
```

Line 135 (same test):
```python
        self.assertEqual(entries[2].printed_page_number, "89")
```

Line 165 (`test_matches_whitespace_leaders_too`):
```python
        self.assertEqual(entries[0].printed_page_number, "12")
```

Line 313 (`test_roman_numeral_page_numbers_accepted_for_front_matter`) -- the regex captured `"vii"` verbatim, so that's now what's stored, not the parsed int `7`:
```python
        self.assertEqual(foreword.printed_page_number, "vii")
```

Line 344 (`test_bare_page_number_line_adopts_preceding_title_line`):
```python
        self.assertEqual(adopted.printed_page_number, "45")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmentation.py -k "find_toc_candidates or FindTocCandidates" -v`
Expected: FAIL -- `_valid_entries` still stores the parsed int.

(If the class/test names don't match this filter, run the whole file and look for the 5 failures at the lines listed above: `uv run pytest tests/test_segmentation.py -v 2>&1 | grep -B5 "AssertionError.*printed_page_number"`.)

- [ ] **Step 3: Implement -- update `_valid_entries`**

In `src/chapter_segmentation/segmentation.py`, inside `find_toc_candidates`'s `_valid_entries` (around lines 389-437), change:

Old (lines 389-391 and 433-437):
```python
            title = m.group("title").strip(" .")
            page_number = _parse_toc_page_number(m.group("page"))
            if page_number is None or page_number > max_plausible_page_number:
                continue
```
```python
            out.append(TocEntry(
                title=title, printed_page_number=page_number, source_page_index=page_index,
                printed_roman=not m.group("page").isdigit(),
                title_variants=tuple(variants),
            ))
```

New:
```python
            title = m.group("title").strip(" .")
            # _parse_toc_page_number is still the plausibility gate here
            # (an implausible value never becomes an entry at all) -- but
            # the TEXT stored on the entry is the original captured
            # string, not the parsed int, so a page reads back exactly as
            # printed.
            parsed_page_number = _parse_toc_page_number(m.group("page"))
            if parsed_page_number is None or parsed_page_number > max_plausible_page_number:
                continue
```
```python
            out.append(TocEntry(
                title=title, printed_page_number=m.group("page"), source_page_index=page_index,
                printed_roman=not m.group("page").isdigit(),
                title_variants=tuple(variants),
            ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -v 2>&1 | tail -40`
Expected: The 5 assertions above now pass. Some failures remain (Tasks 4-6 territory) -- confirm no NEW failures appeared beyond what Step 5 of Task 1 already flagged, and that the ones from this task's lines are gone.

- [ ] **Step 5: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "$(cat <<'EOF'
feat: store regex-heuristic printed_page_number verbatim, not parsed

find_toc_candidates now stores the TOC line's originally-captured page
text (e.g. "vii") rather than the value _parse_toc_page_number derived
from it (7) -- _parse_toc_page_number remains the plausibility gate
deciding whether an entry gets constructed at all, it just no longer
also decides what text ends up stored on it.
EOF
)"
```

---

### Task 4: `_toc_declared_page` digit heuristic + `_chapters_from_located` arithmetic fix

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:1091-1114` (`_toc_declared_page`), `src/chapter_segmentation/segmentation.py:1416-1424` (`_chapters_from_located`'s fast end-page path)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the new failing tests**

Add to `tests/test_segmentation.py`'s `TestTocDeclaredPage` class (around line 927-955):

```python
    def test_prefixed_marker_with_digit_returned_verbatim(self):
        # The real reported bug this whole change exists for: "R42" is
        # neither a pure digit run nor a valid roman numeral, so
        # _parse_toc_page_number can't interpret it -- but it carries a
        # digit, so it's a real alternate-scheme page marker, not noise.
        entry = TocEntry(title="Appendix", printed_page_number="R42", source_page_index=-1)
        self.assertEqual(_toc_declared_page(entry, total_pages=200), "R42")

    def test_digitless_implausible_string_returns_none(self):
        # "mmmm" has no digit at all and isn't a valid roman numeral (4
        # "m"s is invalid roman notation) -- far more likely OCR/model
        # noise than a real page marker, so still "unknown", matching the
        # old int-sentinel path's behavior for this exact case.
        entry = TocEntry(title="Introduction", printed_page_number="mmmm", source_page_index=-1)
        self.assertIsNone(_toc_declared_page(entry, total_pages=200))
```

Existing tests in this class (lines 928-955) all construct with bare ints or `-1` and only assert on `_toc_declared_page`'s return value (already a string, or `None`) -- these need NO changes; they should keep passing once Step 3 lands, confirming the refactor preserves exact behavior for every digit/roman case.

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_segmentation.py -k TestTocDeclaredPage -v`
Expected: FAIL for the two new tests -- `_toc_declared_page` still does `value = entry.printed_page_number; if value <= 0 ...` which raises `TypeError` comparing a `str`/`None` with `0` (or, if `TocEntry` accepted the string in this dataclass revision already, a `TypeError` on `<=` between `str` and `int`).

- [ ] **Step 3: Implement -- update `_toc_declared_page`**

Replace lines 1109-1114 of `src/chapter_segmentation/segmentation.py`:

Old:
```python
    value = entry.printed_page_number
    if value <= 0 or value > total_pages * _TOC_MAX_PAGE_NUMBER_RATIO:
        return None
    if entry.printed_roman and value > _ROMAN_PAGE_MAX_VALUE:
        return None
    return _format_page_number(value, entry.printed_roman)
```

New:
```python
    raw = entry.printed_page_number
    if raw is None:
        return None
    value = _parse_toc_page_number(raw)
    if value is None:
        # A real alternate-scheme page marker ("R42", "12a") always
        # carries at least one digit -- return it verbatim. A string with
        # no digit at all that also isn't a valid roman numeral ("mmmm",
        # "civil") is far more likely OCR/model noise than a genuine page
        # marker -- treat it as unknown, same outcome the old int-
        # sentinel path already produced for this case.
        return raw if any(ch.isdigit() for ch in raw) else None
    if value <= 0 or value > total_pages * _TOC_MAX_PAGE_NUMBER_RATIO:
        return None
    if entry.printed_roman and value > _ROMAN_PAGE_MAX_VALUE:
        return None
    return _format_page_number(value, entry.printed_roman)
```

Also update this function's docstring (lines 1091-1108) -- replace the sentence `"-1 is the sentinel both sources use for "not identified" ..."` with: `"None is what both sources use for "not identified" (a regex-found entry always has a real, valid value -- find_toc_candidates never constructs one otherwise; an LLM-found entry is None when it couldn't read one, see llm_extract_toc_entries)."`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -k TestTocDeclaredPage -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Write the failing test for the `_chapters_from_located` arithmetic fast-path**

Add to `TestChaptersFromLocatedPageNumberPriority` (around line 1281+):

```python
    def test_prefixed_next_entry_marker_skips_fast_path_without_crashing(self):
        # next_entry.printed_page_number is "R42" -- _parse_toc_page_number
        # can't turn that into an int to subtract 1 from, so the fast
        # "next entry's declared start minus one" path must be skipped
        # (not raise a TypeError), falling through to on-page extraction
        # of this chapter's own (trimmed) end page instead. Page numbers
        # kept small (1, not e.g. 10): _toc_declared_page's plausibility
        # ceiling is total_pages * _TOC_MAX_PAGE_NUMBER_RATIO(2.0) -- with
        # only 2 pages in this fixture, anything above 4 would be
        # (correctly) rejected as implausible before this test could even
        # exercise the fast-path-skip it's meant to check.
        pages = [self._FILLER + "\n\n12", self._FILLER + "\n\n13"]
        first = TocEntry(title="Introduction", printed_page_number="1", source_page_index=0)
        second = TocEntry(title="Appendix", printed_page_number="R42", source_page_index=1)
        located = [
            (first, ChapterStartMatch(index=0, score=100.0, margin=20.0)),
            (second, ChapterStartMatch(index=1, score=100.0, margin=20.0)),
        ]
        chapters = _chapters_from_located(pages, located)
        self.assertEqual(chapters[0]["citation_pages"], "1-12")
        self.assertEqual(chapters[0]["page_mapping_confidence"], "high")
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_segmentation.py -k test_prefixed_next_entry_marker_skips_fast_path_without_crashing -v`
Expected: FAIL with `TypeError: unsupported operand type(s) for -: 'str' and 'int'` -- the current code does `next_entry.printed_page_number - 1` directly.

- [ ] **Step 7: Implement -- update `_chapters_from_located`'s fast end-page path**

Replace lines 1416-1424 of `src/chapter_segmentation/segmentation.py`:

Old:
```python
        end_printed = None
        next_entry = located[i + 1][0] if i + 1 < len(located) else None
        if (
            next_entry is not None
            and next_entry.printed_roman == entry.printed_roman
            and _toc_declared_page(next_entry, total_pages) is not None
            and next_entry.printed_page_number - 1 > 0
        ):
            end_printed = _format_page_number(next_entry.printed_page_number - 1, entry.printed_roman)
```

New:
```python
        end_printed = None
        next_entry = located[i + 1][0] if i + 1 < len(located) else None
        next_value = (
            _parse_toc_page_number(next_entry.printed_page_number)
            if next_entry is not None and next_entry.printed_page_number is not None
            else None
        )
        if (
            next_entry is not None
            and next_entry.printed_roman == entry.printed_roman
            and _toc_declared_page(next_entry, total_pages) is not None
            and next_value is not None
            and next_value - 1 > 0
        ):
            end_printed = _format_page_number(next_value - 1, entry.printed_roman)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -k "TestChaptersFromLocatedPageNumberPriority or TestTocDeclaredPage" -v`
Expected: PASS (all tests in both classes, including the two new ones)

- [ ] **Step 9: Run the full test_segmentation.py suite**

Run: `uv run pytest tests/test_segmentation.py -v 2>&1 | tail -40`
Expected: PASS -- this file should now be fully green (Tasks 1-4 covered every `printed_page_number`-touching line in it).

- [ ] **Step 10: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "$(cat <<'EOF'
fix: _toc_declared_page preserves verbatim alternate page markers

A digit-based heuristic distinguishes a real alternate-scheme page
marker ("R42" -- always carries a digit) from OCR/model noise that
merely fails digit/roman parsing ("mmmm" -- no digit, invalid roman
notation): the former is returned verbatim, the latter still maps to
None/unknown exactly as the old int-sentinel path did.
_chapters_from_located's fast end-page path now parses via
_parse_toc_page_number before subtracting, instead of subtracting
directly from what is now a string field.
EOF
)"
```

---

### Task 5: `_candidate_to_toc_entry` bridge

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:1653-1659`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing test**

Add a new test class to `tests/test_segmentation.py` (this function isn't directly unit-tested today). First add `_candidate_to_toc_entry` to the import block (lines 35-45 or the block at line 46, alongside `_chapters_from_located`):

```python
from chapter_segmentation.segmentation import _candidate_to_toc_entry
```

```python
class TestCandidateToTocEntry(unittest.TestCase):
    def test_known_page_number_becomes_str(self):
        candidate = ChapterCandidate(title="Introduction", printed_page_number=42)
        entry = _candidate_to_toc_entry(candidate)
        self.assertEqual(entry.printed_page_number, "42")

    def test_none_page_number_stays_none(self):
        candidate = ChapterCandidate(title="Introduction", printed_page_number=None)
        entry = _candidate_to_toc_entry(candidate)
        self.assertIsNone(entry.printed_page_number)
```

(`ChapterCandidate` is already imported at line 50: `from chapter_segmentation.evidence.types import ChapterCandidate`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segmentation.py -k TestCandidateToTocEntry -v`
Expected: FAIL on `test_known_page_number_becomes_str` -- current code does `printed_page_number=candidate.printed_page_number if ... is not None else -1`, so the int `42` reaches `TocEntry.__post_init__` unchanged (`_normalize_printed_page_number(42)` -> `"42"` actually...) -- wait: check this carefully. Actually this specific assertion may already pass once Task 1 lands, since `__post_init__` normalizes ANY int input including one passed through this old code path. Confirm by running it: if it unexpectedly passes, that's fine -- move directly to Step 3's doc/comment cleanup, which is the real remaining change (the stray `-1` fallback is now redundant/misleading but not a behavioral bug, since `-1` itself normalizes to `None` too). Either way, apply Step 3's edit for clarity and correctness of intent, then confirm both tests pass.

- [ ] **Step 3: Implement -- update `_candidate_to_toc_entry`**

Replace lines 1653-1659 (originally 1668-1674 before Task 1-4 edits shifted line numbers -- locate by function name, not line number, at this point):

Old:
```python
def _candidate_to_toc_entry(candidate: ChapterCandidate) -> TocEntry:
    return TocEntry(
        title=candidate.title,
        authors=candidate.authors,
        printed_page_number=candidate.printed_page_number if candidate.printed_page_number is not None else -1,
        source_page_index=-1,
    )
```

New:
```python
def _candidate_to_toc_entry(candidate: ChapterCandidate) -> TocEntry:
    return TocEntry(
        title=candidate.title,
        authors=candidate.authors,
        printed_page_number=str(candidate.printed_page_number) if candidate.printed_page_number is not None else None,
        source_page_index=-1,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -k TestCandidateToTocEntry -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run the full test_segmentation.py and test_segmentation_strategies.py suites**

Run: `uv run pytest tests/test_segmentation.py tests/test_segmentation_strategies.py tests/test_harness.py -v 2>&1 | tail -40`
Expected: PASS -- `test_segmentation_strategies.py` and `test_harness.py` construct `TocEntry`/`ChapterCandidate` with bare ints throughout but never assert on `.printed_page_number`'s value directly (confirmed while researching this plan), so they need no edits at all.

- [ ] **Step 6: Commit**

```bash
git add src/chapter_segmentation/segmentation.py tests/test_segmentation.py
git commit -m "$(cat <<'EOF'
fix: _candidate_to_toc_entry converts int page number to str explicitly

Was relying on TocEntry's old -1-sentinel convention (candidate's None
-> -1); now explicit str(...)/None, matching the field's new str | None
type directly.
EOF
)"
```

---

### Task 6: `evaluation/dnb_toc_matching.py` -- string equivalence, sorting, serialization

**Files:**
- Modify: `evaluation/dnb_toc_matching.py`
- Test: `tests/test_dnb_toc_matching.py`

- [ ] **Step 1: Write the failing tests for the actual reported bug -- prefixed markers can now align**

Add to `tests/test_dnb_toc_matching.py`'s `TestAlignTocEntries` class:

```python
    def test_matches_on_prefixed_page_marker(self):
        # The real reported bug this whole change exists for: two
        # independent extractions that both correctly read "R42" for the
        # same line must be able to align -- the old int-with--1-sentinel
        # representation collapsed both to the same "unknown" value and
        # skipped them before matching was even attempted.
        a = [_entry("Appendix", "R42")]
        b = [_entry("Appendix", "R42")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_prefixed_marker_case_insensitively(self):
        a = [_entry("Appendix", "R42")]
        b = [_entry("Appendix", "r42")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_roman_numeral_case_difference(self):
        a = [_entry("Foreword", "VII")]
        b = [_entry("Foreword", "vii")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])
```

Update the `_entry` helper's type hint just above (line 14) for clarity (no behavior change, purely a signature update since it's now routinely called with strings too):
```python
def _entry(title: str, page: str | int | None, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0, authors=authors)
```

Update `test_merged_entries_sorted_by_printed_page_number` (line 164-169), which asserts on stored values directly:
```python
    def test_merged_entries_sorted_by_printed_page_number(self):
        h = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        l = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.printed_page_number for e in entries], ["9", "40"])
```

Every other existing test in this file (`test_no_match_when_either_page_unknown` using `-1`, `test_unknown_page_number_becomes_none` using `-1`, `test_known_page_number_becomes_string` expecting `"9"`, and all the plain page-number-as-int construction sites) needs **no changes** -- confirmed while researching this plan: `-1` still normalizes to `None` via `TocEntry.__post_init__`, and `toc_entry_to_gt_dict` already returns the (already-string) field directly either way.

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: FAIL on the 3 new tests (current `align_toc_entries` requires numeric equality, which `_parse_toc_page_number("R42")` can't produce) and on `test_merged_entries_sorted_by_printed_page_number` (currently sorts/returns ints).

- [ ] **Step 3: Implement -- `_pages_equivalent`, `align_toc_entries`, `gate_book`'s sort key, `toc_entry_to_gt_dict`**

In `evaluation/dnb_toc_matching.py`, add this function right after the `_ALIGN_SCORE_THRESHOLD` constant (line 17) and before `_candidate_titles`:

```python
from chapter_segmentation.segmentation import TocEntry, _parse_toc_page_number


def _pages_equivalent(a: str | None, b: str | None) -> bool:
    """True when two entries' printed_page_number values represent the
    same page. None never matches None (or anything else) -- "unknown"
    on either side means there is nothing to compare, same policy the
    old -1-sentinel-skip already enforced. Otherwise: exact string match
    first (handles a shared alternate-scheme marker like "R42" directly,
    with no numeric parsing needed at all); then numeric equality via
    _parse_toc_page_number (handles a case difference in a roman numeral,
    "VII" vs "vii", or a leading zero, "07" vs "7"); then a
    case-insensitive string match (handles a case difference in a
    non-roman marker, "R42" vs "r42")."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    parsed_a, parsed_b = _parse_toc_page_number(a), _parse_toc_page_number(b)
    if parsed_a is not None and parsed_a == parsed_b:
        return True
    return a.casefold() == b.casefold()
```

(Update the existing `from chapter_segmentation.segmentation import TocEntry` line at the top of the file (line 13) to include `_parse_toc_page_number` instead of adding a second import line -- i.e. the final import block should read `from chapter_segmentation.segmentation import TocEntry, _parse_toc_page_number`.)

Replace lines 59-61 inside `align_toc_entries`:

Old:
```python
    for i, entry_a in enumerate(a):
        if entry_a.printed_page_number == -1:
            continue
```

New:
```python
    for i, entry_a in enumerate(a):
        if entry_a.printed_page_number is None:
            continue
```

Replace line 66:

Old:
```python
            if entry_b.printed_page_number != entry_a.printed_page_number:
                continue
```

New:
```python
            if not _pages_equivalent(entry_a.printed_page_number, entry_b.printed_page_number):
                continue
```

Replace line 144 inside `gate_book`:

Old:
```python
    merged.sort(key=lambda e: (e.printed_page_number == -1, e.printed_page_number))
```

New:
```python
    def _page_sort_key(entry: TocEntry) -> tuple:
        value = _parse_toc_page_number(entry.printed_page_number) if entry.printed_page_number else None
        return (entry.printed_page_number is None, value if value is not None else 0, entry.printed_page_number or "")

    merged.sort(key=_page_sort_key)
```

Replace line 165 inside `toc_entry_to_gt_dict`:

Old:
```python
        "printed_page_number": str(entry.printed_page_number) if entry.printed_page_number != -1 else None,
```

New:
```python
        "printed_page_number": entry.printed_page_number,
```

Also update this function's docstring (lines 149-151), which currently says `"printed_page_number as a string, or None for the -1 "unknown" sentinel"` -- change to `"printed_page_number as-is (already str | None on TocEntry)"`.

And update `align_toc_entries`'s own docstring (lines 42-44), which currently says `"A pair (i, j) counts as a match only when both sides have a KNOWN printed_page_number (neither is the -1 "unknown" sentinel) that's numerically equal"` -- change to `"A pair (i, j) counts as a match only when both sides have a KNOWN printed_page_number (neither is None) that's equivalent per _pages_equivalent (exact string match, numeric match, or case-insensitive match)"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: PASS (all tests, including the 3 new ones)

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_matching.py tests/test_dnb_toc_matching.py
git commit -m "$(cat <<'EOF'
fix: align_toc_entries matches on string-equivalent page markers

Replaces numeric-only page comparison with _pages_equivalent (exact
string match, then numeric match via _parse_toc_page_number, then
case-insensitive string match) -- fixes two independent extractions
that both correctly read a section-prefixed marker like "R42" being
unable to align, since both previously collapsed to the same -1
"unknown" sentinel and were skipped before matching was attempted.
gate_book's merge sort and toc_entry_to_gt_dict's serialization are
updated for the str | None field directly, dropping the now-unneeded
-1-sentinel special-casing.
EOF
)"
```

---

### Task 7: `evaluation/scripts/arbitrate_dnb_toc.py` -- `_format_entry`

**Files:**
- Modify: `evaluation/scripts/arbitrate_dnb_toc.py:79-81`
- Test: `tests/test_arbitrate_dnb_toc.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_arbitrate_dnb_toc.py`'s `TestFormatBookReport` class:

```python
    def test_unknown_page_number_renders_as_question_mark(self):
        report = format_book_report(
            "book4", "Some Title", Path("/tmp/book4.pdf"),
            {"model-a": [TocEntry(title="Mystery", printed_page_number=None, source_page_index=0)]},
        )
        self.assertIn("p.   ?", report)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_arbitrate_dnb_toc.py -k test_unknown_page_number_renders_as_question_mark -v`
Expected: FAIL -- current `_format_entry` does `entry.printed_page_number if entry.printed_page_number != -1 else "?"`; since `None != -1` is `True`, it renders `None` itself (via `!s` formatting) as the page, producing `"p.None"` rather than `"p.   ?"`.

- [ ] **Step 3: Implement -- update `_format_entry`**

Replace line 80 of `evaluation/scripts/arbitrate_dnb_toc.py`:

Old:
```python
    page = entry.printed_page_number if entry.printed_page_number != -1 else "?"
```

New:
```python
    page = entry.printed_page_number if entry.printed_page_number is not None else "?"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_arbitrate_dnb_toc.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/arbitrate_dnb_toc.py tests/test_arbitrate_dnb_toc.py
git commit -m "$(cat <<'EOF'
fix: _format_entry checks printed_page_number is not None

Was checking != -1, the old int-sentinel convention -- with the field
now str | None, an unknown page rendered as the literal text "None"
instead of "?".
EOF
)"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest 2>&1 | tail -60`
Expected: PASS, 0 failures. This exercises every file identified during this plan's research as touching `printed_page_number` (`tests/test_segmentation.py`, `tests/test_segmentation_strategies.py`, `tests/test_harness.py`, `tests/test_dnb_toc_matching.py`, `tests/test_dnb_toc_vision.py`, `tests/test_arbitrate_dnb_toc.py`, `tests/test_generate_dnb_toc_ground_truth.py`, `tests/evidence/test_*.py`, `tests/test_nuextract_baseline.py`, `tests/test_nuextract2_common.py`, `tests/test_redaction.py`) plus everything else in the suite, confirming no unrelated regression.

- [ ] **Step 2: Sanity-check against real committed cache data**

Run:
```bash
uv run python -c "
from evaluation.dnb_toc_vision import load_cached_llm_entries
from pathlib import Path

cache_dir = Path('evaluation/corpus/dnb-toc-only/llm-cache')
entries = load_cached_llm_entries(cache_dir, '0745309941', 'qwen3-omni-30b-a3b-instruct')
print(len(entries), 'entries loaded')
print(entries[0].printed_page_number, type(entries[0].printed_page_number))
unknown = [e for e in entries if e.printed_page_number is None]
print(len(unknown), 'entries with unknown page number')
"
```
Expected: prints a positive entry count, `1 <class 'str'>` (or whatever the first entry's actual page reads as -- the key check is that it's a `str`, not an `int`), and a small `unknown` count (this book's cache has none of the 5 known legacy `-1`-sentinel entries found during research, but the load must not raise regardless). This confirms the real, previously-committed `-1`-int-sentinel cache data loads correctly with zero migration.

- [ ] **Step 3: If everything passes, this plan is complete.** No further commit needed for this task (verification only) -- if Step 1 or 2 surfaces a failure, fix it as a small follow-up commit before considering the branch done, following the same TDD discipline as the tasks above (update/add the failing test first, then fix the implementation).
