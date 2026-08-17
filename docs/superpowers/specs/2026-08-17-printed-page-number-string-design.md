# `printed_page_number` as a string, not a lossy int

## Problem

`TocEntry.printed_page_number` (`src/chapter_segmentation/segmentation.py`)
is typed `int`, with `-1` as an in-band sentinel for "unknown." Every
producer of a `TocEntry` -- the regex heuristic (`find_toc_candidates`), the
shared LLM-item parser (`_toc_items_to_entries`, used by both
`llm_extract_toc_entries` and `evaluation/dnb_toc_vision.py`'s vision
extraction) -- runs the raw printed-page text through `_parse_toc_page_number`,
which only understands two shapes: a pure digit run, or a pure roman-numeral
run (`vii`, `XL`). Anything else -- most importantly a real, empirically-seen
section-prefixed page marker like `"R42"` (DNB-digitized TOC scans; see
`evaluation/corpus/dnb-toc-only/`) -- returns `None`, which every call site
then collapses to the *same* `-1` sentinel used for "no page number was
printed at all." The original text is discarded; nothing on `TocEntry`
retains it.

Concretely, this breaks two things:

- **Alignment.** `evaluation/dnb_toc_matching.py`'s `align_toc_entries`
  explicitly skips any entry with `printed_page_number == -1` before trying
  to match it (`if entry_a.printed_page_number == -1: continue`) -- so two
  independent vision extractions that both correctly read `"R42"` for the
  same line can never be aligned, even though they agree. The entry still
  counts in `gate_book`'s agreement-rate denominator (`max(len(a), len(b))`)
  without ever being able to contribute to the numerator, silently dragging
  down (or outright failing) a book's whole-book agreement gate for reasons
  that have nothing to do with actual disagreement.
- **Ground truth fidelity.** In the resulting `.expected.json`, that entry
  surfaces as `"printed_page_number": null`, indistinguishable from a line
  that genuinely has no printed page number -- an information loss baked
  permanently into the ground truth.

The regex heuristic's own `_TOC_LINE_RE` capture group
(`\d{1,4}|[ivxlcdm]{1,6}|[IVXLCDM]{1,6}`) is narrower still -- it wouldn't
even recognize a line like `"Appendix ..... R42"` as a TOC line in the first
place. **Extending that recall is explicitly out of scope for this change**
(confirmed with the user) -- it's a real extraction-quality question with its
own false-positive tradeoffs, orthogonal to fixing how an already-captured
value is represented and compared. This design only concerns text that some
extractor (today: the two LLM-based paths, which already copy verbatim) has
already produced.

## Core change

`TocEntry.printed_page_number` moves from `int` (sentinel `-1`) to
`str | None`, storing the page marker exactly as read (`"42"`, `"vii"`,
`"R42"`), or `None` when genuinely absent/unreadable. `printed_roman`
(`bool`) stays as an explicit, caller-set flag -- unchanged semantics
("this entry lives in the book's roman-paginated front matter"), still set
by the same call sites, just now checked against a string instead of
derived from an already-parsed int.

`ChapterCandidate.printed_page_number` (`src/chapter_segmentation/evidence/types.py`)
is **not** changed -- it stays `int | None`. It already uses real `None`
(no sentinel), its sources (Crossref, the Zotero catalog) are essentially
always plain arabic page numbers, and touching it would ripple through
`evidence/fusion.py`'s and both strategy modules' sort keys
(`(c.printed_page_number is None, c.printed_page_number or 0)`) and
`crossref_strategy.py`'s disk-cache (de)serialization for no observed
benefit. The only place the two types meet is the bridge described below.

## BC mechanism: normalize once, at construction

A single function centralizes every legal input shape:

```python
def _normalize_printed_page_number(value: str | int | float | None) -> str | None:
    """Coerces any of TocEntry.printed_page_number's historical input
    shapes into the canonical str | None form. int/float exists only for
    old callers/cache files (the -1 sentinel, or a bare numeric JSON
    value some LLM response used instead of a string); a str is returned
    verbatim (stripped), preserving whatever text an extractor actually
    read -- this is the whole point of the change."""
    if isinstance(value, str):
        text = value.strip()
        return None if not text or text.lower() == "null" else text
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = int(value)
        return None if value == -1 else str(value)
    return None
```

`TocEntry.__post_init__` (new; the dataclass is `frozen=True`, so this uses
`object.__setattr__`) runs every constructed value through it. This is the
mechanism that makes the change backward-compatible: every existing call
site that builds a `TocEntry(printed_page_number=1, ...)` with a bare int
(most of `tests/test_segmentation_strategies.py`, `tests/test_harness.py`,
`tests/test_arbitrate_dnb_toc.py`'s fixture helper, ...) keeps working
completely unmodified -- the int becomes `"1"` transparently. Any on-disk
cache file written by the *old* code (`-1` sentinel, or a bare int from a
model that ignored the string instruction) loads correctly through
`evaluation/dnb_toc_vision.py`'s `load_cached_llm_entries`, which
constructs `TocEntry` directly from the cached JSON dict with no code
changes needed there at all.

`_toc_items_to_entries` (segmentation.py) simplifies to:

```python
printed_page_number = _normalize_printed_page_number(item.get("printed_page_number"))
printed_roman = (
    printed_page_number is not None
    and not printed_page_number.isdigit()
    and _parse_toc_page_number(printed_page_number) is not None
)
```

Same semantics as today (roman iff it parses via `_parse_toc_page_number`
AND isn't a plain digit string), just checked against the normalized
string instead of the pre-parsed int -- so `"R42"` correctly comes out
`printed_roman=False` (it isn't a digit run, but it also doesn't parse as
a roman numeral either). The int/float-tolerance branch that used to live
here (`if isinstance(printed, (int, float)): printed = str(int(printed))`)
moves into `_normalize_printed_page_number` itself, since it's the same
coercion `TocEntry.__post_init__` now also needs.

The regex heuristic (`find_toc_candidates`'s `_valid_entries`) passes
`m.group("page")` straight through as the string (today it passes
`_parse_toc_page_number(m.group("page"))`, an int) -- construction still
calls `_parse_toc_page_number` first to validate plausibility (implausible
page numbers must still be rejected before an entry is even created; that
gate doesn't move), but the value stored on the `TocEntry` is the original
captured text, not the parsed int.

## Equality and alignment

`evaluation/dnb_toc_matching.py`'s `align_toc_entries` gets a shared
equivalence check, replacing its current int-equality-only comparison:

```python
def _pages_equivalent(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False  # unknown never matches unknown -- same policy as today
    if a == b:
        return True
    parsed_a, parsed_b = _parse_toc_page_number(a), _parse_toc_page_number(b)
    if parsed_a is not None and parsed_a == parsed_b:
        return True  # e.g. "VII" vs "vii", or "07" vs "7"
    return a.casefold() == b.casefold()  # e.g. "R42" vs "r42"
```

This directly fixes the alignment problem: two extractions that both read
`"R42"` now match on the `a == b` branch (or `casefold()`, for a case
difference) without ever needing `_parse_toc_page_number` to understand the
prefix. The `entry_a.printed_page_number == -1: continue` skip is replaced
with `is None`, so a known-but-unparseable value is no longer treated
identically to "no value at all" -- it's now eligible for matching, just via
string equality instead of numeric equality.

`gate_book`'s merge sort (`e.printed_page_number == -1` sorting last)
becomes a three-part key that degrades gracefully for a non-numeric value:

```python
def _page_sort_key(entry: TocEntry) -> tuple:
    value = _parse_toc_page_number(entry.printed_page_number) if entry.printed_page_number else None
    return (entry.printed_page_number is None, value if value is not None else 0, entry.printed_page_number or "")
```

`None` sorts last; a parseable value sorts numerically; an unparseable
value (rare -- only when both sides of a merge genuinely agree on a
prefixed marker) sorts by raw text among other unparseable entries at that
same tier.

`toc_entry_to_gt_dict` simplifies from
`str(entry.printed_page_number) if entry.printed_page_number != -1 else None`
to returning `entry.printed_page_number` directly -- it's already the
target shape. **No change to the on-disk `.expected.json` format**: sampled
files already store this field as a string or `null`
(`evaluation/corpus/dnb-toc-only/*.expected.json`), confirming this was
always the intended serialized shape; only the lossy in-memory
representation was wrong.

## Arithmetic call sites

A few places in `segmentation.py` do integer arithmetic or bounds/ratio
validation directly against `entry.printed_page_number`. Each moves to
parsing on demand via the existing `_parse_toc_page_number`, with today's
"can't resolve, fall back to unknown" behavior preserved rather than
invented fresh:

- **`_toc_declared_page(entry, total_pages)`** -- currently: `value =
  entry.printed_page_number; if value <= 0 or value > ...: return None`.
  Becomes: `if entry.printed_page_number is None: return None`, then parse
  via `_parse_toc_page_number`. If parsing succeeds, keep today's exact
  ratio/roman-max-value validation and `_format_page_number` round-trip
  (unchanged behavior for every digit/roman entry). If parsing fails (a
  prefixed marker like `"R42"`) there is no numeric value to validate a
  ratio against -- return the raw string verbatim rather than `None`,
  since we know something real was printed there and have no basis to
  reject it.
- **`_chapters_from_located`** -- the one place doing subtraction directly
  (`next_entry.printed_page_number - 1`, used to derive a chapter's
  fallback end page as "one before the next chapter's declared start").
  Parses `next_entry.printed_page_number` via `_parse_toc_page_number`
  first; if that's `None` (including the prefixed-marker case), this
  fallback path is skipped exactly as it already is today when
  `_toc_declared_page(next_entry, ...)` returns `None` -- falls through to
  the existing `extract_printed_page_number`/`_infer_printed_page`/
  `_fallback_end_printed` chain, unchanged.
- **`_fallback_end_printed`** -- already operates on strings
  (`start_printed`, `raw`) via `_parse_toc_page_number` and
  `.isdigit()` scheme-match checks; not itself touched, since it never
  read `TocEntry.printed_page_number` directly.
- **`_candidate_to_toc_entry`** (the `ChapterCandidate` -> `TocEntry`
  bridge) -- becomes `str(candidate.printed_page_number) if
  candidate.printed_page_number is not None else None` (was: `... if ...
  is not None else -1`).

## Explicitly out of scope

- **`_TOC_LINE_RE`'s capture pattern** (regex heuristic recall for
  prefixed markers) -- confirmed out of scope with the user; a separate,
  riskier extraction-quality change.
- **`ChapterCandidate.printed_page_number`** and everything downstream of
  it (`evidence/fusion.py`, `crossref_strategy.py`, `zotero_catalog_strategy.py`)
  -- stays `int | None`; no observed non-arabic values from these sources.
- **`evaluation/nuextract_baseline.py`'s `match_toc_entries`/`_predicted_page`** --
  on closer reading, these score NuExtract's raw predicted output against
  `open-access`/`copyrighted-scans` ground truth's `citation_pages` range
  string (via `_expected_start_page`), not against any `TocEntry` or the
  dnb-toc-only corpus's `printed_page_number` field at all. They share the
  same "parse-to-int-first" shape as the bug being fixed, but no prefixed
  marker has been observed in that corpus's `citation_pages` values, and
  wiring in string-equivalence there would require `_expected_start_page`
  to also expose the raw un-parsed substring, purely speculatively. Left
  unchanged; flagged here in case that corpus ever needs it.
- **`evaluation/scripts/ground_truth_helper.py`** -- an independent,
  parallel implementation (its own `_TOC_LINE_RE`, arabic-only) used only
  to produce a first-pass draft a human then verifies by hand (see
  `evaluation/CLAUDE.md` Step 2/3); not wired to `TocEntry` at all.
- **`evaluation/metrics.py`'s `citation_pages_metrics`** -- parses the
  `citation_pages` string's own start/end substrings via
  `_parse_toc_page_number`, already returns `None` gracefully on an
  unparseable value; no `TocEntry` involved, no behavior change needed.

## Touch list

- `src/chapter_segmentation/segmentation.py` -- `TocEntry` (+
  `__post_init__`, new `_normalize_printed_page_number`), `find_toc_candidates`'s
  `_valid_entries`, `_toc_items_to_entries`, `_toc_declared_page`,
  `_chapters_from_located`, `_candidate_to_toc_entry`.
- `evaluation/dnb_toc_matching.py` -- `align_toc_entries` (new
  `_pages_equivalent`), `gate_book` (new `_page_sort_key`),
  `toc_entry_to_gt_dict` (simplifies).
- `evaluation/scripts/arbitrate_dnb_toc.py` -- `_format_entry`'s
  `page = entry.printed_page_number if entry.printed_page_number != -1 else "?"`
  becomes `is not None`.
- `evaluation/dnb_toc_vision.py` -- no code change expected (verify via
  tests); cache round-trip already goes through `TocEntry`'s new
  normalization automatically.
- Tests across `tests/test_segmentation.py`, `tests/test_dnb_toc_matching.py`,
  `tests/test_dnb_toc_vision.py`, `tests/test_arbitrate_dnb_toc.py`,
  `tests/test_generate_dnb_toc_ground_truth.py`,
  `tests/test_segmentation_strategies.py`, `tests/test_harness.py` --
  existing int-based fixtures should keep passing unmodified thanks to
  `__post_init__` normalization; new tests needed for: a prefixed marker
  (`"R42"`) surviving `_toc_items_to_entries` verbatim, `align_toc_entries`
  matching two `"R42"` entries that would previously both collapse to `-1`
  and be skipped, and the `__post_init__` BC normalization itself (old
  `-1` sentinel -> `None`, bare int -> str).

## Testing approach

Per this project's TDD convention: write the failing test for each behavior
above first (a `"R42"` round-trip through `_toc_items_to_entries`; two
`"R42"` `TocEntry`s aligning via `align_toc_entries`; a legacy
`printed_page_number=-1`/`printed_page_number=1` construction normalizing
correctly) before changing the implementation. The existing test suite
(unchanged fixtures using bare ints) is the regression backstop for BC.
