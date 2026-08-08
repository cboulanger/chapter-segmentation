# Improve `citation_pages` / `page_mapping_confidence` accuracy

Status: approved for planning
Date: 2026-08-08

## Problem

`_chapters_from_located` (`src/chapter_segmentation/segmentation.py:1143`)
derives `citation_pages` by calling `extract_printed_page_number` on exactly
the chapter's own `pdf_start_index` and `pdf_end_index` pages, and only those
two pages. `extract_printed_page_number` (line 937) only recognizes an
*isolated* line among the first/last 2 lines of a page that is nothing but
digits or roman numerals — it ignores a number appearing alongside other
text on the same line, and it never looks at any other page in the document.

Three consequences, confirmed empirically:

1. **Narrow per-page matching.** A running header/footer of the form `"12
   Chapter Title"` or `"Some Author  12"` is invisible to the isolated-line
   check — common typesetting puts the page number next to a title/author on
   the same line, alternating by recto/verso. `evaluation/scripts/
   ground_truth_helper.py`'s `extract_printed_number` already solves this for
   ground-truth authoring (an isolated-line check first, then a
   boundary-guarded trailing/leading-number check on the first non-URL line)
   but that logic was never ported into production.

2. **No use of document-level context.** Chapter-opening pages very commonly
   suppress the running header entirely (a standard typesetting convention).
   When that happens today, `citation_pages` goes straight to `null` and
   `page_mapping_confidence` to `"unmappable"` even though neighboring pages
   make the true printed-page offset obvious.

3. **TOC-declared page numbers are already parsed and thrown away.**
   `TocEntry.printed_page_number` (line 211) is populated by both TOC-entry
   sources: the heuristic regex parser (`find_toc_candidates`, always a
   valid, plausibility-checked int whenever a TOC line matches at all — see
   `_TOC_MAX_PAGE_NUMBER_RATIO`) and the LLM (`llm_extract_toc_entries`
   explicitly prompts for it, sentinel `-1` when it can't identify one; see
   line 625). This value is exactly the printed page number a reader would
   use to navigate to that chapter — read from the TOC's own (clean, never
   redaction-corrupted) text, not the located body page's running header —
   yet `_chapters_from_located` never looks at it, re-deriving everything
   from body-page text scanning instead.

Hand-verified ground truth (`evaluation/*.expected.json`) shows only 27/292
(9%) of real chapters are legitimately unmappable — no printed number visible
anywhere near either boundary. Production is doing substantially worse than
this achievable floor, which matters beyond the evaluation score: consuming
code that builds citation metadata for a located chapter (e.g. "pp. 12-34")
cannot do so at all when `citation_pages` is null, and `pdf_start_index`/
`pdf_end_index` are PDF-relative indices, not printed page numbers — they are
useless for that purpose on their own.

Neither field is read by `evaluation/metrics.py` today, so this gap has been
invisible to the evaluation score even as it degrades production usability.

## Fix

Production changes (§1-3c, §5) are all in `src/chapter_segmentation/
segmentation.py`. §1-3c are isolated to the printed-page-number extraction
path called from `_chapters_from_located`, with no signature changes to
`_chapters_from_located` itself or to any `analyze_attachment*` entry point —
every strategy benefits automatically. §5 is isolated to
`llm_extract_toc_entries`'s prompt and parsing, upstream of §1-3c (it
affects what `TocEntry.printed_page_number`/`printed_roman` *contain* for an
LLM-sourced entry, which §3's `_toc_declared_page` then *consumes*).
Evaluation changes (§4) are in `evaluation/metrics.py` and
`evaluation/generate_report.py`, mirroring how `precision_recall_f1` already
lives alongside the production code it scores.

### 1. Widen per-page extraction

Extend `extract_printed_page_number` with a second check, ported from
`ground_truth_helper.py`'s proven `extract_printed_number`: after the
existing isolated-line check fails, look at the first non-URL line of the
page for a boundary-guarded leading or trailing number.

```python
_TRAILING_PAGE_NUM_RE = re.compile(r"(?<![A-Za-z])(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)
_LEADING_PAGE_NUM_RE = re.compile(r"^(\d{1,4}|[ivxlcdm]{1,7})(?![A-Za-z])", re.IGNORECASE)


def _looks_like_url(line: str) -> bool:
    return "doi.org" in line or "http" in line.lower() or line.count(".") >= 3


def extract_printed_page_number(page_text: str) -> str | None:
    """Read the printed page number actually shown on a page: first an
    isolated numeral/roman-numeral header/footer line, then (if that finds
    nothing) a number embedded at either end of the page's first non-URL
    line -- running headers alternate between "<num> <author>" and
    "<title> ... <num>" depending on recto/verso convention. The boundary
    guards on the embedded check ((?<![A-Za-z]) / (?![A-Za-z])) stop a bare
    trailing/leading letter of an ordinary word (e.g. "Afterword", "Index")
    from false-positiving as a roman numeral -- same trap documented in
    evaluation/CLAUDE.md's "Known failure modes", this ports the fix
    already proven there. Returns None if neither check finds anything --
    callers must treat this as "unknown", never guess.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not _looks_like_url(ln)]
    if not lines:
        return None
    candidates = lines[:2] + lines[-2:]
    for line in candidates:
        if _PAGE_NUMBER_TOKEN_RE.match(line):
            return line
    first_line = lines[0]
    if len(first_line) < 120:
        match = _TRAILING_PAGE_NUM_RE.search(first_line)
        if match:
            return match.group(1)
        match = _LEADING_PAGE_NUM_RE.match(first_line)
        if match:
            return match.group(1)
    return None
```

This is strictly additive — every input that returned a value before still
returns the same value (the isolated-line check runs first, unchanged).

### 2. Document-level offset inference

New helper, `_page_number_anchors`, runs once per `_chapters_from_located`
call over every page:

```python
def _page_number_anchors(pages: list[str]) -> list[tuple[int, int, bool]]:
    """Every page whose printed number extract_printed_page_number can read
    directly, parsed to an int. Returns (page_index, value, is_roman)
    triples in page order -- the raw material _infer_printed_page uses to
    recover a page's printed number when its own text has no directly
    readable one (see design spec 2026-08-08).
    """
    anchors = []
    for index, text in enumerate(pages):
        raw = extract_printed_page_number(text)
        if raw is None:
            continue
        value = _parse_toc_page_number(raw)
        if value is not None:
            anchors.append((index, value, not raw.isdigit()))
    return anchors
```

(`_parse_toc_page_number` already exists at line 90 and handles both arabic
and roman parsing with the existing implausible-value guard.)

A new constant bounds how far an inference may reach from its nearest
anchor:

```python
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
    same index will not coincidentally agree on offset (roman "vi" continuing
    arithmetically into arabic numbering doesn't land on the arabic anchor's
    actual value) -- this naturally rejects inference across a numbering-
    scheme change without needing to special-case roman vs. arabic.
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

`_to_roman` is a new small helper (lowercase output, matching the existing
lowercase convention seen in `extract_printed_page_number`'s roman-numeral
test case):

```python
_ROMAN_NUMERALS = [
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
    (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
]


def _to_roman(value: int) -> str:
    result = []
    for amount, numeral in _ROMAN_NUMERALS:
        count, value = divmod(value, amount)
        result.append(numeral * count)
    return "".join(result)
```

### 3. Prefer the TOC-declared page number over body-page scanning

`TocEntry.printed_page_number` (heuristic: always valid when the entry
exists at all; LLM: `-1` sentinel when it couldn't identify one) is exactly
the number a reader would use to find this chapter, read from the TOC's own
text rather than the located body page's running header. It should be tried
*before* any of the page-scanning/interpolation machinery in §1-2, not
instead of it — §1-2 remain the fallback for whichever end this doesn't
cover (routinely the LLM's `-1` case, and always the very last chapter's end,
since there's no "next entry" to derive it from).

```python
def _format_page_number(value: int, is_roman: bool) -> str:
    return _to_roman(value) if is_roman else str(value)


def _toc_declared_page(entry: TocEntry, total_pages: int) -> str | None:
    """entry's own printed_page_number, formatted, when the TOC (heuristic
    or LLM) supplied a plausible one. The plausibility ceiling mirrors
    find_toc_candidates' own guard (_TOC_MAX_PAGE_NUMBER_RATIO) -- the LLM
    path has no equivalent check of its own, and an LLM could hallucinate
    an implausible value where the heuristic regex parser structurally
    cannot.
    """
    value = entry.printed_page_number
    if value is None or value <= 0 or value > total_pages * _TOC_MAX_PAGE_NUMBER_RATIO:
        return None
    return _format_page_number(value, entry.printed_roman)
```

The end page is derived the same way from the *next* located entry (which
may itself be a structural marker like a part divider — it still carries a
real, correctly-parsed `printed_page_number`, so no special-casing needed):
`next_entry.printed_page_number - 1`, only when `next_entry.printed_roman ==
entry.printed_roman` (guards against subtracting 1 across a roman-to-arabic
reset at a front-matter/body boundary — the two numbering schemes don't
share an arithmetic relationship, so a mismatch here means don't use this
path, fall through to §1-2/§3b instead). This is always tagged `"inferred"`
in step 3c below, never `"high"` — unlike the start, the end page's own
printed number is never independently observed here, only computed.

### 3b. Fallback end page, when neither the TOC nor the chapter's own page resolves

A chapter's own end page (blank/divider trimmed) is disproportionately
likely to be the one boundary with no on-page number and no nearby anchor —
trailing pages are often blank or divider pages that suppress the running
header, and `_PAGE_NUMBER_INFERENCE_MAX_GAP` may not reach past them to the
next real anchor. This is the last fallback in the chain (after §3's
TOC-derived next-entry value, §1's direct extraction, and §2's anchor
interpolation have all failed for the end — typically because the *next*
entry's own `printed_page_number` was itself `-1` or zone-mismatched).
Discarding a perfectly good start page just because the end can't be pinned
down loses information a consumer building "good enough" split/citation
metadata could still use — the true aim is splitting book PDFs and producing
usable citation ranges, not exact per-page precision on every boundary.

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
    start_value, end_value = _parse_toc_page_number(start_printed), _parse_toc_page_number(raw)
    if start_value is None or end_value is None or end_value < start_value:
        return None
    if start_printed.isdigit() != raw.isdigit():
        return None
    return raw
```

### 3c. Wire it all together in `_chapters_from_located`

Full priority chain, combining §1-3b. `start_is_high`/`end_is_high` track
whether that side came from an *observed* source (TOC-declared, or read
directly off its own page) as opposed to a *derived* one (anchor
interpolation, or the next-entry/next-page fallbacks) — `page_mapping_
confidence` is `"high"` only when both sides are observed.

```python
anchors = _page_number_anchors(pages)
...
for i, (entry, match) in enumerate(located):
    ...
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
        and _toc_declared_page(next_entry, total_pages) is not None  # validates plausibility
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

`_page_number_anchors(pages)` is computed once per `_chapters_from_located`
call (hoisted above the `for i, (entry, match) in enumerate(located)` loop,
alongside the existing `header_lines` computation), not once per chapter —
it does not depend on `entry`/`match`.

### 4. Evaluation: score start/end printed-page accuracy separately

A single exact-string comparison of the whole `"start-end"` value conflates
two very different failure modes: the start page is the load-bearing part of
a citation (a consumer can't reconstruct it from context), while the end
page can reasonably be approximated (from the next chapter's start, or
end-of-book — see §3b) and only actually hurts usability when it's
*under*-inclusive (cuts off real content). So the metric scores start and
end independently, with an end-page tolerance:

```python
_CITATION_END_OVER_INCLUSION_TOLERANCE = 3  # printed pages; see design spec 2026-08-08


@dataclass(frozen=True)
class CitationPageMetrics:
    start_coverage: float  # non-null found start / GT chapters with a non-null citation_pages, among matched chapters
    start_accuracy: float  # exact start match / GT chapters with a non-null citation_pages, among matched chapters
    end_coverage: float    # non-null found end / GT chapters with a non-null citation_pages, among matched chapters
    end_accuracy: float    # found end in [expected_end, expected_end + tolerance] / same denominator
    checked_count: int     # shared denominator for all four rates


def _split_citation_pages(value: str | None) -> tuple[str, str] | None:
    if value is None or "-" not in value:
        return None
    start, _, end = value.partition("-")
    return start, end


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
        expected_end_value, found_end_value = _parse_toc_page_number(expected_end), _parse_toc_page_number(found_end) if found_end else None
        end_correct += (
            found_end_value is not None
            and expected_end_value is not None
            and expected_end_value <= found_end_value <= expected_end_value + _CITATION_END_OVER_INCLUSION_TOLERANCE
        )
    n = len(checked)
    return CitationPageMetrics(start_covered / n, start_correct / n, end_covered / n, end_correct / n, n)
```

(`_parse_toc_page_number` is imported from `chapter_segmentation.segmentation`
the same way `evaluation/metrics.py` would need to for this comparison —
it already handles both arabic and roman parsing.)

`MicroAggregate`-style pooling across books is handled the same way the rest
of `generate_report.py` pools counts today: accumulate the five counters
(`checked`, `start_covered`, `start_correct`, `end_covered`, `end_correct`)
across books, divide once at the end.

`generate_report.py` gains these as extra columns (`Start accuracy`, `End
accuracy`) on the existing per-strategy standalone-results table, computed
identically for heuristic/outline/LLM. `RESULTS.md`'s per-strategy table and
surrounding bullets are updated with the real numbers once this is
implemented and re-run — same validation pattern as the LLM-fix spec.

### 5. Fix the LLM's roman-numeral blind spot

`llm_extract_toc_entries`'s prompt schema (`_LLM_TOC_EXTRACTION_PROMPT`,
line 488) currently asks the LLM for `"printed_page_number": 12` — a plain
JSON number, with no way to express a roman-numeral front-matter page
("vii"). The parsing code (line 625) mirrors this:
`int(printed) if isinstance(printed, (int, float)) else -1`, and never sets
`printed_roman` (always its `False` default) on any LLM-sourced `TocEntry`.
Two consequences: a front-matter chapter's real "vii" is either
misrepresented as arabic `7` or dropped to the `-1` sentinel; and even when
a plausible value does come through, §3's `_toc_declared_page`/end-derivation
logic can never treat an LLM-sourced entry as roman-numbered — real roman
front matter always falls through to the slower page-scanning fallback
chain instead of taking the fast, reliable TOC-declared path a
heuristic-found roman entry already gets.

Fix: ask the LLM for the page number **as printed** (a string, copied
verbatim) instead of asking it to normalize to an int, and parse that
string the same way `find_toc_candidates` already does — via the existing
`_parse_toc_page_number`, which handles both arabic and roman numerals with
its own implausibility guard.

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

```python
printed = item.get("printed_page_number")
if isinstance(printed, (int, float)):
    # Tolerate a model that ignores the string instruction and returns a
    # bare number anyway -- still unambiguous for the arabic case.
    printed = str(int(printed))
parsed_value = _parse_toc_page_number(printed.strip()) if isinstance(printed, str) else None
# -1 is a sentinel for "unknown" (LLM returned null, an unparseable
# value, or an implausible one, e.g. a roman numeral over
# _ROMAN_PAGE_MAX_VALUE) -- never a real printed page number.
printed_page_number = parsed_value if parsed_value is not None else -1
printed_roman = parsed_value is not None and not printed.strip().isdigit()
...
entries.append(TocEntry(
    title=title, printed_page_number=printed_page_number, source_page_index=-1,
    authors=authors, printed_roman=printed_roman,
))
```

Backward compatible with every existing test fixture that sends
`printed_page_number` as a raw JSON int (e.g. `1`, `null`) — those still
parse identically (an int is stringified before parsing, a digit string
parses to the same int, `null` still hits the `-1` sentinel). This directly
strengthens §3: an LLM-sourced front-matter entry can now take the same
fast, TOC-declared path a heuristic-found one already does, instead of
always falling through to page-scanning.

## Testing

- `extract_printed_page_number`: existing isolated-line cases unchanged;
  new cases for embedded trailing/leading numbers, the URL-line-skip, the
  boundary guard against "Afterword"/"Index" false-positiving, and the
  `len(first_line) < 120` cutoff.
- `_page_number_anchors`: empty pages, mixed hits/misses, roman entries
  correctly flagged `is_roman=True`.
- `_to_roman`: round-trip a handful of values through the existing
  `_parse_toc_page_number` (`_parse_toc_page_number(_to_roman(n)) == n`) for
  n in a representative range (1-49, staying under `_ROMAN_PAGE_MAX_VALUE`).
- `_infer_printed_page`: brackets on both sides with agreement (infers),
  disagreement (unmappable), gap exceeding `_PAGE_NUMBER_INFERENCE_MAX_GAP`
  on one or both sides (unmappable), only one side present at all
  (unmappable), roman anchor before / arabic anchor after straddling a
  scheme change (unmappable via the natural offset-mismatch, not a
  roman/arabic special case).
- `_toc_declared_page`: valid heuristic value formats correctly (arabic and
  roman); LLM's `-1` sentinel returns `None`; a value exceeding
  `total_pages * _TOC_MAX_PAGE_NUMBER_RATIO` (simulating an LLM
  hallucination) returns `None` even though it's a positive int.
- `_chapters_from_located`, priority order: entry has a valid
  `printed_page_number` -> used directly for the start regardless of what
  on-page scanning would have found, tagged `"high"` (this is the
  "silently trust the TOC-declared value" behavior — no cross-check against
  the on-page scan); entry's value is `-1` -> falls through to on-page
  extraction, then anchor interpolation; next entry has a valid
  `printed_page_number` in the *same* `printed_roman` zone -> end derived
  as `next.printed_page_number - 1`, tagged `"inferred"` even though the
  start in the same chapter is `"high"`; next entry's zone differs
  (`printed_roman` mismatch, e.g. this chapter is the last one before a
  roman-to-arabic transition) -> the derived-end path is skipped entirely,
  falls through to on-page extraction of the chapter's own end page; a case
  where the start page has no on-page number but is bracketed by anchors ->
  `page_mapping_confidence == "inferred"`, `citation_pages` populated; a
  case with anchors on both sides for both endpoints -> `"high"`; a case
  with no usable anchors at all -> `"unmappable"` (unchanged from today).
- `_fallback_end_printed`: start resolved, own end page unresolvable, next
  chapter's start-1 page has a printed number -> that number used, tagged
  `"inferred"`; last chapter in the book -> falls back to the last page's
  printed number; fallback candidate resolves to a *smaller* value than
  start (nonsensical ordering) -> rejected, stays unmappable; fallback
  candidate resolves to a different numbering scheme (roman vs. arabic)
  than start -> rejected, stays unmappable; fallback candidate itself
  unresolvable (direct extraction and interpolation both fail) -> stays
  unmappable.
- `citation_pages_metrics`: matched chapters with correct/incorrect start;
  end exactly matching, end over-inclusive within tolerance (counts
  correct), end over-inclusive beyond tolerance (counts wrong), end
  under-inclusive by even 1 page (counts wrong, no leniency); a `null`
  found `citation_pages` (both coverage rates should reflect it); an
  expected chapter with a null `citation_pages` excluded from the
  denominator; an expected chapter with no found match at all excluded
  from the denominator.
- Existing `tests/test_segmentation_accuracy.py` /
  `tests/test_public_evaluation_cache_parity.py` are unaffected by this
  change in isolation but the accuracy test's real-book pdf ranges are
  unchanged (this fix only affects `citation_pages`/`page_mapping_confidence`,
  never `pdf_start_index`/`pdf_end_index`).
- `llm_extract_toc_entries` parsing (§5): a roman-numeral string
  (`"printed_page_number": "vii"`) -> `printed_page_number == 7`,
  `printed_roman == True`; a plain digit string (`"12"`) ->
  `printed_page_number == 12`, `printed_roman == False`; a legacy bare int
  (`12`, matching every existing test fixture) -> unchanged behavior; `null`
  -> `-1` sentinel, `printed_roman == False`; an implausible roman string
  (e.g. `"mmmm"`, over `_ROMAN_PAGE_MAX_VALUE`) -> `-1` sentinel, same as an
  unparseable value today.

## Validation

After merging, regenerate the report (`uv run python
evaluation/generate_report.py --out /tmp/citation-check`) and update
`evaluation/RESULTS.md` with the new citation coverage/accuracy numbers per
strategy. Also re-run `--mode full` against the real KISSKI API afterward
(same procedure as the LLM-extraction fix) — this time not just because
`_chapters_from_located` is shared by every strategy, but because §5
changes the prompt `llm_extract_toc_entries` actually sends, so every
cached LLM result is stale against the new schema and needs regenerating
regardless of `--mode full`'s usual "did the fixed code change what gets
sent" trigger.

## Out of scope

- Changing `pdf_start_index`/`pdf_end_index` semantics or how chapter
  boundaries are located — this fix only affects the printed-page-number
  metadata attached to an already-located boundary.
- A configurable/tunable `_PAGE_NUMBER_INFERENCE_MAX_GAP` — a fixed constant
  is sufficient; revisit only if real data shows it's mistuned.
- Handling more than two numbering zones per document (e.g. a book with a
  separately-paginated appendix) beyond what nearest-anchor interpolation
  already handles naturally — not observed in the current evaluation corpus.
- The LLM misreading a roman numeral it saw correctly enough to attempt (e.g.
  transcribing "vii" as "vi") — §5 fixes the *schema's* inability to
  represent roman numerals at all, but a model transcription error is a
  general LLM-accuracy risk, the same category as misreading any other
  digit or title text, not something a prompt/parsing change can close.
- Cross-validating the TOC-declared value against the on-page scan when both
  are available — this round always trusts the TOC-declared value, per this
  spec's revision discussion.
