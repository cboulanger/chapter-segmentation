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

Two consequences, confirmed empirically:

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

All changes are in `src/chapter_segmentation/segmentation.py`, isolated to
the printed-page-number extraction path called from `_chapters_from_located`.
No signature changes to `_chapters_from_located` itself or to any
`analyze_attachment*` entry point — every strategy benefits automatically.

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

### 3. Wire into `_chapters_from_located`, add the "inferred" tier

```python
anchors = _page_number_anchors(pages)
...
for i, (entry, match) in enumerate(located):
    ...
    start_printed = extract_printed_page_number(pages[start_index])
    end_printed = extract_printed_page_number(pages[end_index])
    inferred = False
    if start_printed is None:
        start_printed = _infer_printed_page(start_index, anchors)
        inferred = inferred or start_printed is not None
    if end_printed is None:
        end_printed = _infer_printed_page(end_index, anchors)
        inferred = inferred or end_printed is not None
    if start_printed is not None and end_printed is not None:
        citation_pages = f"{start_printed}-{end_printed}"
        page_mapping_confidence = "inferred" if inferred else "high"
    else:
        citation_pages = None
        page_mapping_confidence = "unmappable"
```

`_page_number_anchors(pages)` is computed once per `_chapters_from_located`
call (hoisted above the `for i, (entry, match) in enumerate(located)` loop,
alongside the existing `header_lines` computation), not once per chapter —
it does not depend on `entry`/`match`.

### 4. Evaluation: score `citation_pages` coverage and accuracy

New function in `evaluation/metrics.py`:

```python
@dataclass(frozen=True)
class CitationPageMetrics:
    coverage: float  # non-null citation_pages / GT chapters with a non-null citation_pages, among matched chapters
    accuracy: float  # exact string match / GT chapters with a non-null citation_pages, among matched chapters
    checked_count: int  # denominator for both


def citation_pages_metrics(expected: list[dict], found: list[dict]) -> CitationPageMetrics:
    """Among chapters found's (pdf_start_index, pdf_end_index) correctly
    matches expected's (the same true-positive set precision_recall_f1
    scores), how well does citation_pages do -- restricted to expected
    chapters that themselves have a non-null citation_pages (an expected
    null means no printed number is visible anywhere on that chapter's real
    boundary pages, so there is nothing to score)."""
    found_by_range = {(c["pdf_start_index"], c["pdf_end_index"]): c for c in found}
    checked = [e for e in expected if e.get("citation_pages") is not None and (e["pdf_start_index"], e["pdf_end_index"]) in found_by_range]
    if not checked:
        return CitationPageMetrics(coverage=0.0, accuracy=0.0, checked_count=0)
    covered = sum(1 for e in checked if found_by_range[(e["pdf_start_index"], e["pdf_end_index"])].get("citation_pages") is not None)
    correct = sum(1 for e in checked if found_by_range[(e["pdf_start_index"], e["pdf_end_index"])].get("citation_pages") == e["citation_pages"])
    return CitationPageMetrics(coverage=covered / len(checked), accuracy=correct / len(checked), checked_count=len(checked))
```

`MicroAggregate`-style pooling across books is handled the same way the rest
of `generate_report.py` pools counts today: accumulate `checked`/`covered`/
`correct` totals across books, divide once at the end (mirroring
`MicroAggregate.compute`'s pattern; not adding a second class since these are
three independent counters, not the four `precision_recall_f1` already
pools).

`generate_report.py` gains this as an extra column pair (`Citation coverage`,
`Citation accuracy`) on the existing per-strategy standalone-results table,
computed identically for heuristic/outline/LLM. `RESULTS.md`'s per-strategy
table and surrounding bullets are updated with the real numbers once this is
implemented and re-run — same validation pattern as the LLM-fix spec.

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
- `_chapters_from_located`: a case where the start page has no on-page
  number but is bracketed by anchors -> `page_mapping_confidence ==
  "inferred"`, `citation_pages` populated; a case with anchors on both
  sides for both endpoints -> `"high"`; a case with no usable anchors at
  all -> `"unmappable"` (unchanged from today).
- `citation_pages_metrics`: matched chapters with matching/mismatching/
  missing `citation_pages`, an expected chapter with a null `citation_pages`
  excluded from the denominator, an expected chapter with no found match at
  all excluded from the denominator.
- Existing `tests/test_segmentation_accuracy.py` /
  `tests/test_public_evaluation_cache_parity.py` are unaffected by this
  change in isolation but the accuracy test's real-book pdf ranges are
  unchanged (this fix only affects `citation_pages`/`page_mapping_confidence`,
  never `pdf_start_index`/`pdf_end_index`).

## Validation

After merging, regenerate the report (`uv run python
evaluation/generate_report.py --out /tmp/citation-check`) and update
`evaluation/RESULTS.md` with the new citation coverage/accuracy numbers per
strategy. Also re-run `--mode full` against the real KISSKI API afterward
(same procedure as the LLM-extraction fix) so the LLM cache's
`citation_pages`/`page_mapping_confidence` fields reflect the fixed logic
too, since `_chapters_from_located` is shared by every strategy.

## Out of scope

- Changing `pdf_start_index`/`pdf_end_index` semantics or how chapter
  boundaries are located — this fix only affects the printed-page-number
  metadata attached to an already-located boundary.
- A configurable/tunable `_PAGE_NUMBER_INFERENCE_MAX_GAP` — a fixed constant
  is sufficient; revisit only if real data shows it's mistuned.
- Handling more than two numbering zones per document (e.g. a book with a
  separately-paginated appendix) beyond what nearest-anchor interpolation
  already handles naturally — not observed in the current evaluation corpus.
