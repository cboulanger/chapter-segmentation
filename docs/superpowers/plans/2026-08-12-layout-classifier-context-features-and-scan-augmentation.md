# Layout-Classifier Context Features and Scan-Noise Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sequence-context, per-book-normalized, and text-heading features to the layout TOC/chapter-first classifier, plus deterministic ALTO-level scan-noise augmentation of the open-access training pool, then re-run the LOBO evaluation and report.

**Architecture:** `evaluation/scripts/layout_features.py` grows two page-local features and a new second-pass function `add_book_context_features()` that adds five book-relative/context features (FEATURE_NAMES goes 10 → 17). A new module `evaluation/scripts/alto_scan_noise.py` writes seeded, perturbed copies of cached ALTO XML. `evaluation/scripts/evaluate_layout_toc_classifier.py` wires in the second pass and a `--scan-noise-augment` flag whose augmented rows join training folds but are never evaluated. Spec: `docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md`.

**Tech Stack:** Python 3 via `uv run`, `unittest` (run with `uv run pytest`), `xml.etree.ElementTree`, scikit-learn (unchanged), external `pdfalto` binary at `../pdfalto/pdfalto` (only needed for the final evaluation runs — the cached ALTO XML in `evaluation/corpus/*/.layout-cache/` already exists for all 70 books).

**Branch:** All work happens on a feature branch (`layout-classifier-context-features`), NOT a worktree — per explicit user instruction.

**Conventions that MUST be followed:**
- Tests are `unittest.TestCase` classes (not pytest-style functions), in `tests/`, run with `uv run pytest tests/<file>.py -v`.
- ALTO fixtures are inline module-level XML strings with hand-computed expected values (see `tests/test_layout_features.py`).
- The evaluation script itself is a manual run, never part of pytest.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 0: Create the feature branch

**Files:** none

- [ ] **Step 1: Branch off main**

```bash
cd /Users/cboulanger/Code/chapter-segmentation
git checkout main && git checkout -b layout-classifier-context-features
```

Expected: `Switched to a new branch 'layout-classifier-context-features'`

- [ ] **Step 2: Verify clean state and passing baseline tests**

```bash
git status --short
uv run pytest tests/test_layout_features.py tests/test_evaluate_layout_toc_classifier.py tests/test_layout_labels.py -q
```

Expected: no unexpected modifications (`evaluation/CLAUDE.md` and `evaluation/RESULTS.md` may show as modified from the prior session — leave them untouched); all tests pass.

---

### Task 1: Two new page-local features in `layout_features.py`

`last_text_vpos_fraction` (bottom edge of lowest text line / page height) and `top_line_heading_match` (highest line's text matches a heading pattern), plus two underscore-prefixed raw intermediates `_max_font_size` / `_modal_font_size` needed by Task 2. `FEATURE_NAMES` is split into `PAGE_FEATURE_NAMES` + `CONTEXT_FEATURE_NAMES` (context names defined here, computed in Task 2).

**Files:**
- Modify: `evaluation/scripts/layout_features.py`
- Test: `tests/test_layout_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layout_features.py` (and add `_is_heading_line` to the existing import):

```python
from evaluation.scripts.layout_features import _is_heading_line, extract_page_features
```

```python
class TestIsHeadingLine(unittest.TestCase):
    def test_keyword_headings_match_in_all_three_languages(self):
        for text in ("Chapter 3", "KAPITEL 12", "Chapitre IV", "Part 2",
                     "Teil 1", "Partie 3", "§ 5", "Chapter", "Teil"):
            self.assertTrue(_is_heading_line(text), text)

    def test_bare_numbers_and_roman_numerals_match(self):
        for text in ("3", "12.", "IV", "xii", "1998"):
            self.assertTrue(_is_heading_line(text), text)

    def test_ordinary_text_and_roman_lookalikes_do_not_match(self):
        # "mix"/"did" are spelled entirely with roman-numeral letters but are
        # not roman numerals -- the existing _TRAILING_NUMERAL_RE grammar
        # rejects them and this feature must inherit that.
        for text in ("The Origins of Law", "mix", "did", "Introduction",
                     "Partial results", "Chapters and verses", ""):
            self.assertFalse(_is_heading_line(text), text)


class TestNewPageLocalFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.alto_path = Path(self._tmp.name) / "fixture.alto.xml"
        self.alto_path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
        self.features = extract_page_features(str(self.alto_path))

    def test_last_text_vpos_fraction(self):
        # Page 0: lowest line bottom = VPOS 140 + HEIGHT 12 = 152.
        self.assertAlmostEqual(self.features[0]["last_text_vpos_fraction"], 152 / 600)
        # Page 1: lowest line bottom = VPOS 215 + HEIGHT 12 = 227.
        self.assertAlmostEqual(self.features[1]["last_text_vpos_fraction"], 227 / 600)

    def test_top_line_heading_match_is_zero_for_both_fixture_pages(self):
        # Page 0's top line reads "Intro 5", page 1's reads "Introduction"
        # -- neither is a bare number, roman numeral, or heading keyword.
        self.assertEqual(self.features[0]["top_line_heading_match"], 0.0)
        self.assertEqual(self.features[1]["top_line_heading_match"], 0.0)

    def test_raw_font_intermediates_are_emitted(self):
        self.assertAlmostEqual(self.features[0]["_max_font_size"], 10.0)
        self.assertAlmostEqual(self.features[0]["_modal_font_size"], 10.0)
        self.assertAlmostEqual(self.features[1]["_max_font_size"], 24.0)
        self.assertAlmostEqual(self.features[1]["_modal_font_size"], 10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: the new tests FAIL (`ImportError: cannot import name '_is_heading_line'`); all pre-existing tests still pass.

- [ ] **Step 3: Implement in `layout_features.py`**

Replace the `FEATURE_NAMES` block with:

```python
PAGE_FEATURE_NAMES = [
    "line_count",
    "width_mean",
    "width_var",
    "left_margin_mean",
    "left_margin_var",
    "trailing_number_fraction",
    "font_size_max_ratio",
    "top_block_is_large_font",
    "first_text_vpos_fraction",
    "line_density",
    "last_text_vpos_fraction",
    "top_line_heading_match",
]

# Computed by add_book_context_features (second pass), not by
# extract_page_features -- they need neighboring pages and book-level
# aggregates a single-page parse can't see.
CONTEXT_FEATURE_NAMES = [
    "prev_last_text_vpos_fraction",
    "prev_line_count_rel",
    "line_count_rel",
    "font_size_max_ratio_book",
    "page_position_fraction",
]

FEATURE_NAMES = PAGE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES
```

Add below `_TRAILING_NUMERAL_RE`:

```python
# A chapter/part heading keyword (English/German/French), optionally
# followed by an arabic or roman number. The bare-number and bare-roman
# branches of heading detection reuse _TRAILING_NUMERAL_RE so its
# lookalike rejections ("mix", "did", "civic") carry over.
_HEADING_KEYWORD_RE = re.compile(
    r"^(?:chapter|kapitel|chapitre|part|teil|partie|§)\.?\s*(?:\d{1,4}|[ivxlcdm]{1,7})?$",
    re.IGNORECASE,
)


def _is_heading_line(text: str) -> bool:
    """Whether a line of text looks like a chapter/part heading: a heading
    keyword (optionally numbered), a bare arabic number, or a bare roman
    numeral. Content-based, so it survives OCR'd scans whose font metadata
    is unreliable."""
    stripped = text.strip().rstrip(".:").strip()
    if not stripped:
        return False
    if _TRAILING_NUMERAL_RE.match(stripped):
        return True
    return _HEADING_KEYWORD_RE.match(stripped) is not None
```

Change `_font_ratio_and_top_block_flag` to also return the raw sizes (update its docstring's first sentence accordingly and the default return):

```python
def _font_ratio_and_top_block_flag(
    lines: list[ET.Element], font_sizes: dict[str, float], page_height: float
) -> tuple[float, float, float, float]:
```

with `return 1.0, 0.0, 0.0, 0.0` in the no-resolvable-line branch and

```python
    return font_size_max_ratio, top_block_is_large_font, max_size, modal_size
```

at the end (everything in between is unchanged).

In `extract_page_features`, update the empty-page branch:

```python
        if not lines:
            features[page_index] = {
                **{name: 0.0 for name in PAGE_FEATURE_NAMES},
                "_max_font_size": 0.0,
                "_modal_font_size": 0.0,
            }
            continue
```

update the unpacking call site:

```python
        font_size_max_ratio, top_block_is_large_font, max_font_size, modal_font_size = (
            _font_ratio_and_top_block_flag(lines, font_sizes, page_height)
        )
```

compute the two new features before the dict literal:

```python
        line_bottoms = [
            float(line.get("VPOS")) + float(line.get("HEIGHT", 0.0)) for line in lines
        ]
        top_line = min(lines, key=lambda line: float(line.get("VPOS")))
        top_line_text = " ".join(
            s.get("CONTENT", "") for s in top_line.findall(_ALTO_NS + "String")
        )
```

and extend the per-page dict with:

```python
            "last_text_vpos_fraction": max(line_bottoms) / page_height,
            "top_line_heading_match": float(_is_heading_line(top_line_text)),
            "_max_font_size": max_font_size,
            "_modal_font_size": modal_font_size,
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: all PASS (pre-existing assertions are per-key, so the extra keys don't break them).

- [ ] **Step 5: Run the classifier test file too (it imports FEATURE_NAMES)**

```bash
uv run pytest tests/test_evaluate_layout_toc_classifier.py -q
```

Expected: PASS — `_feature_row` builds vectors from `FEATURE_NAMES`, which is still a flat list of floats-keyed names.

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/layout_features.py tests/test_layout_features.py
git commit -m "feat: add last_text_vpos_fraction and top_line_heading_match page features

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `add_book_context_features()` second pass

**Files:**
- Modify: `evaluation/scripts/layout_features.py`
- Test: `tests/test_layout_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layout_features.py` (extend the import again):

```python
from evaluation.scripts.layout_features import (
    FEATURE_NAMES,
    PAGE_FEATURE_NAMES,
    _is_heading_line,
    add_book_context_features,
    extract_page_features,
)
```

```python
def _synthetic_page(line_count: float, last_vpos: float, max_font: float, modal_font: float) -> dict:
    page = {name: 0.0 for name in PAGE_FEATURE_NAMES}
    page["line_count"] = line_count
    page["last_text_vpos_fraction"] = last_vpos
    page["_max_font_size"] = max_font
    page["_modal_font_size"] = modal_font
    return page


class TestAddBookContextFeatures(unittest.TestCase):
    def setUp(self):
        # Three-page book: a sparse chapter-opening-like page between two
        # dense body pages. Line counts 30/8/20 -> median 20; modal fonts
        # 10/10/12 -> body font median 10.
        self.pages = {
            0: _synthetic_page(30.0, 0.95, 10.0, 10.0),
            1: _synthetic_page(8.0, 0.4, 24.0, 10.0),
            2: _synthetic_page(20.0, 0.9, 12.0, 12.0),
        }
        self.result = add_book_context_features(self.pages, total_pages=3)

    def test_output_keys_are_exactly_feature_names(self):
        for page in self.result.values():
            self.assertEqual(set(page.keys()), set(FEATURE_NAMES))

    def test_page_zero_sentinels(self):
        self.assertEqual(self.result[0]["prev_last_text_vpos_fraction"], 0.0)
        self.assertEqual(self.result[0]["prev_line_count_rel"], 0.0)

    def test_prev_page_wiring(self):
        self.assertAlmostEqual(self.result[1]["prev_last_text_vpos_fraction"], 0.95)
        self.assertAlmostEqual(self.result[1]["prev_line_count_rel"], 30.0 / 20.0)
        self.assertAlmostEqual(self.result[2]["prev_last_text_vpos_fraction"], 0.4)
        self.assertAlmostEqual(self.result[2]["prev_line_count_rel"], 8.0 / 20.0)

    def test_book_relative_normalization(self):
        self.assertAlmostEqual(self.result[0]["line_count_rel"], 30.0 / 20.0)
        self.assertAlmostEqual(self.result[1]["line_count_rel"], 8.0 / 20.0)
        self.assertAlmostEqual(self.result[1]["font_size_max_ratio_book"], 24.0 / 10.0)
        self.assertAlmostEqual(self.result[2]["font_size_max_ratio_book"], 12.0 / 10.0)

    def test_page_position_fraction(self):
        self.assertAlmostEqual(self.result[0]["page_position_fraction"], 0.0)
        self.assertAlmostEqual(self.result[1]["page_position_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(self.result[2]["page_position_fraction"], 2.0 / 3.0)

    def test_empty_pages_are_excluded_from_book_medians(self):
        pages = {
            0: _synthetic_page(0.0, 0.0, 0.0, 0.0),  # blank page
            1: _synthetic_page(10.0, 0.9, 10.0, 10.0),
            2: _synthetic_page(20.0, 0.9, 10.0, 10.0),
        }
        result = add_book_context_features(pages, total_pages=3)
        # Median over non-empty pages only: (10+20)/2 = 15, not median(0,10,20)=10.
        self.assertAlmostEqual(result[1]["line_count_rel"], 10.0 / 15.0)
        # A page with no resolvable font gets the neutral 1.0, matching
        # font_size_max_ratio's existing default.
        self.assertEqual(result[0]["font_size_max_ratio_book"], 1.0)

    def test_all_empty_book_falls_back_without_dividing_by_zero(self):
        pages = {0: _synthetic_page(0.0, 0.0, 0.0, 0.0)}
        result = add_book_context_features(pages, total_pages=1)
        self.assertEqual(result[0]["line_count_rel"], 0.0)
        self.assertEqual(result[0]["font_size_max_ratio_book"], 1.0)

    def test_raw_keys_are_stripped(self):
        for page in self.result.values():
            self.assertNotIn("_max_font_size", page)
            self.assertNotIn("_modal_font_size", page)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: FAIL with `ImportError: cannot import name 'add_book_context_features'`.

- [ ] **Step 3: Implement `add_book_context_features` in `layout_features.py`**

Append to the module:

```python
def add_book_context_features(
    page_features: dict[int, dict[str, float]], total_pages: int
) -> dict[int, dict[str, float]]:
    """Second pass over one book's extract_page_features output: adds the
    CONTEXT_FEATURE_NAMES (previous-page context, book-relative
    normalization, position in book) and strips the underscore-prefixed raw
    intermediates, returning vectors keyed exactly by FEATURE_NAMES.

    Book medians are computed over non-empty pages only (a scan's blank
    versos would otherwise drag them toward zero). The book-level body-font
    estimate is the median of per-page modal font sizes -- stable over the
    hundreds of near-identical jittery sizes OCR produces, where a single
    page's mode is arbitrary. Pages with no resolvable font get the neutral
    ratio 1.0, matching font_size_max_ratio's existing default."""
    non_empty = [f for f in page_features.values() if f["line_count"] > 0]
    median_line_count = (
        statistics.median(f["line_count"] for f in non_empty) if non_empty else 1.0
    ) or 1.0
    modal_sizes = [f["_modal_font_size"] for f in non_empty if f["_modal_font_size"] > 0]
    body_font_size = statistics.median(modal_sizes) if modal_sizes else 0.0

    result: dict[int, dict[str, float]] = {}
    for page_index, page in page_features.items():
        prev = page_features.get(page_index - 1)
        out = {name: page[name] for name in PAGE_FEATURE_NAMES}
        out["prev_last_text_vpos_fraction"] = (
            prev["last_text_vpos_fraction"] if prev is not None else 0.0
        )
        out["prev_line_count_rel"] = (
            prev["line_count"] / median_line_count if prev is not None else 0.0
        )
        out["line_count_rel"] = page["line_count"] / median_line_count
        max_font = page["_max_font_size"]
        out["font_size_max_ratio_book"] = (
            max_font / body_font_size if body_font_size > 0 and max_font > 0 else 1.0
        )
        out["page_position_fraction"] = page_index / total_pages if total_pages else 0.0
        result[page_index] = out
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/layout_features.py tests/test_layout_features.py
git commit -m "feat: add book-context second pass (prev-page, per-book normalization, position)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Wire the second pass into `build_feature_table`

**Files:**
- Modify: `evaluation/scripts/evaluate_layout_toc_classifier.py`
- Test: `tests/test_evaluate_layout_toc_classifier.py`

- [ ] **Step 1: Update the existing `TestBuildFeatureTable` tests**

`build_feature_table` will now pipe `extract_page_features` output through `add_book_context_features`, whose real implementation would KeyError on the tests' minimal `{"line_count": 1.0}` fakes. All four tests in `TestBuildFeatureTable` patch the pipeline already; add one more patch to each of the four `with patch(...)` blocks, alongside the existing two:

```python
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.add_book_context_features",
            side_effect=lambda page_features, total_pages: page_features,
        ):
```

Then add one new test to `TestBuildFeatureTable` that pins the wiring itself:

```python
    def test_pipes_extracted_features_through_book_context_pass(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "other"]},
        ]
        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}}
        context_pass = Mock(side_effect=lambda page_features, total_pages: {
            index: {**page, "context_added": 1.0} for index, page in page_features.items()
        })

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.add_book_context_features",
            context_pass,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        context_pass.assert_called_once_with(fake_features, 2)
        self.assertEqual(rows[0]["features"]["context_added"], 1.0)
```

- [ ] **Step 2: Run tests to verify the new one fails**

```bash
uv run pytest tests/test_evaluate_layout_toc_classifier.py -v
```

Expected: `test_pipes_extracted_features_through_book_context_pass` FAILS (`add_book_context_features` not imported / never called); the four updated tests still pass (patching a not-yet-used name is harmless with `patch`... it is NOT — `patch` raises `AttributeError` if the attribute doesn't exist on the module). So expected here: ALL FIVE fail with `AttributeError: <module ...> does not have the attribute 'add_book_context_features'`. That's the correct failing state.

- [ ] **Step 3: Implement the wiring**

In `evaluate_layout_toc_classifier.py`, change the import:

```python
from evaluation.scripts.layout_features import (
    FEATURE_NAMES,
    add_book_context_features,
    extract_page_features,
)
```

and in `build_feature_table`, replace

```python
        page_features = extract_page_features(str(alto_path))
```

with

```python
        page_features = add_book_context_features(
            extract_page_features(str(alto_path)), len(book["labels"])
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_evaluate_layout_toc_classifier.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/evaluate_layout_toc_classifier.py tests/test_evaluate_layout_toc_classifier.py
git commit -m "feat: run book-context feature pass in build_feature_table

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `alto_scan_noise.py` augmentation module

**Files:**
- Create: `evaluation/scripts/alto_scan_noise.py`
- Test: `tests/test_alto_scan_noise.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alto_scan_noise.py`:

```python
"""Unit tests for evaluation/scripts/alto_scan_noise.py's deterministic
ALTO-level scan-noise augmentation, against the same style of hand-built
fixture used by tests/test_layout_features.py."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from evaluation.scripts.alto_scan_noise import write_augmented_alto
from evaluation.scripts.layout_features import extract_page_features

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"

_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Styles>
    <TextStyle ID="body" FONTFAMILY="serif" FONTSIZE="10.0" FONTTYPE="serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
    <TextStyle ID="title" FONTFAMILY="sans" FONTSIZE="24.0" FONTTYPE="sans-serif" FONTWIDTH="proportional" FONTCOLOR="000000"/>
  </Styles>
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p1_w1" CONTENT="Introduction" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p1_w2" CONTENT="Body" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="48" VPOS="215" WIDTH="340" HEIGHT="12">
            <String ID="p1_w3" CONTENT="More" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t4" HPOS="48" VPOS="230" WIDTH="340" HEIGHT="12">
            <String ID="p1_w4" CONTENT="Text" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


class TestWriteAugmentedAlto(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_path = Path(self._tmp.name)
        self.source = tmp_path / "book.alto.xml"
        self.source.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
        self.output = tmp_path / "book.aug.alto.xml"
        write_augmented_alto(self.source, self.output, "book-key")

    def test_same_seed_is_byte_identical(self):
        second = Path(self._tmp.name) / "again.aug.alto.xml"
        write_augmented_alto(self.source, second, "book-key")
        self.assertEqual(self.output.read_bytes(), second.read_bytes())

    def test_different_key_differs(self):
        other = Path(self._tmp.name) / "other.aug.alto.xml"
        write_augmented_alto(self.source, other, "another-key")
        self.assertNotEqual(self.output.read_bytes(), other.read_bytes())

    def test_page_line_and_string_counts_preserved(self):
        src_root = ET.parse(self.source).getroot()
        aug_root = ET.parse(self.output).getroot()
        for tag in ("Page", "TextLine", "String"):
            self.assertEqual(
                len(list(src_root.iter(_ALTO_NS + tag))),
                len(list(aug_root.iter(_ALTO_NS + tag))),
                tag,
            )

    def test_string_content_is_untouched(self):
        src_texts = [s.get("CONTENT") for s in ET.parse(self.source).getroot().iter(_ALTO_NS + "String")]
        aug_texts = [s.get("CONTENT") for s in ET.parse(self.output).getroot().iter(_ALTO_NS + "String")]
        self.assertEqual(src_texts, aug_texts)

    def test_title_contrast_is_compressed(self):
        # The fixture's title/body ratio is 24/10 = 2.4; augmentation must
        # strictly compress it (alpha < 1 pulls sizes toward the body size).
        src_ratio = extract_page_features(str(self.source))[0]["font_size_max_ratio"]
        aug_ratio = extract_page_features(str(self.output))[0]["font_size_max_ratio"]
        self.assertAlmostEqual(src_ratio, 2.4)
        self.assertLess(aug_ratio, src_ratio)
        self.assertGreater(aug_ratio, 1.0)

    def test_font_sizes_are_jittered_into_clones(self):
        aug_root = ET.parse(self.output).getroot()
        sizes = {s.get("FONTSIZE") for s in aug_root.iter(_ALTO_NS + "TextStyle")}
        # Original 2 styles plus jittered clones -> strictly more distinct sizes.
        self.assertGreater(len(sizes), 2)

    def test_geometry_is_perturbed_but_bounded(self):
        src_lines = list(ET.parse(self.source).getroot().iter(_ALTO_NS + "TextLine"))
        aug_lines = list(ET.parse(self.output).getroot().iter(_ALTO_NS + "TextLine"))
        moved = 0
        for src, aug in zip(src_lines, aug_lines):
            for attr in ("HPOS", "VPOS", "WIDTH"):
                src_val = float(src.get(attr))
                aug_val = float(aug.get(attr))
                self.assertGreaterEqual(aug_val, 0.0)
                self.assertLess(abs(aug_val - src_val), 20.0, attr)
                if aug_val != src_val:
                    moved += 1
        self.assertGreater(moved, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_alto_scan_noise.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.alto_scan_noise'`.

- [ ] **Step 3: Create `evaluation/scripts/alto_scan_noise.py`**

```python
"""Deterministic ALTO-level scan-noise augmentation: rewrites a cached
pdfalto ALTO XML file to look like the OCR output of a degraded scan,
so the born-digital open-access training pool can teach the layout
classifier scan-shaped feature distributions. The three perturbations
each mimic a property measured in the real copyrighted-scans ALTO (see
docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md):
font-size jitter into many near-identical style clones, title/body
contrast compression, and small geometry noise. All randomness is seeded
from the book key, so output is reproducible and cacheable."""

import random
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

_ALTO_NS_URI = "http://www.loc.gov/standards/alto/ns-v3#"
_ALTO_NS = "{" + _ALTO_NS_URI + "}"

_STYLE_CLONES = 4  # jittered copies per original TextStyle
_FONT_JITTER = (0.96, 1.04)  # per-clone multiplicative font-size noise
_CONTRAST_ALPHA = (0.3, 0.7)  # per-book pull of every size toward the body size
_GEOMETRY_JITTER = (0.99, 1.01)  # per-value multiplicative box noise
_PAGE_OFFSET = (-5.0, 5.0)  # per-page global drift, in ALTO points


def write_augmented_alto(source_path: Path, output_path: Path, book_key: str) -> Path:
    """Writes a scan-noise-augmented copy of source_path to output_path.
    Page/line/String structure and all CONTENT text are preserved -- only
    font styles and box geometry change -- so the source book's page labels
    apply to the augmented copy unchanged."""
    rng = random.Random(f"scan-noise:{book_key}")
    ET.register_namespace("", _ALTO_NS_URI)
    tree = ET.parse(source_path)
    root = tree.getroot()

    styles_parent = root.find(_ALTO_NS + "Styles")
    original_styles = (
        list(styles_parent.iter(_ALTO_NS + "TextStyle")) if styles_parent is not None else []
    )
    sizes_by_id = {
        style.get("ID"): float(style.get("FONTSIZE"))
        for style in original_styles
        if style.get("ID") and style.get("FONTSIZE")
    }

    # Body size = usage-weighted modal font size over String style refs,
    # so a heavily-used body style outweighs a rarely-used title style.
    used_sizes = []
    for string in root.iter(_ALTO_NS + "String"):
        refs = (string.get("STYLEREFS") or "").split()
        if refs and refs[0] in sizes_by_id:
            used_sizes.append(sizes_by_id[refs[0]])
    body_size = statistics.mode(used_sizes) if used_sizes else 0.0
    alpha = rng.uniform(*_CONTRAST_ALPHA)

    clone_ids: dict[str, list[str]] = {}
    for style in original_styles:
        style_id = style.get("ID")
        if style_id not in sizes_by_id:
            continue
        compressed = (
            body_size + (sizes_by_id[style_id] - body_size) * alpha
            if body_size > 0
            else sizes_by_id[style_id]
        )
        ids = []
        for i in range(_STYLE_CLONES):
            clone = ET.SubElement(styles_parent, _ALTO_NS + "TextStyle", dict(style.attrib))
            clone_id = f"{style_id}_aug{i}"
            clone.set("ID", clone_id)
            clone.set("FONTSIZE", f"{compressed * rng.uniform(*_FONT_JITTER):.3f}")
            ids.append(clone_id)
        clone_ids[style_id] = ids

    for string in root.iter(_ALTO_NS + "String"):
        refs = (string.get("STYLEREFS") or "").split()
        if refs and refs[0] in clone_ids:
            string.set("STYLEREFS", rng.choice(clone_ids[refs[0]]))

    for page in root.iter(_ALTO_NS + "Page"):
        page_dx = rng.uniform(*_PAGE_OFFSET)
        page_dy = rng.uniform(*_PAGE_OFFSET)
        for line in page.iter(_ALTO_NS + "TextLine"):
            for attr, drift in (("HPOS", page_dx), ("VPOS", page_dy)):
                value = line.get(attr)
                if value is not None:
                    jittered = (float(value) + drift) * rng.uniform(*_GEOMETRY_JITTER)
                    line.set(attr, f"{max(0.0, jittered):.2f}")
            width = line.get("WIDTH")
            if width is not None:
                line.set("WIDTH", f"{float(width) * rng.uniform(*_GEOMETRY_JITTER):.2f}")

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_alto_scan_noise.py -v
```

Expected: all PASS. (If `test_geometry_is_perturbed_but_bounded` fails on the 20.0 bound: VPOS 230 * 1.01 + 5 ≈ 237.4, well inside; the bound only breaks if constants were changed.)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/alto_scan_noise.py tests/test_alto_scan_noise.py
git commit -m "feat: add deterministic ALTO-level scan-noise augmentation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `--scan-noise-augment` flag and the LOBO leakage rule

**Files:**
- Modify: `evaluation/scripts/evaluate_layout_toc_classifier.py`
- Test: `tests/test_evaluate_layout_toc_classifier.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate_layout_toc_classifier.py`:

```python
class TestBuildFeatureTableAugmentation(unittest.TestCase):
    def _books(self):
        return [
            {"key": "oa-book", "corpus": "open-access", "pdf_path": Path("/fake/oa-book.pdf"),
             "labels": ["toc", "other"]},
            {"key": "scan-book", "corpus": "copyrighted-scans", "pdf_path": Path("/fake/scan-book.pdf"),
             "labels": ["toc", "other"]},
        ]

    def test_augment_adds_marked_rows_for_open_access_only(self):
        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}}

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/cached.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.add_book_context_features",
            side_effect=lambda page_features, total_pages: page_features,
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.write_augmented_alto",
            return_value=Path("/fake/cached.aug.alto.xml"),
        ) as fake_augment, patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.Path.exists",
            return_value=False,
        ):
            rows = build_feature_table(
                self._books(), lambda corpus: Path("/fake/cache"), "pdfalto", augment=True
            )

        # oa-book contributes 2 plain + 2 augmented rows; scan-book only 2 plain.
        oa_rows = [r for r in rows if r["book_key"] == "oa-book"]
        scan_rows = [r for r in rows if r["book_key"] == "scan-book"]
        self.assertEqual(len(oa_rows), 4)
        self.assertEqual(len(scan_rows), 2)
        self.assertEqual(sum(1 for r in oa_rows if r.get("augmented")), 2)
        self.assertEqual(sum(1 for r in scan_rows if r.get("augmented")), 0)
        fake_augment.assert_called_once()

    def test_augment_off_by_default_adds_nothing(self):
        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}}
        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/cached.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.add_book_context_features",
            side_effect=lambda page_features, total_pages: page_features,
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.write_augmented_alto",
        ) as fake_augment:
            rows = build_feature_table(self._books(), lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(len(rows), 4)
        self.assertFalse(any(r.get("augmented") for r in rows))
        fake_augment.assert_not_called()


class TestAugmentedRowFoldRules(unittest.TestCase):
    def test_augmented_rows_train_other_folds_but_are_never_evaluated(self):
        # Two clean books plus augmented twins of both. The held-out book's
        # augmented twin must not appear in its test set (total_pages counts
        # only real pages), while other books' augmented rows still train.
        rows = []
        books = []
        for book_key in ("book-a", "book-b", "book-c"):
            book_rows, book = _clean_book(book_key)
            rows.extend(book_rows)
            books.append(book)
            for row in book_rows:
                rows.append({**row, "augmented": True})

        summary = evaluate_leave_one_book_out(rows, books)

        # Each synthetic book has 5 real pages; if augmented twins leaked
        # into the test set this would read 10.
        for result in summary["per_book"]:
            self.assertEqual(result["total_pages"], 5)
        # Perfectly-separable synthetic data must still fully recall.
        self.assertEqual(summary["full_recall_fraction"], 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evaluate_layout_toc_classifier.py -v
```

Expected: the two `TestBuildFeatureTableAugmentation` tests FAIL (`AttributeError` — `write_augmented_alto` not on the module / unexpected `augment` kwarg). `TestAugmentedRowFoldRules` FAILS: `total_pages` reads 10, not 5.

- [ ] **Step 3: Implement**

In `evaluate_layout_toc_classifier.py`, add the import:

```python
from evaluation.scripts.alto_scan_noise import write_augmented_alto
```

Replace `build_feature_table`'s book loop (keeping the docstring, extending it with a sentence: augmented rows carry `"augmented": True`, are labeled identically to their source book, and only open-access books are augmented; extraction-skip warnings are only counted for real rows):

```python
def build_feature_table(
    books: list[dict], cache_dir_for, pdfalto_bin: str, augment: bool = False
) -> list[dict]:
    rows = []
    skipped_by_label: dict[str, int] = {}
    for book in books:
        cache_dir = cache_dir_for(book["corpus"])
        alto_path = ensure_alto_xml(book["pdf_path"], cache_dir, pdfalto_bin)
        sources = [(alto_path, False)]
        if augment and book["corpus"] == "open-access":
            aug_path = cache_dir / f"{book['pdf_path'].stem}.aug.alto.xml"
            if not aug_path.exists():
                write_augmented_alto(alto_path, aug_path, book["key"])
            sources.append((aug_path, True))
        for source_path, is_augmented in sources:
            page_features = add_book_context_features(
                extract_page_features(str(source_path)), len(book["labels"])
            )
            for page_index, label in enumerate(book["labels"]):
                features = page_features.get(page_index)
                if features is None:
                    if not is_augmented:
                        skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                    continue
                row = {"book_key": book["key"], "features": features, "label": label}
                if is_augmented:
                    row["augmented"] = True
                rows.append(row)
    ...  # (warning block unchanged)
    return rows
```

In `evaluate_leave_one_book_out`, change two lines (and add to the docstring: rows marked `"augmented": True` join every training fold whose held-out book differs, but are never themselves evaluated — the held-out book's augmented twin is dropped for that fold):

```python
    book_keys = sorted({row["book_key"] for row in rows if not row.get("augmented")})
```

```python
        test_rows = [
            r for r in rows if r["book_key"] == held_out and not r.get("augmented")
        ]
```

(`train_rows` stays exactly as-is: augmented rows of other books are included automatically, and the held-out book's own augmented rows are excluded because they share its `book_key`.)

In `main()`, add the flag and pass it through:

```python
    parser.add_argument(
        "--scan-noise-augment",
        action="store_true",
        help=(
            "Augment each open-access book's training rows with a scan-noise-"
            "perturbed copy of its ALTO XML (cached as <key>.aug.alto.xml). "
            "Augmented rows are only ever used for training, never evaluated."
        ),
    )
```

```python
    rows = build_feature_table(
        books, cache_dir_for, pdfalto_bin, augment=args.scan_noise_augment
    )
```

- [ ] **Step 4: Run the full affected test suite**

```bash
uv run pytest tests/test_evaluate_layout_toc_classifier.py tests/test_layout_features.py tests/test_alto_scan_noise.py -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/evaluate_layout_toc_classifier.py tests/test_evaluate_layout_toc_classifier.py
git commit -m "feat: add --scan-noise-augment training augmentation with leakage-safe folds

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Run the two evaluation passes over the real 70-book corpus

**Files:** none modified — manual runs whose output feeds Task 7. The cached ALTO XML already exists for all 70 books; only the augmentation pass writes new `.aug.alto.xml` files (into `evaluation/corpus/open-access/.layout-cache/`, which is already git-ignored alongside the existing cache — verify with `git check-ignore` below).

- [ ] **Step 1: Verify the aug cache location is ignored**

```bash
git check-ignore -v evaluation/corpus/open-access/.layout-cache/x.aug.alto.xml || echo "NOT IGNORED"
```

Expected: a matching gitignore rule is printed. If `NOT IGNORED` appears, stop and add the pattern to the same gitignore that covers `.layout-cache/` before running.

- [ ] **Step 2: Run 2 — new features, no augmentation**

```bash
cd /Users/cboulanger/Code/chapter-segmentation
uv run python evaluation/scripts/evaluate_layout_toc_classifier.py \
  --pdfalto-bin ../pdfalto/pdfalto 2>&1 | tee /tmp/layout-run2-features.txt
```

Expected: completes in ~3 minutes; prints `Books evaluated: 70`, the aggregate lines, 70 per-book lines, and a MET/NOT MET verdict. Save the full output — Task 7 needs it verbatim.

- [ ] **Step 3: Run 3 — new features + scan-noise augmentation**

```bash
uv run python evaluation/scripts/evaluate_layout_toc_classifier.py \
  --pdfalto-bin ../pdfalto/pdfalto --scan-noise-augment 2>&1 | tee /tmp/layout-run3-augmented.txt
```

Expected: first run also writes 57 `.aug.alto.xml` files (a few extra minutes); same output shape. If any `WARNING` about skipped pages appears that did not appear in Run 2, note it for the report.

- [ ] **Step 4: Compute per-corpus breakdowns for both runs**

The script prints only the flat aggregate; split it by corpus with:

```bash
uv run python3 - <<'EOF'
import re
from pathlib import Path

corpus_dir = Path("evaluation/corpus")
corpus_of = {}
for corpus in ("open-access", "copyrighted-scans"):
    for pdf in (corpus_dir / corpus).glob("*.pdf"):
        corpus_of[pdf.stem] = corpus

line_re = re.compile(
    r"\s+(\S+): toc_recall=(n/a|\d+)%?, chapter_first_recall=(n/a|\d+)%?, "
    r"candidate_fraction=([\d.]+)%"
)
TOLERANCE = 90

for run_file in ("/tmp/layout-run2-features.txt", "/tmp/layout-run3-augmented.txt"):
    print(f"== {run_file}")
    rows = []
    for line in Path(run_file).read_text().splitlines():
        m = line_re.match(line)
        if m:
            rows.append(m.groups())
    print(f"parsed {len(rows)} books (expected 70)")
    for corpus in ("open-access", "copyrighted-scans"):
        sub = [r for r in rows if corpus_of.get(r[0]) == corpus]
        full = sum(
            1 for key, toc, chap, frac in sub
            if (toc == "n/a" or int(toc) > 0) and (chap == "n/a" or int(chap) >= TOLERANCE)
        )
        avg_frac = sum(float(r[3]) for r in sub) / len(sub)
        print(f"  {corpus}: {len(sub)} books, full_recall={full}/{len(sub)}"
              f" ({full/len(sub):.1%}), avg_candidate_fraction={avg_frac:.1f}%")
EOF
```

Expected: `parsed 70 books` for each run, then two per-corpus lines per run. (Note the toc pass condition `toc > 0`: the printed integer percent means any nonzero toc recall implies at least one caught page. `n/a` is the vacuous pass.)

- [ ] **Step 5: Sanity-check against the recorded baseline**

Baseline (from `evaluation/RESULTS.md`, "Follow-up: re-run over the grown 70-book corpus"): overall 64% / 10.0%; open-access 77.2% / 10.7%; copyrighted-scans 7.7% / 6.9%. If Run 2 or Run 3 is dramatically *worse* overall (e.g. below 50%), stop and investigate before reporting — a wiring bug (e.g. context features misaligned with pages) is more likely than a genuine regression.

---

### Task 7: Report in RESULTS.md, note in CLAUDE.md

**Files:**
- Modify: `evaluation/RESULTS.md` (new `###` subsection immediately after "### Follow-up: re-run over the grown 70-book corpus", before `## LLM-fallback results`)
- Modify: `evaluation/CLAUDE.md` (targeted-acquisition note)

- [ ] **Step 1: Write the RESULTS.md subsection**

Heading: `### Follow-up: context/normalized features and scan-noise augmentation`. Match the file's existing prose-plus-tables style (see the neighboring subsections). It MUST contain:

1. One paragraph linking the change to the spec
   (`docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md`)
   and naming the 7 new features and the augmentation flag.
2. The three-run comparison table (fill in actual numbers from Task 6):

```markdown
| run | full_recall_fraction | avg_candidate_fraction |
| --- | --- | --- |
| baseline (10 features) | 64% | 10.0% |
| + context/normalized features (17) | N% | N.N% |
| + `--scan-noise-augment` | N% | N.N% |
```

3. The per-corpus table for all three runs (baseline numbers above; Runs 2-3 from Task 6 Step 4).
4. Per-book callouts: what happened to the six 0%-`toc_recall` open-access books
   (`9781783748471`, `9782375460122`, `9783837660944`, `9783839447529`, `9783839468937`, `9783839470619`)
   and to the `copyrighted-scans` set (especially `9783848704316`, `9783789057366`, `9783899496291`, `dnb-36942798X`).
5. A verdict paragraph against the unchanged 90%/15% bar, and — if still NOT MET —
   naming the deferred directions (TOC-anchored matching, document-image deep
   learning) as next places to look, without scoping them.

- [ ] **Step 2: Add the acquisition note to `evaluation/CLAUDE.md`**

In the section describing how to choose/add new ground-truth books, add a short paragraph: when growing the corpus for the layout classifier, prefer scans, books with unnumbered first chapters, and books with weak title/body font contrast — the learning-curve check (RESULTS.md, model-architecture follow-up) showed generic well-produced open-access books are saturated for this classifier.

- [ ] **Step 3: Commit**

```bash
git add evaluation/RESULTS.md evaluation/CLAUDE.md
git commit -m "docs: report context-feature and scan-noise-augmentation evaluation results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Note: `evaluation/RESULTS.md` and `evaluation/CLAUDE.md` carry uncommitted changes from the prior session (the 70-book re-run report and the pdfalto binary-location note). Committing them together with this task's additions is correct — they are prerequisites of this follow-up's narrative. Mention this in the commit body if desired.

- [ ] **Step 4: Full test suite as a final gate**

```bash
uv run pytest tests/ -q
```

Expected: everything passes.

---

## Self-Review (completed at plan-writing time)

- **Spec coverage:** spec §1 → Task 1; §2 → Tasks 2-3; §3 → Task 4; §4 → Task 5; §5 → test steps of Tasks 1-5; §6 → Tasks 6-7; §7 → Task 7 Step 2. No gaps.
- **Placeholders:** none — every code step carries the full code; Task 7's `N%` cells are measurement outputs by design, with the exact source commands in Task 6.
- **Type consistency:** `add_book_context_features(page_features, total_pages)` used identically in Tasks 2, 3, 5; `write_augmented_alto(source_path, output_path, book_key)` identical in Tasks 4, 5; `PAGE_FEATURE_NAMES`/`CONTEXT_FEATURE_NAMES`/`FEATURE_NAMES` defined once in Task 1 and only referenced after.
