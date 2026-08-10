# Layout-based TOC/Chapter-First-Page Classifier Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrofit the ground-truth corpus with a `"toc"` page-range field, wire it into the automated CrossRef GT pipeline, then build and run a leave-one-book-out pilot that decides whether a `pdfalto`-layout-feature classifier is a viable pre-filter for TOC/chapter-first pages.

**Architecture:** Two small, independently-testable pure-function modules
(`layout_labels.py` for ground-truth label derivation, `layout_features.py`
for ALTO-XML-to-feature-vector parsing) plus a thin `pdfalto_runner.py`
subprocess/cache wrapper, composed by a manual-run pilot script that trains
a shallow scikit-learn classifier per leave-one-book-out fold and reports
against the design spec's recall-first decision bar.

**Tech Stack:** Python 3.12, pypdf, scikit-learn (new eval-only optional
dependency), the locally-built `pdfalto` binary (external, not vendored),
`unittest`-style tests (this project's existing convention — see
`tests/test_harness.py`).

**Spec:** `docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`

---

### Task 1: `toc_page_range()` helper in `ground_truth_helper.py`

**Files:**
- Modify: `evaluation/scripts/ground_truth_helper.py` (insert after `find_toc_pages`, currently ending at line 103)
- Test: `tests/test_ground_truth_helper.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ground_truth_helper.py`:

```python
"""Unit tests for evaluation/scripts/ground_truth_helper.py's
toc_page_range()."""

import unittest

from evaluation.scripts.ground_truth_helper import toc_page_range


class TestTocPageRange(unittest.TestCase):
    def test_empty_set_returns_none(self):
        self.assertIsNone(toc_page_range(set()))

    def test_single_page(self):
        self.assertEqual(toc_page_range({5}), (5, 5))

    def test_contiguous_run(self):
        self.assertEqual(toc_page_range({7, 5, 6}), (5, 7))

    def test_two_separate_runs_returns_none(self):
        self.assertIsNone(toc_page_range({5, 6, 20}))

    def test_two_adjacent_singletons_with_gap_returns_none(self):
        self.assertIsNone(toc_page_range({5, 7}))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ground_truth_helper.py -v`
Expected: FAIL with `ImportError: cannot import name 'toc_page_range'`

- [ ] **Step 3: Add the function**

In `evaluation/scripts/ground_truth_helper.py`, insert this new function
immediately after `find_toc_pages` (after line 103, before
`locate_chapter_start`):

```python
def toc_page_range(toc_pages: set[int]) -> tuple[int, int] | None:
    """Collapses find_toc_pages' candidate index set into a single
    contiguous (start, end) range. Returns None if the set is empty or
    spans more than one contiguous run -- e.g. a back-matter index page
    that also matched the same "title ... number" structural pattern --
    since that ambiguity should be resolved by a human, not guessed."""
    if not toc_pages:
        return None
    ordered = sorted(toc_pages)
    runs = [[ordered[0]]]
    for i in ordered[1:]:
        if i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    if len(runs) != 1:
        return None
    return runs[0][0], runs[0][-1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ground_truth_helper.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/ground_truth_helper.py tests/test_ground_truth_helper.py
git commit -m "feat: add toc_page_range helper for collapsing TOC-page candidates into a range"
```

---

### Task 2: `layout_labels.py` — ground-truth page labels

**Files:**
- Create: `evaluation/scripts/layout_labels.py`
- Test: `tests/test_layout_labels.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_layout_labels.py`:

```python
"""Unit tests for evaluation/scripts/layout_labels.py's page_labels()."""

import unittest

from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST, LABEL_OTHER, LABEL_TOC, page_labels


class TestPageLabels(unittest.TestCase):
    def test_missing_toc_key_returns_none(self):
        expected = {"chapters": []}
        self.assertIsNone(page_labels(expected, total_pages=5))

    def test_null_toc_labels_only_chapters(self):
        expected = {"chapters": [{"pdf_start_index": 2, "pdf_end_index": 4}], "toc": None}
        labels = page_labels(expected, total_pages=5)
        self.assertEqual(
            labels,
            [LABEL_OTHER, LABEL_OTHER, LABEL_CHAPTER_FIRST, LABEL_OTHER, LABEL_OTHER],
        )

    def test_toc_range_and_chapters_labeled(self):
        expected = {
            "chapters": [
                {"pdf_start_index": 3, "pdf_end_index": 6},
                {"pdf_start_index": 7, "pdf_end_index": 9},
            ],
            "toc": {"toc_start_index": 1, "toc_end_index": 2},
        }
        labels = page_labels(expected, total_pages=10)
        self.assertEqual(
            labels,
            [
                LABEL_OTHER, LABEL_TOC, LABEL_TOC, LABEL_CHAPTER_FIRST,
                LABEL_OTHER, LABEL_OTHER, LABEL_OTHER, LABEL_CHAPTER_FIRST,
                LABEL_OTHER, LABEL_OTHER,
            ],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layout_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.layout_labels'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/scripts/layout_labels.py`:

```python
"""Per-page ground-truth labels (toc / chapter_first / other) derived from
an .expected.json-shaped dict, for the layout-classifier pilot -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

LABEL_TOC = "toc"
LABEL_CHAPTER_FIRST = "chapter_first"
LABEL_OTHER = "other"


def page_labels(expected: dict, total_pages: int) -> list[str] | None:
    """Returns a total_pages-length list of per-page labels, or None if this
    book has no usable "toc" field yet -- the key being entirely absent
    means "not yet retrofitted / flagged for manual review", which is
    different from an explicit "toc": null ("confirmed, no TOC page
    exists"), a book that IS usable and simply contributes no
    toc-labeled pages."""
    if "toc" not in expected:
        return None

    labels = [LABEL_OTHER] * total_pages
    toc = expected["toc"]
    if toc is not None:
        for index in range(toc["toc_start_index"], toc["toc_end_index"] + 1):
            labels[index] = LABEL_TOC
    for chapter in expected["chapters"]:
        labels[chapter["pdf_start_index"]] = LABEL_CHAPTER_FIRST
    return labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layout_labels.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/layout_labels.py tests/test_layout_labels.py
git commit -m "feat: add page_labels() for deriving per-page toc/chapter_first/other labels"
```

---

### Task 3: `layout_features.py` — ALTO-XML-to-feature-vector parsing

**Files:**
- Create: `evaluation/scripts/layout_features.py`
- Test: `tests/test_layout_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_layout_features.py`. The fixture is a hand-built,
two-page ALTO XML document: page 1 mimics a TOC page (three short lines,
each ending in a bare page-number token, uniform left margin, no font-size
spike); page 2 mimics a chapter-opening page (a large-font title line
followed by two body-text lines with a consistent, narrower left margin).
Expected feature values are hand-computed from the fixture's coordinates.

```python
"""Unit tests for evaluation/scripts/layout_features.py's ALTO-XML-to-
feature-vector parsing, against a small hand-built two-page fixture (one
TOC-shaped page, one chapter-opening-shaped page) with hand-computed
expected values."""

import tempfile
import unittest
from pathlib import Path

from evaluation.scripts.layout_features import extract_page_features

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
          <TextLine ID="p1_t1" HPOS="50" VPOS="100" WIDTH="100" HEIGHT="12">
            <String ID="p1_w1" CONTENT="Intro" HPOS="50" WIDTH="80" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w2" CONTENT="5" HPOS="130" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="50" VPOS="120" WIDTH="120" HEIGHT="12">
            <String ID="p1_w3" CONTENT="ChapterTwo" HPOS="50" WIDTH="100" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w4" CONTENT="12" HPOS="150" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="50" VPOS="140" WIDTH="140" HEIGHT="12">
            <String ID="p1_w5" CONTENT="ChapterThree" HPOS="50" WIDTH="120" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p1_w6" CONTENT="99" HPOS="170" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p2_b1">
          <TextLine ID="p2_t1" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p2_w1" CONTENT="Introduction" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p2_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p2_w2" CONTENT="This" HPOS="48" WIDTH="30" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w3" CONTENT="is" HPOS="80" WIDTH="20" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w4" CONTENT="text" HPOS="102" WIDTH="290" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p2_t3" HPOS="48" VPOS="215" WIDTH="340" HEIGHT="12">
            <String ID="p2_w5" CONTENT="More" HPOS="48" WIDTH="40" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w6" CONTENT="body" HPOS="90" WIDTH="40" HEIGHT="12" STYLEREFS="body"/>
            <String ID="p2_w7" CONTENT="content" HPOS="132" WIDTH="260" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


class TestExtractPageFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.alto_path = Path(self._tmp.name) / "fixture.alto.xml"
        self.alto_path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
        self.features = extract_page_features(str(self.alto_path))

    def test_returns_zero_based_page_indices(self):
        self.assertEqual(set(self.features.keys()), {0, 1})

    def test_toc_like_page_features(self):
        f = self.features[0]
        self.assertEqual(f["line_count"], 3.0)
        self.assertAlmostEqual(f["width_mean"], 120.0)
        self.assertAlmostEqual(f["width_var"], 400.0)
        self.assertAlmostEqual(f["left_margin_mean"], 50.0)
        self.assertAlmostEqual(f["left_margin_var"], 0.0)
        self.assertEqual(f["trailing_number_fraction"], 1.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 1.0)
        self.assertEqual(f["top_block_is_large_font"], 0.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 100 / 600)
        self.assertAlmostEqual(f["line_density"], 3 / 600)

    def test_chapter_opening_page_features(self):
        f = self.features[1]
        self.assertEqual(f["line_count"], 3.0)
        self.assertAlmostEqual(f["width_mean"], 830 / 3)
        self.assertAlmostEqual(f["width_var"], 12033.333333333334, places=2)
        self.assertAlmostEqual(f["left_margin_mean"], 296 / 3)
        self.assertAlmostEqual(f["left_margin_var"], 7701.333333333333, places=2)
        self.assertEqual(f["trailing_number_fraction"], 0.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 2.4)
        self.assertEqual(f["top_block_is_large_font"], 1.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 50 / 600)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layout_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.layout_features'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/scripts/layout_features.py`:

```python
"""Per-page geometric feature extraction from a pdfalto ALTO XML file --
the classifier's input for
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import re
import statistics
import xml.etree.ElementTree as ET

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"
_TRAILING_NUMERAL_RE = re.compile(r"^[0-9]{1,4}$|^[ivxlcdm]{1,7}$", re.IGNORECASE)

FEATURE_NAMES = [
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
]


def _parse_font_sizes(root: ET.Element) -> dict[str, float]:
    """Maps each TextStyle ID to its FONTSIZE, from the document's Styles
    block."""
    sizes = {}
    for style in root.iter(_ALTO_NS + "TextStyle"):
        style_id = style.get("ID")
        font_size = style.get("FONTSIZE")
        if style_id and font_size:
            sizes[style_id] = float(font_size)
    return sizes


def _line_font_size(line: ET.Element, font_sizes: dict[str, float]) -> float | None:
    """Font size of a TextLine's first String, or None if unavailable."""
    string = line.find(_ALTO_NS + "String")
    if string is None:
        return None
    style_ref = string.get("STYLEREFS")
    if not style_ref:
        return None
    return font_sizes.get(style_ref.split()[0])


def extract_page_features(alto_xml_path: str) -> dict[int, dict[str, float]]:
    """Parses a pdfalto ALTO XML file into a per-page feature dict, keyed by
    0-based PDF page index (ALTO's PHYSICAL_IMG_NR is 1-based). A page with
    no text lines at all gets an all-zero feature vector."""
    tree = ET.parse(alto_xml_path)
    root = tree.getroot()
    font_sizes = _parse_font_sizes(root)

    features: dict[int, dict[str, float]] = {}
    for page in root.iter(_ALTO_NS + "Page"):
        page_index = int(page.get("PHYSICAL_IMG_NR")) - 1
        page_height = float(page.get("HEIGHT"))
        lines = list(page.iter(_ALTO_NS + "TextLine"))

        if not lines:
            features[page_index] = {name: 0.0 for name in FEATURE_NAMES}
            continue

        widths = [float(line.get("WIDTH")) for line in lines]
        left_margins = [float(line.get("HPOS")) for line in lines]
        vpositions = [float(line.get("VPOS")) for line in lines]

        trailing_hits = 0
        for line in lines:
            strings = line.findall(_ALTO_NS + "String")
            if strings and _TRAILING_NUMERAL_RE.match(strings[-1].get("CONTENT", "").strip()):
                trailing_hits += 1

        line_sizes = [
            s for s in (_line_font_size(line, font_sizes) for line in lines) if s is not None
        ]
        if line_sizes:
            modal_size = statistics.mode(line_sizes)
            max_size = max(line_sizes)
            font_size_max_ratio = max_size / modal_size if modal_size else 1.0
            max_size_line_index = line_sizes.index(max_size)
            top_block_is_large_font = float(
                font_size_max_ratio > 1.3 and vpositions[max_size_line_index] < page_height / 5
            )
        else:
            font_size_max_ratio = 1.0
            top_block_is_large_font = 0.0

        features[page_index] = {
            "line_count": float(len(lines)),
            "width_mean": statistics.mean(widths),
            "width_var": statistics.variance(widths) if len(widths) > 1 else 0.0,
            "left_margin_mean": statistics.mean(left_margins),
            "left_margin_var": statistics.variance(left_margins) if len(left_margins) > 1 else 0.0,
            "trailing_number_fraction": trailing_hits / len(lines),
            "font_size_max_ratio": font_size_max_ratio,
            "top_block_is_large_font": top_block_is_large_font,
            "first_text_vpos_fraction": min(vpositions) / page_height,
            "line_density": len(lines) / page_height,
        }

    return features
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layout_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/layout_features.py tests/test_layout_features.py
git commit -m "feat: add extract_page_features() for ALTO-XML-to-feature-vector parsing"
```

---

### Task 4: `pdfalto_runner.py` — subprocess wrapper with caching

**Files:**
- Create: `evaluation/scripts/pdfalto_runner.py`
- Test: `tests/test_pdfalto_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pdfalto_runner.py`:

```python
"""Unit tests for evaluation/scripts/pdfalto_runner.py's pure logic: binary
resolution and cache-hit/cache-miss behavior. Running the real pdfalto
binary against a real PDF is exercised manually -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary


class TestResolvePdfaltoBinary(unittest.TestCase):
    def test_explicit_arg_wins(self):
        self.assertEqual(resolve_pdfalto_binary("/custom/pdfalto"), "/custom/pdfalto")

    @patch.dict("os.environ", {"PDFALTO_BIN": "/env/pdfalto"}, clear=True)
    def test_env_var_used_when_no_arg(self):
        self.assertEqual(resolve_pdfalto_binary(None), "/env/pdfalto")

    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_to_bare_name(self):
        self.assertEqual(resolve_pdfalto_binary(None), "pdfalto")


class TestEnsureAltoXml(unittest.TestCase):
    def test_runs_pdfalto_on_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "book.pdf"
            pdf_path.write_bytes(b"%PDF-fake")
            cache_dir = Path(tmp) / "cache"

            def fake_run(cmd, capture_output, text):
                Path(cmd[-1]).write_text("<alto/>", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch(
                "evaluation.scripts.pdfalto_runner.subprocess.run", side_effect=fake_run
            ) as mock_run:
                output_path = ensure_alto_xml(pdf_path, cache_dir, "pdfalto")
                self.assertTrue(output_path.exists())
                self.assertEqual(mock_run.call_count, 1)

    def test_skips_pdfalto_on_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "book.pdf"
            pdf_path.write_bytes(b"%PDF-fake")
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            (cache_dir / "book.alto.xml").write_text("<alto/>", encoding="utf-8")

            with patch("evaluation.scripts.pdfalto_runner.subprocess.run") as mock_run:
                output_path = ensure_alto_xml(pdf_path, cache_dir, "pdfalto")
                self.assertTrue(output_path.exists())
                mock_run.assert_not_called()

    def test_raises_on_pdfalto_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "book.pdf"
            pdf_path.write_bytes(b"%PDF-fake")
            cache_dir = Path(tmp) / "cache"

            with patch(
                "evaluation.scripts.pdfalto_runner.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="boom"),
            ):
                with self.assertRaises(RuntimeError):
                    ensure_alto_xml(pdf_path, cache_dir, "pdfalto")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pdfalto_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.pdfalto_runner'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/scripts/pdfalto_runner.py`:

```python
"""Runs the pdfalto binary (https://github.com/kermitt2/pdfalto) against a
PDF and caches its ALTO XML output -- not vendored, matching the Kreuzberg
OCR sidecar precedent of treating an external tool as developer-provided,
not bundled. See
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import os
import subprocess
from pathlib import Path


def resolve_pdfalto_binary(cli_arg: str | None) -> str:
    """Resolves the pdfalto binary path: explicit --pdfalto-bin flag, then
    the PDFALTO_BIN environment variable, then bare "pdfalto" on PATH."""
    if cli_arg:
        return cli_arg
    env_value = os.environ.get("PDFALTO_BIN")
    if env_value:
        return env_value
    return "pdfalto"


def alto_xml_path(pdf_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{pdf_path.stem}.alto.xml"


def ensure_alto_xml(pdf_path: Path, cache_dir: Path, pdfalto_bin: str) -> Path:
    """Returns the cached ALTO XML path for pdf_path, running pdfalto only
    if the cache entry doesn't already exist. Raises RuntimeError if
    pdfalto exits non-zero or doesn't produce the expected output file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = alto_xml_path(pdf_path, cache_dir)
    if output_path.exists():
        return output_path

    result = subprocess.run(
        [pdfalto_bin, "-skipGraphs", str(pdf_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"pdfalto failed on {pdf_path} (exit {result.returncode}): {result.stderr}"
        )
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pdfalto_runner.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/pdfalto_runner.py tests/test_pdfalto_runner.py
git commit -m "feat: add pdfalto subprocess wrapper with output caching"
```

---

### Task 5: `add_toc_ground_truth.py` — retrofit the existing corpus

**Files:**
- Create: `evaluation/scripts/add_toc_ground_truth.py`
- Test: `tests/test_add_toc_ground_truth.py`
- Modify (real data): every `evaluation/corpus/{open-access,copyrighted-scans}/*.expected.json` in this worktree

- [ ] **Step 1: Write the failing test**

Create `tests/test_add_toc_ground_truth.py`:

```python
"""Unit tests for evaluation/scripts/add_toc_ground_truth.py's retrofit_book()
pure logic. The real file-walking main() is exercised manually against the
real evaluation corpus -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import unittest

from evaluation.scripts.add_toc_ground_truth import retrofit_book


def _toc_like_page(entries: list[tuple[str, int]]) -> str:
    """Builds page text with 3+ "title ... number" lines, matching
    ground_truth_helper._TOC_LINE_RE, so find_toc_pages structurally
    detects it as a TOC page."""
    return "\n".join(f"{title} {'.' * 10} {number}" for title, number in entries)


_TOC_TEXT = _toc_like_page([("Chapter One", 5), ("Chapter Two", 12), ("Chapter Three", 25)])


class TestRetrofitBook(unittest.TestCase):
    def test_skips_when_toc_key_already_present_and_not_forced(self):
        expected = {"chapters": [], "toc": None}
        updated, message = retrofit_book(["page text"], expected, force=False)
        self.assertIsNone(updated)
        self.assertTrue(message.startswith("SKIP"))

    def test_writes_toc_null_when_no_toc_page_found(self):
        pages = ["Chapter One\n\nSome body text with no listing lines at all."]
        expected = {"chapters": []}
        updated, message = retrofit_book(pages, expected, force=False)
        self.assertEqual(updated["toc"], None)
        self.assertTrue(message.startswith("OK"))

    def test_writes_toc_range_for_contiguous_toc_pages(self):
        pages = ["Front cover", _TOC_TEXT, "Chapter One\n\nBody text starts here."]
        expected = {"chapters": []}
        updated, message = retrofit_book(pages, expected, force=False)
        self.assertEqual(updated["toc"], {"toc_start_index": 1, "toc_end_index": 1})
        self.assertTrue(message.startswith("OK"))

    def test_flags_non_contiguous_toc_pages_for_manual_review(self):
        pages = [_TOC_TEXT, "unrelated page", _TOC_TEXT]
        expected = {"chapters": []}
        updated, message = retrofit_book(pages, expected, force=False)
        self.assertIsNone(updated)
        self.assertTrue(message.startswith("NEEDS REVIEW"))

    def test_force_recomputes_even_when_toc_key_present(self):
        pages = [_TOC_TEXT]
        expected = {"chapters": [], "toc": {"toc_start_index": 99, "toc_end_index": 99}}
        updated, message = retrofit_book(pages, expected, force=True)
        self.assertEqual(updated["toc"], {"toc_start_index": 0, "toc_end_index": 0})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_add_toc_ground_truth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.add_toc_ground_truth'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/scripts/add_toc_ground_truth.py`:

```python
#!/usr/bin/env python3
"""Retrofits existing evaluation/corpus/*/*.expected.json files with a
"toc" field, using the same structural TOC-page detection
ground_truth_helper.py already uses to exclude TOC pages from chapter-start
search. See
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md.

Auto-written entries still need a spot-check (open the PDF at
toc_start_index/toc_end_index, confirm) before being trusted -- this script
finds the best-scoring structural match, not necessarily the correct one,
same caveat as ground_truth_helper.py's chapter-boundary drafts.

Usage:
    uv run python evaluation/scripts/add_toc_ground_truth.py
    uv run python evaluation/scripts/add_toc_ground_truth.py --force
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader

from evaluation.scripts.ground_truth_helper import find_toc_pages, toc_page_range

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
_CORPORA = ["open-access", "copyrighted-scans"]


def retrofit_book(pages: list[str], expected: dict, force: bool) -> tuple[dict | None, str]:
    """Returns (updated expected dict or None if unchanged, status message).
    Pure function -- no file I/O -- so it's independently testable."""
    if "toc" in expected and not force:
        return None, "SKIP: already has a toc field"

    toc_pages = find_toc_pages(pages)
    toc_range = toc_page_range(toc_pages)

    if toc_pages and toc_range is None:
        return None, f"NEEDS REVIEW: non-contiguous TOC-like pages found: {sorted(toc_pages)}"

    updated = dict(expected)
    if toc_range is None:
        updated["toc"] = None
        return updated, "OK: no TOC page found, wrote toc=null"

    updated["toc"] = {"toc_start_index": toc_range[0], "toc_end_index": toc_range[1]}
    return updated, f"OK: wrote toc={updated['toc']}"


def _load_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="Re-run books that already have a toc field")
    args = parser.parse_args()

    needs_review = []
    n_written = 0
    n_skipped = 0

    for corpus in _CORPORA:
        corpus_dir = _CORPUS_DIR / corpus
        for expected_path in sorted(corpus_dir.glob("*.expected.json")):
            key = expected_path.name.removesuffix(".expected.json")
            pdf_path = corpus_dir / f"{key}.pdf"
            if not pdf_path.exists():
                print(f"[{corpus}/{key}] SKIP: no PDF found at {pdf_path}")
                n_skipped += 1
                continue

            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            pages = _load_pages(pdf_path)
            updated, message = retrofit_book(pages, expected, args.force)
            print(f"[{corpus}/{key}] {message}")

            if updated is not None:
                expected_path.write_text(
                    json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                n_written += 1
            elif message.startswith("NEEDS REVIEW"):
                needs_review.append(f"{corpus}/{key}")
            else:
                n_skipped += 1

    print(f"\n{n_written} book(s) written, {n_skipped} skipped, {len(needs_review)} need manual review")
    if needs_review:
        print("Needs manual review:")
        for entry in needs_review:
            print(f"  - {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_add_toc_ground_truth.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit the code**

```bash
git add evaluation/scripts/add_toc_ground_truth.py tests/test_add_toc_ground_truth.py
git commit -m "feat: add add_toc_ground_truth.py to retrofit corpus with toc page ranges"
```

- [ ] **Step 6: Run the retrofit against the real corpus**

This worktree already has all 50 GT PDFs symlinked in from the main
checkout (`evaluation/corpus/{open-access,copyrighted-scans}/*.pdf`). Run
the script for real:

```bash
uv run python evaluation/scripts/add_toc_ground_truth.py 2>&1 | tee /tmp/toc_retrofit_output.txt
```

Expected: a line per book (`OK: wrote toc=...` or `NEEDS REVIEW: ...`), and
a summary line `N book(s) written, M skipped, K need manual review`. Every
`.expected.json` under `evaluation/corpus/open-access/` and
`evaluation/corpus/copyrighted-scans/` now has a `"toc"` key.

- [ ] **Step 7: Spot-check a sample of auto-written entries**

Per the design spec, auto-written entries get a spot-check, not full
verification. Pick 8-10 books written as a range (not `null`) spanning both
corpora, and for each confirm the PDF's `toc_start_index`..`toc_end_index`
pages really are the table of contents (open the PDF at that physical page
-- 0-based index, so PDF-viewer page N+1):

```bash
uv run python -c "
import json
from pathlib import Path
from pypdf import PdfReader

for corpus in ['open-access', 'copyrighted-scans']:
    corpus_dir = Path('evaluation/corpus') / corpus
    for expected_path in sorted(corpus_dir.glob('*.expected.json'))[:5]:
        expected = json.loads(expected_path.read_text())
        toc = expected.get('toc')
        if not toc:
            continue
        key = expected_path.name.removesuffix('.expected.json')
        reader = PdfReader(str(corpus_dir / f'{key}.pdf'))
        text = reader.pages[toc['toc_start_index']].extract_text() or ''
        print(f'{corpus}/{key} toc={toc}')
        print('  first lines:', repr(text[:150]))
"
```

If any spot-checked entry is wrong, hand-fix that book's `.expected.json`
`"toc"` field directly (open the PDF, find the real range) before
proceeding -- do not trust the script blindly, same discipline
`evaluation/CLAUDE.md` requires for chapter boundaries.

- [ ] **Step 8: Commit the retrofitted ground truth**

```bash
git add evaluation/corpus/open-access/*.expected.json evaluation/corpus/copyrighted-scans/*.expected.json
git commit -m "chore: retrofit evaluation corpus with toc ground-truth field"
```

---

### Task 6: Wire `"toc"` into the CrossRef auto-GT pipeline

**Files:**
- Modify: `evaluation/scripts/build_crossref_gt_ground_truth.py:48` (import) and `:206-210` (write)

- [ ] **Step 1: Update the import**

In `evaluation/scripts/build_crossref_gt_ground_truth.py`, change line 48
from:

```python
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages
```

to:

```python
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
```

- [ ] **Step 2: Write the toc field alongside chapters**

Change the write block (originally lines 206-210):

```python
    _OPEN_ACCESS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, target_pdf)
    target_expected.write_text(
        json.dumps({"chapters": confirmed}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
```

to:

```python
    _OPEN_ACCESS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, target_pdf)
    toc_range = toc_page_range(toc_pages)
    toc_field = (
        {"toc_start_index": toc_range[0], "toc_end_index": toc_range[1]} if toc_range else None
    )
    target_expected.write_text(
        json.dumps({"chapters": confirmed, "toc": toc_field}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
```

(`toc_pages` is already in scope here -- it's computed once at the top of
`process_book`, originally line 146: `toc_pages = find_toc_pages(pages)`.)

- [ ] **Step 3: Smoke-test with --dry-run**

This repo has no PDFs cached under `evaluation/crossref_gt/` yet (they're
fetched separately via `fetch_crossref_gt_corpus.py`, a network operation
out of scope here), so every book will report "no PDF" -- this run only
confirms the edited file still imports and runs cleanly:

```bash
uv run python evaluation/scripts/build_crossref_gt_ground_truth.py --dry-run
```

Expected: no traceback; every line reads
`SKIP: no PDF (fetch_crossref_gt_corpus.py first)`; final summary line
`0/43 book(s) would be migrated to open-access/`.

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/build_crossref_gt_ground_truth.py
git commit -m "feat: populate toc field when migrating CrossRef-sourced ground truth"
```

---

### Task 7: Document the new field in `evaluation/CLAUDE.md`

**Files:**
- Modify: `evaluation/CLAUDE.md` (insert a new section after "Step 4", before "Known failure modes")

- [ ] **Step 1: Insert the new section**

In `evaluation/CLAUDE.md`, insert this section immediately after "Step 4:
Write the final `.expected.json` and sanity-check it" (i.e. right before
the "## Known failure modes" heading):

```markdown
## Step 5: TOC ground truth

`.expected.json` also carries an optional `"toc"` field, sibling to
`"chapters"`, used by the layout-based TOC-classifier pilot (see
`docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`):

```json
{"toc_start_index": 7, "toc_end_index": 8}
```

Same 0-based-physical-page convention as `pdf_start_index`/`pdf_end_index`.
Three states, not interchangeable:

- **Key absent**: not yet retrofitted for this field.
- **`"toc": null`**: confirmed -- this book has no locatable printed TOC page.
- **`"toc": {"toc_start_index": ..., "toc_end_index": ...}`**: TOC located
  at this contiguous physical-page range.

For a book you're adding by hand, run
`evaluation/scripts/add_toc_ground_truth.py` after finishing Step 4 -- it
reuses the same structural TOC-page detection (`find_toc_pages`) the
chapter-locating step already excludes TOC pages with, so it costs nothing
extra to run. It writes automatically when the detected TOC pages form one
contiguous block; otherwise it leaves the book alone and reports it as
needing manual review (open the PDF, find the real range, write the field
by hand). Spot-check any auto-written range before trusting it, same
discipline as the chapter-boundary draft in Step 2/3 -- this script also
finds the best structural match, not necessarily the correct one.
```

- [ ] **Step 2: Verify the doc renders sensibly**

```bash
uv run python -c "
text = open('evaluation/CLAUDE.md').read()
assert '## Step 5: TOC ground truth' in text
assert text.index('## Step 5: TOC ground truth') < text.index('## Known failure modes')
print('OK: Step 5 section present and correctly ordered')
"
```

Expected: `OK: Step 5 section present and correctly ordered`

- [ ] **Step 3: Commit**

```bash
git add evaluation/CLAUDE.md
git commit -m "docs: document the toc ground-truth field in evaluation/CLAUDE.md"
```

---

### Task 8: `.gitignore` and the scikit-learn dependency

**Files:**
- Modify: `evaluation/.gitignore`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the layout-cache gitignore entry**

In `evaluation/.gitignore`, add a new line after `.ocr-cache/`:

```
*.pdf
manifest.local.json
.ocr-cache/
.layout-cache/
```

- [ ] **Step 2: Add the scikit-learn optional dependency**

In `pyproject.toml`, under `[project.optional-dependencies]` (currently
lines 17-19), add a new extra:

```toml
[project.optional-dependencies]
kreuzberg = ["httpx>=0.27.0"]
tesseract = ["pytesseract>=0.3.10", "pymupdf>=1.24.0", "pillow>=10.0.0"]
llm-eval = ["openai>=1.0.0", "httpx>=0.27.0"]
layout-classifier = ["scikit-learn>=1.4.0"]
```

- [ ] **Step 3: Sync and verify the install**

```bash
uv sync --extra layout-classifier
uv run python -c "from sklearn.ensemble import HistGradientBoostingClassifier; print('OK')"
```

Expected: `OK` printed, no import errors.

- [ ] **Step 4: Commit**

```bash
git add evaluation/.gitignore pyproject.toml uv.lock
git commit -m "chore: add scikit-learn as an eval-only optional dependency, ignore layout-cache"
```

---

### Task 9: `select_threshold()` — recall-targeted probability cutoff

**Files:**
- Create: `evaluation/scripts/evaluate_layout_toc_classifier.py` (this task adds only `select_threshold`; later tasks extend the same file)
- Test: `tests/test_evaluate_layout_toc_classifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate_layout_toc_classifier.py`:

```python
"""Unit tests for evaluation/scripts/evaluate_layout_toc_classifier.py's
pure logic. The real pdfalto-subprocess-driven, real-corpus leave-one-book-out
run is exercised manually -- see
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import unittest

from evaluation.scripts.evaluate_layout_toc_classifier import select_threshold


class TestSelectThreshold(unittest.TestCase):
    def test_no_positives_returns_one(self):
        self.assertEqual(select_threshold([0.1, 0.9], [False, False], recall_target=0.9), 1.0)

    def test_single_positive_needs_its_own_probability(self):
        probs = [0.2, 0.8, 0.5]
        labels = [False, True, False]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.9), 0.8)

    def test_target_below_full_recall_picks_higher_cutoff(self):
        # Four positives at probabilities 0.9, 0.8, 0.7, 0.2 -- targeting
        # 75% recall needs the top 3 (round(0.75*4)=3), so the cutoff is
        # the third-highest positive probability, 0.7.
        probs = [0.9, 0.8, 0.7, 0.2]
        labels = [True, True, True, True]
        self.assertEqual(select_threshold(probs, labels, recall_target=0.75), 0.7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.evaluate_layout_toc_classifier'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/scripts/evaluate_layout_toc_classifier.py`:

```python
#!/usr/bin/env python3
"""Pilot: leave-one-book-out evaluation of a layout-geometry TOC/
chapter-first-page classifier. See
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md.

Manual run, not part of `uv run pytest` -- same convention as
evaluation/scripts/fetch_evaluation_pdfs.py.

Usage:
    uv run python evaluation/scripts/evaluate_layout_toc_classifier.py
    uv run python evaluation/scripts/evaluate_layout_toc_classifier.py --pdfalto-bin /path/to/pdfalto
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_RECALL_TARGET = 0.90  # threshold picked per fold to hit this recall on training pages


def select_threshold(
    train_probs: list[float], train_labels: list[bool], recall_target: float
) -> float:
    """Picks the highest probability threshold that still achieves at least
    recall_target on the training positives -- a lower threshold always
    yields recall >= a higher one, so the highest satisfying threshold is
    the most precise choice that still clears the bar. Returns 1.0 (accept
    nothing) if there are no positives to calibrate against."""
    positive_probs = sorted(
        (p for p, is_positive in zip(train_probs, train_labels) if is_positive), reverse=True
    )
    if not positive_probs:
        return 1.0
    n_needed = max(1, round(recall_target * len(positive_probs)))
    return positive_probs[n_needed - 1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/evaluate_layout_toc_classifier.py tests/test_evaluate_layout_toc_classifier.py
git commit -m "feat: add select_threshold() for recall-targeted classifier cutoffs"
```

---

### Task 10: `load_book_corpus()` and `build_feature_table()`

**Files:**
- Modify: `evaluation/scripts/evaluate_layout_toc_classifier.py` (append)
- Test: `tests/test_evaluate_layout_toc_classifier.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate_layout_toc_classifier.py` (add these
imports to the existing `from evaluation... import` line and add the new
test class):

```python
from unittest.mock import patch

from evaluation.scripts.evaluate_layout_toc_classifier import build_feature_table, select_threshold


class TestBuildFeatureTable(unittest.TestCase):
    def test_joins_features_with_labels_per_page(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "chapter_first", "other"]},
        ]

        fake_features = {0: {"line_count": 1.0}, 1: {"line_count": 2.0}, 2: {"line_count": 3.0}}

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {"book_key": "book-a", "features": {"line_count": 1.0}, "label": "toc"})
        self.assertEqual(
            rows[1], {"book_key": "book-a", "features": {"line_count": 2.0}, "label": "chapter_first"}
        )
        self.assertEqual(rows[2], {"book_key": "book-a", "features": {"line_count": 3.0}, "label": "other"})

    def test_skips_pages_pdfalto_did_not_extract(self):
        books = [
            {"key": "book-a", "corpus": "open-access", "pdf_path": Path("/fake/book-a.pdf"),
             "labels": ["toc", "other"]},
        ]
        # pdfalto only produced a feature vector for page 0.
        fake_features = {0: {"line_count": 1.0}}

        with patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.ensure_alto_xml",
            return_value=Path("/fake/book-a.alto.xml"),
        ), patch(
            "evaluation.scripts.evaluate_layout_toc_classifier.extract_page_features",
            return_value=fake_features,
        ):
            rows = build_feature_table(books, lambda corpus: Path("/fake/cache"), "pdfalto")

        self.assertEqual(len(rows), 1)
```

Add `from pathlib import Path` to the test file's imports at the top if not
already present (it is not, in the file as created in Task 9).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_feature_table'`

- [ ] **Step 3: Write the implementation**

Append to `evaluation/scripts/evaluate_layout_toc_classifier.py` (after
`select_threshold`, and add the new imports at the top of the file
alongside the existing `sys.path.insert` block):

```python
import json

from pypdf import PdfReader

from evaluation.scripts.layout_features import extract_page_features
from evaluation.scripts.layout_labels import page_labels
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
_CORPORA = ["open-access", "copyrighted-scans"]
```

(Place these new imports directly under the existing `sys.path.insert(...)`
line, before `_RECALL_TARGET`.)

Then append these two functions after `select_threshold`:

```python
def load_book_corpus() -> list[dict]:
    """Returns one entry per book with a usable "toc" field: {"key",
    "corpus", "pdf_path", "labels"} -- books whose .expected.json has no
    "toc" key at all are excluded entirely (not yet retrofitted, or
    flagged for manual review), per the design spec."""
    books = []
    for corpus in _CORPORA:
        corpus_dir = _CORPUS_DIR / corpus
        for expected_path in sorted(corpus_dir.glob("*.expected.json")):
            key = expected_path.name.removesuffix(".expected.json")
            pdf_path = corpus_dir / f"{key}.pdf"
            if not pdf_path.exists():
                continue
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            if "toc" not in expected:
                continue
            total_pages = len(PdfReader(str(pdf_path)).pages)
            labels = page_labels(expected, total_pages)
            books.append({"key": key, "corpus": corpus, "pdf_path": pdf_path, "labels": labels})
    return books


def build_feature_table(books: list[dict], cache_dir_for, pdfalto_bin: str) -> list[dict]:
    """Runs pdfalto (cached via cache_dir_for(corpus) -> Path) over every
    book and returns one row per page with an extracted feature vector:
    {"book_key", "features": {...}, "label": "toc"|"chapter_first"|"other"}.
    Pages pdfalto didn't produce a feature vector for (should not normally
    happen) are silently skipped rather than crashing the whole run."""
    rows = []
    for book in books:
        cache_dir = cache_dir_for(book["corpus"])
        alto_path = ensure_alto_xml(book["pdf_path"], cache_dir, pdfalto_bin)
        page_features = extract_page_features(str(alto_path))
        for page_index, label in enumerate(book["labels"]):
            features = page_features.get(page_index)
            if features is None:
                continue
            rows.append({"book_key": book["key"], "features": features, "label": label})
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/evaluate_layout_toc_classifier.py tests/test_evaluate_layout_toc_classifier.py
git commit -m "feat: add load_book_corpus() and build_feature_table() to the pilot script"
```

---

### Task 11: `evaluate_leave_one_book_out()` and `main()`

**Files:**
- Modify: `evaluation/scripts/evaluate_layout_toc_classifier.py` (append)
- Test: `tests/test_evaluate_layout_toc_classifier.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate_layout_toc_classifier.py`:

```python
from evaluation.scripts.evaluate_layout_toc_classifier import evaluate_leave_one_book_out
from evaluation.scripts.layout_features import FEATURE_NAMES


def _feature_row(book_key: str, label: str, value: float) -> dict:
    return {"book_key": book_key, "features": {name: value for name in FEATURE_NAMES}, "label": label}


class TestEvaluateLeaveOneBookOut(unittest.TestCase):
    def test_perfectly_separable_data_gets_full_recall(self):
        # Three synthetic books, each with 5 pages: page 0 is "toc"
        # (features all 5.0), page 1 is "chapter_first" (features all
        # -5.0), pages 2-4 are "other" (features all 0.0) -- identical,
        # trivially separable pattern across every book.
        rows = []
        for book_key in ("book-a", "book-b", "book-c"):
            rows.append(_feature_row(book_key, "toc", 5.0))
            rows.append(_feature_row(book_key, "chapter_first", -5.0))
            rows.extend(_feature_row(book_key, "other", 0.0) for _ in range(3))

        summary = evaluate_leave_one_book_out(rows)

        self.assertEqual(summary["full_recall_fraction"], 1.0)
        self.assertLessEqual(summary["avg_candidate_fraction"], 0.45)
        self.assertEqual(len(summary["per_book"]), 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate_leave_one_book_out'`

- [ ] **Step 3: Write the implementation**

Add this import to the top of `evaluation/scripts/evaluate_layout_toc_classifier.py`,
alongside the other imports added in Task 10:

```python
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluation.scripts.layout_features import FEATURE_NAMES
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST, LABEL_TOC
```

(`extract_page_features` is no longer needed as a separate import target
here since `FEATURE_NAMES` now covers what this function needs from
`layout_features`; keep the existing `from evaluation.scripts.layout_features import extract_page_features`
import from Task 10 as-is, and add `FEATURE_NAMES` to the same import line:
`from evaluation.scripts.layout_features import FEATURE_NAMES, extract_page_features`.)

Append `evaluate_leave_one_book_out` and `main` after `build_feature_table`:

```python
def evaluate_leave_one_book_out(rows: list[dict]) -> dict:
    """Runs leave-one-book-out cross-validation, returns per-book results
    and an aggregate summary matching the design spec's decision criteria."""
    book_keys = sorted({row["book_key"] for row in rows})
    per_book_results = []

    for held_out in book_keys:
        train_rows = [r for r in rows if r["book_key"] != held_out]
        test_rows = [r for r in rows if r["book_key"] == held_out]

        X_train = [[r["features"][name] for name in FEATURE_NAMES] for r in train_rows]
        X_test = [[r["features"][name] for name in FEATURE_NAMES] for r in test_rows]

        result: dict = {"book_key": held_out, "total_pages": len(test_rows)}
        candidate_pages: set[int] = set()

        for label in (LABEL_TOC, LABEL_CHAPTER_FIRST):
            y_train = [r["label"] == label for r in train_rows]
            if sum(y_train) == 0:
                continue
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
            clf.fit(X_train, y_train)
            train_probs = [p[1] for p in clf.predict_proba(X_train)]
            threshold = select_threshold(train_probs, y_train, _RECALL_TARGET)

            test_probs = [p[1] for p in clf.predict_proba(X_test)]
            true_positive_indices = {i for i, r in enumerate(test_rows) if r["label"] == label}
            predicted_indices = {i for i, p in enumerate(test_probs) if p >= threshold}

            result[f"{label}_recall"] = (
                len(true_positive_indices & predicted_indices) / len(true_positive_indices)
                if true_positive_indices
                else None
            )
            candidate_pages |= predicted_indices

        result["candidate_fraction"] = len(candidate_pages) / result["total_pages"]
        per_book_results.append(result)

    n_books = len(per_book_results)
    n_full_recall = sum(
        1
        for r in per_book_results
        if r.get(f"{LABEL_TOC}_recall") in (None, 1.0)
        and r.get(f"{LABEL_CHAPTER_FIRST}_recall") in (None, 1.0)
    )
    avg_candidate_fraction = sum(r["candidate_fraction"] for r in per_book_results) / n_books

    return {
        "per_book": per_book_results,
        "full_recall_fraction": n_full_recall / n_books,
        "avg_candidate_fraction": avg_candidate_fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdfalto-bin", default=None)
    args = parser.parse_args()
    pdfalto_bin = resolve_pdfalto_binary(args.pdfalto_bin)

    books = load_book_corpus()
    if not books:
        print("No books with a 'toc' field found -- run add_toc_ground_truth.py first.")
        return 1

    def cache_dir_for(corpus: str) -> Path:
        return _CORPUS_DIR / corpus / ".layout-cache"

    rows = build_feature_table(books, cache_dir_for, pdfalto_bin)
    summary = evaluate_leave_one_book_out(rows)

    print(f"Books evaluated: {len(books)}")
    print(
        f"Books with full recall (toc + all chapter-first pages retained): "
        f"{summary['full_recall_fraction']:.0%}"
    )
    print(f"Average candidate-page fraction: {summary['avg_candidate_fraction']:.1%}")
    print()
    for r in summary["per_book"]:
        print(
            f"  {r['book_key']}: toc_recall={r.get('toc_recall')}, "
            f"chapter_first_recall={r.get('chapter_first_recall')}, "
            f"candidate_fraction={r['candidate_fraction']:.1%}"
        )

    meets_bar = summary["full_recall_fraction"] >= 0.90 and summary["avg_candidate_fraction"] <= 0.15
    print(
        f"\nDecision bar (>=90% full recall, <=15% avg candidate fraction): "
        f"{'MET' if meets_bar else 'NOT MET'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full test suite to check nothing else broke**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass (pre-existing tests plus every test added in
Tasks 1-11).

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/evaluate_layout_toc_classifier.py tests/test_evaluate_layout_toc_classifier.py
git commit -m "feat: add leave-one-book-out evaluation loop and CLI to the pilot script"
```

---

### Task 12: Run the pilot for real and record the decision

**Files:** none (execution + a decision, no code changes)

- [ ] **Step 1: Run the pilot against the real, retrofitted corpus**

Everything needed is already in this worktree: the locally-built `pdfalto`
binary (`/Users/cboulanger/Code/pdfalto/pdfalto`), the 50 symlinked GT
PDFs, the `"toc"` field retrofitted in Task 5, and `scikit-learn` installed
in Task 8.

```bash
uv run python evaluation/scripts/evaluate_layout_toc_classifier.py \
  --pdfalto-bin /Users/cboulanger/Code/pdfalto/pdfalto \
  2>&1 | tee /tmp/layout_classifier_pilot_output.txt
```

Expected: a per-book line for every book with a usable `"toc"` field, then
the aggregate `full_recall_fraction`, `avg_candidate_fraction`, and a final
`Decision bar (>=90% full recall, <=15% avg candidate fraction): MET` or
`NOT MET` line. First run will be slower (building the `.layout-cache/`
ALTO XML cache for all 50 books); reruns are fast.

- [ ] **Step 2: Read and report the result**

Look at `/tmp/layout_classifier_pilot_output.txt`. Per the design spec's
decision criteria:

- **MET**: the approach is viable as a pre-filter. Next steps (production
  `TocExtractionStrategy`/pre-filter wiring, and/or expanding
  `evaluation/crossref_gt/manifest.json` for more GT) are explicitly out of
  scope for this plan -- report the result and let the user decide whether
  to open a follow-up spec for either.
- **NOT MET**: report which bar failed (recall, candidate-set size, or
  both) and which books drove the failure (visible in the per-book
  breakdown) -- this is useful diagnostic signal for whether the feature
  set needs another pass, even though redesigning the feature set is not
  part of this plan.

Do not write the results into `evaluation/RESULTS.md` as part of this
task -- per the design spec, that write-up is deliberately deferred to
whatever follow-up the result justifies, not baked into this pilot's own
implementation.

---

## Self-Review Notes

- **Spec coverage**: GT schema retrofit (Task 1, 5), CrossRef workflow
  update (Task 6), CLAUDE.md docs (Task 7), pdfalto caching + feature
  extraction (Tasks 3, 4), scikit-learn dependency (Task 8), leave-one-book-out
  pilot with the spec's exact recall-first decision bar (Tasks 9-12) --
  every section of the design spec has a corresponding task.
- **Placeholder scan**: no TBD/TODO markers; every step has complete,
  concrete code.
- **Type consistency**: `toc_page_range`, `page_labels`/`LABEL_TOC`/
  `LABEL_CHAPTER_FIRST`/`LABEL_OTHER`, `extract_page_features`/
  `FEATURE_NAMES`, `resolve_pdfalto_binary`/`ensure_alto_xml`,
  `retrofit_book`, `select_threshold`, `load_book_corpus`,
  `build_feature_table`, `evaluate_leave_one_book_out` are named and typed
  identically everywhere they're defined, imported, and called across
  Tasks 1-12.
