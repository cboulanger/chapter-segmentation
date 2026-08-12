# Automated Crossref-sourced GT corpus expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate discovery of new open-access edited-volume GT candidates via Crossref, and gate their admission into `evaluation/corpus/pending/` on a layout-feature novelty check, so corpus growth targets the classifier's actual feature-space gaps instead of adding redundant volume.

**Architecture:** Three independent pieces, wired into the existing `evaluation/crossref_gt/` reconciliation pipeline rather than replacing it: (1) a new `discover_crossref_candidates.py` that finds and appends new manifest candidates, (2) an extension to the existing `build_crossref_gt_ground_truth.py` that adds a novelty gate + diagnostic outline cross-check on top of its unchanged offset-consensus/content-search confirmation, migrating into `pending/` instead of `open-access/`, and (3) a small `--corpora` flag on `evaluate_layout_toc_classifier.py` so `pending/` books can be evaluated before manual promotion.

**Tech Stack:** Python, `httpx` (sync client, matching `fetch_crossref_gt_corpus.py`), `pypdf`, `rapidfuzz`, `scikit-learn` (`StandardScaler`), `numpy`, `unittest`.

Full design context: `docs/superpowers/specs/2026-08-11-crossref-gt-corpus-expansion-design.md`.

---

## Task 1: Backfill `discovery_source: "manual"` onto existing manifest entries

**Files:**
- Modify: `evaluation/crossref_gt/manifest.json`

This is a one-time data edit, not new logic -- no test file, just a verification step.

- [ ] **Step 1: Run the backfill**

```bash
uv run python -c "
import json
from pathlib import Path
path = Path('evaluation/crossref_gt/manifest.json')
manifest = json.loads(path.read_text(encoding='utf-8'))
for book in manifest['books']:
    book.setdefault('discovery_source', 'manual')
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f\"{len(manifest['books'])} book(s) backfilled\")
"
```

Expected output: `43 book(s) backfilled`.

- [ ] **Step 2: Verify every entry now has the field**

```bash
uv run python -c "
import json
manifest = json.loads(open('evaluation/crossref_gt/manifest.json').read_text(encoding='utf-8'))
missing = [b['isbn'] for b in manifest['books'] if b.get('discovery_source') != 'manual']
assert not missing, f'missing/wrong discovery_source: {missing}'
print('OK: all 43 entries have discovery_source=manual')
"
```

Expected output: `OK: all 43 entries have discovery_source=manual`.

- [ ] **Step 3: Commit**

```bash
git add evaluation/crossref_gt/manifest.json
git commit -m "chore: backfill discovery_source=manual on existing crossref_gt manifest entries"
```

---

## Task 2: Add `--corpora` flag to `evaluate_layout_toc_classifier.py`

**Files:**
- Modify: `evaluation/scripts/evaluate_layout_toc_classifier.py:82-101` (`load_book_corpus`), `:291-354` (`main`)
- Test: `tests/test_evaluate_layout_toc_classifier.py`

`load_book_corpus()` currently always scans the hardcoded module-level `_CORPORA = ["open-access", "copyrighted-scans"]`. This task adds an optional `corpora` parameter (defaulting to `_CORPORA`, so every existing caller/test keeps working unchanged) and a `--corpora` CLI flag, so `evaluation/corpus/pending/` books can be evaluated on demand without editing the script -- needed by Task 6's novelty baseline (which must load *only* `open-access`) and by manual pre-promotion review of a `pending/` book.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluate_layout_toc_classifier.py`, in the same test class as the existing `load_book_corpus` test (`TestLoadBookCorpus` -- check the existing test's class name in the file and add this method alongside it):

```python
    def test_corpora_param_restricts_which_directories_are_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            oa_dir = tmp_path / "open-access"
            cs_dir = tmp_path / "copyrighted-scans"
            oa_dir.mkdir()
            cs_dir.mkdir()

            (oa_dir / "book-a.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )
            (oa_dir / "book-a.pdf").write_bytes(b"%PDF-fake")

            (cs_dir / "book-c.expected.json").write_text(
                json.dumps({"toc": None, "chapters": []}), encoding="utf-8"
            )
            (cs_dir / "book-c.pdf").write_bytes(b"%PDF-fake")

            fake_reader = Mock()
            fake_reader.pages = [Mock()] * 5

            with patch(
                "evaluation.scripts.evaluate_layout_toc_classifier._CORPUS_DIR", tmp_path
            ), patch(
                "evaluation.scripts.evaluate_layout_toc_classifier.PdfReader",
                return_value=fake_reader,
            ):
                books = load_book_corpus(corpora=["open-access"])

        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["key"], "book-a")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -k corpora_param -v`
Expected: FAIL with `TypeError: load_book_corpus() got an unexpected keyword argument 'corpora'`

- [ ] **Step 3: Update `load_book_corpus`**

In `evaluation/scripts/evaluate_layout_toc_classifier.py`, replace:

```python
def load_book_corpus() -> list[dict]:
    """Returns one entry per book with a usable "toc" field: {"key",
    "corpus", "pdf_path", "labels"} -- books whose .expected.json has no
    "toc" key at all are excluded entirely (not yet retrofitted, or
    flagged for manual review), per the design spec."""
    books = []
    for corpus in _CORPORA:
```

with:

```python
def load_book_corpus(corpora: list[str] | None = None) -> list[dict]:
    """Returns one entry per book with a usable "toc" field: {"key",
    "corpus", "pdf_path", "labels"} -- books whose .expected.json has no
    "toc" key at all are excluded entirely (not yet retrofitted, or
    flagged for manual review), per the design spec. `corpora` defaults
    to every corpus this pilot normally scores against (_CORPORA);
    pass an explicit subset (e.g. ["open-access"]) to restrict it, e.g.
    to evaluate evaluation/corpus/pending/ candidates before promotion,
    or to compute a novelty baseline from open-access alone."""
    books = []
    for corpus in corpora if corpora is not None else _CORPORA:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -k corpora_param -v`
Expected: PASS

- [ ] **Step 5: Add the `--corpora` CLI flag**

In `evaluation/scripts/evaluate_layout_toc_classifier.py`'s `main()`, after the existing `--chapter-first-recall-tolerance` argument:

```python
    parser.add_argument(
        "--corpora",
        default=",".join(_CORPORA),
        help=(
            "Comma-separated list of evaluation/corpus/ subdirectories to load books "
            "from (e.g. 'open-access,pending' to include unreviewed candidates). "
            f"Default: {','.join(_CORPORA)}."
        ),
    )
```

And change:

```python
    books = load_book_corpus()
```

to:

```python
    books = load_book_corpus(corpora=[c.strip() for c in args.corpora.split(",") if c.strip()])
```

- [ ] **Step 6: Run the full existing test file to confirm no regression**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: PASS (all tests, including the pre-existing `load_book_corpus` test with no `corpora` argument -- the default keeps it scanning both `_CORPORA` entries unchanged)

- [ ] **Step 7: Verify `--help` renders the new flag**

Run: `uv run python evaluation/scripts/evaluate_layout_toc_classifier.py --help`
Expected: output includes a `--corpora` option with the default `open-access,copyrighted-scans`.

- [ ] **Step 8: Commit**

```bash
git add evaluation/scripts/evaluate_layout_toc_classifier.py tests/test_evaluate_layout_toc_classifier.py
git commit -m "feat: add --corpora flag to evaluate_layout_toc_classifier.py"
```

---

## Task 3: Regression test confirming TOC/page-number pattern matching is language-agnostic

**Files:**
- Modify: `tests/test_ground_truth_helper.py`

`evaluation/scripts/ground_truth_helper.py`'s `find_toc_pages`/`extract_printed_number` are already purely structural (digit/roman-numeral shape, not English keywords) -- confirmed by inspection during design. This task adds a regression test guarding that claim against a future change silently introducing an English-only assumption (e.g. a keyword-based "Contents" search). No production code changes in this task.

- [ ] **Step 1: Write the test**

Add to `tests/test_ground_truth_helper.py`:

```python
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range


class TestLanguageAgnosticPatternMatching(unittest.TestCase):
    """Guards find_toc_pages/extract_printed_number against a future change
    that silently introduces an English-only assumption (e.g. a
    keyword-based "Contents" search) -- both currently key off page-number
    *shape* (digits/roman numerals), not language-specific words, so they
    must work identically on German and French TOC/page-number text."""

    def test_finds_german_toc_page(self):
        pages = [
            "Vorwort\n\n\nSeite 3",
            "Inhaltsverzeichnis\n\nEinleitung .......... 7\nKapitel 1 .......... 15\nKapitel 2 .......... 42\n",
            "Einleitung\n\nDies ist der erste Absatz.",
        ]
        self.assertEqual(find_toc_pages(pages), {1})

    def test_finds_french_toc_page(self):
        pages = [
            "Préface\n\n\nPage 3",
            "Table des matières\n\nIntroduction .......... 7\nChapitre 1 .......... 15\nChapitre 2 .......... 42\n",
            "Introduction\n\nCeci est le premier paragraphe.",
        ]
        self.assertEqual(find_toc_pages(pages), {1})

    def test_extracts_german_footer_page_number(self):
        text = "Einleitung\n\nDies ist der erste Absatz der Einleitung.\n\n7"
        self.assertEqual(extract_printed_number(text), "7")

    def test_extracts_french_footer_page_number(self):
        text = "Introduction\n\nCeci est le premier paragraphe.\n\n7"
        self.assertEqual(extract_printed_number(text), "7")
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_ground_truth_helper.py -k LanguageAgnostic -v`
Expected: PASS immediately -- no production code change needed. This test documents already-correct behavior; if it ever fails after a future change to `ground_truth_helper.py`, that change introduced a language-specific regression.

- [ ] **Step 3: Run the full file to confirm no regression**

Run: `uv run pytest tests/test_ground_truth_helper.py -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_ground_truth_helper.py
git commit -m "test: add language-agnosticism regression test for TOC/page-number pattern matching"
```

---

## Task 4: Novelty-gate pure functions

**Files:**
- Modify: `evaluation/scripts/build_crossref_gt_ground_truth.py` (add functions near the top, after existing imports/constants)
- Test: `tests/test_build_crossref_gt_ground_truth.py`

Adds three small, dependency-free functions: nearest-neighbor distance, a percentile-based novelty threshold derived from the existing corpus's own leave-one-out distances, and the keep/discard decision. These are unit-testable in isolation from pdfalto/PDF-reading, which Task 6 wires in.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_build_crossref_gt_ground_truth.py`:

```python
from evaluation.scripts.build_crossref_gt_ground_truth import (
    _is_novel,
    _nearest_neighbor_distance,
    _novelty_threshold,
    _toc_field_for,
)


class TestNearestNeighborDistance(unittest.TestCase):
    def test_returns_distance_to_closest_point(self):
        # Distance to [1.0, 0.0] is 1.0; to [3.0, 4.0] is 5.0 -- the
        # nearer one wins.
        distance = _nearest_neighbor_distance([0.0, 0.0], [[3.0, 4.0], [1.0, 0.0]])
        self.assertAlmostEqual(distance, 1.0)

    def test_single_other_point(self):
        distance = _nearest_neighbor_distance([0.0], [[5.0]])
        self.assertAlmostEqual(distance, 5.0)


class TestNoveltyThreshold(unittest.TestCase):
    def test_percentile_of_leave_one_out_distances(self):
        # 1-D vectors [0, 1, 2, 3, 10]. Leave-one-out nearest-neighbor
        # distance is 1.0 for every point except 10.0 (nearest is 3.0,
        # distance 7.0): [1, 1, 1, 1, 7]. numpy's default linear-
        # interpolation 90th percentile of that sorted list is 4.6.
        vectors = [[0.0], [1.0], [2.0], [3.0], [10.0]]
        threshold = _novelty_threshold(vectors, percentile=90)
        self.assertAlmostEqual(threshold, 4.6)


class TestIsNovel(unittest.TestCase):
    def setUp(self):
        self.existing = [[0.0], [1.0], [2.0]]

    def test_candidate_close_to_existing_is_not_novel(self):
        self.assertFalse(_is_novel([[0.5]], self.existing, threshold=1.5))

    def test_candidate_far_from_existing_is_novel(self):
        self.assertTrue(_is_novel([[10.0]], self.existing, threshold=1.5))

    def test_novel_if_any_candidate_page_qualifies(self):
        # First candidate page is close (not novel alone); second is far.
        # Keep-if-at-least-one-page-is-novel per the design spec.
        self.assertTrue(_is_novel([[0.5], [10.0]], self.existing, threshold=1.5))
```

(Keep the existing `TestTocFieldFor` class and its tests in the file unchanged.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -k "NearestNeighbor or NoveltyThreshold or IsNovel" -v`
Expected: FAIL with `ImportError: cannot import name '_nearest_neighbor_distance'` (and similarly for the other two)

- [ ] **Step 3: Implement the three functions**

In `evaluation/scripts/build_crossref_gt_ground_truth.py`, add near the top, after the existing imports (`from pypdf import PdfReader` / `from rapidfuzz import fuzz` / the two `chapter_segmentation`/`evaluation.scripts.ground_truth_helper` imports):

```python
import math

import numpy as np
```

Then add these three functions after the existing constants (`_MIN_CONFIRMED_CHAPTERS = 3`), before `_citation_start`:

```python
def _nearest_neighbor_distance(vector: list[float], others: list[list[float]]) -> float:
    """Euclidean distance from `vector` to the closest point in `others`."""
    return min(math.dist(vector, other) for other in others)


def _novelty_threshold(vectors: list[list[float]], percentile: float) -> float:
    """The `percentile`-th percentile of `vectors`' own leave-one-out
    nearest-neighbor distances -- pages farther than this from the
    existing corpus sit outside how tightly that corpus already clusters.
    `vectors` must have at least 2 entries (a single vector has no
    leave-one-out neighbor to measure against)."""
    loo_distances = [
        _nearest_neighbor_distance(vector, vectors[:i] + vectors[i + 1 :])
        for i, vector in enumerate(vectors)
    ]
    return float(np.percentile(loo_distances, percentile))


def _is_novel(
    candidate_vectors: list[list[float]], existing_vectors: list[list[float]], threshold: float
) -> bool:
    """True if at least one candidate page's nearest-neighbor distance to
    the existing corpus meets or exceeds `threshold` -- the design spec's
    "keep if at least one page is novel" rule, since a book that's mostly
    ordinary but has one unusual chapter-opening is still worth keeping."""
    return any(
        _nearest_neighbor_distance(vector, existing_vectors) >= threshold
        for vector in candidate_vectors
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -k "NearestNeighbor or NoveltyThreshold or IsNovel" -v`
Expected: PASS

- [ ] **Step 5: Run the full test file to confirm no regression**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -v`
Expected: PASS (all tests, including the pre-existing `TestTocFieldFor` ones)

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/build_crossref_gt_ground_truth.py tests/test_build_crossref_gt_ground_truth.py
git commit -m "feat: add layout-feature novelty-gate functions to build_crossref_gt_ground_truth.py"
```

---

## Task 5: Diagnostic outline cross-check pure functions

**Files:**
- Modify: `evaluation/scripts/build_crossref_gt_ground_truth.py`
- Test: `tests/test_build_crossref_gt_ground_truth.py`

Adds `_flatten_outline` (turns a pypdf nested outline into flat `(title, page_index)` pairs) and `_outline_agreement_report` (fuzzy-matches each confirmed chapter's title against those pairs and logs disagreement -- purely informational, per the design spec's "diagnostic-only this round" decision). `_MIN_MATCH_SCORE` (already defined, `85.0`) is reused as the "confident outline match" bar, matching the confidence bar the existing content-search confirmation already uses -- no new magic number introduced.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_build_crossref_gt_ground_truth.py`. First add this import at the top of the file, alongside the existing `unittest` import:

```python
from unittest.mock import Mock
```

Then add:

```python
class _FakeDestination:
    def __init__(self, title: str):
        self.title = title


class TestFlattenOutline(unittest.TestCase):
    def test_flattens_nested_outline_in_order(self):
        outline = [
            _FakeDestination("Part I"),
            [_FakeDestination("Chapter 1"), _FakeDestination("Chapter 2")],
        ]
        reader = Mock()
        reader.get_destination_page_number.side_effect = [0, 5, 12]
        result = _flatten_outline(outline, reader)
        self.assertEqual(result, [("Part I", 0), ("Chapter 1", 5), ("Chapter 2", 12)])

    def test_skips_entries_with_no_title(self):
        outline = [_FakeDestination("")]
        reader = Mock()
        result = _flatten_outline(outline, reader)
        self.assertEqual(result, [])
        reader.get_destination_page_number.assert_not_called()

    def test_skips_entries_pypdf_cannot_resolve_a_page_for(self):
        outline = [_FakeDestination("Broken Entry")]
        reader = Mock()
        reader.get_destination_page_number.side_effect = Exception("unresolvable")
        result = _flatten_outline(outline, reader)
        self.assertEqual(result, [])


class TestOutlineAgreementReport(unittest.TestCase):
    def test_no_outline_returns_empty(self):
        chapters = [{"title": "Introduction", "pdf_start_index": 5}]
        self.assertEqual(_outline_agreement_report([], chapters), [])

    def test_agreement_produces_no_log_line(self):
        outline_entries = [("Introduction", 5)]
        chapters = [{"title": "Introduction", "pdf_start_index": 5}]
        self.assertEqual(_outline_agreement_report(outline_entries, chapters), [])

    def test_disagreement_logs_a_line(self):
        outline_entries = [("Introduction", 7)]
        chapters = [{"title": "Introduction", "pdf_start_index": 5}]
        lines = _outline_agreement_report(outline_entries, chapters)
        self.assertEqual(len(lines), 1)
        self.assertIn("outline=7", lines[0])
        self.assertIn("content-search=5", lines[0])

    def test_low_confidence_match_is_ignored(self):
        outline_entries = [("Completely Unrelated Text About Something Else", 99)]
        chapters = [{"title": "Introduction", "pdf_start_index": 5}]
        self.assertEqual(_outline_agreement_report(outline_entries, chapters), [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -k "FlattenOutline or OutlineAgreementReport" -v`
Expected: FAIL with `ImportError: cannot import name '_flatten_outline'`

Also update the import line at the top of `tests/test_build_crossref_gt_ground_truth.py` (from Task 4) to include the two new names:

```python
from evaluation.scripts.build_crossref_gt_ground_truth import (
    _flatten_outline,
    _is_novel,
    _nearest_neighbor_distance,
    _novelty_threshold,
    _outline_agreement_report,
    _toc_field_for,
)
```

- [ ] **Step 3: Implement the two functions**

In `evaluation/scripts/build_crossref_gt_ground_truth.py`, add after `_is_novel` (from Task 4), before `_citation_start`:

```python
def _flatten_outline(outline: list, reader: PdfReader) -> list[tuple[str, int]]:
    """Flattens a (possibly nested) pypdf outline -- a list where nested
    lists represent nesting -- into (title, page_index) pairs in reading
    order. Skips entries with no title, or whose page pypdf can't resolve
    (get_destination_page_number raising means the destination is
    malformed or points outside this PDF -- skip it, don't crash the
    whole reconciliation over one bad bookmark)."""
    entries: list[tuple[str, int]] = []
    for item in outline:
        if isinstance(item, list):
            entries.extend(_flatten_outline(item, reader))
            continue
        title = getattr(item, "title", None)
        if not title:
            continue
        try:
            page_index = reader.get_destination_page_number(item)
        except Exception:
            continue
        entries.append((title, page_index))
    return entries


def _outline_agreement_report(
    outline_entries: list[tuple[str, int]], confirmed_chapters: list[dict]
) -> list[str]:
    """Diagnostic-only cross-check (see design spec's "Outline scope"
    decision): for each confirmed chapter, fuzzy-matches its title against
    outline_entries and logs a line when the best confident match (score
    >= _MIN_MATCH_SCORE, the same bar the content-search confirmation
    itself uses) disagrees with the content-search-confirmed
    pdf_start_index. Returns an empty list when there's no outline, or
    when everything agrees -- this has zero effect on the migration
    decision, purely evidence for a future decision on whether outline
    agreement should be allowed to rescue failed confirmations."""
    if not outline_entries:
        return []
    lines = []
    for chapter in confirmed_chapters:
        best_page, best_score = None, 0.0
        for title, page_index in outline_entries:
            score = fuzz.partial_ratio(chapter["title"].lower(), title.lower())
            if score > best_score:
                best_page, best_score = page_index, score
        if best_score >= _MIN_MATCH_SCORE and best_page != chapter["pdf_start_index"]:
            lines.append(
                f'outline disagreement: chapter "{chapter["title"]}" '
                f"outline={best_page} content-search={chapter['pdf_start_index']}"
            )
    return lines
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -k "FlattenOutline or OutlineAgreementReport" -v`
Expected: PASS

- [ ] **Step 5: Run the full test file to confirm no regression**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/build_crossref_gt_ground_truth.py tests/test_build_crossref_gt_ground_truth.py
git commit -m "feat: add diagnostic-only outline cross-check to build_crossref_gt_ground_truth.py"
```

---

## Task 6: Wire the novelty gate + outline diagnostic + `pending/` routing into `process_book`

**Files:**
- Modify: `evaluation/scripts/build_crossref_gt_ground_truth.py:1-33` (module docstring + imports/constants), `:142-278` (`process_book`/`main`)

This task integrates Tasks 4-5's pure functions into the real per-book pipeline, changes the migration target from `open-access/` to `pending/`, and adds a one-time-per-run "novelty baseline" loader. The baseline-loading and PDF-feature-extraction parts touch real pdfalto/corpus IO, so beyond the one narrow unit test below, this task is verified by a real dry run against the existing corpus (Step 8) rather than further mocking -- consistent with how `process_book`'s existing PDF-reading logic has never been unit-tested either (see `tests/test_build_crossref_gt_ground_truth.py`'s module docstring).

- [ ] **Step 1: Update imports and add the `NoveltyBaseline` dataclass**

In `evaluation/scripts/build_crossref_gt_ground_truth.py`, replace the import block:

```python
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
from rapidfuzz import fuzz

from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
```

with:

```python
import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
from rapidfuzz import fuzz
from sklearn.preprocessing import StandardScaler

from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.scripts.evaluate_layout_toc_classifier import build_feature_table, load_book_corpus
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
from evaluation.scripts.layout_features import FEATURE_NAMES, extract_page_features
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary
```

Then replace:

```python
_CROSSREF_DIR = Path(__file__).resolve().parent.parent / "crossref_gt"
_OPEN_ACCESS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "open-access"

_MIN_OFFSET_VOTES = 5  # pages agreeing on the same offset, before it's trusted
_WINDOW = 6  # +/- pages searched around each offset-derived candidate index
_MIN_MATCH_SCORE = 85.0
_MIN_CONFIRMED_FRACTION = 0.8
_MIN_CONFIRMED_CHAPTERS = 3
```

with:

```python
_CROSSREF_DIR = Path(__file__).resolve().parent.parent / "crossref_gt"
_OPEN_ACCESS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "open-access"
_PENDING_DIR = Path(__file__).resolve().parent.parent / "corpus" / "pending"
_ALTO_CACHE_DIR = _CROSSREF_DIR / ".layout-cache"

_MIN_OFFSET_VOTES = 5  # pages agreeing on the same offset, before it's trusted
_WINDOW = 6  # +/- pages searched around each offset-derived candidate index
_MIN_MATCH_SCORE = 85.0
_MIN_CONFIRMED_FRACTION = 0.8
_MIN_CONFIRMED_CHAPTERS = 3
_DEFAULT_NOVELTY_PERCENTILE = 90.0  # see design spec's "Novelty metric" decision
```

Add the module docstring note about the new behavior. Replace the module docstring's second paragraph (the "A book where the offset can't be derived..." paragraph) by adding this paragraph immediately after it, before the closing `"""`:

```python

Migrated books land in evaluation/corpus/pending/, not open-access/ --
confirmation + novelty give high GT confidence, but open-access/ is what
this pilot's canonical numbers are measured against, so a human gets a
chance to spot-check (or evaluate in isolation via
evaluate_layout_toc_classifier.py's --corpora flag) before a book affects
those numbers. See
docs/superpowers/specs/2026-08-11-crossref-gt-corpus-expansion-design.md.
```

Now add the `NoveltyBaseline` dataclass and its loader function, after `_is_novel` (Task 4) and `_outline_agreement_report` (Task 5), before `_citation_start`:

```python
@dataclass
class NoveltyBaseline:
    """Precomputed once per script run (not once per candidate book) --
    reloading and re-fitting against the whole open-access corpus for
    every candidate would be correct but wasteful."""

    pdfalto_bin: str
    scaler: StandardScaler
    chapter_first_vectors: list[list[float]]
    threshold: float


def _load_novelty_baseline(pdfalto_bin: str, novelty_percentile: float) -> NoveltyBaseline:
    """Loads the current evaluation/corpus/open-access/ corpus, fits a
    StandardScaler on its full feature-row set (the same normalization
    evaluate_layout_toc_classifier's own LogisticRegression classifier
    uses, so novelty distances are computed in the same space it reasons
    in), and returns the scaled chapter_first vectors plus the novelty
    distance threshold derived from their own leave-one-out
    nearest-neighbor distances."""

    def cache_dir_for(corpus: str) -> Path:
        return _OPEN_ACCESS_DIR.parent / corpus / ".layout-cache"

    books = load_book_corpus(corpora=["open-access"])
    rows = build_feature_table(books, cache_dir_for, pdfalto_bin)
    X = [[row["features"][name] for name in FEATURE_NAMES] for row in rows]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).tolist()
    chapter_first_vectors = [
        vector for vector, row in zip(X_scaled, rows) if row["label"] == LABEL_CHAPTER_FIRST
    ]
    threshold = _novelty_threshold(chapter_first_vectors, novelty_percentile)
    return NoveltyBaseline(pdfalto_bin, scaler, chapter_first_vectors, threshold)
```

- [ ] **Step 2: Update `process_book`'s signature and migration target**

Replace the `process_book` function's signature and the two `SKIP: no PDF`/`already migrated` early-return lines:

```python
def process_book(book: dict, dry_run: bool) -> tuple[str, str]:
    """Returns (isbn, outcome_message)."""
    isbn = book["isbn"]
    pdf_path = _CROSSREF_DIR / f"{isbn}.pdf"
    crossref_path = _CROSSREF_DIR / f"{isbn}.crossref.json"
    target_pdf = _OPEN_ACCESS_DIR / f"{isbn}.pdf"
    target_expected = _OPEN_ACCESS_DIR / f"{isbn}.expected.json"

    if not pdf_path.exists():
        return isbn, "SKIP: no PDF (fetch_crossref_gt_corpus.py first)"
    if not crossref_path.exists():
        return isbn, "SKIP: no crossref.json"
    if target_expected.exists():
        return isbn, "SKIP: already migrated (evaluation/corpus/open-access/*.expected.json exists)"
```

with:

```python
def process_book(book: dict, dry_run: bool, novelty_baseline: NoveltyBaseline) -> tuple[str, str]:
    """Returns (isbn, outcome_message)."""
    isbn = book["isbn"]
    pdf_path = _CROSSREF_DIR / f"{isbn}.pdf"
    crossref_path = _CROSSREF_DIR / f"{isbn}.crossref.json"
    target_pdf = _PENDING_DIR / f"{isbn}.pdf"
    target_expected = _PENDING_DIR / f"{isbn}.expected.json"

    if not pdf_path.exists():
        return isbn, "SKIP: no PDF (fetch_crossref_gt_corpus.py first)"
    if not crossref_path.exists():
        return isbn, "SKIP: no crossref.json"
    if target_expected.exists():
        return isbn, "SKIP: already migrated (evaluation/corpus/pending/*.expected.json exists)"
```

- [ ] **Step 3: Add the novelty gate + outline diagnostic after the existing confirmation gate**

The existing confirmation-fraction check and sanity check stay exactly as they are. Immediately after:

```python
    error = _sanity_check(confirmed, total_pages)
    if error:
        return isbn, f"SKIP: sanity check failed after reconciliation: {error}"

    toc_field, write_toc_key, toc_status = _toc_field_for(toc_pages)
```

insert the novelty check and outline diagnostic, so the block becomes:

```python
    error = _sanity_check(confirmed, total_pages)
    if error:
        return isbn, f"SKIP: sanity check failed after reconciliation: {error}"

    alto_path = ensure_alto_xml(pdf_path, _ALTO_CACHE_DIR, novelty_baseline.pdfalto_bin)
    page_features = extract_page_features(str(alto_path))
    candidate_rows = [
        [page_features[c["pdf_start_index"]][name] for name in FEATURE_NAMES]
        for c in confirmed
        if c["pdf_start_index"] in page_features
    ]
    if not candidate_rows:
        return isbn, "SKIP: no layout features extracted for confirmed chapter-first pages"
    candidate_vectors = novelty_baseline.scaler.transform(candidate_rows).tolist()
    if not _is_novel(candidate_vectors, novelty_baseline.chapter_first_vectors, novelty_baseline.threshold):
        return isbn, (
            f"SKIP: not novel (no confirmed chapter-first page's nearest-neighbor "
            f"distance reaches the {novelty_baseline.threshold:.3f} threshold)"
        )

    outline_entries = _flatten_outline(reader.outline or [], reader)
    for line in _outline_agreement_report(outline_entries, confirmed):
        print(f"  [{isbn}] {line}")

    toc_field, write_toc_key, toc_status = _toc_field_for(toc_pages)
```

- [ ] **Step 4: Update the migration write block to target `_PENDING_DIR`**

Replace:

```python
    _OPEN_ACCESS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, target_pdf)
    payload = {"chapters": confirmed}
    if write_toc_key:
        payload["toc"] = toc_field
    target_expected.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_path = _OPEN_ACCESS_DIR / "manifest.json"
```

with:

```python
    _PENDING_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, target_pdf)
    payload = {"chapters": confirmed}
    if write_toc_key:
        payload["toc"] = toc_field
    target_expected.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_path = _PENDING_DIR / "manifest.json"
```

(The rest of that block -- reading the manifest, appending the entry, writing it back -- is unchanged; it now reads/writes `evaluation/corpus/pending/manifest.json` instead of `open-access/`'s.)

- [ ] **Step 5: Update `main()` to build the novelty baseline once and pass it through**

Replace:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing files")
    args = parser.parse_args()

    manifest = json.loads((_CROSSREF_DIR / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for book in manifest["books"]:
        result = process_book(book, args.dry_run)
        results.append(result)
        print(f"[{result[0]}] {result[1]}")

    n_ok = sum(1 for _, msg in results if msg.startswith("OK"))
    print(f"\n{n_ok}/{len(results)} book(s) {'would be ' if args.dry_run else ''}migrated to open-access/")
```

with:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing files")
    parser.add_argument("--pdfalto-bin", default=None, help="Path to the pdfalto binary (see pdfalto_runner.py)")
    parser.add_argument(
        "--novelty-percentile",
        type=float,
        default=_DEFAULT_NOVELTY_PERCENTILE,
        help=(
            "Percentile of the existing open-access chapter-first corpus's own "
            "leave-one-out nearest-neighbor distances used as the novelty threshold. "
            f"Default: {_DEFAULT_NOVELTY_PERCENTILE}."
        ),
    )
    args = parser.parse_args()
    pdfalto_bin = resolve_pdfalto_binary(args.pdfalto_bin)
    novelty_baseline = _load_novelty_baseline(pdfalto_bin, args.novelty_percentile)

    manifest = json.loads((_CROSSREF_DIR / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for book in manifest["books"]:
        result = process_book(book, args.dry_run, novelty_baseline)
        results.append(result)
        print(f"[{result[0]}] {result[1]}")

    n_ok = sum(1 for _, msg in results if msg.startswith("OK"))
    print(f"\n{n_ok}/{len(results)} book(s) {'would be ' if args.dry_run else ''}migrated to pending/")
```

- [ ] **Step 6: Run the full test file to confirm no regression**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -v`
Expected: PASS (all tests -- `_toc_field_for` and Tasks 4/5's tests exercise the pure functions in isolation; `process_book`/`main`/`_load_novelty_baseline` remain untested by pytest, matching this file's existing convention of verifying the PDF-reading pipeline manually)

- [ ] **Step 7: Verify `--help` renders the new flags**

Run: `uv run python evaluation/scripts/build_crossref_gt_ground_truth.py --help`
Expected: output includes `--pdfalto-bin` and `--novelty-percentile` (default `90.0`) alongside the existing `--dry-run`.

- [ ] **Step 8: Real dry run against the existing corpus**

```bash
uv run python evaluation/scripts/build_crossref_gt_ground_truth.py --dry-run
```

Expected: runs to completion without a traceback. Every one of the 43 existing `crossref_gt/manifest.json` entries reports either `SKIP: already migrated (evaluation/corpus/pending/*.expected.json exists)` (none should, since none have ever been migrated to `pending/`) or a real outcome -- the 31 previously-`open-access/`-migrated books will report `SKIP: already migrated...`? No: `target_expected` now points at `_PENDING_DIR`, which won't have those 31 books' `.expected.json` files (they're in `open-access/`), so those 31 will actually re-run the full confirmation + novelty pipeline and either report `OK (dry-run): ...` (ready to land in `pending/` too, redundantly, since they're already in `open-access/`) or `SKIP: not novel` (expected for most, since they're already-known templates the classifier's `open-access/` corpus was fitted on). **This is expected, not a bug** -- confirm by spot-checking a few of the 31 known-migrated ISBNs' dry-run output line and noting whether they say `SKIP: not novel` (the common case) or `OK`. Do not act on this output (no real run, `--dry-run` only) -- it's a smoke test that the new code path executes cleanly end-to-end against real PDFs/pdfalto, not a decision point. If pdfalto is not installed/resolvable in this environment, this step will fail with a `RuntimeError` from `ensure_alto_xml` -- if so, note that limitation in the final report rather than treating it as a code bug, and skip to Step 9.

- [ ] **Step 9: Commit**

```bash
git add evaluation/scripts/build_crossref_gt_ground_truth.py
git commit -m "feat: wire novelty gate + outline diagnostic + pending/ routing into build_crossref_gt_ground_truth.py"
```

---

## Task 7: `discover_crossref_candidates.py`

**Files:**
- Create: `evaluation/scripts/discover_crossref_candidates.py`
- Test: `tests/test_discover_crossref_candidates.py`

Seed publisher list with real, verified Crossref member IDs (looked up live against `https://api.crossref.org/members` and `https://api.crossref.org/prefixes/<prefix>` during planning -- not guessed):

| Publisher | Member ID | Default language | Verified via |
| --- | --- | --- | --- |
| Open Book Publishers | 4923 | en | member search, DOI prefix 10.11647 matches an existing manifest entry |
| transcript Verlag | 5471 | de | member search, DOI prefix 10.14361 matches an existing manifest entry |
| UCL Press | 5433 | en | member search, DOI prefix 10.14324 matches an existing manifest entry |
| Athabasca University Press | 5869 | en | member search, DOI prefix 10.15215 matches an existing manifest entry |
| Springer Science and Business Media LLC | 297 | en | `/prefixes/10.1007` lookup (existing manifest's Springer entries all use this prefix) |
| OpenEdition | 2399 | fr | `/prefixes/10.4000` lookup -- **note:** the design spec's seed list named "Presses universitaires de Rennes" and "Africae" as if they were separate members; both are actually OpenEdition imprints sharing this one Crossref member ID (confirmed live), so they're consolidated into one seed entry here. See this plan's closing "Alternatives not taken" section. |

- [ ] **Step 1: Write the failing tests for the small pure functions**

Create `tests/test_discover_crossref_candidates.py`:

```python
"""Unit tests for evaluation/scripts/discover_crossref_candidates.py's pure
logic (URL/license resolution, dedup, language-priority ranking) against
mocked httpx responses -- no live network. The real Crossref-search
orchestration (discover()/main()) is exercised manually, matching
fetch_crossref_gt_corpus.py's existing convention of no pytest coverage for
its own network-calling main() entry point."""

import unittest
from collections import Counter
from unittest.mock import Mock

from evaluation.scripts.discover_crossref_candidates import (
    _crossref_link_pdf_url,
    _crossref_publisher_works,
    _is_new_candidate,
    _item_isbn,
    _item_title,
    _language_priority,
    _openalex_pdf_url,
    _select_candidates,
    _unpaywall_pdf_url,
    resolve_download_url,
)


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


class TestItemIsbnAndTitle(unittest.TestCase):
    def test_item_isbn_takes_first_entry(self):
        self.assertEqual(_item_isbn({"ISBN": ["9781234567897", "9781234567904"]}), "9781234567897")

    def test_item_isbn_none_when_absent(self):
        self.assertIsNone(_item_isbn({}))

    def test_item_title_takes_first_entry(self):
        self.assertEqual(_item_title({"title": ["A Book Title"]}), "A Book Title")

    def test_item_title_none_when_absent(self):
        self.assertIsNone(_item_title({}))


class TestCrossrefLinkPdfUrl(unittest.TestCase):
    def test_finds_application_pdf_content_type(self):
        item = {"link": [{"URL": "https://x/y.pdf", "content-type": "application/pdf"}]}
        self.assertEqual(_crossref_link_pdf_url(item), "https://x/y.pdf")

    def test_finds_unspecified_content_type(self):
        # Some publishers (e.g. Open Book Publishers) register their real
        # PDF link with content-type "unspecified" rather than the correct
        # MIME type -- confirmed against a live Crossref record.
        item = {"link": [{"URL": "https://obp/z.pdf", "content-type": "unspecified"}]}
        self.assertEqual(_crossref_link_pdf_url(item), "https://obp/z.pdf")

    def test_skips_non_pdf_content_type(self):
        item = {"link": [{"URL": "https://x/y.html", "content-type": "text/html"}]}
        self.assertIsNone(_crossref_link_pdf_url(item))

    def test_returns_none_when_no_link(self):
        self.assertIsNone(_crossref_link_pdf_url({}))


class TestUnpaywallPdfUrl(unittest.TestCase):
    def test_returns_url_for_pdf(self):
        client = Mock()
        client.get.return_value = _json_response(
            {"best_oa_location": {"url_for_pdf": "https://repo/paper.pdf"}}
        )
        self.assertEqual(_unpaywall_pdf_url("10.1/x", client, None), "https://repo/paper.pdf")

    def test_none_when_no_doi(self):
        client = Mock()
        self.assertIsNone(_unpaywall_pdf_url(None, client, None))
        client.get.assert_not_called()

    def test_none_when_request_fails(self):
        client = Mock()
        client.get.side_effect = Exception("network error")
        self.assertIsNone(_unpaywall_pdf_url("10.1/x", client, None))


class TestOpenAlexPdfUrl(unittest.TestCase):
    def test_returns_pdf_url(self):
        client = Mock()
        client.get.return_value = _json_response({"best_oa_location": {"pdf_url": "https://repo/paper.pdf"}})
        self.assertEqual(_openalex_pdf_url("10.1/x", client), "https://repo/paper.pdf")

    def test_none_when_no_doi(self):
        client = Mock()
        self.assertIsNone(_openalex_pdf_url(None, client))
        client.get.assert_not_called()


class TestResolveDownloadUrl(unittest.TestCase):
    def test_prefers_crossref_link(self):
        item = {"link": [{"URL": "https://crossref/x.pdf", "content-type": "application/pdf"}]}
        client = Mock()
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), ("https://crossref/x.pdf", "crossref"))
        client.get.assert_not_called()

    def test_falls_back_to_unpaywall(self):
        item = {}
        client = Mock()
        client.get.return_value = _json_response(
            {"best_oa_location": {"url_for_pdf": "https://unpaywall/x.pdf"}}
        )
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), ("https://unpaywall/x.pdf", "unpaywall"))

    def test_falls_back_to_openalex_when_unpaywall_empty(self):
        item = {}
        client = Mock()
        client.get.side_effect = [
            _json_response({"best_oa_location": None}),
            _json_response({"best_oa_location": {"pdf_url": "https://openalex/x.pdf"}}),
        ]
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), ("https://openalex/x.pdf", "openalex"))

    def test_none_when_all_three_fail(self):
        item = {}
        client = Mock()
        client.get.return_value = _json_response({"best_oa_location": None})
        url, source = resolve_download_url(item, "10.1/x", client, None)
        self.assertEqual((url, source), (None, None))


class TestIsNewCandidate(unittest.TestCase):
    def test_new_isbn_is_new(self):
        self.assertTrue(_is_new_candidate("111", "10.1/a", {"999"}, {"10.1/z"}))

    def test_known_isbn_is_not_new(self):
        self.assertFalse(_is_new_candidate("111", "10.1/a", {"111"}, set()))

    def test_known_doi_is_not_new_even_with_new_isbn(self):
        self.assertFalse(_is_new_candidate("111", "10.1/a", set(), {"10.1/a"}))

    def test_no_isbn_is_not_new(self):
        self.assertFalse(_is_new_candidate(None, "10.1/a", set(), set()))


class TestLanguagePriority(unittest.TestCase):
    def test_ranks_least_represented_first(self):
        counts = Counter({"en": 30, "de": 9, "fr": 2})
        self.assertEqual(_language_priority(counts, {"en", "de", "fr"}), ["fr", "de", "en"])

    def test_unseen_language_ranks_first(self):
        counts = Counter({"en": 30})
        self.assertEqual(_language_priority(counts, {"en", "es"}), ["es", "en"])


class TestSelectCandidates(unittest.TestCase):
    def test_caps_per_language_in_priority_order(self):
        by_language = {
            "en": [{"isbn": "1"}, {"isbn": "2"}, {"isbn": "3"}],
            "fr": [{"isbn": "10"}, {"isbn": "11"}],
        }
        selected = _select_candidates(by_language, priority=["fr", "en"], max_per_language=1)
        self.assertEqual(selected, [{"isbn": "10"}, {"isbn": "1"}])


class TestCrossrefPublisherWorks(unittest.TestCase):
    def test_stops_when_page_smaller_than_rows(self):
        client = Mock()
        client.get.return_value = _json_response(
            {"message": {"items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}]}}
        )
        result = _crossref_publisher_works("4923", "monograph", client, None, rows=100)
        self.assertEqual(len(result), 2)
        self.assertEqual(client.get.call_count, 1)

    def test_paginates_across_full_pages(self):
        client = Mock()
        page1 = _json_response({"message": {"items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}]}})
        page2 = _json_response({"message": {"items": [{"DOI": "10.1/c"}]}})
        client.get.side_effect = [page1, page2]
        result = _crossref_publisher_works("4923", "monograph", client, None, rows=2)
        self.assertEqual(len(result), 3)
        self.assertEqual(client.get.call_count, 2)

    def test_network_error_returns_partial_results_without_raising(self):
        import httpx

        client = Mock()
        page1 = _json_response({"message": {"items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}]}})
        client.get.side_effect = [page1, httpx.ConnectError("boom")]
        result = _crossref_publisher_works("4923", "monograph", client, None, rows=2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_discover_crossref_candidates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.discover_crossref_candidates'`

- [ ] **Step 3: Create `evaluation/scripts/discover_crossref_candidates.py`**

```python
#!/usr/bin/env python3
"""Discovers new open-access edited-volume candidates for
evaluation/crossref_gt/manifest.json, seeded from Crossref publishers
already represented there. See
docs/superpowers/specs/2026-08-11-crossref-gt-corpus-expansion-design.md.

Resolves each candidate's direct PDF URL from three independent sources,
tried in order (Crossref's own registered link, then Unpaywall, then
OpenAlex), and prioritizes appending candidates in currently-underrepresented
languages first so automated growth doesn't quietly re-converge on
English-language volumes just because they're easiest to find.

This script only appends to manifest.json -- run
fetch_crossref_gt_corpus.py afterwards to actually download the new
entries' PDFs and Crossref chapter metadata, then
build_crossref_gt_ground_truth.py to reconcile and (if confirmed + novel)
migrate them into evaluation/corpus/pending/.

Usage:
    uv run python evaluation/scripts/discover_crossref_candidates.py
    uv run python evaluation/scripts/discover_crossref_candidates.py --dry-run
    uv run python evaluation/scripts/discover_crossref_candidates.py --max-per-language 3
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)
from evaluation.scripts.fetch_crossref_gt_corpus import (
    _DEFAULT_CONTACT_EMAIL,
    _item_license_url,
    _unpaywall_license_url,
)

_CROSSREF_DIR = Path(__file__).resolve().parent.parent / "crossref_gt"
_OPEN_ACCESS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "open-access"

_UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"
_OPENALEX_BASE_URL = "https://api.openalex.org/works"

_WORK_TYPES = ["monograph", "edited-book"]
_DEFAULT_MAX_PER_LANGUAGE = 5
_DEFAULT_MAX_RESULTS_PER_QUERY = 300

# Real Crossref member IDs, looked up live against
# https://api.crossref.org/members (by publisher name) and
# https://api.crossref.org/prefixes/<prefix> (by cross-checking against
# each publisher's DOI prefix already present in manifest.json) --
# not guessed. "Presses universitaires de Rennes" and "Africae" (both
# named in the design spec) turned out to be OpenEdition imprints sharing
# one Crossref member, not separate members -- consolidated into the
# single OpenEdition entry below; see the implementation plan's
# "Alternatives not taken" section.
_SEED_PUBLISHERS = [
    {"member_id": "4923", "publisher": "Open Book Publishers", "default_language": "en"},
    {"member_id": "5471", "publisher": "transcript Verlag", "default_language": "de"},
    {"member_id": "5433", "publisher": "UCL Press", "default_language": "en"},
    {"member_id": "5869", "publisher": "Athabasca University Press", "default_language": "en"},
    {"member_id": "297", "publisher": "Springer Science and Business Media LLC", "default_language": "en"},
    {"member_id": "2399", "publisher": "OpenEdition", "default_language": "fr"},
]


def _item_isbn(item: dict) -> Optional[str]:
    isbns = item.get("ISBN") or []
    return isbns[0] if isbns else None


def _item_title(item: dict) -> Optional[str]:
    titles = item.get("title") or []
    return titles[0] if titles else None


def _crossref_link_pdf_url(item: dict) -> Optional[str]:
    """The first link entry that looks like a real full-text PDF resource.
    content-type "application/pdf" is the correct MIME type; "unspecified"
    is included too because several publishers (confirmed against a live
    Open Book Publishers record) register their actual PDF link that way
    instead."""
    for link in item.get("link") or []:
        if link.get("content-type") in ("application/pdf", "unspecified"):
            return link.get("URL")
    return None


def _unpaywall_pdf_url(doi: Optional[str], client: httpx.Client, contact_email: Optional[str]) -> Optional[str]:
    """Fallback PDF URL lookup via Unpaywall, tried only when Crossref's
    own link array has none. Never raises -- one bad lookup must not abort
    the batch."""
    if not doi:
        return None
    email = contact_email or _DEFAULT_CONTACT_EMAIL
    try:
        response = client.get(f"{_UNPAYWALL_BASE_URL}/{doi}", params={"email": email}, timeout=10.0)
        response.raise_for_status()
        location = response.json().get("best_oa_location")
    except Exception as exc:
        print(f"  [warn] Unpaywall URL lookup failed for {doi}: {exc}")
        return None
    return location.get("url_for_pdf") if location else None


def _openalex_pdf_url(doi: Optional[str], client: httpx.Client) -> Optional[str]:
    """Second fallback PDF URL lookup via OpenAlex, tried only when
    neither Crossref nor Unpaywall has one. Never raises."""
    if not doi:
        return None
    try:
        response = client.get(f"{_OPENALEX_BASE_URL}/doi:{doi}", timeout=10.0)
        response.raise_for_status()
        location = response.json().get("best_oa_location")
    except Exception as exc:
        print(f"  [warn] OpenAlex URL lookup failed for {doi}: {exc}")
        return None
    return location.get("pdf_url") if location else None


def resolve_download_url(
    item: dict, doi: Optional[str], client: httpx.Client, contact_email: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Tries, in order, Crossref's own link array, then Unpaywall, then
    OpenAlex. Returns (url, source) -- source is "crossref", "unpaywall",
    "openalex", or None if none of the three has one."""
    url = _crossref_link_pdf_url(item)
    if url:
        return url, "crossref"
    url = _unpaywall_pdf_url(doi, client, contact_email)
    if url:
        return url, "unpaywall"
    url = _openalex_pdf_url(doi, client)
    if url:
        return url, "openalex"
    return None, None


def _is_new_candidate(
    isbn: Optional[str], doi: Optional[str], existing_isbns: set[str], existing_dois: set[str]
) -> bool:
    if not isbn or isbn in existing_isbns:
        return False
    if doi and doi in existing_dois:
        return False
    return True


def _language_counts(*manifest_paths: Path) -> Counter:
    counts: Counter = Counter()
    for path in manifest_paths:
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for book in manifest["books"]:
            language = book.get("language")
            if language:
                counts[language] += 1
    return counts


def _language_priority(counts: Counter, languages: set[str]) -> list[str]:
    """Ranks languages by ascending current count (most underrepresented
    first); a language with no entries yet ranks ahead of any with at
    least one."""
    return sorted(languages, key=lambda language: counts.get(language, 0))


def _select_candidates(
    candidates_by_language: dict[str, list[dict]], priority: list[str], max_per_language: int
) -> list[dict]:
    selected = []
    for language in priority:
        selected.extend(candidates_by_language.get(language, [])[:max_per_language])
    return selected


def _crossref_publisher_works(
    member_id: str,
    work_type: str,
    client: httpx.Client,
    contact_email: Optional[str],
    rows: int = 100,
    max_results: int = _DEFAULT_MAX_RESULTS_PER_QUERY,
) -> list[dict]:
    """Paginates https://api.crossref.org/works?filter=member:<id>,type:<work_type>
    via offset, stopping once a page returns fewer than `rows` items or
    `max_results` total items have been collected. Never raises -- a
    network/HTTP/JSON failure for one page is logged and treated as "no
    more pages" rather than aborting the whole discovery run. "language"
    is deliberately not in the select list -- Crossref rejects it as an
    invalid select field for the /works route (confirmed live), and in
    practice book-type records essentially never carry it anyway."""
    items: list[dict] = []
    offset = 0
    while len(items) < max_results:
        params: dict[str, str | int] = {
            "filter": f"member:{member_id},type:{work_type}",
            "select": "DOI,ISBN,title,link,publisher,type,license",
            "rows": rows,
            "offset": offset,
        }
        if contact_email:
            params["mailto"] = contact_email

        response = None
        for _attempt in range(_MAX_RETRIES):
            try:
                response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
            except httpx.HTTPError as exc:
                print(f"  [warn] network error fetching member {member_id}/{work_type}: {exc}")
                return items
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
            time.sleep(delay)
        else:
            print(f"  [warn] exhausted retries (429) fetching member {member_id}/{work_type}")
            return items

        try:
            response.raise_for_status()
            page_items = response.json()["message"]["items"]
        except Exception as exc:
            print(f"  [warn] bad Crossref response for member {member_id}/{work_type}: {exc}")
            return items

        items.extend(page_items)
        if len(page_items) < rows:
            break
        offset += rows
    return items[:max_results]


def discover(max_per_language: int, dry_run: bool, contact_email: Optional[str]) -> int:
    manifest_path = _CROSSREF_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_isbns = {book["isbn"] for book in manifest["books"]}
    existing_dois = {book["doi"] for book in manifest["books"] if book.get("doi")}

    lang_counts = _language_counts(manifest_path, _OPEN_ACCESS_DIR / "manifest.json")

    candidates_by_language: dict[str, list[dict]] = {}
    with httpx.Client(follow_redirects=True) as client:
        for seed in _SEED_PUBLISHERS:
            for work_type in _WORK_TYPES:
                items = _crossref_publisher_works(seed["member_id"], work_type, client, contact_email)
                for item in items:
                    isbn = _item_isbn(item)
                    doi = item.get("DOI")
                    if not _is_new_candidate(isbn, doi, existing_isbns, existing_dois):
                        continue
                    download_url, url_source = resolve_download_url(item, doi, client, contact_email)
                    if not download_url:
                        print(f"  [skip] {isbn}: no download URL found (Crossref/Unpaywall/OpenAlex)")
                        continue
                    license_url = _item_license_url(item)
                    license_source = "crossref" if license_url else None
                    if license_url is None:
                        license_url = _unpaywall_license_url(doi, client, contact_email)
                        license_source = "unpaywall" if license_url else None
                    language = item.get("language") or seed["default_language"]
                    candidate = {
                        "isbn": isbn,
                        "title": _item_title(item),
                        "doi": doi,
                        "domain": None,
                        "language": language,
                        "publisher": seed["publisher"],
                        "download_url": download_url,
                        "license": license_url,
                        "license_source": license_source,
                        "discovery_source": "auto",
                    }
                    candidates_by_language.setdefault(language, []).append(candidate)
                    existing_isbns.add(isbn)
                    if doi:
                        existing_dois.add(doi)

    priority = _language_priority(lang_counts, set(candidates_by_language))
    added = _select_candidates(candidates_by_language, priority, max_per_language)

    if added and not dry_run:
        manifest["books"].extend(added)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'Would add' if dry_run else 'Added'} {len(added)} new candidate(s):")
    for candidate in added:
        print(f"  - {candidate['isbn']} ({candidate['language']}): {candidate['title']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would be added without writing manifest.json")
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=_DEFAULT_MAX_PER_LANGUAGE,
        help=(
            "Cap on how many new candidates of one language get appended per run, "
            f"so one prolific publisher's catalog can't monopolize a run. Default: {_DEFAULT_MAX_PER_LANGUAGE}."
        ),
    )
    parser.add_argument("--contact-email", default=_DEFAULT_CONTACT_EMAIL, help="Crossref/Unpaywall polite-pool contact email")
    args = parser.parse_args()
    return discover(args.max_per_language, args.dry_run, args.contact_email)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_discover_crossref_candidates.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Verify `--help` renders correctly**

Run: `uv run python evaluation/scripts/discover_crossref_candidates.py --help`
Expected: output includes `--dry-run`, `--max-per-language` (default `5`), `--contact-email`.

- [ ] **Step 6: Real dry run against live Crossref**

```bash
uv run python evaluation/scripts/discover_crossref_candidates.py --dry-run
```

Expected: runs to completion without a traceback, printing a `Would add N new candidate(s):` summary line followed by one line per candidate (or `Would add 0 new candidate(s):` if every discovered work is already in the manifest or none resolve a download URL -- both are legitimate outcomes, not failures). Skim the output for `[warn]`/`[skip]` lines to sanity-check they look like genuine misses (no PDF link found, 429 exhaustion) rather than a systematic bug (e.g. every single candidate being skipped would warrant investigation before proceeding).

- [ ] **Step 7: Commit**

```bash
git add evaluation/scripts/discover_crossref_candidates.py tests/test_discover_crossref_candidates.py
git commit -m "feat: add discover_crossref_candidates.py for automated GT corpus discovery"
```

---

## Task 8: Run the real end-to-end pipeline and update documentation

**Files:**
- Modify: `evaluation/crossref_gt/README.md`

This task actually runs the new pipeline for real (not `--dry-run`) against live Crossref/Unpaywall/OpenAlex and the real corpus, so its outcome depends on what those live services return at run time -- unlike every other task in this plan, the exact result can't be pinned down in advance. Record what actually happened in the final report (per this plan's "document alternatives not taken" instruction covering open items generally).

- [ ] **Step 1: Run discovery for real**

```bash
uv run python evaluation/scripts/discover_crossref_candidates.py
```

This appends new `discovery_source: "auto"` entries to `evaluation/crossref_gt/manifest.json` (or reports `Added 0 new candidate(s)` -- also a valid outcome). Inspect the diff:

```bash
git diff evaluation/crossref_gt/manifest.json
```

- [ ] **Step 2: Fetch the new candidates' PDFs and Crossref metadata**

```bash
uv run python evaluation/scripts/fetch_crossref_gt_corpus.py
```

(Unchanged script -- skips every already-downloaded entry, so this only fetches whatever Step 1 actually added.)

- [ ] **Step 3: Run reconciliation for real**

```bash
uv run python evaluation/scripts/build_crossref_gt_ground_truth.py
```

Read the full output. Each new candidate reports one of: `OK: ... written` (migrated into `evaluation/corpus/pending/`), a confirmation-fraction `SKIP`, or the new `SKIP: not novel` outcome. Note the actual counts for the final report.

- [ ] **Step 4: Inspect what landed in `pending/`**

```bash
git status evaluation/corpus/pending/
```

(These are gitignored PDFs plus a committed `manifest.json`/`.expected.json` per the existing corpus convention -- confirm the expected files are present for whatever the previous step actually migrated.)

- [ ] **Step 5: Update `evaluation/crossref_gt/README.md`**

Add a new section (after the existing "Downloading: host-specific quirks" section, at the end of the file):

```markdown

## Automated discovery

`evaluation/scripts/discover_crossref_candidates.py` finds new candidates
automatically, seeded from the OA publishers already in this manifest
(Crossref member IDs recorded in the script itself), resolving each
candidate's PDF URL via Crossref's own registered link, then Unpaywall,
then OpenAlex. New entries get `"discovery_source": "auto"` (existing
curated entries are `"manual"`). Candidates in currently-underrepresented
languages are prioritized; `--max-per-language` (default 5) caps how many
of one language get added per run.

```bash
uv run python evaluation/scripts/discover_crossref_candidates.py
uv run python evaluation/scripts/discover_crossref_candidates.py --dry-run
```

Run `fetch_crossref_gt_corpus.py` afterwards to download the new entries,
then `build_crossref_gt_ground_truth.py` to reconcile them. Since
`build_crossref_gt_ground_truth.py` now also gates migration on a
layout-feature **novelty check** (in addition to the existing
offset-consensus/content-search confirmation), confirmed candidates land
in `evaluation/corpus/pending/`, not `open-access/` -- see
`docs/superpowers/specs/2026-08-11-crossref-gt-corpus-expansion-design.md`.
Promoting a `pending/` book into `open-access/` (after reviewing it, or
evaluating it in isolation via `evaluate_layout_toc_classifier.py
--corpora open-access,pending`) is still the manual step
`evaluation/CLAUDE.md` already documents.
```

- [ ] **Step 6: Commit**

```bash
git add evaluation/crossref_gt/README.md evaluation/crossref_gt/manifest.json evaluation/corpus/pending/manifest.json
git add evaluation/corpus/pending/*.expected.json 2>/dev/null || true
git commit -m "docs: document automated discovery in crossref_gt/README.md; run first discovery pass"
```

(If Step 1 added zero candidates and Step 3 migrated zero books, this commit will only contain the README update -- that's fine, adjust the commit message to say so, e.g. `docs: document automated discovery in crossref_gt/README.md (first live run added 0 candidates -- see final report)`.)

---

## Alternatives not taken / autonomous design decisions

Per the user's instruction to resolve open questions autonomously during planning/implementation rather than pausing to ask, and document them here:

1. **OpenEdition consolidation (Task 7).** The spec's seed list named "Presses universitaires de Rennes" and "Africae" as if they were independent Crossref members, one per publisher (mirroring how every other seed entry maps 1:1 to a publisher). Live lookup (`/prefixes/10.4000`) found both are actually imprints hosted under a single Crossref member, "OpenEdition" (id 2399) -- there is no separate member ID for either imprint. Rather than fabricate two member IDs that don't exist, or drop language diversity from French-language sources entirely, I consolidated both into one `OpenEdition` seed entry. **Trade-off:** this seed will also surface OpenEdition-hosted books from imprints/domains beyond just those two (OpenEdition hosts many publishers) -- a broader net than the spec implied, but still within "OA publisher already represented in the manifest," and every candidate still goes through the same license/URL/confirmation/novelty gates regardless of which OpenEdition imprint it came from.

2. **`domain` field left `null` for auto-discovered candidates (Task 7).** The existing manifest schema's `domain` field (`"linguistics"`, `"history"`, etc.) is a human subject-classification judgment call with no reliable Crossref equivalent for books. Rather than build subject-classification logic (out of scope, not requested) or guess from `container-title`/publisher (unreliable), auto-discovered entries get `domain: null`. **Trade-off:** the README's existing per-domain coverage table (`evaluation/crossref_gt/README.md`'s "Coverage" section) won't automatically account for auto-discovered books' subject areas; a future pass could backfill this by hand if the domain-coverage table needs to stay accurate.

3. **Crossref `select` field list excludes `language` (Task 7).** Originally planned to request `select=...,language` to read each candidate's language directly from Crossref. A live test query confirmed Crossref's `/works` route rejects `language` as an invalid select field entirely (a `select-not-available` validation failure that would break every discovery request). A further live check of a full (unselected) work record confirmed Crossref book/monograph records essentially never carry a `language` field regardless. The design's fallback path -- each seed publisher's `default_language` -- ends up being the primary source of language info in practice, not just a fallback. **Trade-off:** a genuinely multilingual publisher (e.g. OpenEdition, which hosts non-French OA books too) will have every one of its candidates tagged with the seed's single `default_language` ("fr" for OpenEdition) regardless of the book's actual language, since there's no per-work signal to override it. Accurate per-book language detection would require either a language-detection library (a new dependency, not justified for this round) or reading the PDF/title text -- both explicitly deferred as unnecessary complexity for a first automated pass; a human reviewing a `pending/` book before promotion is the natural point to catch and correct a wrong `language` value.

4. **Crossref `link` content-type acceptance widened to include `"unspecified"` (Task 7).** A live query against Open Book Publishers confirmed their registered PDF links use `content-type: "unspecified"`, not `"application/pdf"` -- a strict MIME-type-only check would silently treat every OBP candidate as having no Crossref-provided PDF URL, always falling through to Unpaywall/OpenAlex unnecessarily. Accepting `"unspecified"` risks occasionally picking a non-PDF "unspecified" link for some other publisher, but `fetch_crossref_gt_corpus.py`'s existing magic-byte check (`content.startswith(b"%PDF-")`) already catches that downstream as a normal "failed" download, so the risk is bounded and non-silent.

5. **Novelty threshold computed via `numpy.percentile` rather than a hand-rolled percentile function.** `numpy` is already an indirect hard dependency (via `scikit-learn`'s `StandardScaler`, already imported in both `evaluate_layout_toc_classifier.py` and now `build_crossref_gt_ground_truth.py`), so using it directly for percentile computation avoids reinventing linear interpolation over a sorted list for no benefit.

6. **`NoveltyBaseline` computed once per `build_crossref_gt_ground_truth.py` run, not once per candidate book.** Recomputing the open-access corpus's scaler and chapter_first vectors for every candidate in the manifest would be correct but wasteful (each recomputation re-runs `build_feature_table` over every existing `open-access/` book, even though `ensure_alto_xml`'s cache makes repeat calls cheap after the first). Computed once in `main()` and threaded through `process_book()` instead.

7. **No defensive validation added to `_novelty_threshold`/`_is_novel` for empty/too-small inputs (Task 4 code-quality review).** Flagged as a theoretical gap during review, but the real `open-access/` corpus always has far more than the minimum 2 chapter-first vectors these functions need; adding guard code for a precondition that can't be violated with real data was declined as unnecessary defensiveness.

8. **`NoveltyBaseline.pdfalto_bin` stored on the dataclass rather than passed as a separate `process_book` parameter (Task 6 code-quality review).** Keeps `process_book`'s signature from growing an extra argument purely to thread through a value that's already logically part of "how to evaluate novelty."

9. **Outline-diagnostic disagreements print directly from `process_book`, not returned to and printed by `main()` alongside the book's status line (Task 6 code-quality review).** Matches the plan's original design; the ordering (diagnostic lines before the `OK`/`SKIP` summary line) is cosmetic only, not fixed.

10. **`_load_novelty_baseline`'s local `cache_dir_for` closure duplicates a near-identical one already in `evaluate_layout_toc_classifier.py`'s `main()` (Task 6 code-quality review).** Accepted as a small, low-value-to-fix duplication rather than introducing a shared-helper module for one three-line closure.

11. **`_language_priority`'s discovery ranking is inert beyond each language's independent `--max-per-language` cap (Task 7 code-quality review).** The ranking determines *which* languages get printed/considered first within a run, but every language still fills up to the same per-language cap regardless of rank -- there's no global weighted budget across languages. This is a genuine gap in the approved spec itself (the spec never called for a cross-language budget), not an implementer bug, so it's documented here rather than silently expanded in scope to fix.

12. **429-retry-loop and license-fallback duplication between `discover_crossref_candidates.py` and `fetch_crossref_gt_corpus.py` left unrefactored.** The design spec explicitly designates `fetch_crossref_gt_corpus.py` "unchanged" for this feature; extracting a shared helper would have required touching it.

13. **`evaluation/crossref_gt/README.md`'s "Coverage" table left describing only the 43 manually-curated books, not refreshed to compute domain/language breakdowns for the 15 auto-discovered ones.** Auto-discovered entries have no `domain` value (see item 2), so they can't be folded into that table's per-domain breakdown without the same subject-classification work already declined there; a one-line note was added pointing this out instead of restating the whole table.

14. **Correction to the original real-run report: pdfalto was in fact available, just not on `PATH`.** The initial live end-to-end run (Task 8) hit `FileNotFoundError: pdfalto` and was reported as blocked by a missing binary after `which pdfalto`, `$PDFALTO_BIN`, and `brew list --formula` all came back empty -- a real environment gap, correctly diagnosed at the time. What that check missed: an earlier session (before this plan existed, during the original layout-classifier pilot's own viability investigation) had already built `pdfalto` from source into `/Users/cboulanger/Code/pdfalto/pdfalto`, a path outside `PATH` and never exported as `$PDFALTO_BIN`. Re-running `build_crossref_gt_ground_truth.py --pdfalto-bin /Users/cboulanger/Code/pdfalto/pdfalto` completed the full reconciliation pass: all 43 pre-existing books correctly `SKIP`ped, and of the 6 newly-fetched Task 8 candidates, 4 confirmed and cleared the novelty gate, landing in `evaluation/corpus/pending/` (see `evaluation/crossref_gt/README.md`'s "Automated discovery" section for the per-book breakdown). This is not a design decision so much as a correction: the novelty-gate code path is now verified end-to-end against real data, not just up to the point of the missing binary.
