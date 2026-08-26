# Retire TOC Classifier Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the layout-based TOC/chapter-first-page classifier pilot (`evaluation/scripts/evaluate_layout_toc_classifier.py` and friends) from this repo — it never cleared its own decision bar across eleven follow-ups and has been superseded by a standalone repo, `toc-page-classifier` (github.com/cboulanger/toc-page-classifier), which now owns this problem with a different methodology. Move the small subset of this code that `evaluation/scripts/build_crossref_gt_ground_truth.py` still actively depends on (its layout-feature "novelty check" gate) into `src/chapter_segmentation/` as real library code, so that dependency survives the pilot's removal. Delete everything else — pilot script, tests, specs/plans, scan-noise-augmentation tooling — and update every doc that references it.

**Architecture:** `pdfalto_runner.py`, `layout_features.py`, and `layout_labels.py` are generic, reusable primitives (run pdfalto, extract per-page ALTO-XML geometry features, derive per-page ground-truth labels) with no evaluation-directory coupling — they move as-is into `src/chapter_segmentation/`. `load_book_corpus`/`build_feature_table` (which scan `evaluation/corpus/`) are evaluation-specific glue built on top of those primitives — they get a new, slimmed-down home at `evaluation/scripts/layout_corpus.py` (stripped of the pilot-only scan-noise-augmentation option, which nothing outside the pilot uses). `build_crossref_gt_ground_truth.py`'s imports get repointed at the new locations. Everything else pilot-specific — the LOBO training/evaluation harness, scan-noise augmentation, the DNB-scan-noise calibration script, their tests, and their design docs — gets deleted outright. Finally, every markdown doc that references the pilot gets updated, following the existing precedent this repo already used when it extracted `dnb-toc-ground-truth` into its own repo (see commit `93e90c1`): a short pointer in root `README.md`'s "Ongoing experiments" section, and prose fixes everywhere else that named the pilot as still-present.

**Tech Stack:** Python, unittest (this repo's test convention), git.

**Cross-repo note:** The pilot's most useful history has already been condensed and copied to `toc-page-classifier/docs/history/chapter-segmentation-layout-classifier-pilot.md` (done directly, not part of this plan — already committed there). This plan only needs to delete the original `evaluation/experiments/toc-classifier-pilot.md` from this repo, not migrate its content.

---

### Task 1: Move generic layout primitives into `src/chapter_segmentation/`

**Files:**
- Move: `evaluation/scripts/pdfalto_runner.py` → `src/chapter_segmentation/pdfalto_runner.py`
- Move: `evaluation/scripts/layout_features.py` → `src/chapter_segmentation/layout_features.py`
- Move: `evaluation/scripts/layout_labels.py` → `src/chapter_segmentation/layout_labels.py`
- Modify: `tests/test_pdfalto_runner.py`, `tests/test_layout_features.py`, `tests/test_layout_labels.py` (import paths only — they stay in `tests/`, which already holds tests for both `src/chapter_segmentation/` and `evaluation/scripts/` code side by side, so no test-file relocation is needed)

**Context for the implementer:** These three files are currently pilot-scoped in name and docstring only — their actual code has no dependency on the pilot's training/evaluation logic, the pilot's decision bar, or the evaluation corpus directory layout. `evaluation/scripts/build_crossref_gt_ground_truth.py` (NOT part of this task — Task 3 handles it) already imports real library code from `chapter_segmentation.segmentation` (`_parse_toc_page_number`), so importing from `chapter_segmentation.pdfalto_runner` etc. is consistent with how that file already treats `src/chapter_segmentation` as its library dependency.

- [ ] **Step 1: Move the three files**

```bash
git mv evaluation/scripts/pdfalto_runner.py src/chapter_segmentation/pdfalto_runner.py
git mv evaluation/scripts/layout_features.py src/chapter_segmentation/layout_features.py
git mv evaluation/scripts/layout_labels.py src/chapter_segmentation/layout_labels.py
```

- [ ] **Step 2: Update each moved file's module docstring**

These docstrings currently describe the code as pilot-specific (e.g. "for the layout-classifier pilot", "the classifier's input for docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md"). That pilot and its design doc are being deleted later in this plan (Task 5), so any reference to them must go. Replace each docstring to describe the module on its own terms, as general-purpose library code, and note its real current consumer:

`src/chapter_segmentation/pdfalto_runner.py` — change the docstring's first sentence from referencing the pilot's design spec to something like: "Runs the pdfalto binary (https://github.com/kermitt2/pdfalto) against a PDF and caches its ALTO XML output -- not vendored, matching the Kreuzberg OCR sidecar precedent of treating an external tool as developer-provided, not bundled. Used by evaluation/scripts/build_crossref_gt_ground_truth.py's layout-feature novelty check." Keep the rest of the file's code and its two functions (`resolve_pdfalto_binary`, `ensure_alto_xml`) and `alto_xml_path` unchanged.

`src/chapter_segmentation/layout_features.py` — change the docstring's first sentence similarly: "Per-page geometric feature extraction from a pdfalto ALTO XML file. Used by evaluation/scripts/build_crossref_gt_ground_truth.py's layout-feature novelty check (formerly also the input to a since-retired layout-only TOC/chapter-page classifier pilot -- see toc-page-classifier's docs/history/ for that pilot's findings, if relevant)." Keep everything else (all functions, `FEATURE_NAMES`, `PAGE_FEATURE_NAMES`, `CONTEXT_FEATURE_NAMES`, the heading-detection regexes) unchanged.

`src/chapter_segmentation/layout_labels.py` — change the docstring similarly: "Per-page ground-truth labels (toc / chapter_first / other) derived from an .expected.json-shaped dict. Used by evaluation/scripts/layout_corpus.py's build_feature_table (in turn used by build_crossref_gt_ground_truth.py's novelty check)." Keep `LABEL_TOC`, `LABEL_CHAPTER_FIRST`, `LABEL_OTHER`, and `page_labels` unchanged.

- [ ] **Step 3: Update the three test files' imports**

In `tests/test_pdfalto_runner.py`, change:
```python
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary
```
to:
```python
from chapter_segmentation.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary
```

In `tests/test_layout_labels.py`, change:
```python
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST, LABEL_OTHER, LABEL_TOC, page_labels
```
to:
```python
from chapter_segmentation.layout_labels import LABEL_CHAPTER_FIRST, LABEL_OTHER, LABEL_TOC, page_labels
```

In `tests/test_layout_features.py`, find the `from evaluation.scripts.layout_features import (...)` block and change `evaluation.scripts.layout_features` to `chapter_segmentation.layout_features`, keeping the imported names identical.

Check `tests/test_layout_features.py` and `tests/test_pdfalto_runner.py` for any other `evaluation.scripts.` references (e.g. `patch("evaluation.scripts.layout_features....")` or similar) and update those too — grep for `evaluation.scripts.layout_features`, `evaluation.scripts.layout_labels`, `evaluation.scripts.pdfalto_runner` across both files to be sure you caught every occurrence, not just the top-level import line.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, same total count as before this task (no tests added or removed, only relocated — check the count before starting so you can confirm it's unchanged).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move pdfalto/layout-feature/layout-label primitives into src/chapter_segmentation"
```

---

### Task 2: Carve out `load_book_corpus`/`build_feature_table` into `evaluation/scripts/layout_corpus.py`

**Files:**
- Create: `evaluation/scripts/layout_corpus.py`
- Create: `tests/test_layout_corpus.py`
- Modify: `tests/test_evaluate_layout_toc_classifier.py` (remove the two test classes being extracted — the rest of this file is deleted wholesale in Task 5, but do this extraction cleanly now so nothing is lost if Task 5 runs later)

**Context for the implementer:** `evaluation/scripts/evaluate_layout_toc_classifier.py` currently contains `load_book_corpus` and `build_feature_table` (reads `evaluation/corpus/<corpus>/*.expected.json` + PDFs, builds a per-page feature+label table) alongside the rest of the pilot's LOBO training/evaluation/reporting logic, which is being deleted (Task 5). `build_crossref_gt_ground_truth.py`'s novelty check needs exactly these two functions to keep working — Task 3 will repoint its imports at this new module. `build_feature_table` currently has an `augment: bool = False` parameter that pulls in `evaluation/scripts/alto_scan_noise.py` (pilot-only scan-noise augmentation, used only by the pilot's own training runs — `build_crossref_gt_ground_truth.py` never passes `augment=True`). Drop that parameter and its scan-noise code path entirely in the new module — don't carry pilot-only functionality into library-adjacent code that doesn't need it.

Current source (`evaluation/scripts/evaluate_layout_toc_classifier.py`, lines 129-202 as of this plan being written — re-locate by function name if line numbers have drifted):

```python
_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
_CORPORA = ["open-access", "copyrighted-scans"]


def load_book_corpus(corpora: list[str] | None = None) -> list[dict]:
    """Returns one entry per book with a usable "toc" field: {"key",
    "corpus", "pdf_path", "labels"} -- books whose .expected.json has no
    "toc" key at all are excluded entirely (not yet retrofitted, or
    flagged for manual review), per the design spec. `corpora` defaults
    to every corpus this pilot normally scores against (_CORPORA);
    pass an explicit subset (e.g. ["open-access"]) to restrict it -- to
    evaluate evaluation/corpus/pending/ candidates before promotion."""
    books = []
    for corpus in corpora if corpora is not None else _CORPORA:
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
    happen, but has been observed for malformed/oversized scanned pages)
    are skipped rather than crashing the whole run -- skip counts are
    tallied per label and, since a dropped "toc" or "chapter_first" row
    silently erodes what a consumer of this feature table is measuring
    against, printed as a warning once the whole table has been built."""
    rows = []
    skipped_by_label: dict[str, int] = {}
    for book in books:
        cache_dir = cache_dir_for(book["corpus"])
        alto_path = ensure_alto_xml(book["pdf_path"], cache_dir, pdfalto_bin)
        page_features = add_book_context_features(
            extract_page_features(str(alto_path)), len(book["labels"])
        )
        for page_index, label in enumerate(book["labels"]):
            features = page_features.get(page_index)
            if features is None:
                skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                continue
            rows.append({"book_key": book["key"], "features": features, "label": label})
    if skipped_by_label:
        total_skipped = sum(skipped_by_label.values())
        breakdown = ", ".join(f"{label}={count}" for label, count in sorted(skipped_by_label.items()))
        print(
            f"WARNING: build_feature_table skipped {total_skipped} page(s) with no "
            f"pdfalto feature vector ({breakdown})",
            file=sys.stderr,
        )
    return rows
```

(The `augment`/`sources`/`is_augmented`/`aug_path` logic and the `write_augmented_alto` import from the original are already removed in the version above — this is the stripped-down target state, not a literal copy of the current file. Diff it against the current file yourself to confirm you're removing exactly the augmentation path and nothing else.)

- [ ] **Step 1: Create `evaluation/scripts/layout_corpus.py`**

```python
"""Builds a per-page layout-feature table from the evaluation corpus --
evaluation-specific glue on top of chapter_segmentation's pdfalto/
layout_features/layout_labels primitives. Used by
evaluation/scripts/build_crossref_gt_ground_truth.py's layout-feature
novelty check."""

import json
import sys
from pathlib import Path

from pypdf import PdfReader

from chapter_segmentation.layout_features import add_book_context_features, extract_page_features
from chapter_segmentation.layout_labels import page_labels
from chapter_segmentation.pdfalto_runner import ensure_alto_xml

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
_CORPORA = ["open-access", "copyrighted-scans"]


def load_book_corpus(corpora: list[str] | None = None) -> list[dict]:
    """Returns one entry per book with a usable "toc" field: {"key",
    "corpus", "pdf_path", "labels"} -- books whose .expected.json has no
    "toc" key at all are excluded entirely (not yet retrofitted, or
    flagged for manual review). `corpora` defaults to open-access and
    copyrighted-scans (_CORPORA); pass an explicit subset (e.g.
    ["open-access"]) to restrict it."""
    books = []
    for corpus in corpora if corpora is not None else _CORPORA:
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
    happen, but has been observed for malformed/oversized scanned pages)
    are skipped rather than crashing the whole run -- skip counts are
    tallied per label and printed as a warning once the whole table has
    been built."""
    rows = []
    skipped_by_label: dict[str, int] = {}
    for book in books:
        cache_dir = cache_dir_for(book["corpus"])
        alto_path = ensure_alto_xml(book["pdf_path"], cache_dir, pdfalto_bin)
        page_features = add_book_context_features(
            extract_page_features(str(alto_path)), len(book["labels"])
        )
        for page_index, label in enumerate(book["labels"]):
            features = page_features.get(page_index)
            if features is None:
                skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                continue
            rows.append({"book_key": book["key"], "features": features, "label": label})
    if skipped_by_label:
        total_skipped = sum(skipped_by_label.values())
        breakdown = ", ".join(f"{label}={count}" for label, count in sorted(skipped_by_label.items()))
        print(
            f"WARNING: build_feature_table skipped {total_skipped} page(s) with no "
            f"pdfalto feature vector ({breakdown})",
            file=sys.stderr,
        )
    return rows
```

- [ ] **Step 2: Create `tests/test_layout_corpus.py`**

Port `TestBuildFeatureTable` and `TestLoadBookCorpus` from `tests/test_evaluate_layout_toc_classifier.py` (read that file yourself for their exact current bodies — 4 tests in `TestBuildFeatureTable`'s non-augmentation form plus `test_pipes_extracted_features_through_book_context_pass`, and 2 tests in `TestLoadBookCorpus`), with two changes to every test:

1. Every `patch("evaluation.scripts.evaluate_layout_toc_classifier.X", ...)` becomes `patch("evaluation.scripts.layout_corpus.X", ...)` — the patch target must match where the name is looked up (this new module's own namespace), not the old pilot module.
2. The top-level imports change from:
```python
from evaluation.scripts.evaluate_layout_toc_classifier import (
    build_feature_table,
    load_book_corpus,
    ...
)
```
to:
```python
from evaluation.scripts.layout_corpus import build_feature_table, load_book_corpus
```
(only import `build_feature_table` and `load_book_corpus` — this new test file does not need `evaluate_leave_one_book_out`, `main`, `select_threshold`, or `_write_classifier_results`, which stay in the file being deleted in Task 5.)

Do not port `TestBuildFeatureTableAugmentation` or the augmentation-specific parts of any test — the new `build_feature_table` has no `augment` parameter.

Give the new file a short module docstring, e.g.: `"""Unit tests for evaluation/scripts/layout_corpus.py's pure logic."""`

- [ ] **Step 3: Remove the ported test classes from `tests/test_evaluate_layout_toc_classifier.py`**

Delete the `TestBuildFeatureTable` and `TestLoadBookCorpus` classes (and the now-unused `build_feature_table`/`load_book_corpus` names from that file's top-level import, if removing them doesn't break anything else in the file that still needs them — check first) from `tests/test_evaluate_layout_toc_classifier.py`. This file is deleted wholesale in Task 5 anyway, but doing this now keeps the repo in a consistent, non-duplicated state at every commit rather than only at the end of the plan.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest`
Expected: PASS. Total test count should be unchanged overall (tests moved from one file to another, none added or removed net) — but `tests/test_layout_corpus.py` should show 7 passed on its own (`uv run pytest tests/test_layout_corpus.py -v`: 5 from `TestBuildFeatureTable`, 2 from `TestLoadBookCorpus`), and `tests/test_evaluate_layout_toc_classifier.py`'s own count should have dropped by exactly 7.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: carve load_book_corpus/build_feature_table into evaluation/scripts/layout_corpus.py"
```

---

### Task 3: Repoint `build_crossref_gt_ground_truth.py`'s imports

**Files:**
- Modify: `evaluation/scripts/build_crossref_gt_ground_truth.py`

**Context for the implementer:** This script's novelty-check gate (`NoveltyBaseline`, `_load_novelty_baseline`, `_candidate_is_novel`) is the one real, active consumer of the classifier-pilot code that must keep working after the pilot itself is deleted. Tasks 1-2 already moved everything it needs to new locations; this task just repoints four import lines. No other code in this file changes.

- [ ] **Step 1: Update the imports**

Find these four lines (currently around lines 73-77, re-locate by content if line numbers have drifted):
```python
from evaluation.scripts.evaluate_layout_toc_classifier import build_feature_table, load_book_corpus
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
from evaluation.scripts.layout_features import FEATURE_NAMES, extract_page_features
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary
```
Replace with (keeping `ground_truth_helper`'s import line unchanged — it isn't moving):
```python
from chapter_segmentation.layout_features import FEATURE_NAMES, extract_page_features
from chapter_segmentation.layout_labels import LABEL_CHAPTER_FIRST
from chapter_segmentation.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
from evaluation.scripts.layout_corpus import build_feature_table, load_book_corpus
```
(Alphabetized by module path, matching this file's existing import-ordering convention — check the surrounding import block for its actual convention and match it rather than assuming.)

Do not change anything else in this file — `NoveltyBaseline`, `_load_novelty_baseline`, `_candidate_is_novel`, and every other function are unaffected; they call the same names, which now resolve to the same underlying code from new locations.

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, including `tests/test_build_crossref_gt_ground_truth.py` (it imports from `evaluation.scripts.build_crossref_gt_ground_truth` at module level, so a broken import there would fail collection entirely — confirm this file's own tests still pass, not just that the suite as a whole didn't regress in count).

- [ ] **Step 3: Sanity-check the import resolves outside pytest too**

Run: `uv run python -c "from evaluation.scripts.build_crossref_gt_ground_truth import NoveltyBaseline; print('ok')"`
Expected: prints `ok` with no traceback.

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/build_crossref_gt_ground_truth.py
git commit -m "refactor: repoint build_crossref_gt_ground_truth.py at the moved layout-feature library code"
```

---

### Task 4: Delete the pilot's own source, tests, and design docs

**Files:**
- Delete: `evaluation/scripts/evaluate_layout_toc_classifier.py`
- Delete: `evaluation/scripts/alto_scan_noise.py`
- Delete: `evaluation/scripts/measure_dnb_scan_noise_stats.py`
- Delete: `tests/test_evaluate_layout_toc_classifier.py`
- Delete: `tests/test_alto_scan_noise.py`
- Delete: `tests/test_measure_dnb_scan_noise_stats.py`
- Delete: `docs/superpowers/plans/2026-08-10-layout-based-toc-classifier-pilot.md`
- Delete: `docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`
- Delete: `docs/superpowers/plans/2026-08-10-layout-toc-classifier-feature-normalization.md`
- Delete: `docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md`
- Delete: `docs/superpowers/plans/2026-08-12-layout-classifier-context-features-and-scan-augmentation.md`
- Delete: `docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md`
- Delete: `evaluation/experiments/toc-classifier-pilot.md`

**Context for the implementer:** Everything in this task is either the pilot's own training/evaluation/reporting code (nothing else in the repo imports it — Task 3 already repointed the one real external dependency away from it), or its planning/design documents, or its already-migrated history doc (a condensed version now lives in the standalone `toc-page-classifier` repo at `docs/history/chapter-segmentation-layout-classifier-pilot.md`, already committed there — this task only deletes the original, it does not need to migrate anything itself). By the time you run this task, `tests/test_evaluate_layout_toc_classifier.py` should already have had its two still-needed test classes removed (Task 2, Step 3) — confirm that happened before deleting the file, so you're not losing coverage.

- [ ] **Step 1: Delete the files**

```bash
git rm evaluation/scripts/evaluate_layout_toc_classifier.py
git rm evaluation/scripts/alto_scan_noise.py
git rm evaluation/scripts/measure_dnb_scan_noise_stats.py
git rm tests/test_evaluate_layout_toc_classifier.py
git rm tests/test_alto_scan_noise.py
git rm tests/test_measure_dnb_scan_noise_stats.py
git rm docs/superpowers/plans/2026-08-10-layout-based-toc-classifier-pilot.md
git rm docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md
git rm docs/superpowers/plans/2026-08-10-layout-toc-classifier-feature-normalization.md
git rm docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md
git rm docs/superpowers/plans/2026-08-12-layout-classifier-context-features-and-scan-augmentation.md
git rm docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md
git rm evaluation/experiments/toc-classifier-pilot.md
```

- [ ] **Step 2: Search for any remaining references to the deleted modules**

```bash
grep -rn "evaluate_layout_toc_classifier\|alto_scan_noise\|measure_dnb_scan_noise_stats" --include="*.py" .
```
Expected: no output. If anything shows up, it's a real remaining dependency this plan didn't anticipate — stop and report it rather than deleting the reference blindly (it may indicate a consumer this plan's discovery pass missed).

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`
Expected: PASS. Total count should have dropped by exactly the number of tests that were in the three deleted test files (check the count before this task to compute the expected drop, and confirm it matches — don't just check for a clean pass, confirm the arithmetic).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete the retired layout-based TOC classifier pilot's source, tests, and design docs

Superseded by the standalone toc-page-classifier repo
(github.com/cboulanger/toc-page-classifier), which now owns this
problem with a different methodology. Its still-active dependent,
build_crossref_gt_ground_truth.py's novelty check, was already
repointed at real library code in src/chapter_segmentation/ (prior
commits on this branch). A condensed history of this pilot's findings
is preserved at toc-page-classifier's docs/history/."
```

---

### Task 5: Update every doc that references the pilot

**Files:**
- Modify: `README.md` (root)
- Modify: `evaluation/CLAUDE.md`
- Modify: `evaluation/RESULTS.md`
- Modify: `evaluation/EXPERIMENTS.md`
- Modify: `evaluation/experiments/README.md`
- Modify: `evaluation/README.md`
- Modify: `evaluation/scripts/README.md`
- Modify: `evaluation/crossref_gt/README.md`

**Context for the implementer:** This repo already retired a sibling experiment once before — `dnb-toc-ground-truth` was extracted into its own repo in commit `93e90c1`. Read that commit (`git show 93e90c1`) before starting this task; it's the precedent to follow for tone and structure (a short "Ongoing experiments" pointer bullet in root `README.md`, and targeted prose fixes elsewhere — not a wholesale rewrite of any file). One difference from that precedent: `dnb-toc-ground-truth` kept a *live* redirect (`DNB_TOC_CORPUS_DIR`) because other pilots still consumed its corpus at runtime — this classifier pilot has no equivalent live consumer to preserve (its one real dependent, `build_crossref_gt_ground_truth.py`, was already repointed at library code in Tasks 1-3, not at an external corpus), so every reference here should be a clean removal/rewrite, not a redirect.

For each file below, here is exactly what needs to change (re-find the referenced text by content — it may have shifted lines since this plan was written):

**`README.md`** — the "Ongoing experiments" section (added by the `93e90c1` precedent) currently reads:
```
Candidate improvements that haven't (yet) been folded into the main
pipeline -- e.g. a layout-based TOC/chapter-page classifier, a small
locally-runnable TOC-extraction model, and the ground-truth-generation
pipeline for a new evaluation corpus -- are tracked as living documents in
`evaluation/experiments/`, see that directory's README for an index.
```
Remove "a layout-based TOC/chapter-page classifier, " from that sentence (the remaining two examples — the NuExtract pilot and dnb-toc-ground-truth — are enough to keep the sentence meaningful; don't add a replacement example just to keep the list length the same). Add one new sentence directly after it, mentioning where the classifier work now lives, matching the existing dnb-toc-ground-truth pointer's tone: something like "A layout/text-based TOC-page classifier, the direct successor to an earlier in-repo pilot that never cleared its own decision bar, now lives at [toc-page-classifier](https://github.com/cboulanger/toc-page-classifier)."

**`evaluation/CLAUDE.md`** — three separate spots:
1. The "Document organization" intro paragraph currently lists "the layout-based TOC/chapter-first-page classifier pilot" as an example of an `evaluation/experiments/` file, and its bullet list of example filenames includes `toc-classifier-pilot.md`. Remove both mentions (the paragraph and the bullet's example list both still read fine naming only `dnb-toc-only` and the NuExtract pilot as examples).
2. "Step 0a"'s paragraph starting "If the book is being added to grow the layout-based TOC/chapter-first-page classifier's training pool specifically..." (with its citation of `evaluation/experiments/toc-classifier-pilot.md`'s learning-curve finding) — this entire paragraph's advice no longer applies (there is no classifier training pool in this repo anymore). Delete the whole paragraph.
3. "Step 5: TOC ground truth" currently frames the `"toc"` field as existing "used by the layout-based TOC-classifier pilot" and links to that pilot's now-deleted design spec. Per this repo's decision to KEEP the `"toc"` field and `add_toc_ground_truth.py` as-is (it's still useful ground-truth metadata, and the new `toc-page-classifier` repo already consumes this exact field as one of its two ground-truth sources), reword this section's framing rather than deleting it: replace the "used by the layout-based TOC-classifier pilot" clause with something like "consumed by the standalone toc-page-classifier repo (github.com/cboulanger/toc-page-classifier) as one of its ground-truth sources" and remove the dead link to the deleted design spec. Do not remove or alter the rest of Step 5's instructions (running `add_toc_ground_truth.py`, the "NEEDS REVIEW" guidance, etc.) — that tooling and workflow are being kept per an explicit decision, only its stated *purpose* needs updating.

**`evaluation/RESULTS.md`** and **`evaluation/EXPERIMENTS.md`** — both have a scope-explanation paragraph listing `evaluation/experiments/dnb-toc-ground-truth.md`, `evaluation/experiments/toc-classifier-pilot.md`, and (implicitly) the NuExtract doc. Remove the `toc-classifier-pilot.md` reference from both (keep the other references; note there is a pre-existing, unrelated stale link to a `dnb-toc-ground-truth.md` file that may or may not still be accurate — that's out of scope for this task, leave it alone unless removing the classifier reference requires touching the same sentence, in which case fix only the classifier part and leave the rest of the sentence structurally as-is).

**`evaluation/experiments/README.md`** — currently has:
```
- [`toc-classifier-pilot.md`](toc-classifier-pilot.md) -- whether a
  classifier trained purely on page-layout geometry (no text content) can
  pre-filter table-of-contents and chapter-opening pages.
```
Replace this bullet, following the exact pattern the existing `dnb-toc-ground-truth` bullet already uses two entries below it in the same file (read that bullet for the pattern before writing this one) — a bullet with no link (since the file is gone) explaining it moved to a standalone repo and why:
```
- **layout-based TOC classifier pilot** -- a classifier trained purely on
  page-layout geometry (no text content) to pre-filter table-of-contents
  and chapter-opening pages never cleared its own decision bar after
  eleven follow-up investigations, and has been superseded by a
  standalone repo, `toc-page-classifier`
  (github.com/cboulanger/toc-page-classifier), using a different
  methodology (DNB-scan-sourced ground truth, text + layout features, not
  layout-only). A condensed history of this pilot's findings is preserved
  there at `docs/history/chapter-segmentation-layout-classifier-pilot.md`.
```
(Adjust wording to fit this file's own voice — the above is a starting point, not a mandate to copy verbatim; keep it in the same register as the neighboring `dnb-toc-ground-truth` bullet.)

**`evaluation/README.md`** — the `dnb-toc-only/` corpus description currently justifies that corpus's existence partly by "training and calibrating layout-only *parts* of the pipeline (currently the scan-noise constants in `evaluation/scripts/alto_scan_noise.py` -- see `evaluation/experiments/toc-classifier-pilot.md`'s 'real-scan measurement' follow-up)". `alto_scan_noise.py` is deleted (Task 4) and that follow-up's document is gone. **Before rewording this paragraph, check whether `dnb-toc-only/` has any other real consumer** — grep for `dnb-toc-only`, `dnb_toc_only`, or `toc_only` across `.py` files in this repo (excluding what Task 4 already deleted) to see whether anything else still reads this corpus. If nothing else uses it, say so plainly in the reworded paragraph rather than inventing a new justification — e.g. "used for training and calibrating layout-only parts of the pipeline; its only prior consumer (`alto_scan_noise.py`'s scan-noise constants) has been retired along with the classifier pilot it supported, and this corpus currently has no other consumer in this repo" — do not silently delete the corpus directory itself or its acquisition tooling (`fetch_dnb_toc_corpus.py`), that's out of scope for this plan; just report the finding accurately in the doc.

**`evaluation/scripts/README.md`** — remove the `## evaluate_layout_toc_classifier.py` section and the `## measure_dnb_scan_noise_stats.py` section entirely (both document scripts deleted in Task 4). Check whether `layout_features.py`, `layout_labels.py`, or `pdfalto_runner.py` have their own sections or mentions in this file (they're referenced as "library code" per the earlier discovery pass) — if so, update any path/location references to reflect their new home in `src/chapter_segmentation/` rather than `evaluation/scripts/`, or remove the section if this file's convention is to only document `evaluation/scripts/`-resident CLI entry points (check the file's own stated scope/intro to decide which). Add a short section (or a note in `build_crossref_gt_ground_truth.py`'s existing section, if the file is organized per-script) for the new `layout_corpus.py`, matching this file's existing per-script documentation style (a `--help`-dump reference — but `layout_corpus.py` has no CLI/`argparse`, it's pure library code with no `main()`, so don't fabricate a `--help` dump for it; a one-paragraph description of its two functions is enough, consistent with how this file would describe any non-CLI library module it documents, if it documents any — check for a precedent before deciding the exact format).

**`evaluation/crossref_gt/README.md`** — the section around "Since `build_crossref_gt_ground_truth.py` now also gates migration on a layout-feature **novelty check**..." and "The novelty check requires a working `pdfalto` binary (see `evaluation/scripts/pdfalto_runner.py`)..." references `pdfalto_runner.py`'s old location and (indirectly, via `evaluate_layout_toc_classifier.py --corpora open-access,pending`) the deleted pilot script as an example command for "evaluating it in isolation." Update the `pdfalto_runner.py` path reference to `src/chapter_segmentation/pdfalto_runner.py`. The "evaluating it in isolation via `evaluate_layout_toc_classifier.py --corpora open-access,pending`" parenthetical describes a workflow that no longer exists (that script is deleted) — remove that specific parenthetical clause rather than pointing it at a replacement, since there is no direct replacement for "evaluate a pending book against the classifier" in this repo anymore (the classifier now lives in `toc-page-classifier`, which has a different evaluation workflow entirely, not a drop-in `--corpora` flag). Keep the rest of that paragraph's guidance (promoting a `pending/` book into `open-access/` after manual review) unchanged.

- [ ] **Step 1: Make all the changes above**

Work through each file listed. For any instruction above that says "check X before writing Y," actually run the check (grep, read the neighboring precedent bullet, read the file's stated scope) rather than guessing — several of these instructions are deliberately open-ended because the exact right wording depends on what you find, not because the specifics don't matter.

- [ ] **Step 2: Search for any remaining stray references**

```bash
grep -rln "toc-classifier-pilot\|evaluate_layout_toc_classifier\|layout-based TOC.*classifier pilot" --include="*.md" .
```
Review every match. A reference inside `evaluation/RESULTS.md`/`EXPERIMENTS.md`'s "History"-style archival sections (if any exist describing *past* superseded work, not current state) may be legitimate to leave as an accurate historical record rather than something to scrub — use judgment per this repo's own documented convention (`evaluation/CLAUDE.md`'s "Document organization" section, if you kept the parts of it that still exist after your Task 5 edits, explains which files are meant to preserve history verbatim vs. describe current state only).

- [ ] **Step 3: Run the full test suite one final time**

Run: `uv run pytest`
Expected: PASS (doc-only changes in this task shouldn't affect test count, but confirm nothing else in the repo broke).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: remove retired classifier-pilot references, point to the standalone toc-page-classifier repo"
```
