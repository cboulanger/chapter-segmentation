# Layout-Classifier Feature Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 4 layout-classifier features that are unnormalized absolute ALTO coordinates (root-caused as the pilot's dominant NOT-MET driver), then re-run the pilot and record the result.

**Architecture:** One code change (divide `width_mean`/`width_var`/`left_margin_mean`/`left_margin_var` by page width, mirroring the existing page-height normalization already applied to `first_text_vpos_fraction`/`line_density`), one test-fixture update to match, then a real re-run of the existing pilot script against the already-cached 50-book corpus with a `RESULTS.md` write-up.

**Tech Stack:** Python, `unittest`, existing `evaluation/scripts/layout_features.py` and `evaluation/scripts/evaluate_layout_toc_classifier.py` (both unchanged in second task).

---

### Task 1: Normalize width/left-margin features by page width

**Files:**
- Modify: `evaluation/scripts/layout_features.py:90-133` (`extract_page_features`)
- Test: `tests/test_layout_features.py:72-95` (`TestExtractPageFeatures`)

- [ ] **Step 1: Update the two hand-computed test cases to expect width-normalized values**

In `tests/test_layout_features.py`, replace the body of `test_toc_like_page_features`:

```python
    def test_toc_like_page_features(self):
        f = self.features[0]
        self.assertEqual(f["line_count"], 3.0)
        self.assertAlmostEqual(f["width_mean"], 120.0 / 500)
        self.assertAlmostEqual(f["width_var"], 400.0 / 500**2)
        self.assertAlmostEqual(f["left_margin_mean"], 50.0 / 500)
        self.assertAlmostEqual(f["left_margin_var"], 0.0 / 500**2)
        self.assertEqual(f["trailing_number_fraction"], 1.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 1.0)
        self.assertEqual(f["top_block_is_large_font"], 0.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 100 / 600)
        self.assertAlmostEqual(f["line_density"], 3 / 600)
```

and the body of `test_chapter_opening_page_features`:

```python
    def test_chapter_opening_page_features(self):
        f = self.features[1]
        self.assertEqual(f["line_count"], 3.0)
        self.assertAlmostEqual(f["width_mean"], (830 / 3) / 500)
        self.assertAlmostEqual(f["width_var"], 12033.333333333334 / 500**2, places=6)
        self.assertAlmostEqual(f["left_margin_mean"], (296 / 3) / 500)
        self.assertAlmostEqual(f["left_margin_var"], 7701.333333333333 / 500**2, places=6)
        self.assertEqual(f["trailing_number_fraction"], 0.0)
        self.assertAlmostEqual(f["font_size_max_ratio"], 2.4)
        self.assertEqual(f["top_block_is_large_font"], 1.0)
        self.assertAlmostEqual(f["first_text_vpos_fraction"], 50 / 600)
```

Both fixture pages in `_FIXTURE_ALTO_XML` have `WIDTH="500"` (the fixture's page width) -- these expected values divide the *existing* raw-point expected values (still correct as intermediate values -- the ALTO data itself hasn't changed) by that page width (linearly for the means, by its square for the variances, since `Var(x/k) = Var(x)/k**2`). `places=6` on the two variance assertions matches the original test's already-loose `places=2` tolerance, scaled down since the expected magnitude shrank from ~12000/~7700 to ~0.048/~0.031.

- [ ] **Step 2: Run the tests to verify they now fail**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: `test_toc_like_page_features` and `test_chapter_opening_page_features` FAIL (actual values still in raw points, e.g. `120.0` vs expected `0.24`); `TestTopBlockLargeFontIndexAlignment` and `TestTrailingNumberFraction` still PASS (they don't touch these four features).

- [ ] **Step 3: Normalize the four features by page width in the implementation**

In `evaluation/scripts/layout_features.py`, inside `extract_page_features` (starting at line 90), add a `page_width` read next to the existing `page_height` read, and divide the two raw-coordinate lists by it:

```python
    for page in root.iter(_ALTO_NS + "Page"):
        page_index = int(page.get("PHYSICAL_IMG_NR")) - 1
        page_height = float(page.get("HEIGHT"))
        page_width = float(page.get("WIDTH"))
        lines = list(page.iter(_ALTO_NS + "TextLine"))

        if not lines:
            features[page_index] = {name: 0.0 for name in FEATURE_NAMES}
            continue

        widths = [float(line.get("WIDTH")) / page_width for line in lines]
        left_margins = [float(line.get("HPOS")) / page_width for line in lines]
        vpositions = [float(line.get("VPOS")) for line in lines]
```

The rest of the function (the `trailing_hits` loop, `_font_ratio_and_top_block_flag` call, and the `features[page_index] = {...}` dict) is unchanged -- `width_mean`/`width_var`/`left_margin_mean`/`left_margin_var` are still computed with `statistics.mean`/`statistics.variance` over `widths`/`left_margins`, which now hold width-fraction values instead of raw points.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_layout_features.py -v
```

Expected: all tests PASS (5 tests: `test_returns_zero_based_page_indices`, `test_toc_like_page_features`, `test_chapter_opening_page_features`, `TestTopBlockLargeFontIndexAlignment`'s test, `TestTrailingNumberFraction`'s test).

- [ ] **Step 5: Run the full test suite to check for unrelated regressions**

```bash
uv run pytest -q
```

Expected: same pass/skip counts as before this change (this feature module has no other consumers whose own tests hardcode raw-point width/margin values -- confirm by reading test output, not by assumption).

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/layout_features.py tests/test_layout_features.py
git commit -m "$(cat <<'EOF'
fix: normalize layout-classifier width/left-margin features by page width

4 of 10 features were raw ALTO point coordinates, never normalized --
unlike first_text_vpos_fraction/line_density, which already divide by
page height. Root-caused as the pilot's dominant NOT-MET driver: page
width varies far more across copyrighted-scans (304-991pt) than
open-access (420-595pt), so a leave-one-book-out-trained classifier's
absolute-coordinate thresholds didn't transfer to differently-sized
held-out books.
EOF
)"
```

---

### Task 2: Re-run the pilot and record the result

**Files:**
- Modify: `evaluation/RESULTS.md` (new section)
- No code changes -- `evaluation/scripts/evaluate_layout_toc_classifier.py` already re-derives features from the cached ALTO XML on every run, so Task 1's fix is exercised automatically with no script changes.

- [ ] **Step 1: Re-run the pilot against the real, already-cached corpus**

The cached ALTO XML from the original pilot run already exists at
`evaluation/corpus/open-access/.layout-cache/` and
`evaluation/corpus/copyrighted-scans/.layout-cache/` -- this run reads
those files as-is (Task 1 changed Python-side feature computation only,
not what `pdfalto` produces), so no `--pdfalto-bin` flag or cache rebuild
is needed unless the cache is somehow missing:

```bash
uv run python evaluation/scripts/evaluate_layout_toc_classifier.py \
  2>&1 | tee /tmp/layout_classifier_normalized_output.txt
```

Expected: a per-book line for every book with a usable `"toc"` field
(same 50 books as the original pilot run), then the aggregate
`full_recall_fraction`, `avg_candidate_fraction`, and a final
`Decision bar (>=90% full recall, <=15% avg candidate fraction): MET` or
`NOT MET` line. If the cache is missing for some reason, this will fall
back to running `pdfalto` (needs `--pdfalto-bin /path/to/pdfalto` or a
`PDFALTO_BIN` env var pointing at the locally-built binary, e.g.
`/Users/cboulanger/Code/pdfalto/pdfalto`) -- expect this run to hit the
cache and finish quickly either way.

- [ ] **Step 2: Compute the open-access vs. copyrighted-scans recall breakdown**

The script's stdout only reports per-book, not per-corpus, aggregates.
Compute the split needed for the write-up (matches the diagnostic
approach already used to root-cause the original NOT MET result) with a
small ad hoc script -- write it to a scratch path (e.g.
`/tmp/corpus_split.py`), not committed to the repo:

```python
import sys
from pathlib import Path

sys.path.insert(0, "/Users/cboulanger/Code/chapter-segmentation/.claude/worktrees/pdf-layout-toc-classifier")

from evaluation.scripts.evaluate_layout_toc_classifier import (
    build_feature_table,
    load_book_corpus,
    _evaluate_label,
    FEATURE_NAMES,
)
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST
from evaluation.scripts.pdfalto_runner import resolve_pdfalto_binary

_CORPUS_DIR = Path("/Users/cboulanger/Code/chapter-segmentation/.claude/worktrees/pdf-layout-toc-classifier/evaluation/corpus")


def cache_dir_for(corpus: str) -> Path:
    return _CORPUS_DIR / corpus / ".layout-cache"


def main() -> int:
    pdfalto_bin = resolve_pdfalto_binary(None)
    books = load_book_corpus()
    books_by_key = {book["key"]: book for book in books}
    rows = build_feature_table(books, cache_dir_for, pdfalto_bin)
    book_keys = sorted({row["book_key"] for row in rows})

    by_corpus: dict[str, list[float]] = {"open-access": [], "copyrighted-scans": []}

    for held_out in book_keys:
        train_rows = [r for r in rows if r["book_key"] != held_out]
        test_rows = [r for r in rows if r["book_key"] == held_out]
        ground_truth_labels = books_by_key[held_out]["labels"]
        corpus = books_by_key[held_out]["corpus"]

        X_train = [[r["features"][name] for name in FEATURE_NAMES] for r in train_rows]
        X_test = [[r["features"][name] for name in FEATURE_NAMES] for r in test_rows]

        ground_truth_count = ground_truth_labels.count(LABEL_CHAPTER_FIRST)
        recall, _passed, _pred = _evaluate_label(
            LABEL_CHAPTER_FIRST, train_rows, test_rows, X_train, X_test, ground_truth_count, held_out
        )
        if recall is not None:
            by_corpus[corpus].append(recall)

    for corpus, recalls in by_corpus.items():
        n = len(recalls)
        avg = sum(recalls) / n if n else float("nan")
        n_zero = sum(1 for r in recalls if r == 0.0)
        n_full = sum(1 for r in recalls if r == 1.0)
        print(f"{corpus}: n={n}, avg_recall={avg:.0%}, n_zero={n_zero}, n_full={n_full}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
uv run python /tmp/corpus_split.py
```

Expected: two lines, one per corpus, each with `n`, `avg_recall`, `n_zero`
(books stuck at 0% recall), `n_full` (books at 100% recall) -- the same
shape of output used to originally diagnose the problem (`open-access:
n=37, avg_recall=83%, n_zero=0, n_full=9` /
`copyrighted-scans: n=13, avg_recall=20%, n_zero=8, n_full=1` before this
fix).

- [ ] **Step 3: Write the RESULTS.md section**

Add a new section to `evaluation/RESULTS.md`, after the existing
"## Per-strategy standalone results (heuristic / outline / LLM)" section
(before "## LLM-fallback results (archived -- script removed)"), titled
`## Layout-based TOC/chapter-first-page classifier pilot`. Write it in
this file's existing prose-plus-tables style (see e.g. "Diverse
real-library evaluation set" above it for the tone: concrete numbers,
root causes traced by hand, explicit about what's still unresolved).
Cover, using the actual numbers from Steps 1-2 above (not the
placeholder numbers below, which are the original pre-fix pilot's
numbers for reference only):

- What this pilot measures and why (one or two sentences, linking
  `docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`),
  and the original run's result: NOT MET, 16% full recall (bar: ≥90%),
  5.3% average candidate fraction (bar: ≤15%, cleared).
- The root cause found afterward and fixed by
  `docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md`:
  4 of 10 features were unnormalized absolute ALTO coordinates, and this
  tracked almost exactly with the open-access/copyrighted-scans corpus
  split (the original `n_zero=8` cluster, all `copyrighted-scans`).
- The fresh numbers from Step 1 (full_recall_fraction, avg_candidate_fraction,
  MET/NOT MET) and the fresh per-corpus breakdown from Step 2, stated
  plainly against the original numbers so the delta is legible.
- If still NOT MET: name the bar-strictness finding from the
  normalization spec's Problem section (100%-recall-per-book requirement;
  even an 80% tolerance only reached 34% full-recall-fraction on the
  original run) as the next-most-likely place to look, without proposing
  or scoping a fix for it here -- this follow-up's own scope is the
  feature-normalization fix only, per
  `docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md`'s
  "Out of scope" section.

- [ ] **Step 4: Commit**

```bash
git add evaluation/RESULTS.md
git commit -m "$(cat <<'EOF'
docs: record layout-classifier pilot result after feature normalization

Reports the fresh leave-one-book-out numbers and open-access vs.
copyrighted-scans breakdown following the width-normalization fix, per
docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md.
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage**: the normalization design spec's three scope items --
  (1) normalize the four features (Task 1, Steps 1-4), (2) update the
  hand-computed test fixtures (Task 1, Steps 1-2, done *before* the
  implementation per TDD), (3) re-run the pilot and write up
  `RESULTS.md` (Task 2) -- each has a corresponding task/step. The
  spec's "Decision criteria" section (no new pass/fail bar beyond
  reporting) is satisfied by Task 2 Step 3 writing up the result either
  way, MET or NOT MET.
- **Placeholder scan**: no TBD/TODO markers. Task 2 Step 3's write-up
  content is described precisely (what to cover, in what order, sourced
  from which prior step's real output) rather than left as "write up the
  results" -- the actual prose is necessarily written from Steps 1-2's
  real numbers, which don't exist until this plan is executed, so it's
  specified as a structured content list rather than verbatim text, the
  same way Task 12 Step 2 of the original pilot plan handled its own
  "report the result" step.
- **Type consistency**: `extract_page_features`, `FEATURE_NAMES`,
  `build_feature_table`, `load_book_corpus`, `_evaluate_label`,
  `LABEL_CHAPTER_FIRST`, `resolve_pdfalto_binary` are named identically
  to their existing definitions in `evaluation/scripts/layout_features.py`
  and `evaluation/scripts/evaluate_layout_toc_classifier.py` (unchanged
  by this plan, confirmed by reading both files directly) -- Task 2's
  diagnostic script only imports and calls existing functions, no new
  symbols introduced.
- **Out-of-scope guard**: neither task touches the decision bar
  (`evaluate_leave_one_book_out`'s `passed = recall == 1.0 if label ==
  LABEL_CHAPTER_FIRST else bool(hit_indices)` logic), the model, other
  features, `.layout-cache/` invalidation, or production wiring -- matching
  the normalization spec's "Out of scope" section exactly.
