# Layout-based TOC/chapter-first-page classifier: GT retrofit + viability pilot

Status: approved for planning
Date: 2026-08-10

## Problem

Every existing TOC-extraction path -- the heuristic, the cloud-LLM
fallback, and the NuExtract spike (`2026-08-09-nuextract-baseline-evaluation-design.md`)
-- operates on whole-book plain text (`_llm_scan_indices`-selected pages at
best, the full page list at worst). The actual signal these strategies need
lives on a tiny fraction of a book's pages: the table-of-contents listing
and each chapter's opening page. Feeding the rest in as noise costs tokens,
latency, and -- worse -- actively distorts results when unrelated pages
fuzzy-match a chapter title or contain their own numbered-list-like text
(bibliographies, indices).

A short investigation (built `pdfalto` from
[kermitt2/pdfalto](https://github.com/kermitt2/pdfalto), ran it against a
sample of this project's own ground-truth PDFs -- both native-text and
scanned/OCR'd) confirmed the premise: TOC pages and chapter-opening pages
have geometrically distinct signatures in the PDF's layout data alone, well
before any text-content matching happens --

- **TOC pages**: many short lines, most ending in a short numeral token
  right-aligned near a consistent x-position, 2-3 discrete left-indent
  levels for hierarchy, high line-width variance.
- **Chapter-opening pages**: one or two large-font lines at the top (title
  -- visible in the ALTO output's per-string `FONTSIZE`, e.g. 31pt vs 9pt
  body text), a byline line, then dense paragraph text with *low* variance
  in both left-margin and line-width (justified body copy) -- the opposite
  shape of a TOC page.

This held even on the scanned half of the corpus, because those PDFs
already carry an embedded OCR text layer (noisy content, e.g.
"Grüßner"→"GriiBner", but correct *positions*) -- so a layout classifier
does not require a fresh OCR pass to work on scanned books. Extraction is
fast (1.2s for a 226-page book with `-skipGraphs`) and pdfalto's embedded-
outline (`-outline`) output is redundant with what `extract_outline_candidates`
(segmentation.py:39) already does via pypdf, so it adds nothing new there --
the value is entirely in the ALTO layout XML.

Two things are missing before this can be measured: (1) the ground-truth
corpus has no record of where each book's TOC actually is (`.expected.json`
tracks chapter boundaries only), and (2) the automated CrossRef GT pipeline
that already structurally detects TOC pages -- purely to exclude them from
its chapter-start search -- throws that information away instead of keeping
it.

Goal of this spec: close both gaps, then run a standalone measurement pilot
(same spirit as the NuExtract baseline spike) that produces a go/no-go
signal for further investment -- production wiring and/or a larger,
purpose-built ground-truth push via CrossRef.

## Scope

### 1. Ground-truth schema retrofit

Add an optional top-level `"toc"` field to the `.expected.json` schema,
sibling to `"chapters"`:

```json
{
  "chapters": [...],
  "toc": {"toc_start_index": 7, "toc_end_index": 8}
}
```

Same 0-based-physical-page convention as `pdf_start_index`/`pdf_end_index`.
Two distinct states matter and are not interchangeable:

- **Key absent**: book not yet retrofitted (transitional state during this
  migration; also the state of any evaluation book added before this spec).
- **`"toc": null`**: retrofitted and confirmed -- this book has no
  locatable printed TOC page (should be rare; e.g. some `copyrighted-scans/`
  personal-library entries like reports or dissertations).
- **`"toc": {"toc_start_index": ..., "toc_end_index": ...}`**: retrofitted,
  TOC located at this contiguous physical-page range.

A shared helper, `toc_page_range(toc_pages: set[int]) -> tuple[int, int] | None`,
is added to `evaluation/scripts/ground_truth_helper.py` next to the
existing `find_toc_pages` (line 97), which it consumes:

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

A new script, `evaluation/scripts/add_toc_ground_truth.py`, retrofits every
existing `.expected.json` under `evaluation/corpus/open-access/` and
`evaluation/corpus/copyrighted-scans/` (globs both; skips `pending/`, which
by definition has no `.expected.json` yet):

1. Skip any file that already has a `"toc"` key (idempotent by default,
   `--force` to re-run), same convention as `build_crossref_gt_ground_truth.py`.
2. Load page text via `pypdf`'s `extract_text()` (existing convention),
   run `find_toc_pages` + `toc_page_range`.
3. If a range is found, write it. If `find_toc_pages` returns empty or a
   non-contiguous set, write nothing and print the book in a "needs manual
   review" list at the end (mirrors `build_crossref_gt_ground_truth.py`'s
   SKIP-with-reason reporting) -- for empty-set books this includes the
   known failure mode where a scanned book's `extract_text()` has no
   readable layer at all pre-OCR-cache, same as chapter-GT building.

Auto-written entries get a spot-check pass (open the PDF at
`toc_start_index`/`toc_end_index`, confirm), not the full-book per-entry
verification `evaluation/CLAUDE.md` mandates for chapter boundaries --
appropriate here because the pilot's decision criteria (below) is a
recall-first filter bar, not a precision-critical production path, so it
tolerates more label noise than chapter-boundary ground truth does.
Anything the spot-check catches as wrong gets hand-fixed same as always.

`evaluation/CLAUDE.md` gains a new "Step 5: TOC ground truth" section for
manually-added books going forward, so the workflow doc doesn't drift
behind the code (per this project's own document-lifetime convention).

### 2. CrossRef GT workflow update (must land first)

`build_crossref_gt_ground_truth.py` already computes
`toc_pages = find_toc_pages(pages)` (line 146) solely to exclude those pages
from `_locate_near`'s chapter-start search, then discards it. Change:
after `_sanity_check` passes, also call `toc_page_range(toc_pages)` and
write the result into the written `.expected.json`'s `"toc"` field (`null`
if `None`). This has to ship before the pilot script is written against the
corpus, so every book this pipeline migrates from now on -- including any
run during this spec's own retrofit pass -- carries the label for free, at
zero incremental annotation cost.

### 3. Pilot script

`evaluation/scripts/evaluate_layout_toc_classifier.py` -- manual run, not
part of `uv run pytest` or CI (same convention as
`evaluate_nuextract_baseline.py` and `fetch_evaluation_pdfs.py`).

**Feature extraction**: shells out to a locally-built `pdfalto` binary
(path from `--pdfalto-bin` or a `PDFALTO_BIN` env var -- not vendored,
matching the Kreuzberg-sidecar precedent of treating an external tool as a
developer-provided dependency, not a bundled one) with `-skipGraphs`, once
per book. Output ALTO XML is cached at
`evaluation/corpus/<corpus>/.layout-cache/<key>.alto.xml`, added to
`evaluation/.gitignore` alongside the existing `.ocr-cache/` entry (same
sensitivity: derived from copyrighted PDFs, not meant to be redistributed).
This cache is a pilot-script speed optimization only -- not a production
artifact; whether/how a real runtime path caches this is out of scope,
deferred to whatever spec follows a successful pilot.

Per page, the ALTO `TextLine`/`String`/`TextStyle` elements are reduced to
a fixed feature vector (~15-20 scalars), covering:

- line count; mean and variance of line width (`WIDTH`) and left-margin
  (`HPOS`) -- TOC pages have high variance here, chapter-body pages low
- fraction of lines whose last token is a short numeral or roman numeral
  (the dot-leader/right-aligned-page-number signal)
- ratio of the page's largest `FONTSIZE` to its modal (most common,
  i.e. body-text) `FONTSIZE`, and whether that max-size text sits in the
  top fifth of the page (title-block signal)
- vertical position (`VPOS`) of the first text block (top-of-page
  whitespace, e.g. a title page's larger top margin)
- line density (line count / page `HEIGHT`)

**Labels**: for each book with a `"toc"` field present (range or
confirmed-null) after step 1/2 above, every page in
`toc_start_index..toc_end_index` is labeled `toc`; every chapter's
`pdf_start_index` page is labeled `chapter_first`; every other page is
`other`. Books still missing a `"toc"` key (not yet retrofitted or flagged
for manual review) are excluded from the pilot corpus entirely, not treated
as confirmed-empty.

**Model**: a single scikit-learn classifier (gradient-boosted trees, e.g.
`HistGradientBoostingClassifier`, or `LogisticRegression` as a simpler
baseline to compare against) over the feature table, with balanced class
weights given the natural rarity of the `toc` class (one short range per
book vs. hundreds of `other` pages). `scikit-learn` is added as a new
`evaluation`-only optional dependency in `pyproject.toml` (same pattern as
the existing `kreuzberg`/`tesseract`/`llm-eval` extras) -- not a runtime
dependency of the `chapter_segmentation` package itself.

**Evaluation protocol**: leave-one-book-out cross-validation across every
book with usable labels -- train the model on all other books' pages,
predict probabilities for the held-out book's pages, repeat for every book,
so the reported numbers reflect generalization across publishers/layouts
rather than memorization of one book's specific template (the realistic
risk given this corpus's size). A book's own true chapter count is *not*
available to the classifier at prediction time -- candidate pages are
selected purely by per-page class probability crossing a threshold τ, where
τ is chosen per fold from the training pages only (never the held-out
book), to avoid leaking test-set information into the threshold itself.

**Reporting**: printed to stdout in the same table shape as
`RESULTS.md`'s existing per-strategy numbers (per-book and aggregate). This
spec does not itself commit a results file -- per this project's
document-lifetime convention, a measured snapshot belongs in a living
results document written *after* the run, not baked into a design spec
written before it. If the pilot succeeds, its numbers get written up in
`evaluation/RESULTS.md` (new section) as part of the follow-up work that
decision unlocks, not as part of this spec.

## Decision criteria

Recall-first, matching the classifier's actual job (narrowing the page set
handed to the real extraction strategies, not making the final call --
false positives cost a little extra downstream work, false negatives are
unrecoverable). "Viable, worth investing in production wiring and/or a
larger CrossRef-sourced GT push" means, across leave-one-book-out folds:

- For **≥90% of books** with usable labels, the candidate page set (pages
  with predicted probability ≥ τ for their respective class) contains
  **every** true `chapter_first` page and **at least one** true `toc`-range
  page.
- The **average candidate-set size is ≤15%** of a book's total page count
  -- otherwise the classifier isn't meaningfully filtering anything.

A result that clears one bar but not the other (e.g. perfect recall but a
30%-of-book candidate set) doesn't kill the idea -- it means the feature
set or threshold-selection strategy needs another pass before production
wiring is worth building, same tempering language the NuExtract spike used
for its own weak-result case.

## Out of scope

- Any `segmentation.py` / `TocExtractionStrategy` wiring, or a production
  pre-filtering layer ahead of the existing cascade -- a follow-up spec's
  job, once this pilot's numbers justify it (mirrors the NuExtract spike's
  own "no production wiring" non-goal).
- TOC-*entry* structuring (parsing title/page-number pairs directly from a
  classified TOC page's layout) -- this pilot is page-*type* triage only.
  Entry-level extraction remains each `TocExtractionStrategy`'s job.
- Expanding `evaluation/crossref_gt/manifest.json` beyond its current 43
  curated ISBNs -- that curation effort is the conditional "if successful,
  get more GT" follow-on this spec's decision criteria gates, not part of
  its own implementation.
- Packaging or vendoring the `pdfalto` binary -- a pilot-runner builds it
  locally per its own README, same as the Kreuzberg OCR sidecar today.
- An image/rendered-page classifier (CNN over page bitmaps) -- deliberately
  out of scope. A geometry-feature classifier is ~100-1000x smaller (tens
  of KB for a shallow tree ensemble vs. 10-45MB+ for even a "small" CNN
  like MobileNetV3-small/ResNet18, plus a `torch`/`tensorflow` dependency
  versus scikit-learn's CPU-only footprint) and faster (no page-rendering
  step; ALTO parsing is already needed as the feature source, at ~5ms/page
  measured). It is also the right fit for this corpus's size -- tens of
  true-TOC positives and a few hundred chapter-first positives across ~50
  books is a low-data regime for a CNN to generalize in without heavy
  transfer learning, but is a normal-sized problem for a shallow tabular
  model.
- Fine-tuning or otherwise adapting `pdfalto` itself -- used as-is via its
  CLI.
