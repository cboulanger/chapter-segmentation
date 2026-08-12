# Layout-based TOC/chapter-first-page classifier: page-width feature normalization

Status: approved for planning
Date: 2026-08-10

## Problem

`2026-08-10-layout-based-toc-classifier-pilot-design.md`'s pilot ran to
completion against the full 50-book corpus and came back **NOT MET**: 16%
`full_recall_fraction` (bar: ≥90%) against a healthy 5.3%
`avg_candidate_fraction` (bar: ≤15%, comfortably cleared). A follow-up
investigation, run directly against the pilot's cached ALTO XML (no
re-extraction needed), root-caused the shortfall to two contributing
factors, one dominant and one secondary:

**Dominant: 4 of the classifier's 10 features are not comparable across
books.** `width_mean`, `width_var`, `left_margin_mean`, and
`left_margin_var` (`evaluation/scripts/layout_features.py:122-133`) are
computed directly from ALTO's `WIDTH`/`HPOS` attributes -- absolute point
coordinates on that book's own page -- and never divided by anything. The
other position-derived features, `first_text_vpos_fraction` and
`line_density`, already divide by page `HEIGHT`. Page width varies far more
across `copyrighted-scans` (304-991pt, stdev 184) than `open-access`
(420-595pt, stdev 50) -- splitting the pilot's leave-one-book-out (LOBO)
results by corpus shows this tracks the failure almost exactly:

| corpus | books | avg `chapter_first` recall | books at 0% recall | books at 100% recall |
| --- | --- | --- | --- | --- |
| open-access | 37 | 83% | 0 | 9 |
| copyrighted-scans | 13 | 20% | 8 | 1 |

Direct inspection of one zero-recall book's own extracted features
(`9780367439712`) shows the signal is genuinely present *within* that
book -- `top_block_is_large_font` and `font_size_max_ratio` cleanly
distinguish its chapter-opening pages from ordinary ones -- so this isn't a
feature-extraction bug or an OCR/typography artifact (an earlier,
now-retracted theory). It's that a classifier trained via LOBO mostly on
narrower open-access pages learns absolute-coordinate thresholds that don't
transfer to a held-out book with an unusually wide or narrow page.

**Secondary: the decision bar requires literal 100% `chapter_first` recall
per book.** Re-scoring the same LOBO run with the pass/fail rule relaxed to
a tolerance shows this alone is not sufficient to close the gap:

| tolerance | full_recall_fraction |
| --- | --- |
| 100% (current) | 16% |
| 95% | 20% |
| 90% | 26% |
| 80% | 34% |

Even an 80%-of-chapters tolerance falls far short of 90%, and the per-book
recall distribution has a genuine long tail (14%, 20%, 27%, 40%, ...), not
just "one page short of every book." This confirms the bar is a real,
compounding factor but not the dominant one -- fixing it alone would not
have gotten the pilot to MET.

This spec addresses the dominant factor only: normalizing the four
unnormalized features by page width. The bar-strictness issue and any
further feature-set work are explicitly deferred (see "Out of scope") so
this follow-up isolates exactly one variable and its effect is
measurable on its own.

## Scope

### 1. Normalize `width_mean`, `width_var`, `left_margin_mean`, `left_margin_var` by page width

In `evaluation/scripts/extract_page_features` (`layout_features.py:90`),
read the page's width alongside its existing height read:

```python
page_height = float(page.get("HEIGHT"))
page_width = float(page.get("WIDTH"))
```

and divide the raw per-line values before computing statistics on them:

```python
widths = [float(line.get("WIDTH")) / page_width for line in lines]
left_margins = [float(line.get("HPOS")) / page_width for line in lines]
```

`width_mean`/`width_var`/`left_margin_mean`/`left_margin_var` are computed
exactly as today (`statistics.mean`/`statistics.variance` over these lists)
-- only the values fed in change, from absolute points to a fraction of
page width, mirroring how `first_text_vpos_fraction`/`line_density` already
divide by page `HEIGHT`. No other feature, `FEATURE_NAMES`, the function
signature, or any downstream caller changes -- this is a pure correction to
what four of the ten numbers *mean*, not a schema change.

### 2. Update hand-computed test fixtures

`tests/test_layout_features.py` has hand-computed expected values for these
four features derived from its existing ALTO fixtures. Update each expected
value to the same computation divided by that fixture's page `WIDTH` (plain
arithmetic against already-known fixture data -- no new fixtures, no new
test cases). This is correcting existing test expectations to match
corrected behavior, not new test coverage, since the behavior being fixed
(scale-comparability across books) was always the implicit intent of a
cross-book LOBO classifier, just not what the code did.

### 3. Re-run the pilot evaluation and report

Re-run `evaluation/scripts/evaluate_layout_toc_classifier.py` exactly as
written -- no script changes, since it already re-derives features from the
cached ALTO XML on every run, so this exercises the fix directly against
real data with no new extraction step. Same 50-book corpus, same 100%
`chapter_first`-recall decision bar as the original pilot (this follow-up
does not touch the bar -- see "Out of scope").

Report, in `evaluation/RESULTS.md` (new section, matching its existing
prose-plus-tables style -- see e.g. "Diverse real-library evaluation set"
for the tone/structure to follow):

- The original pilot's NOT MET result and root cause (the unnormalized
  features, briefly -- full detail lives in this spec).
- The fresh `full_recall_fraction` / `avg_candidate_fraction` / MET-or-NOT
  verdict, plus the same open-access-vs-copyrighted-scans recall breakdown
  used to diagnose the original problem, so the write-up shows plainly
  whether normalization closed the gap, narrowed it, or left it unchanged.
- If still NOT MET: name the bar-strictness finding above (100%→34% recall
  ceiling even at 80% tolerance) as the next-most-likely place to look,
  since that diagnosis already exists and shouldn't be re-derived by
  whoever picks this up next -- without proposing or scoping that fix here.

## Decision criteria

This follow-up's own job is done once the fix is implemented, tested, and
the pilot has been re-run and reported -- there is no new pass/fail bar to
clear beyond the original pilot's existing decision criteria (unchanged
from `2026-08-10-layout-based-toc-classifier-pilot-design.md`: ≥90% full
recall, ≤15% average candidate fraction). Whether the fresh number is MET
or NOT MET, the RESULTS.md write-up is the deliverable either way -- a
still-NOT-MET result with a clearly narrowed gap and a named next suspect
is still forward progress worth recording, same tempering stance the
original pilot spec and the NuExtract spike both took.

## Out of scope

- The bar-strictness issue (100%-recall-per-book requirement) -- quantified
  during this investigation as a real, secondary contributing factor, but
  addressing it is a separate change with its own tradeoffs (what tolerance
  is defensible, whether it changes the spirit of "candidate set must never
  drop a true chapter start") that deserves its own review, not a
  side-effect of a feature-normalization fix. Left for whoever picks up the
  "next-most-likely blocker" this spec's RESULTS.md write-up names, if
  normalization alone doesn't clear the bar.
- Any further feature-set changes (new features, removing weak ones,
  changing the model or threshold-selection strategy) -- this follow-up
  changes exactly one thing (page-width normalization) so its effect is
  measurable in isolation. A muddled result (partial improvement, unclear
  cause) is more useful diagnostically than a bundle of simultaneous
  changes would be.
- Production wiring, `TocExtractionStrategy` integration, or expanding
  `evaluation/crossref_gt/manifest.json` -- unchanged from the original
  pilot spec's own "Out of scope," still gated on a MET result (or a
  deliberate decision to proceed anyway) neither of which this follow-up
  produces on its own.
- Re-running or invalidating the `.layout-cache/` ALTO XML cache -- the fix
  is entirely in Python-side feature computation from already-cached ALTO
  output; the cache itself (raw pdfalto output) is unaffected by this
  change and needs no regeneration.
