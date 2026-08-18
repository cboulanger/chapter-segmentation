# Layout-based TOC/chapter-first-page classifier pilot

**This is a snapshot of an experimental pilot, not a result of the main
chapter-segmentation workflow.** `evaluation/scripts/evaluate_layout_toc_classifier.py`
investigates whether a classifier trained purely on page-layout geometry
(no text content) can identify table-of-contents pages and chapter-opening
pages -- a candidate pre-filter that, if it ever clears its own decision
bar, could feed a downstream stage (e.g. an LLM confirmation pass) instead
of or alongside the production heuristic pipeline. It has not been
integrated into `chapter_segmentation` and does not affect the numbers in
`evaluation/RESULTS.md`. "Current status" below is expected to go stale
and be rewritten whenever the pilot is re-run or changed; "History" holds
the full write-up for every superseded run and follow-up along the way,
kept in full so the reasoning and dead ends behind the current numbers
aren't lost.

## Current status


`evaluation/scripts/evaluate_layout_toc_classifier.py` trains a
leave-one-book-out (LOBO) classifier on geometric layout features
(`evaluation/scripts/layout_features.py`, derived from cached ALTO XML) and
scores whether it can identify table-of-contents pages and chapter-opening
pages purely from page layout, no text content -- see
`docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`
for the pilot's original design and decision bar. `full_recall_fraction` is
the share of books, across LOBO folds, whose predicted candidate pages
include every true chapter-opening page (or, since a later follow-up
relaxed this, at least `chapter_first_recall_tolerance` of them -- default
`0.90`) and at least one true TOC page (bar: ≥90% of books);
`avg_candidate_fraction` is the average share of a book's pages that end up
in that candidate set at all -- how much the classifier actually narrows
the page list down (bar: ≤15%, smaller is better).

This pilot has never cleared its own 90%/15% bar on the full mixed corpus.
Eleven follow-up investigations since the original run -- feature
normalization, an OCR data-quality fix, calibration tuning, a
model-architecture swap, two corpus-growth re-runs, a feature-engineering
pass, a targeted feature swap, a per-corpus calibration study, a
source-type-split study, and finally a change to the candidate-selection
mechanism itself -- progressively improved the numbers without closing the
gap. Full write-ups for all of them are under "History" below. What
follows is the shipped configuration and the two most recent measurements
against it.

**Current shipped configuration:**

- **Model:** `LogisticRegression(class_weight="balanced")` with a per-fold
  `StandardScaler`, replacing the original `HistGradientBoostingClassifier`
  (see "Follow-up: relaxing the per-book bar, and a model-architecture
  swap" under History below).
- **Features (16):** the original ten geometric layout features, plus two
  page-local features (`last_text_vpos_fraction`, `top_line_heading_match`)
  and four book/context features from `add_book_context_features()`
  (`prev_last_text_vpos_fraction`, `prev_line_count_rel`, `line_count_rel`,
  `font_size_max_ratio_book`), plus `edge_distance`, which replaced
  `page_position_fraction` (see "Follow-up: context/normalized features and
  scan-noise augmentation" and "Follow-up: `edge_distance` feature,
  replacing `page_position_fraction`" under History below). A candidate
  18th feature, `early_gap_ratio`, was tried and reverted -- net negative
  once measured across the full LOBO suite (see "Follow-up: is
  `recall_target` worth splitting by source type? And a rejected feature"
  under History below).
- **`chapter_first_recall_tolerance`:** `0.90` (a book counts as "fully
  recalled" at ≥90% of its true `chapter_first` pages, not literal 100%).
- **Candidate selection:** `select_candidates_by_document_budget` /
  `--candidate-fraction-cap` (default `0.15`) is now `main()`'s default
  path, replacing training-quantile-calibrated `--recall-target`
  thresholding as the shipped default -- see the first follow-up directly
  below. `--recall-target` remains available as an explicit override back
  to the legacy strategy.
- **Scan-noise augmentation** (`--scan-noise-augment`,
  `evaluation/scripts/alto_scan_noise.py`): implemented, cached, off by
  default -- tested and found mostly negative (see "Follow-up:
  context/normalized features and scan-noise augmentation" under History
  below).
- **Ground truth fix (applied, not pending):** `9783837660944.expected.json`
  had a spurious `"Inhalt"` chapter entry duplicating that book's own TOC
  page range, which had been overwriting the true `toc` label with
  `chapter_first` and injecting a false positive; the entry has been
  deleted (see "Follow-up: isolating the open-access corpus..." under
  History below).
- **Not implemented:** splitting `recall_target` by `extraction_type`
  (native vs. scan) was measured as a real, reproducible win (+9 points of
  overall `full_recall_fraction` at a matched candidate budget) under the
  old `recall_target`-based calibration, but was superseded before
  implementation by the candidate-fraction-cap mechanism below, which
  reaches comparable numbers with a single constant and no need to detect
  `extraction_type` at all (see "Follow-up: is `recall_target` worth
  splitting by source type?..." under History below).

The two most recent measurements, in full:

### Follow-up: document-relative candidate-budget selection -- implemented and kept

The "is `recall_target` worth splitting by source type?" follow-up's (see
History below) "more principled retry" idea was implemented and
tested rather than left as a proposal: `select_candidates_by_document_budget`
and `evaluate_leave_one_book_out_document_budget`
(`evaluation/scripts/evaluate_layout_toc_classifier.py`, wired up behind a
new `--candidate-fraction-cap` flag, mutually exclusive with
`--recall-target`) replace `select_threshold`'s training-calibrated,
uniformly-applied absolute probability threshold with a per-document one:
for each held-out book, rank its own pages by `max(prob_toc,
prob_chapter_first)` and take the top `candidate_fraction_cap` share as
candidates -- a single shared top-K selection across both labels, requiring
no ground truth for the document being scored, only its own model output.

**Result: a clear, broad improvement, not a regression, at every operating
point checked -- kept, not reverted.** On the full 89-book corpus (the
mixed pool every earlier `recall_target` sweep in this file struggled
with):

| `candidate_fraction_cap` | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| 0.10 | 62% | 9.9% |
| 0.12 | 70% | 11.8% |
| **0.15** | **79%** | **14.9%** |
| 0.18 | 83% | 17.8% |
| 0.20 | 83% | 19.9% |

At `cap=0.15` -- the exact value of the pilot's own candidate-fraction
budget -- this reaches **79% full_recall_fraction at 14.9%
avg_candidate_fraction**, comfortably the best result this file has ever
recorded on the full mixed corpus at or under the 15% cap: far above the
shipped `recall_target=0.90` default's 64%/10.4% at a similar-order
candidate cost, and above even the best single global `recall_target`
found in the previous follow-up's budget-constrained search (73.0%/13.9%
at `recall_target=0.94`) -- approaching the extraction-type-split result
(82.0%/14.5%) with a single constant and no need to know or detect
`extraction_type` at all. On open-access alone, `cap=0.15` reaches
**91.2%/14.9%**, clearing this pilot's 90%/15% bar on that corpus in
isolation (fractionally below the hand-tuned `recall_target=0.988`'s
94.7%/13.8%, but without any corpus-specific retuning -- the same `0.15`
constant that works for the full mixed corpus also clears the bar for
open-access alone). Still **NOT MET overall** on the full 89-book corpus
(79% vs. the 90% bar) -- this doesn't change the pilot's bottom line, but
it is the strongest configuration found for the mixed corpus across this
entire investigation.

**A second, independently useful property, beyond the raw numbers: per-book
candidate_fraction becomes almost perfectly uniform.** Under
`recall_target`-based selection, individual books' `candidate_fraction`
ranged from under 2% to over 58% (see the "isolating the open-access
corpus" follow-up under "History" below; e.g.
`9781783748471` at 58.3% while `9783839470619` sat at 4.8%, both at the
same `recall_target=0.988`) -- a single global threshold produces wildly
different per-book cost depending on how separated that book's own
probabilities happen to be. Under the budget cap, every book in the
89-book corpus landed within 0.7 points of the 15% target (14.5%-15.0%) at
`cap=0.15`, by construction: a well-separated book's threshold
self-adjusts higher (fewer pages clear the bar for the same rank cutoff)
and a noisy book's self-adjusts lower, but the *volume* handed to whatever
consumes these candidates is predictable per document either way. This
matters operationally if candidates feed something with a per-page cost
(e.g. an LLM confirmation pass): a runaway 58%-of-the-book candidate set
for one hard book is a cost spike a uniform ~15% never produces.

Verified this is purely additive, not a change to existing behavior: a
re-run with no `--candidate-fraction-cap` flag reproduces the exact
`recall_target=0.90` baseline (64%/10.4%, identical per-book numbers) from
before this change, since the new code path is only reachable via the new
flag.

**Update (same day): promoted to the default.** The comparison above was
conclusive enough (equal-or-better recall than any `recall_target` value
found in this pilot's history, at comparable-or-tighter candidate cost, on
both the full corpus and open-access alone, with no per-corpus retuning
needed) that `--candidate-fraction-cap` replaced `--recall-target` as
`main()`'s default code path rather than staying a same-session-only
opt-in flag. `_CANDIDATE_FRACTION_CAP = 0.15` is now the module default;
`--recall-target` remains available but is now the explicit override --
passing it switches back to the legacy training-quantile-calibrated
strategy and `--candidate-fraction-cap` is ignored (inverted from this
follow-up's original wiring, where `--candidate-fraction-cap` was the
opt-in). Verified both directions after the flip: no flags reproduces
today's 79%/14.9% (was previously only reachable via
`--candidate-fraction-cap 0.15`), and `--recall-target 0.90` reproduces
the historical 64%/10.4% shipped-default numbers exactly, confirming nothing
else in the evaluation logic changed. Every `recall_target=...` figure
elsewhere in this file (including "shipped default" language in earlier
follow-ups) describes what was true when it was written and remains an
accurate record of that measurement -- it does not describe today's
default going forward.

### Follow-up: real-scan measurement from the DNB TOC-scan corpus (calibration-grade)

`evaluation/scripts/fetch_dnb_toc_corpus.py` and
`evaluation/scripts/measure_dnb_scan_noise_stats.py` were added (see
`docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`)
to eventually replace the "resembles real scan noise" hand-picked
constants in `alto_scan_noise.py` (`_CONTRAST_ALPHA = (0.3, 0.7)`,
`_FONT_JITTER = (0.96, 1.04)`) with numbers measured directly from real
DNB-digitized table-of-contents scans, sourced via the `lobid-resources`
API. An initial 9-book smoke-test batch (see the earlier version of this
subsection, superseded below) proved the acquisition pipeline worked
end-to-end; a real `--from-dump` bulk run against the live ~21.5GB
lobid-resources dump (2026-08-15) then grew the corpus to the design
spec's "few hundred books" decision-criteria scale. That run also
surfaced and fixed two real acquisition bugs along the way (see
`docs/superpowers/plans/2026-08-15-dnb-toc-corpus-corrections.md`): the
original type filter was too broad and let single-author/thesis/textbook
records into a corpus meant to target edited-volume TOC layouts
specifically (256 of the smoke-test batch's 305 books were purged for
this), and 7 of the newly-acquired 549 had a `toc_download_url` pointing
to an HTML link-out page rather than a real PDF (removed). The corpus now holds **542 verified `EditedVolume`-typed books, all with
a PDF present locally** (5 of the original smoke-test batch's PDFs were
briefly lost when their worktree was cleaned up after merging -- a
lesson-learned noted in the corrections plan -- and were re-downloaded
directly from their already-known `toc_download_url` to complete the set).

```
uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py --pdfalto-bin <path-to-pdfalto>

Title/body contrast ratio (font_size_max_ratio, n=1050):
  measured: {'count': 1050, 'min': 1.0, 'max': 11.26, 'mean': 1.29, 'median': 1.10, 'stdev': 0.65}
  current _CONTRAST_ALPHA range: (0.3, 0.7)

Body-line font-size dispersion (ratio to page modal size, within +/-10%, n=37838):
  measured: {'count': 37838, 'min': 0.90, 'max': 1.10, 'mean': 1.00, 'median': 1.0, 'stdev': 0.0166}
  current _FONT_JITTER range: (0.96, 1.04)
```

`n=1050` non-empty pages and `n=37838` body-like line samples across all
542 real DNB TOC scans -- comfortably calibration-grade, not a preview. As
before, `_CONTRAST_ALPHA` isn't directly unit-comparable to the measured
contrast ratio (it's a compression factor applied to born-digital ALTO's
own, much larger, raw title/body ratio, not a target ratio itself), so
this table alone doesn't resolve whether `_CONTRAST_ALPHA` needs to
change -- that requires a follow-up comparing this measured real
distribution against born-digital `open-access` ALTO's own *uncompressed*
title/body ratio, not done here. What it does establish: the real
distribution is right-skewed (median 1.10 well below the mean 1.29, with
a long tail up to 11.26 -- almost certainly a handful of pages with a
genuinely large decorative heading or an OCR font-size misdetection, not
representative of the median case), and most DNB TOC pages have only mild
title/body contrast, consistent with plain "Inhalt"/"Inhaltsverzeichnis"
listings rather than illustrated title pages.

**The dispersion figure is directly comparable, and is a real, actionable
result: `_FONT_JITTER`'s current `(0.96, 1.04)` range looks well-calibrated.**
Measured real body-line sizes have stdev 0.0166 around their page's modal
size; `_FONT_JITTER`'s half-width of 0.04 is about 2.4 real standard
deviations -- a sensible coverage for a uniform jitter range meant to
approximate a real, roughly bell-shaped noise distribution without
needing to change. At this sample size (37,838 line samples, 542 books),
this is a confirmation that the original hand-picked guess was
well-calibrated, not just an unconfirmed guess -- **no change recommended
to `_FONT_JITTER`.**

With the heuristics above, a full run of that now-deleted script
(KISSKI-backed preset) reported **identical numbers to the pure-heuristic
harness, with neither fallback path firing on any book**: TOC extraction
never triggered because the regex path found a usable listing everywhere,
and per-entry ambiguity was resolved heuristically by the TOC-order
constraints before the LLM would be consulted. This run predated the
layout-mode fallback and OCR route described above, and only covered the 7
originally-committed books, not the 10 "diverse real-library" books.

**Deferred, still-untouched directions** (named across several follow-ups
above, not scoped or attempted in any of them): TOC-anchored chapter
matching (parsing a detected TOC page and locating chapter openings by
title/page-number matching against it, to address the `open-access`
TOC-layout failure mode) and document-image deep learning bypassing OCR
font metadata entirely (to address the residual scan `chapter_first_recall`
ceiling).

## History


The subsections below are the full history of this pilot's investigation,
in the order they happened -- see "Current status" above for the shipped
configuration and latest measured numbers.

### Original pilot run and feature-normalization follow-up

The subsections from here through "Follow-up: relaxing the per-book bar"
describe the pilot's history on the pre-growth **50-book** corpus -- every
book count and percentage in them refers to that corpus size. The
2026-08-12 re-run over the grown 70-book corpus, and the feature/
augmentation work that followed it, are covered in the last two follow-up
subsections at the end of this section.

`evaluation/scripts/evaluate_layout_toc_classifier.py` trains a
leave-one-book-out (LOBO) classifier on ten geometric layout features
(`evaluation/scripts/layout_features.py`, derived from cached ALTO XML) and
scores whether it can identify table-of-contents pages and chapter-opening
pages purely from page layout, no text content -- see
`docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`
for the pilot's design and decision bar. `full_recall_fraction` is the
share of books, across LOBO folds, whose predicted candidate pages include
every true chapter-opening page and at least one true TOC page (bar: ≥90%);
`avg_candidate_fraction` is the average share of a book's pages that end up
in that candidate set at all -- how much the classifier actually narrows
the page list down (bar: ≤15%, smaller is better). The original pilot run
came back **NOT MET**: 16% `full_recall_fraction` against a
comfortably-cleared 5.3% `avg_candidate_fraction`.

A follow-up investigation
(`docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md`)
root-caused most of the shortfall to 4 of the 10 features --
`width_mean`, `width_var`, `left_margin_mean`, `left_margin_var` -- being
raw, unnormalized ALTO point coordinates rather than fractions of page
width, unlike the other position-derived features
(`first_text_vpos_fraction`, `line_density`), which already divide by page
height. This tracked almost exactly with the corpus split: page width
varies far more in `copyrighted-scans` (304-991pt) than `open-access`
(420-595pt), and the original run's 8-book cluster stuck at exactly 0%
`chapter_first` recall was entirely `copyrighted-scans` books:

| corpus | books | avg `chapter_first` recall | books at 0% recall | books at 100% recall |
| --- | --- | --- | --- | --- |
| open-access (original) | 37 | 83% | 0 | 9 |
| copyrighted-scans (original) | 13 | 20% | 8 | 1 |

`extract_page_features` was fixed to divide these four features by page
width before computing statistics on them (Task 1 of the normalization
follow-up), and the pilot was re-run against the same cached ALTO XML with
no other changes. The fresh result is still **NOT MET**, but slightly
better: 18% `full_recall_fraction` (vs. 16% before) and 5.4%
`avg_candidate_fraction` (vs. 5.3% before, still comfortably under the 15%
bar). The per-corpus `chapter_first`-recall breakdown, re-computed the same
way as the original diagnosis, shows the corpus split itself is
essentially unchanged by the fix:

| corpus | books | avg `chapter_first` recall | books at 0% recall | books at 100% recall |
| --- | --- | --- | --- | --- |
| open-access (fresh) | 37 | 81% | 1 | 9 |
| copyrighted-scans (fresh) | 13 | 19% | 8 | 0 |

So the width-normalization fix did what it set out to do (the four
previously-unnormalized features are now genuinely comparable across
books of different page widths) without meaningfully closing the gap
between the two corpora, or the overall `full_recall_fraction` bar. This
matches the normalization spec's own root-cause writeup, which named the
unnormalized features as the *dominant* factor but not the only one: it
separately quantified a secondary, compounding issue -- the decision bar's
requirement of literal 100% `chapter_first` recall per book -- and found
that even relaxing it to an 80% tolerance only lifted `full_recall_fraction`
to 34% on the original (pre-normalization) run, still well short of the
90% bar. That bar-strictness finding is the next-most-likely place to look
if this pilot is picked up again; scoping or implementing a fix for it is
explicitly out of scope for both this run and the normalization follow-up
that produced it
(`docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md`'s
"Out of scope" section).

### Follow-up: replacing textless/degenerate-text corpus PDFs with OCR'ed versions

A closer look at the 8-book cluster still stuck at exactly 0%
`chapter_first` recall after the normalization fix found a second, larger
root cause for 6 of the 8: `pdfalto` (the pilot's only extraction tool,
`evaluation/scripts/pdfalto_runner.py`) reads a PDF's own embedded
text/layout directly and has no OCR fallback of its own -- unlike
`chapter_segmentation`'s own text-based pipeline, which recovers usable
text for these exact books via a dedicated OCR pass
(`src/chapter_segmentation/ocr.py`, `.ocr-cache/`, see
`evaluation/RESULTS.md`'s "Diverse real-library evaluation set" section: `9781409403906.pdf`,
`9783465016878.pdf`, `9783848704316.pdf`, `dnb-36942798X.pdf` had "no text
layer at all"; `9780367439712.pdf`, `9783789057366.pdf` had a "degenerate
text layer"). On these 6 books, `pdfalto` extracted zero text lines at all
on many pages (an all-zero feature vector) or near-uniform, unresolvable
font-size data -- no amount of feature normalization can recover a signal
that was never extracted in the first place.

Per this project's evaluation philosophy (production code should OCR at
run time; the *evaluation corpus* should already contain suitable data, so
a pipeline bug and a data-quality gap are never conflated), the fix was to
replace the 6 affected PDFs in the shared corpus
(`evaluation/corpus/copyrighted-scans/`, gitignored, symlinked identically
into every worktree) with OCR'ed versions, not to teach `pdfalto_runner.py`
to run OCR itself. `evaluation/scripts/ocr_evaluation_pdfs.py`'s existing
OCR path only produces cached plain text (`list[str]`), not a new PDF with
an embedded text layer, so it can't be reused for this -- `ocrmypdf`
(installed via `brew install ocrmypdf`, plus `tesseract-lang` for German)
was used instead: `ocrmypdf --force-ocr -l <lang> <original> <output>`,
with each original preserved alongside as `<key>.original.pdf`. Verified
before re-running the pilot: page count unchanged for all 6 (pypdf), and
`pages_need_ocr` (`src/chapter_segmentation/segmentation.py`) now returns
`False` for all 6, where it previously returned `True`.

Re-running the pilot (same cached ALTO XML for the other 44 books; the 6
fixed books' `.layout-cache/` entries were deleted so `pdfalto` re-ran on
the new PDFs) shows real, measurable per-book improvement on 4 of the 6:

| book | chapter_first recall before | chapter_first recall after |
| --- | --- | --- |
| `9783848704316` | 0% | 73% |
| `9781409403906` | 0% | 42% |
| `9783465016878` | 0% | 23% |
| `dnb-36942798X` | 0% | 6% |
| `9780367439712` | 0% | 0% (unchanged) |
| `9783789057366` | 0% | 0% (unchanged) |

The two unchanged books fail for two different, already-diagnosed reasons,
not a leftover OCR problem: `9780367439712`'s chapter-opening pages have
clean, strong font-size signal both before *and* after OCR (12/12 pages
cross the title-detection threshold either way) -- this is a LOBO
model-generalization gap (its true positives top out around probability
0.45 against a 0.68 threshold calibrated from other books), unrelated to
data quality. `9783789057366` still shows weak font-size differentiation
even with real OCR'd text (only 2 of 56 chapter-opening pages cross the
threshold) -- plausibly because this scan's low average image DPI (~91,
per `ocrmypdf`'s own logged warnings) is too coarse for Tesseract's
per-line font-size estimation to reliably separate title-sized from
body-sized text.

Despite that per-book progress, the overall `full_recall_fraction` stayed
flat at **18%** (was 18% after normalization, 16% originally) -- still
**NOT MET**. `avg_candidate_fraction` improved slightly, 5.4% to 4.4%,
still comfortably under the 15% bar. The corpus-wide number held flat
because every held-out book's classifier is trained on all *other* books'
rows in this leave-one-book-out setup: swapping degenerate rows for real
ones in 6 books measurably shifted many *other*, untouched books' scores
too (both up and down), roughly canceling out in the aggregate. This is
further evidence, on top of the normalization follow-up's own finding,
that the decision bar's literal "100% `chapter_first` recall per book"
requirement -- not any single data-quality or feature issue -- is this
pilot's dominant remaining blocker: most of the corpus already sits in a
60-95% per-book recall band, comfortably real signal, just short of the
all-or-nothing bar.

`evaluation/CLAUDE.md`'s "Step 0a" now requires a real, usable embedded
text layer (checked with `pages_need_ocr`) before any scanned PDF is added
to `copyrighted-scans/` going forward, so this gap doesn't recur silently
for future books.

### Follow-up: recall-target tuning, concentrating on the open-access corpus

With the data-quality issues above addressed, this follow-up asked how far
the existing pipeline -- unchanged features, unchanged model, unchanged
decision bar -- could be pushed by tuning `evaluate_layout_toc_classifier.py`'s
own calibration knob, `_RECALL_TARGET`. `select_threshold` picks the highest
per-fold probability threshold that still achieves at least `_RECALL_TARGET`
recall on that fold's *training* positives; the value had been an arbitrary
`0.90` since the pilot's first run. Since `avg_candidate_fraction` had been
sitting far under its 15% budget the whole time (4.4% most recently), there
was untapped room to trade some of that slack for recall by raising the
target. Per the "concentrate on open-access first, fix outliers later"
priority, `copyrighted-scans` books stayed in the training pool throughout
(removing them from training was tried and made open-access recall *worse*,
not better -- see below) but their own scores were tracked separately rather
than optimized for.

A sweep over `_RECALL_TARGET` from 0.90 to 1.00 (LOBO over the full 50-book
corpus) found a steep, then flat, then explosive response:

| `_RECALL_TARGET` | `full_recall_fraction` (all) | `avg_candidate_fraction` (all) |
| --- | --- | --- |
| 0.90 (previous) | 18% | 4.4% |
| 0.95 | 20% | 5.7% |
| 0.96 | 26% | 6.1% |
| **0.97** | **28%** | **7.0%** |
| 0.98 | 28% | 11.1% |
| 0.99 | 28% | 21.9% |
| 1.00 | 80% | 87.3% |

0.97 is the cheapest point on the 0.97-0.99 plateau -- same recall as 0.98
and 0.99, at roughly half to a third of their candidate-fraction cost -- and
comfortably clear of 1.00's cliff, where the "threshold" degenerates to
"flag almost every page" and stops being a useful filter at all. Broken down
by corpus at 0.97:

| corpus | `full_recall_fraction` (before -> after) | `avg_candidate_fraction` |
| --- | --- | --- |
| open-access | 24.3% -> 35.1% | 6.6% |
| copyrighted-scans | 0% -> 7.7% | 7.9% |

Three other tuning ideas were tried and rejected, in the interest of
recording negative results alongside the positive one:

- **Training on open-access books only** (excluding `copyrighted-scans`
  entirely from the LOBO training pool, not just from the target metric):
  this made open-access recall *worse* (16.2% vs. 24.3% at the old 0.90
  target) -- the other corpus's rows, despite their own data-quality
  problems, still contribute generalizable signal rather than just noise.
- **A new feature, `max_font_vpos_fraction`** (the vertical position of the
  page's *largest-font* line, as opposed to `first_text_vpos_fraction`'s
  topmost line regardless of size -- meant to stop a running header above
  the real title from collapsing that signal to near-zero): dropped
  open-access `full_recall_fraction` from 43.2% to 29.7% at the same 0.97
  target. With only ~600 positive `chapter_first` rows across 37 books, an
  11th feature widens the model's overfitting surface faster than it adds
  real signal.
- **Splitting `_RECALL_TARGET` per label** (a low target for `toc`, whose
  pass bar is lenient -- "at least one hit" -- freeing up candidate-fraction
  budget for a higher `chapter_first` target): backfired. A *lower* target
  makes `select_threshold` pick a *higher*, more selective threshold; for
  several books that threshold was high enough to reject every `toc`
  candidate, failing that label's lenient bar outright. `min_samples_leaf`
  values above the existing default of 1 (tried 2, 3, 5, 10) were also all
  strictly worse.

The result is still **NOT MET** (28% vs. the 90% bar), and the previous
follow-up's bar-strictness finding remains the dominant blocker -- this
tuning pass narrows the gap without closing it. But it is a real,
non-cosmetic improvement obtained purely by recalibrating an existing,
already-arbitrary constant: open-access `full_recall_fraction` improved by
nearly half (24.3% to 35.1%) while candidate fraction stayed at less than
half the 15% budget. `copyrighted-scans` remains the weaker corpus by a wide
margin and continues to be treated as the deferred outlier bucket, per the
"open-access first" priority this follow-up was scoped to.

### Follow-up: relaxing the per-book bar, and a model-architecture swap

Two more changes, requested together: relax the pilot's literal "100% of a
book's chapter_first pages" pass bar (the dominant blocker named in every
follow-up above), and investigate whether the classifier's *model* -- not
just its calibration -- was leaving recall on the table.

**Relaxing the bar.** `evaluate_leave_one_book_out` gained a
`chapter_first_recall_tolerance` parameter (module default `0.90`,
`_CHAPTER_FIRST_RECALL_TOLERANCE`), replacing the exact `recall == 1.0`
check with `recall >= tolerance`. This only changes how a book is scored as
"fully recalled" for `full_recall_fraction` -- it has no effect on which
candidate pages get produced. A sweep from 1.00 down to 0.50 (LOBO, full
50-book corpus, still on the tree-based model at the time) showed a steep,
monotonic response with no plateau: 28% (1.00) -> 32% (0.95) -> 40% (0.90)
-> 48% (0.80) -> 60% (0.70) -> 72% (0.50). Even 0.50 didn't reach the
pilot's own 90% "MET" bar, and at that tolerance "passing" only requires
catching half a book's chapter openings -- too weak to call the classifier
a real detector. 0.90 (miss at most 1 in 10 chapter-openings per book) was
chosen as a tolerance that stays meaningful while acknowledging that a
book's natural layout diversity (an unnumbered first chapter, a
part-divider) means no realistic amount of tuning closes the last mile to
literal 100%.

**Model architecture.** Asked directly why recall was still low given the
tuning already done, three classifiers were compared head-to-head on
identical features, at the same recall-calibration target:

| model | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| `HistGradientBoostingClassifier` (previous) | 40% | 7.0% |
| `RandomForestClassifier` | 0% | 0.1% |
| `LogisticRegression` (+ `StandardScaler`) | 44% | 15.0% |

`RandomForestClassifier` is not viable here -- its `predict_proba` is badly
miscalibrated for this data shape (small, imbalanced, near-duplicate rows
within a book), so `select_threshold` picks a threshold that accepts almost
nothing on held-out books. `LogisticRegression` clearly outperforms the
tree-based model on recall, and the mechanism is architectural, not just a
better hyperparameter: `HistGradientBoostingClassifier` partitions feature
space into discrete leaf regions and calibrates its threshold against one
fold's own training-probability distribution -- a held-out book whose
feature distribution sits slightly off from that fold's training pool can
fall in a gap between leaf regions that the threshold doesn't line up with,
even when the page has an obvious, real chapter-opening signal.
`LogisticRegression`'s smooth, continuous, linear score has no such
discontinuity, so it transfers across books far more gracefully -- evidence
that the true relationship between these ten geometric features and
"chapter_first-ness" is close enough to monotonic/linear that a simpler
model generalizes better than a more expressive one. This is consistent
with two earlier negative results in this same investigation
(the rejected `max_font_vpos_fraction` feature and the rejected
higher-`min_samples_leaf` settings): the pipeline was never capacity-limited,
it was data-limited, and a lower-capacity model suits that regime better.

The pilot's model was switched to `LogisticRegression(class_weight="balanced")`
with a per-fold `StandardScaler` (needed because, unlike tree splits, a
linear model's fit is sensitive to feature scale; tree-based
`HistGradientBoostingClassifier` never needed one). `_RECALL_TARGET`'s
default moved from `0.97` (tuned for the tree model) to `0.80` -- the
highest point on `LogisticRegression`'s own curve that still respects the
*previous* 15%-candidate-fraction budget:

| `LogisticRegression` `recall_target` | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| 0.80 (new default) | 44% | 15.0% |
| 0.85 | 54% | 17.0% |
| 0.90 | 58% | 20.5% |
| 0.95 | 68% | 39.2% |
| 0.97 | 78% | 61.4% |

**Both constants are now runtime-configurable**, via `--recall-target` and
`--chapter-first-recall-tolerance` (or as `evaluate_leave_one_book_out`
keyword arguments), rather than requiring a source edit. They control
different things: `recall_target` is the real "how many false-positive
candidate pages am I willing to live with" dial a consumer of the
classifier would actually set (the table above is that trade-off surface);
`chapter_first_recall_tolerance` only changes how *this evaluation script*
scores a book as "fully recalled" for its own aggregate report, with no
effect on inference behavior. The 0.80 default keeps candidate volume
inside the pilot's original budget; a consumer able to tolerate more
false-positive candidate pages (e.g. because a downstream stage reviews
every candidate cheaply, on institutionally-hosted models with no
per-request cost) can raise `--recall-target` for substantially higher
recall, per the curve above.

With the new default (`LogisticRegression`, `recall_target=0.80`,
`chapter_first_recall_tolerance=0.90`), the full 50-book LOBO run gives
`full_recall_fraction=44%`, `avg_candidate_fraction=15.0%` -- still **NOT
MET** against the pilot's 90%/15% bar, but nearly triple the very first
run's 16%, and the highest `full_recall_fraction` reached at or under the
original candidate-fraction budget across every follow-up so far. Per
corpus:

| corpus | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| open-access | 54.1% | 15.2% |
| copyrighted-scans | 15.4% | 14.4% |

**Would more ground truth help further?** A learning-curve check (fixed
15-book held-out test set, `LogisticRegression` retrained on random subsets
of the remaining pool at sizes 10/15/20/25/30/35 books, 5 repeats per size)
found `full_recall_fraction` on that fixed test set flat within noise
across the entire range (25.3%, 28.0%, 22.7%, 22.7%, 24.0%, 26.7% -- no
scaling trend). Combined with `LogisticRegression`'s low parameter count
(11 weights total) making it very unlikely to be data-starved, this
suggests generic corpus growth ("more books like the ones already here")
has limited further upside -- the ceiling looks like a feature/geometry
information limit rather than a training-volume limit. If more ground
truth is collected, it would be higher-leverage to specifically target the
underrepresented templates already characterized in this investigation
(unnumbered first chapters, chapter-openings with no font-size or
whitespace distinction from body text) rather than generic new books, since
those are the specific cases the current features structurally can't
separate from ordinary pages.

### Follow-up: re-run over the grown 70-book corpus

Re-run 2026-08-12 against the full grown corpus (57 `open-access/` + 13
`copyrighted-scans/`, up from 37 + 13), unchanged model/calibration
(`LogisticRegression`, `recall_target=0.80`, `chapter_first_recall_tolerance=0.90`).
This needed the `pdfalto` binary, which isn't packaged or brew-installable
-- it was built from a sibling checkout of
[kermitt2/pdfalto](https://github.com/kermitt2/pdfalto) at `../pdfalto/`
next to this repo (`evaluation/CLAUDE.md` now documents this location).
32 of the 57 `open-access/` books already had a cached ALTO XML from the
prior run; the remaining 25 new `open-access/` books plus all 13
`copyrighted-scans/` books (whose `.layout-cache/` had never been
populated in this worktree) were freshly extracted, ~2.5 minutes total.

| | full_recall_fraction | avg_candidate_fraction |
| --- | --- | --- |
| 50-book corpus (previous) | 44% | 15.0% |
| **70-book corpus (fresh)** | **64%** | **10.0%** |

Still **NOT MET** against the 90%/15% decision bar, but the largest single
jump in `full_recall_fraction` across this pilot's whole history --
achieved purely from more/corrected ground truth, no model or feature
changes -- while `avg_candidate_fraction` actually *dropped* well clear of
its budget rather than trading one for the other. Per corpus:

| corpus | books | full_recall_fraction (before -> after) | avg_candidate_fraction (before -> after) |
| --- | --- | --- | --- |
| open-access | 37 -> 57 | 54.1% -> **77.2%** | 15.2% -> **10.7%** |
| copyrighted-scans | 13 -> 13 | 15.4% -> **7.7%** | 14.4% -> **6.9%** |

**open-access drove the whole improvement.** Of the 57 books, 44 now hit
the full-recall bar and 38 score a literal 100%/100% toc/chapter_first
recall -- most of the 20 net-new books (the crossref_gt reconciliation
batch) are exactly this kind of clean, well-produced text, and the
hand-verification pass that corrected previously-wrong chapter boundaries
(see the "2026-08-12 re-run" note at the top of this file) means the
labels those 44 books are scored against are also more trustworthy than
before. The open-access stragglers are a recognizable, already-diagnosed
shape rather than something new: low `toc_recall` on TOC pages that don't
structurally resemble the training majority (e.g. `9781783748471` at 0%,
`9782375460122` at 0%) and low `chapter_first_recall` on books with weak
title/body font-size contrast (e.g. `9783492021234` at 43%,
`9783907297339` at 45%) -- both failure modes already named in the
model-architecture follow-up above.

**copyrighted-scans regressed slightly despite an unchanged book set** --
the same 13 books, none added or removed, only 2/13 pass now vs. 2/13
before by fraction (15.4% was already ~2/13; 7.7% is 1/13). This isn't a
data change on this corpus's side; the LOBO classifier trained on each
fold now draws from a much larger, open-access-dominated training pool
(56 other books instead of 36), which shifts the decision boundary these
13 harder, higher-page-width-variance books are scored against. The two
per-book swings worth flagging rather than averaging away:
`9783848704316` dropped from 73% to 27% `chapter_first_recall` (was the
biggest post-OCR-fix win in the earlier follow-up, now the biggest
regression), while `dnb-36942798X` improved from 6% to 17%. Both remain
well short of passing either way. The two newest `copyrighted-scans`
books (the hand-built Festschrift volumes, `9783428042241` and
`9783899496291`) are themselves weak: 5% and 24% `chapter_first_recall`
respectively -- consistent with this corpus's standing diagnosis (the
"Follow-up: replacing textless/degenerate-text..." section above) that
`copyrighted-scans` scans have systematically weaker layout signal than
`open-access` books, not a new finding.

Net assessment of the re-run itself: corpus growth plus ground-truth
correction bought real, substantial progress on `open-access` (54% to 77%)
at essentially no `copyrighted-scans` cost (a ~2-book swing among 13 that
were already the weak half of the corpus) -- but the pilot's own 90% bar
was still not met, and the stark per-corpus split above (`open-access`
close to the bar, `copyrighted-scans` nearly the whole shortfall) is
exactly what motivated the next follow-up's two directions (book-context/
normalized features and scan-noise augmentation), which measures against
this run as its baseline.

### Follow-up: context/normalized features and scan-noise augmentation

Per `docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md`,
direct inspection of `copyrighted-scans` ALTO XML found the cause of the
split above: OCR'd scans carry hundreds of near-identical jittery font
sizes with no genuine large title font, so the two strongest existing
features (`font_size_max_ratio`, `top_block_is_large_font`) are degenerate
on exactly the failing corpus, and the per-page `statistics.mode` used as
the body-font estimate is itself noise over that many close values. The
fix has two independent parts, both on top of the unchanged
`LogisticRegression` model and unchanged calibration
(`recall_target=0.80`, `chapter_first_recall_tolerance=0.90`): seven new
features in `evaluation/scripts/layout_features.py` --
two page-local (`last_text_vpos_fraction`, `top_line_heading_match`) and
five book/context features computed in a second pass,
`add_book_context_features()` (`prev_last_text_vpos_fraction`,
`prev_line_count_rel`, `line_count_rel`, `font_size_max_ratio_book`,
`page_position_fraction`) -- taking `FEATURE_NAMES` from 10 to 17; and a
new deterministic ALTO-level scan-noise augmentation module,
`evaluation/scripts/alto_scan_noise.py`, wired in via a `--scan-noise-augment`
flag that perturbs only `open-access` training rows (font-size jitter,
title-contrast compression, geometry jitter) so the born-digital pool
looks scan-shaped during training, with augmented rows excluded from every
fold's own test set to avoid leakage.

The three official LOBO runs, all over the full 70-book corpus at the
calibration in place when this follow-up's feature/augmentation work was
first measured (`recall_target=0.80`, the pilot's default at the time):

| run | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| baseline (10 features) | 45/70 = 64% | 10.0% |
| + context/normalized features (17) | 39/70 = 56% | 7.2% |
| + `--scan-noise-augment` | 40/70 = 57% | 7.2% |

Per corpus, all three runs:

| run | open-access `full_recall_fraction` | open-access `avg_candidate_fraction` | open-access avg `chapter_first_recall` | scans `full_recall_fraction` | scans `avg_candidate_fraction` | scans avg `chapter_first_recall` |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 44/57 = 77.2% | 10.7% | 93.9% | 1/13 = 7.7% | 6.9% | 28.8% |
| + features | 39/57 = 68.4% | 7.5% | 85.8% | 0/13 = 0% | 5.6% | 50.9% |
| + augment | 40/57 = 70.2% | 7.4% | 86.8% | 0/13 = 0% | 6.1% | 51.2% |

**Read at face value this looks like a regression against baseline, and at
`rt=0.80` it genuinely is one -- not just an appearance.** The new
features make the training probabilities more separable, so the per-fold
threshold calibrated to 80% training recall becomes noticeably more
conservative: candidate fraction fell from 10.0% to 7.2%, tight enough
that five `open-access` books flip from passing to failing purely on
`chapter_first_recall` dropping below the 90% tolerance --
`9781783748532` (100%->87%), `9781787359260` (95%->60%),
`9782821895607` (92%->42%, whose `toc_recall` also drops 100%->0%),
`9783031907272` (100%->32%), and `9783837681192` (100%->46%) -- with zero
books flipping the other way, and `full_recall_fraction` drops from 64% to
56% at that operating point. A `recall_target` sweep, run to check whether
this was purely an operating-point effect rather than the features
actually being worse, found that it was -- and changed the pilot's
default as a result:

| config | `full_recall_fraction` | `avg_candidate_fraction` | open-access `full_recall_fraction` | scans `full_recall_fraction` | scans avg `chapter_first_recall` |
| --- | --- | --- | --- | --- | --- |
| 17 features, rt=0.85 | 41/70 = 59% | 7.9% | 70.2% | 1/13 | 57.8% |
| 17 features, rt=0.90 | 47/70 = 67% | 9.0% | 78.9% | 2/13 | 66.8% |
| 17 features + augment, rt=0.90 | 45/70 = 64% | 8.9% | 77.2% | 1/13 | 64.8% |

At `rt=0.90` -- still comfortably inside the 15% candidate budget at 9.0%
-- the same 17 features beat the baseline on every axis at once: 67% vs.
64% full recall, 9.0% vs. 10.0% candidates, 78.9% vs. 77.2% open-access,
2/13 vs. 1/13 scans. Because the 17-feature model genuinely does not
improve on baseline at the *old* default, and does at this point,
`_RECALL_TARGET`'s default was moved from `0.80` to `0.90` in this
follow-up (`evaluate_layout_toc_classifier.py`) rather than leaving the
regression as the shipped behavior -- re-verified directly (not merely
computed from per-book logs) by running the script with no
`--recall-target` flag against the real 70-book corpus and confirming it
prints `67%` / `9.0%`. This mirrors the earlier model-architecture
follow-up's framing of `recall_target` as "the real ... dial a consumer of
the classifier would actually set" -- the dial needed retuning once the
model it's tuning changed underneath it. All `rt=0.80` numbers above and
in the "scan rescue" and "zero-`toc_recall`" paragraphs below describe the
comparison as it was originally measured, before this default change, and
remain accurate as a record of that measurement; they are not this
follow-up's final shipped configuration.

**The scan rescue is real at the recall level, even though it doesn't
flip any scan to "passing" at the default tolerance.** Scans' average
`chapter_first_recall` rose from 28.8% at baseline to 50.9% (17 features,
`rt=0.80`) and 66.8% (`rt=0.90`) -- the largest per-book jumps (baseline
-> +features) are `9783428042241` (5%->68%, 78% with augmentation),
`9780367439712` (8%->67%), `9783465016878` (15%->77%),
`9783848704316` (27%->73%), `9783789057366` (2%->39%), `9783899496291`
(24%->62%), and `9783322969828` (0%->29%). But **no scan clears the
90%-per-book bar at `rt=0.80`** (0/13, and the baseline's one passer,
`9783848736829`, itself drops from 100% to 61%) -- which is exactly why
the corpus-level `full_recall_fraction` for scans reads 0% despite the
underlying recall gains being large and broad-based.

**The six 0%-`toc_recall` open-access books are unchanged at 0% in all
three official `rt=0.80` runs**: `9781783748471`, `9782375460122`,
`9783837660944`, `9783839447529`, `9783839468937`, and `9783839470619`
still score zero `toc_recall` at the default calibration. Expected -- the
new features target chapter-opening detection (book-context, per-book font
normalization, heading-line text), not TOC-page layout, so they have no
mechanism to touch this failure mode. (In the informational `rt=0.90`
sweep, two of the six do pick up nonzero `toc_recall` -- `9782375460122`
at 25%/50% and `9783839468937` at 50% -- but that's the looser threshold
admitting more candidate pages that happen to include a TOC page, not the
model scoring those pages as TOC-like.) TOC-anchored matching (deferred,
see below) remains the untouched direction for these six.

**Augmentation itself is an honest, mostly-negative result.** Against the
un-augmented 17-feature run, `--scan-noise-augment` moves the overall
number by +1 book at `rt=0.80` (56%->57%) and -2 books at `rt=0.90`
(67%->64%), and scans' average `chapter_first_recall` by only +0.3pt at
`rt=0.80` (50.9%->51.2%). A few scans improve slightly with it
(`9783428042241` 68%->78%, `9783492021234` 43%->50%); others drop
(`9780367439712` 67%->58%, `9783465016878` 77%->69%, and
`9783848736829`'s `toc_recall` 67%->33%). The most likely reason: the
per-book font normalization (`font_size_max_ratio_book`) already closes
most of the domain gap the augmentation was designed to target, leaving
little for the synthetic scan-noise to add on top. It stays in the
codebase -- cheap to run, cached (`.aug.alto.xml`, regenerated only when
deleted by hand), off by default -- but should not be relied on as the
mechanism behind the scan-recall gains above; the feature work is.

**Verdict: still NOT MET, in every configuration measured**, against the
unchanged 90%/15% bar. The shipped configuration -- 17 features,
`recall_target=0.90` (now the default) -- reaches 67% / 9.0%, still 23
points short, but is a genuine improvement over the 10-feature baseline on
both axes at once (64%/10.0%), not merely a differently-measured version
of it. Two structural gaps remain untouched by this follow-up: the
open-access TOC-layout failure mode (candidate direction: TOC-anchored
chapter matching, parsing a detected TOC page and locating chapter
openings by title/page-number matching against it) and the residual scan
`chapter_first_recall` ceiling (candidate direction: document-image deep
learning, bypassing OCR font metadata entirely). Both were deferred, not
rejected, at the design stage; neither is scoped here. Candidate volume
still sits well below the 15% budget at the new default (9.0%), so a
further `recall_target` increase remains available if the next follow-up
wants to trade more candidate-page volume for recall.

### Follow-up: re-run over the grown copyrighted-scans corpus (32 books)

`copyrighted-scans/` grew from 13 to 32 books (19 new), targeting exactly
the hard cases called out above -- scans, weak title/body contrast,
unnumbered first chapters. Re-running the shipped configuration (17
features, `recall_target=0.90`) with no other changes, over all 89 books
now available (57 open-access + 32 scans, up from 70):

| corpus | `full_recall_fraction` | `avg_candidate_fraction` | avg `chapter_first_recall` |
| --- | --- | --- | --- |
| all (89 books) | 57/89 = 64% | 10.0% | 88.3% |
| open-access (57, unchanged) | 44/57 = 77.2% | 10.0% | 95.3% |
| copyrighted-scans (32, was 13) | 13/32 = 40.6% | 10.0% | 75.9% |

**Read at face value against the previous 70-book snapshot (67%/9.0%),
this looks like a regression -- it is not one.** It's a corpus-composition
effect: the 19 new books were deliberately acquired as the classifier's
known-hardest case (scans with weak/degenerate title fonts), so folding
them in shifts the overall average toward a harder mix, not toward a
worse model. Isolating the original 13 scans books and re-scoring them
under the new run (more/different training data per LOBO fold, since
every other book's rows are now part of each fold's training set) shows
scans performance actually **improved**: 2/13 (15.4%) full-recall passes
before this corpus growth vs. 4/13 (30.8%) now on the same 13 books, and
13/32 (40.6%) across the full grown scans set -- consistent with the
75.9% avg `chapter_first_recall`, up from 66.8% at the previous snapshot.
Open-access held steady (78.9% -> 77.2%, one book's `chapter_first_recall`
crossing the 90% tolerance in the other direction due to the shifted
training mix -- not a targeted regression).

Of the 19 new scans books, 9 pass the per-book full-recall bar outright
and 10 fail it -- expected, since these were picked specifically to
stress-test the classifier's weak spots rather than to be easy wins:
`9783161538315`, `9783166448978`, `9783428038275`, `9783472611097`,
`9783531120553`, `9783789017483`, `9783830502777`, `9783845271897`, and
`9788814022272` fail on `chapter_first_recall` (typically 0-77%, well
under the 90% tolerance); `9780521650939`, `9781841136400`,
`9781849463812`, `9783406016127`, `9783571092124`, `9783658057022`,
`9783658076078`, `9783658282103`, and `9783810041449` pass outright with
no changes to the model or features.

One new book, `9783406016127`, is an outlier worth flagging rather than
averaging away: it has no `toc` ground truth (`toc_recall=n/a`) and a
60.9% `candidate_fraction` -- six times the corpus average -- meaning the
classifier is flagging most of the book as candidate pages. This single
book adds roughly 0.6 points to the corpus-wide `avg_candidate_fraction`
on its own; worth a closer look (layout/OCR quality, or a ground-truth
issue) before it's used to justify any future calibration change.

**Verdict: still NOT MET** (89-book `full_recall_fraction` 64%, 26 points
short; `avg_candidate_fraction` 10.0%, comfortably inside the 15% budget).
No code or calibration changed in this follow-up -- it's a pure
re-evaluation against new ground truth. The new books confirm rather than
overturn the previous follow-up's read: the context/normalized features
and `recall_target=0.90` genuinely help on scans in aggregate (avg
`chapter_first_recall` keeps climbing), but per-book pass/fail is still
dominated by the residual scan `chapter_first_recall` ceiling that
follow-up already identified as untouched -- document-image deep learning
remains the candidate direction, still not scoped here.

### Follow-up: `edge_distance` feature, replacing `page_position_fraction`

Motivated by a specific failure: `9782821895607` is a French-language book
whose TOC sits at pages 189-190 of 193 -- right at the very *back* of the
book, not the front. The classifier had `toc_recall=0%` on it in every
prior run, with no feature encoding "how far from either edge" -- only the
2026-08-12 follow-up's `page_position_fraction` (a raw `page_index /
total_pages` fraction), which is monotonic and therefore can only reward
one direction (front *or* back, never both).

Checked against ground truth across 86 books with known TOC bounds:
`corr(total_pages, toc_start_index) ~ -0.09` (front-matter length before a
TOC starts doesn't scale with book length) but `corr(total_pages,
toc_len) ~ +0.60` (longer books do have longer TOCs, though modestly in
absolute terms here: mean 2.8 pages, max 9). This favors an *absolute*
page-count distance feature over a fractional one.

Added `edge_distance = min(page_index, total_pages - 1 - page_index)` to
`add_book_context_features` (`evaluation/scripts/layout_features.py`) --
0 at either edge, rising toward the middle. Initial attempt added it
*alongside* `page_position_fraction` and left `9782821895607` unchanged
(`toc_recall` still 0%): diagnosing the trained fold directly showed
`page_position_fraction`'s coefficient (~-9.4, learned from a training set
almost entirely front-located TOCs) canceling out `edge_distance`'s
correctly-signed coefficient (~-10.4) for this page. Removing
`page_position_fraction` and keeping only `edge_distance` fixed it outright
(that book's two TOC pages went from `prob=0.0` to `prob=0.9998`, both
clearing threshold).

Re-running LOBO with `page_position_fraction` replaced by `edge_distance`
(16 features total, no other changes) over all 89 books:

| | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| before (17 features, incl. `page_position_fraction`) | 57/89 = 64.0% | 10.0% |
| after (16 features, `edge_distance` instead) | 58/89 = 65.2% | 10.3% |

`9782821895607` now passes `toc_recall` outright (0% -> 100%), and
`9783166448978` (one of the original hard scans) also improved on `toc`
(0% -> 50%), though its `chapter_first_recall` (47%) keeps it failing the
per-book bar regardless. No book that previously found at least one TOC
page regressed to zero. Net effect: a small, real full-recall gain (+1.2
points) at a small candidate-volume cost (+0.3 points) -- consistent with
the target fix being real rather than noise, but modest, since it only
addresses one specific failure mode (edge-vs-middle position) among
several still-unaddressed ones (OCR/scan-noise degradation, weak
title/body contrast).

**Verdict: still NOT MET** (65.2% full recall, 24.8 points short; 10.3%
avg candidate fraction, still comfortably inside the 15% budget). This
follow-up doesn't change that overall picture -- it's a targeted fix for
one identified false-negative mechanism (mid-book in-chapter mini-TOCs and
back-of-book TOCs sharing a layout signature with real front-matter TOCs),
not a step toward a fundamentally different recall ceiling.

### Follow-up: isolating the open-access corpus -- recall_target retuned, per-feature weighting tested and ruled out

Motivated by a direct question: why does this pilot score so far below its
own 90%/15% bar on `open-access` alone, given that corpus's clean, native,
well-produced layout should carry the strongest geometric signal of any
corpus here -- and would manually reweighting the model's input features
help close the gap? Both questions were tested empirically, restricting
every run in this section to `--corpora open-access`
(`load_book_corpus`'s LOBO training pool then contains *only* open-access
rows -- not just excluded from the reported metric, excluded from training
too, unlike the "concentrate on open-access first" follow-up above, which
kept `copyrighted-scans` in the training pool throughout):

```bash
uv run python evaluation/scripts/evaluate_layout_toc_classifier.py \
  --pdfalto-bin ../pdfalto/pdfalto --corpora open-access --recall-target 0.988
```

At the shipped defaults (17 features, `recall_target=0.90`, unchanged
model): **72% `full_recall_fraction`, 8.0% `avg_candidate_fraction`**
across the 57 open-access books -- still NOT MET, but with less than half
the 15% candidate budget spent, unlike the mixed 89-book run (10.0%) or
`copyrighted-scans`, where candidate-fraction budget is the binding
constraint.

**Per-feature weighting was tested directly and ruled out as the fix.**
Averaging |standardized coefficient| across several LOBO folds' fitted
models found one feature, `edge_distance`, at ~15 for the `toc` label --
three times the next-highest feature (`line_count`, ~5.3) and roughly the
size of the fitted intercept (~-19) with the opposite sign. That looks
exactly like textbook quasi-complete-separation overfitting (real TOC
pages in this corpus sit almost perfectly at the extreme low end of
`edge_distance`, so an under-regularized linear model drives its
coefficient toward the largest value the L2 penalty allows, to squeeze out
training likelihood). The natural fix hypothesis -- strengthen L2
regularization (`LogisticRegression`'s `C`, sklearn default `1.0`) to
shrink this coefficient toward a more modest, better-generalizing value --
was swept from `C=100` down to `C=0.003` and produced **zero change** in
`full_recall_fraction` or in any individual book's `toc_recall`, over three
orders of magnitude. Root cause: `select_threshold` recalibrates the
accept/reject threshold to each fold's own fitted training-probability
distribution, so uniform L2 shrinkage rescales every coefficient (and the
resulting probabilities) together without changing pages' *relative*
ranking within a book -- and it's the ranking, not the raw probability
scale, that decides which pages clear the bar. Only far more extreme
regularization (`C` below roughly `0.0003`, i.e. multiple further orders of
magnitude past any value a normal hyperparameter search would try) started
changing outcomes at all, and even there it flipped only 2 of the 5
originally-zero-`toc_recall` books to 100% while leaving overall
`full_recall_fraction` at 73.7% -- a much weaker, less comprehensive fix
than the one below, reached only by pushing regularization into a regime
that behaves like a blunt "flag nearly everything" fallback (similar in
character to `recall_target -> 1.0`'s own cliff, documented in the first
follow-up in this section). **Conclusion: no realistic amount of feature
reweighting (via regularization or, by the same logic, via hand-assigned
per-feature priors) is the lever that closes this corpus's gap** -- the
features and their learned relative importance are not the bottleneck.

**`recall_target` -- the pilot's existing, already-exposed calibration
knob -- is the lever that works.** Unlike `C`, it does not scale
coefficients uniformly; it moves the accept threshold non-uniformly along
each fold's own training-positive probability distribution, which is
exactly where this corpus had slack. A sweep restricted to open-access
alone:

| `recall_target` | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| 0.80 | 52.6% | 6.6% |
| 0.85 | 64.9% | 7.1% |
| 0.90 (shipped default) | 71.9% | 8.0% |
| 0.93 | 75.4% | 8.7% |
| 0.95 | 82.5% | 9.2% |
| 0.97 | 84.2% | 10.2% |
| 0.98 | 87.7% | 10.7% |
| 0.987 | 93.0% | 12.7% |
| **0.988** | **94.7%** | **13.8%** |
| 0.99 | 94.7% | 15.2% |
| 1.00 | 98.2% | 58.4% |

**At `recall_target=0.988`, open-access alone clears the pilot's own
90%/15% decision bar for the first time in this pilot's whole history** --
94.7% full recall at 13.8% candidate fraction, both sides of the bar
satisfied with real margin (4.7 points of recall headroom, 1.2 points of
candidate headroom). The crossing is still narrow, not a wide plateau like
the earlier corpus-wide 0.97-0.99 plateau: `0.99` itself -- one thousandth
higher -- already exceeds the 15% cap (94.7%/15.2%, same recall, more
candidates), so this operating point sits close to a cliff edge and should
be expected to move if the open-access corpus grows further, unlike the
earlier, comfortably-wide plateaus found in prior follow-ups. (These
numbers reflect the ground-truth fix below -- rerunning the identical
sweep before that fix put the crossing a full point higher, at
`recall_target=0.99`, with a tighter margin.)

Only 3 of 57 books still fail at `recall_target=0.988`, all the same,
already-diagnosed structural gap:

| book | `chapter_first_recall` | root cause |
| --- | --- | --- |
| `9781783749478` | 89% | Weak title/body font-size contrast (already-diagnosed structural gap) |
| `9783837681192` | 85% | Same |
| `9783907297285` | 77% | Same |

A fourth book, `9783837660944`, originally failed here too
(`toc_recall=0%`) -- but this traced to a **ground-truth defect, not a
classifier gap, and has been fixed** rather than left as a residual. Its
`.expected.json` listed a chapter titled `"Inhalt"` (German for "table of
contents") spanning `pdf_start_index=5, pdf_end_index=6` -- the *exact same
page range* as its own `"toc": {"toc_start_index": 5, "toc_end_index": 6}`
field, credited to the same four names as the book's real `"Vorwort"`
chapter (its editors). `layout_labels.page_labels()` applies `chapters`
after `toc`, so this entry overwrote page 5's true `toc` label with
`chapter_first`, leaving only page 6 as a real toc-positive test row
(confirmed: `n_true_pages=1` for this book pre-fix, not the true 2) and
injecting a spurious `chapter_first` positive at a page that is not a real
chapter opening. Per `evaluation/CLAUDE.md`'s own transcription workflow, a
front-matter TOC section is supposed to be recorded as `{"skip": true}`
during Step 1 specifically so it never lands in the final `chapters`
list -- this was exactly that step being missed when this book's ground
truth was built. **Fixed** by deleting the spurious `"Inhalt"` entry from
`evaluation/corpus/open-access/9783837660944.expected.json`; re-verified
with the Step-4 bounds/overlap check, `tests/test_ground_truth_integrity.py`
(89/89 subtests still pass), and `tests/test_segmentation_accuracy.py`
(this book's heuristic-pipeline recall is unaffected -- still clears the
hard `recall > 0` regression gate, at 0.40 -- so the fix is isolated to
this pilot's own labels, not a change to the main harness's behavior for
this book). Post-fix, this book scores `toc_recall=50%` (page 6 now
correctly recognized) and `chapter_first_recall=100%` (the spurious
positive is gone) at `recall_target=0.988` -- a full pass. A structurally
identical overlap exists in `9781783743339` (a `"Preface"` chapter
starting on the toc's own first page) but was left alone there, since that
book's remaining toc page already satisfies the lenient "≥1 hit" bar on
its own, so the overlap costs it nothing.

Two further partial `toc_recall` cases were checked and are *not* the same
bug, and don't fail the aggregate bar either (its "≥1 hit" pass condition
is lenient): `9782375460122` (25%, a 4-page TOC) and `9783031538391` (50%,
also 4 pages) have no chapter/toc range overlap. Multi-page TOCs aren't
rare in this corpus (12 of 57 books have a 4-page TOC, most scoring
cleanly), so length alone doesn't explain either one -- both are left as
an open, lower-priority layout-diversity gap.

**The mini-TOC hypothesis -- the specific exception this investigation was
framed around -- was checked directly and ruled out for every one of the
originally-failing books.** All 5 books that started this investigation at
`toc_recall=0%` have their real TOC at pages 5-8, the extreme front of the
book, and `edge_distance` (added in the previous follow-up specifically to
separate front/back real TOCs from mid-book mini-TOCs) scores it correctly
in every one of them -- it is consistently that page's *single largest*
positive contributor toward the `toc` label (see the coefficient
discussion above). Pages are also scored independently, with no book-level
"top-k" competition, so a mid-book mini-TOC page cannot suppress or
outrank a genuine front-matter TOC's own probability even in principle.
Every one of those 5 traces to something else: one ground-truth defect
(now fixed), three already-diagnosed weak-font-contrast chapter openings,
and one unexplained-but-structurally-unremarkable TOC layout (the other,
`9781783748471`, is fully fixed by the recalibration alone).

**Verdict: MET for open-access, in isolation, at `recall_target=0.988`.**
This is the first configuration in this pilot's whole history to clear its
own 90%/15% bar. Not proposed as a change to the shipped global default
(`_RECALL_TARGET=0.90`): the "recall-target tuning" follow-up above already
found `copyrighted-scans`' candidate-fraction cost rises faster with
`recall_target` than open-access's does, so applying `0.988` to the
mixed/scans corpus would blow the 15% cap rather than approach it. Whether
a per-corpus (or otherwise input-adaptive) `recall_target` is worth
formalizing as a product decision, versus keeping one global knob and
accepting weaker open-access recall as its cost, is taken up in the next
follow-up.

### Follow-up: is `recall_target` worth splitting by source type? And a rejected feature

Two follow-on questions from the previous section, tested empirically
rather than answered speculatively.

**Does `extraction_type` (native vs. scan) predict how much `recall_target`
headroom a book has, well enough to calibrate separately by type?** First
finding: `extraction_type` is *not* the same axis as corpus name --
`copyrighted-scans/`'s 32 books split 18 native / 14 scan (only the latter
are actually scanned; the corpus name predates the manifest field and
groups by provenance, not by extraction method). So "native vs. scan"
and "open-access vs. copyrighted-scans" are different, overlapping splits,
and the more precise one is extraction_type.

A LOBO run over the full 89-book corpus (one shared model per fold, as
always -- only the *threshold* is calibrated separately) with the
per-fold threshold for label L computed twice -- once from that fold's
native-book training rows at `recall_target_native`, once from its
scan-book training rows at `recall_target_scan` -- then applying whichever
threshold matches the held-out book's own type:

| config | overall full_recall | overall avg_cand | native full_recall / avg_cand | scan full_recall / avg_cand |
| --- | --- | --- | --- | --- |
| single `recall_target=0.94` (closest single-knob budget match) | 73.0% | 13.9% | 74.7% / 10.8% | 64.3% / 30.4% |
| split: `native=0.97, scan=0.75` | **82.0%** | **14.5%** | 89.3% / 14.8% | 42.9% / 12.7% |

At essentially the same overall candidate budget, splitting by
`extraction_type` buys **+9 points of overall full_recall_fraction** --
a real, reproducible win, and cheap (no model or feature change, just two
constants instead of one). The mechanism is reallocation, not a new
capability: native books can absorb a much more aggressive threshold than
one global knob calibrated for the whole mixed pool would ever allow them,
without spending scan books' share of the 15% budget on a target that
barely moves their recall anyway. That's also the split's limit --
**scan's own `full_recall_fraction` stays stuck around 43% across every
`recall_target_scan` tested (0.75-0.85 all gave the identical 42.9%,
suspiciously flat -- likely `select_threshold`'s `ceil()` quantile snapping
to the same few discrete thresholds with only 14 scan books' worth of
positive rows to calibrate against)**, consistent with every earlier
follow-up's finding that scans' *underlying* signal, not calibration, is
the ceiling. **Recommendation: yes, split by `extraction_type` if this
pilot ships -- it's a strictly-better use of the same candidate-fraction
budget than one global constant -- but budget for scans staying well under
open-access's recall regardless of how its own target is tuned.** Not
implemented in `evaluate_layout_toc_classifier.py` here, since the pilot
overall is still NOT MET and this is a calibration recommendation for
*if*/when it ships, not a standalone change.

**Would a per-document "retry with different targets" work instead of a
fixed type-based bucket?** Worth separating two different ideas that
phrase could mean. Retrying at *evaluation* time (as this whole
investigation already does, repeatedly) only works because LOBO has
ground truth for the held-out book to score each retry against -- that
signal doesn't exist for a real, unlabeled document in production, so
there's no way to "try `recall_target=0.99`, check the recall, try
`0.95`, check again" against a book with no known chapters. What
*does* carry over to a real, unlabeled document is the other half of what
this whole section has been trading against: not recall, but
**candidate_fraction**, which is directly observable for any document
(count how many of its own pages clear a given threshold) with no ground
truth needed at all. A more principled version of "retry with different
targets" is therefore document-relative rather than type-bucketed: score
every page of the new document, then pick the *lowest* probability
threshold such that this specific document's own resulting
candidate_fraction stays under a fixed cap (e.g. 15%) -- equivalent to
always taking the top-N-percent highest-scoring pages of that document,
rather than a threshold calibrated once on training data and applied
uniformly. This should automatically behave like the type split above
without needing to know or detect extraction_type at all: an easy,
high-contrast book's probabilities are naturally more separated, so the
same candidate-fraction cap admits a much higher effective recall target
for it than for a noisy book where many pages sit near the decision
boundary. This is a different selection strategy, not just a different
constant -- it would replace `select_threshold`'s recall-target semantics
with a candidate-budget-target one -- so it's a real design change,
untested here, not a same-session tweak; flagging it as the more promising
direction if this axis gets picked up again, rather than further
type-bucket tuning.

**A new feature was tried and rejected: `early_gap_ratio`.** Motivated by
the three books still failing `chapter_first_recall` at
`recall_target=0.988` (`9781783749478`, `9783837681192`, `9783907297285`)
being labeled "weak font-size contrast" without that actually being
re-verified for each -- checking directly found `9783907297285`'s misses
are dominated by `left_margin_var`/`left_margin_mean` contributions (large,
irregular left-indentation, not font size), so that label was too coarse.
Hypothesis: a chapter heading can lack font-size contrast with body text
and still leave a visually larger vertical *gap* before the body starts
(extra leading/whitespace after a heading line, same font). Prototyped
directly against cached ALTO XML (not yet wired into the shipped feature
set): for each page, the ratio of the largest gap among its first 8
line-to-line transitions to its own median line-to-line gap. Checked in
isolation against the three target books' true `chapter_first` pages vs. a
random sample of other pages: real separation for 2 of 3
(`9781783749478`: median ratio 14.8 vs. 9.2; `9783907297285`: 12.7 vs.
4.7) but not the third (`9783837681192`: 7.2 vs. 6.2, confirming that
book's issue is something else, consistent with the margin-variance
finding above).

Despite the promising isolated signal, wiring it into
`layout_features.py` as an 18th feature (`early_gap_ratio`, added to
`PAGE_FEATURE_NAMES`) and re-running the full LOBO suite made things
*worse*, not better, at the operating point that matters: open-access
alone at `recall_target=0.988` dropped from 94.7%/13.8% to
93.0%/13.5% -- one point of avg_candidate_fraction improvement bought by
losing a previously-passing book (`9783839446270`'s
`chapter_first_recall` fell from 91% to 73%) for no gain on either of the
two target books it was meant to help (`9783837681192` and
`9783907297285` both scored identically with or without it). The full
89-book corpus at the shipped `recall_target=0.90` default moved
+1 point (64%->65% full_recall, 10.4%->10.3% avg_cand) -- within
noise, not a real signal either way. **Reverted** (net negative at the
one operating point it was targeted at, neutral everywhere else) -- the
same "looked promising in isolation, hurt in the full LOBO context"
pattern the model-architecture follow-up already found once with
`max_font_vpos_fraction`. Isolated single-feature separation checks on a
handful of hand-picked books are not a reliable predictor of a feature's
effect once it's added to a 17-way logistic regression trained across an
89-book pool; only a full LOBO re-run answers that.
