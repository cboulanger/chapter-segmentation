# Layout-based TOC/chapter-first-page classifier: context features, per-book normalization, and ALTO-level scan-noise augmentation

Status: approved for planning
Date: 2026-08-12

## Problem

The 2026-08-12 re-run of the layout classifier pilot over the grown
70-book corpus (see `evaluation/RESULTS.md`, "Follow-up: re-run over the
grown 70-book corpus") came back **NOT MET**: 64% `full_recall_fraction`
(bar: ≥90%) at a healthy 10.0% `avg_candidate_fraction` (bar: ≤15%). The
per-book results split into two nearly disjoint failure modes:

**Open-access fails on TOC pages, not chapter pages.** Six `open-access`
books score 0% `toc_recall` while most simultaneously hit 90–100%
`chapter_first_recall` (e.g. `9783839470619` at 0%/100%). These are
atypical TOC layouts underrepresented in the training pool.

**Copyrighted-scans fail on chapter-first pages — and the cause is now
measured, not hypothesized.** Nearly all 13 `copyrighted-scans` books sit
at 0–46% `chapter_first_recall`. Direct inspection of their cached ALTO
XML shows the OCR text layer carries hundreds of near-identical jittery
font sizes (7.189, 7.191, 7.429, 7.431, …) with no genuine large title
fonts, where born-digital `open-access` books have clean discrete styles
(9pt body, 20–26pt titles). Two consequences for
`evaluation/scripts/layout_features.py`:

- `font_size_max_ratio` and `top_block_is_large_font` — the two title-block
  signals, previously shown to be the strongest chapter-opening features —
  are degenerate on exactly the failing corpus: there is no large max font
  to find.
- The per-page `statistics.mode` used as the body-font estimate is itself
  arbitrary over jittery continuous sizes, so even the ratio's denominator
  is noise on scans.

Two further constraints from earlier follow-ups shape the fix:

- **The pipeline is data-limited, not capacity-limited.** A learning-curve
  check found `full_recall_fraction` flat across training-pool sizes
  10–35 books, and `LogisticRegression` (11 weights) beat both tree
  ensembles. More model capacity on the same features is not the lever;
  neither is generic corpus growth ("more books like the ones already
  here").
- **Cross-book distribution shift is real and measurable.** Growing the
  training pool by 20 `open-access` books moved `copyrighted-scans`
  scores despite zero data change on that corpus (`9783848704316`:
  73% → 27% `chapter_first_recall`), showing the decision boundary is
  sensitive to training-pool composition in a way per-fold
  `StandardScaler` does not absorb.

An options review with the user (model architecture including deep
learning, feature work, more/synthetic data, system-level reframings)
selected two directions to pursue now: **feature work** (sequence
context, per-book normalization, a light text feature) and **targeted
data via ALTO-level scan-noise augmentation** (chosen over image-level
render-degrade-re-OCR for cost and determinism). TOC-anchored chapter
matching and document-image deep learning were explicitly deferred, not
rejected — see "Out of scope".

## Scope

### 1. Two new page-local features (`layout_features.py`)

- `last_text_vpos_fraction` — bottom edge of the lowest text line
  (`max(VPOS + HEIGHT)` over lines) divided by page height. Base value
  for the sequence features below, and a weak direct signal (chapter-end
  pages end short).
- `top_line_heading_match` — 1.0 if the text of the highest-VPOS line
  matches a heading pattern, else 0.0. The pattern covers, case-insensitively:
  a heading keyword (`chapter`, `kapitel`, `chapitre`, `part`, `teil`,
  `partie`, `§`) optionally followed by a number; a bare arabic number;
  or a bare roman numeral — reusing the module's existing validated
  roman-numeral grammar (`_TRAILING_NUMERAL_RE`'s roman branch), which
  already rejects lookalike words ("mix", "did", "civic"). Content-based,
  so it survives OCR where font metadata does not.

### 2. Second-pass book/context features (`layout_features.py`)

`extract_page_features()` stays page-local but additionally emits two raw
intermediate values per page, `_max_font_size` and `_modal_font_size`
(underscore-prefixed, **not** added to `FEATURE_NAMES`, and stripped by
the second pass so they never reach the model as-is).

A new function `add_book_context_features(page_features, total_pages)`
consumes the per-page dict, computes book-level aggregates, and returns
final feature vectors containing five new features:

- `prev_last_text_vpos_fraction` — the previous page's
  `last_text_vpos_fraction`. Low value = previous page ended short or
  blank, the classic chapter-boundary precursor. Page 0 uses 0.0 (a
  "previous page empty" front-matter prior).
- `prev_line_count_rel` — previous page's `line_count` divided by the
  book-median `line_count`; 0.0 for page 0.
- `line_count_rel` — this page's `line_count` / book-median `line_count`.
  Book medians are computed over non-empty pages only (pages with at
  least one text line); if a book has no non-empty pages the divisor
  falls back to 1.0.
- `font_size_max_ratio_book` — this page's `_max_font_size` divided by a
  book-level body-font estimate: the median of per-page
  `_modal_font_size` over non-empty pages. A median over hundreds of
  jittery OCR sizes is stable where a per-page mode is arbitrary — this
  is the feature that rescues the dead font signal on scans. Pages with
  no resolvable font (raw values absent/zero) get 1.0, matching the
  existing `font_size_max_ratio` default.
- `page_position_fraction` — page index / total pages (TOC near the
  front, chapters spread through the body).

`FEATURE_NAMES` grows from 10 to 17.
`evaluate_layout_toc_classifier.py`'s `build_feature_table()` calls
`add_book_context_features` after `extract_page_features` — one call
site; no change to the LOBO/threshold machinery.

Considered and dropped: a recto/verso parity feature. Physical-index
parity maps onto recto differently per book (cover pages and front
matter shift it), so a single global `LogisticRegression` weight cannot
use it; it would be pure noise across books.

### 3. ALTO-level scan-noise augmentation (`evaluation/scripts/alto_scan_noise.py`)

New module with one public function: given a source ALTO XML path, an
output path, and a book key, write a perturbed copy. All randomness comes
from a `random.Random` seeded deterministically from the book key, so
augmented output is reproducible and cacheable. Three perturbations,
each mimicking a property measured in the real `copyrighted-scans` ALTO:

- **Font-size jitter**: each `TextStyle` in the Styles block is split
  into several clones with `FONTSIZE` multiplied by a per-clone factor
  drawn from ~U(0.96, 1.04); each `TextLine`'s `STYLEREFS` is reassigned
  to a randomly chosen clone of its original style. Reproduces the
  "hundreds of near-identical sizes" pattern that breaks per-page modal
  font estimation.
- **Title-contrast compression**: all sizes pulled toward the document's
  body size (`new = body + (old − body) · α`, with α drawn once per book
  from ~U(0.3, 0.7); body = modal `FONTSIZE` across the document).
  Reproduces the missing large title fonts.
- **Geometry jitter**: small multiplicative noise on each line's
  `HPOS`/`VPOS`/`WIDTH` plus a per-page global offset, simulating
  crooked/offset scans.

Exact constants above are starting points, tunable during implementation;
the invariants (deterministic per key; page and line counts preserved;
contrast strictly compressed) are the contract.

Augmented ALTO is cached as `.layout-cache/<key>.aug.alto.xml` next to
the source cache entry and regenerated only if absent — delete the file
to regenerate after a parameter change (same manual-invalidation
convention as the existing `.layout-cache`).

### 4. Evaluation-script integration and the leakage rule

`evaluate_layout_toc_classifier.py` gains a `--scan-noise-augment` flag
(default off, so the baseline stays reproducible). When set:

- Only `open-access` books are augmented — the point is to make the
  born-digital training pool look scan-like; `copyrighted-scans` books
  are never augmented.
- Each augmented book contributes rows with the **original** `book_key`
  plus an `augmented: True` marker, labeled identically to the source
  book (the perturbation never moves content across pages).
- LOBO fold rule: training rows = all rows (augmented or not) whose
  `book_key` differs from the held-out book; test rows = the held-out
  book's **non-augmented** rows only. No fold ever trains on any variant
  of its test book, and augmented pages are never themselves evaluated.

### 5. Tests (TDD, `tests/`)

- Heading-pattern cases in the `top_line_heading_match` regex: keyword
  forms in all three languages, bare numbers, roman numerals, and the
  roman-lookalike rejections ("mix", "did") the existing grammar already
  guards.
- `add_book_context_features` on synthetic page dicts: prev-page wiring,
  page-0 sentinels, book-median computation (non-empty pages only,
  empty-book fallback), raw-key stripping, and `FEATURE_NAMES`
  completeness of the output vectors.
- `alto_scan_noise` on a small ALTO fixture: same seed → identical
  output; page/line/String counts preserved; font-size contrast
  measurably compressed (max/body ratio strictly decreases); geometry
  perturbations within bounds.
- Fold-rule test: with augmented rows present, the held-out book's
  augmented rows appear in no fold's test set and only in other books'
  training sets.

The evaluation script itself stays a manual run (not part of
`uv run pytest`), per existing convention.

### 6. Re-run protocol and reporting

Three LOBO runs over the full 70-book corpus, unchanged calibration
(`recall_target=0.80`, `chapter_first_recall_tolerance=0.90`):

1. Baseline — already recorded (64% / 10.0%).
2. New features, no augmentation.
3. New features + `--scan-noise-augment`.

Report in `evaluation/RESULTS.md` as a new follow-up subsection under the
pilot section, matching the existing prose-plus-tables style: the three
runs' `full_recall_fraction` / `avg_candidate_fraction`, the per-corpus
breakdown for each, and per-book callouts for the previously diagnosed
failure books (the six 0%-`toc_recall` open-access books and the
`copyrighted-scans` set). If still NOT MET, name the deferred directions
(TOC-anchored matching, document-image deep learning) as the next places
to look, without scoping them.

### 7. Targeted-acquisition note (`evaluation/CLAUDE.md`)

Add a short note to the corpus-growth workflow: when adding books, prefer
scans, books with unnumbered first chapters, and books with weak
title/body font contrast — the learning curve shows generic well-produced
open-access books are saturated for this classifier.

## Decision criteria

The RESULTS.md write-up of all three runs is the deliverable either way
— same tempering stance as every prior follow-up. Directional success,
distinct from the pilot's unchanged 90%/15% bar:

- `copyrighted-scans` `full_recall_fraction` and per-book
  `chapter_first_recall` improve over baseline (run 3 vs. run 1 for the
  augmentation claim; run 2 vs. run 1 for the feature claim).
- `open-access` does not regress materially.
- `avg_candidate_fraction` stays ≤15%.

A feature-only improvement with augmentation adding nothing (or vice
versa) is a useful isolated finding, which is why runs 2 and 3 are
separate rather than bundled.

## Out of scope

- **TOC-anchored chapter matching** (parsing detected TOC pages and
  locating chapter openings by title/page-number matching) — potentially
  the highest ceiling, but a system-level reframing that changes what
  the classifier needs to achieve; deferred to its own design.
- **Document-image deep learning** (DiT/LayoutLM-class fine-tuning) —
  the only option that bypasses degenerate OCR font metadata entirely,
  held in reserve if scans remain stuck after this work; heavy
  dependency/GPU cost not justified before the cheap levers are
  exhausted.
- **Image-level augmentation** (render → degrade → re-OCR → re-extract)
  — rejected for now in favor of ALTO-level perturbation: hours vs.
  seconds per corpus, new OCR tooling in the loop, and non-determinism
  across OCR versions.
- **Recall-target or tolerance retuning** — both stay at their current
  defaults so runs 2 and 3 are comparable to the recorded baseline.
- **Production wiring / `TocExtractionStrategy` integration** — still
  gated on a MET result or a deliberate decision to proceed anyway,
  unchanged from the original pilot spec.
- **New ground-truth acquisition itself** — section 7 documents *what*
  to acquire; actually sourcing books stays the existing manual
  `evaluation/CLAUDE.md` workflow.
