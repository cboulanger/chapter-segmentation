# Experiments

Snapshots of in-progress or experimental work that investigates candidate
improvements to chapter-segmentation accuracy or to the evaluation
pipeline itself. None of these are part of the main
`chapter_segmentation` workflow, and none affect the numbers in
`evaluation/RESULTS.md` unless a write-up says otherwise. Each file
carries a "Current status" section that's kept up to date and a "History"
section that keeps every superseded run and dead end, so the reasoning
behind the current numbers isn't lost.

- **layout-based TOC classifier pilot** -- a classifier trained purely on
  page-layout geometry (no text content), meant to pre-filter
  table-of-contents and chapter-opening pages, never cleared its own
  decision bar after eleven follow-up investigations and has moved to its
  own standalone repository,
  [toc-page-classifier](https://github.com/cboulanger/toc-page-classifier),
  which now approaches the same problem with a different methodology
  (DNB-scan-sourced ground truth, text + layout features, not
  layout-only). See that repo's
  `docs/history/chapter-segmentation-layout-classifier-pilot.md` for a
  condensed history of this pilot's findings.
- **NuExtract-2.0-4B fine-tuning pilot** -- whether a small,
  locally-runnable extraction model (`NuExtract-2.0-4B`) can extract a
  book's table of contents directly, as a candidate alternative to the
  LLM-strategy row in `evaluation/RESULTS.md`. A zero-shot baseline plus a
  LoRA fine-tuning pilot showed a promising but genuinely noisy signal
  (aggregate f1 0.52 -> 0.59 on an 11-book held-out split, propped up by
  big per-book wins alongside a new decoding-time repetition failure mode
  on two others) and never reached a clean go/no-go decision before a
  newer model, NuExtract3, superseded it as the more promising target.
  All further NuExtract fine-tuning work (including this pilot's own
  history) has moved to its own standalone repository,
  [finetune-nuextract-toc](https://github.com/cboulanger/finetune-nuextract-toc),
  which now fine-tunes NuExtract3 instead. See that repo's
  `docs/history/chapter-segmentation-nuextract2-finetuning-pilot.md` for a
  condensed history of this pilot's findings.
- **dnb-toc-ground-truth** -- the LLM-gated ground-truth-generation
  pipeline for DNB table-of-contents scans has moved to its own
  standalone repository,
  [dnb-toc-ground-truth](https://github.com/cboulanger/dnb-toc-ground-truth),
  since it grew a real corpus, its own endpoint-configuration system, and
  no meaningful coupling to `chapter_segmentation` beyond a couple of
  vendored data types. See that repo's own `docs/history.md` for its
  experiment history.
