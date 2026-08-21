# Experiments

Snapshots of in-progress or experimental work that investigates candidate
improvements to chapter-segmentation accuracy or to the evaluation
pipeline itself. None of these are part of the main
`chapter_segmentation` workflow, and none affect the numbers in
`evaluation/RESULTS.md` unless a write-up says otherwise. Each file
carries a "Current status" section that's kept up to date and a "History"
section that keeps every superseded run and dead end, so the reasoning
behind the current numbers isn't lost.

- [`toc-classifier-pilot.md`](toc-classifier-pilot.md) -- whether a
  classifier trained purely on page-layout geometry (no text content) can
  pre-filter table-of-contents and chapter-opening pages.
- [`nuextract-finetuning.md`](nuextract-finetuning.md) -- whether a small,
  locally-runnable extraction model (`NuExtract-2.0-4B`) can extract a
  book's table of contents directly, as a candidate alternative to the
  LLM-strategy row in `evaluation/RESULTS.md`.
- **dnb-toc-ground-truth** -- the LLM-gated ground-truth-generation
  pipeline for DNB table-of-contents scans has moved to its own
  standalone repository (`dnb-toc-ground-truth`), since it grew a real
  corpus, its own endpoint-configuration system, and no meaningful
  coupling to `chapter_segmentation` beyond a couple of vendored data
  types. See that repo's own `docs/history.md` for its experiment
  history.
