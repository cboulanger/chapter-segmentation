# NuExtract-2.0-4B LoRA fine-tuning pilot

Status: approved for planning
Date: 2026-08-10

## Problem

The zero-shot `numind/NuExtract-2.0-4B` baseline (`evaluation/RESULTS.md`,
"NuExtract3 dropped; NuExtract-2.0-4B full-corpus zero-shot baseline")
scores **f1=0.39** across the full 50-book, two-corpus evaluation set,
measured through `llama.cpp`/GGUF (the actual deployment backend --
`transformers`/MPS was ruled out for benchmarking after it was found to
silently corrupt output on this model, see that section for the full
writeup). A per-book failure-mode breakdown of that run found the
aggregate is not dominated by a hard capability gap:

- **14 of 50 books (28%), the largest single cluster**: the model
  extracts titles and authors **verbatim correctly**, but returns
  `printed_page_number: null` for every entry, even when the number is
  plainly present in the source text next to the title. Since
  `match_toc_entries` (`evaluation/nuextract_baseline.py`) requires an
  exact page-number match to count a true positive, these books score
  zero recall despite mostly-correct extraction -- inspected two examples
  directly (one clean German text, one OCR-scrambled English text) and
  confirmed the number was visibly present in both. This looks like a
  fixable extraction-formatting habit, not a comprehension failure.
- **10 of 50 books (20%)**: true truncation -- the 1500-token output
  budget was too tight for some books' larger TOCs, producing no valid
  JSON at all. Being retested separately with a raised token budget (see
  `evaluation/RESULTS.md`'s "Output-token-limit retest") -- a free fix,
  not something fine-tuning needs to solve, and this pilot's baseline
  number should be whichever of the two (f1=0.39, or the post-retest
  number) is current once this plan's data-prep work begins.

Before generating more ground truth (the question this pilot exists to
answer, raised when planning this work), we need to know whether LoRA
fine-tuning on the existing ~50-book corpus can measurably close the
null-page-number gap. A positive result makes fine-tuning the
highest-leverage next investment; a negative or noisy result means more
ground truth (or a different model/approach) needs to happen before
fine-tuning is worth revisiting.

## Scope

**This is a go/no-go pilot, not a production fine-tune.** Everything
below targets answering one question as cheaply as possible: does LoRA
fine-tuning move the held-out f1, concentrated in the diagnosed failure
mode.

**Training data**, derived entirely from data already in the repo -- no
new ground-truth curation:

- Input: the same `_llm_scan_indices`-selected scan-window text already
  used for the zero-shot baseline (`chapter_segmentation.segmentation`).
- Target: built from each book's `.expected.json` `chapters` list,
  reshaped into `NUEXTRACT_TEMPLATE`'s exact schema
  (`evaluation/nuextract_baseline.py`): `{"chapters": [{"title":
  c["title"], "authors": c["authors"], "printed_page_number": <start of
  c["citation_pages"], or null if citation_pages is null>}, ...]}`.
  Chapters with a null `citation_pages` keep `printed_page_number: null`
  in the target rather than being dropped entirely -- the title/authors
  are still real, useful supervision even when the page number isn't
  recoverable from the text.
- **Training-data files are not committed.** For `copyrighted-scans/`
  books, a training example embeds real scan-window text from
  non-open-access PDFs -- the same reason those books' full text is never
  committed elsewhere in this repo (`*.pdf`/`manifest.local.json` are
  gitignored; `public-cache/` only stores redacted text for non-OA
  books). The data-prep script's output directory must be gitignored,
  mirroring that existing convention, not a new exception to it.

**Train/eval split**: stratified by corpus (roughly 8 held out from
`open-access`'s 37, 3 from `copyrighted-scans`'s 13 -- a ~78/22 split, 39
train / 11 eval), so both corpora's very different zero-shot baselines
(0.47 vs. 0.17 f1) are represented on both sides rather than one corpus
dominating the held-out set. A single held-out split is the pilot's
default -- cheaper, and sufficient to answer "does this obviously help."
5-fold stratified CV is an explicit non-goal for the first pass (see
below); revisit only if the single-split result is ambiguous.

**Training mechanics**: LoRA via `peft`, targeting the Qwen2.5-VL-3B
language-model attention/MLP projections (`q_proj`/`k_proj`/`v_proj`/
`o_proj`/`gate_proj`/`up_proj`/`down_proj`) -- the vision tower is
irrelevant since every input is plain text. Loss masked to the completion
(the filled JSON) only, using the same `apply_chat_template`-built prompt
already used for inference with the target JSON appended as the
assistant turn. Starting hyperparameters: rank 16, alpha 32, dropout
0.05, learning rate 1e-4-2e-4, 3-5 epochs over ~39 training examples,
batch size 1 with gradient accumulation, held-out f1 checked after each
epoch (via the harness below) to pick the best checkpoint rather than
always the last one.

**Backend split -- train via `transformers`/PEFT, evaluate via
`llama.cpp` only.** LoRA training needs `transformers`+`peft`
(`llama.cpp` has no training path); this session already established
that any trustworthy accuracy number for this model must come from
`llama.cpp`, not `transformers`/MPS (see the baseline writeup above).
Pipeline: train the adapter via `transformers` on this machine's `mps`
backend (untested for *training* specifically -- the known MPS bug was
in generation sampling, not a teacher-forced training forward pass; fall
back to a rented cloud GPU only if MPS training proves infeasible), merge
via `peft`'s `merge_and_unload`, convert to GGUF via `llama.cpp`'s
`convert_hf_to_gguf.py`, quantize to Q4_K_M (matching the zero-shot
baseline), and score the held-out split through the exact same
`score_book`/`MicroAggregate` harness already used for the zero-shot
numbers. **Never trust an eval number produced by any other backend for
this model.**

**Deliverables**: a committed data-prep script
(`evaluation/scripts/prepare_nuextract_finetune_data.py`, writing to a
gitignored output directory per the training-data note above), a
committed training script
(`evaluation/scripts/finetune_nuextract.py`), and a documented (not
necessarily scripted, since it chains an external `llama.cpp` checkout)
merge/convert/quantize procedure -- mirroring how
`evaluate_nuextract_baseline.py` and this spike's other scripts are
already committed, real, runnable tools rather than one-off scratchpad
code.

## Non-goals

- No production wiring into `segmentation.py` or a `TocExtractionStrategy`
  implementation -- a follow-up spec once (and if) this pilot's numbers
  justify it.
- No 5-fold cross-validation on the first pass -- only revisit if the
  single-split result is ambiguous (a small, uncertain improvement that
  could be noise from which specific books landed in eval).
- No cloud GPU rental by default -- local `mps` training first; only
  fall back if that proves infeasible.
- No multimodal/vision fine-tuning -- text-only, matching the zero-shot
  scope.
- No CI/automated retraining -- a manual, one-off pilot run, not a
  regression-guarded pipeline.
- No extending the ground-truth corpus beyond the current 50 books --
  that decision explicitly waits on this pilot's result.

## Decision criteria

Compare held-out-split f1 (via `llama.cpp`) against whichever zero-shot
baseline is current when this pilot runs (f1=0.39, or the
post-token-limit-fix number once available -- that's a free fix with no
training cost, so it's the real number to beat). A pilot worth
escalating to a bigger ground-truth investment must show a clear,
not-explained-by-noise improvement **concentrated in the diagnosed
failure mode** -- held-out books' null-page-number rate should
measurably drop, not just the aggregate f1 creeping up from a couple of
lucky books. Given only ~11 held-out books, treat anything within a few
points of baseline as noise: this pilot is powered to detect "does this
obviously help" or "does this clearly not help," not to measure a
precise effect size.

## Out of scope

- Deciding whether to extend the ground-truth corpus -- a follow-up
  decision made after reading this pilot's result, not part of this spec.
- 5-fold CV, a bigger LoRA rank/longer training sweep, or any other
  hyperparameter search -- only worth doing once the single-split pilot
  shows a promising-but-imprecise signal.
- Production wiring, deployment packaging, or Ollama/GGUF distribution of
  a fine-tuned checkpoint -- entirely contingent on this pilot's result.
