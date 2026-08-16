# Uniform layout-sensitive OCR for dnb-toc-only

Status: proposed
Date: 2026-08-16

Follow-up to `docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md`
(the GT-generation script this modifies). Triggered by two real findings
from smoke-testing that script against the live corpus:

1. `needs_ocr` skips ~30% of sampled books outright — their embedded text
   layer is absent or degenerate, so neither extractor ever runs.
2. Even where a text layer exists, quality is inconsistent across the
   corpus (different scanning/digitization sources), so heuristic and LLM
   extraction quality varies for reasons unrelated to either extractor's
   logic.

This spec covers replacing per-PDF embedded/inconsistent text with a single,
project-controlled, layout-sensitive OCR pass applied uniformly across the
whole corpus, plus a smaller independent fix to the agreement-gate's title
matching that the investigation below turned up along the way.

## 1. Investigation summary

Four things were tested directly against the corpus before proposing a
design, using book `9783518585306.pdf` (the known "shredded", one-word-
per-line worst case) as the running example.

**a) Plain `ocrmypdf --force-ocr -l deu+eng` (tesseract's own reading
order).** Fixes the one-word-per-line fragmentation, but introduces a worse
problem on this book's two-column-ish layout: tesseract's block
segmentation groups *all* title lines into one block, then *all* page
numbers into a separate block that follows, e.g. the reconstructed page
text ends with a disconnected `"...33\n81\n100\n117\n155\n174\n203"` list.
`find_toc_candidates`'s regex requires title and number on the same
physical line, so this produces **zero** heuristic entries. Confirms the
user's original concern: naive OCR does not solve column-based TOCs.

**b) `pdfalto` word-position reconstruction.** `pdfalto` (sibling checkout
at `/Users/cboulanger/Code/pdfalto/pdfalto`, already used elsewhere in this
project — `evaluation/scripts/pdfalto_runner.py`) runs cleanly against an
`ocrmypdf`-produced PDF and emits per-word `HPOS`/`VPOS` coordinates.
Clustering `<String>` tokens by `VPOS` (8px tolerance) into rows and sorting
by `HPOS` within each row reconstructs correct reading order regardless of
tesseract's own block segmentation:

```
240 1. Wozu noch Philosophie? .........000e ee eeee 33
260 . Die Philosophie als Platzhalter und Interpret
279 3. Was Theorien leisten können - und was nicht.
300 Ein Interview ss m onen een ee eee eee ees 81
```

Title and page number now land on the same reconstructed line every time —
this directly fixes problem (a). Confirmed on the full page: 20/20 rows
reconstructed correctly, in printed order.

**c) `tessdata_best` vs. the installed `tessdata_fast` German/English
models.** Hypothesis: a stronger OCR model would produce cleaner dot-leader
text (`.........000e ee eeee` above is tesseract mis-reading a run of dots
as garbage words). Downloaded `tessdata_best/deu.traineddata` (8.6MB vs.
the installed 1.5MB fast model) and re-ran. Result: a **mixed, marginal**
improvement — some lines got cleanly recognized dot runs
(`Vorwort zur Studienausgabe ...................`), but others stayed
garbled differently (`Einleitung .2..0..000000000000 0100`), and the best
model introduced a *new* digit misread it didn't have before
(`155` → `I55`, capital-I for `1`). Not a reliable fix on its own, and it
adds a real-world dependency (downloading ~9-15MB per language from
`tessdata_best` at setup time — not bundled with the `tesseract-lang`
Homebrew formula) for an inconsistent payoff. **Rejected** as the primary
fix for dot-leader garbage; not adopted.

**d) Whether dot-leader garbage actually breaks anything, and how to fix
it independently of OCR quality.** `_TOC_LINE_RE`
(`src/chapter_segmentation/segmentation.py:76`) is permissive about what it
swallows into the title group (`.{3,120}?` matches any character), so
garbled dot-leader text does **not** prevent page-number extraction — it
just pollutes the extracted title string:
`"Ein Interview ss m onen een ee eee eee ees"` instead of `"Ein Interview"`.
That pollution *does* break the agreement gate: `align_toc_entries`
(`evaluation/dnb_toc_matching.py`) scores title similarity with
`fuzz.token_sort_ratio`, which drops to 38.9-47.3 on these polluted/clean
pairs — well under the 70.0 threshold — because token-sort compares the
*entire* token multiset and a handful of garbage tokens dominates a short
real title. Testing `fuzz.partial_ratio` (best-matching contiguous
substring) on the same pairs scores a clean **100.0** on all of them, since
the garbage is a trailing addition rather than an interleaved corruption of
the real text. A negative-control check against four genuinely different
title pairs confirmed `partial_ratio` doesn't inflate false positives
either (max 48.8, still well under threshold) — expected, since alignment
is already gated on an *exact* page-number match before title similarity
is even scored, so `partial_ratio`'s looser substring tolerance only needs
to discriminate between candidates that already share a page number.

Also checked whether ALTO's per-token geometry could drive a targeted
"strip the dot-leader run" cleanup instead: found overlapping/nonsensical
`HPOS`/`WIDTH` boxes on the garbled tokens (e.g. a 4-character token
`"onen"` reported 201px wide — 7x the per-character width of neighboring
real words), meaning tesseract's own bounding boxes for hallucinated
dot-leader "words" aren't reliable enough to key a geometric cleanup off
of. Confirms text-level tolerance (partial_ratio) is the right layer to fix
this at, not more geometry.

**e) Runtime cost.** `ocrmypdf --force-ocr -l deu+eng` + `pdfalto` on the
2-page test book: 8.4s wall-clock. Corpus is 1251 PDFs. At this per-book
cost, sequential processing is ~3 hours; with the existing script's
concurrency pattern (currently `--concurrency 4` for the LLM calls, easily
reused for OCR since `ocrmypdf`/`pdfalto` are both subprocess calls) this
comes down to well under an hour. User has already accepted "a significant
time cost" for this.

## 2. Design

Two independent changes come out of this investigation. They ship
separately since they fix different problems and have very different
cost/risk profiles.

### 2.1 Title-matching robustness fix (small, immediate, no pipeline needed)

In `evaluation/dnb_toc_matching.py`, change the scoring inside
`align_toc_entries`'s candidate-title loop from
`fuzz.token_sort_ratio(...)` to `max(fuzz.token_sort_ratio(...),
fuzz.partial_ratio(...))`. This is a pure quality improvement to the
existing gate — independent of whether/how OCR changes — and should land
first since it's cheap to implement, test, and verify, and it reduces
noise in subsequent OCR-pipeline experiments too.

### 2.2 Uniform layout-sensitive OCR pipeline

**New module** `evaluation/scripts/dnb_toc_ocr.py`:

- `ocr_pdf(src: Path, dest: Path) -> None` — runs
  `ocrmypdf --force-ocr -l deu+eng <src> <dest>` via `subprocess`. Always
  force-OCRs (never conditional on the existing text layer) — this is the
  "uniform" part the user asked for: every book gets the same treatment
  regardless of whether its original text layer was present, absent, or
  degenerate. Removes `needs_ocr` as a skip reason entirely, since there's
  no longer a dependency on the original layer.
- `reconstruct_page_text(alto_xml_path: Path) -> list[str]` — parses
  `pdfalto`'s ALTO output, clusters `<String>` tokens per `<Page>` into
  rows by `VPOS` (reusing the 8px tolerance validated above — tune only if
  a false-merge/false-split shows up in broader testing), sorts tokens
  within a row by `HPOS`, joins with single spaces, joins rows with
  newlines. Returns one string per page, in the same shape
  `extract_page_texts_for_analysis` already returns, so it's a drop-in
  replacement at the call site.
- `get_ocr_page_texts(pdf_path: Path, cache_dir: Path) -> list[str]` —
  orchestrates: check cache (see below) → else run `ocr_pdf` to a temp
  file → run `pdfalto` on it → `reconstruct_page_text` → cache the result →
  return it.

**Caching.** OCR is the expensive step and the corpus is static once
downloaded, so cache the *reconstructed page texts* (not the intermediate
OCR'd PDF or ALTO XML — those are large and only useful as debugging
artifacts) keyed by book id, alongside the existing LLM cache convention:
new `ocr_cache_dir(corpus_name)` helper in `evaluation/harness.py`
mirroring the existing `llm_cache_dir`, writing
`evaluation/corpus/dnb-toc-only/.ocr-cache/<key>.json` (`{"pages": [...]}`).
Gitignored, same as `.lobid-cache/`.

**Integration point.** In
`evaluation/scripts/generate_dnb_toc_ground_truth.py`, `_run_book`
currently does:

```python
pages, _ = extract_page_texts_for_analysis(pdf_path.read_bytes())
```

This becomes a call to `get_ocr_page_texts(pdf_path, ocr_cache_dir(_CORPUS_NAME))`.
Both `find_toc_candidates` and `llm_extract_toc_entries` then read from the
same uniform, layout-reconstructed text — this is the specific "consistent
basis for both heuristic and LLM results" the user asked for. The
`needs_ocr` check and skip branch are deleted (no longer applicable — see
above).

**Not in scope:** re-running this against every consumer of
`extract_page_texts_for_analysis` project-wide. This is scoped to
`dnb-toc-only`'s ground-truth generation specifically, per the parent
spec's boundary. Other corpora keep using the embedded text layer as-is.

## 3. Testing

- Unit tests for `reconstruct_page_text` against a small hand-built ALTO
  XML fixture (a handful of `<String>` elements across two rows and two
  columns) — asserts row grouping and left-to-right ordering, independent
  of any real OCR run.
- Unit tests for `get_ocr_page_texts`'s cache read/write, following the
  existing `_load_cached_llm_entries`/`_write_cached_llm_entries` pattern
  in the same file for consistency.
- `ocr_pdf` itself (a thin subprocess wrapper) is integration-tested via a
  smoke run against 3-5 real corpus books, not unit-tested with mocks —
  matching how `pdfalto_runner.py` is already tested elsewhere in this
  project.
- Re-run the existing 60-book smoke test after integration and compare
  pass-rate/skip-reason breakdown against the last recorded run (6/60
  passed, 35 below_threshold, 19 needs_ocr) to quantify the actual
  improvement before declaring this done.

## 4. Open questions for review

- 8px row-clustering tolerance was validated on one book's font size; if
  the corpus has meaningfully different scan resolutions/font sizes, this
  may need to be relative (e.g. a fraction of median token height on the
  page) rather than a fixed pixel value. Flagging rather than
  pre-solving — worth checking against a wider book sample before
  committing to fixed vs. adaptive.
- `ocrmypdf --force-ocr` on already-OCR'd (or already-good) pages destroys
  the existing layer and fully re-rasterizes+re-recognizes. This is
  intentional (uniformity is the whole point) but means the corpus's disk
  footprint temporarily doubles during the OCR pass (original + `.ocr.pdf`
  intermediate) unless intermediates are written to a temp dir and
  discarded after `pdfalto` runs and the cache is written — the design
  above already does this (only the reconstructed text is cached), but
  flagging so it's an explicit decision, not an oversight.
