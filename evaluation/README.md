# Evaluation data for book segmentation

Every evaluation book lives under `evaluation/corpus/<name>/` -- a
self-contained subfolder per corpus (see "Corpora" below). Ground-truth
chapter boundaries are hand-verified and committed as
`<filename-without-extension>.expected.json` inside that corpus's
directory (schema: see
`docs/superpowers/plans/2026-07-24-chapter-segmentation-linking.md` Task
30). The PDFs themselves are gitignored (`*.pdf`, see `.gitignore` in this
directory) and are not shipped. See
`docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md` for
the full rationale behind the per-corpus split.

This README documents what the evaluation set is and how to run it -- it
changes rarely. For current precision/recall numbers, known gaps, and
investigation findings from the last time each evaluation was actually run,
see **`RESULTS.md`** in this directory instead -- that document is a
snapshot and is expected to be regenerated (or rewritten) whenever the
heuristics, the strategy pipeline, the extraction/OCR path, or the
evaluation set change.

## Corpora

`evaluation.harness.list_corpora()` auto-discovers every subfolder of
`evaluation/corpus/` that has a `manifest.json` -- every runner below loops
over all of them by default, with an optional `--corpus <name>` flag to
restrict to one. Three corpora exist today:

- **`open-access/`** (6 books) -- well-produced, OA, parseable embedded
  TOCs. The case the pure-heuristic pipeline already handles well.
- **`copyrighted/`** (11 books) -- sourced from a real personal Zotero
  library: no DOI, `embedded_toc: false`, native and scanned, German and
  English. The case the outline/Crossref/Zotero-catalog strategies, the
  layout-mode extraction fallback, and the evaluation OCR route were built
  for.
- **`pending/`** (2 books) -- have a manifest entry and PDF but no
  `.expected.json` yet, so they contribute to no evaluation until someone
  builds ground truth for them (see `CLAUDE.md`'s "Step 0a"), at which point
  the entry moves into whichever real corpus it belongs in.

Each corpus directory has the same shape:

```text
evaluation/corpus/<name>/
  manifest.json          # committed -- the source for this corpus
  manifest.local.json    # optional, gitignored (same schema, see below)
  <isbn>.pdf              # gitignored
  <isbn>.expected.json    # committed (except in pending/)
  public-cache/
  .ocr-cache/             # gitignored
  llm-cache/
```

`manifest.json` entries have:

- `filename` — matches a `<name>.pdf` / `<name>.expected.json` pair in the
  same corpus directory
- `title`, `language`, `extraction_type` (`native` or `scan`), `embedded_toc`
- `oa` — whether the book can be legally auto-downloaded
- `doi` — the book's DOI (used both as metadata and, for non-OA books, as
  the pointer a human follows to acquire it)
- `download_url` — direct PDF URL, only meaningful when `oa: true`; `null`
  otherwise

**Fetching the PDFs:**

```bash
uv run python evaluation/scripts/fetch_evaluation_pdfs.py
uv run python evaluation/scripts/fetch_evaluation_pdfs.py --corpus open-access   # just one corpus
```

Downloads every `oa: true` entry that isn't already present. Non-OA entries
are never touched by this script — if one is missing, it prints the DOI and
the exact path to save the file to. Get that book through your institution's
legal access (library subscription, interlibrary loan, etc.), save it at the
printed path, and re-run the tests.

**Adding a new evaluation book** (e.g. a "difficult" PDF the segmentation
heuristics scored low-confidence on during live testing against a real
Zotero library) — see `CLAUDE.md` in this directory for the full step-by-step
workflow, including which corpus it belongs in, the
`evaluation/scripts/ground_truth_helper.py` draft-then-verify process, and
known failure modes. Short version:

1. Decide the corpus (`open-access`/`copyrighted`/`pending` -- see
   `CLAUDE.md`'s "Step 0a").
2. Has a DOI? Add an entry to that corpus's committed `manifest.json`
   (`"oa": false, "download_url": null` if it can't be freely
   redistributed — that's fully supported, it just means
   `fetch_evaluation_pdfs.py` won't auto-download it, only print the DOI
   for manual acquisition). No DOI, or can't be identified/shared at all?
   Add it to that corpus's `manifest.local.json` instead (same schema,
   gitignored, never committed — see `CLAUDE.md`) so it's still exercised
   by your own local test runs.
3. Place (or fetch) the PDF at `evaluation/corpus/<corpus>/<filename>`.
4. Build `evaluation/corpus/<corpus>/<name>.expected.json` by actually
   inspecting the real PDF — never guessed or extrapolated from the TOC
   alone (`CLAUDE.md` explains why).

## Running an evaluation

The harness lives at `tests/test_segmentation_accuracy.py`,
marked `@pytest.mark.integration` so it never runs as part of the default
`uv run pytest` / `npm test` (see `pyproject.toml`'s `addopts`). Run it
directly:

```bash
uv run pytest tests/test_segmentation_accuracy.py -q -s
```

`-s` is required to see the per-book summary lines (`pytest` swallows `print`
output by default). A book is silently skipped, not failed, if its PDF isn't
present locally yet — run `uv run python evaluation/scripts/fetch_evaluation_pdfs.py`
first for the open-access ones, or acquire non-OA books manually (see above)
— or if it needs OCR and the evaluation OCR cache hasn't been populated yet
(see below).

Pages are loaded via `evaluation/harness.py`'s `analysis_pages_for`,
the same way production's `chapter_segmentation.run()` loads them: default
pypdf extraction, falling back to pypdf's `layout` extraction mode when the
default mode finds no table of contents (recovers books whose two-column
TOC the default mode scrambles), and finally the evaluation OCR cache
(`.ocr-cache/` in this directory, gitignored,
content-hash keyed) for books whose text layer is absent or degenerate. A
book that needs OCR and has no cache entry yet prints `SKIPPED (needs OCR
...)` and is excluded from that run rather than scored 0.00 -- populate the
cache first with:

```bash
uv run python evaluation/scripts/ocr_evaluation_pdfs.py
```

This requires the Kreuzberg sidecar container running (see root `CLAUDE.md`'s
Live Server section). It OCRs every evaluation book whose text layer needs
it and caches the result by content hash, so a first run over several full
scanned books can take 1-3 hours but every later run (unchanged PDFs) is
near-instant, cache-hit-only.

For each book, the test runs `chapter_segmentation.analyze_attachment` over
the loaded page text and compares the resulting chapter ranges
against `<name>.expected.json`:

- **Precision** = correctly-found ranges ÷ all ranges the algorithm returned
  (how much of what it found was right)
- **Recall** = correctly-found ranges ÷ all ranges in the ground truth (how
  much of the real content it found)

A "correct" range requires an **exact** `(pdf_start_index, pdf_end_index)`
match — a one-page-off boundary counts as a miss, not partial credit.

This harness is **reported, not gated** (design spec §12): the heuristics are
probabilistic and not expected to hit 100% on real, messy PDFs, so precision/
recall are not asserted against a required minimum. The only hard assertion
is `recall > 0` per book — a regression guard that catches "this book now
finds zero of its known chapters," not a quality bar. A book flagged
`"heuristic_expected_zero": true` in its manifest entry is exempt from this
assertion (see `CLAUDE.md`'s "Step 0b" for exactly when that flag applies and
how to re-check it); `RESULTS.md` documents which books currently carry it
and why.

### Per-strategy evaluation report

`evaluation/generate_report.py` (published automatically to GitHub Pages
on every push to `main` -- see `.github/workflows/publish-results.yml`)
scores the heuristic and outline strategies independently against the
public-cache corpus -- no pipeline merge/fallback decision is involved,
so each strategy's own standalone accuracy is visible, not just which one
a production run happened to pick. It costs no API calls and needs no
PDFs; run it locally the same way CI does:

```bash
uv run python evaluation/generate_report.py --out public/
```

Produces a `public/index.html` landing page linking to one report per
corpus (`public/<corpus>/index.html`, one row per book x strategy, with
precision/recall/F1/time, best-F1 cell per row marked, plus a per-strategy
summary ordered by aggregate F1) and `public/<corpus>/llm/index.html` per
corpus (see "LLM strategy evaluation" below).

### LLM strategy evaluation

Unlike the heuristic and outline strategies, evaluating the LLM strategy
costs real KISSKI API budget, so it is decoupled from report generation:
`evaluation/refresh_llm_cache.py` is the only script that calls an LLM,
and it writes its results into `evaluation/corpus/<corpus>/llm-cache/<book>.json` (raw
chapters found + timing per model, committed to git) rather than printing
a report directly. `evaluation/generate_report.py` then reads that cache
for free on every run -- folding the single best-performing cached model
into the main report as an "LLM (\<model\>)" column, and rendering every
cached model's full breakdown at `public/<corpus>/llm/index.html` per
corpus.

Run it manually, with `KISSKI_API_KEY` in the environment (locally, source
it from `zotero-rag`'s `.env`):

```bash
export KISSKI_API_KEY=$(grep '^KISSKI_API_KEY=' ../zotero-rag/.env | cut -d= -f2-)
uv run python evaluation/refresh_llm_cache.py --mode top5
```

`--mode top5` (the default) refreshes the 5 currently-least-busy KISSKI
models. `--mode fill-gaps` instead finds non-busy models not yet cached
for every book in the corpus and runs up to 5 of those -- this is what
`.github/workflows/refresh-llm-cache.yml`'s nightly schedule uses, so the
cache grows to cover every available/busy model over time without paying
to re-run models it already has complete data for. The same workflow also
exposes a manual `workflow_dispatch` trigger (using `--mode top5`) for an
on-demand refresh, e.g. right after a prompt change, to sanity-check the
current best models. Either trigger commits the updated cache files
straight to `main`, which republishes the report automatically.

### Strategy-pipeline evaluation

`evaluation/scripts/evaluate_chapter_segmentation_strategies.py` runs the same
evaluation set through `analyze_attachment_with_strategies` (see
`docs/superpowers/specs/2026-08-01-chapter-segmentation-strategy-pipeline-design.md`)
instead of the pure-heuristic `analyze_attachment` -- i.e. with the PDF
outline read and Crossref-by-ISBN lookup strategies active (the evaluation
manifest names each PDF after its own ISBN-13, which doubles as the ISBN
this script passes in). Not a pytest test -- makes real, free, cached
Crossref API calls:

```bash
uv run python evaluation/scripts/evaluate_chapter_segmentation_strategies.py
```

Prints the same precision/recall table format as the harnesses above, plus
each book's `strategies_used` diagnostic. Run after any change to the
outline/Crossref/fusion logic to check whether the new strategies are
net-helpful on the real evaluation set, the same operational pattern the
LLM strategy evaluation above already established -- record what you
find in `RESULTS.md`.

(This script evaluates the *merged pipeline's* Crossref/Zotero-catalog
behavior specifically -- for the outline and LLM strategies evaluated
independently of any pipeline decision, see "Per-strategy evaluation
report" and "LLM strategy evaluation" above.)

## Related: Crossref-sourced ground-truth corpus

`evaluation/crossref_gt/` holds a separate, standalone corpus of 46
open-access books with their Crossref-registered chapter metadata,
sourced specifically to eventually evaluate `CrossrefMetadataStrategy`
against ground truth pulled from the same system it queries. It is not
yet wired into any of the harnesses documented above -- see
`evaluation/crossref_gt/README.md` for its own schema and status.

## Evaluation set composition

The `open-access/` corpus (6 books) is small and, per the design
spec's `## 1. Goal` motivation, skews toward well-produced academic books
with a parseable embedded TOC page -- exactly the case the pure-heuristic
pipeline already handles well (all 6 have `embedded_toc: true`).

The `copyrighted/` corpus (11 books) is sourced directly from a real
personal Zotero library. 10 of its 11 books were selected for the opposite
profile from `open-access/`: `embedded_toc: false`, spanning 1967-2020,
native and scanned, German and English, none open access, and none with a
Crossref-registered DOI at all (checked directly against the API, book- or
chapter-level). This set exercises the case the outline/Crossref/
Zotero-catalog strategies were built for, and the case the layout-mode
extraction fallback and evaluation OCR route (see "Running an evaluation"
above) were built for:

| Filename | Title | Year | Extraction |
| --- | --- | --- | --- |
| `9783848736829.pdf` | Politik und Recht | 2017 | native |
| `9783492021234.pdf` | Empirische Rechtssoziologie | 1975 | native |
| `9783789016202.pdf` | Rechtsproduktion und Rechtsbewußtsein | 1988 | native |
| `9783789057366.pdf` | Soziologie des Rechts (Festschrift für Erhard Blankenburg) | 1999 | native |
| `9783899718188.pdf` | Systemtheorie in den Fachwissenschaften | 2011 | native |
| `9780367439712.pdf` | Luhmann and Socio-Legal Research | 2020 | native |
| `9783465016878.pdf` | Historische Soziologie der Rechtswissenschaft | 1986 | scan |
| `9781409403906.pdf` | Central and Eastern Europe After Transition | 2010 | scan |
| `9783848704316.pdf` | Constitutional Jurisprudence | 2016 | scan |
| `dnb-36942798X.pdf` | Studien und Materialien zur Rechtssoziologie | 1967 | scan |

The corpus's 11th book, `9783322969828.pdf` (Jahrbuch für Rechtssoziologie
und Rechtstheorie IV, 1976, scan), is not part of this set of 10 -- it has
a DOI and a Crossref-registered record, so its ground truth was built via
`CLAUDE.md`'s Crossref-page-range shortcut instead, the same way the
`open-access/` corpus's books were.

See `RESULTS.md` for how each of these books currently scores and why.

Ground truth for the other 10 books was built directly from the PDFs via
multimodal reading (visually locating each chapter's opening/closing page
and transcribing its table of contents) rather than `CLAUDE.md`'s
Crossref-page-range shortcut for OA books, since none of the 10 has any
Crossref record to shortcut from. All 10 entries were originally added to
the gitignored `manifest.local.json` for that reason (see `CLAUDE.md`'s
"Step 0b"), then moved into this corpus's committed `manifest.json` once
each gained a `public-cache/` entry -- but the 10 `.expected.json`
ground-truth files themselves (titles, authors, and page indices only, no
copyrighted text) were committed from the start, so anyone who acquires
the same PDFs by ISBN (see filenames) can reuse them directly without
rebuilding ground truth from scratch.

Two other personal-library PDFs were found and rejected as candidates
before landing on this set of 10: both turned out to be heavily abridged
excerpts of much larger books (missing whole chapters, or missing
everything past a certain page, likely a photocopied subset rather than a
full digitization) rather than complete books -- a real, if narrower,
failure mode of sourcing evaluation data this way, distinct from either
`embedded_toc` value and worth checking for (compare the file's actual page
count against its own printed page numbers near the front and back) before
investing time in transcribing ground truth for a new candidate.
