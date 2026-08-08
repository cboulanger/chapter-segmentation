# Multi-corpus evaluation layout

Status: approved for planning
Date: 2026-08-08

## Problem

`evaluation/manifest.json` currently commingles two historically distinct
book sets with very different accuracy profiles:

- 7 well-produced, mostly-OA books with parseable embedded TOCs (aggregate
  precision/recall **0.91/0.91** per `RESULTS.md`'s "Pure-heuristic
  results").
- 10 books sourced from a real personal Zotero library -- no DOI, no
  embedded TOC, native and scanned, needing the layout-mode extraction
  fallback or OCR (aggregate **0.47/0.24** per `RESULTS.md`'s "Diverse
  real-library evaluation set").

These sets used to live in separate files (`manifest.json` vs. the
gitignored `manifest.local.json`) but were merged into one `manifest.json`
once every book gained a `public-cache/` entry (see `CLAUDE.md`'s "Document
organization" and `RESULTS.md` line ~264). The merge lost the ability to
report, evaluate, or reason about the two sets independently, even though
`RESULTS.md` still hand-splits them into two tables because their numbers
are too different to average meaningfully.

Two further books (`9783428042241.pdf`, `9783899496291.pdf`) have manifest
entries, PDFs, and `public-cache/` entries but no `.expected.json` yet --
they exist in the manifest but contribute to no evaluation.

Goal: restructure the evaluation set into self-contained per-corpus
subfolders, with all runners (tests, scripts, report generation)
auto-discovering every corpus, so a corpus can be added, removed, or scored
independently without code changes to the runners themselves.

## Corpus definitions and book assignment

Three corpora to start, each a subfolder of `evaluation/corpus/`:

**`open-access/`** (6 books, all `"oa": true`):
`9783031466373.pdf`, `9781771993661.pdf`, `9783907297339.pdf`,
`9782375460122.pdf`, `9783907297285.pdf`, `9783847432364.pdf`.

**`copyrighted/`** (11 books: the 10-book "diverse real-library" set plus
the one non-OA book -- `9783322969828.pdf`, `oa: false`, has a DOI -- that
used to sit in the same file as the open-access set; per explicit user
decision it moves here rather than staying with `open-access/`):
`9783322969828.pdf`, `9783848736829.pdf`, `9783492021234.pdf`,
`9783789016202.pdf`, `9783789057366.pdf`, `9783899718188.pdf`,
`9780367439712.pdf`, `9783465016878.pdf`, `9781409403906.pdf`,
`9783848704316.pdf`, `dnb-36942798X.pdf`.

**`pending/`** (2 books, no `.expected.json` yet -- not part of either
scored corpus until ground truth is built for them, at which point the
entry moves into whichever real corpus it belongs in):
`9783428042241.pdf`, `9783899496291.pdf`.

6 + 11 + 2 = 19, matching today's `manifest.json` book count exactly -- this
is a pure reorganization, no book is added, dropped, or reclassified beyond
the one explicit open-access → copyrighted move above.

## Directory layout

Each corpus directory is self-contained -- same file roles as today's flat
`evaluation/`, just nested:

```text
evaluation/corpus/<name>/
  manifest.json          # committed
  manifest.local.json    # optional, gitignored (same schema, same purpose)
  <isbn>.pdf              # gitignored
  <isbn>.expected.json    # committed (skip for pending/ books)
  public-cache/
    <isbn>.pages.json
    <isbn>.outline.json
  .ocr-cache/             # gitignored, content-hash keyed
  llm-cache/
    <isbn>.json
```

`evaluation/.gitignore`'s existing patterns (`*.pdf`, `manifest.local.json`,
`.ocr-cache/`) are not anchored to a specific depth, so they continue to
match correctly under the new nested paths without modification.

Everything else stays where it is: `evaluation/harness.py`,
`evaluation/metrics.py`, `evaluation/report_html.py`, `evaluation/kisski.py`,
`evaluation/redaction/`, `evaluation/generate_report.py`,
`evaluation/refresh_llm_cache.py`, `evaluation/scripts/*.py`,
`evaluation/README.md`, `evaluation/CLAUDE.md`, `evaluation/RESULTS.md`.

## `harness.py`: corpus-parameterized API

Path constants become functions of a corpus name; every book-lookup
function gains a leading `corpus: str` parameter:

```python
EVAL_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = EVAL_DIR / "corpus"

def list_corpora() -> list[str]:
    """Sorted names of every subfolder under evaluation/corpus/ that has a
    manifest.json -- the single source of truth for "what corpora exist"
    that every runner below iterates over."""

def corpus_dir(corpus: str) -> Path: ...
def public_cache_dir(corpus: str) -> Path: ...
def ocr_cache_dir(corpus: str) -> Path: ...
def llm_cache_dir(corpus: str) -> Path: ...

def load_manifest_books(corpus: str) -> list[dict]: ...
def available_books(corpus: str) -> list[tuple[Path, Path, dict]]: ...
def available_public_books(corpus: str) -> list[tuple[str, Path, dict]]: ...
def public_pages_for(corpus: str, manifest_key: str) -> Optional[list[str]]: ...
def analysis_pages_for(corpus: str, file_bytes: bytes) -> Optional[list[str]]: ...
def public_outline_candidates_for(corpus: str, manifest_key: str) -> Optional[list[ChapterCandidate]]: ...
```

`outline_candidate_to_dict` / `outline_candidate_from_dict` are pure
serialization helpers with no path dependency -- unchanged.

## Consumers: auto-discover by default, `--corpus` to restrict

Every runner loops `for corpus in list_corpora(): ...` unless a `--corpus
<name>` argument restricts it to one. The two pytest integration tests
below have no such flag -- they are already invoked wholesale
(`pytest tests/test_segmentation_accuracy.py -q -s`, no book-level
selection today either) and stay that way, always covering every corpus in
one run:

- **`evaluation/scripts/fetch_evaluation_pdfs.py`** -- downloads OA entries
  per corpus into that corpus's directory.
- **`evaluation/scripts/ocr_evaluation_pdfs.py`** -- OCRs into
  `ocr_cache_dir(corpus)`.
- **`evaluation/scripts/generate_public_evaluation_cache.py`** -- writes
  into `public_cache_dir(corpus)`.
- **`evaluation/scripts/evaluate_chapter_segmentation_strategies.py`** --
  prints results grouped by corpus (a header line per corpus, same
  per-book format as today underneath).
- **`tests/test_segmentation_accuracy.py`** and
  **`tests/test_public_evaluation_cache_parity.py`** -- iterate every
  corpus's `available_books()` / `available_public_books()` in the same
  test method, `subTest` labeled `f"{corpus}/{book_name}"` so failures
  still pinpoint the exact book. The module-level `skipUnless` guard
  becomes "skip if no corpus has any available book at all."
- **`evaluation/refresh_llm_cache.py`** -- model-selection/coverage logic
  (`_fully_covered_model_ids`, `_all_cached_model_ids`) takes an explicit
  `cache_dir: Path` parameter instead of reading a module-level constant,
  so it can be called once per corpus with that corpus's `llm_cache_dir()`.
  `_main` builds the combined `(corpus, manifest_key)` list across every
  corpus's `available_public_books()` for model selection, then runs each
  selected model against every book, upserting into that book's own
  corpus's `llm-cache/`.
- **`evaluation/scripts/ground_truth_helper.py`** -- unchanged; it already
  takes explicit `--pdf`/`--toc`/`--output` paths, so nested corpus paths
  just get passed in like any other path.

## Reporting: one page per corpus, plus a landing page

`evaluation/generate_report.py`'s `generate(out_dir)` is rewritten as
`generate_corpus(corpus, out_dir)`, producing `public/<corpus>/index.html`
and `public/<corpus>/llm/index.html` (same two-table format and same LLM
best-model-selection logic as today, just scoped to one corpus's books and
that corpus's `llm-cache/`).

A new top-level `public/index.html` landing page lists every corpus that
currently has at least one book in `available_public_books(corpus)`
(skipping `pending/` while it remains empty), linking to each corpus's
`index.html`. `generate()`'s top-level driver becomes: for each corpus with
scorable books, call `generate_corpus`; then write the landing page.

`.github/workflows/publish-results.yml` needs no change -- it already just
invokes `uv run python evaluation/generate_report.py --out public/`, and
the new internal looping is transparent to it.

`.github/workflows/refresh-llm-cache.yml`'s commit step changes:

```diff
- git add evaluation/llm-cache/
+ git add evaluation/corpus/*/llm-cache/
```

## Testing changes

- `tests/test_harness.py` -- update every test that currently patches
  `evaluation.harness.EVAL_DIR` / `PUBLIC_CACHE_DIR` to instead patch
  `evaluation.harness.CORPUS_ROOT` with a temp dir containing a single
  `<corpus-name>/` subfolder, and pass that corpus name into the functions
  under test.
- `tests/test_generate_report.py` -- update fixtures the same way, plus add
  a test that `generate()`'s landing page links only to corpora with
  scorable books, and a test that two corpora produce two independent
  `public/<corpus>/index.html` files with independent aggregates (a
  regression in one corpus's report must not touch the other's numbers).
- `tests/test_refresh_llm_cache.py` -- update to pass `cache_dir` directly
  to `_fully_covered_model_ids`/`_all_cached_model_ids` instead of patching
  a module-level `LLM_CACHE_DIR`.
- No changes needed to `tests/test_metrics.py`, `tests/test_report_html.py`,
  `tests/test_kisski.py`, `tests/test_redaction.py` -- none of them touch
  corpus/path structure.

## Docs

- **`evaluation/README.md`** -- replace the single `manifest.json` schema
  section with a description of the `evaluation/corpus/<name>/` layout,
  the three starting corpora and what distinguishes them, and update every
  command example (`fetch_evaluation_pdfs.py`, `ocr_evaluation_pdfs.py`,
  `generate_report.py`, etc.) to mention the `--corpus` option. The
  "Evaluation set composition" section's two tables move here largely
  as-is, now clearly scoped to `open-access/` and `copyrighted/`.
- **`evaluation/CLAUDE.md`** -- "Step 0" gains a first-level decision above
  the existing DOI/public-cache branch: which corpus does a new book
  belong in? OA (or otherwise well-produced with an embedded/parseable
  TOC) → `open-access/`; anything else with usable ground truth →
  `copyrighted/`; no ground truth built yet → `pending/`. All path
  references (`evaluation/<filename>`, `evaluation/manifest.local.json`)
  become `evaluation/corpus/<name>/<filename>`, etc.
- **`evaluation/RESULTS.md`** -- keeps its existing two per-corpus result
  sections (renamed to match the new folder names), commands updated to
  mention `--corpus` where relevant. Numbers themselves are not expected to
  change from this reorganization (same books, same code path) -- if a
  regeneration after implementation shows different numbers, that is a bug
  to fix before merging, not an expected outcome to document.

## Migration mechanics

1. Split `evaluation/manifest.json` into
   `evaluation/corpus/open-access/manifest.json` (6 entries) and
   `evaluation/corpus/copyrighted/manifest.json` (11 entries, including the
   moved `9783322969828` entry), plus a new
   `evaluation/corpus/pending/manifest.json` (2 entries, `oa`/`doi` fields
   preserved as-is).
2. `git mv` each book's `<isbn>.expected.json` and (for OA books, if
   present locally) `<isbn>.pdf` into its new corpus directory.
   `public-cache/<isbn>.pages.json` and `.outline.json` similarly `git mv`
   into `corpus/<name>/public-cache/`. `llm-cache/<isbn>.json` `git mv`
   into `corpus/<name>/llm-cache/`.
3. Update `harness.py`, then every consumer, per the sections above.
4. Regenerate nothing -- the migration is a pure file move plus API
   signature change; `generate_public_evaluation_cache.py`,
   `ocr_evaluation_pdfs.py`, and `refresh_llm_cache.py` do not need to be
   re-run unless verification (below) finds a mismatch.
5. Verify: `uv run python evaluation/generate_report.py --out
   /tmp/verify-public` before and after the migration (on real committed
   `public-cache/`/`llm-cache/` data, no PDFs needed) and diff the two
   corpus pages' aggregate numbers -- they must match exactly, since no
   book, cache entry, or scoring logic changed, only paths.

## Out of scope

- Any change to the scoring/metrics logic itself (`harness.py`'s page
  loading, `metrics.py`, `report_html.py`'s rendering) -- confirmed with
  the user that scoring stays identical across corpora for now.
- Splitting `RESULTS.md` into multiple files -- it keeps its current
  two-section structure, just re-scoped.
- Deciding the final fate of the two `pending/` books (building their
  ground truth) -- this change only relocates them so they sit under the
  same `corpus/` structure; someone still has to transcribe their TOCs
  per `CLAUDE.md`'s existing workflow before they count toward either real
  corpus.
- A fourth+ corpus -- the auto-discovery mechanism supports adding one
  later with no runner code changes, but no new corpus is added as part of
  this change.
