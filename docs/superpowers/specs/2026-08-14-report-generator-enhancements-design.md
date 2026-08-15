# Report generator: generation date, LLM staleness date, classifier column

Status: approved for planning
Date: 2026-08-14

## Problem

The published report (`evaluation/generate_report.py`, rendered by
`evaluation/report_html.py`, deployed by `.github/workflows/publish-results.yml`
on every push to `main` and indirectly by the nightly
`.github/workflows/refresh-llm-cache.yml` job's push) has three gaps:

1. The footer states the commit the report was built from
   (`Generated from commit {sha}.`) but not when it was built. Since the
   page rebuilds on every commit, readers have no way to tell how fresh a
   given snapshot is without cross-referencing the commit timestamp.
2. The LLM strategy column (`LLM (<model>)`) is folded in from
   `evaluation/corpus/<corpus>/llm-cache/*.json`, which is refreshed on a
   separate nightly cron, independent of the commit that triggers a report
   rebuild -- so the LLM numbers shown can be arbitrarily older than the
   footer's commit/date would suggest. Investigation (this session) also
   found the nightly fill-gaps job is far short of covering every model
   evenly: `refresh_llm_cache.py` processes models and books fully
   sequentially (no concurrency), averaging ~31s/book across ~89 books, so
   one model's full pass takes ~46 minutes against the job's 60-minute
   timeout. Only one model (`apertus-70b-instruct-2509`, always picked
   first as lowest-demand) has reached full coverage across both corpora;
   the other 9 models ever touched sit at partial coverage. That
   inefficiency is a separate, pre-existing bug in the cache-refresh job
   and is explicitly **out of scope** for this spec (see Non-goals) --
   this spec only makes the resulting staleness visible in the report,
   which was the original ask.
3. The layout-geometry TOC/chapter-first-page classifier
   (`evaluation/scripts/evaluate_layout_toc_classifier.py`) has no results
   anywhere in the generated report. Its own leave-one-book-out (LOBO)
   pilot output (`full_recall_fraction`, `avg_candidate_fraction`, and
   per-book `toc_recall`/`chapter_first_recall`/`candidate_fraction`) is
   currently printed to stdout only and hand-copied into
   `evaluation/RESULTS.md`; there is no committed, machine-readable
   artifact for `generate_report.py` to read.

## Goal

1. Add the report's own build date next to the commit id in the footer of
   every generated page (corpus report, LLM detail page, landing page).
2. Add a "freshest cache data" date to wherever an LLM model's name is
   displayed (the merged `LLM (<model>)` column/row on the main corpus
   report, and each model's own column on the `llm/index.html` detail
   page), computed as the latest timestamp across the books that
   contributed to that specific model's numbers.
3. Add the layout/TOC classifier's LOBO results to both the per-document
   and per-strategy ("per strategy (aggregate)") tables on the main corpus
   report, with a visible note that its metric (page-classification
   recall via LOBO cross-validation) is not directly comparable to the
   other rows' chapter-boundary precision/recall/F1, plus its own "as of
   `<date>`" freshness marker.

## Non-goals

- Fixing `refresh_llm_cache.py`'s sequential-processing/coverage-growth
  problem (see Problem #2) -- tracked as a separate follow-up, not part of
  this change.
- Making the classifier evaluation run automatically in CI. It needs
  `scikit-learn` (the `layout-classifier` extra, not installed by either
  existing workflow) and a sibling `pdfalto` binary build that CI does not
  provision. Its results file is a manually-refreshed, committed artifact,
  the same operational model as `evaluation/RESULTS.md`'s classifier
  section today -- just now also machine-readable.
- Changing the classifier's decision bar, features, or LOBO methodology.
- Unifying the classifier's metric with chapter-boundary F1 (e.g. deriving
  an equivalent precision/recall) -- the note-based non-comparability
  callout is the intentional resolution here, not a metric conversion.

## Design

### 1. Footer generation date

`generate_report.py` currently appends the footer via a string `.replace()`
on the rendered HTML:

```python
html = html.replace(
    "</body></html>",
    f"<p>Generated from commit {_git_sha()}.</p></body></html>",
)
```

Add a `_today() -> str` helper returning `datetime.now(timezone.utc)`
formatted as an ISO date (`YYYY-MM-DD`), and change the footer text to
`Generated on {date} from commit {sha}.`. Apply the same change to
`_write_landing_page`'s footer. No change needed in `report_html.py` since
the footer is appended outside `render_strategy_tables`.

### 2. LLM freshness date

**Cache schema change** (`refresh_llm_cache.py`, `_upsert_cache`): add a
per-model timestamp inside each model's entry, alongside the existing
file-level `generated_at` (kept as-is for backward compatibility/other
readers):

```python
data["models"][model_id] = {
    "chapters": chapters,
    "elapsed_seconds": elapsed_seconds,
    "demand_at_run": demand,
    "generated_at": datetime.now(timezone.utc).isoformat(),
}
```

Old cache entries written before this change simply lack the per-model
`generated_at` key; treat a missing key as "date unknown" (render nothing
extra for that model rather than guessing).

**`generate_report.py` changes:**

- `_load_llm_cache` is unchanged (still returns the whole `models` dict per
  book; callers already have access to each entry's fields).
- New helper `_latest_model_date(corpus, model_id, books) -> str | None`:
  scans every book's cache entry for `model_id`, returns the max
  `generated_at` (date part only) found, or `None` if no entry has the
  field.
- `_best_llm_model`'s caller (`generate_corpus`) computes this date for the
  chosen `best_llm_model` and folds it into the label:
  `llm_strategy_name = f"LLM ({best_llm_model}, as of {date})"` if a date
  was found, else falls back to the current `f"LLM ({best_llm_model})"`.
  Because `report_html.render_strategy_tables` only ever consumes
  `strategy_names`/`aggregates` keys as opaque labels (both the
  per-document column header and the per-strategy row label read from the
  same string), no change to `report_html.py` is needed for this part --
  the richer label just flows through.
- `_generate_llm_detail_page`: `strategy_names` currently is
  `sorted(model_ids)` (raw ids used directly as dict keys throughout).
  Change to build a `{model_id: label}` map first (label = `f"{model_id}
  (as of {date})"` or bare `model_id` if no date), use labels as the keys
  for `per_document`/`aggregates`/`aggregate_times`/`citation_aggregates`
  and as `strategy_names`, sourced consistently from the same map so
  lookups by label stay correct.

### 3. Classifier results

**New persisted artifact** — `evaluate_layout_toc_classifier.py` gains a
`--save-results <path>` flag (default: none, i.e. current stdout-only
behavior unchanged when omitted). When given, after computing
`evaluate_leave_one_book_out`'s summary, write:

```json
{
  "generated_at": "2026-08-14T12:00:00+00:00",
  "full_recall_fraction": 0.67,
  "avg_candidate_fraction": 0.09,
  "per_book": {
    "9781234567890": {
      "toc_recall": 1.0,
      "chapter_first_recall": 0.92,
      "candidate_fraction": 0.08
    }
  }
}
```

(`toc_recall`/`chapter_first_recall` may be `null`, matching the existing
"vacuous pass" case where a book has zero ground-truth pages of that
label.) Path convention: `evaluation/corpus/<corpus>/classifier-results.json`,
one file per corpus matching `--corpora` scope -- when the script runs
across multiple corpora in one invocation, split `per_book` results by each
book's own corpus and write one file per corpus (mirroring how
`llm-cache/` is already split per corpus). This file is committed to git,
refreshed by hand (`uv run python evaluation/scripts/evaluate_layout_toc_classifier.py --save-results ...`)
whenever the classifier, its features, or the TOC ground truth changes --
the same trigger conditions `RESULTS.md`'s classifier section already
documents in prose.

**`report_html.py` changes** — `render_strategy_tables` gains one new
optional parameter:

```python
classifier: dict | None = None
# {
#   "label": "Layout/TOC classifier (LOBO, as of <date>)",
#   "note": "<one-sentence non-comparability caveat, rendered near the tables>",
#   "per_document": {book_key: {"toc_recall", "chapter_first_recall", "candidate_fraction"} | None},
#   "full_recall_fraction": float,
#   "avg_candidate_fraction": float,
# }
```

When given:
- Per-document table: one extra `<th>` (the `label`) and, per row, an
  extra `<td>` rendering `TOC recall=X%, chapter-first recall=Y%,
  candidates=Z%` (or the missing piece as `n/a` when a value is `null`),
  or `<td>N/A</td>` when the book has no entry in `per_document` at all
  (book wasn't part of the corpus when the classifier was last run).
- Per-strategy (aggregate) table: one extra row, using `label` as the
  first cell; the existing Precision/Recall/F1/Found-Expected/Time
  (/citation) columns render `N/A` for this row since they don't apply;
  two new columns are added to the table **only when `classifier` is
  given** -- "Full recall" and "Avg candidates" -- populated for the
  classifier row from `full_recall_fraction`/`avg_candidate_fraction` and
  `N/A` for every other (non-classifier) row.
- `note` is rendered as a small `<p>` directly above the per-strategy
  table (or appended to `description_html` by the caller -- caller's
  choice; simplest is `report_html.py` rendering it itself right before
  `<h2>Per strategy...</h2>` so it sits next to the table it explains).

This keeps the classifier's numbers inside the same two tables the other
strategies use (per the original ask), rather than a separate table,
while keeping its differently-shaped metric visually and semantically
distinct via its own cell format, its own aggregate columns, and the note.

**`generate_report.py` changes** — in `generate_corpus`, after loading
`expected_by_key`, attempt to load
`evaluation/corpus/<corpus>/classifier-results.json`; if present, build the
`classifier` dict per the shape above (label includes the file's own
`generated_at` date) and pass it to `render_strategy_tables`. If absent,
pass `None` (today's behavior, no classifier column at all) -- this is the
common case for `copyrighted-scans` until/unless someone runs the
classifier against that corpus and saves results for it too.

## Testing

- `tests/test_report_html.py`: add cases for the new `classifier` param --
  renders the extra column/row correctly, renders `N/A` for a book missing
  from `per_document`, omits the extra columns/row entirely when
  `classifier=None` (existing tests must keep passing unchanged).
- `tests/test_generate_report.py`: add a case with a fixture
  `classifier-results.json` present for a corpus, asserting the label/date
  and per-document values make it into the rendered HTML; add a case
  confirming the footer contains a `Generated on <date>` string; add a
  case confirming an `LLM (<model>, as of <date>)` label appears when
  cache entries carry the new per-model `generated_at`, and confirming the
  old (no per-model date) cache shape still renders (falls back to the
  bare `LLM (<model>)` label) without error.
- A small script-level test or manual invocation confirming
  `--save-results` writes the documented JSON shape (splitting correctly
  by corpus when multiple corpora are in scope).
