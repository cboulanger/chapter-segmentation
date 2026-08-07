# Per-strategy evaluation and reporting

Status: approved for planning
Date: 2026-08-07

## Problem

`evaluation/generate_report.py` only ever measures `analyze_attachment` (the
pure heuristic pipeline). The two other orchestration entry points --
`analyze_attachment_with_llm_fallback` and `analyze_attachment_with_strategies`
-- are strict pipelines: each makes an internal, all-or-nothing decision
about which strategy's output to trust (e.g. `analyze_attachment_with_strategies`
falls back to the heuristic/LLM pipeline entirely if its metadata-strategy
merge comes back empty). That decision can be wrong, and today there is no
way to see how any individual strategy (outline read, LLM extraction, etc.)
performs on its own against the evaluation corpus -- only the pipeline's
net result. This makes it impossible to tell whether a regression comes
from a bad strategy or a bad "which strategy wins" decision.

Goal: evaluate each strategy independently against the same corpus and
ground truth, report per-document and per-strategy (aggregate) results
including timing, and add a manually-triggered, cost-bearing LLM
evaluation whose results get cached and folded into the automated report.

## Strategies covered

Four of the five strategies that exist in `chapter_segmentation`:

1. **Heuristic** (`analyze_attachment`) -- regex TOC scan + fuzzy content
   locate. Existing, unchanged.
2. **Outline** (new `analyze_attachment_outline_only`) -- PDF embedded
   bookmarks.
3. **LLM** (new `analyze_attachment_llm_only`) -- unconditional LLM TOC
   extraction + locate + LLM disambiguation. Manually triggered only (costs
   money); results are cached.
4. Crossref and Zotero-catalog metadata strategies are explicitly **out of
   scope** for this evaluation rewrite: Crossref needs a live network call
   per run and Zotero-catalog needs a real Zotero library, neither of which
   the public, PDF-free evaluation corpus can provide. They remain
   evaluable only through the existing manual
   `evaluate_chapter_segmentation_strategies.py` script, unchanged.

## Production code changes

`src/chapter_segmentation/segmentation.py` gains two new orchestration
functions, both extracted from logic that already exists inside
`analyze_attachment_with_strategies` / `analyze_attachment_with_llm_fallback`
so evaluation exercises the same code paths production does:

- `analyze_attachment_outline_only(pages: list[str], outline_candidates: list[ChapterCandidate]) -> dict`
  Builds full chapter dicts (bounds, citation pages, confidence) from an
  already-extracted outline candidate list -- the same "pre-located
  candidate -> chapter" logic `analyze_attachment_with_strategies` uses for
  candidates that already carry a `pdf_page_index`, factored out so it can
  be called directly. Takes candidates rather than raw PDF bytes so the
  same function works whether candidates come from a live
  `extract_outline_candidates(file_bytes)` call or a cached JSON snapshot.

- `analyze_attachment_llm_only(pages: list[str], llm_client: LLMClient) -> dict`
  Mirrors `analyze_attachment_with_llm_fallback`'s LLM code paths but drops
  the "only if heuristic found nothing" gate -- `llm_extract_toc_entries`
  always runs, followed by locate and (for genuinely ambiguous entries)
  `llm_disambiguate_chapter_start`. This measures the LLM strategy's true
  standalone accuracy, which the existing fallback pipeline structurally
  hides (LLM only ever engages when the heuristic has already failed).

## Evaluation corpus changes

The outline strategy needs raw PDF bytes (`extract_outline_candidates`
reads `reader.outline` via pypdf), but `public-cache/` intentionally holds
only redacted page **text**, not PDF bytes, so CI never needs real books.

`evaluation/scripts/generate_public_evaluation_cache.py` is extended to
also write `public-cache/<manifest_key>.outline.json` -- the *resolved*
`ChapterCandidate` list (title, authors, pdf_page_index, printed page
number; no prose) produced by running `extract_outline_candidates` once,
locally, against the real PDF. This is the same sensitivity level as the
already-committed `.expected.json` ground truth. Books whose PDF has no
outline, or where the outline strategy returns `[]`, get an empty/absent
entry. `evaluation/harness.py` gains `public_outline_candidates_for
(manifest_key) -> Optional[list[ChapterCandidate]]` to load it.

## LLM results cache

`evaluation/llm-cache/<manifest_key>.json`, committed to git:

```json
{
  "generated_at": "2026-08-07T12:00:00Z",
  "models": {
    "<model-id>": {
      "chapters": [ ...same shape as analyze_attachment's "chapters"... ],
      "elapsed_seconds": 12.3,
      "demand_at_run": 0
    }
  }
}
```

Raw results are cached (not just scores) so metrics can be recomputed if
`*.expected.json` ground truth is later corrected, without spending another
API call. Multiple refresh runs upsert by model id -- a model dropped from
the current "top 5 non-busy" selection in a later run keeps its last cached
result rather than being deleted, so the cache can grow to hold more than 5
models over time. This is expected; it is a report input, not something
that needs pruning as part of this change.

## Metrics and rendering (shared code)

- `evaluation/metrics.py` (new): `precision_recall_f1(expected, found)` and
  a micro-average accumulator (pools tp/found/expected counts across
  documents before computing precision/recall/F1 -- matches today's
  aggregate style). F1 is the ranking/best-marking metric throughout.
- `evaluation/report_html.py` (new): one shared table-renderer used by both
  pages, producing:
  1. **Per-document x strategy table** -- rows = documents, columns =
     strategies, each cell shows precision/recall/F1/found-expected/time;
     the highest-F1 cell in each row is visually marked (bold + background
     tint).
  2. **Per-strategy aggregate table** -- one row per strategy, micro
     precision/recall/F1 and *summed* time-spent across all documents,
     rows ordered by F1 descending.

## Report generation

`evaluation/generate_report.py` (rewritten; still runs on every push, still
zero API calls, zero network calls):

- For each book in `available_public_books()`, runs **heuristic** (timed)
  and, if an outline cache entry exists, **outline** (timed) live.
- If `evaluation/llm-cache/<key>.json` exists for a book: computes each
  cached model's micro-F1 aggregate *across the whole corpus*, picks the
  single best-performing model (ties broken by lower total time), and adds
  an "LLM (`<model-id>`)" column to the per-document table and row to the
  aggregate table using that model's cached per-book results. This is one
  consistent model across all books, not a per-book cherry-pick of
  whichever model did best on that specific book -- the latter would not
  reflect a deployable choice.
- Also regenerates `public/llm/index.html`: the same two-table format but
  scoped to LLM models only, one column/row per *every* cached model (not
  just the winner) -- full detail, still free since it only reads the
  cache.
- Writes both `public/index.html` and `public/llm/index.html` every run.

## LLM cache refresh (the only cost-bearing path)

`evaluation/refresh_llm_cache.py` (repurposed from today's
`evaluate_chapter_segmentation_llm_fallback.py` / `evaluate_chapter_segmentation_strategies.py`
LLM-invocation pattern):

- Reads `KISSKI_API_KEY` (and optionally `KISSKI_BASE_URL`, default
  `https://chat-ai.academiccloud.de/v1`) from the environment. Locally,
  source it from `zotero-rag/.env`; in CI it comes from a repo secret.
- `evaluation/kisski.py` (new): `fetch_kisski_models(base_url, api_key)` --
  `POST {base_url}/models`, same request shape as zotero-rag's
  `fetch_kisski_rag_models` -- returns id/name/demand per model, classified
  `available` (demand==0) / `busy` (demand<=5) / `very busy`.
  `select_models(models, n=5)` -- takes all `available` first, fills
  remaining slots from `busy` (skipping `very busy`), in ascending-demand
  order.
- For each selected model, runs `analyze_attachment_llm_only` against
  every book in the full public corpus (same corpus as the main report),
  times it, and upserts the result into that book's `llm-cache/*.json`.
- Not a pytest test; run manually:
  `KISSKI_API_KEY=... uv run python evaluation/refresh_llm_cache.py`.

## CI / workflows

- `.github/workflows/publish-results.yml` -- unchanged trigger (push to
  main). Its single step (`generate_report.py --out public/`) now also
  produces `public/llm/index.html` as part of the same run, from whatever
  is currently committed in `llm-cache/`.
- New `.github/workflows/refresh-llm-cache.yml` -- `workflow_dispatch`
  only. Requires a `KISSKI_API_KEY` repository secret (to be added
  manually by a repo admin -- not something this change can do on its
  own). Runs `refresh_llm_cache.py`, then commits and pushes the updated
  `evaluation/llm-cache/*.json` files directly to `main`. That push
  triggers `publish-results.yml` normally, republishing with fresh LLM
  numbers automatically.

## Testing

- `tests/test_segmentation_accuracy.py` and
  `tests/test_public_evaluation_cache_parity.py` continue to exercise only
  `analyze_attachment` (unchanged) -- they are regression gates for the
  heuristic pipeline specifically, not general strategy benchmarks.
- New unit tests for `analyze_attachment_outline_only` and
  `analyze_attachment_llm_only` (with a fake `LLMClient`) covering the
  extraction/refactor itself, plus tests for `evaluation/metrics.py`'s
  precision/recall/F1 and micro-aggregation, and `evaluation/kisski.py`'s
  demand-classification and model-selection logic (available-first,
  fill-from-busy, skip-very-busy, dedupe).
- `evaluate_chapter_segmentation_strategies.py` (Crossref/Zotero-catalog,
  out of scope here) is left as-is.

## Out of scope

- Crossref and Zotero-catalog strategies (see "Strategies covered" above).
- Any change to `analyze_attachment_with_strategies`'s internal merge/
  fallback decision logic -- this change adds visibility into individual
  strategies, it does not change how the production pipeline picks among
  them.
- Pruning `evaluation/llm-cache/` -- left for a future change if the file
  count/size becomes a real problem.
