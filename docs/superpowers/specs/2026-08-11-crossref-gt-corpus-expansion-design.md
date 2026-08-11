# Automated Crossref-sourced GT corpus expansion

Status: approved for planning
Date: 2026-08-11

## Problem

`evaluation/crossref_gt/manifest.json` (43 books) and the migration it feeds,
`evaluation/scripts/build_crossref_gt_ground_truth.py`, are both fully
manual on the discovery side: a human finds each candidate book, checks its
OA status and download URL, and adds it to the manifest by hand. Migration
itself is already automatic and high-confidence (offset-consensus +
content-search confirmation, >=80% of chapters / >=3 chapters required; 31
of 43 books have cleared that bar with no manual GT verification).

The layout-based TOC/chapter-first-page classifier pilot
(`docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`,
`evaluation/RESULTS.md`) found via a learning-curve experiment that generic
corpus growth (10-35 training books) produces a flat `full_recall_fraction`
trend -- no benefit from volume alone. The higher-leverage target it
identified: books whose chapter-first pages sit in layout-feature regions
the current corpus doesn't already cover (its two named examples:
unnumbered first chapters, and chapter-openings with no font-size/
whitespace distinction from body text). Manually hunting for such books is
slow and there's no way to check a candidate's actual feature-space
novelty before spending time on it.

## Goal

Automate both discovery and admission:

1. **Discovery**: find new open-access edited-volume candidates via
   Crossref (seeded from publishers already represented in the manifest),
   resolving each candidate's direct PDF URL via Crossref/Unpaywall/
   OpenAlex, with no manual searching.
2. **Admission**: after the existing offset-consensus/content-search
   confirmation succeeds, gate migration into `evaluation/corpus/pending/`
   (not `open-access/` -- see "Landing in `pending/`, not `open-access/`"
   below) on a new **novelty check** -- a candidate's confirmed
   chapter-first pages must include at least one page whose layout
   features sit meaningfully outside the region the current corpus already
   covers. Books that pass confirmation but bring nothing new are left
   unmigrated rather than added as redundant volume.

Both discovery and admission must preserve and reinforce **linguistic
diversity** (see "Linguistic diversity" below) -- the point of targeted
growth is defeated if automation quietly re-converges on English-language,
Latin-typeset volumes because those are easiest to find and match.

## Non-goals

- Changing the layout classifier pilot's decision bar, model, or features.
- Outline-based *rescue* of chapters/books that fail the existing
  content-search confirmation bar. This round adds outline cross-checking
  as **diagnostic-only** logging; whether to use it to rescue currently-
  stuck books is a follow-up decision, made from real data this round
  produces, not a guess made now.
- A draft-`.expected.json`-from-outline-only fallback tier, or reverting to
  the fully manual `evaluation/CLAUDE.md` workflow as an automated
  fallback path. Both remain available as manual options if a future
  discovery run's yield turns out too low to be worth automating further;
  neither is built in this round.
- Auto-promoting a book from `pending/` to `open-access/`. That stays a
  manual step (same promotion `evaluation/CLAUDE.md` already documents for
  hand-built `pending/` entries), made once a human has actually reviewed/
  tested the book -- this round only gets it into `pending/` automatically.
- Re-deriving or hand-verifying any of the 31 already-migrated
  `open-access/` books. This work only affects newly discovered candidates,
  and never touches anything already in `open-access/`.

## Design

### Directory / manifest changes

`evaluation/crossref_gt/manifest.json` entries gain one new field,
`discovery_source`: `"manual"` for the 43 existing entries (backfilled),
`"auto"` for anything `discover_crossref_candidates.py` adds. Purely for
provenance/debugging -- nothing reads it to change behavior.

### `discover_crossref_candidates.py` (new script)

Seed list, one entry per publisher already represented in the manifest,
each carrying a Crossref `member` ID and a `default_language` (used only as
a fallback when Crossref's own `language` field is absent -- see
"Linguistic diversity" below):

```python
_SEED_PUBLISHERS = [
    {"member_id": "...", "publisher": "Open Book Publishers", "default_language": "en"},
    {"member_id": "...", "publisher": "transcript Verlag", "default_language": "de"},
    {"member_id": "...", "publisher": "UCL Press", "default_language": "en"},
    {"member_id": "...", "publisher": "Athabasca University Press", "default_language": "en"},
    {"member_id": "...", "publisher": "Springer", "default_language": "en"},
    {"member_id": "...", "publisher": "Presses universitaires de Rennes", "default_language": "fr"},
    {"member_id": "...", "publisher": "Africae", "default_language": "fr"},
]
```

(Exact `member_id` values looked up from Crossref's member API during
implementation -- not guessed here.)

For each seed publisher, query
`https://api.crossref.org/works?filter=member:<id>,type:monograph`
(and a second pass with `type:edited-book`), paginated via `offset`, same
429-aware retry loop already used by `fetch_crossref_gt_corpus.py` and
`crossref_strategy.py`. For each returned work:

1. Extract ISBN (Crossref `ISBN` field), DOI, title, `language`.
2. Skip if the ISBN or DOI already appears in `manifest.json` (dedup).
3. Resolve `download_url` by trying, in order, and recording which one
   answered as `license_source`-style provenance:
   - Crossref's own `link` array (`content-type: application/pdf`).
   - Unpaywall (`https://api.unpaywall.org/v2/{doi}`) `best_oa_location.url_for_pdf`.
   - OpenAlex (`https://api.openalex.org/works/doi:{doi}`)
     `best_oa_location.pdf_url`.
   Drop the candidate if none of the three yields a URL -- there is nothing
   to fetch.
4. Resolve `license` the same way `fetch_crossref_gt_corpus.py` already
   does (Crossref first, Unpaywall fallback) -- reuse those functions
   directly rather than re-implementing.
5. Append the surviving candidate to `manifest.json` with
   `discovery_source: "auto"`.

Never aborts the batch on one publisher's or one work's failure (log and
continue), matching every existing script in this pipeline.

### Linguistic diversity

**Discovery side.** Before appending new candidates, compute the current
language distribution across `evaluation/crossref_gt/manifest.json` +
`evaluation/corpus/open-access/manifest.json` (tally the `language` field).
Rank languages by ascending count (most underrepresented first). When more
candidates are found than are worth adding in one run, process them in that
priority order, and cap how many candidates of an already-well-represented
language get appended per run (`--max-per-language`, default 5) so one
prolific publisher (e.g. transcript Verlag's German catalog) can't
monopolize a single discovery run. A candidate's `language` comes from
Crossref's own field when present; when Crossref has none, fall back to
the seed publisher's `default_language` (a prior, not a guess from title
text -- no new language-detection dependency needed).

**TOC-pattern-matching side.** Checked during this design's research: every
structural pattern the pipeline depends on is already language-agnostic --
`ground_truth_helper._TOC_LINE_RE` (`"title ... number"` line shape),
`_PAGE_NUM_RE`/`_TRAILING_NUM_RE`/`_LEADING_NUM_RE` (digit or roman-numeral
matching only), and `_derive_offset`'s consensus-vote logic all key off
page-number shape, not language-specific words. No code changes are needed
here; this is confirmed by a new test (see "Testing") that exercises
`find_toc_pages`/`extract_printed_number` against synthetic German- and
French-labeled TOC/page-number text, guarding against a future change
silently introducing an English-only assumption (e.g. a keyword-based
"Contents" search). The one place a literal keyword list *is* introduced --
the new outline-title diagnostic cross-check below -- is scoped to fuzzy-
match the book's own chapter title text (whatever language it's actually
in), not a fixed keyword list, so it inherits the same language-agnosticism
by construction.

### `fetch_crossref_gt_corpus.py` -- unchanged

Already downloads any manifest entry not yet on disk (PDF + Crossref
metadata), skips existing files, never aborts the batch on one book's
failure. `discovery_source` is simply an extra field it ignores.

### `build_crossref_gt_ground_truth.py` -- extended

After the existing confirmation gate passes unchanged
(`_MIN_CONFIRMED_FRACTION=0.8`, `_MIN_CONFIRMED_CHAPTERS=3`, no changes to
either constant or to `_derive_offset`/`_locate_near`):

1. Run `pdfalto_runner.resolve_pdfalto_binary` +
   `layout_features.extract_page_features` on the candidate PDF (same
   extraction path `evaluate_layout_toc_classifier.py`'s
   `build_feature_table` already uses), and take the feature vector for
   each confirmed chapter's `pdf_start_index` page.
2. Load the current `evaluation/corpus/open-access/` corpus's
   `LABEL_CHAPTER_FIRST` rows (reusing
   `evaluate_layout_toc_classifier.load_book_corpus`/`build_feature_table`,
   restricted to `open-access`), and fit a `StandardScaler` on its full row
   set -- the same normalization the classifier itself already uses, so
   distances are computed in the same space it reasons in.
3. For each candidate chapter-first page, compute its nearest-neighbor
   Euclidean distance (in scaled space) to the existing corpus's
   chapter_first vectors. Compute a novelty threshold as a percentile
   (default 90th, `--novelty-percentile` CLI flag) of the existing corpus's
   own leave-one-out nearest-neighbor distances -- pages farther than that
   sit outside how tightly the current corpus already clusters.
4. **Migrate into `evaluation/corpus/pending/` (not `open-access/`) only if
   at least one candidate chapter-first page's distance exceeds the
   threshold.** Otherwise skip with a new `"SKIP: not novel (nearest
   chapter-first distance <threshold>)"` message (distinct from the
   existing confirmation-failure message so the two rejection reasons are
   never conflated in the report), leaving the PDF and `.crossref.json` in
   `crossref_gt/` for visibility -- exactly like a confirmation-failure
   skip today. The migrated manifest entry (appended to
   `evaluation/corpus/pending/manifest.json`, same schema
   `process_book` already writes) additionally carries `"embedded_toc"`
   from `bool(reader.outline)` exactly as it does for `open-access/` today.
5. **Diagnostic-only outline cross-check**: flatten `reader.outline`
   (nested `Destination` entries) into `(title, page_index)` pairs via
   `reader.get_destination_page_number`. For each confirmed chapter,
   fuzzy-match its Crossref title against outline entry titles
   (`rapidfuzz.fuzz.partial_ratio`, same library already used by
   `_locate_near`); if the best match's page disagrees with the
   content-search-confirmed `pdf_start_index`, log it in the per-book
   report as `outline disagreement: chapter "<title>" outline=<page>
   content-search=<page>`. This has **no effect on the migration
   decision** -- purely informational, to build evidence for whether an
   outline-based rescue mechanism would be worth adding in a later round.

### TOC field -- unchanged

`_toc_field_for` already handles this inline via the same structural
`find_toc_pages`/`toc_page_range` functions confirmed language-agnostic
above.

### Landing in `pending/`, not `open-access/`

`evaluation/corpus/pending/` already exists as a defined tier
(`evaluation/CLAUDE.md` step 0a: "no ground truth built yet... move into
`open-access/` once its `.expected.json` exists") -- this reuses it exactly
as designed, just with the `.expected.json` arriving automatically instead
of by hand. Confirmation + novelty already give high GT confidence, but
`open-access/` is what the pilot's canonical numbers (`evaluation/
RESULTS.md`) are measured against; landing new books there straight from
an unreviewed automated run would make every future pilot re-run
unreviewable-by-diff. `pending/` gives a human a chance to spot-check a
book (or run the pilot against it in isolation) before it affects the
numbers everyone already relies on.

`evaluate_layout_toc_classifier.py` gets a new `--corpora` CLI flag
(comma-separated, default `"open-access,copyrighted-scans"`, replacing the
hardcoded module-level `_CORPORA` list as the default) so a book in
`pending/` can actually be evaluated before promotion --
`--corpora open-access,pending` runs the full LOBO pilot with the
candidate included; `--corpora pending` (if more than one pending book
exists) isolates just the new arrivals. Promotion itself
(move PDF + `.expected.json`, merge the manifest entry into
`open-access/manifest.json`) stays the manual step
`evaluation/CLAUDE.md` already documents -- not automated in this round
(see "Non-goals").

## Testing

- `discover_crossref_candidates.py`: unit tests against mocked HTTP
  responses (`httpx` mock transport, matching the existing test style) for
  dedup logic (ISBN/DOI already in manifest), the three-source
  `download_url` fallback order, and the language-priority ranking/
  `--max-per-language` cap -- all deterministic, no live network.
- `build_crossref_gt_ground_truth.py` extension: extend
  `tests/test_build_crossref_gt_ground_truth.py` (currently covers
  `_toc_field_for` only) with synthetic feature-vector fixtures for the new
  novelty-gate function (deterministic distances on either side of a fixed
  threshold), and a synthetic-outline fixture for the diagnostic
  cross-check's agreement/disagreement logging.
- New regression test confirming `find_toc_pages`/`extract_printed_number`
  correctly detect TOC-shaped lines and page numbers in synthetic German-
  and French-labeled text (e.g. `"Einleitung .......... 7"`,
  `"Introduction .......... 7"`) -- guards the "already language-agnostic"
  claim above against future regressions.
- `evaluate_layout_toc_classifier.py`'s new `--corpora` flag: a unit test
  confirming it parses a comma-separated list and that `load_book_corpus`
  restricts itself to exactly those corpus directories (using the existing
  synthetic-fixture style in `tests/test_evaluate_layout_toc_classifier.py`).
- No live-network end-to-end test, consistent with every other script in
  this pipeline (`fetch_crossref_gt_corpus.py` has none either).

## Follow-up work (explicitly out of scope here)

- Deciding, from real disagreement-rate data this round's diagnostic
  logging produces, whether outline agreement should be allowed to rescue
  chapters/books that fail content-search confirmation.
- A draft-`.expected.json`-from-outline-only fallback tier, for candidates
  with a good outline but insufficient Crossref chapter-page data, if
  Tier-1 (Crossref + content-search + novelty) yield from real discovery
  runs turns out too low.
- Re-running the layout classifier pilot's full LOBO evaluation once a
  meaningful number of new books have been migrated, to check whether
  targeted growth actually moves `full_recall_fraction` (the empirical
  question the flat learning curve left open).
