# Crossref-sourced ground-truth corpus for chapter segmentation

Status: approved for planning
Date: 2026-08-08

## Problem

The existing evaluation set (`evaluation/manifest.json` + `manifest.local.json`,
17 books total) is small and every ground-truth file
(`<name>.expected.json`) was built by hand: transcribe the TOC, run
`evaluation/scripts/ground_truth_helper.py` to locate each chapter's real PDF
page by content search, then verify every entry visually against the actual
PDF (`evaluation/CLAUDE.md`). That process doesn't scale, and it means the
set is small, skews toward one subject domain (socio-legal studies — see
`evaluation/README.md`'s "Evaluation set composition"), and none of it was
built from the Crossref book-chapter records the `CrossrefMetadataStrategy`
(`src/chapter_segmentation/evidence/crossref_strategy.py`) actually queries —
so that strategy's own accuracy has never been checked against ground truth
sourced from the same system it reads from.

Crossref itself already publishes, for many open-access books, exactly the
chapter metadata (title, authors, chapter DOI, citation page range) needed
for evaluation ground truth — no manual transcription required. This is a
large, mostly untapped source of free ground truth for open-access books.

## Goal

A script that builds a new, separate corpus of ~40-50 open-access books,
spanning diverse academic domains, by downloading each book's PDF and its
Crossref-registered chapter metadata. This corpus is not wired into the
existing evaluation harness yet — that's future work, once it's clear how
Crossref's citation page ranges should be reconciled with the harness's
`pdf_start_index`/`pdf_end_index` PDF-relative indices. For now it exists
standalone, for later integration.

## Non-goals

- Resolving each chapter's citation page range to a `pdf_start_index`/
  `pdf_end_index` (the `ground_truth_helper.py` content-search step). Left
  for a later integration pass.
- Wiring this corpus into `test_segmentation_accuracy.py`,
  `generate_report.py`, or any other existing harness entry point.
- Automated, unsupervised book discovery. The candidate list is
  hand-curated (see below) — this script only fetches and verifies what's
  already been chosen.

## Design

### Directory layout

A new `evaluation/crossref_gt/` directory, fully separate from
`evaluation/manifest.json`/`*.expected.json` (untouched by this work):

```text
evaluation/crossref_gt/
  README.md                  # schema + explicit "not yet wired into the harness" note
  manifest.json               # curated book list (committed)
  <isbn>.pdf                  # downloaded OA PDF (gitignored, like evaluation/*.pdf)
  <isbn>.crossref.json        # fetched Crossref metadata (committed — no copyrighted text)
```

`evaluation/.gitignore` gets a `crossref_gt/*.pdf` entry alongside the
existing `*.pdf` rule if it doesn't already cover the new subdirectory.

### `manifest.json` schema

```json
{
  "books": [
    {
      "isbn": "9783961102546",
      "title": "...",
      "doi": "10.5281/zenodo...",
      "domain": "linguistics",
      "language": "en",
      "publisher": "Language Science Press",
      "download_url": "https://.../book.pdf"
    }
  ]
}
```

Curated by hand (~40-50 entries, ~5-6 per domain across roughly 8 domains:
law, linguistics, computer science, medicine/public health, economics,
history, environmental science, education — exact mix depends on which
candidates actually pan out during research). Every entry must, by
construction, be open access with a working direct-download PDF URL and a
Crossref-registered DOI — candidates that don't satisfy this are dropped
during curation, not added with placeholder fields (unlike
`evaluation/manifest.json`, which deliberately supports non-OA entries this
corpus has no use for).

### `<isbn>.crossref.json` schema

```json
{
  "isbn": "9783961102546",
  "fetched_at": "2026-08-08T12:00:00+00:00",
  "raw_items": [ /* verbatim Crossref message.items entries for this ISBN, type=="book-chapter" only */ ],
  "chapters": [
    {"title": "...", "authors": ["..."], "chapter_doi": "10...", "citation_pages": "19-39"}
  ]
}
```

`raw_items` is the untouched Crossref API response items (the "full
metadata" the task asked for — whatever fields the query selects, kept
verbatim for anything a later integration pass might need that today's
normalized view doesn't capture). `chapters` is a convenience projection:
`title` follows `_parse_crossref_item`'s existing title+subtitle join
convention (`crossref_strategy.py`), `citation_pages` is the raw Crossref
`page` string as-is (e.g. `"19-39"`), not split/parsed.

### Script: `evaluation/scripts/fetch_crossref_gt_corpus.py`

Modeled on `evaluation/scripts/fetch_evaluation_pdfs.py`'s structure
(read manifest, iterate, skip existing, continue past per-book failures),
plus a Crossref fetch step:

1. For each book in the manifest:
   - If `<isbn>.pdf` doesn't exist (or `--force`): download from
     `download_url`, matching the existing script's approach (`httpx`,
     follow redirects, raise on HTTP error but don't abort the batch —
     catch and log per-book instead, since this script iterates ~40-50
     network calls to varied third-party hosts).
   - If `<isbn>.crossref.json` doesn't exist (or `--force`): query
     `https://api.crossref.org/works?filter=isbn:{isbn}` with
     `select=DOI,title,subtitle,author,page,type,container-title,published,
     ISBN` (same field list `_parse_crossref_item` already reads, plus
     `published`/`ISBN` since those cost nothing extra and round out "full
     metadata") and `mailto=<contact-email>` (default
     `boulanger@lhlt.mpg.de`, overridable via `--contact-email`), the same
     429-backoff/retry behavior as `fetch_crossref_chapters` in
     `crossref_strategy.py` (not imported — this script needs `raw_items`
     too, which that function's `ChapterCandidate` conversion discards),
     filter to `type == "book-chapter"`, and write both `raw_items` and the
     normalized `chapters` projection.
2. After fetching, print a warning (not a failure) for any book where:
   - Crossref returned zero `book-chapter` records, or
   - any returned chapter is missing a `page` value.
   These are exactly the signals used to decide, during curation, whether
   a candidate should be dropped and replaced.
3. Summary line at the end: books processed, PDFs downloaded vs. already
   present, chapters fetched total, books flagged for the warnings above.

### `evaluation/crossref_gt/README.md`

Short — schema description (mirroring this doc's two JSON shapes) plus an
explicit statement that this corpus is standalone and not yet consumed by
`test_segmentation_accuracy.py` or `generate_report.py`.

## Testing

No new pytest coverage — this is a data-fetching utility script in the same
category as `fetch_evaluation_pdfs.py` and `ocr_evaluation_pdfs.py`, neither
of which has tests; both are exercised by running them against real network
resources. Manual verification: run the script, confirm PDFs and
`.crossref.json` files land correctly, spot-check a few `.crossref.json`
files' `chapters` against the book's real TOC, confirm the warning path
fires for a deliberately-bad ISBN.

## Follow-up work (explicitly out of scope here)

- Reconciling `citation_pages` to `pdf_start_index`/`pdf_end_index` so this
  corpus can produce real `.expected.json` files and join the existing
  harness.
- Deciding whether the CrossrefMetadataStrategy's own evaluation against
  this corpus should compare citation-page ranges directly (trivially ~1.0
  F1, since both sides read the same Crossref data) versus PDF-index ranges
  (requires the reconciliation step above, and is the only fair way to also
  use this corpus for the heuristic/outline/LLM strategies).
