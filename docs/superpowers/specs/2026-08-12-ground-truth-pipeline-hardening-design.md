# Ground-truth pipeline hardening: shared bounds check, shared license lookup, promotion script

Status: approved for planning
Date: 2026-08-12

## Problem

Two pieces of the evaluation ground-truth workflow currently only exist as
ad-hoc, hand-run code rather than as part of the standard tooling:

1. **The chapter bounds/overlap sanity check.** `evaluation/CLAUDE.md`'s
   Step 4 documents it as a `python -c "..."` one-liner to copy-paste
   before committing a new book. It also already exists, independently
   re-implemented, as a private `_sanity_check()` inside
   `evaluation/scripts/build_crossref_gt_ground_truth.py`, used only on
   that script's own crossref_gt-to-pending/ migration path. Neither
   copy runs automatically against the corpus as a whole. This gap is
   exactly how a 2-page overlap in `9782821895607.expected.json`
   (open-access/) and a 1-page overlap in `9783428042241.expected.json`
   (copyrighted-scans/) both went undetected until manually re-run by
   hand during this spec's own investigation.
2. **Open-access license lookup (Crossref, with Unpaywall fallback).**
   Lives as private (`_`-prefixed) functions inside
   `evaluation/scripts/fetch_crossref_gt_corpus.py`.
   `evaluation/scripts/discover_crossref_candidates.py` already reaches
   across the module boundary to import those private names directly.
   Promoting a book from `pending/` into `open-access/` -- which requires
   resolving `license`/`license_source` per `evaluation/README.md`'s
   manifest schema -- currently has no supported way to do this at all;
   the only precedent is a throwaway script written by hand for one
   migration.

Additionally, there is no script for the move
`evaluation/CLAUDE.md`'s Step 0a describes ("once its `.expected.json`
exists... the entry moves into whichever real corpus it belongs in") --
it's purely a manual file-move + manifest edit today.

## Goal

- A shared, single-implementation bounds/overlap check that runs
  automatically on every default `uv run pytest` invocation (no PDFs
  required for the overlap-among-chapters part), in addition to being
  used as a hard gate wherever ground truth is written or moved.
- A shared, public license-lookup module, ending the private-name
  cross-import between `fetch_crossref_gt_corpus.py` and
  `discover_crossref_candidates.py`.
- A `promote_pending_book.py` script that performs the pending/ ->
  open-access/ (or -> copyrighted-scans/) move using both of the above,
  closing the gap that caused this session's ad-hoc work.

## Non-goals

- Changing where new ground truth is *authored* (`ground_truth_helper.py`,
  `add_toc_ground_truth.py`, or the manual CLAUDE.md workflow) -- this
  spec only changes what happens to a book already in `pending/` with a
  finished `.expected.json`.
- Handling `pending/manifest.local.json` entries in the promotion script.
  A pending book with no DOI has no license to resolve and, per
  `evaluation/CLAUDE.md`'s Step 0b, wouldn't have a `public-cache/` entry
  either -- promoting such a book is rare enough, and different enough
  (it needs a human decision about whether it can ever be shared), that
  it stays a manual operation. The script errors clearly if asked to
  promote an ISBN it can't find in the corpus's *committed*
  `manifest.json`.
- Retrying/hardening the Crossref or Unpaywall HTTP calls beyond what
  `evaluation/scripts/fetch_crossref_gt_corpus.py` already does today
  (429 backoff, never-raises-on-failure). The moved code keeps its
  existing resilience behavior unchanged.
- A `--corpus copyrighted-scans` license lookup. That schema has no
  `license`/`license_source` fields (`evaluation/README.md`) and this
  spec doesn't add any.

## Design

### Part 1: shared bounds/overlap validator

New function in `evaluation/harness.py` (the existing shared home for
corpus-loading logic used by both `evaluation/scripts/` and `tests/`):

```python
def chapter_bounds_errors(chapters: list[dict], total_pages: int | None = None) -> list[str]:
    """Structural sanity check on one book's ground-truth chapter ranges:
    every pdf_start_index <= pdf_end_index, no two chapters' ranges
    overlap, and -- only when total_pages is given -- every
    pdf_end_index < total_pages. Returns human-readable problem
    descriptions, empty list if none. Needs no PDF unless total_pages is
    passed, so it can run on every corpus even before any PDF is fetched
    locally."""
```

Logic ports directly from `build_crossref_gt_ground_truth.py`'s
`_sanity_check` (sort ranges by start, walk pairwise) and from the
`CLAUDE.md` Step 4 one-liner (start<=end, end<total_pages), unified into
one implementation that returns *all* problems found, not just the first
(the one-liner used bare `assert` and stopped at the first failure;
`_sanity_check` returned only one message string). `build_crossref_gt_ground_truth.py`'s
private `_sanity_check` is deleted; its one call site switches to
`harness.chapter_bounds_errors(...)`, joining the returned list into its
existing single `"SKIP: ..."` message.

### Part 2: shared license-lookup module

New `evaluation/oa_license.py` (sibling to `harness.py`, same rationale:
`evaluation/scripts/*.py` must not depend on each other's internals any
more than they depend on the test tree). Moved and made public (dropping
the leading underscore) from `fetch_crossref_gt_corpus.py`:

- `crossref_book_chapter_items(isbn, client, contact_email) -> list[dict]`
  (was `_crossref_book_chapters`) -- GETs
  `.../works?filter=isbn:{isbn}`, returns raw `type=="book-chapter"` items.
- `item_license_url(item: dict) -> str | None` (was `_item_license_url`)
  -- one Crossref item's registered OA license URL, preferring the
  version-of-record entry.
- `book_license_url(raw_items: list[dict]) -> str | None` (was
  `_book_license_url`) -- majority vote across a book's chapters.
- `unpaywall_license_url(doi, client, contact_email) -> str | None` (was
  `_unpaywall_license_url`) -- Unpaywall fallback, SPDX-code-to-URL
  mapping included.
- New: `resolve_license(isbn, doi, client, contact_email) -> tuple[str | None, str | None]`
  -- the one call the promotion script needs: fetches the book's Crossref
  chapter items, tries `book_license_url`, falls back to
  `unpaywall_license_url` if that's `None`, returns `(license_url,
  license_source)` with `license_source` in `{"crossref", "unpaywall",
  None}`. Never raises -- a failed lookup returns `(None, None)`, matching
  the existing non-fatal convention in this codebase's Crossref/Unpaywall
  calls.

`fetch_crossref_gt_corpus.py` keeps its own use of
`crossref_book_chapter_items` (it needs the raw items for the chapters
list too, not just the license) but imports it and the two license
helpers from `evaluation.oa_license` instead of defining them locally.
`discover_crossref_candidates.py` switches its existing private-name
cross-import to the same public names. Neither script's behavior changes.

### Part 3: `evaluation/scripts/promote_pending_book.py`

```
uv run python evaluation/scripts/promote_pending_book.py <isbn> [<isbn> ...] \
    --corpus {open-access,copyrighted-scans} [--contact-email EMAIL] [--dry-run]
```

For each ISBN, in order, printing one `(isbn, outcome)` line each (same
shape as `build_crossref_gt_ground_truth.py`'s reporting) and a final
`N/M promoted` summary:

1. Look up the entry in `evaluation/corpus/pending/manifest.json`
   (`evaluation.harness.load_manifest_books` would also merge in
   `manifest.local.json` -- read `manifest.json` directly instead, since
   this script must reject those; see Non-goals). Missing -> `"SKIP: not
   in pending/manifest.json"`.
2. Require `pending/<isbn>.expected.json` to exist. Missing -> `"SKIP: no
   ground truth yet"`.
3. **Gate:** read `pending/<isbn>.pdf`'s page count (`pypdf.PdfReader`),
   load the `.expected.json` chapters, call
   `harness.chapter_bounds_errors(chapters, total_pages)`. Any error(s) ->
   print them, `"SKIP: bounds/overlap check failed"`, no files touched
   for this ISBN, continue to the next one.
4. If `--corpus open-access`: open an `httpx.Client`, call
   `oa_license.resolve_license(isbn, doi, client, contact_email)`. If it
   returns `(None, None)`, print a warning and continue anyway (writes
   `"license": null, "license_source": null"` -- allowed by the schema,
   see `evaluation/README.md`), rather than skip -- this is metadata
   *nice-to-have*, not a correctness gate like Part 1's check. If
   `--corpus copyrighted-scans`, skip this step entirely (no license
   fields for that corpus).
5. Unless `--dry-run`: `mv` the `.pdf` and `.expected.json` from
   `pending/` into the target corpus directory; append the manifest entry
   (with `license`/`license_source` attached if step 4 ran) to the
   target's `manifest.json`, sorted by `filename`; remove the entry from
   `pending/manifest.json`. With `--dry-run`, print what would happen
   (target path, resolved license) and touch nothing.

## Testing

- New `tests/test_ground_truth_integrity.py`, **not** marked
  `integration` (so it's part of the default `uv run pytest` run,
  same as every other non-PDF-dependent test). For every corpus
  (`evaluation.harness.list_corpora()`) and every book in
  `load_manifest_books(corpus)` that has a `.expected.json` on disk:
  load its chapters; if the corpus's `.pdf` is also present locally, read
  its page count and pass `total_pages`, otherwise pass `None` (still
  checks start<=end and no-overlap even with zero PDFs downloaded). One
  `subTest` per book; assert `chapter_bounds_errors(...) == []`.
- `evaluation/oa_license.py`'s pure functions (`item_license_url`,
  `book_license_url`) get direct unit tests in
  `tests/test_oa_license.py` using small literal Crossref-shaped dicts --
  no network calls. `unpaywall_license_url`/`resolve_license` are
  exercised indirectly (existing scripts already call them against the
  real network; no new network-dependent tests are added by this spec).
- `promote_pending_book.py` gets a `tests/test_promote_pending_book.py`
  covering the two gates against a temp-directory fake corpus (missing
  manifest entry, missing `.expected.json`, a deliberately-overlapping
  `.expected.json` refusing to promote) plus one happy-path run with
  `--dry-run` confirming no files move and the reported plan is correct.
  The real-network license path is smoke-tested manually (documented in
  the script's own docstring), not asserted in the test suite, matching
  how `fetch_crossref_gt_corpus.py`'s tests (there are none today) treat
  network calls.

## Documentation updates

- `evaluation/CLAUDE.md` Step 0a: replace "the entry moves into whichever
  real corpus it belongs in" (currently undescribed *how*) with a pointer
  to `promote_pending_book.py`.
- `evaluation/CLAUDE.md` Step 4: keep the manual one-liner as a quick
  spot-check a human can still run by hand, but add a note that the same
  check now also runs automatically via `tests/test_ground_truth_integrity.py`.
