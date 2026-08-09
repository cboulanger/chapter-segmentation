# Crossref-sourced ground-truth corpus

A corpus of 46 open-access books, each with its Crossref-registered
`book-chapter` metadata, built for evaluating chapter-segmentation
strategies against ground truth sourced from the same system
`CrossrefMetadataStrategy` (`src/chapter_segmentation/evidence/crossref_strategy.py`)
actually queries. Background and design rationale:
`docs/superpowers/specs/2026-08-08-crossref-gt-corpus-design.md`.

**Status: standalone; 31 of 46 books reconciled into `evaluation/corpus/
open-access/`.** This corpus itself is still not read by
`tests/test_segmentation_accuracy.py`, `evaluation/generate_report.py`, or
any other harness entry point -- but the reconciliation the design spec's
"Follow-up work" deferred (mapping each chapter's Crossref
`citation_pages`, a printed page range, to the harness's PDF-relative
`pdf_start_index`/`pdf_end_index`) has been done for the books where it
could be done with high confidence, via
`evaluation/scripts/build_crossref_gt_ground_truth.py`. That script
derives each book's printed-page-number-to-PDF-index offset by consensus
vote across all pages, maps each chapter's Crossref start page through it,
and confirms the result with a title/byline content-search match; a book
is only migrated (PDF + `.expected.json` + manifest entry copied into
`open-access/`) when at least 80% of its page-bearing chapters (and at
least 3) confirm this way. The other 15 books didn't clear that bar
(usually multi-part pagination resetting the offset partway through the
book, or short/generic chapter titles that don't fuzzy-match reliably) and
remain here, unmigrated, for possible manual curation later.

## Directory contents

```text
evaluation/crossref_gt/
  README.md                  # this file
  manifest.json               # curated book list (committed)
  <isbn>.pdf                  # downloaded OA PDF (gitignored, like evaluation/*.pdf)
  <isbn>.crossref.json        # fetched Crossref metadata (committed)
```

Fetch/refresh the PDFs and metadata with:

```bash
uv run python evaluation/scripts/fetch_crossref_gt_corpus.py
uv run python evaluation/scripts/fetch_crossref_gt_corpus.py --force  # refetch everything
```

The script skips any file that already exists (so a partial run resumes
cleanly), never aborts the batch on one book's failure, and prints a
warning for any book with zero `book-chapter` records or a chapter
missing its page range -- see "Known gaps" below for the one such
warning currently present in this corpus.

## `manifest.json` schema

```json
{
  "isbn": "9783961102546",
  "title": "...",
  "doi": "10.5281/zenodo...",
  "domain": "linguistics",
  "language": "en",
  "publisher": "Language Science Press",
  "download_url": "https://.../book.pdf",
  "license": "https://creativecommons.org/licenses/by/4.0",
  "license_source": "crossref"
}
```

`license` is the book's OA license URL, filled in by
`fetch_crossref_gt_corpus.py` from Crossref's own `license` metadata first;
if Crossref has none registered, it falls back to Unpaywall's
`best_oa_location.license` (a different, independent data source -- see
"Crossref vs. Unpaywall" below) before giving up. `license_source` records
which one actually supplied it: `"crossref"`, `"unpaywall"`, or `null` for
the rare book neither source has one for (see "Known gaps").

## `<isbn>.crossref.json` schema

```json
{
  "isbn": "9783961102546",
  "fetched_at": "2026-08-08T12:00:00+00:00",
  "license": "https://creativecommons.org/licenses/by/4.0",
  "license_source": "crossref",
  "raw_items": [ /* verbatim Crossref message.items entries, type=="book-chapter" only */ ],
  "chapters": [
    {"title": "...", "authors": ["..."], "chapter_doi": "10...", "citation_pages": "19-39"}
  ]
}
```

`raw_items` is the untouched Crossref API response for whatever a later
integration pass might need beyond the normalized view (each item now also
carries Crossref's raw `license` array, since `license` is now part of the
`select` fetch). In `chapters`, `title` follows `CrossrefMetadataStrategy`'s
title+subtitle join convention (Crossref splits a chapter's real printed
heading into separate `title`/`subtitle` fields); `citation_pages` is the
raw Crossref `page` string as-is (e.g. `"19-39"`), not split or parsed.

The top-level `license` field is derived by majority vote across every
chapter's own registered license (in practice unanimous -- Crossref
license metadata is registered once per book and inherited by each
chapter), preferring each chapter's version-of-record entry
(`content-version=="vor"`, `delay-in-days==0`) over an embargoed variant
that may be registered alongside it.

### Crossref vs. Unpaywall: not the same data

Crossref's `license` field is *self-reported by the publisher* at
metadata-deposit time -- it's whatever the publisher chose to register
alongside the DOI, and plenty of publishers simply don't bother. Unpaywall
doesn't read that field at all; it independently aggregates OA status and
license information from institutional repositories, publisher landing
pages, and other registries, then reports its own best-guess
`best_oa_location.license` (a short code like `"cc-by-nc"`, not a URL).
The two systems can and do disagree about coverage: in this corpus,
Unpaywall recovers a license for 7 of the 10 books Crossref has none for
(every UCL Press and Athabasca University Press book), by finding it on
the publisher's own site even though it was never deposited with
Crossref. `fetch_crossref_gt_corpus.py` tries Crossref first and falls
back to Unpaywall only when Crossref comes back empty; `license_source`
records which one actually answered.

## Coverage

46 books, 896 chapters total, across 8 domains (5-6 books each) --
deliberately non-overlapping with the existing evaluation set's
socio-legal-studies skew:

| Domain | Books | Languages |
| --- | --- | --- |
| Anthropology/sociology | 6 | 3 German, 3 English |
| Computer science | 6 | 5 English, 1 German |
| Economics | 6 | 5 English, 1 Spanish |
| Education | 5 | 4 English, 1 German |
| Environmental science | 6 | 5 English, 1 German |
| History | 6 | 2 English, 2 German, 2 French |
| Linguistics | 5 | 4 English, 1 German |
| Medicine/public health | 6 | 3 English, 3 German |

**Languages currently covered:** English (31), German (12), French (2),
Spanish (1) -- 32.6% non-English, exceeding the ~25% target set in the
design spec. Extending to more languages (Italian, Portuguese, etc.) in
a future curation round needs no code changes: add manifest entries with
the new `language` value and update this table.

**Publishers:** transcript Verlag, Open Book Publishers, UCL Press,
Springer, OpenEdition Books, Athabasca University Press -- all confirmed
during curation to host their own OA PDFs directly (or via a stable
mirror), avoiding OAPEN's `library.oapen.org` bitstream host, which now
sits behind an Anubis JS proof-of-work bot-wall that returns HTTP 403 to
any non-browser client.

## Known gaps

- `9782753559530` (*Histoire de la haine*, OpenEdition): one of its 15
  Crossref `book-chapter` records (`10.4000/books.pur.138244`, "Financement
  de l'ouvrage" -- a funding-acknowledgment blurb, not a real chapter) has
  no `page` value. Expected Crossref registration noise, not a defect in
  this corpus; the other 14 chapters are complete.
- 7 books have no license registered on Crossref, but Unpaywall recovers
  one: 5 from UCL Press (`9781800088375`, `9781787359260`, `9781800086586`,
  `9781800082731`, `9781800085787`), 2 from Athabasca University Press
  (`9781771993326`, `9781771992862`). Publisher-side Crossref registration
  gap, not a curation error -- `license_source: "unpaywall"` on these.
- **3 transcript Verlag books likely aren't actually open access:**
  `9783839473948`, `9783839413197`, `9783837621310`. Neither Crossref nor
  Unpaywall (`is_oa: false`) has any license for them, DOAB has no record
  of them either, and their PDF text contains no license statement
  anywhere (every genuinely-OA transcript Verlag book in this corpus does
  state one). Their `download_url` filenames also stand out: every other
  transcript Verlag PDF here is named `oa<isbn>...pdf`; these three are
  named `tstw<n>_...pdf` instead -- consistent with transcript Verlag's
  free "Leseprobe" (reading sample) excerpt, which every book gets
  regardless of OA status, not the full open-access edition. Their page
  counts back this up too (13-44 pages for a multi-chapter edited volume).
  These three were very likely miscurated into this corpus as if they
  were OA when they are not. **None of the three were migrated into
  `evaluation/corpus/open-access/`**, so this hasn't caused a
  redistribution problem yet, but they should probably be removed from
  `manifest.json` (or re-sourced from a real OA edition, if one exists)
  rather than left here looking like the other 43 open-access entries.

## Downloading: host-specific quirks

Found empirically while curating and fetching this corpus -- all handled
by `fetch_crossref_gt_corpus.py` already, documented here in case a
future manifest addition needs the same treatment:

- **Open Book Publishers** (`books.openbookpublishers.com`) sits behind
  an AWS WAF that serves an HTTP 202 "challenge" page to requests without
  a browser-like `User-Agent`. The fetch script sends one, but only to
  this host.
- **Springer** (`link.springer.com`) runs a JS "Client Challenge"
  specifically against requests whose `User-Agent` identifies a known
  HTTP library (`python-httpx/...`, `python-requests/...`, ...) --
  serves an HTML challenge page instead of the PDF. A *browser* UA
  triggers the same challenge there (a UA that claims to be a real
  browser but can't run JS is, if anything, more suspicious to that
  check). A `curl`-style UA clears it, and works cleanly against every
  other host in this manifest too -- so the fetch script's default
  download UA is `curl/8.4.0`, with the browser UA reserved for OBP only.
- **OpenEdition Books** (`books.openedition.org`) runs an Anubis-style
  wall too, but only in front of "Freemium"-restricted titles (HTTP 401
  when denied). Genuinely open-access OpenEdition titles serve the PDF
  directly with a plain request.
