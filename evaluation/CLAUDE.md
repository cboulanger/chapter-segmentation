# Building ground truth for a new evaluation book

Read this when the chapter-segmentation heuristics (`src/chapter_segmentation/segmentation.py`)
score a real book with low confidence during live/empirical testing against a
real Zotero library, and you want to add that book to the evaluation set so
the regression is tracked instead of silently re-discovered later.

## Document organization in this directory

Three documents, three different lifetimes -- know which one to write to:

- **`README.md`** — permanent reference: what the evaluation set is, its
  schema, how to fetch/add books, and how to run each evaluation
  (`test_segmentation_accuracy.py`, the per-strategy report, the LLM
  cache refresh script, the strategy-pipeline script). Changes rarely,
  only when the *procedure*
  itself changes (a new evaluation script, a new page-loading mechanism,
  a new manifest field).
- **`RESULTS.md`** — a snapshot: current precision/recall numbers, per-book
  `strategies_used`/recovery-route diagnostics, known remaining gaps, and
  root-cause investigation findings from the last time each evaluation was
  actually run. Expected to go stale and be regenerated or rewritten
  whenever the heuristics, strategy pipeline, extraction/OCR path, or
  evaluation set change. **When you re-run an evaluation and get new
  numbers, update `RESULTS.md`, not `README.md`** -- even if the new
  numbers reveal something surprising enough to want a narrative
  explanation (see `RESULTS.md`'s "Diverse real-library evaluation set"
  section for a worked example of a fairly involved investigation writeup
  that still belongs there, not in `README.md`, because it's tied to a
  specific measured snapshot that will itself go stale).
- **`CLAUDE.md`** (this file) — permanent workflow reference for adding a
  new evaluation book by hand (ground-truth transcription, the helper
  script, verification steps, known failure modes in that *process*, not
  in the heuristics' results).
- **`public-cache/`** (per corpus, i.e.
  `evaluation/corpus/<corpus>/public-cache/`) — a git-tracked snapshot of
  each book's page text. For `oa: true` books this is the real extracted
  text VERBATIM -- the PDF itself is already legally redistributable, so
  there's nothing to redact and no redaction-induced parity risk. Every
  other book gets a redacted snapshot instead (real navigational/
  bibliographic material verbatim, chapter prose replaced with random real
  words). Also writes `<key>.outline.json` per book -- a resolved snapshot
  of `extract_outline_candidates`' output (titles/authors/page indices
  only), letting the outline strategy be evaluated in CI without the real
  PDF. Regenerate it with `uv run python evaluation/scripts/generate_public_evaluation_cache.py`
  whenever `src/chapter_segmentation/segmentation.py` or `src/chapter_segmentation/common.py` changes in a way
  that touches text-matching logic (a new heuristic could read page text
  outside what the redaction pipeline currently preserves) -- for
  non-OA books, the tool's `--verify` step will refuse to write a
  stale/incorrect entry, so a clean run is the confirmation that a change
  didn't need any redaction-pipeline updates. (OA books skip `--verify`
  entirely since there's no redacted/real divergence possible.) A prior
  redacted revision of this corpus had 13 open-access books permanently
  fail `--verify` -- root-caused to two redaction-pipeline gaps (TOC-page
  selection being sensitive to word substitution when a book has a
  competing index/bibliography page with the same line shape, and the
  Faker word pool having no long enough real word for PDF-extraction-glued
  tokens, shrinking a page below the trailing-blank-page threshold) --
  which is exactly why OA books are no longer redacted at all: real text
  has no such failure mode.

If you're unsure which document a change belongs in, ask: would this
sentence still be true after the next code change to the heuristics, even
if no evaluation book changed? If yes, `README.md` or `CLAUDE.md`
(depending on whether it's "what/how" vs. "how to add a book"). If no --
it's describing a specific run's outcome -- `RESULTS.md`.

## Step 0a: Decide which corpus the book belongs in

Every evaluation book lives under `evaluation/corpus/<corpus>/` -- see
`docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md`.
Before anything else, pick one:

- **OA, or otherwise well-produced with a parseable embedded/printed TOC**
  → `open-access/`.
- **Everything else that you can build real ground truth for** (no DOI,
  no embedded TOC, scanned, sourced from a personal library, ...) →
  `copyrighted-scans/`.
- **No ground truth built yet** (you only have the PDF and basic metadata
  so far) → `pending/`. Move the entry into `open-access/` or
  `copyrighted-scans/` once its `.expected.json` exists.

Every path in this document below (`evaluation/<filename>`,
`evaluation/manifest.local.json`, etc.) means
`evaluation/corpus/<corpus>/<filename>` for whichever corpus you picked
here.

## Step 0b: Decide where the book's metadata goes

- **Has a DOI, OR already has a `public-cache/` entry?** → Add it to the
  committed `manifest.json`, even if the book is not open access. Set
  `"oa": false`, `"download_url": null`, and fill in `"doi"` (`null` if none).
  Either condition alone is enough: a DOI lets other developers acquire the
  PDF themselves via institutional access (see the DOI printed by
  `evaluation/scripts/fetch_evaluation_pdfs.py` when the file is missing) and use the
  ground-truth JSON — which contains no copyrighted text, just page indices
  and titles — directly; a `public-cache/` entry lets them reproduce the
  accuracy signal via `test_public_evaluation_cache_parity.py` without the
  PDF at all. A manifest entry that only exists in the gitignored
  `manifest.local.json` makes any `public-cache/` entry for that book
  effectively invisible to everyone else — `load_manifest_books()` only
  merges `manifest.local.json` in on the machine that has it, so a
  fresh-clone `available_public_books()` would never find the committed
  cache file (found empirically: 10 books' worth of already-committed
  `public-cache/` entries were undiscoverable this way until their manifest
  entries were moved over).
- **Neither** (no DOI, no `public-cache/` entry yet, e.g. a personal library
  PDF, an internal document, something you can't confirm the provenance of)
  → add it to `manifest.local.json` instead (create the file if it doesn't
  exist yet; same schema as `manifest.json`, see below). This file is
  gitignored — it never leaves your machine — but the test harness
  (`tests/test_segmentation_accuracy.py`) reads it exactly
  like the committed manifest, so it still gets exercised in your own local
  runs. Once `evaluation/scripts/generate_public_evaluation_cache.py --verify` succeeds
  for such a book, move its entry to `manifest.json` in the same commit that
  adds its `public-cache/` file, so the two never drift apart.

`manifest.local.json` schema — identical to `manifest.json`'s `"books"` list:

```json
{
  "books": [
    {
      "filename": "some-book.pdf",
      "title": "...",
      "language": "en",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": false,
      "doi": null,
      "download_url": null,
      "heuristic_expected_zero": false
    }
  ]
}
```

Place the PDF itself directly in that corpus's directory
(`evaluation/corpus/<corpus>/<filename>`) — both `.gitignore` entries
(`*.pdf`, `manifest.local.json`) mean neither the file nor its local-only
metadata are ever committed.

`"heuristic_expected_zero"` is optional (defaults to `false` when omitted).
Set it to `true` only when the book scores exactly zero recall even after
every available recovery path has been tried and confirmed not to help:
default extraction, the pypdf layout-mode extraction fallback (recovers a
real printed TOC that default-mode extraction scrambled), and — for a book
whose text layer is absent or degenerate (`pages_need_ocr`) — the
evaluation OCR cache (`uv run python evaluation/scripts/ocr_evaluation_pdfs.py`,
requires the Kreuzberg sidecar; see `README.md`'s "Running an evaluation").
`evaluation/harness.py`'s `analysis_pages_for` is what the test
harness actually uses to load pages, so it's what a book must fail through
before the flag is justified. Do not set it just because
`find_toc_candidates` returns empty on raw default-mode extraction — that
used to be treated as "no signal at all," but turned out in practice to
often just mean the TOC needed layout-mode extraction or OCR to become
visible (see "Diverse real-library evaluation set" in `README.md` for the
worked example: 7 of 10 books once flagged this way turned out to have real,
recoverable signal once the right extraction path was tried; only 3 remain
genuinely at zero, each with a specific, traced root cause documented
there — an OCR-quality issue, a back-matter false-positive cluster winning
over a degraded real TOC, and a book with no TOC-line-density anywhere in
the OCR'd text). `test_segmentation_accuracy.py` otherwise
hard-fails any book that scores exactly zero recall, on the assumption
that's always a code regression — true for the curated, DOI-backed set, and
now also true for most of the local, no-DOI set. Leave the flag
`false`/omitted for every other book, including ones the heuristic merely
scores *low* on (only an unconditional zero is special-cased), and re-check
it (don't just trust an old value) whenever the extraction/OCR pipeline
changes or a new evaluation book is added.

## Step 1: Transcribe the table of contents

Open the PDF and write out its chapters as a small JSON file (anywhere,
e.g. `/tmp/<name>_toc.json`):

```json
[
  {"title": "Introduction", "authors": ["Jane Author"]},
  {"title": "Some Chapter Title", "authors": ["First Author", "Second Author"]}
]
```

Include front/back-matter sections that sit *between* two real chapters you
care about (e.g. "Acknowledgements", "Bibliography", a "Part II" divider) as
`{"title": "...", "authors": [], "skip": true}` entries — they're needed to
correctly bound their neighbors' page ranges even though they won't appear in
the final ground truth. Leaving one out is the single most common way to get
a neighboring chapter's `pdf_end_index` wrong (see "Known failure modes"
below).

## Step 2: Run the helper script to get a draft

```bash
uv run python evaluation/scripts/ground_truth_helper.py \
  --pdf evaluation/<name>.pdf \
  --toc /tmp/<name>_toc.json \
  --output /tmp/<name>_draft.json
```

This locates each entry's true chapter-opening page by **content search**
(never by assuming `pdf_index == printed_page_number` — see design spec
section 2/5 for why that assumption is unsafe) and tries to read the printed
page number shown on that page separately, for `citation_pages`.

## Step 3: Verify every entry by hand — do not trust the draft

The script found the *best-scoring* match, not necessarily the *correct*
one. For each chapter in the draft:

1. Open the PDF at `pdf_start_index` (0-based physical page, so PDF viewer
   page N+1) and confirm it's really that chapter's title/byline page, not a
   continuation page or a false match.
2. Do the same for `pdf_end_index` — confirm it's the chapter's actual last
   page of content, not a blank filler page or the start of the next
   section.
3. If `match_score` is below ~90, or `citation_start`/`citation_end` came
   back `null`, double-check by hand rather than assuming the script got it
   right (or leave `citation_pages` as `null` if the page genuinely has no
   visible printed number — never guess one).
4. Compute `citation_pages` as `"<citation_start>-<citation_end>"` (or
   `null` if either half is unavailable).

A fast way to spot-check many pages at once — dump first/last lines of a
page range and eyeball them against expectation:

```bash
uv run python -c "
from pypdf import PdfReader
r = PdfReader('evaluation/<name>.pdf')
for i in [<index>, <index>+1]:
    t = r.pages[i].extract_text() or ''
    lines = [l for l in t.splitlines() if l.strip()]
    print(i, 'FIRST:', repr(lines[0][:60]) if lines else None, ' LAST:', repr(lines[-1][:60]) if lines else None)
"
```

## Step 4: Write the final `.expected.json` and sanity-check it

Save as `evaluation/<name>.expected.json`
(schema: `{"chapters": [{"title", "authors", "pdf_start_index",
"pdf_end_index", "citation_pages"}, ...]}` — see any existing `.expected.json`
in this directory for a worked example). Then run the bounds/overlap sanity
check before committing (or before considering a local-only entry "done"):

```bash
uv run python -c "
import json
from pypdf import PdfReader
book = '<name>'
total = len(PdfReader(f'evaluation/{book}.pdf').pages)
chapters = json.load(open(f'evaluation/{book}.expected.json'))['chapters']
ranges = sorted((c['pdf_start_index'], c['pdf_end_index']) for c in chapters)
for s, e in ranges:
    assert s <= e, f'start>end: {(s,e)}'
    assert e < total, f'end>=total_pages({total}): {(s,e)}'
for (s1,e1),(s2,e2) in zip(ranges, ranges[1:]):
    assert s2 > e1, f'overlap: {(s1,e1)} vs {(s2,e2)}'
print(f'{book}: {len(chapters)} chapters, {total} pages -> OK')
"
```

## Known failure modes (found the hard way while building this evaluation set)

- **PDF-index ≠ printed page number, and the offset is often not constant.**
  Blank filler pages (forcing a chapter to start on an odd page) and "Part N"
  divider pages can consume a printed page number without occupying their
  own distinct content page, or vice versa. Never derive `pdf_start_index`
  from `citation_start` or the TOC's printed page number directly — always
  locate by content search.
- **A book's own table-of-contents page will fuzzy-match its own entries.**
  `ground_truth_helper.py` guards against this by structurally detecting TOC
  pages (3+ "title ... number" lines close together) and excluding them from
  the search — but if a chapter's own opening page happens to contain a
  mini internal outline (sub-sections with their own page numbers), it can
  be mistaken for a TOC page and wrongly excluded too. If a chapter's
  `pdf_start_index` looks suspiciously late, check whether this happened.
- **Running headers can repeat the full chapter title on every page**, not
  just the opening one. The script requires an author's last name to appear
  near the top of the page too, to disambiguate the true opening page from a
  mere continuation page — but this only works if you supplied `authors` in
  the TOC JSON; an empty/missing `authors` list falls back to title-only
  matching, which is much more error-prone for chapters with generic titles.
- **A bare "d", "i", "v", "x", "l", "c", or "m" at the end of an ordinary
  word** (e.g. "Afterword", "Index") can look like a roman-numeral page
  number to a careless regex. `ground_truth_helper.py`'s
  `extract_printed_number` guards against this with a boundary check, but if
  you write your own extraction logic, remember this trap.
- **Very short/generic entries (e.g. "Index", "Postface", "Bibliography")
  are bad fuzzy-match search targets** — short strings tend to
  partial-match almost any page somewhat plausibly. Prefer a more distinctive
  substring of the section's actual heading text as the search title, or
  just determine that boundary by direct inspection instead of trusting the
  script for it.
