# Building ground truth for a new evaluation book

Read this when the chapter-segmentation heuristics (`src/chapter_segmentation/segmentation.py`)
score a real book with low confidence during live/empirical testing against a
real Zotero library, and you want to add that book to the evaluation set so
the regression is tracked instead of silently re-discovered later.

## Document organization in this directory

Four documents, four different lifetimes -- know which one to write to:

- **`README.md`** — permanent reference: what the evaluation set is, its
  schema, how to fetch/add books, and how to run each evaluation
  (`test_segmentation_accuracy.py`, the per-strategy report, the LLM
  cache refresh script, the strategy-pipeline script). Changes rarely,
  only when the *procedure*
  itself changes (a new evaluation script, a new page-loading mechanism,
  a new manifest field). For the exact CLI flags/defaults of any script
  under `evaluation/scripts/`, don't re-derive them from source -- see
  `evaluation/scripts/README.md`, a `--help`-dump reference kept alongside
  the scripts themselves.
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
  specific measured snapshot that will itself go stale). **Always document
  the latest results and data in `RESULTS.md`; move outdated/superseded
  text to `EXPERIMENTS.md`, and leave only a summary/mention linked to
  there.** Concretely: when a new run's numbers supersede an existing
  `RESULTS.md` write-up (a whole "Follow-up: ..." subsection, a superseded
  snapshot table, an experiment that a later one has fully moved past --
  not every single backward-looking sentence; narrative reasoning that
  explains *why* the current numbers look the way they do can stay inline
  in the current write-up), move that old write-up's full text to
  `EXPERIMENTS.md` verbatim (don't delete it, don't shorten it there) and
  replace it in `RESULTS.md` with a short summary (what was tried, the
  headline result, why it was superseded) plus a markdown link into the
  matching `EXPERIMENTS.md` heading. The same "would this sentence still
  be true after the next code change, even if no evaluation book changed"
  test that decides README vs. `RESULTS.md` extends one step further here:
  a `RESULTS.md` write-up graduates to `EXPERIMENTS.md` the moment a newer
  run's numbers make it no longer *the latest data* for its topic -- at
  that point it's no longer describing the current measured state, only
  the history behind it, which is exactly what `EXPERIMENTS.md` is for.
- **`EXPERIMENTS.md`** — the permanent, unabridged archive of every
  superseded `RESULTS.md` write-up: full prose, tables, and root-cause
  detail preserved exactly, organized under the same section headings as
  `RESULTS.md` so a `RESULTS.md` summary's link lands on the matching
  heading here. Never trimmed or rewritten away once something lands here
  -- only appended to, as more of `RESULTS.md` gets superseded over time.
  Reading `RESULTS.md` and `EXPERIMENTS.md` together should never lose
  information that was ever recorded in `RESULTS.md`.
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
  non-OA books, the tool's `--verify` step checks that the redacted text
  still makes `analyze_attachment` find the exact same chapter boundaries
  as the real text, so a clean run (no `WARNING`s) is the confirmation
  that a change didn't need any redaction-pipeline updates. (OA books skip
  `--verify` entirely since there's no redacted/real divergence possible.)
  A prior redacted revision of this corpus had 13 open-access books
  permanently fail `--verify` -- root-caused to two redaction-pipeline
  gaps (TOC-page selection being sensitive to word substitution when a
  book has a competing index/bibliography page with the same line shape,
  and the Faker word pool having no long enough real word for
  PDF-extraction-glued tokens, shrinking a page below the trailing-blank-
  page threshold) -- which is exactly why OA books are no longer redacted
  at all: real text has no such failure mode.

  **A book whose redacted text still doesn't match after self-correction
  gets cached anyway, flagged `"verified": false`, with a `WARNING` printed
  instead of being silently dropped** (changed 2026-08-13 -- see
  `redact_book_until_stable`'s docstring in `evaluation/redaction/redact.py`
  for the self-correction loop itself). The tool's own retry loop already
  widens the forced-verbatim page set automatically when it finds drift; it
  only gives up when a book's drift keeps recurring across attempts without
  shrinking, or the pages involved are so widespread that forcing them all
  verbatim would leak a large fraction of the book's actual prose into a
  file meant to be safe to redistribute. Don't chase the retry loop further
  by hand for such a book (raising `max_attempts` again, or writing a
  smarter drift-page heuristic) before checking whether the fix is
  actually cheap: **`evaluation/corpus/<corpus>/redaction_overrides.json`**
  (create if missing) is a committed, hand-maintained map of manifest key
  → extra page indices to always force fully verbatim, for exactly this
  case --

  ```json
  {"9781841136400": [453, 454]}
  ```

  Diagnose the culprit page(s) by hand rather than guessing: bisect the
  mismatched range reported in the `WARNING`'s boundary diff by forcing
  candidate pages and re-running `analyze_attachment` on both the real and
  redacted text until the exact boundaries match again (see the "found
  manually for `9781841136400`" case in `redact_book_until_stable`'s
  docstring for a worked example -- there, the culprit turned out to be a
  page that was neither the start nor end of any mismatched range, so nothing
  in the automatic loop ever proposed forcing it). If the culprit can't be
  pinned to a small handful of pages -- the drift is scattered across a
  large fraction of the book, or the redacted text loses *every* detected
  chapter outright -- leave the book uncached rather than force enough of
  it verbatim to "fix" the check; a `"verified": false` entry (or no entry
  at all, if you'd rather not cache something this noisy) is an accepted
  outlier, not a build blocker.

  **`--skip-redaction` is a separate, faster escape valve for active
  development** (added 2026-08-13, when a batch of newly-promoted books
  needed *something* cached quickly so `refresh_llm_cache.py` had data to
  read, without waiting on redaction to converge for the stubborn ones):
  it caches a non-OA book's real text verbatim -- no redaction attempted
  at all -- and marks the entry `"needs_redaction": true`. That file is
  real copyrighted prose sitting in what's normally the git-tracked,
  safe-to-publish `public-cache/` directory, so it must never be
  committed -- add its exact path to a `.gitignore` **inside that corpus's
  own directory** (`evaluation/corpus/<corpus>/.gitignore`, create it if
  missing -- corpus-scoped, not the shared `evaluation/.gitignore`, since
  this is specific to that corpus's own books) whenever you use this flag.
  Treat it as temporary: re-run the script on that book without the flag
  once you're ready to redact it for real, which overwrites the file with
  a properly redacted (or `"verified": false`, per the paragraph above)
  entry -- at which point remove its gitignore line (delete the file
  entirely once it's empty; a batch of four books that used this flag
  during the 2026-08-13 migration all ended up redacting properly once
  the boundary-check and heading-window fixes above landed, so the
  gitignore entries came back out and the file was deleted again).

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
  `copyrighted-scans/`. **A scanned PDF must already have a real, usable
  embedded text layer before it goes in** -- check with `pages_need_ocr`
  (`extract_page_texts_for_analysis(pdf_bytes)` then `pages_need_ocr(pages)`,
  both in `src/chapter_segmentation/segmentation.py`); if it returns `True`,
  OCR the PDF itself first (e.g. `ocrmypdf --force-ocr -l <lang> in.pdf
  out.pdf`) and add the OCR'ed version, not the raw scan. **But first check
  whether the scan itself is bad, not just its text layer** -- a black
  scanner-bed background, the scanning operator's hand visible around page
  edges, heavy skew, or wildly inconsistent page sizes/aspect ratios from
  page to page are all pixel-level defects that plain `ocrmypdf --force-ocr`
  does not fix (it re-OCRs whatever image is there, artifacts included --
  and can even misread the artifacts themselves as spurious glyphs). Run
  `evaluation/scripts/clean_scanned_pdf.py` instead in that case -- see
  `README.md`'s "Cleaning a badly-scanned PDF" -- which handles the
  re-OCR step itself via `--ocr-lang`, so it replaces the plain `ocrmypdf`
  invocation rather than running alongside it. `.ocr-cache/`
  (`evaluation/scripts/ocr_evaluation_pdfs.py`) only caches extracted text
  for the accuracy harness -- it does not touch the PDF the layout-based
  TOC/chapter-first-page classifier pilot reads directly (`pdfalto`, via
  `evaluation/scripts/pdfalto_runner.py`; the built binary lives at
  `../pdfalto/pdfalto`, a sibling checkout of
  [kermitt2/pdfalto](https://github.com/kermitt2/pdfalto) next to this repo
  -- pass it via `--pdfalto-bin ../pdfalto/pdfalto` or `PDFALTO_BIN`, since
  it isn't on `PATH` or installable via brew), so a text-layer-less PDF silently
  starves that pilot of signal no matter what the text-based harness sees.
- **No ground truth built yet** (you only have the PDF and basic metadata
  so far) → `pending/`. Once its `.expected.json` exists, promote it into
  `open-access/` or `copyrighted-scans/` with
  `uv run python evaluation/scripts/promote_pending_book.py <isbn> --corpus <open-access|copyrighted-scans>`
  (add `--dry-run` to preview first) -- it re-runs the Step 4 bounds/overlap
  check as a gate and, for `open-access/`, resolves `license`/`license_source`
  via Crossref/Unpaywall automatically. Only entries already in that
  corpus's committed `manifest.json` are supported (not
  `manifest.local.json` -- promote those by hand).

**If the book is being added to grow the layout-based TOC/chapter-first-page
classifier's training pool specifically** (as opposed to the text-heuristic
accuracy harness), prefer scans, books with unnumbered first chapters, and
books with weak title/body font contrast over another generic
well-produced open-access book. A learning-curve check (`RESULTS.md`,
"Follow-up: relaxing the per-book bar, and a model-architecture swap")
found `full_recall_fraction` flat across training-pool sizes 10-35 books --
the classifier is saturated on the kind of book already well-represented in
the corpus, so another book like those adds little signal; the
underrepresented templates it still struggles with are where new ground
truth actually moves the numbers.

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

**`manifest.local.json` is strictly for a book you are *not* committing to
the repo at all** — one you're only exercising in your own local test runs,
with no `public-cache/` entry and no intention of adding one right now. It
is not a parking spot for "no DOI" books in general, and it is not where a
`pending/`-promoted book's ground truth lands by default. The absence of a
DOI only decides whether the manifest entry can carry a real `doi` field —
it says nothing about whether the entry should be committed. If you're
building (or, per the rule above, already have) a `public-cache/` entry for
the book — which is exactly what happens whenever you run
`evaluation/scripts/generate_public_evaluation_cache.py` for it, including
as part of a `pending/`-promotion batch — the manifest entry belongs in the
committed `manifest.json`, DOI or not, in the same commit as that cache
file. Promoting a `pending/` book by hand (the `manifest.local.json` case
Step 0a's pending bullet mentions) does not mean the *destination* entry
stays local too — check this rule again once you've generated its cache
data, and move it to `manifest.json` if you have.

**Check Crossref by ISBN even when Zotero shows no DOI.** A personal-library
item's Zotero catalog entry frequently has no DOI field filled in even
though the book itself has a real Crossref book-level record — a quick
`curl -s "https://api.crossref.org/works?filter=isbn:<isbn>&rows=1"` (or
`.../works/<doi>` if you already spotted one printed on the copyright page)
takes a few seconds and, when it hits, moves the book straight to the
committed `manifest.json` instead of the local one. Found a real DOI this
way for 6 of 10 personal-library books in one batch (2026-08-12) whose
Zotero entries had none.

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

**Before transcribing by hand, check whether a DNB-digitized TOC scan
already exists** for this book: look in
`evaluation/corpus/dnb-toc-only/manifest.json` (see
`evaluation/scripts/fetch_dnb_toc_corpus.py` --
`docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`)
for an entry with this book's ISBN, or query live:
`curl -s "https://lobid.org/resources/search?q=isbn:<isbn>&format=json"`
and check for a populated `tableOfContents` field. When present, it's a
ready-made, already-OCR'd TOC scan to transcribe from instead of locating
and reading the TOC pages inside the raw book PDF from scratch -- you
still verify every entry by hand against the actual book exactly as Step
3 below describes; this only saves the step of finding the TOC pages
visually.

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

**Viewing pages visually (Claude Code's `Read` tool on the PDF itself)** is
the fastest way to actually *look* at a candidate page rather than just its
extracted text — text extraction can be misleadingly clean-looking while
still describing the wrong page. Two gotchas found doing this for real
(2026-08-12 ground-truth batch):

- Its `pages` parameter takes **1-based viewer page numbers**, not 0-based
  physical indices — `pages: "N-M"` renders viewer pages N through M, i.e.
  0-based `pdf_start_index`/`pdf_end_index` values `N-1` through `M-1`. Mixing
  up the two conventions mid-session silently shifts every subsequent
  spot-check by one page; recompute the viewer-page number explicitly
  (`index + 1`) every time rather than trusting a running mental offset.
- It only honors the **first** comma-separated range — `pages: "13-15,69-71"`
  silently renders only pages 13-15 and drops the second range entirely, with
  no error. Make one call per range instead of combining them.

A fast way to spot-check many pages at once via raw text (no rendering) —
dump first/last lines of a page range and eyeball them against expectation:

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
check before committing (or before considering a local-only entry "done") --
the same check also runs automatically for every corpus's ground truth via
`tests/test_ground_truth_integrity.py` (part of the default `uv run pytest`)
and as a hard gate inside `evaluation/scripts/promote_pending_book.py`, but
running it by hand here catches a mistake before it's even written to disk:

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

## Step 5: TOC ground truth

`.expected.json` also carries an optional `"toc"` field, sibling to
`"chapters"`, used by the layout-based TOC-classifier pilot (see
`docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`):

```json
{"toc_start_index": 7, "toc_end_index": 8}
```

Same 0-based-physical-page convention as `pdf_start_index`/`pdf_end_index`.
Three states, not interchangeable:

- **Key absent**: not yet retrofitted for this field.
- **`"toc": null`**: confirmed -- this book has no locatable printed TOC page.
- **`"toc": {"toc_start_index": ..., "toc_end_index": ...}`**: TOC located
  at this contiguous physical-page range.

For a book you're adding by hand, run
`evaluation/scripts/add_toc_ground_truth.py` after finishing Step 4 -- it
reuses the same structural TOC-page detection (`find_toc_pages`) the
chapter-locating step already excludes TOC pages with, so it costs nothing
extra to run. It writes automatically when the detected TOC pages form one
contiguous block; otherwise it leaves the book alone and reports it as
needing manual review (open the PDF, find the real range, write the field
by hand). Spot-check any auto-written range before trusting it, same
discipline as the chapter-boundary draft in Step 2/3 -- this script also
finds the best structural match, not necessarily the correct one.

The script defaults to `open-access`/`copyrighted-scans`, but takes an
explicit `--corpus` flag (one or more names, e.g. `--corpus pending` or
`--corpus pending copyrighted-scans`) for any other corpus, including
`pending/`.

**"NEEDS REVIEW" is the common case, not the exception** -- in a batch of 10
books hand-processed 2026-08-12, all 10 came back "NEEDS REVIEW" with a long
list of non-contiguous "TOC-like" pages scattered across the whole book, not
just the front matter: per-chapter mini-outlines (see the mini-TOC failure
mode below) and citation-/footnote-dense pages elsewhere routinely match the
same "3+ title...number lines" structural pattern the real front-matter TOC
does. A useful (but not fully reliable) triage step before falling back to
fully manual page-turning: filter the detected page set to indices below
roughly 25 (`{p for p in toc_pages if p < 25}`) and recompute the contiguous
range on that subset -- this isolated the real TOC correctly for most of
that batch, but not all of it: it can still include a front-matter false
positive (e.g. a copyright/ISBN page whose imprint text coincidentally
matches the pattern) or miss the real TOC range's own boundary pages
entirely, and on a book whose two-column TOC layout gets scrambled by
default-mode pypdf extraction (see `README.md`'s "Diverse real-library
evaluation set"), the filtered set can come back completely empty even
though the book has a perfectly real, visually obvious TOC. There is no
substitute for opening the PDF (see the visual-viewing note in Step 3 above)
and confirming the exact page range by eye every time.

## Arbitrating below-gate dnb-toc-only books

`evaluation/scripts/generate_dnb_toc_ground_truth.py`'s two-vision-model
gate discards a book outright when the two models disagree too much
(below 0.90 agreement) or one of them fails outright -- but it never
deletes either model's cached raw extraction
(`evaluation/corpus/dnb-toc-only/llm-cache/<key>.<model>.json`). Rather
than re-running the whole book from scratch or leaving it discarded,
walk through the following after a generation run leaves books below
the gate (design spec
docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md):

1. List every book still needing a decision:

   ```bash
   uv run python evaluation/scripts/arbitrate_dnb_toc.py
   ```

   This prints, per book: its title and PDF path, both models' entry
   counts and agreement rate, and every entry each side found that the
   other didn't (or, if only one model produced usable output, that
   model's full list with a note to verify it directly).

2. For each book, read the printed diff. The disagreement patterns
   found in practice so far (`evaluation/RESULTS.md` § "dnb-toc-only
   ground truth: two-vision-model gate") usually make the right call
   obvious from the text alone: one side dropping real content, one
   side including front/back matter or a part-divider that should have
   been skipped, a two-line title wrongly split into two entries, or a
   deeply nested TOC segmented at different granularities.

3. When the text alone doesn't settle it, open the book's actual TOC
   page images directly: use the `Read` tool on the PDF with a `pages`
   parameter (1-based viewer pages, same convention as Step 3 above).

4. Write the final `evaluation/corpus/dnb-toc-only/<key>.expected.json`
   yourself -- same schema as a passing book
   (`{"entries": [...], "verified": true}`, each entry via
   `evaluation.dnb_toc_matching.toc_entry_to_gt_dict`), but with
   `"verified": true` rather than `false`: unlike the bulk-tier gate's
   own output, this went through direct scrutiny (including the images,
   when needed), the same standard `_spot_check`'s docstring in
   `generate_dnb_toc_ground_truth.py` already treats as
   "independently human-verified" -- so it's also correctly excluded
   from that function's own sampling pool going forward.

5. If a book is genuinely unrecoverable (both models hallucinate, the
   scan itself is too degraded to read even directly), record that
   instead of leaving it to resurface every run:

   ```bash
   uv run python evaluation/scripts/arbitrate_dnb_toc.py reject <key> "<short reason>"
   ```

   This writes to the committed
   `evaluation/corpus/dnb-toc-only/arbitration-rejected.json` -- refuses
   (rather than silently overwriting) if `<key>` is already present, so
   re-running this step is safe.

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
  **This is not a rare edge case for some book templates** — a Springer
  "Handbuch"/Reference-series volume can carry a per-chapter mini-"Inhalt"
  box on *nearly every single chapter's* opening or second page (confirmed
  on `9783658076078.expected.json` and `9783658057022.expected.json`,
  2026-08-12), cascading the naive search wildly off-track from the second
  or third chapter onward. If you recognize this template, don't trust the
  script's output past the first couple of entries — locate every remaining
  boundary by direct visual inspection instead.
- **A chapter's title/byline block can extract *after* its own body text**,
  even though it's visually at the top of the page — a PDF content-stream
  text-object ordering quirk seen on at least two otherwise-unrelated books
  (`9783658057022.expected.json`, `9783161538315.expected.json`,
  2026-08-12). `locate_chapter_start`/`ground_truth_helper.py`'s
  title-then-author proximity check only looks at the first ~250 characters
  of extracted text, so on an affected page it skips straight past the real
  opening page and locks onto a *later* page whose running header or a
  same-title in-body mention happens to satisfy the check instead — and
  because the real opening page's title/author text is technically present
  just later in the extraction order, this can produce a **misleadingly
  high match_score (90-100)**, not a low one that would prompt a second
  look. A high score is reassuring but not proof; visually confirm the page
  regardless when a book's chapters otherwise repeat titles in running
  headers (the same precondition as the running-headers failure mode below).
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
- **Personal-library PDFs are often abridged excerpts, not complete books —
  and this is common enough to expect, not a rare surprise.** In a batch of
  14 candidates sourced from one real Zotero library (2026-08-12), 4 (~29%)
  turned out to be a photocopied subset of a much larger volume: one was a
  single chapter with no front matter at all (printed pages 398-454 of a
  much longer monograph), one was the introduction plus one later chapter
  with the entire middle of the book missing (printed pages 9-43, then a
  jump straight to 348-395, then another jump to 412-429, against a TOC
  claiming the book runs to page 432), one was four chapters of "Part I"
  with the book's own foreword referencing an unreached "zweiter Teil", and
  one was a single article whose own first page already started at printed
  page 452. **Checking only the first and last few pages is not enough** —
  the second example above looks perfectly contiguous if you check just the
  front and just the back; the gap is only visible by checking printed page
  numbers *throughout*, e.g. every 20-30 pages or at every apparent
  section/chapter boundary, and cross-checking the PDF's total physical page
  count against the TOC's own last-listed page number (a TOC claiming page
  432 in a 47-page PDF is an immediate, no-page-turning-required red flag).
  If a book fails this check, it does not belong in any corpus (not even
  `pending/`) — there is no partial ground truth worth building from an
  excerpt whose missing chapters you can't get.
- **Zotero's cataloged creator roles (`editor` vs. contributor/honoree) can
  be wrong — always verify against the book's actual title page or
  colophon before writing `authors` into the TOC JSON or the manifest
  `title`.** Found twice in one batch (2026-08-12): a book cataloged with
  two "editors" turned out to have a single actual author (a posthumous
  "collected works" volume) with one of the two names being the real
  volume editor and the other appearing nowhere as an author of any
  individual chapter; a Festschrift cataloged with two "editors" turned out
  to have one real editor, with the second name being the honoree the
  volume was written *for* (and a contributing chapter author in his own
  right), not a co-editor at all. Either mistake, left uncorrected, writes
  a wrong `authors` list that then degrades `locate_chapter_start`'s
  author-confirmation matching for every chapter in the book.
