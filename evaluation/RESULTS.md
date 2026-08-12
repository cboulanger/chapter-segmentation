# Current evaluation results — book segmentation

**This is a snapshot, not permanent documentation.** It reports the numbers,
findings, and known gaps from the last time each evaluation was actually run
against the real PDFs. It is expected to go stale and be regenerated (or
rewritten) whenever the heuristics, the strategy pipeline, the extraction/OCR
path, or the evaluation set itself change — do not treat any number here as a
guarantee. For what the evaluation set is, how it's organized, and how to run
each evaluation, see `README.md` in this directory instead; that document
changes rarely and this one changes often.

> **Always-current numbers:** https://cboulanger.github.io/chapter-segmentation/ (auto-published from `evaluation/generate_report.py`, no hand-written analysis). This file adds mechanism/root-cause commentary the published page deliberately omits, and is only updated by hand.
>
> **Layout note:** every evaluation book now lives under
> `evaluation/corpus/<name>/` (`open-access`, `copyrighted-scans`, `pending`) --
> see `docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md`.
> The two result sections below ("Pure-heuristic results" and "Diverse
> real-library evaluation set") correspond to the `open-access` and
> `copyrighted-scans` corpora respectively. Both corpora have since grown
> substantially (open-access gained 31 books reconciled from
> `evaluation/crossref_gt/`; copyrighted-scans gained 2 hand-built Festschrift
> volumes) -- the tables below predate that growth and are due for a full
> re-run and rewrite; see `README.md`'s "Corpora" section for current counts.

## Pure-heuristic results

From `uv run pytest tests/test_segmentation_accuracy.py -q -s`
(`chapter_segmentation.analyze_attachment`), one row per committed evaluation
book:

| Book (title / filename) | Language | Type | Precision | Recall | Found / Expected |
| --- | --- | --- | --- | --- | --- |
| Transformations of European Welfare States and Social Rights (`9783031466373.pdf`) | en | native | 1.00 | 1.00 | 11/11 found, 11/11 expected |
| Violence, Imagination, and Resistance (`9781771993661.pdf`) | en | native | 1.00 | 1.00 | 10/10 found, 10/10 expected |
| 20 ans de transparence à Genève (`9783907297339.pdf`) | fr | native | 1.00 | 1.00 | 11/11 found, 11/11 expected |
| Accueillir des publics migrants et immigrés (`9782375460122.pdf`) | fr | native | 0.78 | 0.82 | 14/18 found, 14/17 expected |
| Recht in der Krise — APARIUZ XXIII (`9783907297285.pdf`) | de | native | 0.69 | 0.69 | 9/13 found, 9/13 expected |
| Recht umkämpft (`9783847432364.pdf`) | de | native | 0.95 | 0.95 | 20/21 found, 20/21 expected |
| Jahrbuch für Rechtssoziologie und Rechtstheorie IV (`9783322969828.pdf`) | de | scan | 0.96 | 0.92 | 22/23 found, 22/24 expected |

Aggregate (micro): **precision 0.91, recall 0.91** across 107 expected
chapters. These 7 books' numbers are unaffected by the layout-mode
extraction fallback and evaluation OCR cache described in "Diverse
real-library evaluation set" below -- their default-mode pypdf extraction
already finds a usable TOC, so neither addition ever fires for them.

The heuristic pipeline's main mechanisms, in the order they run
(see `src/chapter_segmentation/segmentation.py` for the full details on each):

- `find_toc_candidates` counts only lines that survive the content filters
  (URL/DOI, implausible page numbers) toward the "does this page look like
  a listing" density test, so an imprint/metadata page can't shadow the
  real TOC; it accepts roman-numeral page fields for front-matter entries
  ("Foreword vii"), merges wrapped multi-line titles into alternative
  "variant" readings (the page number sits on the last physical line only),
  adopts the preceding line's text when the page number sits on a bare
  dot-leader line of its own, extends the TOC cluster onto a following page
  holding just the listing's last couple of entries, and recognizes the
  author-line convention ("MOTS ET CHIFFRES ... / par Mustapha Harzoune
  .... 16") where only `par`/`by`-marked entries are chapter-level.
- `locate_chapter_start` strips repeated running headers (detected
  digit-insensitively across the book) before scoring a page's head, so a
  book that stamps every page with a long header doesn't hide its titles.
- `_locate_toc_entries` picks, per entry, whichever variant reading locates
  most trustworthily (ranked by `min(score, margin)` — certainty, not raw
  score), excludes "secondary listing" pages (part-divider pages and
  cover/blurb pages that quote several chapter titles) and everything on or
  before the TOC itself (except for roman-paginated front-matter entries),
  and finally resolves per-entry ambiguity with TOC-order constraints: an
  entry's candidates are pruned to the interval between its already-located
  neighbors, which cleanly separates Introduction/Conclusion pairs sharing
  the book's own title as a suffix.
- `_chapters_from_located` keeps part dividers and standard back-matter
  sections (Index, Contributors, Sommaire, ...) as located *boundaries*
  without emitting them as chapters, and trims chapter ends past trailing
  blank pages and known non-content (TOC/listing) pages.

Known remaining misses, all understood and accepted for now:

- `9782375460122.pdf`: three chapters end on image-only pages that pypdf
  extracts as empty text — the text heuristic cannot distinguish them from
  blank divider pages, so those ends are one page short.
- `9783907297285.pdf`: this book's hand-verified ground truth attaches each
  trailing part-divider page ("Teil 2: ...") to the preceding chapter,
  while `9783322969828.pdf`'s ground truth excludes divider pages from
  chapters — the two conventions are mutually exclusive for a single
  heuristic, and the current behavior matches the latter (more common)
  book. Its first chapter also opens with a per-chapter half-title page two
  pages before the body, which the locate step returns instead of the
  ground truth's body page.
- `9783322969828.pdf` (scan): one chapter's TOC line is OCR-garbled beyond
  what the fuzzy matcher recovers.

`chapter_upload.py`'s `confidence_threshold` default was re-calibrated
against this snapshot (now `0.90`): with ~91% of all proposed chapters
already exactly correct, the sweep shows the threshold no longer buys
precision (0.91 → 0.93 across the whole range) while anything above ~0.94
sharply cuts how many correct chapters survive — the remaining errors are
end-boundary quirks that the start-match confidence cannot see. **Re-run
the calibration sweep (or re-derive it against `analyze_attachment`'s
output) any time `find_toc_candidates`, `locate_chapter_start`, or
`match_confidence` change, or the evaluation set grows** — these numbers
are a snapshot tied to the current heuristics, not a permanent constant.

### Heuristic fix: copyright/imprint page shadowing the real TOC

Found while trying the pipeline against a new real-world book outside the
evaluation set at the time (`9783428042241.pdf`, DOI
10.3790/978-3-428-44224-9, a 1978 German Festschrift, 918 pages): its
copyright page ("© 1978 Duncker & Humblot, Berlin 41", "Gedruckt 1978 bei
...", "ISBN 3 428 04224 1") has exactly three lines that structurally
match the TOC-line pattern (title-like text ending in a trailing number)
-- enough to form the book's first qualifying "TOC cluster" and
permanently shadow the real table of contents three pages later, since
none of the three is a URL/DOI (the only content filter
`find_toc_candidates` had for this category before). `find_toc_candidates`
went from 3 junk entries / near-zero detected chapters to 41 real TOC
entries / 40 correctly detected chapters once a matching
`_looks_like_imprint_line` filter (`© <year>`, `ISBN`, `Gedruckt`,
`Printed in`) was added alongside the existing URL/DOI filter
(`src/chapter_segmentation/segmentation.py`).

**Zero regression on the rest of the evaluation set**: re-running
`test_segmentation_accuracy.py` after the fix reproduces every
number in the tables above and below exactly -- none of the other books'
TOC pages happen to have copyright-page text shaped like this.
`9783428042241.pdf` has no hand-verified ground truth yet (see
`CLAUDE.md`'s "Building ground truth"), so it isn't in any table here --
its 41-entries/40-chapters result is a real, checked improvement, but
unscored, not a precision/recall number.

## Strategy-pipeline results

From `uv run python evaluation/scripts/evaluate_chapter_segmentation_strategies.py`
(`analyze_attachment_with_strategies`):

| Book (filename) | Precision | Recall | Found / Expected | Strategies used |
| --- | --- | --- | --- | --- |
| `9783031466373.pdf` | 0.83 | 0.91 | 10/12 found, 10/11 expected | crossref, outline, outline+crossref |
| `9781771993661.pdf` | 1.00 | 1.00 | 10/10 found, 10/10 expected | (falls back to heuristic) |
| `9783907297339.pdf` | 0.82 | 0.82 | 9/11 found, 9/11 expected | outline |
| `9782375460122.pdf` | 0.78 | 0.82 | 14/18 found, 14/17 expected | (falls back to heuristic) |
| `9783907297285.pdf` | 0.64 | 0.69 | 9/14 found, 9/13 expected | outline |
| `9783847432364.pdf` | 0.95 | 0.95 | 20/21 found, 20/21 expected | crossref |
| `9783322969828.pdf` | 0.96 | 0.96 | 23/24 found, 23/24 expected | crossref |

Aggregate (micro): **precision 0.86, recall 0.89** across 107 expected
chapters -- still below the pure-heuristic baseline's 0.91/0.91 above, so on
this evaluation set the strategies remain a net *regression* when they fire,
not an improvement, though a much smaller one than the previous snapshots.
`analyze_attachment_with_strategies` only falls back to the pure-heuristic
pipeline when a book's merged candidate list is completely empty (two books
above), never when it's merely wrong -- so a confidently-incomplete or
imprecise outline/Crossref result is trusted over what the heuristic
pipeline would have found instead. Root causes found and fixed so far (see
`extract_outline_candidates` in
`src/chapter_segmentation/evidence/outline_strategy.py`,
`_is_non_chapter_structural_title` in `src/chapter_segmentation/common.py`,
`analyze_attachment_with_strategies`'s `exclude_indices` computation and
`merge_metadata_sources`'s single-source sort in
`src/chapter_segmentation/segmentation.py`/`evidence/fusion.py`, and
`_parse_crossref_item`'s subtitle handling in
`src/chapter_segmentation/evidence/crossref_strategy.py` for the code,
`tests/evidence/test_outline_strategy.py` /
`tests/test_segmentation_strategies.py` /
`tests/evidence/test_fusion.py` / `tests/evidence/test_crossref_strategy.py` for
regression coverage):

- **Real chapters nested one level under unlabeled "Part I/II/III" outline
  nodes.** The top-level-only read used to silently accept whatever
  front/back matter survived around the part dividers as "the chapter
  list" (a sparse-but-non-empty result, so the empty-merge fallback never
  triggered) -- now rejected outright the moment a part-divider entry is
  found to have nested children, deferring correctly to the heuristic
  fallback. Recovered both books that hit this to their pre-regression
  1.00/1.00 and 0.78/0.82 scores.
- **PDF-outline-specific bookmark vocabulary** (Cover, Half Title, Title,
  Copyright, Backcover, Impressum, German "...verzeichnis" compound nouns,
  and the book's own title used as its own bookmark) was leaking through as
  false-positive chapters -- none of it appears in a printed table of
  contents, which is what the back-matter title list was originally built
  from.
- **Trailing-page trim over-reach for outline-sourced chapters.** The
  generic "front matter through the printed TOC" exclusion region (built to
  constrain where a *not-yet-located* chapter may start) was also being
  used to trim chapter *ends* -- collapsing a chapter's real content down
  to a single page whenever its true end fell inside that broad region
  (e.g. a Foreword that legitimately lives in the front matter).
- **Crossref/Zotero-catalog candidates were localized against a blind
  15%-of-total-pages front exclusion zone, not the book's actual printed
  TOC.** `analyze_attachment_with_strategies` fed metadata-sourced
  candidates needing content-search localization through
  `_toc_scan_indices(pages)` -- a fixed front-15%/back-5% page-fraction
  region meant only as a *search window* for `find_toc_candidates`, never
  as a "definitely non-chapter-content" guarantee. On books whose front
  matter is much shorter than 15% of total length (all three
  `crossref`-touched books above run 226-419 pages with a 1-2 page printed
  TOC), this blindly excluded real early chapter starts from the candidate
  pool entirely (silently dropped as unlocated) and pushed others onto a
  later page sharing the same running header. Fixed by deriving
  `exclude_indices` from `find_toc_candidates(pages)`'s own detected TOC
  page(s) instead -- the same source `analyze_attachment`/
  `analyze_attachment_with_llm_fallback` already use -- falling back to the
  blind fraction only when no real TOC page is found at all. Recovered
  `9783322969828.pdf` to near-parity with the heuristic baseline (0.95/0.75
  -> 0.96/0.96) and substantially improved the other two Crossref-touched
  books.
- **A single metadata source's candidate list is not guaranteed to be in
  book order.** `merge_metadata_sources` only sorted by
  `printed_page_number` when actually merging TWO non-empty sources
  (`_merge_two_metadata_lists`'s own final sort); with a single source --
  the common case for a Crossref-only book -- the list passed straight
  through in whatever order the Crossref API happened to return it.
  `_locate_toc_entries`'s second-pass "TOC order is book order" ambiguity
  disambiguation relies on list POSITION (not `printed_page_number` value)
  mirroring book order, so an out-of-order single source silently broke
  its `lower`/`upper` bound computation and made a genuinely locatable
  chapter (e.g. a short, ambiguous title with real page-order neighbors
  that should have pinned it down) fall through as unresolved instead.
  Fixed by sorting a single source the same way a merged pair already is.
- **Crossref splits a chapter's real printed heading into separate
  title/subtitle fields, and `_parse_crossref_item` was discarding the
  subtitle.** A truncated title (e.g. "Commons." instead of the real
  "Commons. Was wir brauchen und was uns gemeinsam ist") is a much weaker,
  more ambiguous content-search target than the full heading a PDF-outline
  bookmark supplies for the same chapter -- it can tie or even out-score
  its own true opening page against a brief in-body mention of the same
  short phrase in a nearby, unrelated chapter, and because
  `locate_chapter_start_candidates` transitively chains same-title matches
  within a small page gap (deliberately, to keep a long chapter's own
  repeating running header as one cluster -- see its docstring), a
  spurious short mention 2-3 pages before the real opening page can chain
  onto it and report the earlier, wrong page as the location. Fixed by
  requesting Crossref's `subtitle` field and appending it to `title[0]`
  when present, restoring the full heading. Recovered `9783847432364.pdf`
  from 0.80/0.76 to 0.95/0.95 (20/21 chapters now exactly correct).

Known remaining gaps, not yet fixed:

- A book-specific structural section label with no generic vocabulary
  match (e.g. "Gastbeiträge: ..." -- German for "guest contributions",
  grouping several chapters that follow it) still leaks through as a false
  chapter on `9783907297285.pdf`. The same class of gap causes "Preface" (a
  real outline bookmark, not counted as a ground-truth chapter) to leak
  through as a false positive on `9783031466373.pdf`.
- **The last located chapter in a book has no "next entry" to bound its
  end**, so `_chapters_from_located` defaults to the last PDF page and
  trims backward only past pages its blank-page/non-content checks
  recognize. A publisher's back-cover blurb or colophon page (ISBN,
  website, marketing copy -- real, non-blank text, so the blank-page check
  doesn't fire) is not detected as non-chapter content, over-extending the
  final chapter by a few pages on both `9783847432364.pdf` (350-364 found
  vs. 350-361 expected) and `9783322969828.pdf` (405-418 found vs. 405-416
  expected). Crossref's own "Back Matter" candidate (present in both
  books' raw results, see `crossref_candidates_found` in diagnostics)
  would be the right boundary if it could be located, but "Back Matter" is
  a generic schema label, not literal text printed anywhere in either
  book, so it never resolves to a page and can't serve as one; not yet
  root-caused further.
- Even where the located start page is correct, `_chapters_from_located`'s
  blank-page/trim heuristics still occasionally over- or under-shoot a
  chapter's end by a few pages for outline- and Crossref-sourced chapters
  alike -- the same class of imprecision the pure-heuristic pipeline
  already has (see "Known remaining misses" above).

**Given this net regression, `analyze_book_chapters.py`/the `/api/analyze`
endpoint should not be pointed at the strategy pipeline for production use
until the remaining gaps above are closed** -- re-run this evaluation after
any further change to the outline/Crossref/fusion logic.

## Diverse real-library evaluation set — results

See `README.md`'s "Evaluation set composition" for what these 10 books are
and why they're in the set. They originally lived in the gitignored
`manifest.local.json` (no DOI); all 10 have since moved into the committed
`manifest.json`, once each gained a `public-cache/` entry (see
`CLAUDE.md`'s "Document organization" for the broadened criterion) --
`manifest.local.json` no longer exists at all as of this snapshot.

| Filename | Precision | Recall | Found / Expected | Recovery route |
| --- | --- | --- | --- | --- |
| `9783848736829.pdf` | 1.00 | 1.00 | 23/23, 23/23 | layout-mode fallback |
| `9783492021234.pdf` | 0.27 | 0.29 | 4/15, 4/14 | layout-mode fallback |
| `9783789016202.pdf` | 0.46 | 0.50 | 6/13, 6/12 | layout-mode fallback |
| `9783899718188.pdf` | 0.27 | 0.30 | 3/11, 3/10 | layout-mode fallback |
| `9780367439712.pdf` | 0.36 | 0.42 | 5/14, 5/12 | OCR (degenerate text layer) |
| `9783789057366.pdf` | 0.00 | 0.00 | 0/2, 0/56 | OCR (degenerate text layer) -- still 0 |
| `9783465016878.pdf` | 0.40 | 0.15 | 2/5, 2/13 | OCR (no text layer) |
| `9781409403906.pdf` | 0.10 | 0.08 | 1/10, 1/12 | OCR (no text layer) |
| `9783848704316.pdf` | 0.00 | 0.00 | 0/0, 0/15 | OCR (no text layer) -- still 0 |
| `dnb-36942798X.pdf` | 0.00 | 0.00 | 0/1, 0/18 | OCR (no text layer) -- still 0 |

Aggregate (micro, these 10 books alone): **precision 0.47, recall 0.24**
(44/94 found correctly, 44/185 expected chapters). These are the same
numbers whether run through the pure heuristic (`analyze_attachment`,
`test_segmentation_accuracy.py`) or the full strategy pipeline
(`analyze_attachment_with_strategies`, `strategies_used: []` on all 10) --
none of the 10 has a Crossref record or a usable PDF outline, so the
strategy pipeline always falls straight back to the same heuristic result
on this set; the strategies genuinely add nothing here, for better or
worse.

This is a large change from an earlier, incorrect reading of this same
sample: all 10 books used to be reported as scoring 0.00/0.00 with
`toc_matches_found: 0`, taken as evidence that "none of the three
currently-implemented strategies have any signal to work with on this
sample at all." That conclusion was wrong -- it measured a text-extraction
artifact, not an absence of signal:

- **6 of the 10 books have a real, intact printed table of contents.**
  pypdf's *default* text-extraction mode destroys a two-column TOC layout
  (page numbers land on their own line, separated from titles, or glue onto
  the next title: `'7Vorwort'`, `'123III. Recht zwischen den
  Professionen'`), so `find_toc_candidates`'s `<title> ... <page number>`
  regex never matches a single physical line and reports zero candidates.
  `extract_page_texts_for_analysis` (`src/chapter_segmentation/segmentation.py`) now retries
  with pypdf's `layout` extraction mode -- which preserves column
  alignment -- whenever default-mode extraction finds no TOC, and adopts
  the layout-mode pages only if THAT finds one. This is a pure fallback:
  none of the original 7 committed books' scores changed, since their
  default-mode extraction already found a TOC and the layout attempt never
  fires for them. Four of the ten books recovered this way -- one to a
  perfect 1.00/1.00, the other three to partial (0.29-0.50) scores limited
  by ordinary heuristic imprecision (messy TOC layouts: inline author
  names, a left-column page-number convention), not a missing signal.
- **2 of the 10 books have a *degenerate* text layer**
  (`9783789057366.pdf`, `9780367439712.pdf`): every page extracts as one
  giant line with essentially no newlines, in *both* default and
  layout-mode pypdf extraction (pypdf itself warns "Rotated text
  discovered" while reading them) -- no line-oriented heuristic can work on
  that shape of text regardless of extraction mode. `pages_need_ocr`
  (`src/chapter_segmentation/segmentation.py`) now detects this case (alongside an
  absent/near-absent text layer) and routes both books through OCR instead
  -- the same per-page Kreuzberg OCR path production uses for scans
  (`src/chapter_segmentation/ocr.py`'s `ocr_pdf_pages`), cached by content hash in the gitignored
  each corpus's `evaluation/corpus/<name>/.ocr-cache/` and populated by
  `uv run python evaluation/scripts/ocr_evaluation_pdfs.py` (run once; re-runs are
  instant cache hits). OCR recovers usable line structure for one of the
  two (`9780367439712.pdf`, 0.00 -> 0.42); the other
  (`9783789057366.pdf`) still scores 0.00 -- see "still 0" below.
- **The 4 true scans** (`9783465016878.pdf`, `9781409403906.pdf`,
  `9783848704316.pdf`, `dnb-36942798X.pdf`) had no text layer at all and
  were never exercised by any evaluation script before this change (the
  pytest harness and both `evaluation/scripts/evaluate_chapter_segmentation_*.py`
  scripts fed raw pypdf text straight into analysis, with no OCR step --
  unlike production's `run()`, which already had one). They now go through
  the same OCR-cache route as the two degenerate-text books above. 2 of the
  4 recover partial signal (0.08-0.15); 2 remain at 0.00 -- see below.

**The 3 books still at 0.00 recall after OCR, with actual root causes**
(traced by hand -- inspecting the cached OCR text and `find_toc_candidates`'
raw output directly, not guessed):

- `9783789057366.pdf`: OCR *does* locate the real front-matter TOC page
  (`find_toc_candidates` finds it), but Tesseract's OCR of the dot-leader
  lines is badly garbled on this particular scan -- a line that should read
  something like `"Zueignung .......... 6"` instead OCRs as
  `"ZUEISNUNG .n...onannnnnennenenennsnnennnnmn nen nn nenn nennen rennen"`.
  The dot-leader run of characters is essentially noise, and several
  titles are corrupted too, so almost no entry survives cleanly enough for
  the content-search localization step to find its real opening page. This
  is an OCR-quality problem on a specific page layout convention (dense
  dot leaders), not a missing-TOC problem.
- `dnb-36942798X.pdf`: `find_toc_candidates` finds 6 "candidates," but they
  are OCR'd bibliography/citation-index lines from the back of the book
  (`"Law, Criminology, and Police Science ... 55"`,
  `"... American Journal of Sociology ... 70"`) -- journal citations that
  structurally resemble a TOC line ("title ... number") but aren't one.
  The book's real front-matter TOC apparently didn't survive OCR at a high
  enough line-density to win the "first qualifying cluster" contest against
  this back-matter citation list. This is the same "secondary listing wins
  over the real TOC" failure mode already documented for
  `9783322969828.pdf` in "Known remaining misses" above, here triggered by
  OCR degrading the real TOC's density rather than a structural ambiguity.
- `9783848704316.pdf`: `find_toc_candidates` returns **zero** candidates
  anywhere in the OCR'd text's front/back scan window -- no page has three
  or more lines matching the TOC-line pattern at all, even after OCR. Not
  yet root-caused further (worth checking by hand whether this book's
  printed TOC uses a structurally different convention the regex can't
  match at all, versus OCR quality issues specific to its front matter).

These three keep `"heuristic_expected_zero": true` in `manifest.json` (a
known, currently-accepted limitation, re-checked and re-justified with
real evidence rather than the earlier blanket claim); the other seven of
the ten are now `false` and are held to the same `recall > 0` regression
guard as the rest of the evaluation set.

## Per-strategy standalone results (heuristic / outline / LLM)

From `uv run python evaluation/generate_report.py --out public/` (heuristic,
outline) plus `uv run python evaluation/refresh_llm_cache.py --mode full`
populating each corpus's `evaluation/corpus/<name>/llm-cache/` (LLM) -- each strategy run independently
via `analyze_attachment`, `analyze_attachment_outline_only`,
`analyze_attachment_llm_only` against the full 17-book public-cache corpus,
with no pipeline merge/fallback logic involved (see
`docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md`).
Live numbers (always current, no hand-written commentary):
<https://cboulanger.github.io/chapter-segmentation/>;
full per-model breakdown at its `llm/index.html`. See `README.md`'s
"Per-strategy evaluation report" / "LLM strategy evaluation" for how to
reproduce.

"Start accuracy"/"End accuracy" (added by
`docs/superpowers/plans/2026-08-08-citation-pages-mapping.md`, see that
section below) score `citation_pages` -- the printed-page-number metadata
attached to each chapter -- restricted to chapters whose located PDF page
range exactly matches ground truth AND whose expected `citation_pages` is
non-null. Start requires an exact match (unmappable counts as wrong); end
tolerates being up to 3 printed pages over-inclusive (see
`evaluation/metrics.py`'s `citation_pages_metrics`).

| Strategy | Precision | Recall | F1 | Found / Expected | Total time | Start accuracy | End accuracy | Applicable books |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outline | 0.79 | 0.89 | 0.84 | 31/39 found, 31/35 expected | 0.6s | 0.16 | 0.48 | 3/17 |
| heuristic | 0.71 | 0.50 | 0.58 | 145/204 found, 145/292 expected | 14.8s | 0.99 | 0.99 | 17/17 |
| LLM (`glm-4.7`, best of 10 cached models) | 0.71 | 0.37 | 0.48 | 107/150 found, 107/292 expected | 239.7s | 0.99 | 0.99 | 17/17 |

- **Outline scores highest but applies narrowly.** Only 3 of the 17 corpus
  books carry a real embedded PDF outline/bookmark catalog
  (`9783031466373.pdf`, `9783907297285.pdf`, `9783907297339.pdf`) -- an
  outline entry is already a resolved, book-order-correct chapter
  reference, so once one exists it's nearly free signal (well under a
  second total across all 3, no content search needed beyond confirming
  the mapped page). The other 14 books have no outline to read at all
  (`evaluation/corpus/<name>/public-cache/<key>.outline.json` is `{"candidates": []}`),
  which is a property of the PDF, not a strategy failure -- those books
  render `N/A` in the report rather than being scored as "found 0".
- **Heuristic's 0.58 aggregate here is pulled down entirely by the 10
  "diverse real-library" books** documented above -- restrict to the
  original 7 well-behaved books and it's 0.91/0.91 (see "Pure-heuristic
  results"); the other 10 (OCR/degenerate-text/layout-mode recoveries)
  score 0.47/0.24 as their own aggregate (see "Diverse real-library
  evaluation set" above). This table just confirms the same heuristic run
  through the new standalone harness (`evaluation/metrics.py`) reproduces
  identical per-book numbers.
- **LLM standalone F1 improved from 0.29 to 0.49** after fixing two root
  causes in `llm_extract_toc_entries` found by inspecting this table's
  first run (see
  `docs/superpowers/specs/2026-08-07-llm-toc-extraction-fix-design.md`):
  a hardcoded `max_tokens=1024` output cap was silently truncating the
  LLM's JSON chapter listing on large-TOC books (truncated JSON fails to
  parse, and the failure was swallowed as "found nothing"), and the
  extraction prompt sent the entire blind front-15%/back-5% page fraction
  instead of the much narrower region `find_toc_candidates` can usually
  already pinpoint. Both are now fixed: retry once at `max_tokens=8192` on
  a parse failure, and narrow the scanned page range to the heuristic's
  detected TOC page(s) (+/-1 page) whenever it finds one. The clearest
  before/after evidence: the three books with 21-24 expected chapters that
  previously scored exactly 0/0 across every one of the (then 5) models
  (`9783322969828.pdf`, `9783847432364.pdf`, `9783848736829.pdf`) now
  score F1 0.71-0.86 across 9 of the current 10 cached models (see
  `llm/index.html`) -- the sole holdout, `qwen3.5-122b-a10b`, still scores
  0/0 on two of the three, consistent with it being the weakest model
  overall (F1 0.32, lowest of the ten). `apertus-70b-instruct-2509` --
  previously 0.00 on literally every book because its 65536-token context
  window was smaller than every book's blind-fraction prompt (~98K tokens
  for the largest) -- is no longer categorically incompatible: the
  narrower input now fits its context window on most books, and it scores
  F1 0.42 overall (95/158 found), no longer zero. LLM standalone still
  trails the heuristic's 0.58 and costs far more time, so this remains
  informational, not a recommendation to route production through it --
  but the truncation/input-size bug that was previously suppressing its
  real accuracy is closed. The remaining zero-recall books
  (`9783789057366.pdf`, `9783848704316.pdf`, `dnb-36942798X.pdf`) score at
  or near 0.00 across every LLM model (one exception: `apertus` finds 1 of
  56 expected chapters on `9783789057366.pdf`, recall 0.02), but these are
  the same three books the heuristic pipeline also can't recover any
  signal from (see "Diverse real-library evaluation set" above --
  degenerate/absent text layers, OCR quality) -- not an LLM-specific gap,
  a shared data-quality problem no text-based strategy can solve.
- **The "best" model by F1 changes between runs, and one model's own time
  swung by more than an order of magnitude.** After the full cache
  regeneration below, nine of the ten cached models cluster tightly at F1
  0.43-0.48 (`qwen3.5-122b-a10b` remains a clear outlier at 0.19, the
  weakest model overall); the top four (`glm-4.7`, `deepseek-v4-flash`,
  `mistral-medium-3.5-128b`, `devstral-2-123b-instruct-2512`) all land at
  F1 0.48, and `evaluation/generate_report.py`'s `_best_llm_model()`
  (highest F1, ties broken by lower time) currently picks `glm-4.7`
  (239.7s total). `qwen3.6-27b` -- the previous snapshot's pick at F1 0.49
  and 1922.5s -- dropped to F1 0.43 at 2388.6s this run, on the same
  prompt and the same corpus: real KISSKI model-serving variance (latency
  and occasional zero-chapter responses on individual book/model pairs,
  logged as `giving up (invalid_or_truncated_json)`) is large enough to
  reorder the "best model" pick run over run. Treat any single run's "best
  model" as noisy, not a stable ranking; a deployment decision would look
  at the tightly-clustered F1 0.43-0.48 group as roughly equivalent and
  pick among them on time/cost instead.
- **`citation_pages` mapping accuracy is now measured, and the fix works.**
  Before `docs/superpowers/plans/2026-08-08-citation-pages-mapping.md`,
  `_chapters_from_located` re-derived a chapter's printed start/end pages
  by scanning body-page text, ignoring the printed page number already
  parsed off the table of contents (`TocEntry.printed_page_number`) --
  fragile, and with no metric tracking how often it worked. The fix
  prefers that TOC-declared value first, falls back to cross-page anchor
  interpolation, and (for chapter ends) to reading the page before the
  next chapter's raw start or the book's last page. Measured on the
  full corpus (restricted to chapters whose located PDF range exactly
  matches ground truth and whose expected `citation_pages` is non-null,
  via `citation_pages_metrics`): **heuristic scores 142/144 exact start
  matches (98.6%) and 143/144 end matches within tolerance (99.3%)**;
  the best LLM model scores 105/106 on both (99.1%) -- the LLM strategy's
  own printed-page-number field was blind to roman numerals before Task 6
  of that plan fixed the extraction schema, which this run's numbers now
  reflect. **Outline scores far lower** (5/31 start, 15/31 end, 16%/48%)
  because `extract_outline_candidates` never populates a TOC-declared
  page number at all (a PDF outline/bookmark has no printed-page-number
  field to read) -- every outline-sourced chapter falls through to the
  weaker inference/scan fallbacks, and roughly half don't resolve a page
  at all (`start_coverage`/`end_coverage` both 0.48). This is a real,
  currently-accepted gap in the outline strategy, not a measurement
  artifact; closing it would mean giving `extract_outline_candidates` a
  way to read a printed page number where one is visible near the
  bookmark's target page, not yet attempted.
- The LLM cache now covers 10 distinct KISSKI models, up from the initial
  5: the nightly `fill-gaps` GitHub Actions job added 5 more
  (`qwen3.5-122b-a10b`, `qwen3-coder-next`, `qwen3.6-27b`,
  `qwen3.6-35b-a3b`, `mistral-medium-3.5-128b`) under the *old, buggy*
  extraction code before this fix reached `main`; merging that commit in
  and re-running the new `evaluation/refresh_llm_cache.py --mode full`
  (added specifically for this situation -- `--mode top5`/`--mode
  fill-gaps` alone never force a re-run of an already-cached model)
  regenerated all ten under the fixed code, which is what this table
  reports. `--mode fill-gaps`'s nightly schedule
  (`.github/workflows/refresh-llm-cache.yml`) will keep growing coverage
  to every non-busy KISSKI model on every book over time, once a
  `KISSKI_API_KEY` repository secret is configured.

## Layout-based TOC/chapter-first-page classifier pilot

`evaluation/scripts/evaluate_layout_toc_classifier.py` trains a
leave-one-book-out (LOBO) classifier on ten geometric layout features
(`evaluation/scripts/layout_features.py`, derived from cached ALTO XML) and
scores whether it can identify table-of-contents pages and chapter-opening
pages purely from page layout, no text content -- see
`docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`
for the pilot's design and decision bar. `full_recall_fraction` is the
share of books, across LOBO folds, whose predicted candidate pages include
every true chapter-opening page and at least one true TOC page (bar: ≥90%);
`avg_candidate_fraction` is the average share of a book's pages that end up
in that candidate set at all -- how much the classifier actually narrows
the page list down (bar: ≤15%, smaller is better). The original pilot run
came back **NOT MET**: 16% `full_recall_fraction` against a
comfortably-cleared 5.3% `avg_candidate_fraction`.

A follow-up investigation
(`docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md`)
root-caused most of the shortfall to 4 of the 10 features --
`width_mean`, `width_var`, `left_margin_mean`, `left_margin_var` -- being
raw, unnormalized ALTO point coordinates rather than fractions of page
width, unlike the other position-derived features
(`first_text_vpos_fraction`, `line_density`), which already divide by page
height. This tracked almost exactly with the corpus split: page width
varies far more in `copyrighted-scans` (304-991pt) than `open-access`
(420-595pt), and the original run's 8-book cluster stuck at exactly 0%
`chapter_first` recall was entirely `copyrighted-scans` books:

| corpus | books | avg `chapter_first` recall | books at 0% recall | books at 100% recall |
| --- | --- | --- | --- | --- |
| open-access (original) | 37 | 83% | 0 | 9 |
| copyrighted-scans (original) | 13 | 20% | 8 | 1 |

`extract_page_features` was fixed to divide these four features by page
width before computing statistics on them (Task 1 of the normalization
follow-up), and the pilot was re-run against the same cached ALTO XML with
no other changes. The fresh result is still **NOT MET**, but slightly
better: 18% `full_recall_fraction` (vs. 16% before) and 5.4%
`avg_candidate_fraction` (vs. 5.3% before, still comfortably under the 15%
bar). The per-corpus `chapter_first`-recall breakdown, re-computed the same
way as the original diagnosis, shows the corpus split itself is
essentially unchanged by the fix:

| corpus | books | avg `chapter_first` recall | books at 0% recall | books at 100% recall |
| --- | --- | --- | --- | --- |
| open-access (fresh) | 37 | 81% | 1 | 9 |
| copyrighted-scans (fresh) | 13 | 19% | 8 | 0 |

So the width-normalization fix did what it set out to do (the four
previously-unnormalized features are now genuinely comparable across
books of different page widths) without meaningfully closing the gap
between the two corpora, or the overall `full_recall_fraction` bar. This
matches the normalization spec's own root-cause writeup, which named the
unnormalized features as the *dominant* factor but not the only one: it
separately quantified a secondary, compounding issue -- the decision bar's
requirement of literal 100% `chapter_first` recall per book -- and found
that even relaxing it to an 80% tolerance only lifted `full_recall_fraction`
to 34% on the original (pre-normalization) run, still well short of the
90% bar. That bar-strictness finding is the next-most-likely place to look
if this pilot is picked up again; scoping or implementing a fix for it is
explicitly out of scope for both this run and the normalization follow-up
that produced it
(`docs/superpowers/specs/2026-08-10-layout-toc-classifier-feature-normalization-design.md`'s
"Out of scope" section).

### Follow-up: replacing textless/degenerate-text corpus PDFs with OCR'ed versions

A closer look at the 8-book cluster still stuck at exactly 0%
`chapter_first` recall after the normalization fix found a second, larger
root cause for 6 of the 8: `pdfalto` (the pilot's only extraction tool,
`evaluation/scripts/pdfalto_runner.py`) reads a PDF's own embedded
text/layout directly and has no OCR fallback of its own -- unlike
`chapter_segmentation`'s own text-based pipeline, which recovers usable
text for these exact books via a dedicated OCR pass
(`src/chapter_segmentation/ocr.py`, `.ocr-cache/`, see "Diverse
real-library evaluation set" above: `9781409403906.pdf`,
`9783465016878.pdf`, `9783848704316.pdf`, `dnb-36942798X.pdf` had "no text
layer at all"; `9780367439712.pdf`, `9783789057366.pdf` had a "degenerate
text layer"). On these 6 books, `pdfalto` extracted zero text lines at all
on many pages (an all-zero feature vector) or near-uniform, unresolvable
font-size data -- no amount of feature normalization can recover a signal
that was never extracted in the first place.

Per this project's evaluation philosophy (production code should OCR at
run time; the *evaluation corpus* should already contain suitable data, so
a pipeline bug and a data-quality gap are never conflated), the fix was to
replace the 6 affected PDFs in the shared corpus
(`evaluation/corpus/copyrighted-scans/`, gitignored, symlinked identically
into every worktree) with OCR'ed versions, not to teach `pdfalto_runner.py`
to run OCR itself. `evaluation/scripts/ocr_evaluation_pdfs.py`'s existing
OCR path only produces cached plain text (`list[str]`), not a new PDF with
an embedded text layer, so it can't be reused for this -- `ocrmypdf`
(installed via `brew install ocrmypdf`, plus `tesseract-lang` for German)
was used instead: `ocrmypdf --force-ocr -l <lang> <original> <output>`,
with each original preserved alongside as `<key>.original.pdf`. Verified
before re-running the pilot: page count unchanged for all 6 (pypdf), and
`pages_need_ocr` (`src/chapter_segmentation/segmentation.py`) now returns
`False` for all 6, where it previously returned `True`.

Re-running the pilot (same cached ALTO XML for the other 44 books; the 6
fixed books' `.layout-cache/` entries were deleted so `pdfalto` re-ran on
the new PDFs) shows real, measurable per-book improvement on 4 of the 6:

| book | chapter_first recall before | chapter_first recall after |
| --- | --- | --- |
| `9783848704316` | 0% | 73% |
| `9781409403906` | 0% | 42% |
| `9783465016878` | 0% | 23% |
| `dnb-36942798X` | 0% | 6% |
| `9780367439712` | 0% | 0% (unchanged) |
| `9783789057366` | 0% | 0% (unchanged) |

The two unchanged books fail for two different, already-diagnosed reasons,
not a leftover OCR problem: `9780367439712`'s chapter-opening pages have
clean, strong font-size signal both before *and* after OCR (12/12 pages
cross the title-detection threshold either way) -- this is a LOBO
model-generalization gap (its true positives top out around probability
0.45 against a 0.68 threshold calibrated from other books), unrelated to
data quality. `9783789057366` still shows weak font-size differentiation
even with real OCR'd text (only 2 of 56 chapter-opening pages cross the
threshold) -- plausibly because this scan's low average image DPI (~91,
per `ocrmypdf`'s own logged warnings) is too coarse for Tesseract's
per-line font-size estimation to reliably separate title-sized from
body-sized text.

Despite that per-book progress, the overall `full_recall_fraction` stayed
flat at **18%** (was 18% after normalization, 16% originally) -- still
**NOT MET**. `avg_candidate_fraction` improved slightly, 5.4% to 4.4%,
still comfortably under the 15% bar. The corpus-wide number held flat
because every held-out book's classifier is trained on all *other* books'
rows in this leave-one-book-out setup: swapping degenerate rows for real
ones in 6 books measurably shifted many *other*, untouched books' scores
too (both up and down), roughly canceling out in the aggregate. This is
further evidence, on top of the normalization follow-up's own finding,
that the decision bar's literal "100% `chapter_first` recall per book"
requirement -- not any single data-quality or feature issue -- is this
pilot's dominant remaining blocker: most of the corpus already sits in a
60-95% per-book recall band, comfortably real signal, just short of the
all-or-nothing bar.

`evaluation/CLAUDE.md`'s "Step 0a" now requires a real, usable embedded
text layer (checked with `pages_need_ocr`) before any scanned PDF is added
to `copyrighted-scans/` going forward, so this gap doesn't recur silently
for future books.

### Follow-up: recall-target tuning, concentrating on the open-access corpus

With the data-quality issues above addressed, this follow-up asked how far
the existing pipeline -- unchanged features, unchanged model, unchanged
decision bar -- could be pushed by tuning `evaluate_layout_toc_classifier.py`'s
own calibration knob, `_RECALL_TARGET`. `select_threshold` picks the highest
per-fold probability threshold that still achieves at least `_RECALL_TARGET`
recall on that fold's *training* positives; the value had been an arbitrary
`0.90` since the pilot's first run. Since `avg_candidate_fraction` had been
sitting far under its 15% budget the whole time (4.4% most recently), there
was untapped room to trade some of that slack for recall by raising the
target. Per the "concentrate on open-access first, fix outliers later"
priority, `copyrighted-scans` books stayed in the training pool throughout
(removing them from training was tried and made open-access recall *worse*,
not better -- see below) but their own scores were tracked separately rather
than optimized for.

A sweep over `_RECALL_TARGET` from 0.90 to 1.00 (LOBO over the full 50-book
corpus) found a steep, then flat, then explosive response:

| `_RECALL_TARGET` | `full_recall_fraction` (all) | `avg_candidate_fraction` (all) |
| --- | --- | --- |
| 0.90 (previous) | 18% | 4.4% |
| 0.95 | 20% | 5.7% |
| 0.96 | 26% | 6.1% |
| **0.97** | **28%** | **7.0%** |
| 0.98 | 28% | 11.1% |
| 0.99 | 28% | 21.9% |
| 1.00 | 80% | 87.3% |

0.97 is the cheapest point on the 0.97-0.99 plateau -- same recall as 0.98
and 0.99, at roughly half to a third of their candidate-fraction cost -- and
comfortably clear of 1.00's cliff, where the "threshold" degenerates to
"flag almost every page" and stops being a useful filter at all. Broken down
by corpus at 0.97:

| corpus | `full_recall_fraction` (before -> after) | `avg_candidate_fraction` |
| --- | --- | --- |
| open-access | 24.3% -> 35.1% | 6.6% |
| copyrighted-scans | 0% -> 7.7% | 7.9% |

Three other tuning ideas were tried and rejected, in the interest of
recording negative results alongside the positive one:

- **Training on open-access books only** (excluding `copyrighted-scans`
  entirely from the LOBO training pool, not just from the target metric):
  this made open-access recall *worse* (16.2% vs. 24.3% at the old 0.90
  target) -- the other corpus's rows, despite their own data-quality
  problems, still contribute generalizable signal rather than just noise.
- **A new feature, `max_font_vpos_fraction`** (the vertical position of the
  page's *largest-font* line, as opposed to `first_text_vpos_fraction`'s
  topmost line regardless of size -- meant to stop a running header above
  the real title from collapsing that signal to near-zero): dropped
  open-access `full_recall_fraction` from 43.2% to 29.7% at the same 0.97
  target. With only ~600 positive `chapter_first` rows across 37 books, an
  11th feature widens the model's overfitting surface faster than it adds
  real signal.
- **Splitting `_RECALL_TARGET` per label** (a low target for `toc`, whose
  pass bar is lenient -- "at least one hit" -- freeing up candidate-fraction
  budget for a higher `chapter_first` target): backfired. A *lower* target
  makes `select_threshold` pick a *higher*, more selective threshold; for
  several books that threshold was high enough to reject every `toc`
  candidate, failing that label's lenient bar outright. `min_samples_leaf`
  values above the existing default of 1 (tried 2, 3, 5, 10) were also all
  strictly worse.

The result is still **NOT MET** (28% vs. the 90% bar), and the previous
follow-up's bar-strictness finding remains the dominant blocker -- this
tuning pass narrows the gap without closing it. But it is a real,
non-cosmetic improvement obtained purely by recalibrating an existing,
already-arbitrary constant: open-access `full_recall_fraction` improved by
nearly half (24.3% to 35.1%) while candidate fraction stayed at less than
half the 15% budget. `copyrighted-scans` remains the weaker corpus by a wide
margin and continues to be treated as the deferred outlier bucket, per the
"open-access first" priority this follow-up was scoped to.

### Follow-up: relaxing the per-book bar, and a model-architecture swap

Two more changes, requested together: relax the pilot's literal "100% of a
book's chapter_first pages" pass bar (the dominant blocker named in every
follow-up above), and investigate whether the classifier's *model* -- not
just its calibration -- was leaving recall on the table.

**Relaxing the bar.** `evaluate_leave_one_book_out` gained a
`chapter_first_recall_tolerance` parameter (module default `0.90`,
`_CHAPTER_FIRST_RECALL_TOLERANCE`), replacing the exact `recall == 1.0`
check with `recall >= tolerance`. This only changes how a book is scored as
"fully recalled" for `full_recall_fraction` -- it has no effect on which
candidate pages get produced. A sweep from 1.00 down to 0.50 (LOBO, full
50-book corpus, still on the tree-based model at the time) showed a steep,
monotonic response with no plateau: 28% (1.00) -> 32% (0.95) -> 40% (0.90)
-> 48% (0.80) -> 60% (0.70) -> 72% (0.50). Even 0.50 didn't reach the
pilot's own 90% "MET" bar, and at that tolerance "passing" only requires
catching half a book's chapter openings -- too weak to call the classifier
a real detector. 0.90 (miss at most 1 in 10 chapter-openings per book) was
chosen as a tolerance that stays meaningful while acknowledging that a
book's natural layout diversity (an unnumbered first chapter, a
part-divider) means no realistic amount of tuning closes the last mile to
literal 100%.

**Model architecture.** Asked directly why recall was still low given the
tuning already done, three classifiers were compared head-to-head on
identical features, at the same recall-calibration target:

| model | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| `HistGradientBoostingClassifier` (previous) | 40% | 7.0% |
| `RandomForestClassifier` | 0% | 0.1% |
| `LogisticRegression` (+ `StandardScaler`) | 44% | 15.0% |

`RandomForestClassifier` is not viable here -- its `predict_proba` is badly
miscalibrated for this data shape (small, imbalanced, near-duplicate rows
within a book), so `select_threshold` picks a threshold that accepts almost
nothing on held-out books. `LogisticRegression` clearly outperforms the
tree-based model on recall, and the mechanism is architectural, not just a
better hyperparameter: `HistGradientBoostingClassifier` partitions feature
space into discrete leaf regions and calibrates its threshold against one
fold's own training-probability distribution -- a held-out book whose
feature distribution sits slightly off from that fold's training pool can
fall in a gap between leaf regions that the threshold doesn't line up with,
even when the page has an obvious, real chapter-opening signal.
`LogisticRegression`'s smooth, continuous, linear score has no such
discontinuity, so it transfers across books far more gracefully -- evidence
that the true relationship between these ten geometric features and
"chapter_first-ness" is close enough to monotonic/linear that a simpler
model generalizes better than a more expressive one. This is consistent
with two earlier negative results in this same investigation
(the rejected `max_font_vpos_fraction` feature and the rejected
higher-`min_samples_leaf` settings): the pipeline was never capacity-limited,
it was data-limited, and a lower-capacity model suits that regime better.

The pilot's model was switched to `LogisticRegression(class_weight="balanced")`
with a per-fold `StandardScaler` (needed because, unlike tree splits, a
linear model's fit is sensitive to feature scale; tree-based
`HistGradientBoostingClassifier` never needed one). `_RECALL_TARGET`'s
default moved from `0.97` (tuned for the tree model) to `0.80` -- the
highest point on `LogisticRegression`'s own curve that still respects the
*previous* 15%-candidate-fraction budget:

| `LogisticRegression` `recall_target` | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| 0.80 (new default) | 44% | 15.0% |
| 0.85 | 54% | 17.0% |
| 0.90 | 58% | 20.5% |
| 0.95 | 68% | 39.2% |
| 0.97 | 78% | 61.4% |

**Both constants are now runtime-configurable**, via `--recall-target` and
`--chapter-first-recall-tolerance` (or as `evaluate_leave_one_book_out`
keyword arguments), rather than requiring a source edit. They control
different things: `recall_target` is the real "how many false-positive
candidate pages am I willing to live with" dial a consumer of the
classifier would actually set (the table above is that trade-off surface);
`chapter_first_recall_tolerance` only changes how *this evaluation script*
scores a book as "fully recalled" for its own aggregate report, with no
effect on inference behavior. The 0.80 default keeps candidate volume
inside the pilot's original budget; a consumer able to tolerate more
false-positive candidate pages (e.g. because a downstream stage reviews
every candidate cheaply, on institutionally-hosted models with no
per-request cost) can raise `--recall-target` for substantially higher
recall, per the curve above.

With the new default (`LogisticRegression`, `recall_target=0.80`,
`chapter_first_recall_tolerance=0.90`), the full 50-book LOBO run gives
`full_recall_fraction=44%`, `avg_candidate_fraction=15.0%` -- still **NOT
MET** against the pilot's 90%/15% bar, but nearly triple the very first
run's 16%, and the highest `full_recall_fraction` reached at or under the
original candidate-fraction budget across every follow-up so far. Per
corpus:

| corpus | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| open-access | 54.1% | 15.2% |
| copyrighted-scans | 15.4% | 14.4% |

**Would more ground truth help further?** A learning-curve check (fixed
15-book held-out test set, `LogisticRegression` retrained on random subsets
of the remaining pool at sizes 10/15/20/25/30/35 books, 5 repeats per size)
found `full_recall_fraction` on that fixed test set flat within noise
across the entire range (25.3%, 28.0%, 22.7%, 22.7%, 24.0%, 26.7% -- no
scaling trend). Combined with `LogisticRegression`'s low parameter count
(11 weights total) making it very unlikely to be data-starved, this
suggests generic corpus growth ("more books like the ones already here")
has limited further upside -- the ceiling looks like a feature/geometry
information limit rather than a training-volume limit. If more ground
truth is collected, it would be higher-leverage to specifically target the
underrepresented templates already characterized in this investigation
(unnumbered first chapters, chapter-openings with no font-size or
whitespace distinction from body text) rather than generic new books, since
those are the specific cases the current features structurally can't
separate from ordinary pages.

## LLM-fallback results (archived -- script removed)

**Superseded by "Per-strategy standalone results" above, which measures
what the LLM itself can find rather than whether a merge/fallback path
fires.** `evaluation/scripts/evaluate_chapter_segmentation_llm_fallback.py`,
referenced below, no longer exists; kept here only as a historical record
of `analyze_attachment_with_llm_fallback`'s merged behavior at the time it
was last run. If that merged pipeline needs dedicated evaluation again, a
new script would need to be written -- none currently does this.

With the heuristics above, a full run of that now-deleted script
(KISSKI-backed preset) reported **identical numbers to the pure-heuristic
harness, with neither fallback path firing on any book**: TOC extraction
never triggered because the regex path found a usable listing everywhere,
and per-entry ambiguity was resolved heuristically by the TOC-order
constraints before the LLM would be consulted. This run predated the
layout-mode fallback and OCR route described above, and only covered the 7
originally-committed books, not the 10 "diverse real-library" books.

## NuExtract-1.5-tiny zero-shot baseline (2026-08-09)

`evaluation/scripts/evaluate_nuextract_baseline.py` against Ollama serving
`hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0` (the script's documented
default -- see
`docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md`).
Scores only TOC-listing extraction (title + printed_page_number), not full
chapter-boundary localization.

| Corpus | Books | Expected chapters | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| open-access | 37 | 601 | 0.00 | 0.00 | 0.00 |
| copyrighted-scans | 13 | 312 | 0.00 | 0.00 | 0.00 |
| **Total** | **50** | **913** | **0.00** | **0.00** | **0.00** |

**Decision (per the spec's "Decision criteria"): NuExtract-1.5-tiny does
not clear the bar for the companion spec's production wiring.** The
initial 0.00/0.00 result (via Ollama/GGUF) turned out to be a real but
*partial* explanation -- a follow-up retest via the original `transformers`
checkpoint (see "Follow-up" below) rules out GGUF-conversion fidelity as
the sole cause and shows the model has a genuine, load-bearing capability
gap on this task: it cannot reliably extract structured chapter data from
scan-window text that includes any surrounding non-TOC content, which is
exactly what this project's scan-window selection always produces.

**Root cause: the model stops generating almost immediately (empty output)
on real book front-matter text, independent of this project's code.**
Investigated directly against Ollama's `/api/generate` endpoint, bypassing
`evaluation/nuextract_baseline.py` entirely, to rule out a bug in this
repo's prompt-building or response-parsing:

- A short, clean synthetic example (`<|input|>/### Template/### Text/
  <|output|>` with `"My name is John."` as the text) works correctly --
  the model fills the template as expected (`{"name": "John"}`), and a
  short synthetic TOC snippet ("Introduction ... 1 / Chapter One:
  Foundations ... 12") also produces syntactically valid, mostly-correct
  JSON.
- The *same* template against a real book's front-matter text (a ~1,300-
  token excerpt of `9781771993661.pdf`'s copyright/title-page text, well
  under the 16,384-token `num_ctx` budget -- see `call_ollama`'s
  docstring) reliably produces **zero generated tokens**
  (`eval_count: 1, done_reason: "stop"`, i.e. the very first sampled
  token is an end-of-sequence token) at `temperature=0`. Binary-searching
  truncations of that same text (2,500 / 3,000 / 3,500 / 3,800 / 4,000 /
  4,200 / 4,400 / 4,600 / 4,800 / 4,971 characters) shows this isn't a
  fixed length threshold -- pass/fail flips unpredictably between nearby
  cutoffs, which points to greedy-decoding brittleness on this specific
  (quantized, community-converted) checkpoint rather than a clean
  context-length cliff.
- Raising `temperature` to 0.3 avoids the immediate stop but the model
  then **echoes the input text back verbatim** instead of extracting a
  template -- also not usable output, just a different failure mode.
- This reproduces **identically across two independently-produced
  third-party GGUF conversions** -- `QuantFactory/NuExtract-1.5-tiny-GGUF`
  (this script's default) and `mradermacher/NuExtract-1.5-tiny-GGUF` --
  ruling out a single bad upload. `ollama show <model> --modelfile` on
  both shows Ollama auto-detected a **generic Qwen2 ChatML template**
  (`<|im_start|>`/`<|im_end|>`, with FIM/tool-call scaffolding), not
  NuExtract's own fine-tuned `<|input|>`/`<|output|>` format -- consistent
  with (though not proven to be the sole cause of) the conversion not
  faithfully carrying over whatever made the original HF checkpoint
  reliably respond to that format. (`raw: true` bypasses Ollama's own
  templating for the actual generate call either way -- this is a signal
  about the conversion, not evidence the wrong template was applied to
  our requests.)
- The full 50-book run above is the direct consequence: 44 of 50 books
  return zero predicted entries outright (immediate-stop failure on every
  scanned page range), and the other 6 return exactly one spurious,
  non-matching entry -- none of the 913 expected chapters across either
  corpus are recovered.

### Follow-up: original Hugging Face checkpoint (2026-08-09)

Retested by loading `numind/NuExtract-1.5-tiny` directly through
`transformers` (`AutoModelForCausalLM`/`AutoTokenizer`, fp32, greedy
decoding, run on this machine's Apple Silicon GPU via `mps`), bypassing
GGUF conversion and Ollama entirely -- the one genuinely unmeasured
possibility identified above. Ad hoc script, not committed (probe only);
used `evaluation/nuextract_baseline.py`'s own `build_prompt`/
`parse_response`/`score_book` against real pages loaded via
`evaluation/harness.py`, so results are directly comparable to the table
above.

- **The immediate-EOS failure does not reproduce.** On the exact same real
  front-matter text that produced `eval_count: 1` under Ollama, the
  original checkpoint generates fluently for hundreds of tokens. This
  rules out "GGUF conversion broke the model's stopping behavior" as a
  complete explanation.
- **But it does not extract, either -- it echoes.** Given a scan window
  that mixes the actual table of contents with surrounding noise (the
  book's copyright/front-matter page before it, or the start of the
  Foreword body text after it -- i.e. exactly what
  `chapter_segmentation.segmentation._llm_scan_indices` selects for every
  real book, by design, since it can't know in advance which lines are
  the ToC), the model does not fill the JSON template at all. It falls
  into a degenerate mode: copying the input text back verbatim (with
  occasional token substitutions, e.g. quote-mark hallucinations), then
  looping on a repeated phrase until the token budget runs out. This
  reproduced on all three tested windows (page 3+4+5, page 4+5+6, and the
  original 4-page window) for `9781771993661.pdf`.
- **Given a hand-curated, TOC-only window (no noise), the model does
  attempt genuine extraction** -- it emits syntactically valid JSON
  matching the template, and gets several `printed_page_number` values
  right. But entity binding is unreliable: it collapses numbered chapters
  (e.g. "1. Race and Colonialism in Socio-legal Studies in Canada") down
  to their enclosing part labels ("Part I"), losing the actual chapter
  titles, and it misattributes authors -- reusing chapter 1's author list
  ("Carmela Murdocca, Shaira Vadasaria, Timothy Bryan") for several
  unrelated later entries instead of each entry's real contributors.
  Scored against `9781771993661.expected.json` via `score_book`, this
  best-case, hand-cleaned input still only reaches **precision=0.08,
  recall=0.10, f1=0.09** (1 of 10 expected chapters correctly matched).

**Conclusion: this is a genuine model-capability limit, not a serving-path
artifact.** NuExtract-1.5-tiny's "template-fill" paradigm assumes the
input text is already close to the target structure -- it has no
instruction channel to say "ignore the surrounding noise and extract only
the table of contents," unlike the instruction-following cloud LLMs
(`llm_extract_toc_entries` in `segmentation.py`) this spike was evaluating
it as a local replacement for. Because real scan windows always carry
that noise, and because even the noise-free best case scores far below
useful (f1=0.09 on the one book tested), NuExtract-1.5-tiny at the "tiny"
size is not a viable zero-shot drop-in for this task. See the next section
for a retest against a larger, newer model in the same family, which
closes most of this gap.

## NuExtract3 (4B): a larger, newer model closes the gap (2026-08-09)

This machine (Apple M4, 32 GB unified memory) can comfortably run
NuExtract's newer, larger models locally -- checked actual download sizes
via the Hugging Face API: `numind/NuExtract-2.0-4B` (7.5 GB fp16),
`numind/NuExtract-2.0-8B` (16.6 GB fp16), `numind/NuExtract3` (9.3 GB
fp16, the current flagship, released after this spike's original design
and built on `Qwen3.5-4B` rather than 1.5-tiny's `Qwen2.5-0.5B`). Retested
using `numind/NuExtract3` -- a materially different, more capable
generation, not just a scaled-up copy of the same weak checkpoint. Ad hoc
scripts, not committed; loaded via `transformers`
(`AutoModelForImageTextToText`/`AutoProcessor`, fp16, greedy decoding,
`mps`), reusing this repo's `NUEXTRACT_TEMPLATE`/`parse_response`/
`score_book`/`evaluation/harness.py` for direct comparability with the
tables above. NuExtract3 uses a proper chat template (`apply_chat_template`
with `template`/`instructions`/`enable_thinking` kwargs) rather than
1.5-tiny's raw `<|input|>`/`<|output|>` convention.

- **Single-book check, same noisy window that broke NuExtract-1.5-tiny**
  (`9781771993661.pdf`, pages 3-6, front-matter + ToC + start of Foreword,
  no `instructions` kwarg used): **precision=0.91, recall=1.00, f1=0.95**
  (10/10 expected chapters correctly matched, correct titles, correct
  authors, correct page numbers; one spurious extra entry). Adding an
  `instructions` string ("extract only the table-of-contents entries...
  ignore copyright notices... and body/prose text") or hand-cleaning the
  window down to ToC-only text both produced the same result
  (f1=0.91, one additional spurious "Contributors" entry) -- unlike
  1.5-tiny, this model doesn't need the noise removed to work.
- **10-book random sample from the open-access corpus** (seeded,
  `evaluation/metrics.py`'s `MicroAggregate`, real scan windows via
  `_llm_scan_indices`, no OCR-only books in the sample): **aggregate
  precision=0.95, recall=0.73, f1=0.83** -- 8 of 10 books scored f1
  between 0.72 and 1.00 (four of them exactly 1.00); the other 2 scored
  0.00 and were investigated individually rather than averaged over
  blindly:
  - `9782375460122.pdf` (French-language, a "SOMMAIRE" spanning 6 dense
    pages with many nested sub-headings): a genuine extraction miss, not
    a budget issue -- confirmed by rerunning with the output budget raised
    from 1,500 to 3,500 tokens, which changed nothing (still only 46
    tokens generated). The model latched onto the book's own
    cataloging-in-publication bibliographic entry on the noise page before
    the ToC and extracted that as a single spurious "chapter," never
    reaching the actual SOMMAIRE on the following pages.
  - `9781783741953.pdf` (a deeply hierarchical ToC -- top-level chapters
    each with many numbered sub-sections, e.g. "4.1", "4.2", "4.2.1"):
    **this one was a genuine truncation** at the original 1,500-token
    output budget (0 parseable entries). Raised to 3,500 tokens, the model
    completed cleanly (2,973 output tokens) and correctly extracted the
    *entire* hierarchy -- all top-level chapters and every numbered
    sub-section, with correct titles and page numbers throughout. But the
    ground truth only counts 7 top-level chapters, so the 89 extracted
    entries score precision=0.01 -- **a template/instructions mismatch
    (we never told it to extract top-level entries only), not a
    comprehension failure.** The model demonstrably parsed this book's
    full multi-level structure correctly.

**Decision: NuExtract3 (4B) is a credible local zero-shot candidate for
this task, unlike NuExtract-1.5-tiny, and is worth a full-corpus run and
production-wiring spike before deciding against a local-model approach.**
Both zero-score books point at fixable gaps rather than a hard capability
ceiling: the French-language miss suggests either a prompt-language
adjustment or accepting some recall loss on non-English ToCs, and the
over-extraction case is resolved by adding an `instructions` string that
scopes extraction to top-level entries (already shown above not to hurt
the good case) and/or raising the output token budget for long books.
Trade-offs versus the cloud-LLM baseline this spike was benchmarking
against: **speed** (70-250 seconds per book on this machine's `mps`
backend vs. a cloud API call) and **setup cost** (9.3 GB download, 32 GB
RAM headroom, ~20-90 second model load even from local disk cache) are
real costs a production decision would need to weigh against not
depending on a paid API.

**Recommended next step:** run the full two-corpus, 50-book baseline
(mirroring the table at the top of this document) against NuExtract3 with
an `instructions` string added, and compare against the existing cloud-LLM
strategy numbers before deciding whether to wire this into
`evaluation/scripts/evaluate_nuextract_baseline.py` as a first-class,
committed option.

### Optimization: MLX vs. `transformers`+`mps` (2026-08-09)

Before running the full corpus, checked whether the serving path used
above (`transformers`, fp16, `mps`) was leaving speed on the table. It
was: the model load explicitly warned `The fast path is not available
because one of the required library is not installed` (`flash-linear-
attention`/`causal-conv1d` -- both CUDA/Triton-only, not installable on
Apple Silicon), meaning Qwen3.5's hybrid linear-attention layers were
running through a slow generic fallback rather than a fused kernel.
Retested with `mlx-vlm` against `numind/NuExtract3-mlx-4bits` -- the
NuMind-published 4-bit quantization (3.0 GB vs. 9.3 GB fp16) run through
MLX, Apple's native array framework, instead of PyTorch's `mps` backend.

- **Single-book check** (`9781771993661.pdf`, same window as above):
  **25.3s vs. 69.6s** (2.75x faster), 27 tokens/sec vs. ~8.6 tokens/sec,
  for essentially the same output (f1=0.95 both ways).
- **10-book sample** (same seeded sample as the `transformers` run
  above): total raw generation time **323s vs. 1,080s (3.3x faster)**;
  aggregate **precision=0.82, recall=0.74, f1=0.78** vs. the fp16 run's
  **precision=0.95, recall=0.73, f1=0.83** -- a modest precision dip
  consistent with 4-bit quantization noise, recall essentially unchanged.
  The same two previously-diagnosed problem books (the French-language
  miss and the deeply-nested-ToC over-extraction) show the same failure
  modes, not new ones -- nothing about switching to MLX changed *what*
  fails, only *how fast* the successful cases run.
- Model load is also far cheaper once weights are cached locally: 2.5s
  (MLX, warm) vs. 18.8-381.8s observed for `transformers` (dependent on OS
  disk-cache state).

**Decision: use `numind/NuExtract3-mlx-4bits` via `mlx-vlm` as the runner
for the full-corpus run** -- same architecture, same accuracy class, ~3x
less wall-clock time, and a much smaller download.

### Full 50-book, two-corpus run (2026-08-09)

Same conditions as the NuExtract-1.5-tiny table at the top of this
document -- `numind/NuExtract3-mlx-4bits`, no `instructions` kwarg (this
is the zero-shot baseline; instructions weren't re-added here to keep the
comparison to 1.5-tiny apples-to-apples), `max_tokens=2000`, both
corpora, real scan windows via `_llm_scan_indices`. Ad hoc script, not
committed. Total wall-clock: 3,180s (~53 minutes) for 50 books.

| Corpus | Books | Expected chapters | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| open-access | 37 | 601 | 0.64 | 0.73 | 0.69 |
| copyrighted-scans | 13 | 312 | 0.49 | 0.32 | 0.39 |
| **Total** | **50** | **913** | **0.61** | **0.59** | **0.60** |

(A third `evaluation/corpus/pending/` directory exists with an empty
book list -- `available_books()` correctly returns nothing for it, not a
bug.)

**This is the headline result of the whole NuExtract investigation:
0.00/0.00/0.00 -> 0.61/0.59/0.60**, on the same 50 books, same metric,
same TOC-listing-only scope, just a newer/larger model in the same
family. Two things stand out in the breakdown:

- **`copyrighted-scans` (f1=0.39) trails `open-access` (f1=0.69)
  substantially** -- this corpus is OCR'd (see `evaluation/harness.py`'s
  `analysis_pages_for`), and several books logged `Rotated text
  discovered. Output will be incomplete.` during page extraction, a
  pre-existing OCR-pipeline limitation, not a NuExtract3 issue.
- **The worst-scoring books overlap exactly with an independently
  diagnosed, pre-existing data-quality problem.** `9783789057366.pdf`
  (0/56), `9783848704316.pdf` (0/15), and `dnb-36942798X.pdf` (0/9) all
  score zero here -- and per "Per-strategy standalone results" above,
  these are the *same three books* that score at or near 0.00 across
  **every** cloud LLM model and the heuristic pipeline too, attributed
  there to "degenerate/absent text layers, OCR quality" shared across
  every text-based strategy. NuExtract3 hits the same wall, which is
  reassuring rather than concerning -- it's not a new, NuExtract-specific
  failure, it's the same known corpus limitation every other strategy
  already hits.
- **Two more failure clusters, consistent with what the smaller samples
  already found:** the French-language miss (`9782375460122.pdf`,
  0/17) and the deeply-nested-ToC granularity mismatch
  (`9781783741953.pdf`, 1/7, extracting sub-headings as if they were
  top-level chapters) both reproduce at full scale. A third, new-at-this-
  scale pattern: several books show **long generation times paired with
  zero valid entries** (`9783848704316.pdf` 211s/0 found,
  `9781800649057.pdf` 110s/0 found, `9783839458013.pdf` 113s/0 found,
  `9781783742806.pdf` 109s/0 found) -- consistent with the truncation
  failure mode already root-caused on `9781783741953.pdf` above (output
  budget exhausted before valid JSON completes), suggesting
  `max_tokens=2000` is still too low for a meaningful minority of books
  and a production version of this would need either a larger cap or a
  retry-on-truncation strategy (mirroring the fix already applied to the
  cloud-LLM path's `llm_extract_toc_entries`, per "Per-strategy
  standalone results" above).

**Directional comparison to the cloud-LLM baseline** (caveat: different
corpus -- the LLM standalone table above runs on the 17-book
public-cache corpus, not these 50 books, so this is not a strict
apples-to-apples number): the cloud-LLM cluster there scores F1
0.43-0.48 across its top models. NuExtract3's zero-shot 0.60 overall (and
0.69 on the clean-text `open-access` corpus alone) is in the same range
or better, using a free, local, 4-bit-quantized 4B model with **no
fine-tuning and no `instructions` prompt tuning** -- there's real headroom
left untapped. That's a genuinely striking result for what started this
spike as a "does the tiny model work zero-shot" check.

**Updated recommendation:** this result is strong enough to justify
wiring NuExtract3 into `evaluation/scripts/evaluate_nuextract_baseline.py`
as a first-class, committed option (via `mlx-vlm`, matching the runner
used here) rather than treating it as a dead end -- with two concrete
follow-ups before that: add an `instructions` string scoping extraction
to top-level entries only (already shown above to fix the over-extraction
case without hurting the good case), and raise/retry the output token
budget for the truncation cluster identified above.

### CPU-only deployment check: NuExtract3 vs. NuExtract-2.0-4B (2026-08-09)

MLX is Apple-Silicon-only. A real deployment target under consideration
is a no-GPU, 16 GB RAM, 4-vCPU Linux box (AMD EPYC, KVM-virtualized) --
MLX cannot run there at all, and Ollama currently cannot serve NuExtract3's
Qwen3.5-architecture GGUF either (a known gap: separate `mmproj`
vision-file handling not yet supported in Ollama for this architecture;
`llama.cpp`'s own server/CLI does support it). To get a same-machine,
same-backend comparison isolated from Apple's GPU, both `numind/
NuExtract3-GGUF` (Q4_K_M, 2.71 GB) and `numind/NuExtract-2.0-4B-GGUF`
(Q4_K_M, 1.93 GB, the previous generation, built on the much more
mature/optimized Qwen2.5-VL architecture) were run through `llama-cpp-
python` with `n_gpu_layers=0` and `n_threads=4` (matching the target
box's vCPU count) on this same M4 machine, over a 5-book sample. Ad hoc
script, not committed.

| | NuExtract3 | NuExtract-2.0-4B |
| --- | --- | --- |
| Total time (5-book sample) | 955.6s | 675.0s (**1.42x faster**) |
| Aggregate precision / recall / f1 | 0.97 / 0.94 / 0.96 | 0.99 / 0.96 / 0.97 |
| Model load time | 31.5s | 4.0s |
| Per-book tokens/sec range | 2.0-3.8 | 2.3-6.3 |

NuExtract-2.0-4B was faster on every book in the sample (1.1x-2.2x per
book) and matched or slightly exceeded NuExtract3's accuracy on every one
of them, consistent with `llama.cpp`'s CPU kernels for the older,
standard Qwen2.5-VL transformer architecture being far more mature than
its very recent support for Qwen3.5's hybrid linear-attention layers (the
same gap seen earlier as a missing `flash-linear-attention`/
`causal-conv1d` fast path on the GPU/`transformers` side).

**Caveat:** this ran on the M4's CPU cores, not the target AMD EPYC
vCPUs -- absolute throughput will differ on the real host (their
single-thread performance and memory bandwidth are unknowns from here),
so treat the *relative* result (2.0-4B ~1.4x faster, equal-or-better
accuracy) as the reliable takeaway, not the absolute tokens/sec.

**Recommendation for a no-GPU deployment specifically:**
`numind/NuExtract-2.0-4B` over `NuExtract3` -- faster, smaller download,
equal or better accuracy in this test, and it already has mature Ollama
support today, avoiding the Qwen3.5/Ollama gap entirely.

## NuExtract3 dropped; NuExtract-2.0-4B full-corpus zero-shot baseline (2026-08-10)

Following the CPU-deployment comparison above, NuExtract3 was dropped from
consideration entirely (deleted its `transformers`/GGUF/MLX weight caches,
13.8GB total -- no production code ever referenced it, so nothing else to
remove). All further work targets `numind/NuExtract-2.0-4B` only.

### Backend-dependent bug: `transformers`/MPS silently drops `printed_page_number`

Before trusting a full-corpus number for NuExtract-2.0-4B, a zero-shot run
was attempted via the same `transformers`+`mps`+fp16 path used earlier for
NuExtract3, for consistency. It scored **precision=0.12 recall=0.12
f1=0.12** -- far below the model's own 5-book CPU/`llama.cpp` sample
(f1=0.97) using the *same* five books. Investigation (single-book replay,
`evaluation/corpus/open-access/9781771993661.pdf`) found the root cause:
on `transformers`/MPS, at both fp16 and bf16, the model reliably emits
`"printed_page_number": null` for every chapter entry, even though the
scanned page text plainly contains the printed page numbers (verified by
printing the raw scan window) and the model correctly extracts every
title/author. `match_toc_entries` (`evaluation/nuextract_baseline.py`)
requires a non-null page-number match, so this alone drives recall to
near zero regardless of title-extraction quality. Ruled out tokenization
as the cause (`add_special_tokens=True` vs `False` produced identical
token IDs and identical -- still-null -- output). The same prompt run
through `llama.cpp` (GGUF Q4_K_M, both CPU-only and Metal-offloaded)
correctly filled in every page number and scored f1=0.95 on that book,
reproduced twice. This is a genuine backend-dependent decoding
difference for this model, not a fluke, a tokenization bug, or noise --
likely something in how `llama.cpp`'s own prompt tokenization or KV/RoPE
handling differs subtly from the `transformers` path for this specific
architecture. Since the deployment target is `llama.cpp` on a no-GPU
Linux host anyway, this is moot for production, but it means **any
zero-shot/fine-tuning comparison must go through `llama.cpp`, not
`transformers`/MPS** -- the two backends are not interchangeable for this
model's structured-output behavior.

### Full 50-book, two-corpus run (GGUF Q4_K_M, Metal-offloaded)

Re-ran the full corpus through `llama.cpp` with `n_gpu_layers=-1` (Metal
offload for speed) and `n_ctx=40960` (raised from 8192 after one book's
noisy scan window exceeded it):

| Corpus | Books | Chapters | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| copyrighted-scans | 13 | 312 | 0.30 | 0.12 | 0.17 |
| open-access | 37 | 601 | 0.48 | 0.46 | 0.47 |
| **Total** | **50** | **913** | **0.45** | **0.35** | **0.39** |

Total wall time: 4156s (~69 min). This is NuExtract-2.0-4B's real
zero-shot ceiling on this corpus, through the same backend the target
deployment will use. It is noticeably below NuExtract3's own full-corpus
number on this same corpus (f1=0.60, see above) -- the smaller/older
model is less accurate zero-shot, which was already expected going in;
the CPU-comparison's f1=0.97 was a 5-book best-case sample, not
representative.

**Failure-mode breakdown.** Categorizing all 50 books by outcome:

| Failure type | Books | Share |
| --- | --- | --- |
| True truncation (empty/unparseable output) | 10 | 20% |
| Titles/authors correct, `printed_page_number` null on every entry | 14 | 28% |
| Low but nonzero | 7 | 14% |
| Good (f1 > 0.5) | 19 | 38% |

Truncation (10 books, several with 200-500s generation times before
producing no valid JSON -- the 1500-token output budget is too tight for
these books' larger TOCs) is real but is *not* the dominant cause of the
low aggregate. The larger group (14 books, 28%) is a different, more
specific failure: the model extracts titles and authors **verbatim
correctly** but leaves `printed_page_number` `null` for every entry, even
when the number is plainly present in the scan text next to the title.
Inspected two examples directly:

- `9783847432364.pdf` (German, clean extracted text): titles/authors
  match ground truth exactly; scan text clearly shows `"...das neue
  Gemeinsame   7"`, `"...gekämpft wird   15"` right next to each title,
  yet every predicted `printed_page_number` is `null`.
- `9780367439712.pdf` (English, but a badly OCR-scrambled multi-column
  contents page -- `"l"` for `"1"`, garbled column bleed): same pattern,
  titles correct, page numbers all `null`.

Since `match_toc_entries` requires an exact page-number match to count a
true positive, these 14 books score exactly 0 recall despite mostly-
correct extraction -- title-only accuracy is substantially better than
the f1=0.39 headline suggests. Two candidate sub-causes, not yet
disentangled: a likely non-English weakness (echoes the French-language
miss documented earlier for NuExtract3) and OCR-scrambled TOC layouts.
Unlike the truncation cluster or the earlier `transformers`/MPS bug, this
looks like a fixable extraction-formatting habit rather than a
fundamental capability gap -- the model already has the right
information, it just isn't attaching it to the record -- which is a
reasonable target for LoRA fine-tuning to move.

**This f1=0.39 is the baseline number a fine-tuning pilot needs to beat.**

### Next: fine-tuning feasibility

Before investing in a bigger ground-truth set, the plan is to check
whether LoRA fine-tuning actually moves this number, using a held-out
split of the existing 50-book corpus as a cheap pilot rather than
committing to more ground-truth curation first. Design and implementation
plan: `docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`.

### Output-token-limit retest (2026-08-10)

Before the fine-tuning pilot above, cleared the cheaper explanation
first: the failure-mode breakdown's truncation cluster (10 books, 20%)
used `max_tokens=1500`, and several of those books' generation times
(200-500s) are consistent with hitting that cap rather than reaching a
natural stop. Re-ran the full 50-book corpus with `max_tokens=6000`
(everything else identical: `llama.cpp`, Metal-offloaded, `n_ctx=40960`)
to see how much of the f1=0.39 baseline was an artifact of an
under-provisioned output budget rather than a genuine capability gap.

| Corpus | precision | recall | F1 (1500 tok) | F1 (6000 tok) |
| --- | --- | --- | --- | --- |
| copyrighted-scans | 0.57 | 0.42 | 0.17 | **0.48** |
| open-access | 0.39 | 0.46 | 0.47 | 0.42 |
| **Total** | **0.43** | **0.45** | **0.39** | **0.44** |

**A real but partial fix: f1=0.39 -> f1=0.44, driven by 2 of the 10
originally-truncated books recovering completely, not by the cluster
closing.** Tracing all 10 originally-flagged books individually:

- **2 books recovered to strong scores**, confirming the budget really
  was the cause for these: `9783428042241.pdf` (0/0 found -> 38/41
  found, f1 0.00 -> 0.94) and `9783899496291.pdf` (0/0 -> 53/58 found,
  f1 0.00 -> 0.91). Both are large Festschrift-style
  `copyrighted-scans` books with big TOCs -- exactly the shape of book
  the 1500-token cap was too small for -- and both now score
  essentially as well as the corpus's best books, which is why
  `copyrighted-scans`' aggregate f1 nearly tripled (0.17 -> 0.48).
- **4 books still hit the new, 4x-larger cap and still produce zero**:
  `dnb-36942798X.pdf`, `9783839458013.pdf`, `9781783742806.pdf`,
  `9781783743339.pdf` (all logged `[HIT_MAX_TOKENS]`, taking 220-630s
  each). These have TOCs large/complex enough that even 6000 tokens
  isn't enough, or the model enters a repetition loop that never
  reaches valid JSON regardless of budget -- not distinguished further
  here.
- **1 book (`9783848704316.pdf`) took 979s and still produced 0/0
  without hitting the cap** -- it stopped generating on its own before
  6000 tokens, just never produced valid/matching JSON. A different
  failure shape than budget exhaustion.
- **3 books stopped truncating but now produce wrong content instead of
  no content** -- `9782375460122.pdf` (0/0 -> 0/78 found, still 0 true
  positives -- this is the already-documented French-language/
  cataloging-page miss, see the NuExtract3 section above, not a
  truncation artifact at all), `9783839446270.pdf` (0/0 -> 0/0 found in
  29.5s, a fast empty result, not budget-related), and
  `9783839465776.pdf` (0/0 -> 0/59 found, still 0 true positives). Where
  more output budget just means more room to generate spurious entries,
  it doesn't help.

So raising the token budget is a real, worthwhile, free fix -- worth
keeping in whatever configuration the fine-tuning pilot's evaluation
script uses (`evaluate_nuextract_finetune.py` already defaults to
`--max-tokens 6000`) -- but it only fully resolved 2 of 10 originally-
truncated books; the rest were already, or became once budget was no
longer the bottleneck, cases of the model producing wrong or repetitive
output rather than running out of room. **f1=0.44 (this retest), not
f1=0.39, is the correct "baseline to beat" for the fine-tuning pilot**
per its design spec's decision criteria.

**Follow-up: raising the cap further (12000 tokens) does not rescue the
remaining 4 `[HIT_MAX_TOKENS]` books either.** Retried just those four
(the only ones where more budget could plausibly still be the
bottleneck -- the other zero-scoring books were already ruled out above
as language/content misses, not budget) at `max_tokens=12000`, same
`llama.cpp`/Metal/`n_ctx=40960` setup:

| Book | 6000 tok | 12000 tok |
| --- | --- | --- |
| `dnb-36942798X.pdf` | 288s, `[HIT_MAX_TOKENS]`, 0/0 | 677s, stopped on its own, 0/0 |
| `9783839458013.pdf` | 223s, `[HIT_MAX_TOKENS]`, 0/0 | 545s, `[HIT_MAX_TOKENS]`, 0/0 |
| `9781783742806.pdf` | 419s, `[HIT_MAX_TOKENS]`, 0/0 | 860s, `[HIT_MAX_TOKENS]`, 0/0 |
| `9781783743339.pdf` | 628s, `[HIT_MAX_TOKENS]`, 0/0 | 888s, `[HIT_MAX_TOKENS]`, 0/0 |

All four still score 0/0. Three still hit the (now doubled) cap, and the
fourth ran even longer (677s, up from 288s) before finally stopping on
its own -- still with no valid output. Doubling the budget roughly
doubled the wall-clock time these four burn without moving their score
at all, which is the signature of a genuine repetition/malformed-
generation failure, not a legitimate long-TOC book that just needs more
room. **Not worth raising the cap further** -- these four need a
different fix entirely (e.g. repetition-penalty sampling, detecting and
truncating a repeating n-gram mid-generation, or accepting them as a
known zero-recall cluster) rather than a bigger `max_tokens`. `f1=0.44`
at `max_tokens=6000` stands as the baseline; raising it beyond 6000 buys
nothing further on this corpus and should not be adopted as the
production/evaluation default.
