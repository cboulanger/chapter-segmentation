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
> `copyrighted-scans` corpora respectively.

**2026-08-12 re-run, full expanded corpus.** `open-access/` has grown from
6/37 books (depending which earlier snapshot) to **57**, and
`copyrighted-scans/` from 11 to **13**, following the crossref_gt
reconciliation, the two hand-built Festschrift volumes, and a pass of
hand-verification corrections to previously-wrong chapter boundaries (see
git log around 2026-08-11/12: `0d5ec5a`, `1e8ee92`, `69b9662`). This
snapshot re-runs every *cheap* (no per-request API cost) evaluation --
the pure-heuristic pytest harness, the heuristic/outline per-strategy
report, and the heuristic+outline+Crossref strategy-pipeline script --
against the full grown corpus. The layout-based TOC classifier pilot was
also re-run 2026-08-12 after its `pdfalto` extraction pass over the new
books -- see "Follow-up: re-run over the grown 70-book corpus" (and the
context-features follow-up after it) in the pilot's own section below. The
LLM strategy was **not** re-run (it costs real KISSKI budget) -- its
numbers below are carried over from before the corpus grew and are called
out as stale where they appear.

## Pure-heuristic results

From `uv run pytest tests/test_segmentation_accuracy.py -q -s -m integration`
(`chapter_segmentation.analyze_attachment`), run 2026-08-12 against the full
70-book corpus (57 `open-access/` + 13 `copyrighted-scans/`). Full per-book
detail lives in the published report
(https://cboulanger.github.io/chapter-segmentation/); this is the aggregate:

| Corpus | Books | Precision | Recall | F1 | Found / Expected |
| --- | --- | --- | --- | --- | --- |
| `open-access/` | 57 | 0.46 | 0.51 | 0.48 | 550/1199 found, 550/1074 expected |
| `copyrighted-scans/` | 13 | 0.49 | 0.32 | 0.39 | 99/201 found, 99/307 expected |
| **combined** | **70** | **0.46** | **0.47** | **0.47** | **649/1400 found, 649/1381 expected** |

This is a large drop from the previous 7-book snapshot's 0.91/0.91 -- **not
a code regression**: those original 7 books are still in the corpus and
still score exactly as before (see them called out in the mechanism/misses
notes below), but they're now a small, easy fraction of a much larger and
harder set. The bulk of the new `open-access/` growth (31 books reconciled
from `evaluation/crossref_gt/`) is technical/textbook content where numbered
subsections, figures, and glossary entries get mistaken for chapter titles
-- a real, already-partially-documented heuristic gap (see "Per-strategy
standalone results" below for how much of it the outline/Crossref
strategies recover instead). The pytest run itself now **fails 6 subtests**
(`recall > 0` regression guard) -- see "New zero-recall findings" below for
what each one is and why.

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

### New zero-recall findings from the 2026-08-12 corpus growth

Six books now fail the pytest harness's `recall > 0` regression guard
(none carry `heuristic_expected_zero: true`). Traced by hand rather than
guessed:

- **`open-access/9781800642003.pdf`, `9781805114307.pdf`,
  `9781805115717.pdf`, `9782821895607.pdf`** (0/17, 0/14, 0/57, 0/12
  expected found): all four are `crossref_gt`-migrated technical/textbook
  books -- the same "numbered subsections/figures mistaken for chapters"
  gap already documented for 7 sibling books in this same migration batch
  (which *are* flagged `heuristic_expected_zero: true`; these four simply
  weren't flagged when that batch was reconciled). `analyze_attachment`
  finds real chapter candidates for all four (1-15 of them), just none at
  an exactly-correct boundary -- **and all four recover to 0.58-1.00
  recall under the strategy pipeline** (outline and/or Crossref signal,
  see "Strategy-pipeline results" below), confirming this is the same
  known heuristic-only gap, not a new failure mode. Worth flagging
  `heuristic_expected_zero: true` on these four to match their siblings
  and keep the pytest suite green -- not done here since that's a
  manifest-editing judgment call outside "run the evaluation," not a
  reporting one.
- **`copyrighted-scans/9781409403906.pdf`** (0/12 expected found): this
  book's ground truth was hand-corrected on 2026-08-12 (commit `1e8ee92`)
  -- every chapter after the Introduction had `pdf_start_index`/
  `pdf_end_index` off by 1-3 pages in the old data. The *previously
  reported* 0.08 recall (1/12, in the old "Diverse real-library" table
  below) was true only against that wrong ground truth; the heuristic's
  actual output never changed, and against the corrected boundaries it
  matches zero of them. This is the GT correction **revealing** a
  pre-existing heuristic gap that had been masked by a consistent
  ground-truth offset, not a new regression -- not yet root-caused
  further (has no Crossref/outline signal either, per "Strategy-pipeline
  results" below, so it stays at 0 there too).
- **`copyrighted-scans/9783465016878.pdf`** (0/0 found -- the heuristic
  detects zero candidate chapters at all, not just zero correct ones):
  unlike the book above, this one's ground truth was *not* touched by the
  recent corrections. `find_toc_candidates` returns no TOC cluster at all;
  inspecting the OCR'd front matter directly shows why -- the printed TOC's
  dot-leader page numbers OCR onto their own separate line, detached from
  the title line they belong to (page 4: five titles, then five bare
  numbers `201 / 227 / 253 / 273 / 283` on trailing lines with no
  "title ... number" line ever formed). This is the same class of gap
  already documented for `9783789057366.pdf` below (dense dot-leader OCR
  garbling), not yet root-caused to a specific fix. Previously reported at
  0.15 recall (2/13) -- not yet explained why that used to work; possibly
  a heuristic change in the intervening commits tightened the TOC-cluster
  requirements enough to lose this book's weaker candidate set (not
  confirmed by bisecting).

`chapter_upload.py`'s `confidence_threshold` default (`0.90`) was
calibrated against the old 7-book/~91%-correct snapshot above -- **now
stale** given the much lower correctness rate on the full 70-book corpus.
Re-run the calibration sweep (or re-derive it against `analyze_attachment`'s
output on the current corpus) before trusting that default again; not done
as part of this evaluation re-run.

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

**Update, 2026-08-12:** `9783428042241.pdf` now has hand-verified ground
truth (`evaluation/corpus/copyrighted-scans/9783428042241.expected.json`,
built via the Crossref-page-range shortcut per `CLAUDE.md`) and is scored
in the tables above: `precision=0.40 recall=0.40` (16/40 found, 16/40
expected) under the pure heuristic, recovering to `precision=0.86
recall=0.76` (44/51 found, 44/58 expected -- via `crossref`) under the
strategy pipeline (see below). The unscored 41-entries/40-chapters figure
above was this fix's own before/after check at the time, not a
precision/recall number against real ground truth; the book's actual
measured accuracy is lower than that figure implied; the fix itself (the
`_looks_like_imprint_line` filter) is unaffected -- without it this book
would still be at near-zero detected chapters, not the 16/40 it gets now.

## Strategy-pipeline results

From `uv run python evaluation/scripts/evaluate_chapter_segmentation_strategies.py`
(`analyze_attachment_with_strategies`), re-run 2026-08-12 against the full
70-book corpus. Full per-book detail (including each book's
`strategies_used`) is in the raw run log, not reproduced here given its
size -- this is the aggregate, alongside the pure-heuristic numbers from
above for direct comparison:

| Corpus | Books | Strategy pipeline P / R / F1 | Found / Expected | Pure heuristic P / R / F1 (from above) |
| --- | --- | --- | --- | --- |
| `open-access/` | 57 | 0.78 / 0.85 / 0.81 | 908/1157 found, 908/1074 expected | 0.46 / 0.51 / 0.48 |
| `copyrighted-scans/` | 13 | 0.58 / 0.43 / 0.49 | 131/225 found, 131/307 expected | 0.49 / 0.32 / 0.39 |
| **combined** | **70** | **0.75 / 0.75 / 0.75** | **1039/1382 found, 1039/1381 expected** | **0.46 / 0.47 / 0.47** |

**This reverses the previous snapshot's headline finding.** On the old
7-book set, the strategy pipeline was a net *regression* against the pure
heuristic (0.86/0.89 vs. 0.91/0.91) -- on the full, much larger and harder
corpus, it's a clear net *improvement*, especially on `open-access/` (0.81
vs. 0.48 F1). The mechanism is exactly what the pure-heuristic section
above already names as this corpus's dominant new gap: most of the 31
`crossref_gt`-migrated books are technical/textbook content where the
heuristic's TOC-line regex gets confused by numbered subsections, figures,
and glossary entries, but where a real Crossref book-chapter record or PDF
outline (see "Per-strategy standalone results" below -- 41 of the 57
`open-access/` books now carry a real outline, up from 3 of 17 previously)
gives the pipeline a resolved, book-order-correct chapter list to localize
against instead of ever needing that fragile regex. `copyrighted-scans/`
improves more modestly (0.49 vs. 0.39 F1) because only 3 of its 13 books
have any Crossref/outline signal at all (`9783322969828.pdf`,
`9783428042241.pdf`, `9783899496291.pdf` -- all DOI-backed; the other 10
have neither a DOI nor a PDF outline by construction, see `README.md`'s
"Evaluation set composition") -- for the other 10, `strategies_used` is
always `[]` and the pipeline falls straight back to the pure-heuristic
result, unchanged.

`analyze_attachment_with_strategies` only falls back to the pure-heuristic
pipeline when a book's merged candidate list is completely empty, never
when it's merely wrong -- so a confidently-incomplete or
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

All four of the pure-heuristic section's new unflagged zero-recall
`open-access/` books (`9781800642003.pdf`, `9781805114307.pdf`,
`9781805115717.pdf`, `9782821895607.pdf`) recover to non-trivial recall
here (1.00, 0.93, 0.88, 0.58 respectively, via outline and/or crossref) --
further evidence they're the known heuristic-only textbook gap, not a
missing-signal problem. The two `copyrighted-scans/` zero-recall books
(`9781409403906.pdf`, `9783465016878.pdf`) stay at zero here too, since
neither has any Crossref/outline signal to fall back on.

**Given the corpus-wide reversal above, `analyze_book_chapters.py`/the
`/api/analyze` endpoint routing through the strategy pipeline is now
net-*positive* on `open-access/`-shaped books (a real DOI or PDF outline,
a well-produced printed TOC), though still only par-to-modestly-ahead of
the pure heuristic on `copyrighted-scans/`-shaped books (no DOI, no
outline, sometimes a scan)** -- re-run this evaluation after any further
change to the outline/Crossref/fusion logic, and treat this as a
reversal of the previous snapshot's "should not be pointed at" advice
for the `open-access/`-shaped case specifically, not a blanket "safe for
production everywhere" clearance.

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
| `9783492021234.pdf` | 0.41 | 0.50 | 7/17, 7/14 | layout-mode fallback |
| `9783789016202.pdf` | 0.50 | 0.58 | 7/14, 7/12 | layout-mode fallback |
| `9783899718188.pdf` | 0.27 | 0.30 | 3/11, 3/10 | layout-mode fallback |
| `9780367439712.pdf` | 0.31 | 0.42 | 5/16, 5/12 | OCR (degenerate text layer) |
| `9783789057366.pdf` | 0.09 | 0.02 | 1/11, 1/56 | OCR (degenerate text layer) -- was "still 0," now a sliver of signal (not re-investigated) |
| `9783465016878.pdf` | 0.00 | 0.00 | 0/0, 0/13 | OCR (no text layer) -- **new regression, was 0.15** |
| `9781409403906.pdf` | 0.00 | 0.00 | 0/10, 0/12 | OCR (no text layer) -- **GT-correction-revealed gap, was 0.08** |
| `9783848704316.pdf` | 0.25 | 0.07 | 1/4, 1/15 | OCR (no text layer) -- GT-correction recovered a sliver of signal |
| `dnb-36942798X.pdf` | 0.00 | 0.00 | 0/2, 0/18 | OCR (no text layer) -- still 0 |

See "New zero-recall findings from the 2026-08-12 corpus growth" above for
the two bolded rows' root causes. Aggregate (micro, these 10 books alone):
**precision 0.44, recall 0.25** (47/108 found, 47/185 expected chapters) --
essentially flat versus the previous snapshot's 0.47/0.24 (47 vs. 44
correct matches, same 185 expected chapters), but the underlying mix
shifted: two books that used to show partial recall now show none (see
above), offset by three books recovering more signal (`9783492021234.pdf`,
`9783789016202.pdf`, `9783848704316.pdf`) and one moving off an exact 0.00
(`9783789057366.pdf`). These are the same numbers whether run through the
pure heuristic (`analyze_attachment`, `test_segmentation_accuracy.py`) or
the full strategy pipeline (`analyze_attachment_with_strategies`,
`strategies_used: []` on all 10, unchanged from before) -- none of the 10
has a Crossref record or a usable PDF outline, so the strategy pipeline
always falls straight back to the same heuristic result on this set; the
strategies genuinely add nothing here, for better or worse.

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

**The 3 books originally reported at 0.00 recall after OCR, with actual
root causes** (traced by hand -- inspecting the cached OCR text and
`find_toc_candidates`' raw output directly, not guessed). As of this
2026-08-12 snapshot, 2 of these 3 (`9783789057366.pdf`, `9783848704316.pdf`)
score a small nonzero recall (see the table above) -- not re-investigated
to confirm whether the original root cause below still fully applies, only
that it's no longer literally zero:

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
guard as the rest of the evaluation set. Two of those seven
(`9781409403906.pdf`, `9783465016878.pdf`) now fail that guard as of this
snapshot -- see "New zero-recall findings from the 2026-08-12 corpus
growth" above; they are not flagged here since, unlike the three above,
their zero-recall status has not been confirmed as unrecoverable through
every available extraction/OCR path, only observed in this run.

## Per-strategy standalone results (heuristic / outline / LLM)

From `uv run python evaluation/generate_report.py --out public/` (heuristic,
outline) plus `uv run python evaluation/refresh_llm_cache.py --mode full`
populating each corpus's `evaluation/corpus/<name>/llm-cache/` (LLM) -- each
strategy run independently via `analyze_attachment`,
`analyze_attachment_outline_only`, `analyze_attachment_llm_only`, with no
pipeline merge/fallback logic involved (see
`docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md`).
Live numbers (always current, no hand-written commentary):
https://cboulanger.github.io/chapter-segmentation/;
full per-model breakdown at its `llm/index.html`. See `README.md`'s
"Per-strategy evaluation report" / "LLM strategy evaluation" for how to
reproduce.

**Heuristic and outline rows below are freshly re-run (2026-08-12) against
the full 70-book corpus** (up from 17 in the snapshot this table originally
reported). **The LLM row is carried over unchanged from before the corpus
grew** -- `evaluation/refresh_llm_cache.py` costs real KISSKI API budget, so
it was not re-run as part of this "cheap strategies only" pass; its
"applicable books" denominator below still reflects the old, much smaller
corpus and should not be compared directly against the new heuristic/outline
denominators. `public-cache/` was also regenerated for this pass to cover
the 25 `open-access/` books it was still missing (added by the corpus growth
but never cached) -- **two `copyrighted-scans/` books
(`9780367439712.pdf`, `9781409403906.pdf`) failed the regeneration's
`--verify` self-consistency check and kept their stale (2026-08-08) cached
text**; their heuristic/outline numbers in the per-book report
(`public/copyrighted-scans/index.html`) may not match the authoritative
pytest/strategy-pipeline numbers above, which read the real PDF directly
rather than `public-cache/`. Concretely: the report shows
`9781409403906.pdf` heuristic at precision 0.40/recall 0.33 (4/12), while
the direct-PDF pytest run above shows 0.00/0.00 (0/12) -- **trust the
direct-PDF numbers** for these two books specifically; not yet fixed
(the redaction pipeline's own self-consistency check catching a real
divergence, per `CLAUDE.md`'s "Document organization" section on this
exact failure mode).

"Start accuracy"/"End accuracy" (added by
`docs/superpowers/plans/2026-08-08-citation-pages-mapping.md`, see that
section below) score `citation_pages` -- the printed-page-number metadata
attached to each chapter -- restricted to chapters whose located PDF page
range exactly matches ground truth AND whose expected `citation_pages` is
non-null. Start requires an exact match (unmappable counts as wrong); end
tolerates being up to 3 printed pages over-inclusive (see
`evaluation/metrics.py`'s `citation_pages_metrics`).

| Corpus | Strategy | Precision | Recall | F1 | Found / Expected | Total time | Start accuracy | End accuracy | Applicable books |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| open-access | outline | 0.78 | 0.71 | 0.75 | 556/712 found, 556/780 expected | 14.1s | 0.75 | 0.53 | 41/57 |
| open-access | heuristic | 0.46 | 0.51 | 0.48 | 550/1199 found, 550/1074 expected | 150.3s | 0.97 | 0.98 | 57/57 |
| open-access | LLM (`qwen3-coder-next`, stale/partial) | 0.85 | 0.69 | 0.76 | 57/67 found, 57/83 expected | 97.6s | 0.96 | 0.96 | ~8/57 |
| copyrighted-scans | heuristic | 0.52 | 0.34 | 0.41 | 103/199 found, 103/307 expected | 46.8s | 1.00 | 1.00 | 13/13 |
| copyrighted-scans | outline | 0.40 | 0.42 | 0.41 | 17/42 found, 17/40 expected | 0.8s | 0.00 | 0.00 | 1/13 |
| copyrighted-scans | LLM (`glm-4.7`, stale) | 0.54 | 0.27 | 0.36 | 67/125 found, 67/249 expected | 580.7s | 1.00 | 1.00 | 13/13 |

(The `copyrighted-scans` heuristic row above includes the two stale-cache
books' inflated numbers per the caveat above -- the authoritative
direct-PDF pure-heuristic aggregate for this corpus, from the
"Pure-heuristic results" section above, is precision 0.49/recall 0.32,
99/201 found, 99/307 expected.)

- **Outline no longer applies narrowly on `open-access/`.** The previous
  snapshot found only 3 of 17 corpus books carrying a real embedded PDF
  outline/bookmark catalog; that was an artifact of the small, curated
  original set -- across the grown 57-book `open-access/` corpus, **41 of
  57** now carry one (many of the `crossref_gt`-migrated books turn out to
  have real outlines), which is exactly why the outline strategy now
  dominates the strategy pipeline's `open-access/` improvement documented
  above. `copyrighted-scans/` stays narrow (1 of 13) -- consistent with that
  corpus's "no DOI, mostly no embedded TOC" sourcing criterion. An outline
  entry is already a resolved, book-order-correct chapter reference, so
  once one exists it's nearly free signal; books with none render `N/A` in
  the report rather than being scored as "found 0".
- **Heuristic's F1 0.48/0.41 aggregates are now dominated by the *whole*
  corpus, not pulled down by a small "diverse" sub-sample.** The previous
  snapshot's framing ("0.58 pulled down entirely by 10 diverse-real-library
  books, restrict to the original 7 and it's 0.91/0.91") no longer holds
  as a useful decomposition -- the original 7 `open-access/` books are now
  a small fraction of 57, and most of the new 0.46/0.51 aggregate's misses
  are the `crossref_gt` technical/textbook gap described in
  "Pure-heuristic results" above, not the OCR/scan issues the old framing
  centered on.
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
  F1 0.42 overall (95/158 found), no longer zero. This bullet and the ones
  below describe the LLM-only cache-refresh investigation on the
  pre-corpus-growth 17-book set (the LLM row wasn't re-run for this
  snapshot, see the caveat above) -- LLM standalone trailed the
  then-heuristic's 0.58 aggregate and cost far more time, so this remains
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
  (pre-growth, 17-book) corpus at the time (restricted to chapters whose
  located PDF range exactly matches ground truth and whose expected
  `citation_pages` is non-null, via `citation_pages_metrics`): **heuristic
  scores 142/144 exact start matches (98.6%) and 143/144 end matches within
  tolerance (99.3%)**; this 2026-08-12 re-run's fresh start/end-accuracy
  *fractions* (raw counts not extracted) are in the table above --
  0.97/0.98 for `open-access/` heuristic and 1.00/1.00 for
  `copyrighted-scans/` heuristic, both still consistent with "the fix
  works," just measured on a much larger corpus now;
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

The subsections from here through "Follow-up: relaxing the per-book bar"
describe the pilot's history on the pre-growth **50-book** corpus -- every
book count and percentage in them refers to that corpus size. The
2026-08-12 re-run over the grown 70-book corpus, and the feature/
augmentation work that followed it, are covered in the last two follow-up
subsections at the end of this section.

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

### Follow-up: re-run over the grown 70-book corpus

Re-run 2026-08-12 against the full grown corpus (57 `open-access/` + 13
`copyrighted-scans/`, up from 37 + 13), unchanged model/calibration
(`LogisticRegression`, `recall_target=0.80`, `chapter_first_recall_tolerance=0.90`).
This needed the `pdfalto` binary, which isn't packaged or brew-installable
-- it was built from a sibling checkout of
[kermitt2/pdfalto](https://github.com/kermitt2/pdfalto) at `../pdfalto/`
next to this repo (`evaluation/CLAUDE.md` now documents this location).
32 of the 57 `open-access/` books already had a cached ALTO XML from the
prior run; the remaining 25 new `open-access/` books plus all 13
`copyrighted-scans/` books (whose `.layout-cache/` had never been
populated in this worktree) were freshly extracted, ~2.5 minutes total.

| | full_recall_fraction | avg_candidate_fraction |
| --- | --- | --- |
| 50-book corpus (previous) | 44% | 15.0% |
| **70-book corpus (fresh)** | **64%** | **10.0%** |

Still **NOT MET** against the 90%/15% decision bar, but the largest single
jump in `full_recall_fraction` across this pilot's whole history --
achieved purely from more/corrected ground truth, no model or feature
changes -- while `avg_candidate_fraction` actually *dropped* well clear of
its budget rather than trading one for the other. Per corpus:

| corpus | books | full_recall_fraction (before -> after) | avg_candidate_fraction (before -> after) |
| --- | --- | --- | --- |
| open-access | 37 -> 57 | 54.1% -> **77.2%** | 15.2% -> **10.7%** |
| copyrighted-scans | 13 -> 13 | 15.4% -> **7.7%** | 14.4% -> **6.9%** |

**open-access drove the whole improvement.** Of the 57 books, 44 now hit
the full-recall bar and 38 score a literal 100%/100% toc/chapter_first
recall -- most of the 20 net-new books (the crossref_gt reconciliation
batch) are exactly this kind of clean, well-produced text, and the
hand-verification pass that corrected previously-wrong chapter boundaries
(see the "2026-08-12 re-run" note at the top of this file) means the
labels those 44 books are scored against are also more trustworthy than
before. The open-access stragglers are a recognizable, already-diagnosed
shape rather than something new: low `toc_recall` on TOC pages that don't
structurally resemble the training majority (e.g. `9781783748471` at 0%,
`9782375460122` at 0%) and low `chapter_first_recall` on books with weak
title/body font-size contrast (e.g. `9783492021234` at 43%,
`9783907297339` at 45%) -- both failure modes already named in the
model-architecture follow-up above.

**copyrighted-scans regressed slightly despite an unchanged book set** --
the same 13 books, none added or removed, only 2/13 pass now vs. 2/13
before by fraction (15.4% was already ~2/13; 7.7% is 1/13). This isn't a
data change on this corpus's side; the LOBO classifier trained on each
fold now draws from a much larger, open-access-dominated training pool
(56 other books instead of 36), which shifts the decision boundary these
13 harder, higher-page-width-variance books are scored against. The two
per-book swings worth flagging rather than averaging away:
`9783848704316` dropped from 73% to 27% `chapter_first_recall` (was the
biggest post-OCR-fix win in the earlier follow-up, now the biggest
regression), while `dnb-36942798X` improved from 6% to 17%. Both remain
well short of passing either way. The two newest `copyrighted-scans`
books (the hand-built Festschrift volumes, `9783428042241` and
`9783899496291`) are themselves weak: 5% and 24% `chapter_first_recall`
respectively -- consistent with this corpus's standing diagnosis (the
"Follow-up: replacing textless/degenerate-text..." section above) that
`copyrighted-scans` scans have systematically weaker layout signal than
`open-access` books, not a new finding.

Net assessment of the re-run itself: corpus growth plus ground-truth
correction bought real, substantial progress on `open-access` (54% to 77%)
at essentially no `copyrighted-scans` cost (a ~2-book swing among 13 that
were already the weak half of the corpus) -- but the pilot's own 90% bar
was still not met, and the stark per-corpus split above (`open-access`
close to the bar, `copyrighted-scans` nearly the whole shortfall) is
exactly what motivated the next follow-up's two directions (book-context/
normalized features and scan-noise augmentation), which measures against
this run as its baseline.

### Follow-up: context/normalized features and scan-noise augmentation

Per `docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md`,
direct inspection of `copyrighted-scans` ALTO XML found the cause of the
split above: OCR'd scans carry hundreds of near-identical jittery font
sizes with no genuine large title font, so the two strongest existing
features (`font_size_max_ratio`, `top_block_is_large_font`) are degenerate
on exactly the failing corpus, and the per-page `statistics.mode` used as
the body-font estimate is itself noise over that many close values. The
fix has two independent parts, both on top of the unchanged
`LogisticRegression` model and unchanged calibration
(`recall_target=0.80`, `chapter_first_recall_tolerance=0.90`): seven new
features in `evaluation/scripts/layout_features.py` --
two page-local (`last_text_vpos_fraction`, `top_line_heading_match`) and
five book/context features computed in a second pass,
`add_book_context_features()` (`prev_last_text_vpos_fraction`,
`prev_line_count_rel`, `line_count_rel`, `font_size_max_ratio_book`,
`page_position_fraction`) -- taking `FEATURE_NAMES` from 10 to 17; and a
new deterministic ALTO-level scan-noise augmentation module,
`evaluation/scripts/alto_scan_noise.py`, wired in via a `--scan-noise-augment`
flag that perturbs only `open-access` training rows (font-size jitter,
title-contrast compression, geometry jitter) so the born-digital pool
looks scan-shaped during training, with augmented rows excluded from every
fold's own test set to avoid leakage.

The three official LOBO runs, all over the full 70-book corpus at the
unchanged calibration:

| run | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| baseline (10 features) | 45/70 = 64% | 10.0% |
| + context/normalized features (17) | 39/70 = 56% | 7.2% |
| + `--scan-noise-augment` | 40/70 = 57% | 7.2% |

Per corpus, all three runs:

| run | open-access `full_recall_fraction` | open-access `avg_candidate_fraction` | open-access avg `chapter_first_recall` | scans `full_recall_fraction` | scans `avg_candidate_fraction` | scans avg `chapter_first_recall` |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 44/57 = 77.2% | 10.7% | 93.9% | 1/13 = 7.7% | 6.9% | 28.8% |
| + features | 39/57 = 68.4% | 7.5% | 85.8% | 0/13 = 0% | 5.6% | 50.9% |
| + augment | 40/57 = 70.2% | 7.4% | 86.8% | 0/13 = 0% | 6.1% | 51.2% |

**Read at face value, this looks like a regression -- it isn't; it's an
operating-point artifact of the unchanged `recall_target=0.80`.** The new
features make the training probabilities more separable, so the per-fold
threshold calibrated to 80% training recall becomes noticeably more
conservative: candidate fraction fell from 10.0% to 7.2%, tight enough
that five `open-access` books flip from passing to failing purely on
`chapter_first_recall` dropping below the 90% tolerance --
`9781783748532` (100%->87%), `9781787359260` (95%->60%),
`9782821895607` (92%->42%, whose `toc_recall` also drops 100%->0%),
`9783031907272` (100%->32%), and `9783837681192` (100%->46%) -- with zero
books flipping the other way. An informational sweep of `recall_target`
(not a change to the default; run only to test whether the drop is purely
an operating-point effect) confirms it is:

| config | `full_recall_fraction` | `avg_candidate_fraction` | open-access `full_recall_fraction` | scans `full_recall_fraction` | scans avg `chapter_first_recall` |
| --- | --- | --- | --- | --- | --- |
| 17 features, rt=0.85 | 41/70 = 59% | 7.9% | 70.2% | 1/13 | 57.8% |
| 17 features, rt=0.90 | 47/70 = 67% | 9.0% | 78.9% | 2/13 | 66.8% |
| 17 features + augment, rt=0.90 | 45/70 = 64% | 8.9% | 77.2% | 1/13 | 64.8% |

At `rt=0.90` -- still comfortably inside the 15% candidate budget at 9.0%
-- the same 17 features beat the baseline on every axis at once: 67% vs.
64% full recall, 9.0% vs. 10.0% candidates, 78.9% vs. 77.2% open-access,
2/13 vs. 1/13 scans. This sweep does **not** change the default --
`recall_target` stays `0.80` so this run remains comparable to every prior
follow-up's numbers, per the earlier model-architecture follow-up's
framing above ("`recall_target` is the real ... dial a consumer of the
classifier would actually set"; `chapter_first_recall_tolerance` only
changes how this script scores a book, not inference behavior) -- it's
reported here only to show that the curve genuinely moved, not just the
single point the default happens to sit at.

**The scan rescue is real at the recall level, even though it doesn't
flip any scan to "passing" at the default tolerance.** Scans' average
`chapter_first_recall` rose from 28.8% at baseline to 50.9% (17 features,
`rt=0.80`) and 66.8% (`rt=0.90`) -- the largest per-book jumps (baseline
-> +features) are `9783428042241` (5%->68%, 78% with augmentation),
`9780367439712` (8%->67%), `9783465016878` (15%->77%),
`9783848704316` (27%->73%), `9783789057366` (2%->39%), `9783899496291`
(24%->62%), and `9783322969828` (0%->29%). But **no scan clears the
90%-per-book bar at `rt=0.80`** (0/13, and the baseline's one passer,
`9783848736829`, itself drops from 100% to 61%) -- which is exactly why
the corpus-level `full_recall_fraction` for scans reads 0% despite the
underlying recall gains being large and broad-based.

**The six 0%-`toc_recall` open-access books are unchanged at 0%**:
`9781783748471`, `9782375460122`, `9783837660944`, `9783839447529`,
`9783839468937`, and `9783839470619` all still score zero `toc_recall`
under every configuration above. Expected -- the new features target
chapter-opening detection (book-context, per-book font normalization,
heading-line text), not TOC-page layout, so they have no mechanism to
touch this failure mode. TOC-anchored matching (deferred, see below)
remains the untouched direction for these six.

**Augmentation itself is an honest, mostly-negative result.** Against the
un-augmented 17-feature run, `--scan-noise-augment` moves the overall
number by +1 book at `rt=0.80` (56%->57%) and -2 books at `rt=0.90`
(67%->64%), and scans' average `chapter_first_recall` by only +0.3pt at
`rt=0.80` (50.9%->51.2%). A few scans improve slightly with it
(`9783428042241` 68%->78%, `9783492021234` 43%->50%); others drop
(`9780367439712` 67%->58%, `9783465016878` 77%->69%, and
`9783848736829`'s `toc_recall` 67%->33%). The most likely reason: the
per-book font normalization (`font_size_max_ratio_book`) already closes
most of the domain gap the augmentation was designed to target, leaving
little for the synthetic scan-noise to add on top. It stays in the
codebase -- cheap to run, cached (`.aug.alto.xml`, regenerated only when
deleted by hand), off by default -- but should not be relied on as the
mechanism behind the scan-recall gains above; the feature work is.

**Verdict: still NOT MET, in every configuration measured**, against the
unchanged 90%/15% bar. The best configuration found is 17 features at
`recall_target=0.90` (67% / 9.0%), still 23 points short. The features
moved the curve in the right direction on both corpora simultaneously
without touching the candidate-fraction budget -- a real, if partial,
result -- but two structural gaps remain untouched by this follow-up: the
open-access TOC-layout failure mode (candidate direction: TOC-anchored
chapter matching, parsing a detected TOC page and locating chapter
openings by title/page-number matching against it) and the residual scan
`chapter_first_recall` ceiling (candidate direction: document-image deep
learning, bypassing OCR font metadata entirely). Both were deferred, not
rejected, at the design stage; neither is scoped here. With candidate
volume now sitting well below the 15% budget even at `rt=0.90`, revisiting
the default `recall_target` itself is also worth a future look.

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
