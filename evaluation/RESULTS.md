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
>
> **Full experiment history:** `EXPERIMENTS.md` in this directory holds the
> complete, unabridged write-up for every superseded snapshot and follow-up
> investigation that is mentioned below only in summary form. Follow the
> "See EXPERIMENTS.md § ..." links in the sections below for the full
> numbers, tables, and root-cause detail behind a compressed summary.

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
context-features follow-up after it) under "History" in the pilot's own
section below, or the full write-up in EXPERIMENTS.md. The
LLM strategy was **not** re-run (it costs real KISSKI budget) -- its
numbers below are carried over from before the corpus grew and are called
out as stale where they appear.

**2026-08-13 update.** `copyrighted-scans/` has grown again, from 13 to
**32**, via a targeted acquisition pass following the "prefer scans,
unnumbered-first-chapter books, weak title/body contrast" guidance in this
directory's `CLAUDE.md`. Only the layout classifier pilot has been re-run
against this larger corpus so far -- see "Follow-up: re-run over the grown
copyrighted-scans corpus (32 books)" under "History" in the pilot's
section below, or the full write-up in EXPERIMENTS.md. The
other sections on this page (pure-heuristic, strategy-pipeline, diverse
real-library set, per-strategy standalone) still describe the 57/13 split
and have not been re-verified against the new scans books.

## dnb-toc-only ground truth: two-vision-model gate

Per `docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md` and
`docs/superpowers/plans/2026-08-16-dnb-toc-vision-extraction.md`,
`generate_dnb_toc_ground_truth.py` runs a two-independent-vision-model
gate (each model reads the book's page images directly via `pdftoppm`,
no OCR/text layer at all). Three smoke tests (40% with `gemma-4-31b-it`
as the second model; 60% after swapping to the qwen3.6 family; 53% after
a further granularity-prompt fix, whose aggregate rate didn't improve
because a different disagreement cluster then dominated) diagnosed and
fixed a content-dropping reliability gap and a nested-sub-point
granularity gap, and identified front/back-matter inclusion
disagreements as the next-largest remaining cause of gate failures --
see
[EXPERIMENTS.md § dnb-toc-only ground truth: two-vision-model gate](EXPERIMENTS.md#dnb-toc-only-ground-truth-two-vision-model-gate)
for all three runs' full write-ups.

**Front/back-matter prompt fix, plus an arbitration tool for whatever
still doesn't clear the gate (2026-08-16):** `_VISION_TOC_EXTRACTION_PROMPT`
was made explicit that front matter, back matter, and part/section
dividers never get their own entry, and that a two-line title (main
title + subtitle sharing one page number) is a single entry, not two --
see `docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`.
Rather than treat the gate as the final word, a new
`evaluation/scripts/arbitrate_dnb_toc.py` surfaces exactly what each
model extracted for any book that doesn't clear the gate (or where one
model returned nothing usable), so a Claude Code session can resolve it
by hand -- reading the diff, and opening the actual TOC page images when
the text alone doesn't settle it -- instead of the book being silently
discarded.

**Result on the same 15-book sample: 15/15 (100%) now have ground
truth**, up from 8/15 (53%) auto-gated alone:

```
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 15 --concurrency 4
# 8/15 passed the gate automatically ("source": "bulk_gate")

uv run python evaluation/scripts/arbitrate_dnb_toc.py
# surfaced the remaining 7 (5 below_threshold + 2 empty-response errors)
# each resolved by hand and written with "source": "claude_arbitration"
```

Arbitrating the 7 remaining books surfaced real extraction errors that
neither model's raw output alone would have caught, beyond the
already-diagnosed disagreement categories: an off-by-one page number on
6 entries in `3571092120` (the model read a preceding part-divider's own
page number instead of the actual chapter's, e.g. attributing page 79 --
the "Erkenntnistheorie des Rechts" section header's page -- to the
chapter that starts on page 80), a misspelled author name in two
different books (`3571092120`: "Jürgen Rödiger" for "Jürgen Rödig";
`9783515114868`: "Bodo V. Borries" vs. the correct "Bodo von Borries"
elsewhere in the same book's own author list), a spurious bibliography
entry in `9783842331976` ("Literaturverzeichnis", which the prompt
already says to skip but the model included anyway), and one book
(`3465016874`) where both models badly mishandled a 3-level nested
structure (part headers with roman-numeral subsections) badly enough
that it needed full hand-transcription from the page images rather than
reconciling either model's list.

**Still open, not blocking**: the empty-response failure mode on
`qwen3.6`'s side (2 of these 15 books hit it this run) remains
un-root-caused -- live-service flakiness is the leading hypothesis
(different specific qwen3.6 sub-model each run, and the same book
doesn't fail consistently across runs), but arbitration means it no
longer blocks ground-truth coverage, only adds arbitration work.

**Scaling generation to the rest of the corpus (2026-08-17):**
`generate_dnb_toc_ground_truth.py` was changed to skip books that already
have a `.expected.json` or are in `arbitration-rejected.json` (previously
`--limit N` always re-processed the same first-N books in manifest order,
which made repeated invocations useless for advancing through a large
corpus) -- see the function's own `_still_needs_a_decision` docstring.
Running it in successive `--limit 100` batches, each followed by
`arbitrate_dnb_toc.py`, took `dnb-toc-only` from 15 to 170 books with
ground truth. KISSKI rate-limited 30-60% of a typical 100-book batch even
at `--concurrency 4`; per-book errors are cheap to retry (a book without
`.expected.json` just gets re-attempted in the next batch, and any model
whose response was already cached is reused for free), so this cost wall
time, not correctness.

A real batch run did stall completely for several minutes with zero
throughput, though -- root cause: `_run_book`'s concurrency semaphore
wrapped the *entire* per-model retry sequence, including the backoff
`sleep()` between attempts. When enough books hit `RateLimitError` around
the same time, every concurrency slot ended up asleep in backoff
simultaneously, blocking all other pending books from even starting a new
attempt -- confirmed live via `ps` (the stalled process had accrued only
~8s of CPU time across ~3h of wall time) and `lsof` (zero open
connections, so it wasn't hung on a frozen socket, just sleeping). Fixed
by narrowing the semaphore to wrap only the individual API call inside
the retry closure, so it's released during backoff and other books can
proceed -- regression test:
`test_semaphore_is_released_during_backoff_sleep` in
`tests/test_generate_dnb_toc_ground_truth.py`. The same fix incidentally
un-broke a test-suite slowdown introduced alongside the rate-limit-aware
backoff (attempts 3->6, base delay 1s->2s): tests exercising `_run_book`'s
retry paths didn't inject a mock `sleep`, so they burned real wall time on
every real backoff -- `_run_book` now accepts an injectable `sleep` too.

A second, distinct stall showed up immediately after that fix, on the very
next batch: `lsof` on the running process showed 4 TCP connections to
KISSKI's real host stuck `ESTABLISHED` for 20+ minutes with the process
barely using any CPU -- not a client-side backoff sleep this time (no
connections would be open for that), but requests actually in flight and
not returning. Root cause: the `AsyncOpenAI` client was constructed with
no explicit `timeout`, so it fell back to the SDK's own default (600s read
timeout) -- one slow/stuck KISSKI response could occupy a concurrency
slot for up to 10 minutes per attempt, times up to 6 retry attempts, a
worst case over an hour for a single book. Fixed by passing `timeout=90.0`
to the `AsyncOpenAI(...)` call in `_generate` -- generous for a 1-4 page
TOC scan's vision call, but bounds the worst case to something a retry
loop can actually recover from within a batch's lifetime.

**Third stall, same session: a genuine daily quota, not a bug.** After
both fixes above, a fresh batch still made zero progress in its first
3 minutes, repeatedly -- but an isolated single call (no retry wrapper,
no concurrency) failed in 1.6s with `RateLimitError`, ruling out a hang.
The 429 response's own headers settled it precisely (`e.response.headers`
on the raised `RateLimitError`):
`x-ratelimit-limit-day: 1000` / `x-ratelimit-remaining-day: 0` /
`retry-after: 54179` (seconds) -- the account's daily quota (1000
requests) was fully spent by this session's batches 1-4, resetting at a
clean midnight UTC (minute/hour/month limits still had headroom, so day
was specifically the binding one). No further batches are worth
attempting until the daily reset; `_call_with_retry`'s backoff, however
long, cannot recover from a quota that's already at zero. Corpus stood at
206/1251 `dnb-toc-only` books with ground truth (up from 15) when this
was hit.

**Fourth exhaustion, `retry-after` header now confirms hour AND day are both
binding simultaneously (2026-08-18):** a fresh batch resumed once daily
quota reset, made real progress (206 -> 238 books, ~748 individual
model-calls cached), then hit a wall again. Direct header inspection at the
moment of failure: `x-ratelimit-limit-day: 1000` / `remaining-day: 0`,
`x-ratelimit-limit-hour: 200` / `remaining-hour: 0`,
`x-ratelimit-limit-minute: 30` / `remaining-minute: 30` (untouched),
`retry-after: 65597` (~18.2h, resetting at the same clean-midnight-UTC
pattern). Motivated a retry-scheduling fix:
`_call_with_retry` (`evaluation/scripts/generate_dnb_toc_ground_truth.py`)
now reads `retry-after`/`x-ratelimit-remaining-<window>` directly off a 429
response and sleeps the server's own reported delay for an inline-
recoverable "hour"/"minute" window, but gives up immediately (no further
attempts) when the binding window is "day" -- a day-scale reset cannot
happen within one script invocation, so the prior blind linear backoff (up
to ~5min/book x up to 6 attempts) burned real wall time re-discovering
that same fact one book at a time (the ~6.5h run above lost ~91% of its
attempted books this way). See `_binding_rate_limit_window`/
`_retry_after_seconds`'s own docstrings for the exact window-priority
logic; regression tests in `tests/test_generate_dnb_toc_ground_truth.py`
(`TestCallWithRetry`, `TestBindingRateLimitWindow`, `TestRetryAfterSeconds`).

**Spot-check of the bulk-tier gate's real precision (2026-08-19):** the
two-vision-model >=90%-agreement gate only measures whether the two models
*agree*, not whether they're both right -- raising the question of whether
a same-family model pairing (`qwen3-omni-30b-a3b-instruct` +
`qwen3.6-<N>`, see below) might share a correlated blind spot invisible to
the gate itself. Measured directly: 25 books randomly sampled from the 179
`"verified": false` bulk-tier books, each visually reviewed against its
real PDF scan (5 background Claude Code subagents, 5 books each, using the
`Read` tool's image rendering the same way `arbitrate_dnb_toc.py`'s
human-in-the-loop review already works) -- effectively running
`generate_dnb_toc_ground_truth.py --spot-check`'s Accept/Reject protocol
without needing a live terminal or new KISSKI calls.

Naive result: only 7/25 (28%) fully matched their scans. But 16 of the 25
sampled books turned out to still be pre-2026-08-17-schema files (no
`skip` field on any entry at all -- not yet reprocessed by the current
pipeline, purely a backlog/staleness artifact, see `_is_stale_bulk_gate_entry`),
and EVERY one of those 16 failed for the exact same, already-diagnosed,
already-fixed reason: front-matter/back-matter/part-divider lines silently
omitted rather than recorded with `skip: true`. Restricting to the 9
sampled books already on the current schema -- i.e. what the pipeline
actually produces today -- gives **7/9 (78%) precision**, a small but
real sample.

The two current-schema rejects are genuinely informative:

- `9783495485019`: a two-line heading ("Einleitung: / Endlichkeit und
  Verantwortung", both halves on one page) was wrongly split into two
  separate entries, and a page-number-less "Anhang:" divider was
  duplicated into two hallucinated entries instead of appearing once.
- `0292746245`: an author name typo ("Irving Davis" for the correctly-
  printed "Irvine Davis"), plus an internally-inconsistent `skip`
  classification (its own "Index" entry marked `skip: false` despite the
  file correctly marking its own "Contents" entry `skip: true`).

Neither looks like the two models independently making the *same*
mistake (contrast with the earlier `gemma-4-31b-it` content-dropping bug,
independently confirmed via page-range comparison against a third
reading) -- both are more consistent with a **`gate_book` merge-policy
gap**: a matched pair unconditionally keeps side `a`'s title/authors
verbatim once the fuzzy-similarity threshold is cleared (no exact-match
cross-check), and a singleton entry found by only one model is
unconditionally trusted into the merged result on the theory that it's a
real line the other model missed. Both design choices (deliberate, see
`gate_book`'s own docstring) let a single model's individual error survive
into `"verified": false` ground truth undetected -- a risk that would
exist for any two-model pairing, not specifically a same-family one. Real
chapter-level content (titles/authors/page numbers for actual chapters,
as opposed to the divider/front/back-matter lines `skip` exists to mark)
was reliably accurate across nearly all 25 books, current- and
stale-schema alike; the handful of exceptions were isolated single-
character OCR-style typos (e.g. "Urteitskraft"/"Urteilskraft",
"Cotidianeidad"/"Cotidianidad"), not a systematic pattern.

**Conclusion**: the same-family model pairing is not obviously the
dominant risk here -- the measured 78% current-schema precision is
already explained by `gate_book`'s lenient merge policy (structural,
model-family-independent) plus the two isolated single-model errors above,
with no case found of both models independently producing the identical
wrong answer. The bulk of the naive 28% number is pipeline staleness
(pending regeneration once quota allows), not a correlated-bias finding.
Not yet acted on: tightening `gate_book` to flag a near-but-not-exact
matched-pair title (or an unconfirmed singleton) for arbitration instead
of silently trusting it would directly address the two real defects found
here, at the cost of routing more books to `arbitrate_dnb_toc.py` instead
of the fully-automatic bulk tier.

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

`evaluation/scripts/evaluate_layout_toc_classifier.py` trains a
leave-one-book-out (LOBO) classifier on geometric layout features
(`evaluation/scripts/layout_features.py`, derived from cached ALTO XML) and
scores whether it can identify table-of-contents pages and chapter-opening
pages purely from page layout, no text content -- see
`docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md`
for the pilot's original design and decision bar. `full_recall_fraction` is
the share of books, across LOBO folds, whose predicted candidate pages
include every true chapter-opening page (or, since a later follow-up
relaxed this, at least `chapter_first_recall_tolerance` of them -- default
`0.90`) and at least one true TOC page (bar: ≥90% of books);
`avg_candidate_fraction` is the average share of a book's pages that end up
in that candidate set at all -- how much the classifier actually narrows
the page list down (bar: ≤15%, smaller is better).

This pilot has never cleared its own 90%/15% bar on the full mixed corpus.
Eleven follow-up investigations since the original run -- feature
normalization, an OCR data-quality fix, calibration tuning, a
model-architecture swap, two corpus-growth re-runs, a feature-engineering
pass, a targeted feature swap, a per-corpus calibration study, a
source-type-split study, and finally a change to the candidate-selection
mechanism itself -- progressively improved the numbers without closing the
gap. Each is summarized under "History" below, with the full write-up for
each in
[EXPERIMENTS.md § Layout-based TOC/chapter-first-page classifier pilot](EXPERIMENTS.md#layout-based-tocchapter-first-page-classifier-pilot).
What follows is the shipped configuration and the two most recent
measurements against it.

**Current shipped configuration:**

- **Model:** `LogisticRegression(class_weight="balanced")` with a per-fold
  `StandardScaler`, replacing the original `HistGradientBoostingClassifier`
  (see "Follow-up: relaxing the per-book bar, and a model-architecture
  swap" under History below).
- **Features (16):** the original ten geometric layout features, plus two
  page-local features (`last_text_vpos_fraction`, `top_line_heading_match`)
  and four book/context features from `add_book_context_features()`
  (`prev_last_text_vpos_fraction`, `prev_line_count_rel`, `line_count_rel`,
  `font_size_max_ratio_book`), plus `edge_distance`, which replaced
  `page_position_fraction` (see "Follow-up: context/normalized features and
  scan-noise augmentation" and "Follow-up: `edge_distance` feature,
  replacing `page_position_fraction`" under History below). A candidate
  18th feature, `early_gap_ratio`, was tried and reverted -- net negative
  once measured across the full LOBO suite (see "Follow-up: is
  `recall_target` worth splitting by source type? And a rejected feature"
  under History below).
- **`chapter_first_recall_tolerance`:** `0.90` (a book counts as "fully
  recalled" at ≥90% of its true `chapter_first` pages, not literal 100%).
- **Candidate selection:** `select_candidates_by_document_budget` /
  `--candidate-fraction-cap` (default `0.15`) is now `main()`'s default
  path, replacing training-quantile-calibrated `--recall-target`
  thresholding as the shipped default -- see the first follow-up directly
  below. `--recall-target` remains available as an explicit override back
  to the legacy strategy.
- **Scan-noise augmentation** (`--scan-noise-augment`,
  `evaluation/scripts/alto_scan_noise.py`): implemented, cached, off by
  default -- tested and found mostly negative (see "Follow-up:
  context/normalized features and scan-noise augmentation" under History
  below).
- **Ground truth fix (applied, not pending):** `9783837660944.expected.json`
  had a spurious `"Inhalt"` chapter entry duplicating that book's own TOC
  page range, which had been overwriting the true `toc` label with
  `chapter_first` and injecting a false positive; the entry has been
  deleted (see "Follow-up: isolating the open-access corpus..." under
  History below).
- **Not implemented:** splitting `recall_target` by `extraction_type`
  (native vs. scan) was measured as a real, reproducible win (+9 points of
  overall `full_recall_fraction` at a matched candidate budget) under the
  old `recall_target`-based calibration, but was superseded before
  implementation by the candidate-fraction-cap mechanism below, which
  reaches comparable numbers with a single constant and no need to detect
  `extraction_type` at all (see "Follow-up: is `recall_target` worth
  splitting by source type?..." under History below).

The two most recent measurements, in full:

### Follow-up: document-relative candidate-budget selection -- implemented and kept

The "is `recall_target` worth splitting by source type?" follow-up's (see
History below) "more principled retry" idea was implemented and
tested rather than left as a proposal: `select_candidates_by_document_budget`
and `evaluate_leave_one_book_out_document_budget`
(`evaluation/scripts/evaluate_layout_toc_classifier.py`, wired up behind a
new `--candidate-fraction-cap` flag, mutually exclusive with
`--recall-target`) replace `select_threshold`'s training-calibrated,
uniformly-applied absolute probability threshold with a per-document one:
for each held-out book, rank its own pages by `max(prob_toc,
prob_chapter_first)` and take the top `candidate_fraction_cap` share as
candidates -- a single shared top-K selection across both labels, requiring
no ground truth for the document being scored, only its own model output.

**Result: a clear, broad improvement, not a regression, at every operating
point checked -- kept, not reverted.** On the full 89-book corpus (the
mixed pool every earlier `recall_target` sweep in this file struggled
with):

| `candidate_fraction_cap` | `full_recall_fraction` | `avg_candidate_fraction` |
| --- | --- | --- |
| 0.10 | 62% | 9.9% |
| 0.12 | 70% | 11.8% |
| **0.15** | **79%** | **14.9%** |
| 0.18 | 83% | 17.8% |
| 0.20 | 83% | 19.9% |

At `cap=0.15` -- the exact value of the pilot's own candidate-fraction
budget -- this reaches **79% full_recall_fraction at 14.9%
avg_candidate_fraction**, comfortably the best result this file has ever
recorded on the full mixed corpus at or under the 15% cap: far above the
shipped `recall_target=0.90` default's 64%/10.4% at a similar-order
candidate cost, and above even the best single global `recall_target`
found in the previous follow-up's budget-constrained search (73.0%/13.9%
at `recall_target=0.94`) -- approaching the extraction-type-split result
(82.0%/14.5%) with a single constant and no need to know or detect
`extraction_type` at all. On open-access alone, `cap=0.15` reaches
**91.2%/14.9%**, clearing this pilot's 90%/15% bar on that corpus in
isolation (fractionally below the hand-tuned `recall_target=0.988`'s
94.7%/13.8%, but without any corpus-specific retuning -- the same `0.15`
constant that works for the full mixed corpus also clears the bar for
open-access alone). Still **NOT MET overall** on the full 89-book corpus
(79% vs. the 90% bar) -- this doesn't change the pilot's bottom line, but
it is the strongest configuration found for the mixed corpus across this
entire investigation.

**A second, independently useful property, beyond the raw numbers: per-book
candidate_fraction becomes almost perfectly uniform.** Under
`recall_target`-based selection, individual books' `candidate_fraction`
ranged from under 2% to over 58% (see the "isolating the open-access
corpus" follow-up in EXPERIMENTS.md; e.g.
`9781783748471` at 58.3% while `9783839470619` sat at 4.8%, both at the
same `recall_target=0.988`) -- a single global threshold produces wildly
different per-book cost depending on how separated that book's own
probabilities happen to be. Under the budget cap, every book in the
89-book corpus landed within 0.7 points of the 15% target (14.5%-15.0%) at
`cap=0.15`, by construction: a well-separated book's threshold
self-adjusts higher (fewer pages clear the bar for the same rank cutoff)
and a noisy book's self-adjusts lower, but the *volume* handed to whatever
consumes these candidates is predictable per document either way. This
matters operationally if candidates feed something with a per-page cost
(e.g. an LLM confirmation pass): a runaway 58%-of-the-book candidate set
for one hard book is a cost spike a uniform ~15% never produces.

Verified this is purely additive, not a change to existing behavior: a
re-run with no `--candidate-fraction-cap` flag reproduces the exact
`recall_target=0.90` baseline (64%/10.4%, identical per-book numbers) from
before this change, since the new code path is only reachable via the new
flag.

**Update (same day): promoted to the default.** The comparison above was
conclusive enough (equal-or-better recall than any `recall_target` value
found in this pilot's history, at comparable-or-tighter candidate cost, on
both the full corpus and open-access alone, with no per-corpus retuning
needed) that `--candidate-fraction-cap` replaced `--recall-target` as
`main()`'s default code path rather than staying a same-session-only
opt-in flag. `_CANDIDATE_FRACTION_CAP = 0.15` is now the module default;
`--recall-target` remains available but is now the explicit override --
passing it switches back to the legacy training-quantile-calibrated
strategy and `--candidate-fraction-cap` is ignored (inverted from this
follow-up's original wiring, where `--candidate-fraction-cap` was the
opt-in). Verified both directions after the flip: no flags reproduces
today's 79%/14.9% (was previously only reachable via
`--candidate-fraction-cap 0.15`), and `--recall-target 0.90` reproduces
the historical 64%/10.4% shipped-default numbers exactly, confirming nothing
else in the evaluation logic changed. Every `recall_target=...` figure
elsewhere in this file (including "shipped default" language in earlier
follow-ups) describes what was true when it was written and remains an
accurate record of that measurement -- it does not describe today's
default going forward.

### Follow-up: real-scan measurement from the DNB TOC-scan corpus (calibration-grade)

`evaluation/scripts/fetch_dnb_toc_corpus.py` and
`evaluation/scripts/measure_dnb_scan_noise_stats.py` were added (see
`docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`)
to eventually replace the "resembles real scan noise" hand-picked
constants in `alto_scan_noise.py` (`_CONTRAST_ALPHA = (0.3, 0.7)`,
`_FONT_JITTER = (0.96, 1.04)`) with numbers measured directly from real
DNB-digitized table-of-contents scans, sourced via the `lobid-resources`
API. An initial 9-book smoke-test batch (see the earlier version of this
subsection, superseded below) proved the acquisition pipeline worked
end-to-end; a real `--from-dump` bulk run against the live ~21.5GB
lobid-resources dump (2026-08-15) then grew the corpus to the design
spec's "few hundred books" decision-criteria scale. That run also
surfaced and fixed two real acquisition bugs along the way (see
`docs/superpowers/plans/2026-08-15-dnb-toc-corpus-corrections.md`): the
original type filter was too broad and let single-author/thesis/textbook
records into a corpus meant to target edited-volume TOC layouts
specifically (256 of the smoke-test batch's 305 books were purged for
this), and 7 of the newly-acquired 549 had a `toc_download_url` pointing
to an HTML link-out page rather than a real PDF (removed). The corpus now holds **542 verified `EditedVolume`-typed books, all with
a PDF present locally** (5 of the original smoke-test batch's PDFs were
briefly lost when their worktree was cleaned up after merging -- a
lesson-learned noted in the corrections plan -- and were re-downloaded
directly from their already-known `toc_download_url` to complete the set).

```
uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py --pdfalto-bin <path-to-pdfalto>

Title/body contrast ratio (font_size_max_ratio, n=1050):
  measured: {'count': 1050, 'min': 1.0, 'max': 11.26, 'mean': 1.29, 'median': 1.10, 'stdev': 0.65}
  current _CONTRAST_ALPHA range: (0.3, 0.7)

Body-line font-size dispersion (ratio to page modal size, within +/-10%, n=37838):
  measured: {'count': 37838, 'min': 0.90, 'max': 1.10, 'mean': 1.00, 'median': 1.0, 'stdev': 0.0166}
  current _FONT_JITTER range: (0.96, 1.04)
```

`n=1050` non-empty pages and `n=37838` body-like line samples across all
542 real DNB TOC scans -- comfortably calibration-grade, not a preview. As
before, `_CONTRAST_ALPHA` isn't directly unit-comparable to the measured
contrast ratio (it's a compression factor applied to born-digital ALTO's
own, much larger, raw title/body ratio, not a target ratio itself), so
this table alone doesn't resolve whether `_CONTRAST_ALPHA` needs to
change -- that requires a follow-up comparing this measured real
distribution against born-digital `open-access` ALTO's own *uncompressed*
title/body ratio, not done here. What it does establish: the real
distribution is right-skewed (median 1.10 well below the mean 1.29, with
a long tail up to 11.26 -- almost certainly a handful of pages with a
genuinely large decorative heading or an OCR font-size misdetection, not
representative of the median case), and most DNB TOC pages have only mild
title/body contrast, consistent with plain "Inhalt"/"Inhaltsverzeichnis"
listings rather than illustrated title pages.

**The dispersion figure is directly comparable, and is a real, actionable
result: `_FONT_JITTER`'s current `(0.96, 1.04)` range looks well-calibrated.**
Measured real body-line sizes have stdev 0.0166 around their page's modal
size; `_FONT_JITTER`'s half-width of 0.04 is about 2.4 real standard
deviations -- a sensible coverage for a uniform jitter range meant to
approximate a real, roughly bell-shaped noise distribution without
needing to change. At this sample size (37,838 line samples, 542 books),
this is a confirmation that the original hand-picked guess was
well-calibrated, not just an unconfirmed guess -- **no change recommended
to `_FONT_JITTER`.**

With the heuristics above, a full run of that now-deleted script
(KISSKI-backed preset) reported **identical numbers to the pure-heuristic
harness, with neither fallback path firing on any book**: TOC extraction
never triggered because the regex path found a usable listing everywhere,
and per-entry ambiguity was resolved heuristically by the TOC-order
constraints before the LLM would be consulted. This run predated the
layout-mode fallback and OCR route described above, and only covered the 7
originally-committed books, not the 10 "diverse real-library" books.

**Deferred, still-untouched directions** (named across several follow-ups
above, not scoped or attempted in any of them): TOC-anchored chapter
matching (parsing a detected TOC page and locating chapter openings by
title/page-number matching against it, to address the `open-access`
TOC-layout failure mode) and document-image deep learning bypassing OCR
font metadata entirely (to address the residual scan `chapter_first_recall`
ceiling).

### History (superseded snapshots and follow-ups)

- **Original pilot run and feature-normalization follow-up.** The pilot's
  first run (pre-growth 50-book corpus, ten raw layout features) came back
  NOT MET: 16% `full_recall_fraction` against a comfortably-cleared 5.3%
  `avg_candidate_fraction`, with an 8-book cluster (all `copyrighted-scans`)
  stuck at exactly 0% `chapter_first` recall. A follow-up root-caused most
  of the shortfall to four features being raw, unnormalized ALTO point
  coordinates rather than page-width fractions; fixing that improved the
  result only slightly (18%/5.4%) and left the `open-access`/
  `copyrighted-scans` split essentially unchanged, pointing at the decision
  bar's literal 100%-per-book requirement as the larger, still-unaddressed
  factor. Superseded by every later corpus-growth and calibration follow-up
  below. Full write-up:
  [EXPERIMENTS.md § Original pilot run and feature-normalization follow-up](EXPERIMENTS.md#original-pilot-run-and-feature-normalization-follow-up).
- **Follow-up: replacing textless/degenerate-text corpus PDFs with OCR'ed
  versions.** Traced 6 of the 8 zero-recall `copyrighted-scans` books to
  `pdfalto` (the pilot's only extraction tool) having no OCR fallback,
  unlike the text-based pipeline; replaced those 6 PDFs in the shared
  corpus with `ocrmypdf`-OCR'ed versions. Per-book `chapter_first_recall`
  improved sharply on 4 of the 6, but the corpus-wide `full_recall_fraction`
  stayed flat at 18% because the LOBO setup redistributes the effect across
  other books' folds -- further evidence that the 100%-per-book bar, not
  data quality, is the dominant blocker. Superseded by later corpus growth.
  Full write-up:
  [EXPERIMENTS.md § Follow-up: replacing textless/degenerate-text corpus PDFs with OCR'ed versions](EXPERIMENTS.md#follow-up-replacing-textlessdegenerate-text-corpus-pdfs-with-ocred-versions).
- **Follow-up: recall-target tuning, concentrating on the open-access
  corpus.** Swept the `_RECALL_TARGET` calibration knob from 0.90 to 1.00
  and found a plateau at 0.97 (28% `full_recall_fraction`, 7.0%
  `avg_candidate_fraction`) -- the cheapest point on the plateau before
  `recall_target=1.00`'s degenerate cliff. Three other tuning ideas
  (training on `open-access` only, a new `max_font_vpos_fraction` feature,
  per-label recall targets) were tried and rejected as negative results.
  Superseded by the model-architecture follow-up's own recalibration. Full
  write-up:
  [EXPERIMENTS.md § Follow-up: recall-target tuning, concentrating on the open-access corpus](EXPERIMENTS.md#follow-up-recall-target-tuning-concentrating-on-the-open-access-corpus).
- **Follow-up: relaxing the per-book bar, and a model-architecture swap.**
  Relaxed the pilot's exact 100%-per-book bar to a tunable
  `chapter_first_recall_tolerance` (default 0.90) and compared three
  classifier architectures head-to-head; `LogisticRegression` clearly
  outperformed the incumbent `HistGradientBoostingClassifier` (44% vs. 40%
  `full_recall_fraction` at matched `avg_candidate_fraction`) because its
  smooth linear score generalizes across books better than the tree model's
  discrete leaf regions. Became the pilot's shipped model from this point
  on, with `_RECALL_TARGET` retuned to 0.80. Superseded numerically by
  later corpus growth and recalibration, though the model choice and the
  tolerance mechanism itself remain in the current configuration. Full
  write-up:
  [EXPERIMENTS.md § Follow-up: relaxing the per-book bar, and a model-architecture swap](EXPERIMENTS.md#follow-up-relaxing-the-per-book-bar-and-a-model-architecture-swap).
- **Follow-up: re-run over the grown 70-book corpus.** Re-ran the unchanged
  `LogisticRegression`/`recall_target=0.80` configuration against the
  newly-grown 70-book corpus (up from 50): `full_recall_fraction` jumped to
  64% (from 44%), the largest single jump in the pilot's history, achieved
  purely from more/corrected ground truth. `open-access` drove nearly the
  entire improvement (54.1% -> 77.2%); `copyrighted-scans` regressed
  slightly (15.4% -> 7.7%) due to a shifted LOBO training pool, not a data
  change. Superseded by the next follow-up's feature/calibration change.
  Full write-up:
  [EXPERIMENTS.md § Follow-up: re-run over the grown 70-book corpus](EXPERIMENTS.md#follow-up-re-run-over-the-grown-70-book-corpus).
- **Follow-up: context/normalized features and scan-noise augmentation.**
  Added seven book-context/normalized features (10 -> 17 total) and a
  scan-noise data-augmentation module; at the old `recall_target=0.80` this
  looked like a regression (56% vs. 64% baseline) but a sweep showed it was
  purely an operating-point effect -- at `recall_target=0.90` the
  17-feature model beat baseline on every axis (67% vs. 64% full recall,
  9.0% vs. 10.0% candidates), so `_RECALL_TARGET`'s default moved from 0.80
  to 0.90. Scan-noise augmentation itself was a mostly-negative result and
  stayed off by default. Superseded numerically by later corpus growth and
  the candidate-budget-cap follow-up, though the 0.90 recall-target default
  and augmentation-off decision held until candidate-budget selection
  replaced the mechanism entirely. Full write-up:
  [EXPERIMENTS.md § Follow-up: context/normalized features and scan-noise augmentation](EXPERIMENTS.md#follow-up-contextnormalized-features-and-scan-noise-augmentation).
- **Follow-up: re-run over the grown copyrighted-scans corpus (32 books).**
  Re-ran the shipped 17-feature/`recall_target=0.90` configuration after
  `copyrighted-scans` grew from 13 to 32 books (89 total): the headline
  number read as a regression (64% vs. the 70-book snapshot's 67%) but was
  a corpus-composition effect -- the 19 new books were deliberately
  acquired as the classifier's hardest known case. Re-scored on just the
  original 13 scans, performance actually improved (2/13 -> 4/13
  full-recall passes). Superseded by the `edge_distance` feature follow-up
  immediately after. Full write-up:
  [EXPERIMENTS.md § Follow-up: re-run over the grown copyrighted-scans corpus (32 books)](EXPERIMENTS.md#follow-up-re-run-over-the-grown-copyrighted-scans-corpus-32-books).
- **Follow-up: `edge_distance` feature, replacing `page_position_fraction`.**
  Added an `edge_distance` feature (`min(page_index, total_pages - 1 -
  page_index)`) to fix a specific failure: a book whose TOC sits at the
  very back rather than front of the book, which the existing
  `page_position_fraction` feature (monotonic, so it can only reward one
  direction) couldn't encode. Replacing `page_position_fraction` with
  `edge_distance` fixed that book outright and produced a small overall
  gain (64.0% -> 65.2% full recall over 89 books). This feature swap is
  part of the current shipped configuration above. Full write-up:
  [EXPERIMENTS.md § Follow-up: `edge_distance` feature, replacing `page_position_fraction`](EXPERIMENTS.md#follow-up-edge_distance-feature-replacing-page_position_fraction).
- **Follow-up: isolating the open-access corpus -- recall_target retuned,
  per-feature weighting tested and ruled out.** Asked why the pilot scores
  so far below its bar on `open-access` alone despite that corpus's clean
  layout, and whether reweighting the model's features could close the
  gap. Per-feature reweighting (via L2 regularization strength) was tested
  and ruled out -- it doesn't change the within-book *ranking* the
  threshold depends on. Retuning `recall_target` specifically for
  `open-access` alone did work: at `recall_target=0.988`, `open-access` in
  isolation cleared the pilot's 90%/15% bar for the first time in the
  pilot's history (94.7%/13.8%). Also fixed a real ground-truth defect
  found along the way (the `9783837660944` `"Inhalt"` entry, see the
  current configuration above). Superseded as a shipped mechanism by the
  candidate-budget-cap follow-up, which reaches comparable or better
  numbers without per-corpus retuning; the ground-truth fix remains
  applied. Full write-up:
  [EXPERIMENTS.md § Follow-up: isolating the open-access corpus -- recall_target retuned, per-feature weighting tested and ruled out](EXPERIMENTS.md#follow-up-isolating-the-open-access-corpus----recall_target-retuned-per-feature-weighting-tested-and-ruled-out).
- **Follow-up: is `recall_target` worth splitting by source type? And a
  rejected feature.** Tested whether calibrating `recall_target`
  separately for native-extraction vs. scanned books (rather than one
  global constant) could buy more recall for the same candidate-fraction
  budget: it did (82.0% vs. 73.0% full recall at a matched budget), but
  scans' own ceiling stayed flat around 43% regardless of their own
  target, confirming scans are limited by signal, not calibration --
  recommended if the pilot ships, but not implemented, since the whole
  `recall_target` calibration mechanism was replaced shortly after (see
  "document-relative candidate-budget selection" above). Also proposed and
  rejected an `early_gap_ratio` feature (real isolated signal on 2 of 3
  target books, but net negative once wired into the full 89-book LOBO
  run) -- reverted, not part of the shipped feature set. Full write-up:
  [EXPERIMENTS.md § Follow-up: is `recall_target` worth splitting by source type? And a rejected feature](EXPERIMENTS.md#follow-up-is-recall_target-worth-splitting-by-source-type-and-a-rejected-feature).

## NuExtract: lessons learned from earlier models (2026-08-09)

Before settling on NuExtract-2.0-4B (below), two earlier/larger models in
the same family were evaluated and rejected: **NuExtract-1.5-tiny**
(zero-shot f1=0.00 across a full 50-book run -- a genuine model-capability
limit, not a serving-path artifact: no instruction channel to ignore
surrounding non-ToC noise) and **NuExtract3 (4B)** (f1=0.60 zero-shot,
competitive with the cloud-LLM baseline, but dropped for CPU-only
deployment reasons -- NuExtract-2.0-4B is 1.42x faster with equal-or-better
accuracy on the actual no-GPU, 16GB RAM Linux target). Full investigation
detail for both: [EXPERIMENTS.md § NuExtract: lessons learned from earlier models (2026-08-09)](EXPERIMENTS.md#nuextract-lessons-learned-from-earlier-models-2026-08-09).

## NuExtract-2.0-4B zero-shot baseline (2026-08-10)

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

### Full 50-book, two-corpus run (GGUF Q4_K_M, Metal-offloaded) — superseded

A first full 50-book run at `max_tokens=1500` scored **f1=0.39**
(copyrighted-scans 0.17, open-access 0.47; total wall time 4156s) --
noticeably below NuExtract3's own f1=0.60 on the same corpus, as expected
for the smaller/older model. A failure-mode breakdown found the low
aggregate was dominated less by truncation (20% of books) than by a
`printed_page_number`-stays-null pattern despite otherwise-correct
title/author extraction (28% of books) -- meaning title-only accuracy was
substantially better than the headline f1 suggested. Superseded by the
output-token-limit retest below, which found much of the truncation
cluster was a fixable `max_tokens` artifact and established f1=0.44, not
f1=0.39, as the real baseline to beat. Full write-up:
[EXPERIMENTS.md § Full 50-book, two-corpus run (GGUF Q4_K_M, Metal-offloaded)](EXPERIMENTS.md#full-50-book-two-corpus-run-gguf-q4_k_m-metal-offloaded).

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

## NuExtract-2.0-4B LoRA fine-tuning pilot: result (2026-08-14)

Ran the pilot for real on MPCDF Raven (see
`docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`
for the design, `evaluation/hpc/README.md` for the HPC deployment itself
-- getting it running surfaced a long chain of environment/dependency
bugs, all fixed and documented there and in `nuextract.def`/
`run_pilot.slurm`'s own comments; nothing environment-specific belongs
here). Trained a LoRA adapter (rank 16, 4 epochs, gradient checkpointing)
on 78 books' TOC-scan-window text (89-book corpus, 78 train / 11 eval,
stratified split seed 42), merged, converted to GGUF Q4_K_M, and scored
both the fine-tuned and unmodified base checkpoint on the same 11-book
held-out split via the same `llama.cpp`-only scoring path
(`evaluate_nuextract_finetune.py`) -- an apples-to-apples comparison,
not directly comparable to the full-corpus `f1=0.44` baseline above
(different, much smaller subset).

| | precision | recall | f1 |
| --- | --- | --- | --- |
| Fine-tuned | 0.83 | 0.46 | **0.59** |
| Base (same split, same code) | 0.57 | 0.48 | **0.52** |

**Aggregate f1 improved (0.52 -> 0.59), but the per-book picture is more
complicated than "fine-tuning helps" -- it's propped up by big wins on a
few books while masking a real regression on two others:**

| Book | Fine-tuned f1 | Base f1 | Δ |
| --- | --- | --- | --- |
| copyrighted-scans/9783848736829 | 0.98 | 0.47 | +0.51 |
| copyrighted-scans/9783161538315 | 0.00 | 0.00 | — (both fail, pre-existing) |
| copyrighted-scans/9783428042241 | 0.95 | 0.94 | ~even |
| open-access/9781800641648 | **0.00** | 0.96 | **−0.96** |
| open-access/9781771993661 | 0.95 | 0.95 | ~even |
| open-access/9783839458013 | **0.00** | 0.30 | **−0.30** |
| open-access/9781906924874 | 0.96 | 0.90 | +0.06 |
| open-access/9783031466373 | 1.00 | 0.91 | +0.09 |
| open-access/9783907297285 | 0.96 | 0.96 | ~even |
| open-access/9781805111856 | 0.49 | 0.00 | +0.49 |
| open-access/9781805115717 | 0.00 | 0.00 | — (both fail, pre-existing) |

**Root cause of the two collapses: a decoding-time degenerate-repetition
loop, not a fine-tuning capability regression.** Added `--dump-dir` to
`evaluate_nuextract_finetune.py` (writes each book's raw completion
text, `finish_reason`, and parsed/expected chapters) to inspect why two
books scored 0/0 despite the model clearly having the right knowledge.
Both books' raw output showed `finish_reason: length` -- the model
correctly extracted several early chapters completely correctly (real
titles, real authors, real page numbers) before falling into an
infinite loop (`9781800641648`: repeating Hebrew transliteration
diacritics; `9783839458013`: repeating `…`) that burned the entire
`--max-tokens` budget without ever closing the JSON, so `parse_response`
saw truncated/invalid JSON and scored 0/0 -- not because the model didn't
know the answer, but because greedy decoding (`temperature=0.0`, no
repetition penalty) got stuck. `9783839458013` was already a known
repetition-prone book in the zero-shot baseline above (one of the four
`[HIT_MAX_TOKENS]` books that didn't recover even at `max_tokens=12000`)
-- this pilot didn't introduce that tendency, though `9781800641648` collapsing
is new (it scored 0.96 zero-shot in this same run).

**Tried fixing it with a repeat penalty; made the aggregate worse both
times.** The zero-shot baseline section above speculated
"repetition-penalty sampling" as a possible fix for exactly this failure
shape -- tested it for real here, twice:

| Config | `9781800641648` | `9783839458013` | 4 other previously-fine books | Aggregate f1 |
| --- | --- | --- | --- | --- |
| No penalty (baseline) | 0.00 | 0.00 | all 0.49-1.00 | **0.59** |
| `repeat_penalty=1.1`, 64-token window (llama-cpp-python default) | 0.83 | 0.09 | 3 collapsed to 0.00, 1 dropped to 0.62 | 0.41 |
| `repeat_penalty=1.1`, 16-token window | 0.38 | 0.22 | 2 still 0.00, 1 dropped to 0.56 | 0.34 |

Fixed the two target books (partially) but broke others every time: our
output is a JSON *list* of chapter dicts, repeating the same field names
(`"title"`/`"authors"`/`"printed_page_number"`) every ~20-40 tokens --
any blanket repeat penalty, at any window size tried, seems to disrupt
this model's ability to produce that legitimate, required repetition,
not just the genuine 1-4-token degenerate loops. Reverted to no penalty
(`--repeat-penalty`/`--repeat-last-n` remain available as documented,
off-by-default flags for future experimentation, not because they're
expected to work as-is). A more promising direction, if this failure
rate turns out to matter: detect the loop and salvage the valid JSON
prefix generated before it, at the application layer instead of the
sampler.

**Against the design spec's actual decision criterion -- a promising
but genuinely noisy signal, not a clean go.** The spec calls for
checking whether the *null-page-number rate* specifically dropped, not
just the aggregate f1; that specific check wasn't done here (would need
inspecting more of the `--dump-dir` output by hand across the 7 correctly
-scoring books), but the raw dumps for the two collapsed books are
suggestive -- every chapter extracted before the loop struck had a
correct, non-null `printed_page_number`, not the null-page-number
failure the pilot was meant to fix. Per the spec's own caution, an
11-book split is "powered to detect obviously helps' vs 'clearly
doesn't help,' not to measure a precise effect size" -- and this result
is neither: real wins on some books, a real new failure mode on two
others, aggregate f1 up but not overwhelmingly so. Worth a follow-up
decision (extend ground truth for a bigger/more stable split? pursue the
JSON-prefix-salvage fix for the repetition failures first?) rather than
either shipping this adapter or abandoning the approach on this result
alone.
