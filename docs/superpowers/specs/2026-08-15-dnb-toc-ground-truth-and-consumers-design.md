# Structured ground truth for dnb-toc-only, and its three consumers

Status: approved for planning (high-level / program spec — see "How to use this
document")
Date: 2026-08-15

## How to use this document

This is a **program-level spec, not a single implementation plan.** It covers
four pieces of work that share one input (structured ground truth built from
the `dnb-toc-only` corpus) but are otherwise independent: the GT-generation
phase itself, plus three separate downstream consumers. Each of the four
sections below (§2-§5) is scoped enough to become its own focused
brainstorm → design → plan cycle later, in the priority order given here.
This document's own deliverable is the shared understanding and schema those
later specs will build on — it does not, by itself, authorize writing code.

## 1. Problem and context

`evaluation/corpus/dnb-toc-only/` (542 books today, see
`docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`)
holds real DNB-digitized table-of-contents scans acquired specifically to
calibrate layout-only parts of the pipeline against real scan noise. That
spec explicitly deferred building any structured ground truth from these
scans ("Generating `<id>.expected.json` chapter lists... needs its own
extraction/verification step... this spec doesn't design or build" it) —
this document is that follow-up.

Three existing subsystems could plausibly benefit from structured
`{title, authors, printed_page_number}` ground truth built from this corpus,
but each has a different real shape of need, confirmed by inspecting the
current code rather than assumed:

- **The layout-based TOC/chapter-first-page classifier**
  (`evaluation/scripts/evaluate_layout_toc_classifier.py`,
  `layout_features.py`) is geometry-only (ALTO layout features, no text
  content), requires per-book context features (`page_position_fraction`/
  `edge_distance`, previous-page stats) computed from a full surrounding
  book, and — confirmed by direct inspection — **is purely an evaluation
  pilot today**: no production code path in `src/chapter_segmentation/`
  references it, and the trained model is never saved, only retrained fresh
  every run. A standalone TOC-only scan (no surrounding book) cannot supply
  the context features this classifier's current design depends on.
- **NuExtract-2.0-4B fine-tuning**
  (`evaluation/scripts/prepare_nuextract_finetune_data.py`,
  `evaluation/nuextract2_common.py`'s `build_target`) trains on
  `{"text": <TOC-window page text>, "target": {"chapters": [{"title",
  "authors", "printed_page_number"}, ...]}}` pairs. Confirmed by reading
  `prepare_nuextract_finetune_data.py`: `pdf_start_index`/`pdf_end_index`
  are read into each JSONL row only for later eval scoring
  (`evaluate_nuextract_finetune.py`), never used to build the training
  target itself. **A training example needs nothing but the TOC listing's
  own text and its own printed page numbers** — exactly what a TOC-only
  scan can supply, no full book required.
- **The heuristic's TOC-line parser** (`find_toc_candidates`, `TocEntry`,
  `_TOC_LINE_RE`, `_parse_toc_page_number` in
  `src/chapter_segmentation/segmentation.py`) has, confirmed by search, **no
  existing accuracy harness at all** — `tests/test_segmentation.py` checks
  it only against small hand-crafted synthetic strings, and every
  `evaluation/` accuracy script scores end-to-end `pdf_start_index`/
  `pdf_end_index` chapter boundaries, never whether a TOC line was parsed
  into the right title/page-number. A large, diverse, real-OCR sample with
  no full-book requirement is exactly what's missing to build one.

This asymmetry is the spine of the whole document: dnb-toc-only is a strong
match for NuExtract fine-tuning and for a new heuristic line-parsing harness,
and a weak match for the layout classifier as currently architected.

There is currently **no schema anywhere in this repo** for standalone
structured TOC-entry ground truth (confirmed: `add_toc_ground_truth.py`'s
`"toc"` field stores only `{toc_start_index, toc_end_index}`, never parsed
entries). §2 defines one.

## 2. Ground-truth generation (do this first)

### 2.1 Schema

New file per book, `evaluation/corpus/dnb-toc-only/<id>.expected.json`:

```json
{
  "entries": [
    {"title": "Einleitung", "authors": [], "printed_page_number": "9"},
    {"title": "Zur Soziologie des Rechts", "authors": ["Max Mustermann"], "printed_page_number": "17"},
    {"title": "Bibliographie", "authors": [], "printed_page_number": null}
  ],
  "verified": false
}
```

- `title`/`authors`/`printed_page_number` deliberately reuse `TocEntry`'s own
  field shapes and `NUEXTRACT_TEMPLATE`'s target shape exactly — every
  consumer in §3-§5 reads this schema with zero translation.
- Every line the TOC scan actually prints gets an entry, including ones a
  full-book `.expected.json` would mark `skip: true` (front/back matter,
  bibliography, index) — this file measures extraction fidelity against
  what the page prints, not "which of these are real chapters," so nothing
  here is filtered out the way it is for `open-access`/`copyrighted-scans`.
- `printed_page_number: null` is a legitimate value (a line with no visible
  printed number), same convention as the existing corpora — never guessed.
- `"verified"` reuses the exact flag and meaning `generate_public_evaluation_cache.py`
  already established for "cached but not hand-confirmed clean": `false` for
  the bulk tier (§2.3), `true` for the held-out eval tier (§2.4).
- No `pdf_start_index`/`pdf_end_index`/`toc_start_index` fields — there is no
  full book here, only the TOC scan itself, so those concepts don't apply.
  (`dnb-toc-only`'s existing `manifest.json` already establishes that this
  corpus doesn't follow the other corpora's directory/field shape; this file
  is this corpus's own equivalent of `.expected.json`, not a repurposing of
  the existing one.)

### 2.2 Why two tiers, not one review pass over everything

For a TOC-only scan there is no separate "the real book" to cross-check a
draft against — the scan image the OCR text came from *is* the source of
truth, so verifying one entry is a single-page read, not a multi-page search
through a different document the way `CLAUDE.md`'s existing chapter-locate
workflow requires. That makes a full by-hand pass over ~1000 books far
cheaper per book than the existing corpora's ground truth, but still not
free — the win below comes from not needing to look at most of them at all,
not from each look being fast.

### 2.3 Bulk tier — agreement-gated dual extraction, target ~450-950 books, `verified: false`

Run two extractors per book against the DNB scan's own OCR text, independent
of each other by construction:

1. The existing heuristic regex path — `find_toc_candidates`/`TocEntry`,
   unmodified.
2. A cheap zero-shot LLM pass — `llm_extract_toc_entries` (a KISSKI call,
   already used elsewhere in this project). Deliberately **not** NuExtract
   itself: using NuExtract's own zero-shot output to help build data that
   later trains and evaluates NuExtract would be circular.

Where the two agree — fuzzy title match (reuse `fusion.py`'s
`_ALIGN_SCORE_THRESHOLD` convention) and exact `printed_page_number` match —
auto-accept the entry with **no human review**. Agreement between two
structurally unrelated extraction methods is the correctness signal; a later
spec should spot-check a random sample of auto-accepted entries against the
scan image to measure the real precision of this proxy before trusting it at
full scale. Entries where the two disagree, or where one or both find
nothing, are queued rather than guessed at — either reviewed by hand at
lower volume than a full pass, or the whole book is simply left out of the
bulk tier if too little of it resolves cleanly (same "leave it uncached
rather than force a fix" stance `CLAUDE.md`'s redaction section already
takes for a structurally similar problem).

### 2.4 Eval tier — hand-verified, held out, ~50-100 books, `verified: true`

Carved out up front (e.g. randomly sampled, or stratified across the
publishers/eras/decades the acquisition metadata already carries) and never
entered into any consumer's training data. Transcribed by direct visual
read of the scan image — same discipline as `CLAUDE.md`'s existing
ground-truth workflow, minus the chapter-locate step, so meaningfully
cheaper per book. This is what gives §3-§5 a trustworthy, non-circular
number to measure against; it must not be drafted by whichever method it's
later used to evaluate (e.g. don't use NuExtract-drafted entries, reviewed
or not, as the eval set NuExtract's own fine-tuning is judged against).

### 2.5 Corpus growth (trivial prerequisite)

`uv run python evaluation/scripts/fetch_dnb_toc_corpus.py --from-dump
--limit 1000` — the acquisition pipeline is already built, validated, and
produces `EditedVolume`-filtered, PDF-verified records (see the 2026-08-14
spec and its corrections plan). No new design needed; just run it with a
higher limit before or alongside §2.3-2.4.

## 3. Consumer priority 1 — NuExtract-2.0-4B fine-tuning

The cleanest match, per §1: a bulk-tier or eval-tier entry maps directly
onto the existing training-example shape with no schema translation.
Follow-up spec should cover: extending `prepare_nuextract_finetune_data.py`
(or a sibling script) to fold dnb-toc-only's bulk tier into `train.jsonl`
and its eval tier into a held-out split never mixed with training data;
whether/how to weight or cap dnb-toc-only rows relative to the existing
89-book corpus's rows (real scans vs. mostly-native PDFs — avoid letting one
source's formatting quirks dominate); and re-running
`evaluate_nuextract_finetune.py`-style scoring against the new eval tier to
check whether the dominant zero-shot failure mode this pilot already
identified (`printed_page_number` staying null) measurably improves. Since
these are real bitonal DNB scans, this also grows the training pool's real
scan-OCR representation specifically, which the current 89-book corpus (bulk
of it native-PDF `open-access`) under-represents relative to the actual
scanned-personal-library deployment target.

## 4. Consumer priority 2 — Heuristic TOC-line-parsing accuracy harness

Since none exists today (§1), build one: a new script structurally
analogous to `nuextract_baseline.py`'s `match_toc_entries`/`score_book`, but
scoring `find_toc_candidates`/`TocEntry` output against dnb-toc-only's eval
tier specifically (never the bulk tier, for the same non-circularity reason
as §3 — the bulk tier's agreement-gating already leans partly on the
heuristic's own output). This isolates line-parsing accuracy from full
chapter-boundary accuracy for the first time in this project, against a
sample far larger and more diverse (multiple publishers, eras, OCR
qualities) than the current 70-89-book evaluation corpora provide. A
follow-up spec should scope this against the specific failure modes
`RESULTS.md` already documents and treats as unresolved: dense dot-leader
OCR garbling losing the title-number line shape entirely, a secondary
listing (bibliography/citation index) outscoring a degraded real TOC, and
multi-line title merging. This harness becomes the regression guard and
tuning target for any future change to `_TOC_LINE_RE`/`find_toc_candidates`.

## 5. Consumer priority 3 — Layout-based TOC classifier (weak match, capped ROI — defer)

Per §1, this classifier's current design (geometry-only, per-book context
features, no production wiring, model never persisted) is a poor fit for
what dnb-toc-only's *structured entry* ground truth specifically adds — it
already gets free, GT-free positive-page examples from the corpus
acquisition alone (every page in every book is a confirmed `toc` page by
construction, per the 2026-08-14 spec), and §2's title/author/page-number
transcription is a different signal type (text content) than what this
classifier consumes (page geometry). A later spec, if picked up, has two
directions to choose between rather than one obvious path: (a) restructure
into a two-stage pipeline — a context-free, page-local pre-filter trained on
a much larger pool (dnb-toc-only positives plus negative/non-TOC pages
sourced from the existing corpora) feeding into the existing context-aware
re-scorer only for pages that survive stage 1; or (b) keep dnb-toc-only to
the out-of-population recall sanity check the 2026-08-14 spec already
scoped, and invest no further. **Recommendation: defer real investment
here until this classifier has an actual production consumer** — right now
it is an evaluation pilot with no deployment path, so improving its
LOBO numbers further has no measurable downstream effect.

## 6. Recommended sequencing for follow-up specs

1. §2 (ground-truth generation) — everything else depends on this.
2. §3 (NuExtract fine-tuning) — highest confirmed payoff, cleanest schema
   match, most directly addresses an already-documented, already-measured
   failure mode.
3. §4 (heuristic line-parsing harness) — fills a real, currently-total gap
   in what this project can measure, moderate implementation cost (mirrors
   an existing pattern in `nuextract_baseline.py`).
4. §5 (layout classifier) — defer; revisit only once/if the classifier gains
   a production consumer, at which point re-evaluate which of the two
   directions in §5 actually matters.

## 7. Decision criteria

- §2 ships a defined `<id>.expected.json` schema, a bulk tier covering a
  clear majority of the (grown, ~1000-book) corpus with `verified: false`,
  and a held-out eval tier of ~50-100 books with `verified: true` that was
  never drafted by any method it will later be used to score.
- A spot-check of the bulk tier's auto-accepted entries against real scan
  images reports a measured precision (not assumed) for the
  agreement-gating proxy, in whatever document that follow-up spec's own
  results land in.
- §3's follow-up reports a before/after comparison of NuExtract fine-tuning
  eval numbers (F1, and specifically the null-`printed_page_number` rate)
  with vs. without dnb-toc-only rows in the training set.
- §4's follow-up produces a working, reusable line-parsing accuracy script
  and a first baseline number for `find_toc_candidates` against the eval
  tier — closing the gap named in §1, whether or not that baseline number
  is good.

## 8. Out of scope (for this document and, unless a later spec says
   otherwise, for §2-§4 as well)

- Any change to `evaluate_layout_toc_classifier.py`'s model, features, or
  production wiring (§5 is deferred, not designed here).
- Automating Zenodo publication of the dnb-toc-only PDF corpus (already out
  of scope per the 2026-08-14 spec; unaffected by this document).
- Growing `open-access`/`copyrighted-scans` using dnb-toc-only data — DNB
  digitizes only the TOC excerpt, never the surrounding book, so this
  source still cannot supply new full-book evaluation entries (unchanged
  from the 2026-08-14 spec's own "out of scope").
- Building the review-app UI extension mentioned conversationally as a
  possible efficiency booster for §2.3's disagreement queue — worth
  considering inside that follow-up spec, not decided here.
