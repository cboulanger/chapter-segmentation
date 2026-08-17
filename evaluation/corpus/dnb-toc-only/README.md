# `dnb-toc-only` corpus

Real, born-scanned table-of-contents pages acquired from the **Deutsche
Nationalbibliothek's "Kataloganreicherung" program** via the `lobid-resources`
API (`lobid.org/resources`), not from this project's own books. Each entry's
`.pdf` is a short (1-4 page), 300dpi bitonal scan of *just the TOC*, with an
embedded OCR text layer, released under **CC0** (no restriction, no
attribution) — full provenance and acquisition details in
`docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`.

This is why the corpus doesn't follow the shape `evaluation/README.md`
describes for every other corpus: there is no full book here, no
`pdf_start_index`/`pdf_end_index` chapter boundaries to locate, and no
`public-cache/` (nothing to redact — the scan itself already only ever
contains the TOC page's own printed text). `.expected.json` here is a flat
list of what the TOC page prints, not a chapter-location result:

```json
{
  "entries": [
    {"title": "...", "authors": ["..."], "printed_page_number": "N"}
  ],
  "verified": true,
  "source": "claude_arbitration"
}
```

`manifest.json` carries `"toc_only": true` at the top level and, per book,
`filename`, `title`, `language`, `doi`, `toc_download_url` (the original DNB
scan URL), `license`/`license_source` (`"CC0-1.0"`/`"dnb"` for essentially
every entry), and `lobid_url` (the source bibliographic record — also where
to re-fetch metadata if a field looks wrong).

## Why it exists

Two consumers, one acquisition pipeline:

- **Layout-based TOC/chapter-first-page classifier training data** — real
  scan noise (skew, bitonal artifacts, genuine font/contrast variation)
  at a scale (~1,251 books) this project's own scanned corpus
  (`copyrighted-scans/`, 13 books) can't approach, replacing the
  synthetic scan-degradation model `alto_scan_noise.py` used before.
- **A large, cheap "does automated TOC extraction get this right" ground
  truth set** — since each book is just its TOC page(s), building ground
  truth here means transcribing what's printed, not locating chapter
  boundaries in a full book — cheap enough to attempt at scale via two
  independent vision models instead of hand-transcribing every book.

## Ground-truth generation: two-tier workflow

Both tiers write `<id>.expected.json` in the schema above; only the
`"verified"`/`"source"` values and how much human judgment went in differ.

**Bulk tier** (`"verified": false, "source": "bulk_gate"`) — automated, no
human review. `evaluation/scripts/generate_dnb_toc_ground_truth.py` renders
each book's page images (`pdftoppm`, no OCR) and sends them to two
independent vision-capable KISSKI models
(`evaluation.dnb_toc_vision.vision_extract_toc_entries`); a book's
`.expected.json` is written only when the two models agree on at least 90%
of entries (`evaluation.dnb_toc_matching.gate_book`). Already-decided books
(an existing `.expected.json`) and rejected ones (below) are skipped
automatically, so re-running the same command just picks up where the last
run left off:

```bash
export KISSKI_API_KEY=$(grep '^KISSKI_API_KEY=' ../zotero-rag/.env | cut -d= -f2-)
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 100 --concurrency 4
```

**Arbitration** (`"verified": true, "source": "claude_arbitration"`) — for
books the bulk tier skipped (models disagreed, or one/both failed outright).
`evaluation/scripts/arbitrate_dnb_toc.py` surfaces each one's two raw
extractions (from `llm-cache/<key>.<model>.json`, kept regardless of gate
outcome) side by side; a human (or Claude Code, per
`evaluation/CLAUDE.md`'s "Arbitrating below-gate dnb-toc-only books") reads
the disagreement, opens the actual TOC page images when the text diff alone
doesn't settle it, and hand-writes the final `.expected.json`. A book that
turns out genuinely unrecoverable (both models hallucinate, the scan is too
degraded to read even directly) gets recorded in `arbitration-rejected.json`
via `arbitrate_dnb_toc.py reject <key> "<reason>"` instead of resurfacing on
every run.

See `evaluation/README.md`'s "Building dnb-toc-only ground truth" for the
eval-tier (fully hand-transcribed, held-out) sample and the bulk-tier
spot-check procedure, and `RESULTS.md`'s "dnb-toc-only ground truth:
two-vision-model gate" section for current coverage numbers, observed
failure modes, and the KISSKI rate-limit characteristics of running this at
scale.
