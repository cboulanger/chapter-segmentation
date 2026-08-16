# Evaluation scripts reference

One-line description plus a full `--help` dump for every script in this
directory, alphabetically. For *why*/*when* to run each one, see
`evaluation/README.md` and `evaluation/CLAUDE.md` — this page exists so the
exact flag names/defaults don't have to be re-derived from source every time;
it's a dump of `--help`, not a workflow guide. Regenerate an entry by running
`uv run python evaluation/scripts/<name>.py --help` (or `uv run review --help`
/ `uv run review-stop --help` for `review_app.py`) whenever that script's
arguments change.

Four files in this directory are helper modules, not standalone scripts (no
`argparse`, nothing to run directly): `ground_truth_helper.py` is a script
itself but also somewhat unusual (see below); `layout_features.py`,
`layout_labels.py`, and `pdfalto_runner.py` are pure library code imported by
`evaluate_layout_toc_classifier.py` and `build_crossref_gt_ground_truth.py`.
`__init__.py` is an empty package marker.

## `add_toc_ground_truth.py`

Retrofits existing `.expected.json` files with a `"toc"` field, using the same
structural TOC-page detection `ground_truth_helper.py` already uses to
exclude TOC pages from chapter-start search.

```
usage: add_toc_ground_truth.py [-h] [--force] [--corpus NAME [NAME ...]]

options:
  -h, --help            show this help message and exit
  --force               Re-run books that already have a toc field
  --corpus NAME [NAME ...]
                        Corpus/corpora to process, e.g. 'pending'. Defaults to
                        ['open-access', 'copyrighted-scans'].
```

## `arbitrate_dnb_toc.py`

Surfaces dnb-toc-only books whose two vision-model TOC extractions didn't
clear `generate_dnb_toc_ground_truth.py`'s agreement gate, so a Claude Code
session can arbitrate the conflict directly (design spec
`docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`). Reports
and records rejections only -- never writes `.expected.json` itself; see
`evaluation/CLAUDE.md`'s "Arbitrating below-gate dnb-toc-only books" for the
full workflow.

```
usage: arbitrate_dnb_toc.py [-h] {list,reject} ...

Surfaces dnb-toc-only books whose two vision-model TOC extractions didn't
clear generate_dnb_toc_ground_truth.py's agreement gate, so a Claude Code
session can arbitrate the conflict directly -- see design spec
docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md. This script
only REPORTS and records rejections; it never decides. The arbitrator reads a
book's report, opens the PDF's actual TOC pages via the Read tool when the
text alone doesn't settle it, then either writes evaluation/corpus/dnb-toc-
only/<key>.expected.json directly (same schema as a passing book, "verified":
true) or runs this script's `reject` subcommand to permanently record the book
as unrecoverable.

positional arguments:
  {list,reject}
    list         List books needing arbitration (default)
    reject       Permanently mark a book as unrecoverable

options:
  -h, --help     show this help message and exit
```

The `reject` subcommand:

```
usage: arbitrate_dnb_toc.py reject [-h] key reason

positional arguments:
  key
  reason

options:
  -h, --help  show this help message and exit
```

## `build_crossref_gt_ground_truth.py`

One-time reconciliation: turns `evaluation/crossref_gt/` (Crossref-sourced
book-chapter metadata) into real `evaluation/corpus/pending/` ground truth.

```
usage: build_crossref_gt_ground_truth.py [-h] [--dry-run]
                                         [--pdfalto-bin PDFALTO_BIN]
                                         [--novelty-percentile NOVELTY_PERCENTILE]

options:
  -h, --help            show this help message and exit
  --dry-run             Report what would happen without writing files
  --pdfalto-bin PDFALTO_BIN
                        Path to the pdfalto binary (see pdfalto_runner.py)
  --novelty-percentile NOVELTY_PERCENTILE
                        Percentile of the existing open-access chapter-first
                        corpus's own leave-one-out nearest-neighbor distances
                        used as the novelty threshold. Default: 90.0.
```

## `clean_scanned_pdf.py`

Cleans a badly-scanned PDF (black scanner background, stray hand/fingers,
skew) via unpaper's auto page-content detection, then optionally re-OCRs the
result.

```
usage: clean_scanned_pdf.py [-h] [--dpi DPI] [--start-page START_PAGE]
                            [--end-page END_PAGE] [--ocr-lang OCR_LANG]
                            [--no-normalize-page-size] [--page-size PAGE_SIZE]
                            [--color-mode {grayscale,monochrome,color}]
                            [--optimize {1,2,3}] [--jbig2-lossy]
                            [--pdftoppm-bin PDFTOPPM_BIN]
                            [--unpaper-bin UNPAPER_BIN]
                            [--img2pdf-bin IMG2PDF_BIN]
                            [--magick-bin MAGICK_BIN]
                            [--ocrmypdf-bin OCRMYPDF_BIN] [--keep-workdir]
                            input_pdf output_pdf

positional arguments:
  input_pdf
  output_pdf

options:
  -h, --help            show this help message and exit
  --dpi DPI             Rasterization resolution (default: 300, good for re-
                        OCR).
  --start-page START_PAGE
                        1-based first page to process (default: 1).
  --end-page END_PAGE   1-based last page to process (default: last page).
  --ocr-lang OCR_LANG   Re-OCR the cleaned pages with ocrmypdf --force-ocr
                        using this tesseract language code (e.g. 'deu'). Omit
                        to skip re-OCR and produce an image-only PDF (no text
                        layer).
  --no-normalize-page-size
                        Skip normalizing to a shared page size -- leave every
                        page's PDF page size matching its own unpaper-detected
                        content size.
  --page-size PAGE_SIZE
                        Target page size to normalize to (ignored if --no-
                        normalize-page-size is given): 'auto' (default) uses
                        the max width/height actually seen across the
                        processed pages, so no page ever needs shrinking; a
                        standard name ('a5', 'a4', 'a3', 'letter', 'legal');
                        or an explicit WIDTHxHEIGHT with mm/cm/in units (e.g.
                        '148mmx210mm'). Pages larger than the target in either
                        dimension are shrunk to fit it first (preserving
                        aspect ratio, via ImageMagick); every page is then
                        centered on the shared target size unscaled.
  --color-mode {grayscale,monochrome,color}
                        Rasterization color mode (default: grayscale -- a
                        black-and-white book scan doesn't need full color's 3x
                        file size). 'monochrome' renders 1-bit black/white
                        (smaller still, but can lose antialiasing detail that
                        helps OCR). 'color' keeps the original RGB.
  --optimize {1,2,3}    Post-OCR optimization level, passed to ocrmypdf's -O
                        (only takes effect when --ocr-lang is given, since
                        that's what runs ocrmypdf): 1 (default) applies only
                        safe lossless recompression -- already using JBIG2 for
                        monochrome pages if jbig2enc is installed. 2 and 3
                        additionally allow lossy JPEG recompression of
                        grayscale/color images (smaller, minor quality loss);
                        irrelevant for monochrome pages on their own -- pair
                        with --jbig2-lossy for smaller monochrome output too.
  --jbig2-lossy         Allow ocrmypdf's JBIG2 encoder to use lossy symbol
                        substitution for monochrome images -- meaningfully
                        smaller than the lossless JBIG2 --optimize already
                        uses, but can occasionally substitute a similar-
                        looking glyph for another. Only takes effect when
                        --ocr-lang is given.
  --pdftoppm-bin PDFTOPPM_BIN
  --unpaper-bin UNPAPER_BIN
  --img2pdf-bin IMG2PDF_BIN
  --magick-bin MAGICK_BIN
  --ocrmypdf-bin OCRMYPDF_BIN
  --keep-workdir        Don't delete the temporary per-page working directory;
                        print its path for inspection.
```

Requires `pdftoppm`+`img2pdf` (poppler), `unpaper`, ImageMagick's `magick`,
and `ocrmypdf` on `PATH` (or the matching `--<tool>-bin` flag /
`<TOOL>_BIN` env var) — `brew install poppler unpaper imagemagick ocrmypdf`.

## `discover_crossref_candidates.py`

Discovers new open-access edited-volume candidates for
`evaluation/crossref_gt/manifest.json`, seeded from Crossref publishers
already represented there.

```
usage: discover_crossref_candidates.py [-h] [--dry-run]
                                       [--max-per-language MAX_PER_LANGUAGE]
                                       [--contact-email CONTACT_EMAIL]

options:
  -h, --help            show this help message and exit
  --dry-run             Report what would be added without writing
                        manifest.json
  --max-per-language MAX_PER_LANGUAGE
                        Cap on how many new candidates of one language get
                        appended per run, so one prolific publisher's catalog
                        can't monopolize a run. Default: 5.
  --contact-email CONTACT_EMAIL
                        Crossref/Unpaywall polite-pool contact email
```

## `evaluate_chapter_segmentation_strategies.py`

Runs every evaluation corpus through `analyze_attachment_with_strategies`
(outline read + Crossref-by-ISBN lookup strategies active, not the
pure-heuristic `analyze_attachment`) and prints the same precision/recall
table format `tests/test_segmentation_accuracy.py` uses, grouped by corpus,
plus per-book `strategies_used` diagnostics. Not a pytest test — makes real
(free, cached) Crossref API calls per book.

```
usage: evaluate_chapter_segmentation_strategies.py [-h] [--no-crossref]

options:
  -h, --help     show this help message and exit
  --no-crossref  Disable the Crossref lookup strategy
```

## `evaluate_layout_toc_classifier.py`

Pilot: leave-one-book-out evaluation of a layout-geometry TOC/chapter-first-
page classifier.

```
usage: evaluate_layout_toc_classifier.py [-h] [--pdfalto-bin PDFALTO_BIN]
                                         [--recall-target RECALL_TARGET]
                                         [--chapter-first-recall-tolerance CHAPTER_FIRST_RECALL_TOLERANCE]
                                         [--corpora CORPORA]
                                         [--scan-noise-augment]
                                         [--candidate-fraction-cap CANDIDATE_FRACTION_CAP]
                                         [--save-results]

Pilot: leave-one-book-out evaluation of a layout-geometry TOC/ chapter-first-
page classifier. See docs/superpowers/specs/2026-08-10-layout-based-toc-
classifier-pilot-design.md.

options:
  -h, --help            show this help message and exit
  --pdfalto-bin PDFALTO_BIN
  --recall-target RECALL_TARGET
                        Legacy selection strategy: per-fold threshold-
                        calibration target -- how much recall on training
                        positives to require before accepting a page as a
                        candidate, applied as one absolute threshold to every
                        held-out book. Passing this explicitly switches
                        selection to this strategy and --candidate-fraction-
                        cap is ignored -- omit both to use the default
                        --candidate-fraction-cap strategy instead. (Historical
                        default: 0.9.)
  --chapter-first-recall-tolerance CHAPTER_FIRST_RECALL_TOLERANCE
                        Per-book chapter_first recall required to count that
                        book as 'fully recalled' in this script's own report.
                        Does not affect which candidate pages are produced.
                        Default: 0.9.
  --corpora CORPORA     Comma-separated list of evaluation/corpus/
                        subdirectories to load books from (e.g. 'open-
                        access,pending' to include unreviewed candidates).
                        Default: open-access,copyrighted-scans.
  --scan-noise-augment  Augment each open-access book's training rows with a
                        scan-noise-perturbed copy of its ALTO XML (cached as
                        <key>.aug.alto.xml). Augmented rows are only ever used
                        for training, never evaluated.
  --candidate-fraction-cap CANDIDATE_FRACTION_CAP
                        Default selection strategy: document-relative
                        candidate selection -- rank each held-out book's own
                        pages by max(prob_toc, prob_chapter_first) and take
                        the top candidate_fraction_cap share as candidates,
                        instead of a threshold calibrated once from training-
                        positive quantiles. Ignored if --recall-target is
                        explicitly set. Default: 0.15. See
                        select_candidates_by_document_budget's docstring.
  --save-results        Write per-book results to
                        evaluation/corpus/<corpus>/classifier-results.json
                        (split by each book's own corpus), for
                        generate_report.py to fold into the published report.
                        Default: off (stdout-only, current behavior).
```

## `fetch_crossref_gt_corpus.py`

Fetches the Crossref-sourced ground-truth corpus (`evaluation/crossref_gt/`).

```
usage: fetch_crossref_gt_corpus.py [-h] [--corpus-dir CORPUS_DIR] [--force]
                                   [--force-metadata]
                                   [--contact-email CONTACT_EMAIL]

options:
  -h, --help            show this help message and exit
  --corpus-dir CORPUS_DIR
                        Directory containing manifest.json
  --force               Refetch PDFs and Crossref metadata even if present
  --force-metadata      Refetch only the Crossref metadata (not PDFs) even if
                        present -- e.g. after a select-field change
  --contact-email CONTACT_EMAIL
                        Crossref polite-pool contact email
```

## `fetch_dnb_toc_corpus.py`

Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API into evaluation/corpus/dnb-toc-only/.

```
usage: fetch_dnb_toc_corpus.py [-h] (--from-dump | --isbns-file ISBNS_FILE)
                               [--dump-url DUMP_URL] [--limit LIMIT]
                               [--rate-limit-seconds RATE_LIMIT_SECONDS]
                               [--max-retries MAX_RETRIES]

Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API (lobid.org/resources) into evaluation/corpus/dnb-toc-only/ -- see
docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md.

options:
  -h, --help            show this help message and exit
  --from-dump           Scan the full lobid-resources JSON-Lines dump for
                        matching records (hours-long; see module docstring)
  --isbns-file ISBNS_FILE
                        Path to a text file of ISBNs (one per line, '#'
                        comments allowed) to look up individually
  --dump-url DUMP_URL   lobid-resources dump URL for --from-dump (default:
                        https://lobid.org/download/dumps/lobid-
                        resources/latestLobidResources.jsonl.gz)
  --limit LIMIT         Stop after acquiring this many new books
  --rate-limit-seconds RATE_LIMIT_SECONDS
                        Delay after each TOC PDF download, to stay polite to
                        DNB's servers (default: 1.0)
  --max-retries MAX_RETRIES
                        For --from-dump: how many times to reconnect and
                        rescan after a dropped connection before giving up
                        (default: 5)
```

## `fetch_evaluation_pdfs.py`

Downloads the open-access chapter-segmentation evaluation PDFs on demand
(every `"oa": true` manifest entry not already present locally). Non-OA
entries are never auto-downloaded — the script prints the DOI and target
path instead.

```
usage: fetch_evaluation_pdfs.py [-h] [--force] [--corpus CORPUS]

options:
  -h, --help       show this help message and exit
  --force          Re-download even if the PDF already exists
  --corpus CORPUS  Only fetch this corpus (default: every corpus under
                   evaluation/corpus/)
```

## `generate_dnb_toc_ground_truth.py`

Generates bulk-tier `dnb-toc-only` ground truth by sending each book's page
images to two independent vision-capable KISSKI models and writing
`.expected.json` only when they agree well enough -- see
`evaluation/README.md`'s "Building dnb-toc-only ground truth".

```
usage: generate_dnb_toc_ground_truth.py [-h] [--limit LIMIT]
                                        [--concurrency CONCURRENCY]
                                        [--spot-check N]

Generates bulk-tier structured ground truth for dnb-toc-only (design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md, which
supersedes the two-text-extractor design in
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md).
For every manifest book not held out in eval_tier_ids.json (see
select_dnb_toc_eval_sample.py and evaluation/README.md's "Building dnb-toc-
only ground truth"), sends the book's page images to two independent vision-
capable KISSKI models (evaluation.dnb_toc_vision.vision_extract_toc_entries)
and writes <id>.expected.json with "verified": false only when they agree well
enough (evaluation.dnb_toc_matching.gate_book, >=0.90 whole-book agreement).
Books that don't clear the gate are skipped and reported, not partially
written.

options:
  -h, --help            show this help message and exit
  --limit LIMIT         Process at most this many books (smoke-test
                        convenience)
  --concurrency CONCURRENCY
                        How many books to process concurrently (default: 4)
  --spot-check N        Instead of generating, sample N passing bulk-tier
                        books and walk through a visual Accept/Reject check
```

## `generate_public_evaluation_cache.py`

Generates each corpus's `public-cache/` (a git-trackable, safe-to-distribute
snapshot of page text — verbatim for OA books, redacted for everyone else)
plus a resolved outline-strategy candidate snapshot per book. Run by a
maintainer who has the real books locally.

```
usage: generate_public_evaluation_cache.py [-h] [--book BOOK]
                                           [--corpus CORPUS] [--no-verify]
                                           [--skip-redaction]

options:
  -h, --help        show this help message and exit
  --book BOOK       Only regenerate this manifest key (filename stem)
  --corpus CORPUS   Only regenerate this corpus (default: every corpus under
                    evaluation/corpus/)
  --no-verify       Skip the exact-boundary-match check
  --skip-redaction  Cache non-OA books' REAL text verbatim, unredacted -- see
                    module docstring. Do NOT commit the resulting public-
                    cache/ files.
```

`--skip-redaction` writes real copyrighted prose into the normally
git-tracked `public-cache/` directory — see `evaluation/CLAUDE.md`'s
"`--skip-redaction` is a separate, faster escape valve" section before using
it; the caller is responsible for gitignoring the affected files.

## `ground_truth_helper.py`

Drafts a chapter-segmentation ground-truth `.expected.json` from a real PDF
and a hand-transcribed table of contents (JSON: `[{title, authors, skip?}]`
in reading order). A starting point, not an oracle — every entry needs
verifying by hand (see `evaluation/CLAUDE.md`'s "Step 3").

```
usage: ground_truth_helper.py [-h] --pdf PDF --toc TOC [--output OUTPUT]

options:
  -h, --help       show this help message and exit
  --pdf PDF        Path to the real PDF
  --toc TOC        JSON file: [{title, authors, skip?}] in reading order
  --output OUTPUT  Write draft JSON here instead of stdout
```

## `measure_dnb_scan_noise_stats.py`

Measures real font-size contrast/dispersion from the dnb-toc-only corpus's
ALTO to calibrate alto_scan_noise.py's synthetic constants.

```
usage: measure_dnb_scan_noise_stats.py [-h] [--corpus CORPUS]
                                       [--pdfalto-bin PDFALTO_BIN]

Measures real font-size contrast and per-line dispersion from the
dnb-toc-only corpus's ALTO output, to calibrate alto_scan_noise.py's
hand-picked synthetic constants (_CONTRAST_ALPHA, _FONT_JITTER) against
real scanned data -- see
docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md
section 3. Every page in this corpus is a confirmed TOC page by
construction (DNB only digitizes the TOC itself), so no per-page
labeling step is needed.

options:
  -h, --help            show this help message and exit
  --corpus CORPUS       Corpus to measure (default: dnb-toc-only)
  --pdfalto-bin PDFALTO_BIN
                        Path to the pdfalto binary (see pdfalto_runner.py)
```

## `ocr_evaluation_pdfs.py`

OCRs the evaluation books whose text layer is absent or degenerate into each
corpus's gitignored `.ocr-cache/` (content-hash keyed). Already-cached books
are skipped instantly.

```
usage: ocr_evaluation_pdfs.py [-h] [--ocr-backend {kreuzberg,tesseract}]
                              [--kreuzberg-url KREUZBERG_URL]
                              [--corpus CORPUS]

options:
  -h, --help            show this help message and exit
  --ocr-backend {kreuzberg,tesseract}
  --kreuzberg-url KREUZBERG_URL
  --corpus CORPUS       Only OCR this corpus (default: every corpus under
                        evaluation/corpus/)
```

Uses `KreuzbergOcrBackend` by default (requires the Kreuzberg sidecar
container running — see root `CLAUDE.md`'s Live Server section); pass
`--ocr-backend tesseract` for the local-binary path instead.

## `promote_pending_book.py`

Promotes one or more `evaluation/corpus/pending/` books (already carrying a
hand-verified `.expected.json`) into a real corpus, re-running the bounds/
overlap sanity check as a gate.

```
usage: promote_pending_book.py [-h] --corpus {open-access,copyrighted-scans}
                               [--contact-email CONTACT_EMAIL] [--dry-run]
                               isbn [isbn ...]

positional arguments:
  isbn                  One or more pending/ ISBNs (matching <isbn>.pdf) to
                        promote

options:
  -h, --help            show this help message and exit
  --corpus {open-access,copyrighted-scans}
                        Target corpus
  --contact-email CONTACT_EMAIL
                        Crossref/Unpaywall polite-pool contact email
  --dry-run             Report what would happen without moving files or
                        writing manifests
```

## `review_app.py`

Starts a local static server for the repo root and opens the ground-truth
review app (`evaluation/app/`) in the browser, pointed at one corpus. macOS
only. Not run directly — exposed as two console scripts (see
`pyproject.toml`'s `[project.scripts]`): `review` (`main()`) and
`review-stop` (`stop()`).

```
$ uv run review --help
usage: review [-h] [--index INDEX] [--port PORT] corpus

positional arguments:
  corpus         corpus directory name under evaluation/corpus/, e.g. open-
                 access

options:
  -h, --help     show this help message and exit
  --index INDEX  starting book index (default: 0)
  --port PORT    local server port (default: 8743)
```

```
$ uv run review-stop --help
usage: review-stop [-h] [--port PORT]

options:
  -h, --help   show this help message and exit
  --port PORT  port to stop (default: 8743)
```
