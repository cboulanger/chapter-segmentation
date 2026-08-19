# chapter-segmentation

Standalone PDF chapter-boundary detection: given a book's page text (or raw
PDF bytes), finds where each chapter starts and ends using a table-of-contents
heuristic, an optional PDF-outline/Crossref/catalog metadata fusion pipeline,
and an optional LLM fallback for irregular layouts.

Extracted from [zotero-rag](https://github.com/cboulanger/zotero-rag), where
it originated as part of a Zotero chapter-linking feature — see that
project's `docs/superpowers/specs/2026-08-06-chapter-segmentation-extraction-design.md`
for the extraction rationale.

## Install

```bash
pip install "chapter-segmentation[tesseract]"   # local OCR, no container
# or
pip install "chapter-segmentation[kreuzberg]"   # OCR via a Kreuzberg sidecar
```

## Standalone CLI

```bash
chapter-segmentation analyze mybook.pdf
```

Requires the `tesseract` binary on `PATH` if the PDF needs OCR (see
"OCR backends" below) — `apt-get install tesseract-ocr tesseract-ocr-deu
tesseract-ocr-fra tesseract-ocr-spa` (Debian/Ubuntu) or `brew install
tesseract tesseract-lang` (macOS).

## OCR backends

Two `OcrBackend` implementations ship in `chapter_segmentation.ocr_backends`:

- `TesseractOcrBackend` (`[tesseract]` extra) — local binary, no container, no network. The CLI's default.
- `KreuzbergOcrBackend` (`[kreuzberg]` extra) — calls a running Kreuzberg sidecar's HTTP API.

Implement `chapter_segmentation.ocr.OcrBackend` yourself for anything else.

## Evaluation

See `evaluation/README.md` for the ground-truth corpus, how to run the
accuracy suite, and how to add a new evaluation book. Current numbers:
https://cboulanger.github.io/chapter-segmentation/ (auto-published, no
hand-written analysis) and `evaluation/RESULTS.md` (hand-maintained,
includes mechanism/root-cause notes).

### Ongoing experiments

Candidate improvements that haven't (yet) been folded into the main
pipeline -- e.g. a layout-based TOC/chapter-page classifier, a small
locally-runnable TOC-extraction model, and the ground-truth-generation
pipeline for a new evaluation corpus -- are tracked as living documents in
`evaluation/experiments/`, see that directory's README for an index.

## Development

```bash
uv sync
uv run pytest
```
