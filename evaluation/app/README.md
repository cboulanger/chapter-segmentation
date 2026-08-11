# Ground-truth review app

Local, backend-free tool for spot-checking `.expected.json` ground truth
against the actual PDF pages. See
`docs/superpowers/specs/2026-08-11-ground-truth-review-app-design.md` for
the full design.

## Run it

From the repo root:

    python3 -m http.server 8000

Then open:

    http://localhost:8000/evaluation/app/index.html?corpus=open-access

Replace `open-access` with any corpus directory name under
`evaluation/corpus/`.

## Usage

- Each screen shows one book: its ground-truth TOC pages (blue
  background, only shown if the book's `.expected.json` has a `toc`
  field) and the first page of every chapter per `pdf_start_index` (green
  background).
- Click a thumbnail to see it larger; click it again (or press Escape) to
  close.
- `Accept`/`Reject` records a verdict and moves to the next book. `Prev`
  goes back to change a verdict — the button matching your last verdict
  for that book is outlined.
- Your position (`index`) is kept in the URL so you can resume later;
  verdicts are kept in this browser's `localStorage`, scoped per corpus
  (`gt-review:<corpus>:decisions`).
- After the last book, a rejected-ISBN list downloads automatically as
  `<corpus>-rejected.txt`, one isbn per line.
- If a book's PDF or `.expected.json` isn't present on disk, a `Skip`
  button advances past it without recording a verdict.

## Tests

Pure logic (URL parsing, decision-list building, page-range math) has
unit tests, no browser needed:

    node --test evaluation/app/lib.test.js
