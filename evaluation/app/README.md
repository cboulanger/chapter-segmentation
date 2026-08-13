# Ground-truth review app

Local, backend-free tool for spot-checking `.expected.json` ground truth
against the actual PDF pages. See
`docs/superpowers/specs/2026-08-11-ground-truth-review-app-design.md` for
the full design.

## Run it

macOS: one command starts the server and opens the app in your browser:

    uv run review open-access

Replace `open-access` with any corpus directory name under
`evaluation/corpus/`. Add `--index N` to start at a specific book, or
`--port` to use a port other than the default (8743, chosen to avoid
colliding with commonly-used ports like 8000/8080). Press Ctrl+C to stop
the server.

On other platforms (or if you'd rather do it by hand), from the repo
root:

    python3 -m http.server 8743

Then open:

    http://localhost:8743/evaluation/app/index.html?corpus=open-access

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
- A book you've already accepted shows a green header with a checkmark
  the next time it loads, so a re-review pass makes it obvious which
  books still need attention.
- `Open in VS Code` deep-links to the book's `.expected.json` via
  `vscode://file/...`, using the local filesystem path the `review`
  command passes in as `?repoRoot=`. macOS-only, and only appears when
  `repoRoot` is present in the URL (i.e. when launched via `uv run
  review`, not the manual `python3 -m http.server` route).
- Your position (`index`) is kept in the URL so you can resume later;
  verdicts are kept in this browser's `localStorage`, scoped per corpus
  (`gt-review:<corpus>:decisions`).
- After the last book, `Download rejected list` saves the rejected ISBNs
  as `<corpus>-rejected.txt`, one per line. `Clear rejected list` removes
  rejected verdicts from storage (keeping accepted ones) so you can
  re-review just the books you fixed. `Review rejected` (shown only when
  there are any) jumps back to the first rejected book with `Rejected
  only` pre-checked. `Clear all corpus data` (after a confirmation
  prompt) wipes every accepted and rejected verdict for the current
  corpus, resetting it to a fresh review.
- If a book's PDF or `.expected.json` isn't present on disk, a `Skip`
  button advances past it without recording a verdict.
- `Rejected only` filters navigation (`Accept`/`Reject`/`Prev`/`Skip`) to
  books currently marked `rejected`, jumping straight to the next/previous
  one and skipping everything else — handy for a follow-up pass after
  fixing some ground truth. Unchecking it returns to the normal
  book-by-book order.
- `Skip accepted` filters navigation to books that are either `rejected`
  or have no verdict yet, skipping over ones you've already accepted —
  handy for getting through the remaining unreviewed books without
  re-seeing ones you've signed off on. If `Rejected only` is also
  checked, it takes precedence (rejected is a subset of not-accepted).

## Tests

Pure logic (URL parsing, decision-list building, page-range math) has
unit tests, no browser needed:

    node --test evaluation/app/lib.test.js
