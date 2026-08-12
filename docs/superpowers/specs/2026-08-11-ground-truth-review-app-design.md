# Ground-truth review app design

## 1. Goal

Ground truth (`<isbn>.expected.json`) for `evaluation/corpus/*` is currently
verified by hand (open the PDF, jump to `pdf_start_index`, eyeball it — see
`evaluation/CLAUDE.md` Step 3). There's no fast way to re-spot-check an
entire corpus after the fact, e.g. after the `toc_start_index`/
`toc_end_index` fields were added to every `.expected.json` in the
`pdf-layout-toc-classifier` worktree (assumed already merged for this
design).

This adds a small, local-only, backend-free web app that lets a human
page through a corpus's books, see a thumbnail of what the ground truth
*claims* is the TOC and the opening page of every chapter, and mark each
book Accept/Reject. At the end it downloads a text file listing the
rejected books' ISBNs, for follow-up correction.

## 2. Non-goals

- No editing of `.expected.json` from the app — this is a read-only
  correctness *check*, not a ground-truth editor.
- No server/API. The app is static files; the only "backend" is a plain
  static file server for local use (`python3 -m http.server`), because
  `fetch()` cannot read `file://` URLs reliably.
- No auto-discovery of corpora (no directory-listing parsing) — the corpus
  name is a required URL parameter.
- No build step, no framework, no npm dependency in this otherwise
  Python-only repo.

## 3. Location and files

```
evaluation/app/
  index.html   # shell: corpus/progress header, thumbnail grid, controls, lightbox
  app.js       # all logic (ES module)
  styles.css   # thumbnail grid, TOC vs. chapter background colors, lightbox
  README.md    # how to run it
```

PDF.js is loaded from a CDN `<script>` tag in `index.html` (confirmed
acceptable — this is a local dev tool, network access is assumed
available when running it).

## 4. Running it

```bash
cd /Users/cboulanger/Code/chapter-segmentation
python3 -m http.server 8000
# open:
http://localhost:8000/evaluation/app/index.html?corpus=open-access
```

Serving from the repo root means `evaluation/app/app.js` can fetch
`../corpus/<corpus>/manifest.json`, `<isbn>.expected.json`, and
`<isbn>.pdf` with plain relative paths.

## 5. URL and persisted state

- `?corpus=<name>` (required) — selects `evaluation/corpus/<name>/`.
- `?index=<N>` (optional, default `0`) — position within
  `manifest.json`'s `books` array. Updated via `history.replaceState` as
  the user advances/goes back, so the address bar always reflects the
  current book and the tab can be closed and reopened from the same URL.
- `localStorage["gt-review:<corpus>:decisions"]` — `{ [isbn]:
  "accepted" | "rejected" }`, built up as the user makes decisions. This
  is what survives if the browser is closed mid-review (the URL alone
  only remembers *position*, not prior verdicts) and what the final
  rejected-list download is built from. Not cleared automatically —
  re-reviewing a book just overwrites its entry.

`isbn` throughout = the manifest entry's `filename` with the `.pdf`
extension stripped.

## 6. Per-book screen

For `manifest.books[index]`:

1. Fetch `<isbn>.expected.json`. 404 (or the PDF 404s) → render an inline
   "not available locally" message for this book with a single `Skip`
   button (advances `index` without touching the decisions map — this is
   expected for corpus PDFs that aren't downloaded on this machine, not a
   verdict on the ground truth).
2. Load `<isbn>.pdf` via `pdfjsLib.getDocument`.
3. Render thumbnails into two visually distinct sections (different
   background colors, per explicit request, so TOC and chapter pages are
   never confused at a glance):
   - **TOC** (only if `expected.toc` is present): one canvas per page in
     `[toc_start_index, toc_end_index]` inclusive, light-blue background,
     labeled `TOC — p.<pdfIndex>`.
   - **Chapters**: one canvas per entry in `expected.chapters`, at
     `pdf_start_index`, light-green background, labeled with the
     chapter's `title`, `authors`, and `citation_pages`.
4. Each canvas renders at a fixed target width (~600px, computed from the
   page's natural viewport so aspect ratio is preserved) — big enough to
   read titles/running heads. Clicking a thumbnail re-renders that same
   page at a larger scale into a full-screen lightbox overlay for closer
   inspection; clicking the overlay (or Esc) closes it.
5. Header shows `Book <index+1> of <total> — <isbn> — <title>`.
6. Controls: `Reject`, `Accept`, `Prev` (disabled at index 0). `Accept`/
   `Reject` write the decision for the current isbn into the decisions
   map and advance `index` by 1 (see §7 for what happens at the end).
   If the current isbn already has a stored decision (e.g. the user came
   back via `Prev`), that button is visually highlighted as "current
   choice" on render.

## 7. Completion

When `Accept`/`Reject` is pressed on the last book (`index === total -
1`), instead of advancing to an out-of-range index, show a "Review
complete" screen:

- Collect every isbn in the decisions map with value `"rejected"`.
- Build a newline-joined text blob and trigger a download named
  `<corpus>-rejected.txt` via `Blob` + a synthetic `<a download>` click
  (fired from within the button handler, so it's a trusted user gesture
  and isn't blocked as a popup).
- Show the count of accepted/rejected/total plus a manual "Download
  again" button, in case the automatic download was blocked or missed.

If the app is loaded with `index` already `>= total` (e.g. a stale
bookmarked URL from a since-shrunk manifest), it goes straight to this
completion screen.

## 8. Error handling

- Missing/invalid `corpus` param → static message: "Add `?corpus=<name>`
  to the URL, e.g. `?corpus=open-access`." No attempt to enumerate valid
  corpus names (would require directory listing).
- `manifest.json` fetch failure (bad corpus name) → similar static error
  message with the attempted path, so it's obvious what to fix.
- Per-book PDF/expected.json 404 → handled per §6.1 (Skip), not a
  fatal error for the whole app.

## 9. Testing

This is a manual visual tool with no business logic worth unit-testing
beyond "does it run." Verification is: run it against `open-access`
locally, page through a handful of books including at least one with a
`toc` field and one without, confirm thumbnails match the ground truth,
confirm Accept/Reject/Prev/resume-from-URL and the final download all
work as described.
