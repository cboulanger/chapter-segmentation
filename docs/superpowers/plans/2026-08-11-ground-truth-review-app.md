# Ground-Truth Review App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evaluation/app/`, a static, backend-free web app that pages through a corpus's books, renders thumbnails of the ground-truth TOC pages and each chapter's opening page, and lets a human Accept/Reject each book, ending in a downloaded `<corpus>-rejected.txt`.

**Architecture:** Two ES modules — `lib.js` (pure, unit-tested logic: URL/param parsing, ISBN derivation, page-range math, decision-list building) and `app.js` (DOM orchestration: fetches, PDF.js rendering, event wiring) — plus a static `index.html`/`styles.css` shell. No build step, no framework.

**Tech Stack:** Vanilla JS (ES modules), PDF.js loaded from the cdnjs CDN, `node --test` for the pure-logic unit tests, `python3 -m http.server` for local serving.

Design reference: `docs/superpowers/specs/2026-08-11-ground-truth-review-app-design.md`.

---

### Task 1: Feature branch and test fixture

**Files:**
- Modify: `evaluation/corpus/open-access/9781771993661.expected.json`

The design was clarified against a newer `.expected.json` schema (adds a
top-level `"toc": {"toc_start_index": N, "toc_end_index": M}` field) that
exists today only in the `pdf-layout-toc-classifier` worktree at
`.claude/worktrees/pdf-layout-toc-classifier/`. That worktree isn't being
merged as part of this plan — we only need one book with a real `toc`
field on disk to manually verify the TOC-thumbnail code path later, and
one book without it (every other book in `evaluation/corpus/open-access/`
already lacks the field, so that path needs no setup).

- [ ] **Step 1: Create and switch to a feature branch**

```bash
git checkout -b ground-truth-review-app
```

Expected: `Switched to a new branch 'ground-truth-review-app'`.

- [ ] **Step 2: Copy the `toc`-enabled fixture from the worktree**

```bash
cp .claude/worktrees/pdf-layout-toc-classifier/evaluation/corpus/open-access/9781771993661.expected.json \
   evaluation/corpus/open-access/9781771993661.expected.json
```

- [ ] **Step 3: Verify the copied file has the `toc` field**

```bash
python3 -c "import json; d=json.load(open('evaluation/corpus/open-access/9781771993661.expected.json')); print(d['toc'])"
```

Expected: `{'toc_start_index': 4, 'toc_end_index': 5}`

- [ ] **Step 4: Commit**

```bash
git add evaluation/corpus/open-access/9781771993661.expected.json
git commit -m "test: import toc-enabled expected.json fixture for review-app testing"
```

---

### Task 2: Pure logic module (`lib.js`)

**Files:**
- Create: `evaluation/app/lib.js`
- Create: `evaluation/app/lib.test.js`

This is the only part of the app with automated tests — everything else
is DOM/canvas orchestration verified manually (see design spec §9).

- [ ] **Step 1: Write the test file**

Create `evaluation/app/lib.test.js`:

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseParams,
  isbnFromFilename,
  decisionsStorageKey,
  rejectedListText,
  tocPageRange,
  computeScale,
  isComplete,
  normalizeIndex,
} from './lib.js';

test('parseParams reads corpus and index', () => {
  assert.deepEqual(parseParams('?corpus=open-access&index=5'), { corpus: 'open-access', index: 5 });
});

test('parseParams defaults index to 0 when absent', () => {
  assert.deepEqual(parseParams('?corpus=open-access'), { corpus: 'open-access', index: 0 });
});

test('parseParams returns null corpus when absent', () => {
  assert.deepEqual(parseParams(''), { corpus: null, index: 0 });
});

test('parseParams ignores a non-numeric index', () => {
  assert.deepEqual(parseParams('?corpus=x&index=abc'), { corpus: 'x', index: 0 });
});

test('parseParams ignores a negative index', () => {
  assert.deepEqual(parseParams('?corpus=x&index=-3'), { corpus: 'x', index: 0 });
});

test('isbnFromFilename strips .pdf', () => {
  assert.equal(isbnFromFilename('9783839468937.pdf'), '9783839468937');
});

test('decisionsStorageKey is corpus-scoped', () => {
  assert.equal(decisionsStorageKey('open-access'), 'gt-review:open-access:decisions');
});

test('rejectedListText only includes rejected isbns, sorted', () => {
  const decisions = { b: 'rejected', a: 'accepted', c: 'rejected' };
  assert.equal(rejectedListText(decisions), 'b\nc');
});

test('rejectedListText returns empty string when nothing rejected', () => {
  assert.equal(rejectedListText({ a: 'accepted' }), '');
});

test('tocPageRange expands an inclusive range', () => {
  assert.deepEqual(tocPageRange({ toc_start_index: 4, toc_end_index: 6 }), [4, 5, 6]);
});

test('tocPageRange returns an empty array when toc is absent', () => {
  assert.deepEqual(tocPageRange(undefined), []);
  assert.deepEqual(tocPageRange(null), []);
});

test('computeScale divides target width by natural width', () => {
  assert.equal(computeScale(500, 600), 1.2);
});

test('computeScale falls back to 1 for a non-positive natural width', () => {
  assert.equal(computeScale(0, 600), 1);
  assert.equal(computeScale(-10, 600), 1);
});

test('isComplete is true once index reaches total, and for an empty manifest', () => {
  assert.equal(isComplete(3, 3), true);
  assert.equal(isComplete(2, 3), false);
  assert.equal(isComplete(0, 0), true);
});

test('normalizeIndex clamps a negative or non-finite value to 0', () => {
  assert.equal(normalizeIndex(-1), 0);
  assert.equal(normalizeIndex(Number.NaN), 0);
  assert.equal(normalizeIndex(5), 5);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
node --test evaluation/app/lib.test.js
```

Expected: fails with `Cannot find module '.../evaluation/app/lib.js'`.

- [ ] **Step 3: Write `lib.js`**

Create `evaluation/app/lib.js`:

```js
export function parseParams(search) {
  const params = new URLSearchParams(search);
  const corpus = params.get('corpus');
  const rawIndex = params.get('index');
  const parsedIndex = rawIndex === null ? 0 : Number.parseInt(rawIndex, 10);
  const index = Number.isFinite(parsedIndex) && parsedIndex >= 0 ? parsedIndex : 0;
  return { corpus: corpus || null, index };
}

export function isbnFromFilename(filename) {
  return filename.replace(/\.pdf$/i, '');
}

export function decisionsStorageKey(corpus) {
  return `gt-review:${corpus}:decisions`;
}

export function rejectedListText(decisions) {
  return Object.keys(decisions)
    .filter((isbn) => decisions[isbn] === 'rejected')
    .sort()
    .join('\n');
}

export function tocPageRange(toc) {
  if (!toc || typeof toc.toc_start_index !== 'number' || typeof toc.toc_end_index !== 'number') {
    return [];
  }
  const pages = [];
  for (let i = toc.toc_start_index; i <= toc.toc_end_index; i += 1) {
    pages.push(i);
  }
  return pages;
}

export function computeScale(naturalWidth, targetWidth) {
  if (!naturalWidth || naturalWidth <= 0) return 1;
  return targetWidth / naturalWidth;
}

export function isComplete(index, total) {
  return total <= 0 || index >= total;
}

export function normalizeIndex(rawIndex) {
  return Number.isFinite(rawIndex) && rawIndex >= 0 ? Math.floor(rawIndex) : 0;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
node --test evaluation/app/lib.test.js
```

Expected: `# pass 15`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add evaluation/app/lib.js evaluation/app/lib.test.js
git commit -m "feat: add pure logic module for ground-truth review app"
```

---

### Task 3: Static app shell

**Files:**
- Create: `evaluation/app/index.html`
- Create: `evaluation/app/styles.css`
- Create: `evaluation/app/app.js` (stub)

- [ ] **Step 1: Write `index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Ground-truth review</title>
  <link rel="stylesheet" href="styles.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
</head>
<body>
  <div id="app">Loading…</div>
  <script>
    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  </script>
  <script type="module" src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `styles.css`**

```css
body {
  font-family: system-ui, sans-serif;
  margin: 0;
  padding: 1rem 2rem 4rem;
}

header {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  display: block;
}

.section h2 {
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #444;
}

.section.hidden {
  display: none;
}

.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 2rem;
}

.thumb {
  margin: 0;
  padding: 0.5rem;
  border-radius: 6px;
  max-width: 620px;
}

.thumb canvas {
  display: block;
  max-width: 100%;
  cursor: zoom-in;
  border: 1px solid #ccc;
}

.thumb figcaption {
  font-size: 0.8rem;
  margin-top: 0.4rem;
  max-width: 600px;
}

.toc-thumb {
  background: #eaf4ff;
}

.chapter-thumb {
  background: #eefaf0;
}

.controls {
  position: sticky;
  bottom: 0;
  background: white;
  padding: 1rem 0;
  display: flex;
  gap: 1rem;
  border-top: 1px solid #ddd;
}

.controls button {
  font-size: 1rem;
  padding: 0.6rem 1.2rem;
  cursor: pointer;
}

.accept {
  background: #d7f7dd;
}

.reject {
  background: #ffe0e0;
}

.current-choice {
  outline: 3px solid #333;
}

.error {
  color: #a00;
}

#lightbox {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: zoom-out;
  z-index: 10;
}

#lightbox canvas {
  max-width: 95vw;
  max-height: 95vh;
}
```

- [ ] **Step 3: Write a stub `app.js`**

```js
document.getElementById('app').textContent = 'App scaffold OK';
```

- [ ] **Step 4: Verify the shell loads**

```bash
python3 -m http.server 8000 &
sleep 1
curl -s http://localhost:8000/evaluation/app/index.html | grep -o '<title>.*</title>'
kill %1
```

Expected: `<title>Ground-truth review</title>`. Then open
`http://localhost:8000/evaluation/app/index.html` in a real browser and
confirm the page shows "App scaffold OK" with no console errors (the
PDF.js CDN scripts should load fine even though nothing calls them yet).

- [ ] **Step 5: Commit**

```bash
git add evaluation/app/index.html evaluation/app/styles.css evaluation/app/app.js
git commit -m "feat: add static shell for ground-truth review app"
```

---

### Task 4: App skeleton — data loading, navigation, persistence, completion

**Files:**
- Modify: `evaluation/app/app.js` (replace stub entirely)

This builds the full interactive flow — corpus/manifest loading, per-book
navigation, Accept/Reject/Prev, `localStorage` persistence, URL sync, and
the completion/download screen — using a text placeholder in place of
real thumbnails. Task 5 swaps the placeholder for actual PDF rendering
without touching any of this control flow.

- [ ] **Step 1: Replace `evaluation/app/app.js` with the skeleton**

```js
import {
  parseParams,
  isbnFromFilename,
  decisionsStorageKey,
  rejectedListText,
  isComplete,
  normalizeIndex,
} from './lib.js';

const state = {
  corpus: null,
  index: 0,
  manifest: [],
  decisions: {},
};

function $(id) {
  return document.getElementById(id);
}

function showError(message) {
  $('app').innerHTML = `<p class="error">${message}</p>`;
}

function loadDecisions(corpus) {
  const raw = localStorage.getItem(decisionsStorageKey(corpus));
  return raw ? JSON.parse(raw) : {};
}

function saveDecisions(corpus, decisions) {
  localStorage.setItem(decisionsStorageKey(corpus), JSON.stringify(decisions));
}

function updateUrl(corpus, index) {
  const url = new URL(window.location.href);
  url.searchParams.set('corpus', corpus);
  url.searchParams.set('index', String(index));
  window.history.replaceState({}, '', url);
}

async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

function headerHtml(book, isbn) {
  return `Book ${state.index + 1} of ${state.manifest.length} — ${isbn} — ${book.title}`;
}

function renderSkippable(book, isbn, message) {
  $('app').innerHTML = `
    <header>${headerHtml(book, isbn)}</header>
    <p class="error">${message}</p>
    <div class="controls"><button id="skip">Skip</button></div>
  `;
  $('skip').addEventListener('click', () => {
    state.index += 1;
    render();
  });
}

function highlightDecision(isbn) {
  const decision = state.decisions[isbn];
  if (!decision) return;
  const btn = $(decision === 'accepted' ? 'accept' : 'reject');
  if (btn) btn.classList.add('current-choice');
}

function decide(isbn, verdict) {
  state.decisions[isbn] = verdict;
  saveDecisions(state.corpus, state.decisions);
  state.index += 1;
  render();
}

function goPrev() {
  state.index = Math.max(0, state.index - 1);
  render();
}

function renderBookPlaceholder(book, isbn, expected) {
  $('app').innerHTML = `
    <header>${headerHtml(book, isbn)}</header>
    <p>toc: ${expected.toc ? JSON.stringify(expected.toc) : 'none'}</p>
    <p>${expected.chapters.length} chapters (thumbnails land in a later task)</p>
    <div class="controls">
      <button id="prev" ${state.index === 0 ? 'disabled' : ''}>Prev</button>
      <button id="reject" class="reject">Reject</button>
      <button id="accept" class="accept">Accept</button>
    </div>
  `;
  highlightDecision(isbn);
  $('prev').addEventListener('click', goPrev);
  $('accept').addEventListener('click', () => decide(isbn, 'accepted'));
  $('reject').addEventListener('click', () => decide(isbn, 'rejected'));
}

function downloadRejected(text) {
  const blob = new Blob([text ? `${text}\n` : ''], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${state.corpus}-rejected.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function renderComplete(total) {
  const rejectedText = rejectedListText(state.decisions);
  const rejectedCount = rejectedText ? rejectedText.split('\n').length : 0;
  const acceptedCount = Object.values(state.decisions).filter((v) => v === 'accepted').length;
  $('app').innerHTML = `
    <h1>Review complete</h1>
    <p>${acceptedCount} accepted, ${rejectedCount} rejected, ${total} total.</p>
    <button id="download">Download rejected list</button>
  `;
  $('download').addEventListener('click', () => downloadRejected(rejectedText));
  if (rejectedCount > 0) downloadRejected(rejectedText);
}

async function render() {
  const total = state.manifest.length;
  updateUrl(state.corpus, state.index);
  if (isComplete(state.index, total)) {
    renderComplete(total);
    return;
  }

  const book = state.manifest[state.index];
  const isbn = isbnFromFilename(book.filename);

  let expected;
  try {
    expected = await fetchJson(`../corpus/${state.corpus}/${isbn}.expected.json`);
  } catch (err) {
    renderSkippable(book, isbn, `expected.json not available: ${err.message}`);
    return;
  }

  renderBookPlaceholder(book, isbn, expected);
}

async function init() {
  const { corpus, index } = parseParams(window.location.search);
  if (!corpus) {
    showError('Add ?corpus=&lt;name&gt; to the URL, e.g. ?corpus=open-access');
    return;
  }
  state.corpus = corpus;
  state.decisions = loadDecisions(corpus);
  state.index = normalizeIndex(index);

  try {
    const manifest = await fetchJson(`../corpus/${corpus}/manifest.json`);
    state.manifest = manifest.books;
  } catch (err) {
    showError(`Could not load manifest for corpus "${corpus}": ${err.message}`);
    return;
  }

  await render();
}

init();
```

- [ ] **Step 2: Start the local server**

```bash
python3 -m http.server 8000
```

Leave it running in this terminal for the rest of this task and Tasks 5-7.

- [ ] **Step 3: Verify the missing-corpus error**

Open `http://localhost:8000/evaluation/app/index.html` (no query string).
Expected: the page shows "Add ?corpus=&lt;name&gt; to the URL, e.g.
?corpus=open-access" instead of crashing.

- [ ] **Step 4: Verify the bad-corpus error**

Open `http://localhost:8000/evaluation/app/index.html?corpus=nope`.
Expected: a "Could not load manifest for corpus "nope"" message
mentioning a 404.

- [ ] **Step 5: Verify navigation, decisions, and persistence**

Open `http://localhost:8000/evaluation/app/index.html?corpus=open-access`.
Expected: header reads "Book 1 of 37 — 9783031466373 — Transformations of
European Welfare States and Social Rights". Click **Accept**. Expected: the address bar's
`index` query param becomes `1` without a page reload, and the header
now shows "Book 2 of 37 — ...". Open the browser devtools console and run:

```js
localStorage.getItem('gt-review:open-access:decisions')
```

Expected: a JSON string containing the first book's isbn mapped to
`"accepted"`. Click **Prev**. Expected: back on book 1, with the
**Accept** button visually outlined (the `current-choice` style).

- [ ] **Step 6: Verify completion and download**

Edit the URL's `index` param directly to `36` (one less than the total
book count printed in Step 5) and reload. Click **Reject**. Expected: a
"Review complete" screen showing counts, and a file named
`open-access-rejected.txt` downloads automatically containing at least
the isbn you just rejected (open it to confirm — one isbn per line).
Click "Download rejected list" again and confirm it re-downloads the same
content.

- [ ] **Step 7: Reset local state before moving on**

```js
localStorage.removeItem('gt-review:open-access:decisions')
```

Run this in the devtools console so Task 5/6 manual checks start clean.

- [ ] **Step 8: Commit**

```bash
git add evaluation/app/app.js
git commit -m "feat: add navigation, decisions, and completion flow to review app"
```

---

### Task 5: Real thumbnail rendering (TOC + chapter pages)

**Files:**
- Modify: `evaluation/app/app.js`

Replaces `renderBookPlaceholder` with real PDF.js-rendered thumbnails,
and adds the PDF-load failure path alongside the existing
expected.json-load failure path.

- [ ] **Step 1: Apply the following changes to `evaluation/app/app.js`**

Change the import line to add two more pure helpers:

```js
import {
  parseParams,
  isbnFromFilename,
  decisionsStorageKey,
  rejectedListText,
  tocPageRange,
  computeScale,
  isComplete,
  normalizeIndex,
} from './lib.js';
```

Add a constant right after the imports:

```js
const THUMBNAIL_TARGET_WIDTH = 600;
```

Add a new function (anywhere above `render`):

```js
async function renderPageThumb(pdf, pdfIndex, label, cssClass) {
  const page = await pdf.getPage(pdfIndex + 1);
  const viewport = page.getViewport({ scale: 1 });
  const scale = computeScale(viewport.width, THUMBNAIL_TARGET_WIDTH);
  const scaledViewport = page.getViewport({ scale });

  const canvas = document.createElement('canvas');
  canvas.width = scaledViewport.width;
  canvas.height = scaledViewport.height;
  const ctx = canvas.getContext('2d');
  await page.render({ canvasContext: ctx, viewport: scaledViewport }).promise;

  const figure = document.createElement('figure');
  figure.className = `thumb ${cssClass}`;
  figure.appendChild(canvas);
  const caption = document.createElement('figcaption');
  caption.textContent = label;
  figure.appendChild(caption);
  return figure;
}
```

Replace `renderBookPlaceholder` entirely with:

```js
async function renderBook(book, isbn, expected, pdf) {
  $('app').innerHTML = `
    <header>${headerHtml(book, isbn)}</header>
    <section id="toc-section" class="section toc-section">
      <h2>Table of contents</h2>
      <div id="toc-grid" class="grid"></div>
    </section>
    <section class="section chapters-section">
      <h2>Chapters</h2>
      <div id="chapters-grid" class="grid"></div>
    </section>
    <div class="controls">
      <button id="prev" ${state.index === 0 ? 'disabled' : ''}>Prev</button>
      <button id="reject" class="reject">Reject</button>
      <button id="accept" class="accept">Accept</button>
    </div>
  `;

  const tocPages = tocPageRange(expected.toc);
  if (tocPages.length === 0) {
    $('toc-section').classList.add('hidden');
  } else {
    const tocGrid = $('toc-grid');
    for (const pdfIndex of tocPages) {
      tocGrid.appendChild(await renderPageThumb(pdf, pdfIndex, `TOC — p.${pdfIndex}`, 'toc-thumb'));
    }
  }

  const chaptersGrid = $('chapters-grid');
  for (const chapter of expected.chapters) {
    const authors = (chapter.authors || []).join(', ');
    const label = `${chapter.title} — ${authors} (${chapter.citation_pages ?? '?'})`;
    chaptersGrid.appendChild(
      await renderPageThumb(pdf, chapter.pdf_start_index, label, 'chapter-thumb')
    );
  }

  highlightDecision(isbn);
  $('prev').addEventListener('click', goPrev);
  $('accept').addEventListener('click', () => decide(isbn, 'accepted'));
  $('reject').addEventListener('click', () => decide(isbn, 'rejected'));
}
```

In `render()`, replace the final line `renderBookPlaceholder(book, isbn,
expected);` with a PDF load step:

```js
  let pdf;
  try {
    pdf = await window.pdfjsLib.getDocument(`../corpus/${state.corpus}/${isbn}.pdf`).promise;
  } catch (err) {
    renderSkippable(book, isbn, `PDF not available: ${err.message}`);
    return;
  }

  await renderBook(book, isbn, expected, pdf);
```

- [ ] **Step 2: Verify TOC thumbnails render for the fixture book**

With the server from Task 4 still running, open
`http://localhost:8000/evaluation/app/index.html?corpus=open-access&index=0`
and click **Prev**/**Accept** (or edit the URL's `index`) until you reach
isbn `9781771993661` (the book patched in Task 1). Expected: a "Table of
contents" section with a light-blue-background thumbnail of page index 4
(PDF.js page 5), showing that book's actual TOC page.

- [ ] **Step 3: Verify chapter thumbnails render, and TOC hides when absent**

On that same book, expected: a "Chapters" section below with one
light-green-background thumbnail per chapter, each showing what should be
that chapter's opening page, captioned with title/authors/citation pages.
Then navigate to any other book in the corpus (which lacks a `toc` field
per Task 1's note). Expected: the "Table of contents" heading and grid
are not shown at all for that book (only "Chapters").

- [ ] **Step 4: Verify the PDF-missing path**

Rename a PDF temporarily to simulate it being absent from disk, e.g.:

```bash
mv evaluation/corpus/open-access/9781800648234.pdf /tmp/9781800648234.pdf.bak
```

Navigate the app to that book (`?corpus=open-access&index=<N>` for
whichever index it is in `manifest.json`). Expected: "PDF not available:
..." with a **Skip** button that advances past it. Restore the file:

```bash
mv /tmp/9781800648234.pdf.bak evaluation/corpus/open-access/9781800648234.pdf
```

- [ ] **Step 5: Commit**

```bash
git add evaluation/app/app.js
git commit -m "feat: render TOC and chapter-start thumbnails via PDF.js"
```

---

### Task 6: Lightbox zoom on thumbnail click

**Files:**
- Modify: `evaluation/app/app.js`

- [ ] **Step 1: Apply the following changes to `evaluation/app/app.js`**

Add a second constant next to `THUMBNAIL_TARGET_WIDTH`:

```js
const LIGHTBOX_TARGET_WIDTH = 1400;
```

Add a new function (near `renderPageThumb`):

```js
async function openLightbox(pdf, pdfIndex) {
  const page = await pdf.getPage(pdfIndex + 1);
  const viewport = page.getViewport({ scale: 1 });
  const scale = computeScale(viewport.width, LIGHTBOX_TARGET_WIDTH);
  const scaledViewport = page.getViewport({ scale });

  const canvas = document.createElement('canvas');
  canvas.width = scaledViewport.width;
  canvas.height = scaledViewport.height;
  const ctx = canvas.getContext('2d');
  await page.render({ canvasContext: ctx, viewport: scaledViewport }).promise;

  const overlay = document.createElement('div');
  overlay.id = 'lightbox';
  overlay.appendChild(canvas);
  function onKey(e) {
    if (e.key === 'Escape') close();
  }
  function close() {
    overlay.remove();
    document.removeEventListener('keydown', onKey);
  }
  overlay.addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
}
```

In `renderPageThumb`, add a click listener on the canvas right after its
`await page.render(...)` line:

```js
  canvas.addEventListener('click', () => openLightbox(pdf, pdfIndex));
```

- [ ] **Step 2: Verify the lightbox**

Reload `http://localhost:8000/evaluation/app/index.html?corpus=open-access&index=0`
and click any thumbnail. Expected: a full-screen dark overlay showing that
same page re-rendered much larger. Click the overlay (anywhere).
Expected: it closes. Reopen it and press `Escape`. Expected: it closes
the same way.

- [ ] **Step 3: Commit**

```bash
git add evaluation/app/app.js
git commit -m "feat: add click-to-zoom lightbox for review thumbnails"
```

---

### Task 7: README and end-to-end verification

**Files:**
- Create: `evaluation/app/README.md`

- [ ] **Step 1: Write `evaluation/app/README.md`**

```md
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
```

- [ ] **Step 2: Full end-to-end pass**

With the local server still running, open
`http://localhost:8000/evaluation/app/index.html?corpus=open-access`
and page through at least 5 books using **Accept**/**Reject**, including
the fixture book `9781771993661`, confirming for each: the header text is
correct, thumbnails look like genuinely correct pages for that ground
truth, the URL's `index` advances, and clicking a thumbnail zooms it.
Then jump (via the URL bar) to the last valid index, make a decision, and
confirm the rejected-list file downloads and its contents match every
book you rejected during this pass.

- [ ] **Step 3: Re-run the unit tests one more time**

```bash
node --test evaluation/app/lib.test.js
```

Expected: `# pass 15`, `# fail 0`.

- [ ] **Step 4: Stop the local server**

```bash
# in the terminal running `python3 -m http.server 8000`
Ctrl+C
```

- [ ] **Step 5: Commit**

```bash
git add evaluation/app/README.md
git commit -m "docs: add README for ground-truth review app"
```
