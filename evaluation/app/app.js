import {
  parseParams,
  isbnFromFilename,
  decisionsStorageKey,
  rejectedListText,
  tocPageRange,
  computeScale,
  isComplete,
  normalizeIndex,
  vscodeFileUri,
  clearRejectedDecisions,
  tabTitle,
  pageHeading,
} from './lib.js';

const THUMBNAIL_TARGET_WIDTH = 120;
const LIGHTBOX_TARGET_WIDTH = 1400;

const state = {
  corpus: null,
  index: 0,
  manifest: [],
  decisions: {},
  repoRoot: null,
};

function $(id) {
  return document.getElementById(id);
}

function showError(message) {
  $('app').innerHTML = `<p class="error">${message}</p>`;
}

function loadDecisions(corpus) {
  const raw = localStorage.getItem(decisionsStorageKey(corpus));
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
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
  const accepted = state.decisions[isbn] === 'accepted';
  const checkmark = accepted ? ' ✓' : '';
  return `<span class="${accepted ? 'accepted-header' : ''}">Book ${state.index + 1} of ${state.manifest.length} — ${isbn} — ${book.title}${checkmark}</span>`;
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
  render(true);
}

function goPrev() {
  state.index = Math.max(0, state.index - 1);
  render();
}

async function openLightbox(pdf, pdfIndex) {
  const overlay = document.createElement('div');
  overlay.id = 'lightbox';
  function onKey(e) {
    if (e.key === 'Escape') close();
  }
  function close() {
    overlay.remove();
    document.removeEventListener('keydown', onKey);
  }
  overlay.addEventListener('click', close);
  document.addEventListener('keydown', onKey);

  try {
    const page = await pdf.getPage(pdfIndex + 1);
    const viewport = page.getViewport({ scale: 1 });
    const scale = computeScale(viewport.width, LIGHTBOX_TARGET_WIDTH);
    const scaledViewport = page.getViewport({ scale });
    const canvas = document.createElement('canvas');
    canvas.width = scaledViewport.width;
    canvas.height = scaledViewport.height;
    const ctx = canvas.getContext('2d');
    await page.render({ canvasContext: ctx, viewport: scaledViewport }).promise;
    overlay.appendChild(canvas);
  } catch (err) {
    const message = document.createElement('p');
    message.className = 'error';
    message.textContent = `Failed to render page: ${err.message}`;
    overlay.appendChild(message);
  }

  document.body.appendChild(overlay);
}

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
  canvas.addEventListener('click', () => openLightbox(pdf, pdfIndex));

  const figure = document.createElement('figure');
  figure.className = `thumb ${cssClass}`;
  figure.appendChild(canvas);
  const caption = document.createElement('figcaption');
  caption.textContent = label;
  figure.appendChild(caption);
  return figure;
}

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
      ${state.repoRoot ? `<a id="open-vscode" class="button" href="${vscodeFileUri(state.repoRoot, state.corpus, isbn)}">Open in VS Code</a>` : ''}
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

function renderComplete(total, fromDecision) {
  const rejectedText = rejectedListText(state.decisions);
  const rejectedCount = rejectedText ? rejectedText.split('\n').length : 0;
  const acceptedCount = Object.values(state.decisions).filter((v) => v === 'accepted').length;
  $('app').innerHTML = `
    <h1>Review complete</h1>
    <p>${acceptedCount} accepted, ${rejectedCount} rejected, ${total} total.</p>
    <button id="download">Download rejected list</button>
    <button id="clear-rejected">Clear rejected list</button>
  `;
  $('download').addEventListener('click', () => downloadRejected(rejectedText));
  $('clear-rejected').addEventListener('click', () => {
    state.decisions = clearRejectedDecisions(state.decisions);
    saveDecisions(state.corpus, state.decisions);
    state.index = 0;
    render();
  });
  if (fromDecision && rejectedCount > 0) downloadRejected(rejectedText);
}

async function render(fromDecision = false) {
  const total = state.manifest.length;
  updateUrl(state.corpus, state.index);
  const displayIndex = Math.min(state.index + 1, total);
  document.title = tabTitle(state.corpus, displayIndex, total);
  $('page-heading').textContent = pageHeading(state.corpus, displayIndex, total);
  if (isComplete(state.index, total)) {
    renderComplete(total, fromDecision);
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

  let pdf;
  try {
    pdf = await window.pdfjsLib.getDocument(`../corpus/${state.corpus}/${isbn}.pdf`).promise;
  } catch (err) {
    renderSkippable(book, isbn, `PDF not available: ${err.message}`);
    return;
  }

  try {
    await renderBook(book, isbn, expected, pdf);
  } catch (err) {
    renderSkippable(book, isbn, `Failed to render pages: ${err.message}`);
  }
}

async function init() {
  const { corpus, index, repoRoot } = parseParams(window.location.search);
  if (!corpus) {
    showError('Add ?corpus=&lt;name&gt; to the URL, e.g. ?corpus=open-access');
    return;
  }
  state.corpus = corpus;
  state.decisions = loadDecisions(corpus);
  state.index = normalizeIndex(index);
  state.repoRoot = repoRoot;

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
