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
  render(true);
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

function renderComplete(total, fromDecision) {
  const rejectedText = rejectedListText(state.decisions);
  const rejectedCount = rejectedText ? rejectedText.split('\n').length : 0;
  const acceptedCount = Object.values(state.decisions).filter((v) => v === 'accepted').length;
  $('app').innerHTML = `
    <h1>Review complete</h1>
    <p>${acceptedCount} accepted, ${rejectedCount} rejected, ${total} total.</p>
    <button id="download">Download rejected list</button>
  `;
  $('download').addEventListener('click', () => downloadRejected(rejectedText));
  if (fromDecision && rejectedCount > 0) downloadRejected(rejectedText);
}

async function render(fromDecision = false) {
  const total = state.manifest.length;
  updateUrl(state.corpus, state.index);
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
