export function parseParams(search) {
  const params = new URLSearchParams(search);
  const corpus = params.get('corpus');
  const rawIndex = params.get('index');
  const parsedIndex = rawIndex === null ? 0 : Number.parseInt(rawIndex, 10);
  const index = Number.isFinite(parsedIndex) && parsedIndex >= 0 ? parsedIndex : 0;
  const repoRoot = params.get('repoRoot');
  return { corpus: corpus || null, index, repoRoot: repoRoot || null };
}

export function vscodeFileUri(repoRoot, corpus, isbn) {
  return `vscode://file${repoRoot}/evaluation/corpus/${corpus}/${isbn}.expected.json`;
}

export function clearRejectedDecisions(decisions) {
  return Object.fromEntries(Object.entries(decisions).filter(([, verdict]) => verdict !== 'rejected'));
}

export function tabTitle(corpus, oneBasedIndex, total) {
  return `${corpus} ${oneBasedIndex}/${total}`;
}

export function pageHeading(corpus, oneBasedIndex, total) {
  return `Corpus '${corpus}' ${oneBasedIndex} of ${total}`;
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
