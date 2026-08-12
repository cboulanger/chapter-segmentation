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
  vscodeFileUri,
  clearRejectedDecisions,
} from './lib.js';

test('parseParams reads corpus and index', () => {
  assert.deepEqual(parseParams('?corpus=open-access&index=5'), { corpus: 'open-access', index: 5, repoRoot: null });
});

test('parseParams defaults index to 0 when absent', () => {
  assert.deepEqual(parseParams('?corpus=open-access'), { corpus: 'open-access', index: 0, repoRoot: null });
});

test('parseParams returns null corpus when absent', () => {
  assert.deepEqual(parseParams(''), { corpus: null, index: 0, repoRoot: null });
});

test('parseParams ignores a non-numeric index', () => {
  assert.deepEqual(parseParams('?corpus=x&index=abc'), { corpus: 'x', index: 0, repoRoot: null });
});

test('parseParams ignores a negative index', () => {
  assert.deepEqual(parseParams('?corpus=x&index=-3'), { corpus: 'x', index: 0, repoRoot: null });
});

test('parseParams reads repoRoot', () => {
  assert.deepEqual(parseParams('?corpus=x&repoRoot=%2FUsers%2Fme%2Frepo'), {
    corpus: 'x',
    index: 0,
    repoRoot: '/Users/me/repo',
  });
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

test('vscodeFileUri builds an absolute vscode:// deep link', () => {
  assert.equal(
    vscodeFileUri('/Users/me/repo', 'open-access', '9781234567890'),
    'vscode://file/Users/me/repo/evaluation/corpus/open-access/9781234567890.expected.json'
  );
});

test('clearRejectedDecisions drops rejected entries and keeps the rest', () => {
  assert.deepEqual(clearRejectedDecisions({ a: 'accepted', b: 'rejected', c: 'rejected' }), { a: 'accepted' });
});

test('clearRejectedDecisions is a no-op when nothing is rejected', () => {
  assert.deepEqual(clearRejectedDecisions({ a: 'accepted' }), { a: 'accepted' });
});
