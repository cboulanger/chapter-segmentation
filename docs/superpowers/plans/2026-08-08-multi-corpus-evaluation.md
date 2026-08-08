# Multi-corpus evaluation layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `evaluation/manifest.json`'s commingled book sets into self-contained `evaluation/corpus/<name>/` subfolders (`open-access`, `copyrighted`, `pending`), with every runner (tests, scripts, report generation) auto-discovering every corpus.

**Architecture:** `evaluation/harness.py`'s path constants become functions of a `corpus: str` name; every book-lookup function gains a leading `corpus` parameter. Every consumer loops `for corpus in list_corpora(): ...` by default, with an optional `--corpus` flag on scripts to restrict to one. `generate_report.py` produces one report page per corpus plus a landing page.

**Tech Stack:** Python 3.12, pytest, uv. See design spec `docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md` for full rationale.

---

## Book-to-corpus assignment (reference for every task below)

**`open-access`** (6 books): `9783031466373`, `9781771993661`, `9783907297339`, `9782375460122`, `9783907297285`, `9783847432364`.

**`copyrighted`** (11 books): `9783322969828`, `9783848736829`, `9783492021234`, `9783789016202`, `9783789057366`, `9783899718188`, `9780367439712`, `9783465016878`, `9781409403906`, `9783848704316`, `dnb-36942798X`.

**`pending`** (2 books, no `.expected.json` yet): `9783428042241`, `9783899496291`.

---

### Task 1: Capture a baseline report for regression verification

**Files:**
- Create (untracked, outside the repo): `/tmp/baseline-report/`

- [ ] **Step 1: Run the current report generator and save its output**

```bash
uv run python evaluation/generate_report.py --out /tmp/baseline-report
```

Expected: exits 0, writes `/tmp/baseline-report/index.html` and `/tmp/baseline-report/llm/index.html`.

- [ ] **Step 2: Spot-check it has per-book rows**

```bash
grep -c "9783031466373\|9783322969828\|dnb-36942798X" /tmp/baseline-report/index.html
```

Expected: `3` (all three sample books present as table rows). Keep this directory around — Task 15 diffs against it.

---

### Task 2: Split `manifest.json` into three corpus manifests

**Files:**
- Create: `evaluation/corpus/open-access/manifest.json`
- Create: `evaluation/corpus/copyrighted/manifest.json`
- Create: `evaluation/corpus/pending/manifest.json`

- [ ] **Step 1: Create the open-access manifest**

```bash
mkdir -p evaluation/corpus/open-access evaluation/corpus/copyrighted evaluation/corpus/pending
```

Write `evaluation/corpus/open-access/manifest.json`:

```json
{
  "books": [
    {
      "filename": "9783031466373.pdf",
      "title": "Transformations of European Welfare States and Social Rights",
      "language": "en",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": true,
      "doi": "10.1007/978-3-031-46637-3",
      "download_url": "https://library.oapen.org/bitstream/handle/20.500.12657/86934/978-3-031-46637-3.pdf?sequence=1"
    },
    {
      "filename": "9781771993661.pdf",
      "title": "Violence, Imagination, and Resistance: Socio-legal Interrogations of Power",
      "language": "en",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": true,
      "doi": "10.15215/aupress/9781778290022.01",
      "download_url": "https://www.aupress.ca/app/uploads/120313_Alam_et_al_2023-Violence_Imagination_and_Resistance.pdf"
    },
    {
      "filename": "9783907297339.pdf",
      "title": "20 ans de transparence à Genève",
      "language": "fr",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": true,
      "doi": "10.38107/033",
      "download_url": "https://library.oapen.org/bitstream/handle/20.500.12657/61692/oa_pdf-033-1675100892.pdf?sequence=1"
    },
    {
      "filename": "9782375460122.pdf",
      "title": "Accueillir des publics migrants et immigrés. Interculturalité en bibliothèque",
      "language": "fr",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": true,
      "doi": "10.4000/books.pressesenssib.7527",
      "download_url": "https://books.openedition.org/pressesenssib/pdf/7527"
    },
    {
      "filename": "9783907297285.pdf",
      "title": "Recht in der Krise (APARIUZ XXIII)",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": true,
      "doi": "10.38107/028",
      "download_url": "https://library.oapen.org/bitstream/handle/20.500.12657/58534/oa_pdf-028-1-1663335379.pdf?sequence=1"
    },
    {
      "filename": "9783847432364.pdf",
      "title": "Recht umkämpft",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": true,
      "doi": "10.3224/84743101",
      "download_url": "https://library.oapen.org/bitstream/handle/20.500.12657/101141/UTF-89783847432364.pdf?sequence=1"
    }
  ]
}
```

- [ ] **Step 2: Create the copyrighted manifest**

Write `evaluation/corpus/copyrighted/manifest.json`:

```json
{
  "books": [
    {
      "filename": "9783322969828.pdf",
      "title": "Jahrbuch für Rechtssoziologie und Rechtstheorie IV",
      "language": "de",
      "extraction_type": "scan",
      "embedded_toc": false,
      "oa": false,
      "doi": "10.1007/978-3-322-96982-8",
      "download_url": null
    },
    {
      "filename": "9783848736829.pdf",
      "title": "Politik und Recht: Umrisse eines politikwissenschaftlichen Forschungsfeldes",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null
    },
    {
      "filename": "9783492021234.pdf",
      "title": "Empirische Rechtssoziologie",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null
    },
    {
      "filename": "9783789016202.pdf",
      "title": "Rechtsproduktion und Rechtsbewußtsein",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null
    },
    {
      "filename": "9783789057366.pdf",
      "title": "Soziologie des Rechts. Festschrift für Erhard Blankenburg zum 60. Geburtstag",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null,
      "heuristic_expected_zero": true
    },
    {
      "filename": "9783899718188.pdf",
      "title": "Systemtheorie in den Fachwissenschaften: Zugänge, Methoden, Probleme",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null
    },
    {
      "filename": "9780367439712.pdf",
      "title": "Luhmann and Socio-Legal Research: An Empirical Agenda for Social Systems Theory",
      "language": "en",
      "extraction_type": "native",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null
    },
    {
      "filename": "9783465016878.pdf",
      "title": "Historische Soziologie der Rechtswissenschaft",
      "language": "de",
      "extraction_type": "scan",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null
    },
    {
      "filename": "9781409403906.pdf",
      "title": "Central and Eastern Europe After Transition: Towards a New Socio-Legal Semantics",
      "language": "en",
      "extraction_type": "scan",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null
    },
    {
      "filename": "9783848704316.pdf",
      "title": "Constitutional Jurisprudence: Function, Impact, and Challenges of Constitutional Courts",
      "language": "en",
      "extraction_type": "scan",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null,
      "heuristic_expected_zero": true
    },
    {
      "filename": "dnb-36942798X.pdf",
      "title": "Studien und Materialien zur Rechtssoziologie",
      "language": "de",
      "extraction_type": "scan",
      "embedded_toc": false,
      "oa": false,
      "doi": null,
      "download_url": null,
      "heuristic_expected_zero": true
    }
  ]
}
```

- [ ] **Step 3: Create the pending manifest**

Write `evaluation/corpus/pending/manifest.json`:

```json
{
  "books": [
    {
      "filename": "9783428042241.pdf",
      "title": "Recht und Gesellschaft: Festschrift für Helmut Schelsky zum 65. Geburtstag",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": true,
      "oa": false,
      "doi": "10.3790/978-3-428-44224-9",
      "download_url": null
    },
    {
      "filename": "9783899496291.pdf",
      "title": "Festschrift 200 Jahre Juristische Fakultät der Humboldt-Universität zu Berlin: Geschichte, Gegenwart und Zukunft",
      "language": "de",
      "extraction_type": "native",
      "embedded_toc": false,
      "oa": false,
      "doi": "10.1515/9783899496307",
      "download_url": null
    }
  ]
}
```

- [ ] **Step 4: Stage the three new manifests**

```bash
git add evaluation/corpus/open-access/manifest.json evaluation/corpus/copyrighted/manifest.json evaluation/corpus/pending/manifest.json
git status --short evaluation/corpus/
```

Expected: three new files shown as `A` (added).

---

### Task 3: Move PDFs, ground truth, and caches into their corpus directories

**Files:** moves ~19 `.pdf`, 17 `.expected.json`, ~36 `public-cache/*`, 17 `llm-cache/*.json`, and the `.ocr-cache/` directory; deletes the old flat `evaluation/manifest.json`.

- [ ] **Step 1: Run the migration script**

```bash
set -e

declare -A CORPUS_FOR=(
  [9783031466373]=open-access [9781771993661]=open-access [9783907297339]=open-access
  [9782375460122]=open-access [9783907297285]=open-access [9783847432364]=open-access
  [9783322969828]=copyrighted [9783848736829]=copyrighted [9783492021234]=copyrighted
  [9783789016202]=copyrighted [9783789057366]=copyrighted [9783899718188]=copyrighted
  [9780367439712]=copyrighted [9783465016878]=copyrighted [9781409403906]=copyrighted
  [9783848704316]=copyrighted [dnb-36942798X]=copyrighted
  [9783428042241]=pending [9783899496291]=pending
)

for key in "${!CORPUS_FOR[@]}"; do
  corpus="${CORPUS_FOR[$key]}"
  mkdir -p "evaluation/corpus/$corpus/public-cache" "evaluation/corpus/$corpus/llm-cache"
  [ -f "evaluation/${key}.pdf" ] && mv "evaluation/${key}.pdf" "evaluation/corpus/$corpus/${key}.pdf"
  [ -f "evaluation/${key}.expected.json" ] && git mv "evaluation/${key}.expected.json" "evaluation/corpus/$corpus/${key}.expected.json"
  [ -f "evaluation/public-cache/${key}.pages.json" ] && git mv "evaluation/public-cache/${key}.pages.json" "evaluation/corpus/$corpus/public-cache/${key}.pages.json"
  [ -f "evaluation/public-cache/${key}.outline.json" ] && git mv "evaluation/public-cache/${key}.outline.json" "evaluation/corpus/$corpus/public-cache/${key}.outline.json"
  [ -f "evaluation/llm-cache/${key}.json" ] && git mv "evaluation/llm-cache/${key}.json" "evaluation/corpus/$corpus/llm-cache/${key}.json"
done

# .ocr-cache/ is gitignored and content-hash keyed (not book-name keyed).
# Every currently-cached entry belongs to a copyrighted-corpus book (see
# design spec's book-assignment table and RESULTS.md's "Recovery route"
# column), so the whole directory moves as one, with plain mv (untracked).
[ -d evaluation/.ocr-cache ] && mv evaluation/.ocr-cache evaluation/corpus/copyrighted/.ocr-cache

git rm evaluation/manifest.json
rmdir evaluation/public-cache evaluation/llm-cache 2>/dev/null || true
```

- [ ] **Step 2: Verify counts**

```bash
find evaluation/corpus -maxdepth 1 -mindepth 1 -type d | sort
find evaluation/corpus -name "*.pdf" | wc -l          # expect 19
find evaluation/corpus -name "*.expected.json" | wc -l # expect 17
find evaluation/corpus -name "*.pages.json" | wc -l    # expect 17 (matches expected.json count)
find evaluation/corpus -name "*.outline.json" | wc -l  # expect 15 (9783428042241/9783899496291 never had one)
find evaluation/corpus -name "*.json" -path "*/llm-cache/*" | wc -l  # expect 17
ls evaluation/ | grep -v '^corpus$\|^scripts$\|^redaction$\|\.py$\|\.md$\|__pycache__\|\.gitignore'
```

Expected: the corpus listing shows exactly `open-access`, `copyrighted`, `pending`; the last command prints nothing left over (no stray `.pdf`/`.expected.json`/`manifest.json` at the old flat location).

- [ ] **Step 3: Commit the migration**

```bash
git add -A evaluation/corpus/ evaluation/
git status --short
git commit -m "$(cat <<'EOF'
refactor: split evaluation set into evaluation/corpus/<name>/ subfolders

Moves the commingled open-access/copyrighted book sets (plus 2 books
pending ground truth) into self-contained evaluation/corpus/open-access/,
evaluation/corpus/copyrighted/, and evaluation/corpus/pending/ directories,
per docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md.
Code updates to consume the new layout follow in subsequent commits.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rewrite `evaluation/harness.py` with a corpus-parameterized API

**Files:**
- Modify: `evaluation/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Rewrite the failing test file first**

Replace the full contents of `tests/test_harness.py`:

```python
"""Unit tests for evaluation/harness.py -- the pages-loading logic
shared by the pytest accuracy harness and the evaluation scripts. The
extraction and cache primitives are patched; what's under test is the
routing: healthy pages pass through, OCR-shaped pages come from the eval
OCR cache, and a cache miss returns None (caller skips the book)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chapter_segmentation.evidence.types import ChapterCandidate
from evaluation.harness import (
    analysis_pages_for,
    available_public_books,
    list_corpora,
    outline_candidate_from_dict,
    outline_candidate_to_dict,
    public_outline_candidates_for,
    public_pages_for,
)

_HEALTHY_PAGES = ["Zeile\n" * 200] * 40
_OCR_PAGES = ["ocr text\n" * 100] * 40


class TestListCorpora(unittest.TestCase):
    def test_lists_only_subfolders_with_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus-a").mkdir()
            (root / "corpus-a" / "manifest.json").write_text('{"books": []}', encoding="utf-8")
            (root / "corpus-b").mkdir()
            (root / "corpus-b" / "manifest.json").write_text('{"books": []}', encoding="utf-8")
            (root / "not-a-corpus").mkdir()  # no manifest.json
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertEqual(list_corpora(), ["corpus-a", "corpus-b"])

    def test_returns_empty_list_when_corpus_root_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.harness.CORPUS_ROOT", Path(tmp) / "does-not-exist"):
                self.assertEqual(list_corpora(), [])


class TestAnalysisPagesFor(unittest.TestCase):
    def test_returns_extracted_pages_when_text_layer_is_usable(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=(_HEALTHY_PAGES, False)), \
             patch("evaluation.harness.load_cached_ocr") as mock_cache:
            pages = analysis_pages_for("test-corpus", b"%PDF-fake")
        self.assertEqual(pages, _HEALTHY_PAGES)
        mock_cache.assert_not_called()

    def test_returns_cached_ocr_pages_for_ocr_shaped_input(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=([""] * 300, False)), \
             patch("evaluation.harness.load_cached_ocr", return_value={"detected_language": "deu", "pages": _OCR_PAGES}):
            pages = analysis_pages_for("test-corpus", b"%PDF-fake")
        self.assertEqual(pages, _OCR_PAGES)

    def test_returns_none_on_ocr_cache_miss(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=([""] * 300, False)), \
             patch("evaluation.harness.load_cached_ocr", return_value=None):
            self.assertIsNone(analysis_pages_for("test-corpus", b"%PDF-fake"))

    def test_returns_none_when_cached_ocr_pages_are_still_degenerate(self):
        with patch("evaluation.harness.extract_page_texts_for_analysis", return_value=([""] * 300, False)), \
             patch("evaluation.harness.load_cached_ocr", return_value={"detected_language": "deu", "pages": [""] * 300}):
            self.assertIsNone(analysis_pages_for("test-corpus", b"%PDF-fake"))


class TestAvailablePublicBooks(unittest.TestCase):
    def test_yields_books_with_a_cache_entry_and_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "test-corpus"
            public_cache_dir = cdir / "public-cache"
            public_cache_dir.mkdir(parents=True)
            (cdir / "9999999.expected.json").write_text("{}", encoding="utf-8")
            (public_cache_dir / "9999999.pages.json").write_text(
                json.dumps({"pages": ["a"]}), encoding="utf-8",
            )
            book = {"filename": "9999999.pdf", "title": "Test Book"}
            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]):
                results = available_public_books("test-corpus")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], "9999999")

    def test_skips_books_with_no_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cdir = root / "test-corpus"
            public_cache_dir = cdir / "public-cache"
            public_cache_dir.mkdir(parents=True)
            (cdir / "9999999.expected.json").write_text("{}", encoding="utf-8")
            book = {"filename": "9999999.pdf", "title": "Test Book"}
            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.harness.load_manifest_books", return_value=[book]):
                results = available_public_books("test-corpus")
            self.assertEqual(results, [])


class TestPublicPagesFor(unittest.TestCase):
    def test_returns_pages_for_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_cache_dir = root / "test-corpus" / "public-cache"
            public_cache_dir.mkdir(parents=True)
            (public_cache_dir / "9999999.pages.json").write_text(
                json.dumps({"pages": ["redacted page text"]}), encoding="utf-8",
            )
            with patch("evaluation.harness.CORPUS_ROOT", root):
                pages = public_pages_for("test-corpus", "9999999")
            self.assertEqual(pages, ["redacted page text"])

    def test_returns_none_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test-corpus" / "public-cache").mkdir(parents=True)
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(public_pages_for("test-corpus", "9999999"))


class TestOutlineCandidateSerialization(unittest.TestCase):
    def test_round_trips_all_fields(self):
        candidate = ChapterCandidate(
            title="Introduction", authors=("Jane Author",), printed_page_number=1,
            pdf_page_index=5, chapter_doi="10.1/x", source="outline", metadata_confidence=0.9,
        )
        restored = outline_candidate_from_dict(outline_candidate_to_dict(candidate))
        self.assertEqual(restored, candidate)

    def test_round_trips_defaults(self):
        candidate = ChapterCandidate(title="Introduction", pdf_page_index=5)
        restored = outline_candidate_from_dict(outline_candidate_to_dict(candidate))
        self.assertEqual(restored, candidate)


class TestPublicOutlineCandidatesFor(unittest.TestCase):
    def test_returns_candidates_for_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_cache_dir = root / "test-corpus" / "public-cache"
            public_cache_dir.mkdir(parents=True)
            candidate = ChapterCandidate(title="Introduction", pdf_page_index=5, source="outline")
            (public_cache_dir / "9999999.outline.json").write_text(
                json.dumps({"candidates": [outline_candidate_to_dict(candidate)]}), encoding="utf-8",
            )
            with patch("evaluation.harness.CORPUS_ROOT", root):
                candidates = public_outline_candidates_for("test-corpus", "9999999")
            self.assertEqual(candidates, [candidate])

    def test_returns_none_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test-corpus" / "public-cache").mkdir(parents=True)
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(public_outline_candidates_for("test-corpus", "9999999"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test file to confirm it fails**

```bash
uv run pytest tests/test_harness.py -q
```

Expected: FAIL/ERROR -- `ImportError: cannot import name 'list_corpora' from 'evaluation.harness'` (the function doesn't exist yet).

- [ ] **Step 3: Rewrite `evaluation/harness.py`**

Replace the full contents of `evaluation/harness.py`:

```python
"""Shared loading helpers for the chapter-segmentation evaluation set.

Single home for the manifest-merging, PDF-availability, and page-loading
logic that tests/test_segmentation_accuracy.py and the
evaluation/scripts/evaluate_chapter_segmentation_*.py scripts previously each
carried their own copy of. Lives under evaluation/ (not tests/) because
evaluation/scripts/ must not depend on the test tree.

Every evaluation book lives under evaluation/corpus/<corpus>/ -- a
self-contained subfolder (manifest.json, optional manifest.local.json,
PDFs, ground truth, and caches) per corpus -- see
docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md.
list_corpora() is the single source of truth for "what corpora exist";
every function below takes a corpus name so callers can target one corpus
or loop list_corpora() to cover all of them.

Page loading mirrors production's chapter_segmentation.run(): default
extraction with the layout-mode fallback, then -- for books whose text
layer is absent or degenerate (pages_need_ocr) -- the content-hash-keyed
OCR cache populated by evaluation/scripts/ocr_evaluation_pdfs.py. A book
whose OCR cache entry is missing loads as None and should be skipped by the
caller with a pointer to that script.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from chapter_segmentation.evidence.types import ChapterCandidate
from chapter_segmentation.ocr import load_cached_ocr
from chapter_segmentation.segmentation import (
    extract_page_texts_for_analysis,
    pages_need_ocr,
)

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = EVAL_DIR / "corpus"


def list_corpora() -> list[str]:
    """Sorted names of every subfolder under evaluation/corpus/ that has a
    manifest.json -- the single source of truth for "what corpora exist"
    that every runner iterates over."""
    if not CORPUS_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in CORPUS_ROOT.iterdir()
        if p.is_dir() and (p / "manifest.json").exists()
    )


def corpus_dir(corpus: str) -> Path:
    return CORPUS_ROOT / corpus


def public_cache_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / "public-cache"


def ocr_cache_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / ".ocr-cache"


def llm_cache_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / "llm-cache"


def load_manifest_books(corpus: str) -> list[dict]:
    """Merge a corpus's committed manifest.json with its gitignored,
    optional manifest.local.json (see evaluation/CLAUDE.md) -- the latter
    holds books that have no DOI or otherwise can't be shared, still
    exercised in local runs on the machine that added them."""
    cdir = corpus_dir(corpus)
    books = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))["books"]
    local_manifest_path = cdir / "manifest.local.json"
    if local_manifest_path.exists():
        books = books + json.loads(local_manifest_path.read_text(encoding="utf-8"))["books"]
    return books


def available_books(corpus: str) -> list[tuple[Path, Path, dict]]:
    """(pdf_path, expected_json_path, manifest_entry) for every book in
    this corpus whose PDF and ground truth are both present locally right
    now."""
    cdir = corpus_dir(corpus)
    triples = []
    for book in load_manifest_books(corpus):
        pdf_path = cdir / book["filename"]
        expected_path = cdir / (Path(book["filename"]).stem + ".expected.json")
        if pdf_path.exists() and expected_path.exists():
            triples.append((pdf_path, expected_path, book))
    return triples


def available_public_books(corpus: str) -> list[tuple[str, Path, dict]]:
    """(manifest_key, expected_json_path, manifest_entry) for every book in
    this corpus with a public-cache entry -- no PDF or .ocr-cache
    required."""
    cdir = corpus_dir(corpus)
    cache_dir = public_cache_dir(corpus)
    triples = []
    for book in load_manifest_books(corpus):
        manifest_key = Path(book["filename"]).stem
        expected_path = cdir / f"{manifest_key}.expected.json"
        cache_path = cache_dir / f"{manifest_key}.pages.json"
        if cache_path.exists() and expected_path.exists():
            triples.append((manifest_key, expected_path, book))
    return triples


def public_pages_for(corpus: str, manifest_key: str) -> Optional[list[str]]:
    """Redacted pages for one book from this corpus's committed
    public-cache, or None if no entry exists yet for this key."""
    cache_path = public_cache_dir(corpus) / f"{manifest_key}.pages.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))["pages"]


def analysis_pages_for(corpus: str, file_bytes: bytes) -> Optional[list[str]]:
    """Page texts for this PDF the same way production run() would see
    them, or None when the book needs OCR and this corpus's eval OCR cache
    has no usable entry yet (run evaluation/scripts/ocr_evaluation_pdfs.py
    to populate it)."""
    pages, _layout_used = extract_page_texts_for_analysis(file_bytes)
    if not pages_need_ocr(pages):
        return pages
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    cached = load_cached_ocr(ocr_cache_dir(corpus), content_hash)
    if cached is not None and not pages_need_ocr(cached["pages"]):
        return cached["pages"]
    return None


def outline_candidate_to_dict(candidate: ChapterCandidate) -> dict:
    return {
        "title": candidate.title,
        "authors": list(candidate.authors),
        "printed_page_number": candidate.printed_page_number,
        "pdf_page_index": candidate.pdf_page_index,
        "chapter_doi": candidate.chapter_doi,
        "source": candidate.source,
        "metadata_confidence": candidate.metadata_confidence,
    }


def outline_candidate_from_dict(data: dict) -> ChapterCandidate:
    return ChapterCandidate(
        title=data["title"],
        authors=tuple(data.get("authors", [])),
        printed_page_number=data.get("printed_page_number"),
        pdf_page_index=data.get("pdf_page_index"),
        chapter_doi=data.get("chapter_doi"),
        source=data.get("source", "outline"),
        metadata_confidence=data.get("metadata_confidence", 1.0),
    )


def public_outline_candidates_for(corpus: str, manifest_key: str) -> Optional[list[ChapterCandidate]]:
    """Cached outline-strategy candidates for one book from this corpus's
    committed public-cache, or None if no entry exists yet (either the
    book's PDF has no outline, or the cache hasn't been generated for it
    yet -- see evaluation/scripts/generate_public_evaluation_cache.py)."""
    cache_path = public_cache_dir(corpus) / f"{manifest_key}.outline.json"
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [outline_candidate_from_dict(c) for c in data["candidates"]]
```

- [ ] **Step 4: Run the test file to confirm it passes**

```bash
uv run pytest tests/test_harness.py -q
```

Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add evaluation/harness.py tests/test_harness.py
git commit -m "$(cat <<'EOF'
refactor: parameterize evaluation/harness.py by corpus

Path constants become functions of a corpus name; every book-lookup
function gains a leading corpus parameter, matching the new
evaluation/corpus/<name>/ layout.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update `evaluation/scripts/fetch_evaluation_pdfs.py`

**Files:**
- Modify: `evaluation/scripts/fetch_evaluation_pdfs.py`

- [ ] **Step 1: Replace the full file contents**

```python
#!/usr/bin/env python3
"""Download the open-access chapter-segmentation evaluation PDFs on demand.

The PDFs themselves are gitignored (evaluation/corpus/*/*.pdf) so they
aren't shipped in the repo. This script reads each corpus's manifest.json
under evaluation/corpus/ (see evaluation/README.md) and downloads each
entry with "oa": true into that same corpus directory if not already
present.

Non-OA books ("oa": false) are perfectly welcome in a manifest — they just
can't be auto-downloaded. If one is missing locally, this script prints its
DOI and the exact path to save it to, so you can fetch it manually through
your institution's legal access and drop it in yourself.

Usage:
    uv run python evaluation/scripts/fetch_evaluation_pdfs.py
    uv run python evaluation/scripts/fetch_evaluation_pdfs.py --force            # re-download even if present
    uv run python evaluation/scripts/fetch_evaluation_pdfs.py --corpus open-access
"""

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir, list_corpora, load_manifest_books


def fetch_corpus(corpus: str, force: bool) -> None:
    cdir = corpus_dir(corpus)
    missing_non_oa = []

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for book in load_manifest_books(corpus):
            target = cdir / book["filename"]
            if not book["oa"]:
                if not target.exists():
                    missing_non_oa.append(book)
                continue
            if target.exists() and not force:
                print(f"[skip] {corpus}/{book['filename']} already present")
                continue
            print(f"[fetch] {corpus}/{book['filename']} <- {book['download_url']}")
            response = client.get(book["download_url"])
            response.raise_for_status()
            target.write_bytes(response.content)
            print(f"        wrote {len(response.content):,} bytes")

    if missing_non_oa:
        print(f"\n[{corpus}] Not auto-downloaded (OA: No — legally cannot be fetched automatically).")
        print("Download each one manually via your institution's access to the DOI below,")
        print("then place it at the path shown, and re-run this script (or just run the tests):\n")
        for book in missing_non_oa:
            doi = book.get("doi")
            doi_url = f"https://doi.org/{doi}" if doi else "(no DOI on file — see manifest.json)"
            print(f"  - {book['filename']}")
            print(f"      DOI: {doi_url}")
            print(f"      Save to: {cdir / book['filename']}")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-download even if the PDF already exists")
    parser.add_argument("--corpus", help="Only fetch this corpus (default: every corpus under evaluation/corpus/)")
    args = parser.parse_args()

    corpora = [args.corpus] if args.corpus else list_corpora()
    for corpus in corpora:
        fetch_corpus(corpus, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

Note: this drops the old `--eval-dir` flag (no longer meaningful now that each corpus resolves its own directory via `corpus_dir()`) and the old importable `fetch_all(eval_dir, force)` function name (renamed `fetch_corpus(corpus, force)`, corpus-scoped). No test file imports either — confirmed by `grep -rln "fetch_evaluation_pdfs" tests/` returning nothing.

- [ ] **Step 2: Confirm no test references the old symbols**

```bash
grep -rln "fetch_evaluation_pdfs\|fetch_all" tests/
```

Expected: no output (nothing to update).

- [ ] **Step 3: Smoke-test it runs (dry, since all 19 PDFs are already present locally)**

```bash
uv run python evaluation/scripts/fetch_evaluation_pdfs.py
```

Expected: prints `[skip] <corpus>/<file> already present` for every book across `open-access`, `copyrighted`, `pending`, exits 0.

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/fetch_evaluation_pdfs.py
git commit -m "$(cat <<'EOF'
refactor: make fetch_evaluation_pdfs.py corpus-aware

Loops evaluation.harness.list_corpora() by default, with an optional
--corpus flag to restrict to one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Update `evaluation/scripts/ocr_evaluation_pdfs.py`

**Files:**
- Modify: `evaluation/scripts/ocr_evaluation_pdfs.py`

- [ ] **Step 1: Replace the full file contents**

```python
#!/usr/bin/env python3
"""OCR the evaluation books whose text layer is absent or degenerate, into
each corpus's gitignored OCR cache (evaluation/corpus/<name>/.ocr-cache/,
content-hash keyed), so the accuracy harness and evaluation scripts can
analyze them the way a real caller would.

Uses KreuzbergOcrBackend by default (pass --ocr-backend tesseract for the
local-binary path instead). Books already cached are skipped instantly, so
re-runs are cheap; the first run over several full scanned books takes a
long time.

    uv run python evaluation/scripts/ocr_evaluation_pdfs.py
    uv run python evaluation/scripts/ocr_evaluation_pdfs.py --corpus copyrighted
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.ocr import detect_language, ocr_pdf_pages
from chapter_segmentation.segmentation import extract_page_texts_for_analysis, pages_need_ocr
from evaluation.harness import available_books, list_corpora, ocr_cache_dir


async def _main(ocr_backend_name: str, kreuzberg_url: str, corpus_filter: str | None) -> int:
    if ocr_backend_name == "tesseract":
        from chapter_segmentation.ocr_backends.tesseract import TesseractOcrBackend
        backend = TesseractOcrBackend()
    else:
        from chapter_segmentation.ocr_backends.kreuzberg import KreuzbergOcrBackend
        backend = KreuzbergOcrBackend(kreuzberg_url=kreuzberg_url)

    corpora = [corpus_filter] if corpus_filter else list_corpora()
    for corpus in corpora:
        for pdf_path, _expected_path, book in available_books(corpus):
            file_bytes = pdf_path.read_bytes()
            pages, _layout_used = extract_page_texts_for_analysis(file_bytes)
            if not pages_need_ocr(pages):
                print(f"{corpus}/{pdf_path.name}: text layer usable, no OCR needed")
                continue
            language = detect_language(book.get("language"), book.get("title", ""))
            print(f"{corpus}/{pdf_path.name}: OCR-ing {len(pages)} pages (language={language}) ...", flush=True)
            try:
                page_texts = await ocr_pdf_pages(
                    file_bytes, backend=backend, cache_dir=ocr_cache_dir(corpus), language=language,
                )
            except Exception as exc:
                print(f"{corpus}/{pdf_path.name}: FAILED ({exc}) -- skipping, will retry on next run", flush=True)
                continue
            print(f"{corpus}/{pdf_path.name}: done, {sum(len(p) for p in page_texts)} chars cached", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-backend", choices=["kreuzberg", "tesseract"], default="kreuzberg")
    parser.add_argument("--kreuzberg-url", default="http://localhost:8100")
    parser.add_argument("--corpus", help="Only OCR this corpus (default: every corpus under evaluation/corpus/)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args.ocr_backend, args.kreuzberg_url, args.corpus)))
```

- [ ] **Step 2: Confirm no test references it**

```bash
grep -rln "ocr_evaluation_pdfs" tests/
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add evaluation/scripts/ocr_evaluation_pdfs.py
git commit -m "$(cat <<'EOF'
refactor: make ocr_evaluation_pdfs.py corpus-aware

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update `evaluation/scripts/generate_public_evaluation_cache.py`

**Files:**
- Modify: `evaluation/scripts/generate_public_evaluation_cache.py`

- [ ] **Step 1: Replace the full file contents**

```python
#!/usr/bin/env python3
"""Generate each corpus's public-cache/ -- a redacted, git-trackable
corpus safe to commit and distribute (real navigational/bibliographic text
kept verbatim, chapter prose replaced with random real words in the book's
own language) plus, per book, a resolved outline-strategy candidate
snapshot (<key>.outline.json -- titles/authors/page indices only, no
prose) so the outline strategy is also testable without the real PDF --
see evaluation/README.md for the redaction rationale and workflow, and
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
for the outline-snapshot rationale.

Run by a maintainer who has the real books locally; not something a
contributor without PDFs needs to run.

    uv run python evaluation/scripts/generate_public_evaluation_cache.py [--book <manifest-key>] [--corpus <name>] [--no-verify]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.evidence.outline_strategy import extract_outline_candidates
from chapter_segmentation.ocr import detect_language
from chapter_segmentation.segmentation import (
    analyze_attachment,
    extract_page_texts_for_analysis,
    pages_need_ocr,
)
from evaluation.harness import (
    analysis_pages_for,
    available_books,
    list_corpora,
    outline_candidate_to_dict,
    public_cache_dir,
)
from evaluation.redaction.redact import redact_book_until_stable

CIPHER_VERSION = 1


def _verify(real_pages: list[str], redacted_pages: list[str]) -> list[str]:
    """Human-readable diff lines, empty when the two page sets produce
    identical detected chapter boundaries."""
    real_result = analyze_attachment(real_pages)
    redacted_result = analyze_attachment(redacted_pages)
    real_boundaries = {(c["pdf_start_index"], c["pdf_end_index"]) for c in real_result["chapters"]}
    redacted_boundaries = {(c["pdf_start_index"], c["pdf_end_index"]) for c in redacted_result["chapters"]}
    if real_boundaries == redacted_boundaries:
        return []
    return [
        f"  real boundaries:     {sorted(real_boundaries)}",
        f"  redacted boundaries: {sorted(redacted_boundaries)}",
    ]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", help="Only regenerate this manifest key (filename stem)")
    parser.add_argument("--corpus", help="Only regenerate this corpus (default: every corpus under evaluation/corpus/)")
    parser.add_argument("--no-verify", action="store_true", help="Skip the exact-boundary-match check")
    args = parser.parse_args()

    failures = 0
    corpora = [args.corpus] if args.corpus else list_corpora()
    for corpus in corpora:
        cache_dir = public_cache_dir(corpus)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path, _expected_path, book in available_books(corpus):
            manifest_key = Path(book["filename"]).stem
            if args.book and manifest_key != args.book:
                continue
            file_bytes = pdf_path.read_bytes()
            real_pages = analysis_pages_for(corpus, file_bytes)
            if real_pages is None:
                print(f"{corpus}/{manifest_key}: SKIPPED (needs OCR -- run scripts/ocr_evaluation_pdfs.py first)")
                continue
            try:
                raw_pages, _layout_used = extract_page_texts_for_analysis(file_bytes)
                source = "ocr" if pages_need_ocr(raw_pages) else "extracted"
                language = detect_language(book.get("language"), book.get("title", ""))
                redacted_pages, extra_preserved = redact_book_until_stable(
                    real_pages, detected_language=language, book_salt=manifest_key,
                )
                if extra_preserved:
                    print(f"{corpus}/{manifest_key}: self-corrected -- forced {len(extra_preserved)} extra page(s) "
                          f"fully verbatim to resolve a redaction-induced boundary drift: {sorted(extra_preserved)}")
                if not args.no_verify:
                    # Defense in depth: redact_book_until_stable already verifies
                    # internally on every attempt, so this only fires if
                    # max_attempts was exhausted without full convergence.
                    diff = _verify(real_pages, redacted_pages)
                    if diff:
                        print(f"{corpus}/{manifest_key}: VERIFY FAILED -- redaction changed detected chapter boundaries "
                              f"even after self-correction")
                        print("\n".join(diff))
                        failures += 1
                        continue
            except Exception as exc:
                # One book's failure must not strand the rest of the batch --
                # same catch-log-continue shape as scripts/ocr_evaluation_pdfs.py.
                print(f"{corpus}/{manifest_key}: FAILED ({exc}) -- skipping")
                failures += 1
                continue
            cache_path = cache_dir / f"{manifest_key}.pages.json"
            cache_path.write_text(
                json.dumps(
                    {"cipher_version": CIPHER_VERSION, "source": source, "pages": redacted_pages},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"{corpus}/{manifest_key}: OK, wrote {cache_path}")
            outline_candidates = extract_outline_candidates(file_bytes)
            outline_cache_path = cache_dir / f"{manifest_key}.outline.json"
            outline_cache_path.write_text(
                json.dumps(
                    {"candidates": [outline_candidate_to_dict(c) for c in outline_candidates]},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"{corpus}/{manifest_key}: wrote {len(outline_candidates)} outline candidate(s) to {outline_cache_path}")
    if failures:
        print(f"{failures} book(s) failed -- see above")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 2: Confirm no test references it**

```bash
grep -rln "generate_public_evaluation_cache" tests/
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add evaluation/scripts/generate_public_evaluation_cache.py
git commit -m "$(cat <<'EOF'
refactor: make generate_public_evaluation_cache.py corpus-aware

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Update `evaluation/scripts/evaluate_chapter_segmentation_strategies.py`

**Files:**
- Modify: `evaluation/scripts/evaluate_chapter_segmentation_strategies.py`

- [ ] **Step 1: Replace the full file contents**

```python
#!/usr/bin/env python3
# evaluation/scripts/evaluate_chapter_segmentation_strategies.py
"""Runs every evaluation corpus (see evaluation/corpus/) through
analyze_attachment_with_strategies instead of the pure-heuristic
analyze_attachment, and prints the same precision/recall table format
tests/test_segmentation_accuracy.py already uses, grouped by corpus, plus
per-book strategies_used diagnostics.

Not a pytest test -- makes real (free, cached) Crossref API calls per book:

    uv run python scripts/evaluate_chapter_segmentation_strategies.py

Pass --no-crossref to disable the Crossref lookup and see outline-only
numbers:

    uv run python scripts/evaluate_chapter_segmentation_strategies.py --no-crossref

See docs/superpowers/specs/2026-08-01-chapter-segmentation-strategy-pipeline-design.md section 12.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import analysis_pages_for, available_books, list_corpora
from chapter_segmentation.evidence.crossref_strategy import CrossrefMetadataStrategy
from chapter_segmentation.evidence.types import BookContext
from chapter_segmentation.evidence.zotero_catalog_strategy import ZoteroCatalogMetadataStrategy
from chapter_segmentation.segmentation import analyze_attachment_with_strategies


async def _main(enable_crossref: bool) -> int:
    corpora = list_corpora()
    if not any(available_books(corpus) for corpus in corpora):
        print("No evaluation PDFs present -- run: uv run python scripts/fetch_evaluation_pdfs.py")
        return 1

    zotero_catalog_strategy = ZoteroCatalogMetadataStrategy({})  # no live library in this script

    async with httpx.AsyncClient() as http_client:
        crossref_strategy = (
            CrossrefMetadataStrategy(http_client, cache_dir=Path("data/crossref_cache"), contact_email=None)
            if enable_crossref else None
        )
        for corpus in corpora:
            triples = available_books(corpus)
            if not triples:
                continue
            print(f"=== {corpus} ===")
            for pdf_path, expected_path, book in triples:
                expected = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
                file_bytes = pdf_path.read_bytes()
                pages = analysis_pages_for(corpus, file_bytes)
                if pages is None:
                    print(f"{pdf_path.name}: SKIPPED (needs OCR — populate the cache with: "
                          f"uv run python scripts/ocr_evaluation_pdfs.py)")
                    continue
                # The evaluation manifest names each PDF after its own ISBN-13
                # (see evaluation/README.md), so the
                # filename stem doubles as the ISBN BookContext needs.
                isbn = Path(book["filename"]).stem
                context = BookContext(
                    item_key=book["filename"], isbn=isbn, title=book["title"],
                    editors=(), publisher=None, year=None,
                )
                result = await analyze_attachment_with_strategies(
                    pages, file_bytes, context, zotero_catalog_strategy,
                    crossref_strategy=crossref_strategy,
                )

                expected_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in expected}
                found_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in result["chapters"]}
                true_positives = expected_ranges & found_ranges

                precision = len(true_positives) / len(found_ranges) if found_ranges else 0.0
                recall = len(true_positives) / len(expected_ranges) if expected_ranges else 0.0
                diag = result["diagnostics"]
                print(
                    f"{pdf_path.name}: precision={precision:.2f} recall={recall:.2f} "
                    f"({len(true_positives)}/{len(found_ranges)} found, {len(true_positives)}/{len(expected_ranges)} expected) "
                    f"strategies_used={diag.get('strategies_used')}"
                )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-crossref", action="store_true", help="Disable the Crossref lookup strategy")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(enable_crossref=not args.no_crossref)))
```

- [ ] **Step 2: Confirm no test references it**

```bash
grep -rln "evaluate_chapter_segmentation_strategies" tests/
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add evaluation/scripts/evaluate_chapter_segmentation_strategies.py
git commit -m "$(cat <<'EOF'
refactor: make evaluate_chapter_segmentation_strategies.py corpus-aware

Groups output by corpus; scoring logic unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Update the two integration test files

**Files:**
- Modify: `tests/test_segmentation_accuracy.py`
- Modify: `tests/test_public_evaluation_cache_parity.py`

- [ ] **Step 1: Replace the full contents of `tests/test_segmentation_accuracy.py`**

```python
"""Precision/recall scoring for chapter_segmentation.analyze_attachment
against the real, hand-verified ground-truth books in
evaluation/corpus/ (design spec §5, §12; see also
docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md).

The PDFs themselves are gitignored — run
`uv run python scripts/fetch_evaluation_pdfs.py` first to download the
open-access ones. A book is skipped (not failed) if its PDF isn't present
locally yet, or if it needs OCR and the evaluation OCR cache hasn't been
populated (run `uv run python scripts/ocr_evaluation_pdfs.py` with the
Kreuzberg sidecar up) — both are real, checkable states, not placeholders.

Pages are loaded exactly the way production's run() sees them (layout-mode
fallback + OCR cache) via evaluation/harness.py. Every corpus under
evaluation/corpus/ is exercised in the same test method.

Marked "integration" so it's excluded from the default `uv run pytest` /
`npm test` run (see pyproject.toml's addopts) -- this is a reported, not
gated, benchmark (design spec §12: probabilistic, not pass/fail), not
something that should ever block CI. Run it directly:

    uv run pytest tests/test_segmentation_accuracy.py -q -s

`-s` is required to see the per-book summary lines (pytest swallows `print`
output by default).
"""

import json
import unittest

import pytest

from evaluation.harness import analysis_pages_for, available_books, list_corpora
from chapter_segmentation.segmentation import analyze_attachment

pytestmark = pytest.mark.integration


def _any_books_available() -> bool:
    return any(available_books(corpus) for corpus in list_corpora())


@unittest.skipUnless(
    _any_books_available(),
    "No evaluation PDFs present — run: uv run python scripts/fetch_evaluation_pdfs.py",
)
class TestChapterSegmentationAccuracy(unittest.TestCase):
    # The default 30s global timeout (pyproject.toml) is sized for the
    # open-access corpus; the layout-mode re-extraction pass on large
    # copyrighted-corpus books is slow (whole-book re-extraction per book
    # that triggers it), so give the single all-books method plenty of room.
    @pytest.mark.timeout(900)
    def test_boundary_precision_recall_per_book(self):
        for corpus in list_corpora():
            for pdf_path, expected_path, book in available_books(corpus):
                with self.subTest(book=f"{corpus}/{pdf_path.name}"):
                    expected = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
                    pages = analysis_pages_for(corpus, pdf_path.read_bytes())
                    if pages is None:
                        print(f"{corpus}/{pdf_path.name}: SKIPPED (needs OCR — populate the cache with: "
                              f"uv run python scripts/ocr_evaluation_pdfs.py)")
                        continue
                    result = analyze_attachment(pages)

                    expected_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in expected}
                    found_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in result["chapters"]}
                    true_positives = expected_ranges & found_ranges

                    precision = len(true_positives) / len(found_ranges) if found_ranges else 0.0
                    recall = len(true_positives) / len(expected_ranges) if expected_ranges else 0.0
                    print(f"{corpus}/{pdf_path.name}: precision={precision:.2f} recall={recall:.2f} "
                          f"({len(true_positives)}/{len(found_ranges)} found, {len(true_positives)}/{len(expected_ranges)} expected)")
                    if book.get("heuristic_expected_zero", False):
                        # This book is a known, accepted heuristic limitation --
                        # zero recall even after the layout fallback and OCR
                        # route (see evaluation/RESULTS.md) -- so zero is
                        # the expected outcome here, not a regression.
                        continue
                    # Reported, not gated (design spec §12: probabilistic, not pass/fail) —
                    # this assertion only catches a total regression to zero detection.
                    self.assertGreater(recall, 0.0, f"{corpus}/{pdf_path.name}: detected zero of {len(expected_ranges)} known chapters")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Replace the full contents of `tests/test_public_evaluation_cache_parity.py`**

```python
"""Aggregate precision/recall parity check for every corpus's redacted
public-cache (evaluation/corpus/<name>/public-cache/) -- see
docs/superpowers/specs/2026-08-05-evaluation-corpus-redaction-design.md
section 9.

Needs no PDFs or .ocr-cache/ -- only the committed public-cache/
directories, so this is what CI and contributors without the source books
actually run.

Marked "integration" so it's excluded from the default `uv run pytest` run
(see pyproject.toml's addopts) -- reported, not gated, same as
test_chapter_segmentation_accuracy.py.

    uv run pytest tests/test_public_evaluation_cache_parity.py -q -s
"""

import json
import unittest

import pytest

from evaluation.harness import available_public_books, list_corpora, public_pages_for
from evaluation.metrics import precision_recall_f1
from chapter_segmentation.segmentation import analyze_attachment

pytestmark = pytest.mark.integration


def _any_public_books_available() -> bool:
    return any(available_public_books(corpus) for corpus in list_corpora())


@unittest.skipUnless(
    _any_public_books_available(),
    "No public-cache entries present -- run: "
    "uv run python scripts/generate_public_evaluation_cache.py",
)
class TestPublicEvaluationCacheParity(unittest.TestCase):
    def test_boundary_precision_recall_per_book(self):
        for corpus in list_corpora():
            for manifest_key, expected_path, _book in available_public_books(corpus):
                with self.subTest(book=f"{corpus}/{manifest_key}"):
                    expected = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
                    pages = public_pages_for(corpus, manifest_key)
                    result = analyze_attachment(pages)

                    metrics = precision_recall_f1(expected, result["chapters"])
                    print(f"{corpus}/{manifest_key}: precision={metrics.precision:.2f} recall={metrics.recall:.2f} "
                          f"({metrics.true_positives}/{metrics.found_count} found, "
                          f"{metrics.true_positives}/{metrics.expected_count} expected)")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run both integration tests**

```bash
uv run pytest tests/test_segmentation_accuracy.py tests/test_public_evaluation_cache_parity.py -q -s -m integration
```

Expected: both pass (19 subtests in the first, 17 in the second — `pending/`'s 2 books have no `.expected.json` so `available_books`/`available_public_books` naturally exclude them), with per-book precision/recall lines printed, prefixed by corpus name.

- [ ] **Step 4: Commit**

```bash
git add tests/test_segmentation_accuracy.py tests/test_public_evaluation_cache_parity.py
git commit -m "$(cat <<'EOF'
refactor: run the accuracy/parity integration tests across every corpus

subTest labels now include the corpus name for pinpointing failures.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Rewrite `evaluation/refresh_llm_cache.py`

**Files:**
- Modify: `evaluation/refresh_llm_cache.py`
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Rewrite the failing test file first**

Replace the full contents of `tests/test_refresh_llm_cache.py`:

```python
"""Unit tests for evaluation/refresh_llm_cache.py's pure logic: coverage
computation and cache upserts. The network-calling _main() orchestration
is exercised manually (see evaluation/README.md), not here."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.refresh_llm_cache import _all_cached_model_ids, _fully_covered_model_ids, _upsert_cache


class TestFullyCoveredModelIds(unittest.TestCase):
    def test_no_cache_files_means_nothing_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set(),
            )

    def test_model_present_in_every_book_is_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for key in ("book-a", "book-b"):
                (cache_dir / f"{key}.json").write_text(
                    json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                    encoding="utf-8",
                )
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x"},
            )

    def test_model_missing_from_one_book_is_not_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set(),
            )

    def test_a_book_with_no_cache_file_at_all_means_zero_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _fully_covered_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set(),
            )

    def test_books_from_different_corpus_cache_dirs_are_covered_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_a_dir = Path(tmp) / "corpus-a"
            corpus_b_dir = Path(tmp) / "corpus-b"
            corpus_a_dir.mkdir()
            corpus_b_dir.mkdir()
            (corpus_a_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (corpus_b_dir / "book-b.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _fully_covered_model_ids([(corpus_a_dir, "book-a"), (corpus_b_dir, "book-b")]), {"model-x"},
            )


class TestAllCachedModelIds(unittest.TestCase):
    def test_no_cache_files_means_no_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertEqual(_all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), set())

    def test_unions_model_ids_across_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(
                json.dumps({"models": {"model-y": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x", "model-y"},
            )

    def test_model_present_in_only_one_book_still_counts(self):
        # Unlike _fully_covered_model_ids (intersection), this is a union
        # -- a model doesn't need to be cached for EVERY book to count.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            (cache_dir / "book-b.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            self.assertEqual(
                _all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x"},
            )

    def test_a_book_with_no_cache_file_at_all_is_skipped_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                _all_cached_model_ids([(cache_dir, "book-a"), (cache_dir, "book-b")]), {"model-x"},
            )


class TestUpsertCache(unittest.TestCase):
    def test_creates_new_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _upsert_cache(cache_dir, "book-a", "model-x", [{"title": "Intro"}], 1.5, demand=0)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertEqual(data["models"]["model-x"]["elapsed_seconds"], 1.5)
            self.assertEqual(data["models"]["model-x"]["demand_at_run"], 0)

    def test_preserves_other_models_when_upserting(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "book-a.json").write_text(
                json.dumps({"models": {"model-old": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}}),
                encoding="utf-8",
            )
            _upsert_cache(cache_dir, "book-a", "model-new", [], 2.0, demand=1)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertIn("model-old", data["models"])
            self.assertIn("model-new", data["models"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test file to confirm it fails**

```bash
uv run pytest tests/test_refresh_llm_cache.py -q
```

Expected: FAIL -- `TypeError: _fully_covered_model_ids() takes 1 positional argument but ...` (the old signature took `manifest_keys: list[str]`, not `book_specs: list[tuple[Path, str]]`).

- [ ] **Step 3: Rewrite `evaluation/refresh_llm_cache.py`**

Replace the full contents:

```python
#!/usr/bin/env python3
"""Refreshes each corpus's llm-cache/ -- the only script in this repo that
spends real KISSKI API budget. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"LLM cache refresh" and
docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md.

Reads KISSKI_API_KEY from the environment. Locally, source it from
zotero-rag's .env, e.g.:

    export KISSKI_API_KEY=$(grep '^KISSKI_API_KEY=' ../zotero-rag/.env | cut -d= -f2-)
    uv run python evaluation/refresh_llm_cache.py --mode top5

In CI it comes from a repository secret (see
.github/workflows/refresh-llm-cache.yml). Not a pytest test.

--mode top5 (default): refreshes the current 5 least-busy models,
unconditionally, even if already cached -- a quick manual sanity check.

--mode fill-gaps: finds non-"very busy" models not yet cached for EVERY
book across every corpus's current public books, and runs up to 5 of
those -- how the cache grows to cover every model over time (see the
nightly schedule in the workflow above).

--mode full: re-runs EVERY model that already has at least one cached
entry (its full historical footprint), across all books in all corpora,
regardless of current busy/demand status -- use after a change to the
extraction logic itself (prompt, max_tokens, page selection, ...) makes
every existing cache entry potentially stale, not just the 5 models
--mode top5 happens to touch. A cached model no longer offered by KISSKI is
skipped with a warning (nothing to run it against).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapter_segmentation.segmentation import analyze_attachment_llm_only
from evaluation.harness import available_public_books, list_corpora, llm_cache_dir, public_pages_for
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5


class _OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) backed by
    any OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float,
        is_valid: Optional[Callable[[str], bool]] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def _fully_covered_model_ids(book_specs: list[tuple[Path, str]]) -> set[str]:
    """book_specs: [(cache_dir, manifest_key), ...] -- one entry per book,
    each carrying its own corpus's llm_cache_dir(). A model id counts as
    covered only if EVERY given book's cache entry already has it. A book
    with no cache file at all has zero coverage -- every model is still a
    gap for it."""
    per_book_model_ids = []
    for cache_dir, manifest_key in book_specs:
        cache_path = cache_dir / f"{manifest_key}.json"
        if not cache_path.exists():
            return set()
        models = json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})
        per_book_model_ids.append(set(models))
    return set.intersection(*per_book_model_ids) if per_book_model_ids else set()


def _all_cached_model_ids(book_specs: list[tuple[Path, str]]) -> set[str]:
    """Every model id with at least one cached entry across any book --
    the "full regeneration" scope (a model's full historical footprint),
    unlike _fully_covered_model_ids' intersection-based "covered
    everywhere" definition used for gap-filling."""
    ids: set[str] = set()
    for cache_dir, manifest_key in book_specs:
        cache_path = cache_dir / f"{manifest_key}.json"
        if not cache_path.exists():
            continue
        ids.update(json.loads(cache_path.read_text(encoding="utf-8")).get("models", {}))
    return ids


def _upsert_cache(cache_dir: Path, manifest_key: str, model_id: str, chapters: list[dict], elapsed_seconds: float, demand: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{manifest_key}.json"
    data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"models": {}}
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["models"][model_id] = {"chapters": chapters, "elapsed_seconds": elapsed_seconds, "demand_at_run": demand}
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def _main(mode: str, base_url: str) -> int:
    api_key = os.environ["KISSKI_API_KEY"]
    # (corpus, manifest_key, cache_dir) for every scorable book across every corpus.
    book_entries: list[tuple[str, str, Path]] = [
        (corpus, manifest_key, llm_cache_dir(corpus))
        for corpus in list_corpora()
        for manifest_key, _expected_path, _book in available_public_books(corpus)
    ]
    if not book_entries:
        print("No public-cache evaluation books present.")
        return 1
    book_specs = [(cache_dir, manifest_key) for _corpus, manifest_key, cache_dir in book_entries]

    all_models = fetch_kisski_models(base_url, api_key)
    if mode == "top5":
        selected = select_top5(all_models)
    elif mode == "fill-gaps":
        selected = select_gap_fill(all_models, _fully_covered_model_ids(book_specs))
    else:
        cached_ids = _all_cached_model_ids(book_specs)
        selected = select_full_regen(all_models, cached_ids)
        retired = sorted(cached_ids - {m.id for m in all_models})
        if retired:
            print(f"Skipping cached models no longer offered by KISSKI: {retired}")

    if not selected:
        if mode == "fill-gaps":
            print("No models to run (fill-gaps: every non-busy model already fully covered).")
        elif mode == "full":
            print("No models to run (full: no models are cached yet).")
        else:
            print("No models to run.")
        return 0

    print(f"Selected models: {[m.id for m in selected]}")
    for model in selected:
        llm_client = _OpenAICompatibleLLMClient(model=model.id, base_url=base_url, api_key=api_key)
        for corpus, manifest_key, cache_dir in book_entries:
            try:
                pages = public_pages_for(corpus, manifest_key)
                start = time.perf_counter()
                result = await analyze_attachment_llm_only(pages, llm_client)
                elapsed = time.perf_counter() - start
                _upsert_cache(cache_dir, manifest_key, model.id, result["chapters"], elapsed, model.demand)
                print(f"{corpus}/{manifest_key} / {model.id}: {len(result['chapters'])} chapters, {elapsed:.1f}s")
            except Exception as exc:
                # One book/model failure must not strand the whole batch or
                # discard cache entries already written for other books/
                # models in this same run -- same catch-log-continue
                # convention as generate_public_evaluation_cache.py.
                print(f"{corpus}/{manifest_key} / {model.id}: FAILED ({exc}) -- skipping")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["top5", "fill-gaps", "full"], default="top5")
    parser.add_argument("--base-url", default=DEFAULT_KISSKI_BASE_URL)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(mode=args.mode, base_url=args.base_url)))
```

- [ ] **Step 4: Run the test file to confirm it passes**

```bash
uv run pytest tests/test_refresh_llm_cache.py -q
```

Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "$(cat <<'EOF'
refactor: make refresh_llm_cache.py corpus-aware

Coverage/upsert helpers take an explicit cache_dir instead of a module-
level constant, so _main can pool (corpus, book) pairs across every
corpus while writing each result into that book's own corpus's llm-cache/.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Rewrite `evaluation/generate_report.py` for per-corpus pages + a landing page

**Files:**
- Modify: `evaluation/generate_report.py`
- Test: `tests/test_generate_report.py`

- [ ] **Step 1: Rewrite the failing test file first**

Replace the full contents of `tests/test_generate_report.py`:

```python
"""Unit tests for evaluation/generate_report.py -- the auto-published,
zero-API-call report covering heuristic, outline, and (if cached) the
best-performing LLM model, one page per evaluation corpus plus a landing
page."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.generate_report import _best_llm_model, generate, generate_corpus


def _expected_json(chapters: list[dict]) -> str:
    return json.dumps({"chapters": chapters})


def _write_corpus_fixture(root: Path, corpus: str, chapters: list[dict], pages: list[str]) -> None:
    cdir = root / corpus
    public_cache_dir = cdir / "public-cache"
    llm_cache_dir = cdir / "llm-cache"
    public_cache_dir.mkdir(parents=True)
    llm_cache_dir.mkdir(parents=True)
    (cdir / "manifest.json").write_text(
        json.dumps({"books": [{"filename": "book-a.pdf", "title": "Book A"}]}), encoding="utf-8",
    )
    (cdir / "book-a.expected.json").write_text(_expected_json(chapters), encoding="utf-8")
    (public_cache_dir / "book-a.pages.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")


class TestBestLlmModel(unittest.TestCase):
    def test_returns_none_with_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test-corpus" / "llm-cache").mkdir(parents=True)
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(
                    _best_llm_model("test-corpus", [("book-a", [{"pdf_start_index": 0, "pdf_end_index": 5}])]),
                )

    def test_picks_the_higher_scoring_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_cache_dir = root / "test-corpus" / "llm-cache"
            llm_cache_dir.mkdir(parents=True)
            expected = [{"pdf_start_index": 0, "pdf_end_index": 5}]
            (llm_cache_dir / "book-a.json").write_text(json.dumps({
                "models": {
                    "good-model": {"chapters": expected, "elapsed_seconds": 1.0, "demand_at_run": 0},
                    "bad-model": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0},
                }
            }), encoding="utf-8")
            with patch("evaluation.harness.CORPUS_ROOT", root):
                best = _best_llm_model("test-corpus", [("book-a", expected)])
            self.assertEqual(best, "good-model")


class TestGenerateCorpus(unittest.TestCase):
    def test_writes_main_report_and_llm_detail_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                wrote = generate_corpus("test-corpus", out_dir)

            self.assertTrue(wrote)
            self.assertTrue((out_dir / "index.html").exists())
            self.assertTrue((out_dir / "llm" / "index.html").exists())
            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("book-a", main_html)
            self.assertIn("N/A", main_html)  # outline has no cache entry in this fixture
            llm_html = (out_dir / "llm" / "index.html").read_text(encoding="utf-8")
            self.assertIn("No cached LLM results yet", llm_html)

    def test_main_report_includes_citation_accuracy_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3, "citation_pages": "1-4"}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.\n\n1", "2", "3", "4"])
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Start accuracy", main_html)
            self.assertIn("End accuracy", main_html)

    def test_returns_false_and_writes_nothing_for_a_corpus_with_no_scorable_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            cdir = root / "empty-corpus"
            cdir.mkdir(parents=True)
            (cdir / "manifest.json").write_text(json.dumps({"books": []}), encoding="utf-8")
            out_dir = tmp_path / "public" / "empty-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root):
                wrote = generate_corpus("empty-corpus", out_dir)

            self.assertFalse(wrote)
            self.assertFalse(out_dir.exists())


class TestGenerate(unittest.TestCase):
    def test_landing_page_links_only_to_corpora_with_scorable_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "scored-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            empty_dir = root / "empty-corpus"
            empty_dir.mkdir(parents=True)
            (empty_dir / "manifest.json").write_text(json.dumps({"books": []}), encoding="utf-8")
            out_dir = tmp_path / "public"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            self.assertTrue((out_dir / "scored-corpus" / "index.html").exists())
            self.assertFalse((out_dir / "empty-corpus" / "index.html").exists())
            landing_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("scored-corpus", landing_html)
            self.assertNotIn("empty-corpus", landing_html)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test file to confirm it fails**

```bash
uv run pytest tests/test_generate_report.py -q
```

Expected: FAIL -- `ImportError: cannot import name 'generate_corpus' from 'evaluation.generate_report'`.

- [ ] **Step 3: Rewrite `evaluation/generate_report.py`**

Replace the full contents:

```python
#!/usr/bin/env python3
"""Generates a prose-free static results page per evaluation corpus from
the committed public-cache data -- see design specs
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md and
docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md.
Runs the heuristic and outline strategies live (no network/API calls); if
a corpus's llm-cache/ has cached LLM results, folds in the single
best-performing cached model too. Writes public/<corpus>/index.html and
public/<corpus>/llm/index.html for every corpus with at least one
scorable book, plus a public/index.html landing page linking to each. No
LLM call anywhere in this path; a plain f-string template, no
templating-engine dependency.

    uv run python evaluation/generate_report.py --out public/
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapter_segmentation.segmentation import analyze_attachment, analyze_attachment_outline_only
from evaluation.harness import (
    available_public_books,
    list_corpora,
    llm_cache_dir,
    public_outline_candidates_for,
    public_pages_for,
)
from evaluation.metrics import CitationPageAggregate, MicroAggregate, citation_pages_metrics, precision_recall_f1
from evaluation.report_html import render_strategy_tables

HEURISTIC = "heuristic"
OUTLINE = "outline"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_llm_cache(corpus: str, manifest_key: str) -> dict:
    cache_path = llm_cache_dir(corpus) / f"{manifest_key}.json"
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})


def _best_llm_model(corpus: str, books: list[tuple[str, list[dict]]]) -> str | None:
    """books: [(manifest_key, expected_chapters)]. Picks the cached model
    with the highest micro-F1 aggregated across every book in this corpus
    that has a cache entry for it (a model with partial corpus coverage is
    still scored, on however many books it has -- see design spec's "LLM
    results cache"), ties broken by lower total time. Returns None if no
    book has any cached LLM result at all."""
    per_model_aggregate: dict[str, MicroAggregate] = {}
    for manifest_key, expected in books:
        for model_id, entry in _load_llm_cache(corpus, manifest_key).items():
            agg = per_model_aggregate.setdefault(model_id, MicroAggregate())
            agg.add(precision_recall_f1(expected, entry["chapters"]), entry["elapsed_seconds"])
    if not per_model_aggregate:
        return None
    return max(
        per_model_aggregate,
        key=lambda model_id: (
            per_model_aggregate[model_id].compute().f1,
            -per_model_aggregate[model_id].total_elapsed_seconds,
        ),
    )


def generate_corpus(corpus: str, out_dir: Path) -> bool:
    """Writes out_dir/index.html and out_dir/llm/index.html for one
    corpus. Returns False (and writes nothing) if the corpus has no
    scorable book yet, so callers can skip linking to it from the landing
    page."""
    books = available_public_books(corpus)
    if not books:
        return False
    expected_by_key = {
        key: json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
        for key, expected_path, _book in books
    }

    best_llm_model = _best_llm_model(corpus, list(expected_by_key.items()))
    llm_strategy_name = f"LLM ({best_llm_model})" if best_llm_model else None
    strategy_names = [HEURISTIC, OUTLINE] + ([llm_strategy_name] if llm_strategy_name else [])

    per_document: dict[str, dict] = {}
    heuristic_agg, outline_agg, llm_agg = MicroAggregate(), MicroAggregate(), MicroAggregate()
    heuristic_citation_agg, outline_citation_agg, llm_citation_agg = (
        CitationPageAggregate(), CitationPageAggregate(), CitationPageAggregate(),
    )

    for manifest_key, _expected_path, _book in books:
        pages = public_pages_for(corpus, manifest_key)
        expected = expected_by_key[manifest_key]
        cells: dict = {}

        start = time.perf_counter()
        heuristic_result = analyze_attachment(pages)
        heuristic_elapsed = time.perf_counter() - start
        heuristic_metrics = precision_recall_f1(expected, heuristic_result["chapters"])
        heuristic_agg.add(heuristic_metrics, heuristic_elapsed)
        heuristic_citation_agg.add(citation_pages_metrics(expected, heuristic_result["chapters"]))
        cells[HEURISTIC] = (heuristic_metrics, heuristic_elapsed)

        outline_candidates = public_outline_candidates_for(corpus, manifest_key)
        if outline_candidates:
            # Falsy also catches `[]` (a real, resolved cache entry for a
            # book whose PDF genuinely has no embedded outline) -- not just
            # `None` (no cache entry at all). Both mean "this strategy has
            # nothing to contribute for this book," so both render as N/A
            # and are excluded from the aggregate; folding `[]` into "ran
            # and scored 0" would silently drag the outline strategy's
            # aggregate F1 down by counting books it structurally can never
            # help with as failures.
            start = time.perf_counter()
            outline_result = analyze_attachment_outline_only(pages, outline_candidates)
            outline_elapsed = time.perf_counter() - start
            outline_metrics = precision_recall_f1(expected, outline_result["chapters"])
            outline_agg.add(outline_metrics, outline_elapsed)
            outline_citation_agg.add(citation_pages_metrics(expected, outline_result["chapters"]))
            cells[OUTLINE] = (outline_metrics, outline_elapsed)
        else:
            cells[OUTLINE] = None

        if llm_strategy_name:
            llm_entry = _load_llm_cache(corpus, manifest_key).get(best_llm_model)
            if llm_entry:
                llm_metrics = precision_recall_f1(expected, llm_entry["chapters"])
                llm_agg.add(llm_metrics, llm_entry["elapsed_seconds"])
                llm_citation_agg.add(citation_pages_metrics(expected, llm_entry["chapters"]))
                cells[llm_strategy_name] = (llm_metrics, llm_entry["elapsed_seconds"])
            else:
                cells[llm_strategy_name] = None

        per_document[manifest_key] = cells

    aggregates = {HEURISTIC: heuristic_agg.compute(), OUTLINE: outline_agg.compute()}
    aggregate_times = {HEURISTIC: heuristic_agg.total_elapsed_seconds, OUTLINE: outline_agg.total_elapsed_seconds}
    citation_aggregates = {HEURISTIC: heuristic_citation_agg.compute(), OUTLINE: outline_citation_agg.compute()}
    if llm_strategy_name:
        aggregates[llm_strategy_name] = llm_agg.compute()
        aggregate_times[llm_strategy_name] = llm_agg.total_elapsed_seconds
        citation_aggregates[llm_strategy_name] = llm_citation_agg.compute()

    description = f"""<p>Corpus: <code>{corpus}</code> (see the <a href="../index.html">corpus index</a>
for the others). Each book has a hand-verified <code>*.expected.json</code> ground truth (real
chapter boundaries as exact PDF page ranges). Each strategy below is run
independently against the same pages -- no pipeline merge/fallback logic
is involved, so this reflects each strategy's own standalone accuracy, not
a production routing decision. A match requires the exact same page range
-- no partial credit. For per-book root-cause notes, see
<a href="https://github.com/cboulanger/chapter-segmentation/blob/main/evaluation/RESULTS.md">RESULTS.md</a>.
The full breakdown of every LLM model ever evaluated (not just the best)
is at <a href="llm/index.html">llm/index.html</a>.</p>"""

    html = render_strategy_tables(
        title=f"chapter-segmentation: {corpus} corpus results",
        description_html=description,
        strategy_names=strategy_names,
        per_document=per_document,
        aggregates=aggregates,
        aggregate_times=aggregate_times,
        citation_aggregates=citation_aggregates,
    )
    html = html.replace(
        "</body></html>",
        f"<p>Generated from commit {_git_sha()}.</p></body></html>",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    _generate_llm_detail_page(corpus, out_dir, list(expected_by_key.items()))
    return True


def _generate_llm_detail_page(corpus: str, out_dir: Path, books: list[tuple[str, list[dict]]]) -> None:
    model_ids: set[str] = set()
    for manifest_key, _expected in books:
        model_ids.update(_load_llm_cache(corpus, manifest_key).keys())

    if not model_ids:
        html = (
            '<!doctype html><html><head><meta charset="utf-8">'
            f"<title>chapter-segmentation {corpus} LLM results</title></head><body>"
            f"<h1>chapter-segmentation: {corpus} LLM strategy results</h1>"
            "<p>No cached LLM results yet -- run "
            "<code>evaluation/refresh_llm_cache.py</code>.</p></body></html>"
        )
    else:
        per_document: dict[str, dict] = {}
        aggregates_acc = {model_id: MicroAggregate() for model_id in model_ids}
        citation_aggregates_acc = {model_id: CitationPageAggregate() for model_id in model_ids}
        for manifest_key, expected in books:
            cache = _load_llm_cache(corpus, manifest_key)
            cells: dict = {}
            for model_id in model_ids:
                entry = cache.get(model_id)
                if entry is None:
                    cells[model_id] = None
                    continue
                metrics = precision_recall_f1(expected, entry["chapters"])
                aggregates_acc[model_id].add(metrics, entry["elapsed_seconds"])
                citation_aggregates_acc[model_id].add(citation_pages_metrics(expected, entry["chapters"]))
                cells[model_id] = (metrics, entry["elapsed_seconds"])
            per_document[manifest_key] = cells

        aggregates = {model_id: acc.compute() for model_id, acc in aggregates_acc.items()}
        aggregate_times = {model_id: acc.total_elapsed_seconds for model_id, acc in aggregates_acc.items()}
        citation_aggregates = {model_id: acc.compute() for model_id, acc in citation_aggregates_acc.items()}
        html = render_strategy_tables(
            title=f"chapter-segmentation: {corpus} LLM strategy results (all cached models)",
            description_html=(
                "<p>Every KISSKI model ever evaluated by "
                "<code>evaluation/refresh_llm_cache.py</code> against the "
                f"<code>{corpus}</code> corpus, run standalone via "
                "<code>analyze_attachment_llm_only</code> (no heuristic fallback). "
                'See <a href="../index.html">the main report</a> for how the single '
                "best-performing model compares against the heuristic and outline "
                "strategies.</p>"
            ),
            strategy_names=sorted(model_ids),
            per_document=per_document,
            aggregates=aggregates,
            aggregate_times=aggregate_times,
            citation_aggregates=citation_aggregates,
        )

    llm_dir = out_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    (llm_dir / "index.html").write_text(html, encoding="utf-8")


def _write_landing_page(out_dir: Path, scored_corpora: list[str]) -> None:
    links = "".join(f'<li><a href="{corpus}/index.html">{corpus}</a></li>' for corpus in scored_corpora)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>chapter-segmentation evaluation results</title></head>
<body>
<h1>chapter-segmentation evaluation results</h1>
<p>One report per evaluation corpus -- see
<a href="https://github.com/cboulanger/chapter-segmentation/blob/main/evaluation/README.md">evaluation/README.md</a>
for what distinguishes them.</p>
<ul>
{links}
</ul>
<p>Generated from commit {_git_sha()}.</p>
</body></html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def generate(out_dir: Path) -> None:
    scored_corpora = [corpus for corpus in list_corpora() if generate_corpus(corpus, out_dir / corpus)]
    _write_landing_page(out_dir, scored_corpora)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="public")
    args = parser.parse_args()
    generate(Path(args.out))
```

- [ ] **Step 4: Run the test file to confirm it passes**

```bash
uv run pytest tests/test_generate_report.py -q
```

Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add evaluation/generate_report.py tests/test_generate_report.py
git commit -m "$(cat <<'EOF'
refactor: generate one report page per corpus plus a landing page

generate_corpus(corpus, out_dir) replaces the single-corpus generate();
generate() now loops every corpus with scorable books and writes a
public/index.html landing page linking to each.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Update `.github/workflows/refresh-llm-cache.yml`

**Files:**
- Modify: `.github/workflows/refresh-llm-cache.yml`

- [ ] **Step 1: Update the commit step's `git add` path**

Change:

```yaml
      - name: Commit and push
        if: always()
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add evaluation/llm-cache/
```

to:

```yaml
      - name: Commit and push
        if: always()
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add evaluation/corpus/*/llm-cache/
```

- [ ] **Step 2: Verify the workflow YAML is still well-formed**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/refresh-llm-cache.yml'))" && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/refresh-llm-cache.yml
git commit -m "$(cat <<'EOF'
ci: commit llm-cache from every corpus subfolder, not one flat directory

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Update `evaluation/README.md`

**Files:**
- Modify: `evaluation/README.md`

- [ ] **Step 1: Replace the manifest-schema and fetch-PDFs section**

Old text (lines 1-37 of the current file):

```markdown
# Evaluation data for book segmentation

Ground-truth chapter boundaries for each book are hand-verified and committed
alongside this README as `<filename-without-extension>.expected.json` (schema:
see `docs/superpowers/plans/2026-07-24-chapter-segmentation-linking.md` Task 30).
The PDFs themselves are gitignored (`*.pdf`, see `.gitignore` in this directory)
and are not shipped.

This README documents what the evaluation set is and how to run it -- it
changes rarely. For current precision/recall numbers, known gaps, and
investigation findings from the last time each evaluation was actually run,
see **`RESULTS.md`** in this directory instead -- that document is a
snapshot and is expected to be regenerated (or rewritten) whenever the
heuristics, the strategy pipeline, the extraction/OCR path, or the
evaluation set change.

`manifest.json` is the source for this evaluation set. Each entry has:

- `filename` — matches a `<name>.pdf` / `<name>.expected.json` pair here
- `title`, `language`, `extraction_type` (`native` or `scan`), `embedded_toc`
- `oa` — whether the book can be legally auto-downloaded
- `doi` — the book's DOI (used both as metadata and, for non-OA books, as
  the pointer a human follows to acquire it)
- `download_url` — direct PDF URL, only meaningful when `oa: true`; `null`
  otherwise

**Fetching the PDFs:**

```bash
uv run python evaluation/scripts/fetch_evaluation_pdfs.py
```

Downloads every `oa: true` entry that isn't already present. Non-OA entries
are never touched by this script — if one is missing, it prints the DOI and
the exact path to save the file to. Get that book through your institution's
legal access (library subscription, interlibrary loan, etc.), save it there,
and re-run the tests.

**Adding a new evaluation book** (e.g. a "difficult" PDF the segmentation
heuristics scored low-confidence on during live testing against a real
Zotero library) — see `CLAUDE.md` in this directory for the full step-by-step
workflow, including the `evaluation/scripts/ground_truth_helper.py` draft-then-verify
process and known failure modes. Short version:

1. Has a DOI? Add an entry to the committed `manifest.json` (`"oa": false,
   "download_url": null` if it can't be freely redistributed — that's fully
   supported, it just means `fetch_evaluation_pdfs.py` won't auto-download
   it, only print the DOI for manual acquisition). No DOI, or can't be
   identified/shared at all? Add it to `manifest.local.json` instead (same
   schema, gitignored, never committed — see `CLAUDE.md`) so it's still
   exercised by your own local test runs.
2. Place (or fetch) the PDF at `<filename>` here.
3. Build `<name>.expected.json` by actually inspecting the real PDF — never
   guessed or extrapolated from the TOC alone (`CLAUDE.md` explains why).
```

New text:

```markdown
# Evaluation data for book segmentation

Every evaluation book lives under `evaluation/corpus/<name>/` -- a
self-contained subfolder per corpus (see "Corpora" below). Ground-truth
chapter boundaries are hand-verified and committed as
`<filename-without-extension>.expected.json` inside that corpus's
directory (schema: see
`docs/superpowers/plans/2026-07-24-chapter-segmentation-linking.md` Task
30). The PDFs themselves are gitignored (`*.pdf`, see `.gitignore` in this
directory) and are not shipped. See
`docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md` for
the full rationale behind the per-corpus split.

This README documents what the evaluation set is and how to run it -- it
changes rarely. For current precision/recall numbers, known gaps, and
investigation findings from the last time each evaluation was actually run,
see **`RESULTS.md`** in this directory instead -- that document is a
snapshot and is expected to be regenerated (or rewritten) whenever the
heuristics, the strategy pipeline, the extraction/OCR path, or the
evaluation set change.

## Corpora

`evaluation.harness.list_corpora()` auto-discovers every subfolder of
`evaluation/corpus/` that has a `manifest.json` -- every runner below loops
over all of them by default, with an optional `--corpus <name>` flag to
restrict to one. Three corpora exist today:

- **`open-access/`** (6 books) -- well-produced, OA, parseable embedded
  TOCs. The case the pure-heuristic pipeline already handles well.
- **`copyrighted/`** (11 books) -- sourced from a real personal Zotero
  library: no DOI, `embedded_toc: false`, native and scanned, German and
  English. The case the outline/Crossref/Zotero-catalog strategies, the
  layout-mode extraction fallback, and the evaluation OCR route were built
  for.
- **`pending/`** (2 books) -- have a manifest entry and PDF but no
  `.expected.json` yet, so they contribute to no evaluation until someone
  builds ground truth for them (see `CLAUDE.md`'s "Step 0"), at which point
  the entry moves into whichever real corpus it belongs in.

Each corpus directory has the same shape:

```text
evaluation/corpus/<name>/
  manifest.json          # committed -- the source for this corpus
  manifest.local.json    # optional, gitignored (same schema, see below)
  <isbn>.pdf              # gitignored
  <isbn>.expected.json    # committed (except in pending/)
  public-cache/
  .ocr-cache/             # gitignored
  llm-cache/
```

`manifest.json` entries have:

- `filename` — matches a `<name>.pdf` / `<name>.expected.json` pair in the
  same corpus directory
- `title`, `language`, `extraction_type` (`native` or `scan`), `embedded_toc`
- `oa` — whether the book can be legally auto-downloaded
- `doi` — the book's DOI (used both as metadata and, for non-OA books, as
  the pointer a human follows to acquire it)
- `download_url` — direct PDF URL, only meaningful when `oa: true`; `null`
  otherwise

**Fetching the PDFs:**

```bash
uv run python evaluation/scripts/fetch_evaluation_pdfs.py
uv run python evaluation/scripts/fetch_evaluation_pdfs.py --corpus open-access   # just one corpus
```

Downloads every `oa: true` entry that isn't already present. Non-OA entries
are never touched by this script — if one is missing, it prints the DOI and
the exact path to save the file to. Get that book through your institution's
legal access (library subscription, interlibrary loan, etc.), save it at the
printed path, and re-run the tests.

**Adding a new evaluation book** (e.g. a "difficult" PDF the segmentation
heuristics scored low-confidence on during live testing against a real
Zotero library) — see `CLAUDE.md` in this directory for the full step-by-step
workflow, including which corpus it belongs in, the
`evaluation/scripts/ground_truth_helper.py` draft-then-verify process, and
known failure modes. Short version:

1. Decide the corpus (`open-access`/`copyrighted`/`pending` -- see
   `CLAUDE.md`'s "Step 0").
2. Has a DOI? Add an entry to that corpus's committed `manifest.json`
   (`"oa": false, "download_url": null` if it can't be freely
   redistributed — that's fully supported, it just means
   `fetch_evaluation_pdfs.py` won't auto-download it, only print the DOI
   for manual acquisition). No DOI, or can't be identified/shared at all?
   Add it to that corpus's `manifest.local.json` instead (same schema,
   gitignored, never committed — see `CLAUDE.md`) so it's still exercised
   by your own local test runs.
3. Place (or fetch) the PDF at `evaluation/corpus/<corpus>/<filename>`.
4. Build `evaluation/corpus/<corpus>/<name>.expected.json` by actually
   inspecting the real PDF — never guessed or extrapolated from the TOC
   alone (`CLAUDE.md` explains why).
```

- [ ] **Step 2: Update the one remaining bare-path reference**

After Step 1's rewrite, exactly one bare-path reference remains further
down the file (in the "LLM strategy evaluation" section). Confirm and fix it:

```bash
grep -n "evaluation/manifest\|evaluation/public-cache\|evaluation/llm-cache\|evaluation/\.ocr-cache\|evaluation/<filename>\|evaluation/<name>" evaluation/README.md
```

Expected: one match, `evaluation/llm-cache/<book>.json`. Change:

```markdown
Unlike the heuristic and outline strategies, evaluating the LLM strategy
costs real KISSKI API budget, so it is decoupled from report generation:
`evaluation/refresh_llm_cache.py` is the only script that calls an LLM,
and it writes its results into `evaluation/llm-cache/<book>.json` (raw
```

to:

```markdown
Unlike the heuristic and outline strategies, evaluating the LLM strategy
costs real KISSKI API budget, so it is decoupled from report generation:
`evaluation/refresh_llm_cache.py` is the only script that calls an LLM,
and it writes its results into `evaluation/corpus/<corpus>/llm-cache/<book>.json` (raw
```

Also update the "Per-strategy evaluation report" section's description of what `generate_report.py` produces -- change:

```markdown
Produces two pages, both using the same table format: `public/index.html`
(one row per book x strategy, with precision/recall/F1/time, best-F1 cell
per row marked, plus a per-strategy summary ordered by aggregate F1) and
`public/llm/index.html` (see "LLM strategy evaluation" below).
```

to:

```markdown
Produces a `public/index.html` landing page linking to one report per
corpus (`public/<corpus>/index.html`, one row per book x strategy, with
precision/recall/F1/time, best-F1 cell per row marked, plus a per-strategy
summary ordered by aggregate F1) and `public/<corpus>/llm/index.html` per
corpus (see "LLM strategy evaluation" below).
```

- [ ] **Step 3: Update "Evaluation set composition"**

Change the section heading and lead-in from describing "the committed
manifest.json set" and "manifest.local.json" as two separate historical
sources to describing them as the `open-access/` and `copyrighted/`
corpora respectively (the two per-book tables that already exist in this
section stay as-is, just re-scoped under the new heading names -- no book
data changes).

Replace:

```markdown
## Evaluation set composition

The committed `manifest.json` set (7 books) is small and, per the design
```

with:

```markdown
## Evaluation set composition

The `open-access/` corpus (6 books) is small and, per the design
```

Replace:

```markdown
`manifest.local.json` (gitignored, per `CLAUDE.md`'s "no DOI" rule above)
adds 10 more books sourced directly from a real personal Zotero library,
```

with:

```markdown
The `copyrighted/` corpus (11 books) is sourced directly from a real
personal Zotero library,
```

- [ ] **Step 4: Commit**

```bash
git add evaluation/README.md
git commit -m "$(cat <<'EOF'
docs: update evaluation/README.md for the corpus/<name>/ layout

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Update `evaluation/CLAUDE.md`

**Files:**
- Modify: `evaluation/CLAUDE.md`

- [ ] **Step 1: Add a corpus-selection step above the existing "Step 0"**

Change:

```markdown
## Step 0: Decide where the book's metadata goes

- **Has a DOI, OR already has a `public-cache/` entry?** → Add it to the
```

to:

```markdown
## Step 0a: Decide which corpus the book belongs in

Every evaluation book lives under `evaluation/corpus/<corpus>/` -- see
`docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md`.
Before anything else, pick one:

- **OA, or otherwise well-produced with a parseable embedded/printed TOC**
  → `open-access/`.
- **Everything else that you can build real ground truth for** (no DOI,
  no embedded TOC, scanned, sourced from a personal library, ...) →
  `copyrighted/`.
- **No ground truth built yet** (you only have the PDF and basic metadata
  so far) → `pending/`. Move the entry into `open-access/` or
  `copyrighted/` once its `.expected.json` exists.

Every path in this document below (`evaluation/<filename>`,
`evaluation/manifest.local.json`, etc.) means
`evaluation/corpus/<corpus>/<filename>` for whichever corpus you picked
here.

## Step 0b: Decide where the book's metadata goes

- **Has a DOI, OR already has a `public-cache/` entry?** → Add it to the
```

- [ ] **Step 2: Update the remaining path references in "Step 0b"**

Find every remaining bare `evaluation/` path reference in the rest of the
file:

```bash
grep -n "evaluation/<filename>\|evaluation/manifest\|Place the PDF" evaluation/CLAUDE.md
```

Update the sentence:

```markdown
Place the PDF itself directly in this directory (`evaluation/<filename>`) — both `.gitignore` entries (`*.pdf`, `manifest.local.json`) mean neither the file nor its local-only metadata are ever committed.
```

to:

```markdown
Place the PDF itself directly in that corpus's directory
(`evaluation/corpus/<corpus>/<filename>`) — both `.gitignore` entries
(`*.pdf`, `manifest.local.json`) mean neither the file nor its local-only
metadata are ever committed.
```

- [ ] **Step 3: Update the "Document organization" section's `public-cache/` bullet**

Change:

```markdown
- **`public-cache/`** — a redacted, git-tracked snapshot of each book's
  page text (real navigational/bibliographic material verbatim, chapter
  prose replaced with random real words) — see
  `docs/superpowers/specs/2026-08-05-evaluation-corpus-redaction-design.md`.
```

to:

```markdown
- **`public-cache/`** (per corpus, i.e.
  `evaluation/corpus/<corpus>/public-cache/`) — a redacted, git-tracked
  snapshot of each book's page text (real navigational/bibliographic
  material verbatim, chapter prose replaced with random real words) — see
  `docs/superpowers/specs/2026-08-05-evaluation-corpus-redaction-design.md`.
```

The rest of that bullet's content (the redaction/outline-snapshot
explanation, the `9783031466373` `--verify` exception) is unchanged, since
none of it is path-specific.

- [ ] **Step 4: Commit**

```bash
git add evaluation/CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: update evaluation/CLAUDE.md for the corpus/<name>/ layout

Adds a "which corpus does this book belong in" decision above the
existing DOI/public-cache branch.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Light-touch update to `evaluation/RESULTS.md`

**Files:**
- Modify: `evaluation/RESULTS.md`

Per the design spec, `RESULTS.md` keeps its existing two per-corpus result
sections and prose as-is (no book, cache entry, or scoring logic changed)
-- only the handful of command/path references need updating.

- [ ] **Step 1: Add a one-line pointer near the top**

After the existing "Always-current numbers" callout line (line 12), add:

```markdown
> **Layout note:** every evaluation book now lives under
> `evaluation/corpus/<name>/` (`open-access`, `copyrighted`, `pending`) --
> see `docs/superpowers/specs/2026-08-08-multi-corpus-evaluation-design.md`.
> The two result sections below ("Pure-heuristic results" and "Diverse
> real-library evaluation set") correspond to the `open-access` and
> `copyrighted` corpora respectively.
```

- [ ] **Step 2: Update the four command/path references found earlier**

```bash
grep -n "evaluation/\.ocr-cache/\|evaluation/llm-cache/\|evaluation/public-cache/" evaluation/RESULTS.md
```

For each of the three matches (around lines 325-326, 381, 414), change the
bare path to note it's now per-corpus, e.g.:

Change:

```markdown
  `evaluation/.ocr-cache/` and populated by
```

to:

```markdown
  each corpus's `evaluation/corpus/<name>/.ocr-cache/` and populated by
```

Change:

```markdown
populating `evaluation/llm-cache/` (LLM) -- each strategy run independently
```

to:

```markdown
populating each corpus's `evaluation/corpus/<name>/llm-cache/` (LLM) -- each strategy run independently
```

Change:

```markdown
  (`evaluation/public-cache/<key>.outline.json` is `{"candidates": []}`),
```

to:

```markdown
  (`evaluation/corpus/<name>/public-cache/<key>.outline.json` is `{"candidates": []}`),
```

- [ ] **Step 3: Commit**

```bash
git add evaluation/RESULTS.md
git commit -m "$(cat <<'EOF'
docs: point RESULTS.md at the new evaluation/corpus/<name>/ paths

Light touch only -- the two existing result sections and their numbers
are unchanged, since no book, cache entry, or scoring logic changed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit test suite**

```bash
uv run pytest -q
```

Expected: PASS, no failures. (Integration tests are excluded by default per `pyproject.toml`'s `addopts`.)

- [ ] **Step 2: Run both integration tests**

```bash
uv run pytest tests/test_segmentation_accuracy.py tests/test_public_evaluation_cache_parity.py -q -s -m integration
```

Expected: PASS. 19 subtests in the first (across all three corpora, `pending/`'s 2 books skipped since they have no `.expected.json`), 17 in the second.

- [ ] **Step 3: Generate the new per-corpus reports**

```bash
uv run python evaluation/generate_report.py --out /tmp/after-report
find /tmp/after-report -name "index.html"
```

Expected: `/tmp/after-report/index.html` (landing page), `/tmp/after-report/open-access/index.html`, `/tmp/after-report/open-access/llm/index.html`, `/tmp/after-report/copyrighted/index.html`, `/tmp/after-report/copyrighted/llm/index.html`. No `pending/` subdirectory (it has no scorable books).

- [ ] **Step 4: Diff per-book numbers against the Task 1 baseline**

The baseline (`/tmp/baseline-report/index.html`) was one combined table;
the new output splits the same books across
`/tmp/after-report/open-access/index.html` and
`/tmp/after-report/copyrighted/index.html`. Since no scoring code, page
data, or ground truth changed -- only which file each book's data lives in
-- every book's precision/recall/F1 numbers must be byte-identical before
and after. Spot-check a sample from each corpus:

```bash
for book in 9783031466373 9783322969828 9783848736829 dnb-36942798X; do
  echo "=== $book ==="
  grep -A1 "$book" /tmp/baseline-report/index.html | grep "P=" 
  grep -A1 "$book" /tmp/after-report/open-access/index.html /tmp/after-report/copyrighted/index.html 2>/dev/null | grep "P="
done
```

Expected: for each book, the `P=... R=... F1=...` line from the baseline
matches the line found in whichever corpus page now contains that book.

- [ ] **Step 5: Confirm the CI workflow files are still valid YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/publish-results.yml')); yaml.safe_load(open('.github/workflows/refresh-llm-cache.yml'))" && echo OK
```

Expected: `OK`.

- [ ] **Step 6: Clean up temp verification directories**

```bash
rm -rf /tmp/baseline-report /tmp/after-report
```

No commit for this task -- it's verification only, confirming Tasks 1-15 didn't change any evaluation number, only where the data lives.
