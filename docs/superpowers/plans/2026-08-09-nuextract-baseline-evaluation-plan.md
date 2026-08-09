# NuExtract-1.5-tiny Zero-Shot TOC-Extraction Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, manually-run evaluation script that measures NuExtract-1.5-tiny's zero-shot precision/recall at extracting TOC entries (title + printed page number) from the chapter-segmentation evaluation corpora, as the go/no-go signal for further NuExtract investment (spec: `docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md`).

**Architecture:** A small pure-logic library module (`evaluation/nuextract_baseline.py` — prompt building, response parsing, scoring, and the Ollama HTTP call) backed by unit tests with a mocked HTTP layer, plus a thin CLI runner (`evaluation/scripts/evaluate_nuextract_baseline.py`) that loops over `evaluation/corpus/*/` the same way `evaluate_chapter_segmentation_strategies.py` already does. No production code (`segmentation.py`, `cli.py`) is touched.

**Tech Stack:** Python 3.12, `httpx` (already used elsewhere in `evaluation/`), `rapidfuzz` (already a dependency), a local Ollama server, `pytest`/`unittest.mock` for the new unit tests.

**One deviation from the spec, decided during planning:** The spec's "Serving" section anticipated needing to convert the Hugging Face checkpoint to GGUF via `llama.cpp` if no Ollama tag existed. Research during planning found that's unnecessary: `numind/NuExtract-1.5-tiny` already has third-party GGUF builds on Hugging Face (e.g. `QuantFactory/NuExtract-1.5-tiny-GGUF`, confirmed to contain a `NuExtract-1.5-tiny.Q8_0.gguf` file), and Ollama can pull a GGUF file directly from a Hugging Face repo with `ollama pull hf.co/<repo>:<quant>` — no local conversion step needed. (Ollama's own published `nuextract` tag is a *different*, older, larger model — the original phi-3-based NuExtract v1 — not NuExtract-1.5-tiny, so it must not be used here.) This still satisfies the spec's actual constraint ("not automated — no network fetch of arbitrary model conversions belongs in a test/eval script"): the pull remains a manual, one-time, documented step: `ollama pull hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0`.

---

### Task 1: Prompt-building helper

**Files:**
- Create: `evaluation/nuextract_baseline.py`
- Test: `tests/test_nuextract_baseline.py`

NuExtract does not follow chat instructions — it fills a JSON template from a template+text pair, using the exact input format documented on its model card:

```
<|input|>
### Template:
{template_json}
### Text:
{text}

<|output|>
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nuextract_baseline.py`:

```python
"""Unit tests for evaluation/nuextract_baseline.py -- NuExtract-1.5-tiny
zero-shot TOC-extraction baseline spike. See design spec
docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md."""

import unittest
from unittest.mock import MagicMock, patch

from evaluation.nuextract_baseline import build_prompt


class TestBuildPrompt(unittest.TestCase):
    def test_includes_template_and_page_text(self):
        pages = ["front matter", "Contents\nIntro ... 1", "back matter"]
        prompt = build_prompt(pages, [1])
        self.assertIn("### Template:", prompt)
        self.assertIn('"chapters"', prompt)
        self.assertIn("### Text:", prompt)
        self.assertIn("Contents\nIntro ... 1", prompt)
        self.assertTrue(prompt.startswith("<|input|>"))
        self.assertTrue(prompt.endswith("<|output|>"))

    def test_joins_multiple_scan_indices_in_order_and_skips_others(self):
        pages = ["p0", "p1", "p2"]
        prompt = build_prompt(pages, [0, 2])
        text_section = prompt.split("### Text:\n")[1]
        self.assertTrue(text_section.startswith("p0"))
        self.assertIn("p2", text_section)
        self.assertNotIn("p1", text_section)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: `ModuleNotFoundError: No module named 'evaluation.nuextract_baseline'`

- [ ] **Step 3: Create the library module with the template and `build_prompt`**

Create `evaluation/nuextract_baseline.py`:

```python
"""NuExtract-1.5-tiny zero-shot TOC-extraction baseline spike. See design
spec docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md.

Scores only the TOC-*listing* step (title + printed_page_number pairs)
llm_extract_toc_entries (segmentation.py) would replace -- not full
chapter-boundary localization. This module never runs
_locate_toc_entries/_chapters_from_located.
"""

import json

import httpx
from rapidfuzz import fuzz

from chapter_segmentation._llm_json import parse_json_object
from chapter_segmentation.evidence.fusion import _ALIGN_SCORE_THRESHOLD
from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.metrics import Metrics

# Mirrors TocEntry's fields (segmentation.py) -- title/authors/
# printed_page_number, the same shape llm_extract_toc_entries asks a
# cloud LLM for, formatted as NuExtract's own template convention instead
# of a free-form instruction.
NUEXTRACT_TEMPLATE = {
    "chapters": [
        {"title": "", "authors": [""], "printed_page_number": ""},
    ],
}


def build_prompt(pages: list[str], scan_indices: list[int]) -> str:
    """Formats NuExtract's documented <|input|>/### Template/### Text/
    <|output|> input convention. Unlike _LLM_TOC_EXTRACTION_PROMPT
    (segmentation.py), no "[PAGE i]" markers are included -- NuExtract
    copies values verbatim from the text rather than following
    instructions, and this spike does not use page indices at all (see
    module docstring), only whatever printed page number already appears
    next to each title in the raw text."""
    text = "\n\n".join(pages[i] for i in scan_indices)
    template_json = json.dumps(NUEXTRACT_TEMPLATE)
    return f"<|input|>\n### Template:\n{template_json}\n### Text:\n{text}\n\n<|output|>"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/nuextract_baseline.py tests/test_nuextract_baseline.py
git commit -m "feat: add NuExtract prompt-building helper for baseline spike"
```

---

### Task 2: Response parsing

**Files:**
- Modify: `evaluation/nuextract_baseline.py`
- Modify: `tests/test_nuextract_baseline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nuextract_baseline.py`:

```python
from evaluation.nuextract_baseline import build_prompt, parse_response


class TestParseResponse(unittest.TestCase):
    def test_extracts_chapters_list(self):
        raw = '{"chapters": [{"title": "Intro", "authors": ["A"], "printed_page_number": "1"}]}'
        self.assertEqual(
            parse_response(raw),
            [{"title": "Intro", "authors": ["A"], "printed_page_number": "1"}],
        )

    def test_strips_code_fence(self):
        raw = '```json\n{"chapters": []}\n```'
        self.assertEqual(parse_response(raw), [])

    def test_returns_empty_on_malformed_json(self):
        self.assertEqual(parse_response("not json at all"), [])

    def test_returns_empty_when_chapters_not_a_list(self):
        self.assertEqual(parse_response('{"chapters": "oops"}'), [])
```

(Keep the existing `TestBuildPrompt` class above it in the same file.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: `ImportError: cannot import name 'parse_response'`

- [ ] **Step 3: Add `parse_response`**

Add to `evaluation/nuextract_baseline.py`, after `build_prompt`:

```python
def parse_response(raw: str) -> list[dict]:
    """Parses NuExtract's filled-template output into the "chapters" list.
    Returns [] for empty/unparseable output or a malformed "chapters"
    field -- treated as "no signal", mirroring llm_extract_toc_entries'
    own failure handling (segmentation.py) rather than raising."""
    try:
        data = parse_json_object(raw)
    except ValueError:
        return []
    chapters = data.get("chapters")
    return chapters if isinstance(chapters, list) else []
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/nuextract_baseline.py tests/test_nuextract_baseline.py
git commit -m "feat: parse NuExtract's filled-template response"
```

---

### Task 3: Title+page matching

**Files:**
- Modify: `evaluation/nuextract_baseline.py`
- Modify: `tests/test_nuextract_baseline.py`

Per the spec's "Scoring" section: match a predicted entry against ground
truth using the same fuzzy title-match convention `fusion.py`'s `_align`
already uses (`rapidfuzz.fuzz.token_sort_ratio`, threshold
`_ALIGN_SCORE_THRESHOLD` = 70), **plus** an exact printed-page-number
match. `expected.json` stores the page as `citation_pages` (e.g.
`"1-31"` or `"vii-ix"`) — only the first half is the chapter's own start
page.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nuextract_baseline.py`:

```python
from evaluation.nuextract_baseline import build_prompt, match_toc_entries, parse_response


class TestMatchTocEntries(unittest.TestCase):
    def test_matches_on_title_and_page(self):
        predicted = [{"title": "Introduction", "printed_page_number": "1"}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 1)

    def test_rejects_page_mismatch(self):
        predicted = [{"title": "Introduction", "printed_page_number": "2"}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)

    def test_rejects_title_mismatch(self):
        predicted = [{"title": "Completely Different", "printed_page_number": "1"}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)

    def test_handles_roman_numerals(self):
        predicted = [{"title": "Foreword", "printed_page_number": "vii"}]
        expected = [{"title": "Foreword", "citation_pages": "vii-ix"}]
        self.assertEqual(match_toc_entries(predicted, expected), 1)

    def test_is_monotonic_like_fusion_align(self):
        # "Chapter Two" is predicted first and consumes expected[1]; the
        # later "Chapter One" prediction can then only search from
        # expected[2:] onward (mirrors fusion._align's "TOC order is book
        # order" assumption), so it finds nothing even though a textual
        # match for it exists earlier in the expected list.
        predicted = [
            {"title": "Chapter Two", "printed_page_number": "20"},
            {"title": "Chapter One", "printed_page_number": "1"},
        ]
        expected = [
            {"title": "Chapter One", "citation_pages": "1-19"},
            {"title": "Chapter Two", "citation_pages": "20-39"},
        ]
        self.assertEqual(match_toc_entries(predicted, expected), 1)

    def test_null_printed_page_number_never_matches(self):
        predicted = [{"title": "Introduction", "printed_page_number": None}]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)

    def test_null_citation_pages_never_matches(self):
        predicted = [{"title": "Introduction", "printed_page_number": "1"}]
        expected = [{"title": "Introduction", "citation_pages": None}]
        self.assertEqual(match_toc_entries(predicted, expected), 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: `ImportError: cannot import name 'match_toc_entries'`

- [ ] **Step 3: Add the page-parsing helpers and `match_toc_entries`**

Add to `evaluation/nuextract_baseline.py`, after `parse_response`:

```python
def _expected_start_page(citation_pages) -> int | None:
    """The chapter's own start page from an expected.json entry's
    "citation_pages" field (e.g. "1-31" -> 1, "vii-ix" -> 7). None if the
    field is null or has no "-" separator."""
    if not citation_pages or "-" not in citation_pages:
        return None
    start, _, _ = citation_pages.partition("-")
    return _parse_toc_page_number(start.strip())


def _predicted_page(printed_page_number) -> int | None:
    """The parsed page number from a NuExtract-predicted entry's
    "printed_page_number" field. None for a missing/null/empty value or
    one _parse_toc_page_number can't interpret."""
    if printed_page_number is None:
        return None
    text = str(printed_page_number).strip()
    if not text or text.lower() == "null":
        return None
    return _parse_toc_page_number(text)


def match_toc_entries(predicted: list[dict], expected: list[dict]) -> int:
    """Counts true positives: a predicted entry matches the next
    unmatched expected chapter whose start page (see
    _expected_start_page) is identical to the predicted entry's own page
    (see _predicted_page) AND whose title fuzzy-matches at or above
    fusion.py's own _ALIGN_SCORE_THRESHOLD. Greedy and order-preserving --
    once an expected index is matched, no earlier expected index can be
    matched again -- exactly mirroring fusion._align's "TOC listing order
    is book order" assumption. An entry on either side with no parseable
    page number can never match (see the two "null ... never matches"
    tests) -- this is a real scope limitation of this metric, not a bug:
    it means a chapter with no visible printed page number is
    unreachable by this scoring, consistent with the spec's "Scoring"
    section requiring an exact page match."""
    last_j = -1
    matched = 0
    for pred in predicted:
        pred_page = _predicted_page(pred.get("printed_page_number"))
        pred_title = str(pred.get("title") or "").strip().lower()
        if pred_page is None or not pred_title:
            continue
        best_j = None
        best_score = 0.0
        for j in range(last_j + 1, len(expected)):
            exp_page = _expected_start_page(expected[j].get("citation_pages"))
            if exp_page is None or exp_page != pred_page:
                continue
            score = fuzz.token_sort_ratio(pred_title, str(expected[j].get("title") or "").strip().lower())
            if score >= _ALIGN_SCORE_THRESHOLD and score > best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            matched += 1
            last_j = best_j
    return matched
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/nuextract_baseline.py tests/test_nuextract_baseline.py
git commit -m "feat: add fuzzy title+page matching for NuExtract baseline scoring"
```

---

### Task 4: Per-book scoring

**Files:**
- Modify: `evaluation/nuextract_baseline.py`
- Modify: `tests/test_nuextract_baseline.py`

Wraps `match_toc_entries` into the same `Metrics` shape
`evaluation/metrics.py` already uses elsewhere, so the CLI runner can
reuse `MicroAggregate` for corpus/grand-total aggregation instead of
reimplementing pooling.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nuextract_baseline.py`:

```python
from evaluation.nuextract_baseline import build_prompt, match_toc_entries, parse_response, score_book


class TestScoreBook(unittest.TestCase):
    def test_computes_precision_and_recall(self):
        predicted = [
            {"title": "Introduction", "printed_page_number": "1"},
            {"title": "Spurious Entry", "printed_page_number": "99"},
        ]
        expected = [{"title": "Introduction", "citation_pages": "1-31"}]
        metrics = score_book(predicted, expected)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.found_count, 2)
        self.assertEqual(metrics.expected_count, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 1.0)

    def test_empty_predicted_scores_zero_precision_and_recall(self):
        metrics = score_book([], [{"title": "Introduction", "citation_pages": "1-31"}])
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: `ImportError: cannot import name 'score_book'`

- [ ] **Step 3: Add `score_book`**

Add to `evaluation/nuextract_baseline.py`, after `match_toc_entries`:

```python
def score_book(predicted: list[dict], expected_chapters: list[dict]) -> Metrics:
    """Precision/recall/F1 over TOC-listing entries (title +
    printed_page_number), NOT chapter-boundary ranges -- a narrower
    metric than precision_recall_f1 (evaluation/metrics.py), which scores
    (pdf_start_index, pdf_end_index) exact matches instead. This spike
    never produces pdf indices at all (see module docstring)."""
    tp = match_toc_entries(predicted, expected_chapters)
    found_count = len(predicted)
    expected_count = len(expected_chapters)
    precision = tp / found_count if found_count else 0.0
    recall = tp / expected_count if expected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(
        precision=precision, recall=recall, f1=f1,
        true_positives=tp, found_count=found_count, expected_count=expected_count,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/nuextract_baseline.py tests/test_nuextract_baseline.py
git commit -m "feat: add per-book Metrics scoring for NuExtract baseline"
```

---

### Task 5: Ollama call

**Files:**
- Modify: `evaluation/nuextract_baseline.py`
- Modify: `tests/test_nuextract_baseline.py`

Calls Ollama's `/api/generate` endpoint in **raw mode** (`"raw": true`),
which bypasses Ollama's own chat templating entirely so NuExtract's
`<|input|>`/`<|output|>` format (built by `build_prompt`) reaches the
model verbatim — the same shape the model card's own `transformers`
example constructs by hand. `num_ctx` is set explicitly because Ollama's
default context window (2048) is too small for a multi-page TOC scan and
would silently truncate the input.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nuextract_baseline.py`:

```python
from evaluation.nuextract_baseline import (
    build_prompt, call_ollama, match_toc_entries, parse_response, score_book,
)


class TestCallOllama(unittest.TestCase):
    def test_posts_raw_mode_request_to_generate_endpoint(self):
        response = MagicMock()
        response.json.return_value = {"response": '{"chapters": []}'}
        response.raise_for_status.return_value = None
        with patch("evaluation.nuextract_baseline.httpx.post", return_value=response) as mock_post:
            result = call_ollama("http://localhost:11434", "hf.co/example:Q8_0", "prompt-text")
        self.assertEqual(result, '{"chapters": []}')
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertIs(kwargs["json"]["raw"], True)
        self.assertEqual(kwargs["json"]["prompt"], "prompt-text")
        self.assertEqual(kwargs["json"]["model"], "hf.co/example:Q8_0")
        self.assertIs(kwargs["json"]["stream"], False)

    def test_strips_trailing_slash_on_base_url(self):
        response = MagicMock()
        response.json.return_value = {"response": ""}
        response.raise_for_status.return_value = None
        with patch("evaluation.nuextract_baseline.httpx.post", return_value=response) as mock_post:
            call_ollama("http://localhost:11434/", "model", "prompt")
        self.assertEqual(mock_post.call_args[0][0], "http://localhost:11434/api/generate")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: `ImportError: cannot import name 'call_ollama'`

- [ ] **Step 3: Add `call_ollama`**

Add to `evaluation/nuextract_baseline.py`, after `score_book`:

```python
def call_ollama(base_url: str, model: str, prompt: str, timeout: float = 300.0) -> str:
    """POSTs to Ollama's /api/generate in raw mode -- bypasses Ollama's
    own chat templating so NuExtract's own <|input|>/<|output|> format
    (see build_prompt) reaches the model verbatim. num_ctx is set well
    above Ollama's 2048-token default since a multi-page TOC scan can
    easily exceed that; num_predict/temperature follow the model card's
    own documented generation config (max_new_tokens=4000, temperature
    at/near 0)."""
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4000, "num_ctx": 8192},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract_baseline.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/nuextract_baseline.py tests/test_nuextract_baseline.py
git commit -m "feat: add Ollama raw-mode generate call for NuExtract baseline"
```

---

### Task 6: CLI runner script

**Files:**
- Create: `evaluation/scripts/evaluate_nuextract_baseline.py`

Mirrors `evaluation/scripts/evaluate_chapter_segmentation_strategies.py`'s
shape (loop `list_corpora()` → `available_books(corpus)` →
`analysis_pages_for`), but calls NuExtract via Ollama instead of the
strategy pipeline, and prints a per-corpus + grand-total aggregate using
`MicroAggregate` (same pooling style `evaluation/metrics.py` already
provides for the other reports).

- [ ] **Step 1: Write the script**

Create `evaluation/scripts/evaluate_nuextract_baseline.py`:

```python
#!/usr/bin/env python3
"""Zero-shot NuExtract-1.5-tiny TOC-extraction baseline over the
chapter-segmentation evaluation corpora. See design spec
docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md.

Scores only the TOC-*listing* step (title + printed_page_number pairs)
llm_extract_toc_entries (segmentation.py) would replace -- not full
chapter-boundary localization. This script never runs
_locate_toc_entries/_chapters_from_located, and does not touch
segmentation.py, cli.py, or any production strategy code.

One-time setup: pull the model into a local Ollama server. There is no
official Ollama tag for NuExtract-1.5-tiny -- Ollama's own "nuextract" tag
is a different, older, larger model (the original phi-3-based NuExtract
v1). Pull a third-party GGUF build straight from Hugging Face instead:

    ollama pull hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0

Then run (needs the evaluation PDFs -- see evaluation/README.md's
"Fetching the PDFs"):

    uv run python evaluation/scripts/evaluate_nuextract_baseline.py
    uv run python evaluation/scripts/evaluate_nuextract_baseline.py --corpus open-access
    uv run python evaluation/scripts/evaluate_nuextract_baseline.py --model hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q4_K_M
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.segmentation import _llm_scan_indices
from evaluation.harness import analysis_pages_for, available_books, list_corpora
from evaluation.metrics import MicroAggregate
from evaluation.nuextract_baseline import build_prompt, call_ollama, parse_response, score_book

_DEFAULT_MODEL = "hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _run_corpus(corpus: str, model: str, ollama_url: str, timeout: float, grand_total: MicroAggregate) -> None:
    triples = available_books(corpus)
    if not triples:
        return
    print(f"=== {corpus} ===")
    corpus_total = MicroAggregate()
    for pdf_path, expected_path, _book in triples:
        expected_chapters = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
        file_bytes = pdf_path.read_bytes()
        pages = analysis_pages_for(corpus, file_bytes)
        if pages is None:
            print(f"{pdf_path.name}: SKIPPED (needs OCR -- see evaluation/README.md)")
            continue
        scan_indices = _llm_scan_indices(pages)
        if not scan_indices:
            print(f"{pdf_path.name}: SKIPPED (no TOC-scan pages found)")
            continue
        prompt = build_prompt(pages, scan_indices)
        try:
            raw = call_ollama(ollama_url, model, prompt, timeout=timeout)
        except Exception as exc:
            print(f"{pdf_path.name}: FAILED ({exc})")
            continue
        predicted = parse_response(raw)
        metrics = score_book(predicted, expected_chapters)
        corpus_total.add(metrics)
        grand_total.add(metrics)
        print(
            f"{pdf_path.name}: precision={metrics.precision:.2f} recall={metrics.recall:.2f} "
            f"({metrics.true_positives}/{metrics.found_count} found, "
            f"{metrics.true_positives}/{metrics.expected_count} expected)"
        )
    agg = corpus_total.compute()
    print(f"[{corpus}] aggregate: precision={agg.precision:.2f} recall={agg.recall:.2f} f1={agg.f1:.2f}\n")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", help="Only evaluate this corpus (default: every corpus)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help=f"Ollama model tag (default: {_DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=_DEFAULT_OLLAMA_URL, help=f"Ollama server base URL (default: {_DEFAULT_OLLAMA_URL})")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-book request timeout in seconds (default: 300)")
    args = parser.parse_args()

    corpora = [args.corpus] if args.corpus else list_corpora()
    if not any(available_books(corpus) for corpus in corpora):
        print("No evaluation PDFs present -- run: uv run python evaluation/scripts/fetch_evaluation_pdfs.py")
        return 1

    grand_total = MicroAggregate()
    for corpus in corpora:
        _run_corpus(corpus, args.model, args.ollama_url, args.timeout, grand_total)

    total = grand_total.compute()
    print(f"=== TOTAL: precision={total.precision:.2f} recall={total.recall:.2f} f1={total.f1:.2f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 2: Sanity-check the script imports and parses args cleanly**

Run: `uv run python evaluation/scripts/evaluate_nuextract_baseline.py --help`
Expected: prints the argparse usage/help text (module docstring plus
`--corpus`/`--model`/`--ollama-url`/`--timeout`) and exits 0 — this
confirms every import resolves without needing a running Ollama server or
any PDFs present yet.

- [ ] **Step 3: Commit**

```bash
git add evaluation/scripts/evaluate_nuextract_baseline.py
git commit -m "feat: add NuExtract baseline CLI runner"
```

---

### Task 7: Document the spike in evaluation/README.md

**Files:**
- Modify: `evaluation/README.md`

- [ ] **Step 1: Add a new subsection**

In `evaluation/README.md`, immediately after the existing "### LLM
strategy evaluation" subsection (the one describing
`evaluation/refresh_llm_cache.py`, ending around the KISSKI
`--mode fill-gaps` paragraph) and before "### Strategy-pipeline
evaluation", insert:

```markdown
### NuExtract baseline spike

`evaluation/scripts/evaluate_nuextract_baseline.py` measures
NuExtract-1.5-tiny's zero-shot accuracy at the TOC-*listing* extraction
step `llm_extract_toc_entries` (segmentation.py) performs today via a
cloud LLM -- title + printed_page_number pairs only, not full
chapter-boundary localization. See
`docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md`
for the full rationale and decision criteria.

One-time setup -- pull the model into a local Ollama server (no official
Ollama tag exists for NuExtract-1.5-tiny; Ollama's own `nuextract` tag is
a different, older, larger model):

```bash
ollama pull hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0
```

Then run, with the evaluation PDFs present (see "Fetching the PDFs"
above) and Ollama serving locally:

```bash
uv run python evaluation/scripts/evaluate_nuextract_baseline.py
```

Not a pytest test and not part of any CI workflow -- a manual, one-off
measurement, same operational pattern as the LLM strategy evaluation
above but with no API cost.
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/README.md
git commit -m "docs: document the NuExtract baseline spike"
```

---

### Task 8: Manual smoke test (requires a local Ollama server)

This task cannot be executed by an agent without a running Ollama
server and local model weights — it must be run by a human (or an agent
with genuine local Ollama access) on the development machine.

- [ ] **Step 1: Pull the model**

```bash
ollama pull hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0
```

Expected: Ollama downloads the ~676MB Q8_0 GGUF file and registers it
under that tag (`ollama list` should then show it).

- [ ] **Step 2: Fetch at least one open-access evaluation PDF if not already present**

```bash
uv run python evaluation/scripts/fetch_evaluation_pdfs.py --corpus open-access
```

- [ ] **Step 3: Run the script against just that corpus**

```bash
uv run python evaluation/scripts/evaluate_nuextract_baseline.py --corpus open-access
```

Expected: one printed line per book (`precision=... recall=...`), an
`[open-access] aggregate: ...` line, and a final `=== TOTAL: ... ===`
line — no unhandled exceptions. A `FAILED (...)` line for a specific book
is fine to investigate separately; a total failure (e.g. a connection
error on every book) usually means Ollama isn't running
(`ollama serve`) or the model tag is wrong (`ollama list` to check).

- [ ] **Step 4: Spot-check one book's output makes sense**

Compare one book's printed precision/recall against its
`evaluation/corpus/open-access/<isbn>.expected.json` chapter count by eye
— confirms the scoring isn't systematically broken (e.g. every book
scoring exactly 0.00 would indicate a real bug, not just a weak model).

No commit for this task — it's a verification step, not a code change.

---

### Task 9: Run the full baseline and record the result

Requires Task 8's setup to already be working. Also run against
`copyrighted-scans` (needs those PDFs acquired per `evaluation/README.md`
— skip any book you don't have locally; the script already
skips missing/OCR-pending books gracefully).

- [ ] **Step 1: Run the full corpus**

```bash
uv run python evaluation/scripts/evaluate_nuextract_baseline.py
```

- [ ] **Step 2: Record the result in evaluation/RESULTS.md**

Per `evaluation/CLAUDE.md`'s document-organization rules, a newly-measured
number belongs in `RESULTS.md`, not `README.md`. Add a new top-level
section (placement: after the existing per-strategy sections, before any
trailing appendix) using this structure, filling in the real numbers from
Step 1's output:

```markdown
## NuExtract-1.5-tiny zero-shot baseline (<today's date>)

`evaluation/scripts/evaluate_nuextract_baseline.py` against Ollama
serving `hf.co/QuantFactory/NuExtract-1.5-tiny-GGUF:Q8_0`. Scores only
TOC-listing extraction (title + printed_page_number), not full
chapter-boundary localization -- see
docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md.

| Corpus | Precision | Recall | F1 |
| --- | --- | --- | --- |
| open-access | <fill in> | <fill in> | <fill in> |
| copyrighted-scans | <fill in> | <fill in> | <fill in> |
| **Total** | **<fill in>** | **<fill in>** | **<fill in>** |

Decision (per the spec's "Decision criteria"): <promising -- in the
neighborhood of the heuristic pipeline's 0.91/0.91 aggregate, or
meaningfully better than 0 on copyrighted-scans / not promising --
fill in based on the actual numbers above, and note which case applies>.
```

- [ ] **Step 3: Commit**

```bash
git add evaluation/RESULTS.md
git commit -m "docs: record NuExtract-1.5-tiny zero-shot baseline results"
```

---

## Self-review notes

- **Spec coverage:** script path (Task 6) ✓; page selection reuses
  `_llm_scan_indices` unchanged (Task 6) ✓; template mirrors `TocEntry`
  fields (Task 1) ✓; NuExtract's documented template+text input
  convention, not the free-form chat prompt (Task 1) ✓; local Ollama
  serving (Tasks 5/6/8, with the documented HF-direct-pull deviation
  noted at the top of this plan) ✓; scoring uses the same fuzzy
  title-match convention + threshold as `fusion.py`'s `_align`, plus an
  exact `printed_page_number` match (Task 3) ✓; per-corpus + aggregate
  reporting in the same table shape other evaluation scripts use (Task
  6) ✓; non-goals (no fine-tuning, no production wiring, no multimodal,
  no pytest/CI integration) — nothing in this plan touches
  `segmentation.py`/`cli.py`, adds no image handling, and the CLI script
  lives under `evaluation/scripts/` (outside `pyproject.toml`'s
  `testpaths = ["tests"]`) so it's never pytest-collected ✓; decision
  criteria applied in Task 9 ✓.
- **Placeholder scan:** the only "fill in" markers are in Task 9's
  results template, which by definition can't be filled before the
  script actually runs — not a plan-authoring placeholder.
- **Type consistency:** `Metrics`/`MicroAggregate` (from
  `evaluation/metrics.py`) are used with the same field names
  (`precision`, `recall`, `f1`, `true_positives`, `found_count`,
  `expected_count`) across Task 4's `score_book` and Task 6's CLI script.
  `build_prompt`/`parse_response`/`match_toc_entries`/`score_book`/
  `call_ollama` signatures introduced in Tasks 1–5 are used identically
  in Task 6's `_run_corpus`.
