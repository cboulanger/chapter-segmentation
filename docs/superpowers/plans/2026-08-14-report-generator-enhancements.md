# Report Generator Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a build-date footer, per-model "as of" freshness dates on the LLM columns, and a layout/TOC classifier column (with a non-comparability note) to the published evaluation report.

**Architecture:** All changes live in `evaluation/generate_report.py` (orchestration), `evaluation/report_html.py` (shared table renderer), `evaluation/refresh_llm_cache.py` (adds a per-model timestamp to the LLM cache schema), and `evaluation/scripts/evaluate_layout_toc_classifier.py` (adds a `--save-results` flag that persists a new `evaluation/corpus/<corpus>/classifier-results.json` artifact, read by `generate_report.py`). No new files besides that JSON artifact's schema.

**Tech Stack:** Python 3.12, stdlib `unittest`, `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-08-14-report-generator-enhancements-design.md`

---

### Task 1: Footer generation date

**Files:**
- Modify: `evaluation/generate_report.py:1-46` (imports, add `_today` helper), `evaluation/generate_report.py:174-177` (`generate_corpus` footer), `evaluation/generate_report.py:243-259` (`_write_landing_page` footer)
- Test: `tests/test_generate_report.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_generate_report.py`, inside `class TestGenerateCorpus` (after `test_writes_main_report_and_llm_detail_page`):

```python
    def test_main_report_footer_includes_generation_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertRegex(main_html, r"Generated on \d{4}-\d{2}-\d{2} from commit")
```

Add to `tests/test_generate_report.py`, inside `class TestGenerate` (after `test_landing_page_links_only_to_corpora_with_scorable_books`):

```python
    def test_landing_page_footer_includes_generation_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "scored-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            out_dir = tmp_path / "public"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate(out_dir)

            landing_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertRegex(landing_html, r"Generated on \d{4}-\d{2}-\d{2} from commit")
```

Also add `generate` to the existing import line at the top of the file (it's already imported: `from evaluation.generate_report import _best_llm_model, generate, generate_corpus` -- no change needed there).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_report.py -k generation_date -v`
Expected: FAIL (`Generated from commit ...` does not match `Generated on \d{4}-\d{2}-\d{2} from commit`)

- [ ] **Step 3: Implement**

In `evaluation/generate_report.py`, change the imports block (currently just `import argparse`, `import json`, `import subprocess`, `import sys`, `import time`, `from pathlib import Path`) to add a datetime import:

```python
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
```

Add a new helper right after `_git_sha`:

```python
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
```

In `generate_corpus`, change:

```python
    html = html.replace(
        "</body></html>",
        f"<p>Generated from commit {_git_sha()}.</p></body></html>",
    )
```

to:

```python
    html = html.replace(
        "</body></html>",
        f"<p>Generated on {_today()} from commit {_git_sha()}.</p></body></html>",
    )
```

In `_write_landing_page`, change:

```python
<p>Generated from commit {_git_sha()}.</p>
</body></html>
"""
```

to:

```python
<p>Generated on {_today()} from commit {_git_sha()}.</p>
</body></html>
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_report.py -v`
Expected: PASS (all tests, including the two new ones and every pre-existing one in this file)

- [ ] **Step 5: Commit**

```bash
git add evaluation/generate_report.py tests/test_generate_report.py
git commit -m "feat: add generation date to report footer"
```

---

### Task 2: Per-model timestamp in the LLM cache

**Files:**
- Modify: `evaluation/refresh_llm_cache.py:128-134` (`_upsert_cache`)
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_refresh_llm_cache.py`, inside `class TestUpsertCache` (after `test_preserves_other_models_when_upserting`):

```python
    def test_records_a_per_model_generated_at_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _upsert_cache(cache_dir, "book-a", "model-x", [], 1.0, demand=0)
            data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            self.assertRegex(data["models"]["model-x"]["generated_at"], r"^\d{4}-\d{2}-\d{2}T")

    def test_a_second_models_generated_at_does_not_overwrite_the_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            _upsert_cache(cache_dir, "book-a", "model-old", [], 1.0, demand=0)
            first_data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))
            first_timestamp = first_data["models"]["model-old"]["generated_at"]

            _upsert_cache(cache_dir, "book-a", "model-new", [], 1.0, demand=0)
            second_data = json.loads((cache_dir / "book-a.json").read_text(encoding="utf-8"))

            self.assertEqual(second_data["models"]["model-old"]["generated_at"], first_timestamp)
            self.assertIn("generated_at", second_data["models"]["model-new"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_refresh_llm_cache.py -k generated_at -v`
Expected: FAIL (`KeyError: 'generated_at'`)

- [ ] **Step 3: Implement**

In `evaluation/refresh_llm_cache.py`, change `_upsert_cache`:

```python
def _upsert_cache(cache_dir: Path, manifest_key: str, model_id: str, chapters: list[dict], elapsed_seconds: float, demand: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{manifest_key}.json"
    data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"models": {}}
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["models"][model_id] = {"chapters": chapters, "elapsed_seconds": elapsed_seconds, "demand_at_run": demand}
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

to:

```python
def _upsert_cache(cache_dir: Path, manifest_key: str, model_id: str, chapters: list[dict], elapsed_seconds: float, demand: int) -> None:
    """Writes/updates model_id's cache entry for one book. Each model entry
    carries its own generated_at timestamp (not just the file-level one,
    which reflects whichever model was upserted most recently across the
    whole file) -- generate_report.py needs a per-model date to show how
    fresh THAT model's specific numbers are, since different models in the
    same file can have been refreshed on different nights."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{manifest_key}.json"
    data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"models": {}}
    now = datetime.now(timezone.utc).isoformat()
    data["generated_at"] = now
    data["models"][model_id] = {
        "chapters": chapters,
        "elapsed_seconds": elapsed_seconds,
        "demand_at_run": demand,
        "generated_at": now,
    }
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v`
Expected: PASS (all tests in this file)

- [ ] **Step 5: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "feat: record a per-model timestamp in the LLM cache"
```

---

### Task 3: "As of" date on the main report's merged LLM column

**Files:**
- Modify: `evaluation/generate_report.py:78-94` (`generate_corpus`, add `_latest_model_date` helper and use it)
- Test: `tests/test_generate_report.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_generate_report.py`, a new test class after `class TestBestLlmModel` (before `class TestGenerateCorpus`):

```python
class TestLatestModelDate(unittest.TestCase):
    def test_returns_none_when_no_entry_has_a_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_cache_dir = root / "test-corpus" / "llm-cache"
            llm_cache_dir.mkdir(parents=True)
            (llm_cache_dir / "book-a.json").write_text(json.dumps({
                "models": {"model-x": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0}}
            }), encoding="utf-8")
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(_latest_model_date("test-corpus", "model-x", ["book-a"]))

    def test_returns_the_max_date_across_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_cache_dir = root / "test-corpus" / "llm-cache"
            llm_cache_dir.mkdir(parents=True)
            (llm_cache_dir / "book-a.json").write_text(json.dumps({
                "models": {"model-x": {
                    "chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0,
                    "generated_at": "2026-08-10T00:00:00+00:00",
                }}
            }), encoding="utf-8")
            (llm_cache_dir / "book-b.json").write_text(json.dumps({
                "models": {"model-x": {
                    "chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0,
                    "generated_at": "2026-08-14T00:00:00+00:00",
                }}
            }), encoding="utf-8")
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertEqual(_latest_model_date("test-corpus", "model-x", ["book-a", "book-b"]), "2026-08-14")

    def test_ignores_books_missing_the_requested_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm_cache_dir = root / "test-corpus" / "llm-cache"
            llm_cache_dir.mkdir(parents=True)
            (llm_cache_dir / "book-a.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertIsNone(_latest_model_date("test-corpus", "model-x", ["book-a"]))
```

Add `_latest_model_date` to the existing import line:

```python
from evaluation.generate_report import _best_llm_model, _latest_model_date, generate, generate_corpus
```

Add to `class TestGenerateCorpus` (after `test_writes_main_report_and_llm_detail_page`):

```python
    def test_main_report_llm_column_shows_as_of_date_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            (root / "test-corpus" / "llm-cache" / "book-a.json").write_text(json.dumps({
                "models": {"good-model": {
                    "chapters": chapters, "elapsed_seconds": 1.0, "demand_at_run": 0,
                    "generated_at": "2026-08-14T03:00:00+00:00",
                }}
            }), encoding="utf-8")
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("LLM (good-model, as of 2026-08-14)", main_html)

    def test_main_report_llm_column_falls_back_without_a_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            # Old-format cache entry: no per-model "generated_at" key.
            (root / "test-corpus" / "llm-cache" / "book-a.json").write_text(json.dumps({
                "models": {"good-model": {"chapters": chapters, "elapsed_seconds": 1.0, "demand_at_run": 0}}
            }), encoding="utf-8")
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("LLM (good-model)", main_html)
            self.assertNotIn("as of", main_html)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_report.py -k "LatestModelDate or as_of_date or falls_back_without" -v`
Expected: FAIL (`ImportError: cannot import name '_latest_model_date'`)

- [ ] **Step 3: Implement**

In `evaluation/generate_report.py`, add this helper right after `_best_llm_model` (which ends at line 75):

```python
def _latest_model_date(corpus: str, model_id: str, manifest_keys: list[str]) -> str | None:
    """Latest per-model generated_at (date part only, YYYY-MM-DD) across
    every book's cache entry for model_id -- "since not all model runs
    necessarily take place at the same time, use the latest date" (books
    can be refreshed for the same model on different nights). Returns None
    if no book's entry for this model carries the field at all (cache
    entries written before the per-model timestamp was added)."""
    dates = [
        entry["generated_at"][:10]
        for manifest_key in manifest_keys
        if (entry := _load_llm_cache(corpus, manifest_key).get(model_id)) and "generated_at" in entry
    ]
    return max(dates) if dates else None
```

In `generate_corpus`, change:

```python
    best_llm_model = _best_llm_model(corpus, list(expected_by_key.items()))
    llm_strategy_name = f"LLM ({best_llm_model})" if best_llm_model else None
    strategy_names = [HEURISTIC, OUTLINE] + ([llm_strategy_name] if llm_strategy_name else [])
```

to:

```python
    best_llm_model = _best_llm_model(corpus, list(expected_by_key.items()))
    llm_strategy_name = None
    if best_llm_model:
        llm_date = _latest_model_date(corpus, best_llm_model, list(expected_by_key.keys()))
        llm_strategy_name = (
            f"LLM ({best_llm_model}, as of {llm_date})" if llm_date else f"LLM ({best_llm_model})"
        )
    strategy_names = [HEURISTIC, OUTLINE] + ([llm_strategy_name] if llm_strategy_name else [])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_report.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/generate_report.py tests/test_generate_report.py
git commit -m "feat: show the LLM cache's freshest date on the merged model column"
```

---

### Task 4: "As of" dates on the LLM detail page's per-model columns

**Files:**
- Modify: `evaluation/generate_report.py:186-240` (`_generate_llm_detail_page`)
- Test: `tests/test_generate_report.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestGenerateCorpus` in `tests/test_generate_report.py` (after the two tests added in Task 3):

```python
    def test_llm_detail_page_shows_per_model_as_of_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            (root / "test-corpus" / "llm-cache" / "book-a.json").write_text(json.dumps({
                "models": {
                    "model-fresh": {
                        "chapters": chapters, "elapsed_seconds": 1.0, "demand_at_run": 0,
                        "generated_at": "2026-08-14T00:00:00+00:00",
                    },
                    "model-old": {"chapters": [], "elapsed_seconds": 1.0, "demand_at_run": 0},
                }
            }), encoding="utf-8")
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            llm_html = (out_dir / "llm" / "index.html").read_text(encoding="utf-8")
            self.assertIn("model-fresh (as of 2026-08-14)", llm_html)
            self.assertIn(">model-old<", llm_html)  # no date available -- bare id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_report.py -k per_model_as_of_dates -v`
Expected: FAIL (`AssertionError: 'model-fresh (as of 2026-08-14)' not found`)

- [ ] **Step 3: Implement**

In `evaluation/generate_report.py`, change `_generate_llm_detail_page`'s `else` branch (the one that runs when `model_ids` is non-empty) from:

```python
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
```

to:

```python
    else:
        manifest_keys = [manifest_key for manifest_key, _expected in books]
        label_by_model = {
            model_id: (
                f"{model_id} (as of {date})"
                if (date := _latest_model_date(corpus, model_id, manifest_keys))
                else model_id
            )
            for model_id in model_ids
        }

        per_document: dict[str, dict] = {}
        aggregates_acc = {label: MicroAggregate() for label in label_by_model.values()}
        citation_aggregates_acc = {label: CitationPageAggregate() for label in label_by_model.values()}
        for manifest_key, expected in books:
            cache = _load_llm_cache(corpus, manifest_key)
            cells: dict = {}
            for model_id, label in label_by_model.items():
                entry = cache.get(model_id)
                if entry is None:
                    cells[label] = None
                    continue
                metrics = precision_recall_f1(expected, entry["chapters"])
                aggregates_acc[label].add(metrics, entry["elapsed_seconds"])
                citation_aggregates_acc[label].add(citation_pages_metrics(expected, entry["chapters"]))
                cells[label] = (metrics, entry["elapsed_seconds"])
            per_document[manifest_key] = cells

        aggregates = {label: acc.compute() for label, acc in aggregates_acc.items()}
        aggregate_times = {label: acc.total_elapsed_seconds for label, acc in aggregates_acc.items()}
        citation_aggregates = {label: acc.compute() for label, acc in citation_aggregates_acc.items()}
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
            strategy_names=sorted(label_by_model.values()),
            per_document=per_document,
            aggregates=aggregates,
            aggregate_times=aggregate_times,
            citation_aggregates=citation_aggregates,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_report.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/generate_report.py tests/test_generate_report.py
git commit -m "feat: show per-model as-of dates on the LLM detail page"
```

---

### Task 5: `render_strategy_tables` gains an optional classifier column/row

**Files:**
- Modify: `evaluation/report_html.py` (whole file rewritten below)
- Test: `tests/test_report_html.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_report_html.py`, at the end of `class TestRenderStrategyTables` (before the closing of the class, i.e. after `test_omits_citation_accuracy_columns_when_not_provided`):

```python
    def test_renders_classifier_column_and_row(self):
        classifier = {
            "label": "Layout/TOC classifier (LOBO, as of 2026-08-14)",
            "note": (
                "The layout/TOC classifier row measures per-page classification "
                "recall via leave-one-book-out cross-validation -- not directly "
                "comparable to the chapter-boundary precision/recall/F1 columns above."
            ),
            "per_document": {
                "book-a": {"toc_recall": 1.0, "chapter_first_recall": 0.9, "candidate_fraction": 0.08},
            },
            "full_recall_fraction": 0.67,
            "avg_candidate_fraction": 0.09,
        }
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic"],
            per_document={"book-a": {"heuristic": (_metrics(1.0, 1.0, 1.0), 1.0)}},
            aggregates={"heuristic": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0},
            classifier=classifier,
        )
        self.assertIn("Layout/TOC classifier (LOBO, as of 2026-08-14)", html)
        self.assertIn("TOC recall=100%, chapter-first recall=90%, candidates=8%", html)
        self.assertIn("Full recall", html)
        self.assertIn("Avg candidates", html)
        self.assertIn("67%", html)
        self.assertIn(">9%<", html)
        self.assertIn("not directly comparable", html)

    def test_renders_na_for_book_missing_from_classifier_results(self):
        classifier = {
            "label": "Classifier",
            "note": "note text",
            "per_document": {},
            "full_recall_fraction": 0.5,
            "avg_candidate_fraction": 0.1,
        }
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic"],
            per_document={"book-a": {"heuristic": (_metrics(1.0, 1.0, 1.0), 1.0)}},
            aggregates={"heuristic": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0},
            classifier=classifier,
        )
        doc_section = html[html.index("Per document"):html.index("Per strategy")]
        self.assertIn("<td>N/A</td>", doc_section)

    def test_omits_classifier_extras_when_not_provided(self):
        html = render_strategy_tables(
            title="Test report", description_html="",
            strategy_names=["heuristic"],
            per_document={},
            aggregates={"heuristic": _metrics(1.0, 1.0, 1.0)},
            aggregate_times={"heuristic": 1.0},
        )
        self.assertNotIn("Full recall", html)
        self.assertNotIn("Avg candidates", html)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report_html.py -k classifier -v`
Expected: FAIL (`TypeError: render_strategy_tables() got an unexpected keyword argument 'classifier'`)

- [ ] **Step 3: Implement**

Replace the entire contents of `evaluation/report_html.py` with:

```python
"""Shared HTML table rendering for generate_report.py's main report and
its LLM detail page -- one renderer so both pages look and behave
identically. See design specs
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"Metrics and rendering (shared code)" and
docs/superpowers/specs/2026-08-14-report-generator-enhancements-design.md
(the optional classifier column/row).
"""

from evaluation.metrics import CitationPageMetrics, Metrics

_TableCell = tuple[Metrics, float] | None  # (metrics, elapsed_seconds), or None for "not run"


def _cell_html(cell: _TableCell, is_best: bool) -> str:
    if cell is None:
        return "<td>N/A</td>"
    metrics, elapsed_seconds = cell
    style = ' style="background:#e6ffe6; font-weight:bold;"' if is_best else ""
    return (
        f"<td{style}>P={metrics.precision:.2f} R={metrics.recall:.2f} F1={metrics.f1:.2f}<br>"
        f"{metrics.true_positives}/{metrics.found_count} found, "
        f"{metrics.true_positives}/{metrics.expected_count} expected<br>"
        f"{elapsed_seconds:.2f}s</td>"
    )


def _classifier_cell_html(entry: dict | None) -> str:
    """entry: {"toc_recall": float | None, "chapter_first_recall": float |
    None, "candidate_fraction": float} for one book, or None if that book
    wasn't part of the classifier's last leave-one-book-out run."""
    if entry is None:
        return "<td>N/A</td>"

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    return (
        f"<td>TOC recall={fmt(entry['toc_recall'])}, "
        f"chapter-first recall={fmt(entry['chapter_first_recall'])}, "
        f"candidates={entry['candidate_fraction']:.0%}</td>"
    )


def render_strategy_tables(
    title: str,
    description_html: str,
    strategy_names: list[str],
    per_document: dict[str, dict[str, _TableCell]],
    aggregates: dict[str, Metrics],
    aggregate_times: dict[str, float],
    citation_aggregates: dict[str, CitationPageMetrics] | None = None,
    classifier: dict | None = None,
) -> str:
    """
    strategy_names: column order for the per-document table.
    per_document: {document_key: {strategy_name: (Metrics, elapsed_seconds) or None}}.
        None (or a missing key) means the strategy produced no result for
        that document -- rendered as "N/A".
    aggregates: {strategy_name: Metrics} -- micro-aggregate across every
        document that strategy actually ran on.
    aggregate_times: {strategy_name: total_elapsed_seconds}.
    citation_aggregates: optional {strategy_name: CitationPageMetrics} --
        when given, adds "Start accuracy"/"End accuracy" columns to the
        aggregate table (see design spec 2026-08-08). Omitted entirely
        (no extra columns) when not given, so existing callers/tests are
        unaffected.
    classifier: optional {"label": str, "note": str, "per_document":
        {document_key: {"toc_recall", "chapter_first_recall",
        "candidate_fraction"} | None}, "full_recall_fraction": float,
        "avg_candidate_fraction": float} -- the layout/TOC classifier's
        leave-one-book-out results (see design spec 2026-08-14). Its
        metric (per-page classification recall) isn't the same shape as
        the other strategies' chapter-boundary precision/recall/F1, so it
        gets its own cell format in the per-document table, its own two
        extra columns in the aggregate table ("Full recall"/"Avg
        candidates", rendered "N/A" for every non-classifier row), and
        `note` rendered as a caveat directly above the aggregate table.
        Omitted entirely (no extra column/row/note) when not given.
    Returns a full <html> document string.
    """
    doc_rows = []
    for doc_key in sorted(per_document):
        cells = per_document[doc_key]
        best_f1 = max(
            (cell[0].f1 for cell in cells.values() if cell is not None),
            default=None,
        )
        row_cells = []
        for strategy in strategy_names:
            cell = cells.get(strategy)
            # best_f1 == 0.0 means every strategy found nothing for this
            # book -- that's a shared failure, not a "win" for whichever
            # strategy happens to be listed, so nothing gets highlighted.
            is_best = (
                cell is not None and best_f1 is not None and best_f1 > 0.0 and cell[0].f1 == best_f1
            )
            row_cells.append(_cell_html(cell, is_best))
        if classifier is not None:
            row_cells.append(_classifier_cell_html(classifier["per_document"].get(doc_key)))
        doc_rows.append(f"<tr><td>{doc_key}</td>{''.join(row_cells)}</tr>")

    ranked_strategies = sorted(aggregates, key=lambda s: aggregates[s].f1, reverse=True)
    agg_rows = []
    for strategy in ranked_strategies:
        m = aggregates[strategy]
        t = aggregate_times.get(strategy, 0.0)
        citation_cells = ""
        if citation_aggregates is not None:
            c = citation_aggregates.get(strategy)
            citation_cells = (
                f"<td>{c.start_accuracy:.2f}</td><td>{c.end_accuracy:.2f}</td>" if c else "<td>N/A</td><td>N/A</td>"
            )
        classifier_na_cells = "<td>N/A</td><td>N/A</td>" if classifier is not None else ""
        agg_rows.append(
            f"<tr><td>{strategy}</td><td>{m.precision:.2f}</td><td>{m.recall:.2f}</td>"
            f"<td>{m.f1:.2f}</td><td>{m.true_positives}/{m.found_count} found, "
            f"{m.true_positives}/{m.expected_count} expected</td><td>{t:.2f}s</td>"
            f"{citation_cells}{classifier_na_cells}</tr>"
        )
    if classifier is not None:
        citation_na_cells = "<td>N/A</td><td>N/A</td>" if citation_aggregates is not None else ""
        agg_rows.append(
            f"<tr><td>{classifier['label']}</td><td>N/A</td><td>N/A</td><td>N/A</td>"
            f"<td>N/A</td><td>N/A</td>{citation_na_cells}"
            f"<td>{classifier['full_recall_fraction']:.0%}</td>"
            f"<td>{classifier['avg_candidate_fraction']:.0%}</td></tr>"
        )

    doc_header = "".join(f"<th>{s}</th>" for s in strategy_names)
    if classifier is not None:
        doc_header += f"<th>{classifier['label']}</th>"
    citation_header = "<th>Start accuracy</th><th>End accuracy</th>" if citation_aggregates is not None else ""
    classifier_header = "<th>Full recall</th><th>Avg candidates</th>" if classifier is not None else ""
    classifier_note_html = f"<p><em>{classifier['note']}</em></p>" if classifier is not None else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }}</style>
</head><body>
<h1>{title}</h1>
{description_html}
<h2>Per document</h2>
<table>
<tr><th>Book</th>{doc_header}</tr>
{"".join(doc_rows)}
</table>
{classifier_note_html}<h2>Per strategy (aggregate, ordered by F1)</h2>
<table>
<tr><th>Strategy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Found / Expected</th><th>Total time</th>{citation_header}{classifier_header}</tr>
{"".join(agg_rows)}
</table>
</body></html>
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report_html.py -v`
Expected: PASS (all tests, including every pre-existing one)

- [ ] **Step 5: Commit**

```bash
git add evaluation/report_html.py tests/test_report_html.py
git commit -m "feat: render_strategy_tables gains an optional classifier column/row"
```

---

### Task 6: `--save-results` flag on the layout/TOC classifier pilot

**Files:**
- Modify: `evaluation/scripts/evaluate_layout_toc_classifier.py`
- Test: `tests/test_evaluate_layout_toc_classifier.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluate_layout_toc_classifier.py`, a new test class after `class TestAugmentedRowFoldRules` (before the `if __name__ == "__main__":` line):

```python
class TestWriteClassifierResults(unittest.TestCase):
    def test_splits_results_by_corpus_and_computes_aggregates_per_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "open-access").mkdir()
            (tmp_path / "copyrighted-scans").mkdir()
            books = [
                {"key": "oa-book", "corpus": "open-access"},
                {"key": "scan-book", "corpus": "copyrighted-scans"},
            ]
            summary = {
                "per_book": [
                    {"book_key": "oa-book", "toc_recall": 1.0, "chapter_first_recall": 0.9,
                     "candidate_fraction": 0.1, "full_recall": True},
                    {"book_key": "scan-book", "toc_recall": 0.0, "chapter_first_recall": 0.5,
                     "candidate_fraction": 0.2, "full_recall": False},
                ],
            }
            with patch("evaluation.scripts.evaluate_layout_toc_classifier._CORPUS_DIR", tmp_path):
                _write_classifier_results(summary, books)

            oa_data = json.loads((tmp_path / "open-access" / "classifier-results.json").read_text(encoding="utf-8"))
            self.assertEqual(oa_data["full_recall_fraction"], 1.0)
            self.assertEqual(oa_data["avg_candidate_fraction"], 0.1)
            self.assertEqual(oa_data["per_book"]["oa-book"]["toc_recall"], 1.0)
            self.assertNotIn("scan-book", oa_data["per_book"])
            self.assertRegex(oa_data["generated_at"], r"^\d{4}-\d{2}-\d{2}T")

            scan_data = json.loads(
                (tmp_path / "copyrighted-scans" / "classifier-results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(scan_data["full_recall_fraction"], 0.0)
            self.assertEqual(scan_data["per_book"]["scan-book"]["candidate_fraction"], 0.2)

    def test_preserves_null_recall_for_a_vacuous_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "open-access").mkdir()
            books = [{"key": "book-a", "corpus": "open-access"}]
            summary = {
                "per_book": [
                    {"book_key": "book-a", "toc_recall": None, "chapter_first_recall": 1.0,
                     "candidate_fraction": 0.05, "full_recall": True},
                ],
            }
            with patch("evaluation.scripts.evaluate_layout_toc_classifier._CORPUS_DIR", tmp_path):
                _write_classifier_results(summary, books)

            data = json.loads((tmp_path / "open-access" / "classifier-results.json").read_text(encoding="utf-8"))
            self.assertIsNone(data["per_book"]["book-a"]["toc_recall"])
```

Add `_write_classifier_results` to the existing import at the top of the file:

```python
from evaluation.scripts.evaluate_layout_toc_classifier import (
    build_feature_table,
    evaluate_leave_one_book_out,
    load_book_corpus,
    main,
    select_threshold,
    _write_classifier_results,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -k WriteClassifierResults -v`
Expected: FAIL (`ImportError: cannot import name '_write_classifier_results'`)

- [ ] **Step 3: Implement**

In `evaluation/scripts/evaluate_layout_toc_classifier.py`, add the datetime import (change the top-of-file import block):

```python
import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
```

Add this function right after `evaluate_leave_one_book_out` (before `def main() -> int:`):

```python
def _write_classifier_results(summary: dict, books: list[dict]) -> None:
    """Writes evaluate_leave_one_book_out's per-book summary to
    evaluation/corpus/<corpus>/classifier-results.json, split by each
    book's own corpus (one LOBO run can span multiple corpora at once via
    --corpora) -- generate_report.py reads this to fold the classifier's
    results into the published report. See design spec
    docs/superpowers/specs/2026-08-14-report-generator-enhancements-design.md.

    full_recall_fraction/avg_candidate_fraction are recomputed per corpus
    here rather than reusing summary's own (whole-run) values -- a report
    page is generated per corpus, so its aggregate numbers must reflect
    only that corpus's books, not a blend with whatever other corpus was
    also in scope for this invocation."""
    books_by_key = {book["key"]: book for book in books}
    by_corpus: dict[str, list[dict]] = {}
    for result in summary["per_book"]:
        corpus = books_by_key[result["book_key"]]["corpus"]
        by_corpus.setdefault(corpus, []).append(result)

    generated_at = datetime.now(timezone.utc).isoformat()
    for corpus, results in by_corpus.items():
        n = len(results)
        payload = {
            "generated_at": generated_at,
            "full_recall_fraction": sum(1 for r in results if r["full_recall"]) / n,
            "avg_candidate_fraction": sum(r["candidate_fraction"] for r in results) / n,
            "per_book": {
                r["book_key"]: {
                    "toc_recall": r.get("toc_recall"),
                    "chapter_first_recall": r.get("chapter_first_recall"),
                    "candidate_fraction": r["candidate_fraction"],
                }
                for r in results
            },
        }
        out_path = _CORPUS_DIR / corpus / "classifier-results.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_path} ({n} books)")
```

In `main`'s argument parser, add the new flag (right after the existing `--scan-noise-augment` argument block):

```python
    parser.add_argument(
        "--save-results",
        action="store_true",
        help=(
            "Write per-book results to evaluation/corpus/<corpus>/classifier-results.json "
            "(split by each book's own corpus), for generate_report.py to fold into the "
            "published report. Default: off (stdout-only, current behavior)."
        ),
    )
```

In `main`, after `summary = evaluate_leave_one_book_out(...)` and before the `print(f"Books evaluated: ...")` line, add:

```python
    if args.save_results:
        _write_classifier_results(summary, books)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate_layout_toc_classifier.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/evaluate_layout_toc_classifier.py tests/test_evaluate_layout_toc_classifier.py
git commit -m "feat: add --save-results to persist classifier results for the report"
```

---

### Task 7: Wire classifier results into `generate_report.py`

**Files:**
- Modify: `evaluation/generate_report.py` (imports, `generate_corpus`)
- Test: `tests/test_generate_report.py`

- [ ] **Step 1: Write the failing tests**

Add to `class TestGenerateCorpus` in `tests/test_generate_report.py` (after the test added in Task 4):

```python
    def test_main_report_includes_classifier_column_when_results_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            (root / "test-corpus" / "classifier-results.json").write_text(json.dumps({
                "generated_at": "2026-08-14T09:00:00+00:00",
                "full_recall_fraction": 0.75,
                "avg_candidate_fraction": 0.1,
                "per_book": {
                    "book-a": {"toc_recall": 1.0, "chapter_first_recall": 1.0, "candidate_fraction": 0.05},
                },
            }), encoding="utf-8")
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Layout/TOC classifier", main_html)
            self.assertIn("as of 2026-08-14", main_html)
            self.assertIn("not directly comparable", main_html)

    def test_main_report_omits_classifier_column_when_no_results_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "corpus"
            chapters = [{"pdf_start_index": 0, "pdf_end_index": 3}]
            _write_corpus_fixture(root, "test-corpus", chapters, ["Introduction\nBody text.", "more", "more", "more"])
            out_dir = tmp_path / "public" / "test-corpus"

            with patch("evaluation.harness.CORPUS_ROOT", root), \
                 patch("evaluation.generate_report.public_outline_candidates_for", return_value=None):
                generate_corpus("test-corpus", out_dir)

            main_html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("Layout/TOC classifier", main_html)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_report.py -k classifier -v`
Expected: FAIL (`AssertionError: 'Layout/TOC classifier' not found in ...`)

- [ ] **Step 3: Implement**

In `evaluation/generate_report.py`, change the `evaluation.harness` import block from:

```python
from evaluation.harness import (
    available_public_books,
    list_corpora,
    llm_cache_dir,
    public_outline_candidates_for,
    public_pages_for,
)
```

to:

```python
from evaluation.harness import (
    available_public_books,
    corpus_dir,
    list_corpora,
    llm_cache_dir,
    public_outline_candidates_for,
    public_pages_for,
)
```

Add this helper right after `_load_llm_cache`:

```python
def _load_classifier_results(corpus: str) -> dict | None:
    """evaluation/corpus/<corpus>/classifier-results.json, written by
    evaluate_layout_toc_classifier.py --save-results, or None if that
    script has never been run (with --save-results) against this corpus."""
    path = corpus_dir(corpus) / "classifier-results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
```

In `generate_corpus`, change the `description` assignment and the `render_strategy_tables` call. Currently:

```python
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
```

to:

```python
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

    classifier_data = _load_classifier_results(corpus)
    classifier_param = None
    if classifier_data is not None:
        classifier_param = {
            "label": f"Layout/TOC classifier (LOBO, as of {classifier_data['generated_at'][:10]})",
            "note": (
                "The layout/TOC classifier row/column above measures per-page "
                "table-of-contents/chapter-opening-page classification recall via "
                "leave-one-book-out cross-validation (evaluate_layout_toc_classifier.py) -- "
                "a different methodology than the chapter-boundary precision/recall/F1 the "
                "other rows measure, so it is not directly comparable to them."
            ),
            "per_document": classifier_data["per_book"],
            "full_recall_fraction": classifier_data["full_recall_fraction"],
            "avg_candidate_fraction": classifier_data["avg_candidate_fraction"],
        }

    html = render_strategy_tables(
        title=f"chapter-segmentation: {corpus} corpus results",
        description_html=description,
        strategy_names=strategy_names,
        per_document=per_document,
        aggregates=aggregates,
        aggregate_times=aggregate_times,
        citation_aggregates=citation_aggregates,
        classifier=classifier_param,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_report.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/generate_report.py tests/test_generate_report.py
git commit -m "feat: fold layout/TOC classifier results into the main report"
```

---

### Task 8: Update the scripts `--help` reference doc

**Files:**
- Modify: `evaluation/scripts/README.md:180-214` (the `evaluate_layout_toc_classifier.py` section)

- [ ] **Step 1: Regenerate the `--help` block**

Run:

```bash
uv run python evaluation/scripts/evaluate_layout_toc_classifier.py --help
```

- [ ] **Step 2: Update the doc**

Replace the fenced code block under `## \`evaluate_layout_toc_classifier.py\`` in `evaluation/scripts/README.md` (currently lines 185-214, ending right before `## \`fetch_crossref_gt_corpus.py\``) with the exact stdout from Step 1 (it will be the same as today's block plus a new `--save-results` entry, e.g. immediately after the `--scan-noise-augment` entry:

```
  --save-results        Write per-book results to
                        evaluation/corpus/<corpus>/classifier-results.json
                        (split by each book's own corpus), for
                        generate_report.py to fold into the published report.
                        Default: off (stdout-only, current behavior).
```

Use the real command output rather than retyping it by hand -- argparse's line-wrapping depends on the exact help text and terminal width, so copy the actual printed text verbatim into the fenced block).

- [ ] **Step 3: Commit**

```bash
git add evaluation/scripts/README.md
git commit -m "docs: regenerate --help reference for evaluate_layout_toc_classifier.py"
```

---

### Task 9: Full test suite and a real-corpus smoke check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/ -q`
Expected: PASS, no failures, no new errors relative to the pre-existing baseline (the repo's real LLM cache files include many entries with no per-model `generated_at` yet -- Tasks 3/4's fallback path must not crash on those).

- [ ] **Step 2: Smoke-test the report generator against the real repo data**

Run:

```bash
uv run python evaluation/generate_report.py --out /tmp/report-smoke-test
```

Expected: exits 0, writes `/tmp/report-smoke-test/open-access/index.html` and `/tmp/report-smoke-test/copyrighted-scans/index.html` without raising. Then check the footer and LLM header rendered as expected:

```bash
grep -o "Generated on [0-9-]* from commit" /tmp/report-smoke-test/open-access/index.html
grep -o "LLM ([^)]*)" /tmp/report-smoke-test/open-access/index.html
```

Expected: first command prints one match with today's date; second prints the merged LLM column label (with or without an "as of" date, depending on whether any real cache entry has been refreshed since Task 2 landed -- both are correct outcomes, not a failure, since no real cache file will have the new per-model field until `refresh_llm_cache.py` next runs).

- [ ] **Step 3: Clean up the smoke-test output**

```bash
rm -rf /tmp/report-smoke-test
```

No commit for this task -- verification only.
