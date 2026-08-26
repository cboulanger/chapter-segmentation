# Finetune-nuextract migration & reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the NuExtract-2.0-4B fine-tuning pilot from `chapter-segmentation` (deleting its code/tests/specs, keeping only a condensed history + a pointer to `finetune-nuextract`), and reorganize the standalone `finetune-nuextract` repo into a `cli/data/docs/src/training/tests` layout matching `tei-annotation-model`.

**Architecture:** Two independent repos, two mostly-independent parts. **Part 1** restructures `finetune-nuextract` in place: `git mv` the existing top-level scripts into `cli/`/`training/`, extract two small pieces of shared pure logic into `src/finetune_nuextract/` and `training/scoring.py`, add a `pyproject.toml` (hatchling + uv, mirroring `tei-annotation-model`), and add unit tests for the newly-isolated pure logic. **Part 2** deletes the NuExtract-2.0-4B pilot's code/tests/design-docs from `chapter-segmentation` (it is too entangled with `chapter_segmentation`'s internals — `_llm_json`, `evidence.fusion`, `segmentation._parse_toc_page_number`, `evaluation.harness` — to relocate as working code), writes a condensed findings doc into `finetune-nuextract/docs/history/` (Part 1's new directory), and updates `chapter-segmentation`'s own docs (`evaluation/experiments/README.md`, `evaluation/README.md`, `evaluation/RESULTS.md`, `evaluation/EXPERIMENTS.md`, `pyproject.toml`, `.gitignore`) to match. Part 2's history-doc step depends only on Part 1 having created `finetune-nuextract/docs/history/` (a single `mkdir`), not on the rest of Part 1 — the two parts can otherwise run in either order.

**Tech Stack:** Python 3.12, `uv`/hatchling packaging, `pytest`, `git mv` for history-preserving moves, no new runtime dependencies beyond what's already used.

**Precedent already in this repo:** commit `1e14d1e` ("chore: delete the retired layout-based TOC classifier pilot's source, tests, and design docs") did the exact same kind of retirement for a different pilot (moved to `toc-page-classifier`) — deleted the pilot's specs/plans, scripts, tests, and its own `evaluation/experiments/*.md` write-up, replacing it with a bullet + link in `evaluation/experiments/README.md` pointing at the target repo's `docs/history/*.md`. Part 2 below follows that same pattern.

---

## Part 1 — Reorganize `finetune-nuextract`

All paths in this part are relative to `/Users/cboulanger/Code/finetune-nuextract` unless stated otherwise.

### Task 1: Scaffold the new directory layout and `git mv` existing files

**Files:**
- Create dirs: `cli/`, `src/finetune_nuextract/`, `training/`, `docs/`, `docs/history/`, `tests/`
- Move: `common.py` → `training/common.py`; `finetune.py` → `training/finetune.py`; `merge.py` → `training/merge.py`; `evaluate.py` → `training/evaluate.py`; `nuextract3.def` → `training/nuextract3.def`; `run_nuextract3.slurm` → `training/run_nuextract3.slurm`; `run_baseline_eval.slurm` → `training/run_baseline_eval.slurm`; `prepare_data.py` → `cli/prepare_data.py`; `dataset_stats.py` → `cli/dataset_stats.py`; `upload_model.py` → `cli/upload_model.py`; `upload_space.py` → `cli/upload_space.py`; `model_card.md` → `docs/model_card.md`

- [ ] **Step 1: Create the new directories**

```bash
mkdir -p cli src/finetune_nuextract training docs/history tests
```

- [ ] **Step 2: Move files with `git mv` (preserves history)**

```bash
git mv common.py training/common.py
git mv finetune.py training/finetune.py
git mv merge.py training/merge.py
git mv evaluate.py training/evaluate.py
git mv nuextract3.def training/nuextract3.def
git mv run_nuextract3.slurm training/run_nuextract3.slurm
git mv run_baseline_eval.slurm training/run_baseline_eval.slurm
git mv prepare_data.py cli/prepare_data.py
git mv dataset_stats.py cli/dataset_stats.py
git mv upload_model.py cli/upload_model.py
git mv upload_space.py cli/upload_space.py
git mv model_card.md docs/model_card.md
```

- [ ] **Step 3: Verify the move**

Run: `git status`
Expected: all twelve files show as renamed (`R`), nothing shows as deleted-without-a-corresponding-add.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move scripts into cli/ and training/, docs into docs/"
```

### Task 2: Extract `render_pages_to_pngs` into `src/finetune_nuextract/`

This function is used by both `cli/prepare_data.py` (data prep) and `space/app.py` (the demo, via a bundled copy) — moving it into the installable package matches `tei-annotation-model`'s pattern of putting logic reused across `cli/` in `src/`.

**Files:**
- Create: `src/finetune_nuextract/__init__.py`
- Create: `src/finetune_nuextract/rendering.py`
- Modify: `cli/prepare_data.py`

- [ ] **Step 1: Create the package `__init__.py`**

```bash
touch src/finetune_nuextract/__init__.py
```

- [ ] **Step 2: Write `src/finetune_nuextract/rendering.py`**

```python
"""Renders a PDF's pages to PNG images, scaled to fit within a bounding
box (pdftoppm's -scale-to) -- shared by cli/prepare_data.py (fine-tuning
data prep) and space/app.py (the Gradio demo), so both use the exact
same image-rendering path a fine-tuned checkpoint was trained on."""

import subprocess
import tempfile
from pathlib import Path


def render_pages_to_pngs(pdf_path: Path, max_dimension: int, pdftoppm_bin: str = "pdftoppm") -> list[bytes]:
    """Renders each page scaled to fit within a max_dimension x
    max_dimension box (pdftoppm's -scale-to), NOT a fixed DPI. Found the
    hard way (first two smoke-test HPC runs) that a fixed DPI does not
    bound prompt-token cost: this corpus has a handful of oversized
    scans (one book's pages are 740x1148pt, a double-page spread, vs a
    typical ~595x842pt single page) whose *physical* page size, not page
    count, blew a 2-page book's prompt past 7500 tokens at 150 DPI --
    Qwen3VLProcessor's own image-processor area clamp (65536-16777216 px^2,
    i.e. up to 4096x4096) is far too generous to catch this for a
    plain-text TOC scan, which needs nowhere near that resolution to stay
    legible. -scale-to bounds every page's pixel count up front,
    regardless of the source PDF's physical page size, unlike -r."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        result = subprocess.run(
            [pdftoppm_bin, "-scale-to", str(max_dimension), "-png", str(pdf_path), str(prefix)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed on {pdf_path}: {result.stderr}")
        matches = sorted(Path(tmp).glob("page-*.png"))
        if not matches:
            raise RuntimeError(f"pdftoppm produced no output for {pdf_path}")
        return [p.read_bytes() for p in matches]
```

- [ ] **Step 3: Edit `cli/prepare_data.py`'s run example in its module docstring**

Old:
```
Run locally (needs pdftoppm on PATH, pypdf installed):

    python3 prepare_data.py \\
        --corpus-dir ~/Code/dnb-toc-ground-truth/data/corpus/pilot \\
        --output-dir ./data \\
        [--limit 20] [--eval-count 50] [--seed 42] [--dpi 150]

Then rsync <output-dir> to the HPC checkout alongside finetune.py.
"""
```
New:
```
Run locally (needs pdftoppm on PATH, pypdf installed):

    uv run python3 cli/prepare_data.py \\
        --corpus-dir ~/Code/dnb-toc-ground-truth/data/corpus/pilot \\
        --output-dir ./data \\
        [--limit 20] [--eval-count 50] [--seed 42] [--dpi 150]

Then rsync <output-dir> to the HPC checkout alongside training/finetune.py.
"""
```

- [ ] **Step 4: Edit `cli/prepare_data.py`'s imports and remove the now-duplicated function**

Old:
```python
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL_REPO  # noqa: E402  (unused here, kept for parity/reference)


def render_pages_to_pngs(pdf_path: Path, max_dimension: int, pdftoppm_bin: str = "pdftoppm") -> list[bytes]:
    """Renders each page scaled to fit within a max_dimension x
    max_dimension box (pdftoppm's -scale-to), NOT a fixed DPI. Found the
    hard way (first two smoke-test HPC runs) that a fixed DPI does not
    bound prompt-token cost: this corpus has a handful of oversized
    scans (one book's pages are 740x1148pt, a double-page spread, vs a
    typical ~595x842pt single page) whose *physical* page size, not page
    count, blew a 2-page book's prompt past 7500 tokens at 150 DPI --
    Qwen3VLProcessor's own image-processor area clamp (65536-16777216 px^2,
    i.e. up to 4096x4096) is far too generous to catch this for a
    plain-text TOC scan, which needs nowhere near that resolution to stay
    legible. -scale-to bounds every page's pixel count up front,
    regardless of the source PDF's physical page size, unlike -r."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        result = subprocess.run(
            [pdftoppm_bin, "-scale-to", str(max_dimension), "-png", str(pdf_path), str(prefix)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed on {pdf_path}: {result.stderr}")
        matches = sorted(Path(tmp).glob("page-*.png"))
        if not matches:
            raise RuntimeError(f"pdftoppm produced no output for {pdf_path}")
        return [p.read_bytes() for p in matches]
```
New:
```python
import argparse
import json
import random
import sys
from pathlib import Path

from pypdf import PdfReader

from finetune_nuextract.rendering import render_pages_to_pngs
```

- [ ] **Step 5: Verify no leftover references**

Run: `grep -n "render_pages_to_pngs\|subprocess\|tempfile" cli/prepare_data.py`
Expected: `render_pages_to_pngs` appears only in the new `import` line and in its two call sites further down in the file (unchanged); no `subprocess`/`tempfile` references remain.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: extract render_pages_to_pngs into src/finetune_nuextract/rendering.py"
```

### Task 3: Extract pure scoring logic out of `training/evaluate.py`

`evaluate.py` currently defines `parse_entries`/`_normalize_title`/`_match_key`/`score` inline, but the file also imports `torch`/`peft`/`transformers` at module scope — so today these pure functions can't be imported or unit tested without those (GPU-oriented, not installed locally) dependencies present. Splitting them into `training/scoring.py` mirrors `tei-annotation-model/training/scoring.py` exactly and makes them testable from a plain `uv run pytest`.

**Files:**
- Create: `training/scoring.py`
- Modify: `training/evaluate.py`
- Test: `tests/test_training_scoring.py` (Task 5)

- [ ] **Step 1: Write `training/scoring.py`**

```python
"""Pure scoring logic for evaluate.py: parses a raw NuExtract3 completion
and scores it against ground truth. No torch/transformers/peft import, so
this is unit-testable without a GPU or the fine-tuned model itself."""

import json
import re


def parse_entries(text: str) -> list[dict]:
    """Extracts the JSON array from a raw {"entries": [...]} completion --
    same bracket-substring approach as dnb-toc-ground-truth's
    parse_json_array (toc_entry.py), which discards the outer object
    wrapper for free."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found: {text!r}")
    return json.loads(text[start : end + 1])


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _match_key(entry: dict) -> tuple[str, str | None]:
    return _normalize_title(str(entry.get("title", ""))), entry.get("printed_page_number")


def score(predicted: list[dict], expected: list[dict]) -> tuple[int, int, int]:
    """Returns (true_positives, n_predicted, n_expected) via a
    multiset-safe greedy match on _match_key."""
    remaining = list(expected)
    tp = 0
    for p in predicted:
        key = _match_key(p)
        for i, e in enumerate(remaining):
            if _match_key(e) == key:
                tp += 1
                remaining.pop(i)
                break
    return tp, len(predicted), len(expected)
```

- [ ] **Step 2: Edit `training/evaluate.py`'s imports, dropping the four moved functions**

Old:
```python
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL_REPO, _INSTRUCTIONS, _TEMPLATE, build_messages  # noqa: E402

_MAX_NEW_TOKENS = 4096


def _load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_entries(text: str) -> list[dict]:
    """Extracts the JSON array from a raw {"entries": [...]} completion --
    same bracket-substring approach as dnb-toc-ground-truth's
    parse_json_array (toc_entry.py), which discards the outer object
    wrapper for free."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found: {text!r}")
    return json.loads(text[start : end + 1])


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _match_key(entry: dict) -> tuple[str, str | None]:
    return _normalize_title(str(entry.get("title", ""))), entry.get("printed_page_number")


def score(predicted: list[dict], expected: list[dict]) -> tuple[int, int, int]:
    """Returns (true_positives, n_predicted, n_expected) via a
    multiset-safe greedy match on _match_key."""
    remaining = list(expected)
    tp = 0
    for p in predicted:
        key = _match_key(p)
        for i, e in enumerate(remaining):
            if _match_key(e) == key:
                tp += 1
                remaining.pop(i)
                break
    return tp, len(predicted), len(expected)
```
New:
```python
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_MODEL_REPO, _INSTRUCTIONS, _TEMPLATE, build_messages  # noqa: E402
from scoring import parse_entries, score  # noqa: E402

_MAX_NEW_TOKENS = 4096


def _load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 3: Verify the rest of `training/evaluate.py` is untouched**

Run: `grep -n "parse_entries\|score(" training/evaluate.py`
Expected: both names still appear at their original call sites further down (`predicted = parse_entries(raw)`, `tp, n_pred, n_exp = score(predicted, expected)`), now resolved via the new import.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: extract pure scoring logic from evaluate.py into training/scoring.py"
```

### Task 4: Add `pyproject.toml` and set up `uv`

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "finetune-nuextract"
version = "0.1.0"
description = "LoRA fine-tuning pipeline for NuExtract3 table-of-contents extraction"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pypdf>=5.0.0",
    "huggingface-hub>=0.28.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/finetune_nuextract"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[dependency-groups]
dev = [
    "pytest>=9.1.1",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`, installs `pypdf`/`huggingface-hub`/`pytest`, and installs `finetune_nuextract` itself in editable mode (so `cli/prepare_data.py`'s `from finetune_nuextract.rendering import render_pages_to_pngs` resolves under `uv run`).

- [ ] **Step 3: Verify the package import resolves**

Run: `uv run python3 -c "from finetune_nuextract.rendering import render_pages_to_pngs; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pyproject.toml, package src/finetune_nuextract via uv/hatchling"
```

(`.venv/` should already be covered by a standard Python `.gitignore` pattern — if `git status` shows it as untracked-and-not-ignored, add `.venv/` to `.gitignore` before committing.)

### Task 5: Add unit tests for the newly-isolated pure logic

**Files:**
- Test: `tests/test_training_common.py`
- Test: `tests/test_training_scoring.py`

- [ ] **Step 1: Write the failing tests — `tests/test_training_common.py`**

```python
from training.common import build_messages


def test_build_messages_user_turn_only():
    messages = build_messages(target=None)
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "Extract every table-of-contents entry from this page."}]}
    ]


def test_build_messages_includes_assistant_turn_when_target_given():
    target = {"entries": [{"title": "Vorwort", "authors": [], "printed_page_number": "vii", "skip": True}]}
    messages = build_messages(target=target)
    assert len(messages) == 2
    assert messages[1] == {
        "role": "assistant",
        "content": '{"entries": [{"title": "Vorwort", "authors": [], "printed_page_number": "vii", "skip": true}]}',
    }
```

- [ ] **Step 2: Write the failing tests — `tests/test_training_scoring.py`**

```python
import pytest

from training.scoring import parse_entries, score


def test_parse_entries_extracts_json_array_from_wrapped_text():
    raw = '{"entries": [{"title": "Vorwort", "authors": [], "printed_page_number": "vii", "skip": true}]}'
    assert parse_entries(raw) == [
        {"title": "Vorwort", "authors": [], "printed_page_number": "vii", "skip": True}
    ]


def test_parse_entries_raises_on_missing_array():
    with pytest.raises(ValueError):
        parse_entries("not json at all")


def test_score_counts_true_positives_on_normalized_title_and_page_match():
    predicted = [
        {"title": "  Introduction  ", "printed_page_number": "1"},
        {"title": "Bibliography", "printed_page_number": "200"},
    ]
    expected = [
        {"title": "introduction", "printed_page_number": "1"},
        {"title": "Index", "printed_page_number": "210"},
    ]
    assert score(predicted, expected) == (1, 2, 2)


def test_score_does_not_double_count_duplicate_predictions():
    predicted = [{"title": "Chapter One", "printed_page_number": "5"}] * 2
    expected = [{"title": "Chapter One", "printed_page_number": "5"}]
    assert score(predicted, expected) == (1, 2, 1)
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/ -v`
Expected: 6 tests pass (2 in `test_training_common.py`, 4 in `test_training_scoring.py`). No `torch`/`transformers`/`peft` import is triggered — these tests run without the heavy training dependencies installed.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add unit tests for training/common.py and training/scoring.py"
```

### Task 6: Update the bundling logic in `cli/upload_space.py` and `space/`

`common.py` moved to `training/common.py`; the function `space/app.py` imports from `prepare_data.py` moved to `src/finetune_nuextract/rendering.py`. `cli/upload_space.py` bundles both into `space/` at deploy time (they're gitignored there — not maintained as separate copies).

**Files:**
- Modify: `cli/upload_space.py`
- Modify: `space/app.py`
- Modify: `space/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Edit `cli/upload_space.py`'s module docstring**

Old:
```
Bundles common.py and prepare_data.py from this repo's root into
space/ before uploading, so the demo shares the exact same
prompt/template/rendering as fine-tuning and evaluation -- there is no
separate copy of either kept in space/ to fall out of sync; see
space/README.md.
```
New:
```
Bundles training/common.py and src/finetune_nuextract/rendering.py into
space/ before uploading, so the demo shares the exact same
prompt/template/rendering as fine-tuning and evaluation -- there is no
separate copy of either kept in space/ to fall out of sync; see
space/README.md.
```

- [ ] **Step 2: Edit `cli/upload_space.py`'s `_ROOT`/`_BUNDLED` constants and the bundling loops**

Old:
```python
_ROOT = Path(__file__).resolve().parent
_SPACE_DIR = _ROOT / "space"
_BUNDLED = ["common.py", "prepare_data.py"]
```
New:
```python
_ROOT = Path(__file__).resolve().parents[1]
_SPACE_DIR = _ROOT / "space"
_BUNDLED = [("training/common.py", "common.py"), ("src/finetune_nuextract/rendering.py", "rendering.py")]
```

Old:
```python
    for name in _BUNDLED:
        shutil.copy(_ROOT / name, _SPACE_DIR / name)
    try:
        print(f"Creating/reusing Space {args.space_id} (private={args.private})")
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
        print(f"Uploading {_SPACE_DIR} -> {args.space_id}")
        api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=str(_SPACE_DIR))
        print("Requesting ZeroGPU (zero-a10g) hardware")
        api.request_space_hardware(args.space_id, hardware=SpaceHardware.ZERO_A10G)
        if args.model_repo_id:
            print(f"Setting MODEL_REPO_ID={args.model_repo_id}")
            api.add_space_variable(args.space_id, "MODEL_REPO_ID", args.model_repo_id)
    finally:
        for name in _BUNDLED:
            (_SPACE_DIR / name).unlink(missing_ok=True)
```
New:
```python
    for src_rel, dest_name in _BUNDLED:
        shutil.copy(_ROOT / src_rel, _SPACE_DIR / dest_name)
    try:
        print(f"Creating/reusing Space {args.space_id} (private={args.private})")
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
        print(f"Uploading {_SPACE_DIR} -> {args.space_id}")
        api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=str(_SPACE_DIR))
        print("Requesting ZeroGPU (zero-a10g) hardware")
        api.request_space_hardware(args.space_id, hardware=SpaceHardware.ZERO_A10G)
        if args.model_repo_id:
            print(f"Setting MODEL_REPO_ID={args.model_repo_id}")
            api.add_space_variable(args.space_id, "MODEL_REPO_ID", args.model_repo_id)
    finally:
        for _src_rel, dest_name in _BUNDLED:
            (_SPACE_DIR / dest_name).unlink(missing_ok=True)
```

- [ ] **Step 3: Edit `space/app.py`'s module docstring and import**

Old:
```
Reuses prepare_data.py's render_pages_to_pngs (identical -scale-to
image rendering) and common.py's prompt/template/instructions, so the
demo asks the model exactly the same question fine-tuning and
evaluate.py did -- no separate, driftable copy of either.
```
New:
```
Reuses rendering.py's render_pages_to_pngs (identical -scale-to
image rendering) and common.py's prompt/template/instructions, so the
demo asks the model exactly the same question fine-tuning and
evaluate.py did -- no separate, driftable copy of either.
```

Old:
```python
from common import _INSTRUCTIONS, _TEMPLATE, build_messages
from prepare_data import render_pages_to_pngs
```
New:
```python
from common import _INSTRUCTIONS, _TEMPLATE, build_messages
from rendering import render_pages_to_pngs
```

- [ ] **Step 4: Edit `space/README.md`**

Old:
```
`common.py` and `prepare_data.py` here are bundled in at deploy time by
`upload_space.py` from the training repo's root — not maintained as
separate copies in this directory.
```
New:
```
`common.py` and `rendering.py` here are bundled in at deploy time by
`cli/upload_space.py` from `training/common.py` and
`src/finetune_nuextract/rendering.py` — not maintained as separate
copies in this directory.
```

- [ ] **Step 5: Edit `.gitignore`**

Old:
```
space/common.py
space/prepare_data.py
```
New:
```
space/common.py
space/rendering.py
```

- [ ] **Step 6: Verify `cli/upload_space.py` parses and the bundling copy works (dry run, no upload)**

Run:
```bash
uv run python3 -c "
import sys
sys.argv = ['upload_space.py', '--space-id', 'x/y', '--env-file', '/dev/null']
import importlib.util
spec = importlib.util.spec_from_file_location('upload_space', 'cli/upload_space.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for src_rel, dest_name in mod._BUNDLED:
    assert (mod._ROOT / src_rel).is_file(), f'missing {src_rel}'
print('bundle sources ok:', mod._BUNDLED)
"
```
Expected: prints `bundle sources ok: [('training/common.py', 'common.py'), ('src/finetune_nuextract/rendering.py', 'rendering.py')]` with no `AssertionError`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: bundle training/common.py + src/finetune_nuextract/rendering.py into space/"
```

### Task 7: Fix the `docs/model_card.md` and `cli/upload_model.py` path references

**Files:**
- Modify: `docs/model_card.md`
- Modify: `cli/upload_model.py`

- [ ] **Step 1: Edit `docs/model_card.md`**

Old:
```
Requires the exact same prompt template and instructions used during
fine-tuning — copy `common.py` and `prepare_data.py`'s
`render_pages_to_pngs` from the training repo rather than
reconstructing the prompt by hand, since even small prompt drift will
hurt output quality. Minimal usage:
```
New:
```
Requires the exact same prompt template and instructions used during
fine-tuning — copy `training/common.py` and
`src/finetune_nuextract/rendering.py`'s `render_pages_to_pngs` from the
training repo rather than reconstructing the prompt by hand, since even
small prompt drift will hurt output quality. Minimal usage:
```

- [ ] **Step 2: Edit `cli/upload_model.py`'s default `--model-card` path**

Old:
```python
    parser.add_argument("--model-card", type=Path, default=Path(__file__).resolve().parent / "model_card.md")
```
New:
```python
    parser.add_argument("--model-card", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "model_card.md")
```

- [ ] **Step 3: Verify**

Run: `uv run python3 -c "
import sys; sys.argv = ['upload_model.py', '--repo-id', 'x/y', '--model-dir', '/dev/null', '--env-file', '/dev/null']
import importlib.util
spec = importlib.util.spec_from_file_location('upload_model', 'cli/upload_model.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
"`

Expected: a normal `argparse` run isn't triggered by this import-only check (the script's logic lives in `_main()`, only invoked under `if __name__ == \"__main__\"`), so this just confirms the file has no syntax/import errors. Additionally run: `python3 -c "from pathlib import Path; p = Path('cli/upload_model.py').resolve().parents[1] / 'docs' / 'model_card.md'; assert p.is_file(), p; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: point upload_model.py and model_card.md at docs/model_card.md"
```

### Task 8: Update the SLURM scripts for the new `training/` location and root-level `data/`

`data/` (rendered PNGs + train/eval JSONL) stays a top-level directory, sibling to `training/`, matching `tei-annotation-model`'s `data/` — but the SLURM scripts now live one level down, in `training/`, so their submit-directory-relative paths need adjusting.

**Files:**
- Modify: `training/run_nuextract3.slurm`
- Modify: `training/run_baseline_eval.slurm`

- [ ] **Step 1: Edit `training/run_nuextract3.slurm`'s leading comment**

Old:
```
# Assumes this script's own directory contains
# common.py/finetune.py/merge.py/evaluate.py/nuextract3.sif (already
# built) and data/{train,eval}.jsonl + data/images/ (already rsynced from
# a local prepare_data.py run). Model weights must already be pre-fetched
# into $HF_HOME from a login node -- compute nodes have no internet, same
# constraint as the NuExtract-2.0 pilot (see chapter-segmentation's
# evaluation/hpc/README.md section 4).
```
New:
```
# Assumes this script's own directory (training/) contains
# common.py/scoring.py/finetune.py/merge.py/evaluate.py/nuextract3.sif
# (already built), a repo-root-level data/{train,eval}.jsonl +
# data/images/ sibling (already rsynced from a local cli/prepare_data.py
# run). Model weights must already be pre-fetched into $HF_HOME from a
# login node -- compute nodes have no internet.
```

- [ ] **Step 2: Edit `training/run_nuextract3.slurm`'s path variables**

Old:
```bash
DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PTMP="/ptmp/${USER}"
SIF="${DIR}/nuextract3.sif"
BIND="${DIR}:${DIR},${PTMP}"
```
New:
```bash
DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PTMP="/ptmp/${USER}"
SIF="${DIR}/nuextract3.sif"
REPO_ROOT="$(cd "${DIR}/.." && pwd)"
BIND="${REPO_ROOT}:${REPO_ROOT},${PTMP}"
```

Old:
```bash
TRAIN_JSONL="${DIR}/data/train.jsonl"
EVAL_JSONL="${DIR}/data/eval.jsonl"
if [[ ! -f "$TRAIN_JSONL" || ! -f "$EVAL_JSONL" ]]; then
    echo "Missing $TRAIN_JSONL and/or $EVAL_JSONL." >&2
    echo "Run prepare_data.py locally and rsync data/ over first (see README.md)." >&2
    exit 1
fi
```
New:
```bash
TRAIN_JSONL="${REPO_ROOT}/data/train.jsonl"
EVAL_JSONL="${REPO_ROOT}/data/eval.jsonl"
if [[ ! -f "$TRAIN_JSONL" || ! -f "$EVAL_JSONL" ]]; then
    echo "Missing $TRAIN_JSONL and/or $EVAL_JSONL." >&2
    echo "Run cli/prepare_data.py locally and rsync data/ over first (see README.md)." >&2
    exit 1
fi
```

- [ ] **Step 3: Edit `training/run_baseline_eval.slurm`'s leading comment**

Old:
```
# Baseline comparison run: scores the UNMODIFIED numind/NuExtract3 base
# model (no --adapter-dir/--model-dir) on the same held-out eval.jsonl
# used to score the fine-tuned checkpoint in run_nuextract3.slurm (job
# 29510176, f1=0.7544) -- lets us see how much the LoRA fine-tune actually
# moved the needle vs. zero-shot on this corpus.
```
(No change needed here — job IDs/scores are historical fact, unaffected by the move. Skip.)

- [ ] **Step 4: Edit `training/run_baseline_eval.slurm`'s path variables**

Old:
```bash
DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PTMP="/ptmp/${USER}"
SIF="${DIR}/nuextract3.sif"
BIND="${DIR}:${DIR},${PTMP}"
```
New:
```bash
DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PTMP="/ptmp/${USER}"
SIF="${DIR}/nuextract3.sif"
REPO_ROOT="$(cd "${DIR}/.." && pwd)"
BIND="${REPO_ROOT}:${REPO_ROOT},${PTMP}"
```

Old:
```bash
EVAL_JSONL="${DIR}/data/eval.jsonl"
if [[ ! -f "$EVAL_JSONL" ]]; then
    echo "Missing $EVAL_JSONL." >&2
    exit 1
fi
```
New:
```bash
EVAL_JSONL="${REPO_ROOT}/data/eval.jsonl"
if [[ ! -f "$EVAL_JSONL" ]]; then
    echo "Missing $EVAL_JSONL." >&2
    exit 1
fi
```

- [ ] **Step 5: Verify with `bash -n` (syntax check only, no cluster needed)**

Run: `bash -n training/run_nuextract3.slurm && bash -n training/run_baseline_eval.slurm && echo OK`
Expected: `OK`, no syntax errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "fix: point SLURM scripts at repo-root data/ from their new training/ location"
```

### Task 9: Write `training/README.md`

**Files:**
- Create: `training/README.md`

- [ ] **Step 1: Write the file**

```markdown
# NuExtract3 LoRA fine-tuning pipeline

Runbook for training and evaluating the LoRA adapter described in the
[top-level README](../README.md). Read that file first for the task,
prompt contract, and repo layout; this file only covers the mechanics of
running a pilot end to end.

## Layout

- `common.py` — shared template/instructions/prompt-building, kept in sync with production
- `finetune.py` — LoRA fine-tuning (run on a GPU node)
- `scoring.py` — pure parse/match/score logic for `evaluate.py` (no GPU/model needed; unit tested in `../tests/`)
- `merge.py` — merges the trained adapter into the base weights → a standalone HF checkpoint
- `evaluate.py` — scores a checkpoint (base, adapter, or merged) on the held-out eval split
- `nuextract3.def` — Apptainer container definition (PyTorch 2.8.0/CUDA 12.6 + transformers/peft/accelerate/torchvision)
- `run_nuextract3.slurm` — SLURM batch script chaining finetune → merge → evaluate
- `run_baseline_eval.slurm` — scores the unmodified base model, for comparison

Training data (repo-root `data/`, rendered PNGs + JSONL) and run artifacts
(`adapter/`, `merged/`, container `.sif`, job logs) are **not** committed —
see the repo's `.gitignore`. Everything needed to regenerate them is here.

## 1. Prepare training data (locally)

Requires a checkout of `dnb-toc-ground-truth` with its pilot corpus at
`data/corpus/pilot/{pdf,ground-truth}/`, plus `pdftoppm` (poppler) on
`PATH` and `pypdf` installed. Run from the repo root so the output lands
at the repo-root `data/` directory (a sibling of `training/`, which the
SLURM scripts below expect):

```bash
uv run python3 cli/prepare_data.py \
    --corpus-dir ~/Code/dnb-toc-ground-truth/data/corpus/pilot \
    --output-dir data \
    [--limit 20] [--eval-count 50] [--seed 42]
```

This renders each book's TOC page(s) to PNGs (scaled to fit within
`--max-dimension` pixels — bounds prompt-token cost regardless of the
source PDF's physical page size) and writes `data/{train,eval}.jsonl` plus
`data/images/<key>/page-N.png`. Books with a TOC longer than `--max-pages`
(default 3) are skipped. Use `--limit` for a fast smoke test before
committing to a full-corpus run.

## 2. Ship code + data to the HPC cluster

From the repo root:

```bash
rsync -avz . <cluster>:projects/finetune-nuextract/ --exclude='.git' --exclude='.venv'
```

(The whole repo, not just `training/` — `data/` rsyncs too, since it's
excluded from git, not from the cluster copy.)

## 3. Build the container (from a login node)

```bash
cd training
module load apptainer/1.4.3
apptainer build --fakeroot nuextract3.sif nuextract3.def
```

## 4. Pre-fetch base model weights (from a login node — needs internet)

Compute nodes on most HPC clusters (this pipeline was built against MPCDF
Raven) have no internet access, so `numind/NuExtract3`'s weights must be
cached into `$HF_HOME` *before* submitting the batch job:

```bash
export HF_HOME=/ptmp/$USER/huggingface   # match run_nuextract3.slurm's HF_HOME
apptainer exec nuextract3.sif python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('numind/NuExtract3')"
```

The SLURM script runs with `HF_HUB_OFFLINE=1`, so a missed file here fails
the job immediately rather than silently trying to hit the network.

## 5. Run the pipeline

From `training/`:

```bash
sbatch run_nuextract3.slurm
```

This chains three stages, each `srun apptainer exec --nv ...`:

1. **`finetune.py`** — trains a LoRA adapter (rank 16, target modules =
   all attention + MLP projections) at `batch_size=1` with gradient
   accumulation (effective batch size = `--gradient-accumulation-steps`,
   default 8), 3 epochs by default. Loss is masked to the completion only
   — built manually rather than via `return_assistant_tokens_mask=True`,
   which returns an all-zero mask for this model's `Qwen3VLProcessor` once
   images are involved (verified empirically; text-only inputs mask
   correctly). Adapter → `$PTMP/nuextract3-pilot/adapter`.
2. **`merge.py`** — merges the adapter into the base weights, saving a
   standalone HF checkpoint (weights + processor/tokenizer, ready to load
   with `AutoModelForImageTextToText.from_pretrained` or serve via vLLM
   with no PEFT dependency) → `$PTMP/nuextract3-pilot/merged`.
3. **`evaluate.py`** — scores the merged checkpoint on the held-out split
   via direct `transformers` generation on GPU, computing a minimal
   entry-level precision/recall/F1 (an entry matches when its normalized
   title *and* `printed_page_number` both match; `skip` correctness is
   not part of the match key). Per-book predictions dumped to
   `$PTMP/nuextract3-pilot/predictions-finetuned/<key>.json` when
   `--dump-dir` is set; summary → `$PTMP/nuextract3-pilot/eval-finetuned.txt`.

Adjust `--time`, `--mem`, `--cpus-per-task`, and the GPU `--gres` in
`run_nuextract3.slurm` to fit your corpus size and cluster. Each stage can
also be run standalone (e.g. to re-evaluate an existing checkpoint,
or evaluate the unmodified base model by omitting `--adapter-dir`/`--model-dir`).

To score the **unmodified base model** on the same eval split, for
comparison against the fine-tuned checkpoint:

```bash
sbatch run_baseline_eval.slurm
```

## Notes

- `nuextract3.sif` is ~7GB and rebuilds from `nuextract3.def` in a few
  minutes — not worth committing or transferring once you can rebuild it.
- `evaluate.py`'s scoring (`scoring.py`) is intentionally simple (exact
  normalized-title + page-number match) — a go/no-go pilot signal, not a
  replacement for dnb-toc-ground-truth's own fuzzy-matching evaluation
  harness if a checkpoint gets promoted past the pilot stage.
```

- [ ] **Step 2: Commit**

```bash
git add training/README.md
git commit -m "docs: add training/README.md runbook"
```

### Task 10: Rewrite the root `README.md` and update `AGENTS.md` path references

**Files:**
- Modify: `README.md` (full rewrite)
- Modify: `AGENTS.md` (path references)

- [ ] **Step 1: Rewrite `README.md`**

```markdown
# Finetuning NuExtract3 for TOC parsing

LoRA fine-tuning pipeline for [numind/NuExtract3](https://huggingface.co/numind/NuExtract3)
(a Qwen3.5-based vision-language model) on the table-of-contents extraction
task from [dnb-toc-ground-truth](https://github.com/cboulanger/dnb-toc-ground-truth/) —
turning a scanned TOC page image into structured JSON:

```json
{
  "entries": [
    {
      "title": "Vorwort",
      "authors": [],
      "printed_page_number": "vii",
      "skip": true
    },
    {
      "title": "1. Introduction",
      "authors": ["Jane Doe"],
      "printed_page_number": "1",
      "skip": false
    }
  ]
}
```

`skip` marks a line that isn't an actual chapter — front matter (e.g.
a preface, list of contributors), back matter (e.g. bibliography,
index), or a part/section divider. It's still emitted as its own
entry with its own title/page, never omitted; only its own true/false
correctness is excluded from `training/scoring.py`'s precision/recall/F1
match key (see "Results" below) — the entry itself is still scored on
title+page like any other.

The prompt/template/instructions the model is trained on are copied
verbatim from dnb-toc-ground-truth's production zero-shot NuExtract3
integration (`training/common.py`), so the fine-tune target matches
exactly what production inference asks the model to do — no train/serve
skew.

Mirrors chapter-segmentation's earlier NuExtract-2.0 LoRA pilot in spirit
(LoRA via `peft`, loss masked to the completion only), but built from
scratch: NuExtract3 needs real image inputs (`Qwen3VLProcessor`), and
NuMind hasn't published a fine-tuning cookbook for it (only for
NuExtract-2.0). A condensed history of that earlier pilot — retired from
chapter-segmentation once this repo's NuExtract3 pipeline superseded it —
is preserved at
[`docs/history/chapter-segmentation-nuextract2-finetuning-pilot.md`](docs/history/chapter-segmentation-nuextract2-finetuning-pilot.md).

If you're an AI agent running a pilot or documenting its results, see
[AGENTS.md](AGENTS.md) for the exact sequence and bookkeeping
conventions this repo expects.

## Layout

- `cli/` — scripts you run locally (not on the GPU node): `prepare_data.py`
  (build training data), `dataset_stats.py` (train/eval split stats),
  `upload_model.py`/`upload_space.py` (publish to the Hugging Face Hub)
- `src/finetune_nuextract/` — the installable package: `rendering.py`
  (`render_pages_to_pngs`, shared by `cli/prepare_data.py` and the
  `space/` demo)
- `training/` — the LoRA fine-tuning pipeline itself (`common.py`,
  `finetune.py`, `scoring.py`, `merge.py`, `evaluate.py`, the Apptainer/
  SLURM files); see [`training/README.md`](training/README.md) for the
  full runbook
- `docs/` — `model_card.md` (pushed as the Hugging Face model repo's
  README.md by `cli/upload_model.py`) and `history/` (condensed writeups
  of superseded/retired predecessor pilots from other repos)
- `space/` — a Gradio demo (upload a TOC PDF, see the extraction);
  `cli/upload_space.py` deploys it to a Hugging Face Space
- `tests/` — unit tests for the pure logic in `training/common.py` and
  `training/scoring.py`

Training data (`data/`, rendered PNGs + JSONL) and run artifacts (`adapter/`,
`merged/`, container `.sif`, job logs) are **not** committed — see
`.gitignore`. Everything needed to regenerate them is here.

## Prepare, ship, and run the fine-tuning pipeline

See [`training/README.md`](training/README.md) for the full runbook:
preparing training data locally, shipping code + data to an HPC cluster,
building the Apptainer container, pre-fetching model weights, and
submitting the SLURM job that chains fine-tuning → merging → evaluation.

## Publish to the Hugging Face Hub

Two independent, optional publishing steps — both need an `HF_TOKEN` in
a `.env` file (kept off the cluster's home dir permissions and out of
this repo; see `.gitignore`), and take a `--private`/`--public` flag
(default private).

**The model**, from a merged checkpoint (`training/merge.py`'s output):

```bash
python3 cli/upload_model.py --repo-id <your-hf-namespace>/nuextract3-toc \
    --model-dir $PTMP/nuextract3-pilot/merged \
    --env-file ~/projects/finetune/.env
```

This pushes the checkpoint plus `docs/model_card.md` (update its results
table from `RESULTS.md`'s latest entry first — the script uploads it
verbatim, it doesn't regenerate anything).

**The demo Space** (`space/` — a Gradio app: upload a TOC PDF, see the
extraction), bundling `training/common.py` and
`src/finetune_nuextract/rendering.py` in at deploy time so the demo
shares the exact prompt/template/rendering used for training and
evaluation:

```bash
python3 cli/upload_space.py --space-id <your-hf-namespace>/nuextract3-toc-demo \
    --model-repo-id <your-hf-namespace>/nuextract3-toc \
    --env-file ~/projects/finetune/.env
```

`--model-repo-id` sets the Space's `MODEL_REPO_ID` variable to that
repo id, so the demo serves the fine-tuned checkpoint rather than the
unmodified base model (its default if unset) — upload the model first,
or pass this again later to point the Space at a different checkpoint.

## Results

Latest full-corpus pilot (624 train / 44 eval books from
dnb-toc-ground-truth's pilot corpus, 15,372 / 1,050 TOC entries,
seed=42, `--max-pages 3`) — full details, dataset stats, and per-book
notes in [RESULTS.md](RESULTS.md):

| | Baseline (zero-shot) | Fine-tuned (LoRA, 3 epochs) | Δ |
| --- | --- | --- | --- |
| Precision | 0.6131 | 0.7555 | +0.142 |
| Recall | 0.7590 | 0.7533 | −0.006 |
| F1 | 0.6783 | 0.7544 | +0.076 |

Fine-tuning's gain is almost entirely a precision story: the base model
over-generates (1300 predicted vs. 1050 expected entries); fine-tuning
brought predicted count in line with expected without giving up recall.

## Notes

- To deploy a merged checkpoint to production, `cli/upload_model.py` it
  to the Hub and point production at it — it's a drop-in replacement for
  `numind/NuExtract3` (same architecture, same template/instructions
  contract).
```

- [ ] **Step 2: Edit `AGENTS.md`'s path references**

Old:
```
1. **Prepare data locally** — `prepare_data.py` (needs `pdftoppm` +
```
New:
```
1. **Prepare data locally** — `cli/prepare_data.py` (needs `pdftoppm` +
```

Old:
```
4. **Submit the pipeline**: `sbatch run_nuextract3.slurm` (fine-tune →
   merge → evaluate) and, for comparison, `sbatch run_baseline_eval.slurm`
   (scores the unmodified base model on the same eval split).
```
New:
```
4. **Submit the pipeline** (from `training/`): `sbatch run_nuextract3.slurm`
   (fine-tune → merge → evaluate) and, for comparison,
   `sbatch run_baseline_eval.slurm` (scores the unmodified base model on
   the same eval split).
```

Old:
```
1. **Compute dataset stats**: run `dataset_stats.py` against the exact
   `data/{train,eval}.jsonl` this run used (pull the actual files back
   from wherever they were rsynced/generated — don't reconstruct them
   from memory or approximate). Pass `--corpus-dir` too, for the
   corpus-level context line.
   ```bash
   python3 dataset_stats.py --train-jsonl data/train.jsonl --eval-jsonl data/eval.jsonl \
       --corpus-dir <path-to-dnb-toc-ground-truth>/data/corpus/pilot
   ```
```
New:
```
1. **Compute dataset stats**: run `cli/dataset_stats.py` against the exact
   `data/{train,eval}.jsonl` this run used (pull the actual files back
   from wherever they were rsynced/generated — don't reconstruct them
   from memory or approximate). Pass `--corpus-dir` too, for the
   corpus-level context line.
   ```bash
   python3 cli/dataset_stats.py --train-jsonl data/train.jsonl --eval-jsonl data/eval.jsonl \
       --corpus-dir <path-to-dnb-toc-ground-truth>/data/corpus/pilot
   ```
```

Old:
```
`upload_model.py`/`upload_space.py` push a merged checkpoint and the
```
New:
```
`cli/upload_model.py`/`cli/upload_space.py` push a merged checkpoint and the
```

Old:
```
do this: update `model_card.md`'s
```
New:
```
do this: update `docs/model_card.md`'s
```

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: rewrite README.md and update AGENTS.md for the new cli/src/training layout"
```

### Task 11: Final verification of Part 1

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 2: Confirm no stray references to the old top-level paths remain**

Run: `grep -rn "python3 prepare_data.py\|python3 finetune.py\|python3 merge.py\|python3 evaluate.py\|python3 dataset_stats.py\|python3 upload_model.py\|python3 upload_space.py" . --include='*.md' --include='*.py' --include='*.slurm' 2>/dev/null | grep -v '.venv/'`
Expected: no output (every remaining reference should already carry a `cli/`/`training/` prefix from the edits above).

- [ ] **Step 3: Confirm the top-level directory listing matches the target layout**

Run: `find . -maxdepth 1 -not -path '.' -not -name '.git' -not -name '.venv' | sort`
Expected: `./.env`, `./.gitignore`, `./AGENTS.md`, `./CLAUDE.local.md`, `./README.md`, `./RESULTS.md`, `./cli`, `./docs`, `./pyproject.toml`, `./space`, `./src`, `./tests`, `./training`, `./uv.lock`.

---

## Part 2 — Retire the NuExtract-2.0-4B pilot from `chapter-segmentation`

All paths in this part are relative to `/Users/cboulanger/Code/chapter-segmentation` unless stated otherwise. **This part deletes code, it does not move it** — `evaluation/nuextract_baseline.py` and `evaluation/nuextract2_common.py` import directly from `chapter_segmentation._llm_json`, `chapter_segmentation.evidence.fusion`, `chapter_segmentation.segmentation`, and `evaluation.harness`/`evaluation.metrics` (confirmed via `grep` — see investigation notes above), so there is no working standalone copy to relocate; only the empirical results are worth preserving, which Task 13 does as a condensed prose write-up in `finetune-nuextract`.

**Depends on:** Task 13 needs `finetune-nuextract/docs/history/` to exist (created in Part 1, Task 1) — if Part 1 hasn't run yet, `mkdir -p /Users/cboulanger/Code/finetune-nuextract/docs/history` first.

### Task 12: Delete the pilot's code, tests, and design docs

**Files to delete:**
- `evaluation/nuextract2_common.py`
- `evaluation/nuextract_baseline.py`
- `evaluation/scripts/finetune_nuextract.py`
- `evaluation/scripts/evaluate_nuextract_finetune.py`
- `evaluation/scripts/merge_nuextract_lora.py`
- `evaluation/scripts/prepare_nuextract_finetune_data.py`
- `evaluation/scripts/evaluate_nuextract_baseline.py`
- `evaluation/hpc/` (entire directory: `README.md`, `nuextract.def`, `nuextract-viper.def`, `run_pilot.slurm`, `run_pilot-viper.slurm`)
- `tests/test_nuextract2_common.py`
- `tests/test_nuextract_baseline.py`
- `evaluation/experiments/nuextract-finetuning.md`
- `docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md`
- `docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`
- `docs/superpowers/plans/2026-08-09-nuextract-baseline-evaluation-plan.md`
- `docs/superpowers/plans/2026-08-10-nuextract2-finetuning-pilot.md`

- [ ] **Step 1: Confirm nothing else in the codebase imports these modules (safety check before deleting)**

Run:
```bash
grep -rln "nuextract2_common\|nuextract_baseline\|finetune_nuextract\|evaluate_nuextract_finetune\|merge_nuextract_lora\|prepare_nuextract_finetune_data\|evaluate_nuextract_baseline" \
  --include='*.py' . | grep -v -E '^\./(evaluation/nuextract2_common\.py|evaluation/nuextract_baseline\.py|evaluation/scripts/(finetune_nuextract|evaluate_nuextract_finetune|merge_nuextract_lora|prepare_nuextract_finetune_data|evaluate_nuextract_baseline)\.py|tests/test_nuextract2_common\.py|tests/test_nuextract_baseline\.py)$'
```
Expected: no output — confirms these files only reference each other, nothing outside the set to delete imports them. (Already verified during planning; re-run here as a pre-flight check in case the tree has changed.)

- [ ] **Step 2: Delete the files**

```bash
git rm evaluation/nuextract2_common.py
git rm evaluation/nuextract_baseline.py
git rm evaluation/scripts/finetune_nuextract.py
git rm evaluation/scripts/evaluate_nuextract_finetune.py
git rm evaluation/scripts/merge_nuextract_lora.py
git rm evaluation/scripts/prepare_nuextract_finetune_data.py
git rm evaluation/scripts/evaluate_nuextract_baseline.py
git rm -r evaluation/hpc/
git rm tests/test_nuextract2_common.py
git rm tests/test_nuextract_baseline.py
git rm evaluation/experiments/nuextract-finetuning.md
git rm docs/superpowers/specs/2026-08-09-nuextract-baseline-evaluation-design.md
git rm docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md
git rm docs/superpowers/plans/2026-08-09-nuextract-baseline-evaluation-plan.md
git rm docs/superpowers/plans/2026-08-10-nuextract2-finetuning-pilot.md
```

- [ ] **Step 3: Verify the test suite still collects cleanly (no dangling imports of deleted test files)**

Run: `uv run pytest --collect-only -q 2>&1 | tail -20`
Expected: no `ImportError`/`ModuleNotFoundError` mentioning `nuextract2_common`, `nuextract_baseline`, or any of the deleted script names.

(Don't commit yet — Task 15 makes one combined commit for all of Part 2's deletions and doc edits, matching how commit `1e14d1e` bundled the equivalent layout-classifier retirement into a single commit.)

### Task 13: Write the condensed history doc in `finetune-nuextract`

**Files:**
- Create: `/Users/cboulanger/Code/finetune-nuextract/docs/history/chapter-segmentation-nuextract2-finetuning-pilot.md`

This condenses `evaluation/experiments/nuextract-finetuning.md` (being deleted in Task 12) down to the findings worth keeping, in the same format as `toc-page-classifier`'s own `docs/history/chapter-segmentation-layout-classifier-pilot.md` (the precedent for this exact kind of cross-repo history preservation).

- [ ] **Step 1: Ensure the target directory exists**

```bash
mkdir -p /Users/cboulanger/Code/finetune-nuextract/docs/history
```

- [ ] **Step 2: Write the file**

```markdown
# Prior art: the NuExtract-2.0-4B fine-tuning pilot in `chapter-segmentation`

**Historical document, imported verbatim-in-spirit (condensed) from
`chapter-segmentation`'s `evaluation/experiments/nuextract-finetuning.md`
on 2026-08-26, when that pilot was retired from the `chapter-segmentation`
repo.** This repo (`finetune-nuextract`) is not a continuation of that
pilot -- it fine-tunes a different, newer model
([NuExtract3](https://huggingface.co/numind/NuExtract3), Qwen3.5-based)
targeting a different deployment path (vLLM on GPU, not llama.cpp on
CPU), against a different, larger ground-truth corpus (776 DNB books via
`dnb-toc-ground-truth`, not chapter-segmentation's 89-book evaluation
corpus). It's preserved here because the failure modes, model-selection
reasoning, and methodology lessons below are directly relevant to this
repo's own fine-tuning work, and the original document no longer exists
in `chapter-segmentation` once this file was written.

## What it was

A go/no-go pilot: could `NuExtract-2.0-4B` (a small, locally-runnable,
no-GPU-target extraction model) extract a book's table of contents
directly, as a candidate alternative/supplement to the cloud-LLM strategy
row in chapter-segmentation's own evaluation results? Two phases: a
zero-shot baseline across the full 50-book evaluation corpus, then a LoRA
fine-tuning pilot (`peft`, loss masked to the completion only) on a
78-train/11-eval split of the same corpus.

## Model selection: two earlier/larger models tried and rejected first

- **NuExtract-1.5-tiny: a real capability ceiling, not a serving-path
  bug.** Zero-shot full-corpus run scored 0.00/0.00/0.00 -- root-caused
  (direct Ollama API probing plus a `transformers`-checkpoint retest,
  ruling out both a prompt bug and GGUF-conversion fidelity) to the model
  having no instruction channel to ignore surrounding non-TOC noise: given
  a real scan window, it either stopped generating immediately or echoed
  the input back verbatim. Even hand-curated, TOC-only input only reached
  f1=0.09.
- **NuExtract3 (4B, Qwen3.5-based): worked well zero-shot (f1=0.60 on the
  same 50-book corpus via `mlx-vlm`), but dropped for deployment reasons,
  not accuracy** -- the actual target was a no-GPU, 16GB RAM, 4-vCPU Linux
  box, where MLX can't run at all and Ollama didn't yet support
  Qwen3.5-architecture GGUFs. `NuExtract-2.0-4B` benchmarked 1.42x faster
  with equal-or-better accuracy on that exact hardware profile (5-book
  CPU/`llama.cpp` sample), on top of already having mature Ollama support
  via its older Qwen2.5-VL architecture. This is the model
  `finetune-nuextract` itself does **not** use -- by the time this repo
  started, `NuExtract3` had matured enough (and `llama.cpp`/Ollama gained
  the needed architecture support) to become the better target instead,
  which is why this repo fine-tunes `NuExtract3`, not `NuExtract-2.0-4B`.

## Zero-shot baseline: f1=0.44 (the number the fine-tuning pilot needed to beat)

Full 50-book, two-corpus run via `llama.cpp` (Metal-offloaded,
`n_ctx=40960`, `max_tokens=6000`): precision=0.45, recall=0.35, **f1=0.39**
at the originally-tested `max_tokens=1500`, rising to **f1=0.44** once
retested at `max_tokens=6000` (raising the output-token budget recovered
2 of 10 originally-truncated books completely; the other 8 were already,
or became once budget was no longer the bottleneck, genuine
wrong-or-repetitive-output failures, not truncation). Raising the cap
further to 12000 tokens did not rescue the remaining 4
`[HIT_MAX_TOKENS]` books -- confirmed as a genuine decoding
repetition/malformed-generation failure, not a legitimate long-TOC book
that just needed more room.

**Dominant zero-shot failure mode (28% of books): titles/authors
extracted correctly, but `printed_page_number` left `null` on every
entry**, even when the number was plainly visible in the scan text next
to the title. Since the scoring match key requires an exact page-number
match, these books scored 0 recall despite mostly-correct extraction --
title-only accuracy was substantially better than the f1 headline
suggested. This was the specific failure mode the fine-tuning pilot below
targeted.

**A genuine backend-dependent decoding bug, not a tokenization or prompt
bug: on `transformers`/MPS (Apple Silicon), NuExtract-2.0-4B reliably
emitted `printed_page_number: null` for every entry even when the model
clearly had the information (title/author extraction was correct on the
same page), while the identical prompt through `llama.cpp` (GGUF Q4_K_M,
CPU or Metal-offloaded) filled in every page number correctly.**
Reproduced twice on the same book; ruled out `add_special_tokens`
tokenization differences (identical token IDs, still-null output on
`transformers`). Since the actual deployment target was `llama.cpp` on a
CPU-only Linux host anyway, this was moot for production -- but it means
**any zero-shot/fine-tuning comparison for this model family must go
through the same backend the deployment target uses; `transformers` and
`llama.cpp` are not interchangeable for this model's structured-output
behavior.**

## Fine-tuning pilot result: a promising but genuinely noisy signal, not a clean go

Trained a LoRA adapter (rank 16, 4 epochs, gradient checkpointing) on 78
books' TOC-scan-window text (89-book corpus, 78 train / 11 eval,
stratified split), scored both the fine-tuned and unmodified base
checkpoint on the same 11-book held-out split via the same
`llama.cpp`-only scoring path (not directly comparable to the full-corpus
f1=0.44 baseline above -- a different, much smaller subset):

| | precision | recall | f1 |
| --- | --- | --- | --- |
| Fine-tuned | 0.83 | 0.46 | **0.59** |
| Base (same split, same code) | 0.57 | 0.48 | **0.52** |

**Aggregate f1 improved (0.52 → 0.59), but the per-book picture was more
complicated than "fine-tuning helps" -- big wins on some books, masking a
real regression on two others.** Root cause of the two collapses: a
**decoding-time degenerate-repetition loop, not a fine-tuning capability
regression.** Both books' raw output showed `finish_reason: length` -- the
model correctly extracted several early chapters completely correctly
before falling into an infinite repetition loop that burned the entire
token budget without ever closing the JSON, so the parser saw
truncated/invalid JSON and scored 0/0, even though the model demonstrably
had the right knowledge.

**Tried fixing the repetition loop with a repeat penalty; made the
aggregate worse both times tried.** Any blanket repeat penalty (tested at
two window sizes) partially fixed the two target books but broke several
previously-fine books -- because the output is a JSON *list* of chapter
dicts, legitimately repeating the same field names every ~20-40 tokens;
a repeat penalty can't distinguish that legitimate structural repetition
from the genuine 1-4-token degenerate loop. Reverted to no penalty.

**Against the pilot's own decision criterion (whether the specific
null-page-number rate dropped, not just aggregate f1): a promising but
genuinely noisy signal, not a clean go/no-go.** The raw dumps for the two
collapsed books were suggestive (every chapter extracted before the loop
struck had a correct, non-null page number, not the null-page-number
failure the pilot was meant to fix), but the check wasn't completed
before the newer `NuExtract3` model made this line of work moot -- see
"Model selection" above.

## Findings that generalize beyond this specific pilot

- **A model's decoding behavior for structured output can differ
  materially between `transformers` and `llama.cpp` even at matched
  precision/prompt** -- always benchmark and fine-tune through the same
  backend the deployment target actually uses, never assume parity.
- **A blanket repetition penalty is a poor fix for a degenerate-loop
  failure mode when the correct output legitimately contains repeated
  structural tokens** (e.g. a JSON list of same-shaped records) -- it
  tends to trade one failure mode for several new ones. A more promising
  direction, not implemented here: detect the loop and salvage the valid
  JSON prefix generated before it, at the application layer instead of
  the sampler.
- **Raising an output token budget has a real but bounded effect on a
  truncation-driven failure cluster** -- worth doing once (cheap, no
  retraining), but a book that still hits a much larger cap, or still
  produces nothing after a much longer runtime without hitting the cap,
  is very unlikely to be a token-budget problem at all.
- **Model selection for a CPU-only/no-GPU deployment target should
  benchmark on that exact target hardware/backend, not on a
  higher-powered dev machine's best-supported path** -- the model that
  won zero-shot on a Mac (`NuExtract3` via `mlx-vlm`) was provisionally
  dropped in favor of a smaller, older-architecture model purely because
  the actual deployment target (CPU-only Linux) couldn't run `mlx-vlm` at
  all and had better tooling support for the older architecture at the
  time -- a decision that later reversed once tooling caught up (see
  "Model selection" above).
```

- [ ] **Step 3: Verify the file was written correctly**

Run: `wc -l /Users/cboulanger/Code/finetune-nuextract/docs/history/chapter-segmentation-nuextract2-finetuning-pilot.md`
Expected: a non-zero line count (~140-160 lines).

- [ ] **Step 4: Commit (in the `finetune-nuextract` repo)**

```bash
cd /Users/cboulanger/Code/finetune-nuextract
git add docs/history/chapter-segmentation-nuextract2-finetuning-pilot.md
git commit -m "docs: preserve condensed history of chapter-segmentation's retired NuExtract-2.0-4B pilot"
```

### Task 14: Update `chapter-segmentation`'s own docs and dependency list

**Files:**
- Modify: `evaluation/experiments/README.md`
- Modify: `evaluation/README.md`
- Modify: `evaluation/RESULTS.md`
- Modify: `evaluation/EXPERIMENTS.md`
- Modify: `pyproject.toml`
- Modify: `evaluation/.gitignore`

- [ ] **Step 1: Replace the `nuextract-finetuning.md` bullet in `evaluation/experiments/README.md`**

Old:
```
- [`nuextract-finetuning.md`](nuextract-finetuning.md) -- whether a small,
  locally-runnable extraction model (`NuExtract-2.0-4B`) can extract a
  book's table of contents directly, as a candidate alternative to the
  LLM-strategy row in `evaluation/RESULTS.md`.
```
New:
```
- **NuExtract-2.0-4B fine-tuning pilot** -- whether a small,
  locally-runnable extraction model (`NuExtract-2.0-4B`) can extract a
  book's table of contents directly, as a candidate alternative to the
  LLM-strategy row in `evaluation/RESULTS.md`. A zero-shot baseline plus a
  LoRA fine-tuning pilot showed a promising but genuinely noisy signal
  (aggregate f1 0.52 -> 0.59 on an 11-book held-out split, propped up by
  big per-book wins alongside a new decoding-time repetition failure mode
  on two others) and never reached a clean go/no-go decision before a
  newer model, NuExtract3, superseded it as the more promising target.
  All further NuExtract fine-tuning work (including this pilot's own
  history) has moved to its own standalone repository,
  [finetune-nuextract-toc](https://github.com/cboulanger/finetune-nuextract-toc),
  which now fine-tunes NuExtract3 instead. See that repo's
  `docs/history/chapter-segmentation-nuextract2-finetuning-pilot.md` for a
  condensed history of this pilot's findings.
```

- [ ] **Step 2: Remove the two NuExtract sections from `evaluation/README.md`**

Old (the entire block from the `### NuExtract baseline spike` heading through the paragraph immediately before `### Strategy-pipeline evaluation`):
```
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

### NuExtract-2.0-4B fine-tuning pilot

A go/no-go check on whether LoRA fine-tuning closes NuExtract-2.0-4B's
dominant zero-shot failure mode (titles/authors correct,
`printed_page_number` null -- see `RESULTS.md`'s failure-mode
breakdown), using only the existing 89-book corpus, no new ground
truth. See
`docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md`
for the full rationale and decision criteria.

One-time setup -- install the fine-tuning dependency group:

```bash
uv pip install --python .venv -e ".[nuextract-finetune]"
```

1. **Prepare data** (needs the evaluation PDFs -- see "Fetching the PDFs"
   above; no ML dependencies needed for this step):

   ```bash
   uv run python evaluation/scripts/prepare_nuextract_finetune_data.py
   ```

   Writes `evaluation/finetune/data/{train,eval}.jsonl` and `split.json`
   -- all gitignored (copyrighted-scans books' scan-window text can't be
   redistributed, same reason `*.pdf` is gitignored).

2. **Train the LoRA adapter** (`transformers`+`peft`; `--device mps` on
   Apple Silicon, `--device cuda` on an NVIDIA GPU):

   ```bash
   uv run python evaluation/scripts/finetune_nuextract.py \
       --output-dir evaluation/finetune/adapter
   ```

3. **Merge the adapter into a standalone checkpoint**:

   ```bash
   uv run python evaluation/scripts/merge_nuextract_lora.py \
       --adapter-dir evaluation/finetune/adapter \
       --output-dir evaluation/finetune/merged
   ```

4. **Convert to GGUF and quantize**, via a local `llama.cpp` checkout
   (not a dependency of this repo -- clone it separately):

   ```bash
   python /path/to/llama.cpp/convert_hf_to_gguf.py \
       evaluation/finetune/merged \
       --outfile evaluation/finetune/merged.fp16.gguf
   /path/to/llama.cpp/llama-quantize \
       evaluation/finetune/merged.fp16.gguf \
       evaluation/finetune/merged.Q4_K_M.gguf \
       Q4_K_M
   ```

5. **Score the held-out split** and compare against the base model:

   ```bash
   # fine-tuned
   uv run python evaluation/scripts/evaluate_nuextract_finetune.py \
       --gguf-path evaluation/finetune/merged.Q4_K_M.gguf

   # base model, same split, same code -- the number to beat
   uv run python evaluation/scripts/evaluate_nuextract_finetune.py
   ```

Not a pytest test and not part of any CI workflow -- a manual, one-off
pilot, same operational pattern as the NuExtract baseline spike above.
Record the result (both f1 numbers, and whether the null-page-number
rate dropped on the held-out books) in `RESULTS.md`.

**Running this on MPCDF HPC instead of locally:** see
[`evaluation/hpc/README.md`](hpc/README.md) for an Apptainer/SLURM
package that runs the same five steps above on a GPU node (Raven),
including a container definition and a ready-to-submit batch script.

```
New: *(nothing — delete the whole block above, leaving `### Strategy-pipeline evaluation` immediately after the preceding section's closing paragraph, with the usual single blank line between sections)*

- [ ] **Step 3: Fix `evaluation/RESULTS.md`'s "Scope" paragraph**

Old:
```
**Scope: the main `chapter_segmentation` workflow only** (`open-access/`
and `copyrighted-scans/` corpora) — the pure heuristic, the strategy
pipeline, and the standalone heuristic/outline/LLM strategies it's built
from. Experiments toward optimizing individual pieces of that pipeline
that haven't been integrated yet (the `dnb-toc-only` ground-truth
generation pipeline, the NuExtract fine-tuning pilot) each have their own
dedicated snapshot under `evaluation/experiments/` instead of living here
— see `evaluation/experiments/dnb-toc-ground-truth.md` and
`evaluation/experiments/nuextract-finetuning.md`.
```
New:
```
**Scope: the main `chapter_segmentation` workflow only** (`open-access/`
and `copyrighted-scans/` corpora) — the pure heuristic, the strategy
pipeline, and the standalone heuristic/outline/LLM strategies it's built
from. Experiments toward optimizing individual pieces of that pipeline
that haven't been integrated yet (the `dnb-toc-only` ground-truth
generation pipeline) have their own dedicated snapshot under
`evaluation/experiments/` instead of living here — see
`evaluation/experiments/README.md` for the current list, including
experiments that have since moved to their own standalone repositories.
```

- [ ] **Step 4: Fix `evaluation/EXPERIMENTS.md`'s equivalent paragraph**

Old:
```
For the not-yet-integrated experiments this file used to also cover
(`dnb-toc-only` ground-truth generation and the NuExtract fine-tuning
pilot), see their own dedicated files under `evaluation/experiments/`
instead -- `evaluation/experiments/dnb-toc-ground-truth.md` and
`evaluation/experiments/nuextract-finetuning.md` -- each of which now
holds both its current status and its own full history in one place
(moved there 2026-08-19, see `evaluation/RESULTS.md`'s "Scope" note).
```
New:
```
For the not-yet-integrated experiments this file used to also cover
(`dnb-toc-only` ground-truth generation and, formerly, the NuExtract
fine-tuning pilot), see `evaluation/experiments/README.md` for the
current list -- each still-tracked experiment holds both its current
status and its own full history in one place (moved there 2026-08-19,
see `evaluation/RESULTS.md`'s "Scope" note); retired ones point onward to
whichever standalone repository now owns that work.
```

- [ ] **Step 5: Remove the `nuextract-finetune` optional-dependency group from `pyproject.toml`**

Old:
```toml
[project.optional-dependencies]
kreuzberg = ["httpx>=0.27.0"]
tesseract = ["pytesseract>=0.3.10", "pymupdf>=1.24.0", "pillow>=10.0.0"]
llm-eval = ["openai>=1.0.0", "httpx>=0.27.0"]
nuextract-finetune = [
    "torch>=2.4.0",
    "transformers>=4.46.0",
    "peft>=0.13.0",
    "accelerate>=0.34.0",
    "llama-cpp-python>=0.3.0",
    "huggingface-hub>=0.25.0",
]
layout-classifier = ["scikit-learn>=1.4.0"]
```
New:
```toml
[project.optional-dependencies]
kreuzberg = ["httpx>=0.27.0"]
tesseract = ["pytesseract>=0.3.10", "pymupdf>=1.24.0", "pillow>=10.0.0"]
llm-eval = ["openai>=1.0.0", "httpx>=0.27.0"]
layout-classifier = ["scikit-learn>=1.4.0"]
```

(`layout-classifier` is left as-is here — it's a pre-existing dangling
extra from the earlier, already-completed layout-classifier retirement,
unrelated to this task; out of scope to fix here.)

- [ ] **Step 6: Remove the pilot's entries from `evaluation/.gitignore`**

Old:
```
*.pdf
manifest.local.json
.ocr-cache/
finetune/
hpc/*.sif
hpc/job.out.*
hpc/job.err.*
.layout-cache/
.lobid-cache/
```
New:
```
*.pdf
manifest.local.json
.ocr-cache/
.layout-cache/
.lobid-cache/
```

- [ ] **Step 7: Verify the edited markdown renders sensibly (no doubled/missing blank lines at the deletion seam)**

Run: `grep -n "^###\|^## " evaluation/README.md | sed -n '/NuExtract/,+2p; /Strategy-pipeline/p'`
Expected: no `NuExtract` heading remains; `### Strategy-pipeline evaluation` is present and immediately follows the LLM-cache-refresh section with normal spacing (spot check with `grep -n -B3 "Strategy-pipeline evaluation" evaluation/README.md` if unsure).

- [ ] **Step 8: Sync `uv.lock` after the `pyproject.toml` edit**

Run: `uv sync`
Expected: `uv.lock` updates to drop `torch`/`transformers`/`peft`/`accelerate`/`llama-cpp-python` (unless still pulled in transitively by another extra — check the diff).

### Task 15: Commit Part 2

- [ ] **Step 1: Review the full diff**

Run: `git status && git diff --stat`
Expected: deletions matching Task 12's list, plus edits to `evaluation/experiments/README.md`, `evaluation/README.md`, `evaluation/RESULTS.md`, `evaluation/EXPERIMENTS.md`, `pyproject.toml`, `evaluation/.gitignore`, `uv.lock`.

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "chore: retire the NuExtract-2.0-4B fine-tuning pilot's source, tests, and design docs

Superseded by the standalone finetune-nuextract-toc repo
(github.com/cboulanger/finetune-nuextract-toc), which now fine-tunes
NuExtract3 instead. A condensed history of this pilot's findings is
preserved at that repo's docs/history/."
```

- [ ] **Step 3: Run the full test suite one more time**

Run: `uv run pytest -q`
Expected: passes, same pass count as before minus the two deleted test files' test counts (no new failures).

---

## Self-review notes

- **Spec coverage:** "move the finetune experiment out of chapter-segmentation into finetune-nuextract... keep the results... replace with a reference in evaluation/experiments/README.md" → Part 2 (Tasks 12-15). "reorganize finetune-nuextract to use cli/data/docs/src/training/tests" → Part 1 (Tasks 1-11). `data/` itself is never created as a tracked directory (it's gitignored, user-generated by `cli/prepare_data.py`) — matches `tei-annotation-model`'s own `data/` being real but partially gitignored content, not an empty scaffolded directory.
- **Why code isn't relocated in Part 2:** confirmed during planning that `evaluation/nuextract_baseline.py` imports `chapter_segmentation._llm_json`, `chapter_segmentation.evidence.fusion._ALIGN_SCORE_THRESHOLD`, `chapter_segmentation.segmentation._parse_toc_page_number`, and `evaluation.metrics`/`evaluation.harness`; `evaluation/scripts/prepare_nuextract_finetune_data.py` additionally imports `chapter_segmentation.segmentation._llm_scan_indices` and `evaluation.harness`. None of this is portable to a standalone repo without either vendoring core chapter-segmentation internals or dragging along the evaluation corpus/harness — the user's own instruction ("no need to preserve the code... just keep the results") anticipated exactly this outcome, and it matches the precedent set by the layout-classifier pilot's retirement (commit `1e14d1e`).
