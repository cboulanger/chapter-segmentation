# Ground-Truth Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the chapter bounds/overlap sanity check and the open-access license lookup from ad-hoc, hand-run code into shared, tested, always-on pipeline components, and add the missing `promote_pending_book.py` script that uses both to move a book from `pending/` into a real corpus.

**Architecture:** A single `chapter_bounds_errors()` function in `evaluation/harness.py` (already the shared home for corpus-loading logic) replaces two independent implementations (a CLAUDE.md copy-paste one-liner and a private duplicate inside `build_crossref_gt_ground_truth.py`), backed by an always-on `tests/test_ground_truth_integrity.py` that walks every real corpus. A new `evaluation/oa_license.py` module extracts the Crossref/Unpaywall license-lookup functions currently private to `fetch_crossref_gt_corpus.py` (which `discover_crossref_candidates.py` already reaches into across the module boundary to use), adding one new convenience function, `resolve_license()`. Both get wired into a new `evaluation/scripts/promote_pending_book.py` as a hard gate (bounds check) and an automatic step (license lookup).

**Tech Stack:** Python 3.12, pypdf, httpx, pytest/unittest (existing stack, no new dependencies).

---

## Spec coverage checklist (self-review)

- Part 1 (shared validator + always-on test) → Tasks 1, 2, 3.
- Part 2 (shared license module) → Tasks 4, 5, 6.
- Part 3 (promotion script) → Task 7.
- Documentation updates → Task 8.
- Non-goals (no `manifest.local.json` support, no copyrighted-scans license fields, no new HTTP retry logic) are respected by Task 7's design as written below.

## Task 1: `chapter_bounds_errors()` in `evaluation/harness.py`

**Files:**
- Modify: `evaluation/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

Open `tests/test_harness.py`. In the import block (lines 14-22), add `chapter_bounds_errors` alphabetically:

```python
from evaluation.harness import (
    analysis_pages_for,
    available_public_books,
    chapter_bounds_errors,
    list_corpora,
    outline_candidate_from_dict,
    outline_candidate_to_dict,
    public_outline_candidates_for,
    public_pages_for,
)
```

Immediately after the `TestListCorpora` class (after its closing `test_returns_empty_list_when_corpus_root_missing` method, before `class TestAnalysisPagesFor`), insert:

```python
class TestChapterBoundsErrors(unittest.TestCase):
    def test_no_errors_for_valid_non_overlapping_chapters(self):
        chapters = [
            {"pdf_start_index": 0, "pdf_end_index": 4},
            {"pdf_start_index": 5, "pdf_end_index": 9},
        ]
        self.assertEqual(chapter_bounds_errors(chapters, total_pages=10), [])

    def test_flags_start_after_end(self):
        chapters = [{"pdf_start_index": 5, "pdf_end_index": 2}]
        errors = chapter_bounds_errors(chapters)
        self.assertEqual(len(errors), 1)
        self.assertIn("start>end", errors[0])

    def test_flags_overlap_between_chapters(self):
        chapters = [
            {"pdf_start_index": 0, "pdf_end_index": 10},
            {"pdf_start_index": 8, "pdf_end_index": 15},
        ]
        errors = chapter_bounds_errors(chapters)
        self.assertEqual(len(errors), 1)
        self.assertIn("overlap", errors[0])

    def test_flags_end_at_or_past_total_pages_only_when_given(self):
        chapters = [{"pdf_start_index": 0, "pdf_end_index": 10}]
        self.assertEqual(
            chapter_bounds_errors(chapters, total_pages=10),
            ["end>=total_pages(10): (0, 10)"],
        )
        self.assertEqual(chapter_bounds_errors(chapters, total_pages=None), [])

    def test_reports_every_problem_not_just_the_first(self):
        chapters = [
            {"pdf_start_index": 5, "pdf_end_index": 2},   # start>end
            {"pdf_start_index": 3, "pdf_end_index": 20},  # end>=total_pages, and overlaps the next range
            {"pdf_start_index": 6, "pdf_end_index": 8},
        ]
        errors = chapter_bounds_errors(chapters, total_pages=10)
        self.assertEqual(len(errors), 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_harness.py -k ChapterBoundsErrors -v`
Expected: FAIL with `ImportError: cannot import name 'chapter_bounds_errors'`

- [ ] **Step 3: Implement `chapter_bounds_errors()`**

Open `evaluation/harness.py`. Append this function at the end of the file (after `public_outline_candidates_for`):

```python
def chapter_bounds_errors(chapters: list[dict], total_pages: Optional[int] = None) -> list[str]:
    """Structural sanity check on one book's ground-truth chapter ranges:
    every pdf_start_index <= pdf_end_index, no two chapters' ranges
    overlap, and -- only when total_pages is given -- every
    pdf_end_index < total_pages. Returns every problem found (empty list
    if none), not just the first -- needs no PDF unless total_pages is
    passed, so it can run against every corpus even before any PDF has
    been fetched locally."""
    ranges = sorted((c["pdf_start_index"], c["pdf_end_index"]) for c in chapters)
    errors = []
    for start, end in ranges:
        if start > end:
            errors.append(f"start>end: {(start, end)}")
        if total_pages is not None and end >= total_pages:
            errors.append(f"end>=total_pages({total_pages}): {(start, end)}")
    for (_, end1), (start2, _) in zip(ranges, ranges[1:]):
        if start2 <= end1:
            errors.append(f"overlap: end {end1} vs next start {start2}")
    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add evaluation/harness.py tests/test_harness.py
git commit -m "feat: add chapter_bounds_errors() shared ground-truth validator"
```

## Task 2: Always-on corpus-wide integrity test

**Files:**
- Create: `tests/test_ground_truth_integrity.py`

- [ ] **Step 1: Write the test**

Create `tests/test_ground_truth_integrity.py`:

```python
"""Structural sanity check for every corpus's ground truth --
evaluation/harness.py's chapter_bounds_errors() run against every
<isbn>.expected.json present, for every corpus. NOT marked integration:
the overlap/ordering checks need no PDF at all, so this runs in the
default `uv run pytest` even with zero evaluation PDFs downloaded; the
pdf_end_index<total_pages check only applies to books whose PDF happens
to be present locally too. See
docs/superpowers/specs/2026-08-12-ground-truth-pipeline-hardening-design.md."""

import json
import unittest

from pypdf import PdfReader

from evaluation.harness import chapter_bounds_errors, corpus_dir, list_corpora, load_manifest_books


class TestGroundTruthIntegrity(unittest.TestCase):
    def test_no_bounds_or_overlap_errors(self):
        for corpus in list_corpora():
            cdir = corpus_dir(corpus)
            for book in load_manifest_books(corpus):
                isbn = book["filename"].removesuffix(".pdf")
                expected_path = cdir / f"{isbn}.expected.json"
                if not expected_path.exists():
                    continue
                with self.subTest(book=f"{corpus}/{isbn}"):
                    chapters = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
                    pdf_path = cdir / book["filename"]
                    total_pages = len(PdfReader(str(pdf_path)).pages) if pdf_path.exists() else None
                    errors = chapter_bounds_errors(chapters, total_pages)
                    self.assertEqual(errors, [], f"{corpus}/{isbn}: {errors}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and confirm it passes against the real corpus**

Run: `uv run pytest tests/test_ground_truth_integrity.py -v`
Expected: PASS. This is a regression/integrity test over already-fixed real data (both known overlaps -- `9782821895607` in open-access/ and `9783428042241` in copyrighted-scans/ -- were fixed by hand before this plan was written), not new production code being test-driven, so a passing result on first run is the correct outcome here, not a red flag.

If it unexpectedly FAILs, do not "fix" it by loosening the check -- read the reported `corpus/isbn` and problem, open that book's real PDF at the reported page indices (same process as `evaluation/CLAUDE.md`'s Step 3), and correct the `.expected.json` boundary, the same way the two known issues were fixed. Then re-run this step.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ground_truth_integrity.py
git commit -m "test: add always-on ground-truth bounds/overlap regression test"
```

## Task 3: Switch `build_crossref_gt_ground_truth.py` to the shared validator

**Files:**
- Modify: `evaluation/scripts/build_crossref_gt_ground_truth.py`

- [ ] **Step 1: Remove the private `_sanity_check` duplicate**

In `evaluation/scripts/build_crossref_gt_ground_truth.py`, delete this function (currently lines 299-309):

```python
def _sanity_check(chapters: list[dict], total_pages: int) -> str | None:
    ranges = sorted((c["pdf_start_index"], c["pdf_end_index"]) for c in chapters)
    for start, end in ranges:
        if start > end:
            return f"start>end: {(start, end)}"
        if end >= total_pages:
            return f"end>=total_pages({total_pages}): {(start, end)}"
    for (_, end1), (start2, _) in zip(ranges, ranges[1:]):
        if start2 <= end1:
            return f"overlap: end {end1} vs next start {start2}"
    return None
```

- [ ] **Step 2: Import the shared function**

In the same file's import block (currently):

```python
from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.scripts.evaluate_layout_toc_classifier import build_feature_table, load_book_corpus
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
from evaluation.scripts.layout_features import FEATURE_NAMES, extract_page_features
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary
```

change it to:

```python
from chapter_segmentation.segmentation import _parse_toc_page_number
from evaluation.harness import chapter_bounds_errors
from evaluation.scripts.evaluate_layout_toc_classifier import build_feature_table, load_book_corpus
from evaluation.scripts.ground_truth_helper import extract_printed_number, find_toc_pages, toc_page_range
from evaluation.scripts.layout_features import FEATURE_NAMES, extract_page_features
from evaluation.scripts.layout_labels import LABEL_CHAPTER_FIRST
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary
```

- [ ] **Step 3: Update the call site**

In `process_book()`, change:

```python
    error = _sanity_check(confirmed, total_pages)
    if error:
        return isbn, f"SKIP: sanity check failed after reconciliation: {error}"
```

to:

```python
    errors = chapter_bounds_errors(confirmed, total_pages)
    if errors:
        return isbn, f"SKIP: sanity check failed after reconciliation: {'; '.join(errors)}"
```

- [ ] **Step 4: Verify nothing broke**

Run: `uv run pytest tests/test_build_crossref_gt_ground_truth.py -v`
Expected: PASS (this file never imported or tested `_sanity_check` directly, so this only confirms the module still imports and its other tested functions are unaffected).

Run: `uv run python -c "import evaluation.scripts.build_crossref_gt_ground_truth"`
Expected: no output, exit code 0 (confirms no leftover reference to the deleted `_sanity_check` name and no import errors).

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/build_crossref_gt_ground_truth.py
git commit -m "refactor: use shared chapter_bounds_errors() in build_crossref_gt_ground_truth.py"
```

## Task 4: `evaluation/oa_license.py` shared license-lookup module

**Files:**
- Create: `evaluation/oa_license.py`
- Test: `tests/test_oa_license.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_oa_license.py`:

```python
"""Unit tests for evaluation/oa_license.py's pure logic (item_license_url,
book_license_url) against literal Crossref-shaped dicts -- no network.
unpaywall_license_url/resolve_license are exercised indirectly by the
scripts that call them against the real network (fetch_crossref_gt_corpus.py,
discover_crossref_candidates.py, promote_pending_book.py); this file only
covers what needs no mocking to test meaningfully."""

import unittest

from evaluation.oa_license import book_license_url, item_license_url


class TestItemLicenseUrl(unittest.TestCase):
    def test_prefers_version_of_record_with_no_delay(self):
        item = {
            "license": [
                {"URL": "https://embargoed", "content-version": "am", "delay-in-days": 365},
                {"URL": "https://vor", "content-version": "vor", "delay-in-days": 0},
            ]
        }
        self.assertEqual(item_license_url(item), "https://vor")

    def test_falls_back_to_first_entry_when_no_vor_matches(self):
        item = {"license": [{"URL": "https://only-one", "content-version": "am", "delay-in-days": 365}]}
        self.assertEqual(item_license_url(item), "https://only-one")

    def test_none_when_no_license_key(self):
        self.assertIsNone(item_license_url({}))


class TestBookLicenseUrl(unittest.TestCase):
    def test_majority_vote_across_chapters(self):
        items = [
            {"license": [{"URL": "https://a", "content-version": "vor", "delay-in-days": 0}]},
            {"license": [{"URL": "https://a", "content-version": "vor", "delay-in-days": 0}]},
            {"license": [{"URL": "https://b", "content-version": "vor", "delay-in-days": 0}]},
        ]
        self.assertEqual(book_license_url(items), "https://a")

    def test_none_when_no_chapter_has_a_license(self):
        self.assertIsNone(book_license_url([{}, {}]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_oa_license.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.oa_license'`

- [ ] **Step 3: Create `evaluation/oa_license.py`**

```python
"""Shared Crossref/Unpaywall open-access license lookup.

Single home for the license-resolution logic evaluation/scripts/
fetch_crossref_gt_corpus.py, discover_crossref_candidates.py, and
promote_pending_book.py all need. Previously duplicated as private
functions inside fetch_crossref_gt_corpus.py, with discover_crossref_candidates.py
reaching across the module boundary to import its private names directly
-- see docs/superpowers/specs/2026-08-12-ground-truth-pipeline-hardening-design.md.
Lives under evaluation/ (not evaluation/scripts/) for the same reason
harness.py does: evaluation/scripts/*.py must not depend on each other's
internals any more than they depend on the test tree.
"""

import time
from collections import Counter
from typing import Optional

import httpx

from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)

DEFAULT_CONTACT_EMAIL = "boulanger@lhlt.mpg.de"
UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"

# Unpaywall reports a short SPDX-ish code (e.g. "cc-by-nc"), not a URL --
# mapped to the CC 4.0 URL, since every book this mapping has been
# checked against so far is a 2020s publication and 4.0 is the only
# version any of its publishers use for new titles (found empirically:
# every Crossref-registered license in this corpus that DOES carry an
# explicit version is 4.0).
_UNPAYWALL_LICENSE_URLS = {
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc-by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "cc-by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "pd": "https://creativecommons.org/publicdomain/mark/1.0/",
}


def crossref_book_chapter_items(isbn: str, client: httpx.Client, contact_email: Optional[str]) -> list[dict]:
    """GET .../works?filter=isbn:{isbn}, returning raw type=="book-chapter"
    items. Any network/HTTP/JSON failure is printed and treated as an
    empty result -- never raises, so one bad book never aborts a batch."""
    params: dict[str, str | int] = {
        "filter": f"isbn:{isbn}",
        "select": "DOI,title,subtitle,author,page,type,container-title,published,ISBN,license",
        "rows": 100,
    }
    if contact_email:
        params["mailto"] = contact_email

    response = None
    for _attempt in range(_MAX_RETRIES):
        try:
            response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
        except httpx.HTTPError as exc:
            print(f"  [warn] network error fetching Crossref metadata for {isbn}: {exc}")
            return []
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
        time.sleep(delay)
    else:
        print(f"  [warn] exhausted retries (429) fetching Crossref metadata for {isbn}")
        return []

    try:
        response.raise_for_status()
        items = response.json()["message"]["items"]
    except Exception as exc:
        print(f"  [warn] bad Crossref response for {isbn}: {exc}")
        return []

    return [item for item in items if item.get("type") == "book-chapter"]


def item_license_url(item: dict) -> Optional[str]:
    """The registered OA license URL for one Crossref item, preferring the
    version-of-record entry (content-version=="vor", delay-in-days==0) --
    the license that actually applies to the publicly available PDF, not
    an embargoed accepted-manuscript variant Crossref may register
    alongside it."""
    licenses = item.get("license") or []
    for entry in licenses:
        if entry.get("content-version") == "vor" and entry.get("delay-in-days", 0) == 0:
            return entry.get("URL")
    return licenses[0].get("URL") if licenses else None


def book_license_url(raw_items: list[dict]) -> Optional[str]:
    """The book's OA license URL, by majority vote across its chapters'
    individually-registered licenses -- in practice unanimous, since
    Crossref licenses are registered once per book and inherited by every
    chapter, but a vote is cheap insurance against one mis-registered
    chapter. None if no chapter has a registered license at all."""
    urls = [url for item in raw_items if (url := item_license_url(item)) is not None]
    if not urls:
        return None
    return Counter(urls).most_common(1)[0][0]


def unpaywall_license_url(doi: Optional[str], client: httpx.Client, contact_email: Optional[str]) -> Optional[str]:
    """Fallback license lookup via Unpaywall, tried only when Crossref has
    no license registered for a book. Crossref's license field is
    self-reported by the publisher at metadata-deposit time and often left
    blank; Unpaywall aggregates OA status from institutional repositories
    and publisher landing pages instead, so it is NOT the same data.
    Returns None if there's no DOI, no Unpaywall OA record, no license, or
    the request fails -- never raises, so one bad lookup never aborts a
    batch."""
    if not doi:
        return None
    email = contact_email or DEFAULT_CONTACT_EMAIL
    try:
        response = client.get(f"{UNPAYWALL_BASE_URL}/{doi}", params={"email": email}, timeout=10.0)
        response.raise_for_status()
        location = response.json().get("best_oa_location")
    except Exception as exc:
        print(f"  [warn] Unpaywall lookup failed for {doi}: {exc}")
        return None
    code = location.get("license") if location else None
    return _UNPAYWALL_LICENSE_URLS.get(code)


def resolve_license(
    isbn: str, doi: Optional[str], client: httpx.Client, contact_email: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """The one call most callers need: fetches isbn's Crossref
    book-chapter items, tries book_license_url on them, falls back to
    unpaywall_license_url(doi, ...) if that's None. Returns (license_url,
    license_source) with license_source in {"crossref", "unpaywall",
    None}. Never raises."""
    raw_items = crossref_book_chapter_items(isbn, client, contact_email)
    license_url = book_license_url(raw_items)
    if license_url is not None:
        return license_url, "crossref"
    license_url = unpaywall_license_url(doi, client, contact_email)
    return license_url, ("unpaywall" if license_url else None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_oa_license.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/oa_license.py tests/test_oa_license.py
git commit -m "feat: extract shared Crossref/Unpaywall license lookup into evaluation/oa_license.py"
```

## Task 5: Point `fetch_crossref_gt_corpus.py` at the shared module

**Files:**
- Modify: `evaluation/scripts/fetch_crossref_gt_corpus.py`

- [ ] **Step 1: Replace the import block**

Change (currently lines 29-44):

```python
import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx

from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)
```

to:

```python
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx

from evaluation.oa_license import (
    DEFAULT_CONTACT_EMAIL,
    book_license_url,
    crossref_book_chapter_items,
    unpaywall_license_url,
)
```

(`time`, `Counter`, and the three `chapter_segmentation.evidence.crossref_strategy` names are no longer used directly in this file -- they were only referenced inside the functions moved to `evaluation/oa_license.py` in Task 4.)

- [ ] **Step 2: Delete the moved constant and functions**

Delete this line (currently line 47):

```python
_DEFAULT_CONTACT_EMAIL = "boulanger@lhlt.mpg.de"
```

Delete this function (currently lines 74-109):

```python
def _crossref_book_chapters(isbn: str, client: httpx.Client, contact_email: Optional[str]) -> list[dict]:
    """GET .../works?filter=isbn:{isbn}, returning raw type=="book-chapter"
    items. Any network/HTTP/JSON failure is printed and treated as an empty
    result -- never raises, so one bad book never aborts the batch."""
    params: dict[str, str | int] = {
        "filter": f"isbn:{isbn}",
        "select": "DOI,title,subtitle,author,page,type,container-title,published,ISBN,license",
        "rows": 100,
    }
    if contact_email:
        params["mailto"] = contact_email

    response = None
    for _attempt in range(_MAX_RETRIES):
        try:
            response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
        except httpx.HTTPError as exc:
            print(f"  [warn] network error fetching Crossref metadata for {isbn}: {exc}")
            return []
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
        time.sleep(delay)
    else:
        print(f"  [warn] exhausted retries (429) fetching Crossref metadata for {isbn}")
        return []

    try:
        response.raise_for_status()
        items = response.json()["message"]["items"]
    except Exception as exc:
        print(f"  [warn] bad Crossref response for {isbn}: {exc}")
        return []

    return [item for item in items if item.get("type") == "book-chapter"]
```

Delete this function (currently lines 140-150):

```python
def _item_license_url(item: dict) -> Optional[str]:
    """The registered OA license URL for one Crossref item, preferring the
    version-of-record entry (content-version=="vor", delay-in-days==0) --
    the license that actually applies to the publicly available PDF, not
    an embargoed accepted-manuscript variant Crossref may register
    alongside it."""
    licenses = item.get("license") or []
    for entry in licenses:
        if entry.get("content-version") == "vor" and entry.get("delay-in-days", 0) == 0:
            return entry.get("URL")
    return licenses[0].get("URL") if licenses else None
```

Delete this function (currently lines 153-162):

```python
def _book_license_url(raw_items: list[dict]) -> Optional[str]:
    """The book's OA license URL, by majority vote across its chapters'
    individually-registered licenses -- in practice unanimous, since
    Crossref licenses are registered once per book and inherited by every
    chapter, but a vote is cheap insurance against one mis-registered
    chapter. None if no chapter has a registered license at all."""
    urls = [url for item in raw_items if (url := _item_license_url(item)) is not None]
    if not urls:
        return None
    return Counter(urls).most_common(1)[0][0]
```

Delete this line (currently line 165):

```python
_UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"
```

Delete this dict (currently lines 168-181, including its leading comment):

```python
# Unpaywall reports a short SPDX-ish code (e.g. "cc-by-nc"), not a URL --
# mapped to the CC 4.0 URL, since every book in this corpus is a 2020s
# publication and 4.0 is the only version any of its publishers use for
# new titles (found empirically: every Crossref-registered license in
# this same corpus that DOES carry an explicit version is 4.0).
_UNPAYWALL_LICENSE_URLS = {
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc-by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "cc-by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "pd": "https://creativecommons.org/publicdomain/mark/1.0/",
}
```

Delete this function (currently lines 184-206):

```python
def _unpaywall_license_url(doi: Optional[str], client: httpx.Client, contact_email: Optional[str]) -> Optional[str]:
    """Fallback license lookup via Unpaywall, tried only when Crossref has
    no license registered for a book. Crossref's license field is
    self-reported by the publisher at metadata-deposit time and often left
    blank; Unpaywall aggregates OA status from institutional repositories
    and publisher landing pages instead, so it is NOT the same data --
    found empirically: it recovers a license for every UCL Press /
    Athabasca University Press book in this corpus that Crossref has none
    for. Returns None if there's no DOI, no Unpaywall OA record, no
    license, or the request fails -- never raises, so one bad lookup never
    aborts the batch."""
    if not doi:
        return None
    email = contact_email or _DEFAULT_CONTACT_EMAIL
    try:
        response = client.get(f"{_UNPAYWALL_BASE_URL}/{doi}", params={"email": email}, timeout=10.0)
        response.raise_for_status()
        location = response.json().get("best_oa_location")
    except Exception as exc:
        print(f"  [warn] Unpaywall lookup failed for {doi}: {exc}")
        return None
    code = location.get("license") if location else None
    return _UNPAYWALL_LICENSE_URLS.get(code)
```

- [ ] **Step 3: Update the remaining call sites**

In `_fetch_crossref_metadata`, change:

```python
    raw_items = _crossref_book_chapters(isbn, client, contact_email)
    chapters = [c for item in raw_items if (c := _normalize_chapter(item)) is not None]
    license_url = _book_license_url(raw_items)
    license_source = "crossref" if license_url else None
    if license_url is None:
        license_url = _unpaywall_license_url(doi, client, contact_email)
        license_source = "unpaywall" if license_url else None
```

to:

```python
    raw_items = crossref_book_chapter_items(isbn, client, contact_email)
    chapters = [c for item in raw_items if (c := _normalize_chapter(item)) is not None]
    license_url = book_license_url(raw_items)
    license_source = "crossref" if license_url else None
    if license_url is None:
        license_url = unpaywall_license_url(doi, client, contact_email)
        license_source = "unpaywall" if license_url else None
```

In `main()`, change:

```python
    parser.add_argument("--contact-email", default=_DEFAULT_CONTACT_EMAIL, help="Crossref polite-pool contact email")
```

to:

```python
    parser.add_argument("--contact-email", default=DEFAULT_CONTACT_EMAIL, help="Crossref polite-pool contact email")
```

- [ ] **Step 4: Verify nothing broke**

Run: `uv run python -c "import evaluation.scripts.fetch_crossref_gt_corpus"`
Expected: no output, exit code 0

Run: `uv run python evaluation/scripts/fetch_crossref_gt_corpus.py --help`
Expected: prints the usage help text with no traceback

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/fetch_crossref_gt_corpus.py
git commit -m "refactor: fetch_crossref_gt_corpus.py imports license lookup from evaluation.oa_license"
```

## Task 6: Point `discover_crossref_candidates.py` at the shared module

**Files:**
- Modify: `evaluation/scripts/discover_crossref_candidates.py`

- [ ] **Step 1: Replace the cross-script private import**

Change (currently lines 47-57):

```python
from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)
from evaluation.scripts.fetch_crossref_gt_corpus import (
    _DEFAULT_CONTACT_EMAIL,
    _UNPAYWALL_BASE_URL,
    _item_license_url,
    _unpaywall_license_url,
)
```

to:

```python
from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)
from evaluation.oa_license import (
    DEFAULT_CONTACT_EMAIL,
    UNPAYWALL_BASE_URL,
    item_license_url,
    unpaywall_license_url,
)
```

(The `chapter_segmentation.evidence.crossref_strategy` import stays -- this script's own `_crossref_publisher_works` pagination logic uses those three names independently of license lookup.)

- [ ] **Step 2: Update the four renamed references**

In `_unpaywall_pdf_url`, change:

```python
    email = contact_email or _DEFAULT_CONTACT_EMAIL
    try:
        response = client.get(f"{_UNPAYWALL_BASE_URL}/{doi}", params={"email": email}, timeout=10.0)
```

to:

```python
    email = contact_email or DEFAULT_CONTACT_EMAIL
    try:
        response = client.get(f"{UNPAYWALL_BASE_URL}/{doi}", params={"email": email}, timeout=10.0)
```

In `discover()`, change:

```python
                    license_url = _item_license_url(item)
                    license_source = "crossref" if license_url else None
                    if license_url is None:
                        license_url = _unpaywall_license_url(doi, client, contact_email)
                        license_source = "unpaywall" if license_url else None
```

to:

```python
                    license_url = item_license_url(item)
                    license_source = "crossref" if license_url else None
                    if license_url is None:
                        license_url = unpaywall_license_url(doi, client, contact_email)
                        license_source = "unpaywall" if license_url else None
```

In `main()`, change:

```python
    parser.add_argument("--contact-email", default=_DEFAULT_CONTACT_EMAIL, help="Crossref/Unpaywall polite-pool contact email")
```

to:

```python
    parser.add_argument("--contact-email", default=DEFAULT_CONTACT_EMAIL, help="Crossref/Unpaywall polite-pool contact email")
```

- [ ] **Step 3: Verify the existing test file still passes**

Run: `uv run pytest tests/test_discover_crossref_candidates.py -v`
Expected: PASS (this test file never imports the two renamed license functions directly -- it only imports `discover_crossref_candidates`'s own local functions -- so this run just confirms the module-level import change didn't break anything else in the file).

Run: `uv run python evaluation/scripts/discover_crossref_candidates.py --help`
Expected: prints the usage help text with no traceback

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/discover_crossref_candidates.py
git commit -m "refactor: discover_crossref_candidates.py imports license lookup from evaluation.oa_license"
```

## Task 7: `evaluation/scripts/promote_pending_book.py`

**Files:**
- Create: `evaluation/scripts/promote_pending_book.py`
- Test: `tests/test_promote_pending_book.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_promote_pending_book.py`:

```python
"""Unit tests for evaluation/scripts/promote_pending_book.py's
promote_book() against temp-directory fake corpora -- no real
evaluation/corpus/ data touched. The bounds/overlap gate and the
missing-manifest-entry/missing-ground-truth gates need no network; the
open-access license-resolution path is exercised with a mocked httpx
client (same convention as tests/test_discover_crossref_candidates.py)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from pypdf import PdfWriter

from evaluation.scripts.promote_pending_book import promote_book


def _write_blank_pdf(path: Path, num_pages: int) -> None:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)


def _write_manifest(path: Path, books: list[dict]) -> None:
    path.write_text(json.dumps({"books": books}), encoding="utf-8")


def _write_expected(path: Path, chapters: list[dict]) -> None:
    path.write_text(json.dumps({"chapters": chapters}), encoding="utf-8")


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


_BOOK = {
    "filename": "9781234567897.pdf",
    "title": "Test Book",
    "language": "en",
    "extraction_type": "native",
    "embedded_toc": True,
    "oa": True,
    "doi": "10.1/test",
    "download_url": "https://example.org/test.pdf",
}

_VALID_CHAPTERS = [
    {"title": "Introduction", "authors": [], "pdf_start_index": 0, "pdf_end_index": 4, "citation_pages": "1-5"},
    {"title": "Chapter One", "authors": [], "pdf_start_index": 5, "pdf_end_index": 9, "citation_pages": "6-10"},
]

_OVERLAPPING_CHAPTERS = [
    {"title": "Introduction", "authors": [], "pdf_start_index": 0, "pdf_end_index": 6, "citation_pages": "1-7"},
    {"title": "Chapter One", "authors": [], "pdf_start_index": 5, "pdf_end_index": 9, "citation_pages": "6-10"},
]


class _CorpusFixture:
    """Builds a temp pending_dir + target_dir pair, both starting with an
    empty manifest.json, seeded by with_pending_book()."""

    def __init__(self, tmp: Path):
        self.pending_dir = tmp / "pending"
        self.target_dir = tmp / "open-access"
        self.pending_dir.mkdir()
        self.target_dir.mkdir()
        _write_manifest(self.pending_dir / "manifest.json", [])
        _write_manifest(self.target_dir / "manifest.json", [])

    def with_pending_book(self, isbn: str, num_pages: int, chapters: list[dict]) -> "_CorpusFixture":
        manifest = json.loads((self.pending_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["books"].append({**_BOOK, "filename": f"{isbn}.pdf"})
        _write_manifest(self.pending_dir / "manifest.json", manifest["books"])
        _write_blank_pdf(self.pending_dir / f"{isbn}.pdf", num_pages)
        _write_expected(self.pending_dir / f"{isbn}.expected.json", chapters)
        return self


class TestPromoteBookGates(unittest.TestCase):
    def test_skips_isbn_not_in_pending_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp))
            _isbn, outcome = promote_book(
                "0000000000000", fixture.pending_dir, fixture.target_dir, "open-access", Mock(), None, dry_run=True
            )
        self.assertTrue(outcome.startswith("SKIP: not in pending/manifest.json"))

    def test_skips_when_no_expected_json_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp))
            manifest = json.loads((fixture.pending_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["books"].append({**_BOOK, "filename": "9781234567897.pdf"})
            _write_manifest(fixture.pending_dir / "manifest.json", manifest["books"])
            _write_blank_pdf(fixture.pending_dir / "9781234567897.pdf", 10)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir, "open-access", Mock(), None, dry_run=True
            )
        self.assertTrue(outcome.startswith("SKIP: no ground truth yet"))

    def test_skips_when_bounds_overlap_check_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _OVERLAPPING_CHAPTERS)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir, "open-access", Mock(), None, dry_run=True
            )
        self.assertTrue(outcome.startswith("SKIP: bounds/overlap check failed"))
        self.assertIn("overlap", outcome)


class TestPromoteBookDryRun(unittest.TestCase):
    def test_dry_run_moves_nothing_and_reports_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _VALID_CHAPTERS)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir,
                "copyrighted-scans", Mock(), None, dry_run=True,
            )
            self.assertTrue(outcome.startswith("OK (dry-run): would move to copyrighted-scans/"))
            self.assertTrue((fixture.pending_dir / "9781234567897.pdf").exists())
            self.assertTrue((fixture.pending_dir / "9781234567897.expected.json").exists())
            self.assertFalse((fixture.target_dir / "9781234567897.pdf").exists())
            target_manifest = json.loads((fixture.target_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(target_manifest["books"], [])


class TestPromoteBookRealMove(unittest.TestCase):
    def test_moves_files_and_updates_both_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _VALID_CHAPTERS)
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir,
                "copyrighted-scans", Mock(), None, dry_run=False,
            )
            self.assertTrue(outcome.startswith("OK: moved to copyrighted-scans/"))
            self.assertFalse((fixture.pending_dir / "9781234567897.pdf").exists())
            self.assertFalse((fixture.pending_dir / "9781234567897.expected.json").exists())
            self.assertTrue((fixture.target_dir / "9781234567897.pdf").exists())
            self.assertTrue((fixture.target_dir / "9781234567897.expected.json").exists())

            pending_manifest = json.loads((fixture.pending_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(pending_manifest["books"], [])
            target_manifest = json.loads((fixture.target_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(target_manifest["books"]), 1)
            self.assertEqual(target_manifest["books"][0]["filename"], "9781234567897.pdf")
            self.assertNotIn("license", target_manifest["books"][0])

    def test_open_access_target_resolves_and_writes_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _CorpusFixture(Path(tmp)).with_pending_book("9781234567897", 10, _VALID_CHAPTERS)
            client = Mock()
            client.get.return_value = _json_response({
                "message": {"items": [{
                    "type": "book-chapter",
                    "license": [{
                        "URL": "https://creativecommons.org/licenses/by/4.0/",
                        "content-version": "vor",
                        "delay-in-days": 0,
                    }],
                }]}
            })
            _isbn, outcome = promote_book(
                "9781234567897", fixture.pending_dir, fixture.target_dir,
                "open-access", client, None, dry_run=False,
            )
            self.assertTrue(outcome.startswith("OK: moved to open-access/"))
            target_manifest = json.loads((fixture.target_dir / "manifest.json").read_text(encoding="utf-8"))
            book = target_manifest["books"][0]
            self.assertEqual(book["license"], "https://creativecommons.org/licenses/by/4.0/")
            self.assertEqual(book["license_source"], "crossref")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_promote_pending_book.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.promote_pending_book'`

- [ ] **Step 3: Create `evaluation/scripts/promote_pending_book.py`**

```python
#!/usr/bin/env python3
"""Promotes one or more evaluation/corpus/pending/ books -- ones that
already have a hand-verified .expected.json -- into a real corpus
(open-access/ or copyrighted-scans/), completing the step
evaluation/CLAUDE.md's Step 0a otherwise only described as manual ("the
entry moves into whichever real corpus it belongs in").

Two things this automates instead of doing by hand each time (see
docs/superpowers/specs/2026-08-12-ground-truth-pipeline-hardening-design.md):
1. The bounds/overlap sanity check (evaluation/harness.py's
   chapter_bounds_errors) -- refuses to promote a book whose
   .expected.json has a structural defect (overlapping chapter ranges,
   start>end, or an end index past the PDF's real page count), rather
   than letting it slip into a real corpus undetected.
2. Open-access license resolution (evaluation/oa_license.py) -- for
   --corpus open-access, looks up license/license_source via Crossref
   (falling back to Unpaywall) instead of leaving those fields to be
   filled in by hand. --corpus copyrighted-scans never adds these fields
   (that corpus's manifest schema has no license/license_source, per
   evaluation/README.md).

A pending/ entry that only exists in manifest.local.json (no DOI) is not
supported here -- promoting one is rare and needs a human decision about
whether the book can ever be shared at all, so it stays a manual
operation.

Usage:
    uv run python evaluation/scripts/promote_pending_book.py 9781234567897 --corpus open-access
    uv run python evaluation/scripts/promote_pending_book.py 9781234567897 9781234567904 --corpus copyrighted-scans
    uv run python evaluation/scripts/promote_pending_book.py 9781234567897 --corpus open-access --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
from pypdf import PdfReader

from evaluation.harness import chapter_bounds_errors, corpus_dir
from evaluation.oa_license import DEFAULT_CONTACT_EMAIL, resolve_license


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest["books"] = sorted(manifest["books"], key=lambda b: b["filename"])
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def promote_book(
    isbn: str,
    pending_dir: Path,
    target_dir: Path,
    target_corpus: str,
    client: httpx.Client,
    contact_email: Optional[str],
    dry_run: bool,
) -> tuple[str, str]:
    """Returns (isbn, outcome_message). Never raises -- one bad ISBN in a
    batch must not abort the rest."""
    pending_manifest_path = pending_dir / "manifest.json"
    pending_manifest = _load_manifest(pending_manifest_path)
    book = next((b for b in pending_manifest["books"] if b["filename"] == f"{isbn}.pdf"), None)
    if book is None:
        return isbn, "SKIP: not in pending/manifest.json (manifest.local.json entries aren't supported)"

    pending_pdf = pending_dir / f"{isbn}.pdf"
    pending_expected = pending_dir / f"{isbn}.expected.json"
    if not pending_expected.exists():
        return isbn, "SKIP: no ground truth yet (pending/<isbn>.expected.json missing)"
    if not pending_pdf.exists():
        return isbn, "SKIP: no PDF (pending/<isbn>.pdf missing)"

    total_pages = len(PdfReader(str(pending_pdf)).pages)
    chapters = json.loads(pending_expected.read_text(encoding="utf-8"))["chapters"]
    errors = chapter_bounds_errors(chapters, total_pages)
    if errors:
        return isbn, f"SKIP: bounds/overlap check failed: {'; '.join(errors)}"

    book = dict(book)
    if target_corpus == "open-access":
        license_url, license_source = resolve_license(isbn, book.get("doi"), client, contact_email)
        book["license"] = license_url
        book["license_source"] = license_source
        if license_url is None:
            print(f"  [warn] {isbn}: no license found on Crossref or Unpaywall")

    license_note = (
        f" (license={book.get('license')!r} via {book.get('license_source')})"
        if target_corpus == "open-access" else ""
    )

    if dry_run:
        return isbn, f"OK (dry-run): would move to {target_corpus}/{license_note}"

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pending_pdf), str(target_dir / f"{isbn}.pdf"))
    shutil.move(str(pending_expected), str(target_dir / f"{isbn}.expected.json"))

    target_manifest_path = target_dir / "manifest.json"
    target_manifest = _load_manifest(target_manifest_path)
    target_manifest["books"].append(book)
    _write_manifest(target_manifest_path, target_manifest)

    pending_manifest["books"] = [b for b in pending_manifest["books"] if b["filename"] != f"{isbn}.pdf"]
    _write_manifest(pending_manifest_path, pending_manifest)

    return isbn, f"OK: moved to {target_corpus}/{license_note}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("isbn", nargs="+", help="One or more pending/ ISBNs (matching <isbn>.pdf) to promote")
    parser.add_argument(
        "--corpus", required=True, choices=["open-access", "copyrighted-scans"], help="Target corpus"
    )
    parser.add_argument(
        "--contact-email", default=DEFAULT_CONTACT_EMAIL, help="Crossref/Unpaywall polite-pool contact email"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would happen without moving files or writing manifests"
    )
    args = parser.parse_args()

    pending_dir = corpus_dir("pending")
    target_dir = corpus_dir(args.corpus)

    results = []
    with httpx.Client(follow_redirects=True) as client:
        for isbn in args.isbn:
            result = promote_book(
                isbn, pending_dir, target_dir, args.corpus, client, args.contact_email, args.dry_run
            )
            results.append(result)
            print(f"[{result[0]}] {result[1]}")

    n_ok = sum(1 for _, msg in results if msg.startswith("OK"))
    print(f"\n{n_ok}/{len(results)} book(s) {'would be ' if args.dry_run else ''}promoted to {args.corpus}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_promote_pending_book.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Smoke-test the CLI against the real corpus with `--dry-run`**

Run: `uv run python evaluation/scripts/promote_pending_book.py 0000000000000 --corpus open-access --dry-run`
Expected: prints `[0000000000000] SKIP: not in pending/manifest.json (manifest.local.json entries aren't supported)` followed by a `0/1 book(s) would be promoted to open-access/` summary line, exit code 0. This confirms the script runs end-to-end against the real `evaluation/corpus/pending/` directory (via `corpus_dir("pending")`) without needing a real ISBN to exercise the "not found" path.

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/promote_pending_book.py tests/test_promote_pending_book.py
git commit -m "feat: add promote_pending_book.py script with bounds gate and license lookup"
```

## Task 8: Documentation updates and final verification

**Files:**
- Modify: `evaluation/CLAUDE.md`

- [ ] **Step 1: Update Step 0a's pending-corpus bullet**

In `evaluation/CLAUDE.md`, change:

```markdown
- **No ground truth built yet** (you only have the PDF and basic metadata
  so far) → `pending/`. Move the entry into `open-access/` or
  `copyrighted-scans/` once its `.expected.json` exists.
```

to:

```markdown
- **No ground truth built yet** (you only have the PDF and basic metadata
  so far) → `pending/`. Once its `.expected.json` exists, promote it into
  `open-access/` or `copyrighted-scans/` with
  `uv run python evaluation/scripts/promote_pending_book.py <isbn> --corpus <open-access|copyrighted-scans>`
  (add `--dry-run` to preview first) -- it re-runs the Step 4 bounds/overlap
  check as a gate and, for `open-access/`, resolves `license`/`license_source`
  via Crossref/Unpaywall automatically. Only entries already in that
  corpus's committed `manifest.json` are supported (not
  `manifest.local.json` -- promote those by hand).
```

- [ ] **Step 2: Update Step 4's sanity-check paragraph**

In `evaluation/CLAUDE.md`, change:

```markdown
in this directory for a worked example). Then run the bounds/overlap sanity
check before committing (or before considering a local-only entry "done"):
```

to:

```markdown
in this directory for a worked example). Then run the bounds/overlap sanity
check before committing (or before considering a local-only entry "done") --
the same check also runs automatically for every corpus's ground truth via
`tests/test_ground_truth_integrity.py` (part of the default `uv run pytest`)
and as a hard gate inside `evaluation/scripts/promote_pending_book.py`, but
running it by hand here catches a mistake before it's even written to disk:
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests, including every one added or modified by this plan; integration-marked tests are excluded by default per `pyproject.toml`'s `addopts`)

- [ ] **Step 4: Commit**

```bash
git add evaluation/CLAUDE.md
git commit -m "docs: point CLAUDE.md at promote_pending_book.py and the automatic bounds check"
```
