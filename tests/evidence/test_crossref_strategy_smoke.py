"""Real-network smoke test for CrossrefMetadataStrategy against every
evaluation/corpus/open-access/ book.

31 of the 37 books' ground truth (evaluation/corpus/open-access/*.expected.json)
was built directly from Crossref-registered book-chapter records
(evaluation/scripts/build_crossref_gt_ground_truth.py, reconciling
evaluation/crossref_gt/) -- for those, a real CrossrefMetadataStrategy.fetch()
lookup should recover every expected chapter almost by construction. Titles
are matched with rapidfuzz's partial_ratio (the same fuzzy-match convention
build_crossref_gt_ground_truth.py itself uses to confirm a candidate page,
via _locate_near), not exact equality, to tolerate two known, harmless
sources of divergence between what's stored in .expected.json and what a
fresh Crossref query returns right now: hand-transcribed titles that used a
straight apostrophe where Crossref's typographic record uses a curly one
(9783031466373), and one book whose .expected.json predates the
title+subtitle join fix in _parse_crossref_item, so it stores the bare,
Crossref-truncated title instead of the full heading (9783847432364; see
test_crossref_strategy.py's test_appends_subtitle_to_the_truncated_crossref_title
for the same underlying Crossref title/subtitle split).

The other 6 books' ground truth was hand-built the older way (TOC
transcription + evaluation/scripts/ground_truth_helper.py's content search),
not reconciled from Crossref at all (they aren't listed in
evaluation/crossref_gt/manifest.json). Of those 6, two (9783031466373,
9783847432364) still happen to have real Crossref book-chapter records
under their own ISBN, so they're scored normally above. The remaining four
-- 9781771993661, 9782375460122, 9783907297285, 9783907297339 -- have been
confirmed (by direct query against api.crossref.org) to have *zero*
book-chapter records under their ISBN at all: Crossref only holds a
book-level record (or, for 9782375460122, no record whatsoever) for them.
No amount of fuzzy matching can recover chapters Crossref was never asked
to register, so CrossrefMetadataStrategy.fetch() is expected to return an
empty list for these four, and that expectation is asserted explicitly
rather than silently skipped -- if Crossref ever gains chapter-level
records for one of these ISBNs, this test starts failing and is the signal
to move that book's ground truth over to the crossref_gt reconciliation
path (see evaluation/CLAUDE.md).

Marked "integration": makes ~37 real, uncached GET requests to
api.crossref.org (no mocking -- that's the point). Excluded from the
default `uv run pytest` run (pyproject.toml addopts). Run directly:

    uv run pytest tests/evidence/test_crossref_strategy_smoke.py -q -s
"""

import json
import unittest
from pathlib import Path

import httpx
import pytest
from rapidfuzz import fuzz

from chapter_segmentation.evidence.crossref_strategy import fetch_crossref_chapters
from evaluation.harness import corpus_dir, load_manifest_books

pytestmark = pytest.mark.integration

_TITLE_MATCH_THRESHOLD = 95.0
_CONTACT_EMAIL = "boulanger@lhlt.mpg.de"

# See module docstring: these four books' ground truth was hand-built from
# the printed TOC, and Crossref has no book-chapter records at all under
# their ISBN (verified directly against the API) -- a real lookup is
# expected to come back empty, not to recover the expected chapters.
_KNOWN_NO_CROSSREF_CHAPTERS = {
    "9781771993661",
    "9782375460122",
    "9783907297285",
    "9783907297339",
}


class TestCrossrefStrategyAgainstOpenAccessCorpus(unittest.IsolatedAsyncioTestCase):
    async def test_every_expected_chapter_is_found_by_a_real_lookup(self):
        books = load_manifest_books("open-access")
        cdir = corpus_dir("open-access")
        self.assertTrue(books, "evaluation/corpus/open-access/manifest.json has no books")

        async with httpx.AsyncClient() as http_client:
            for book in books:
                isbn = Path(book["filename"]).stem
                expected_path = cdir / f"{isbn}.expected.json"
                if not expected_path.exists():
                    continue
                with self.subTest(isbn=isbn, title=book["title"]):
                    expected = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
                    fetched = await fetch_crossref_chapters(
                        isbn, http_client, cache_dir=None, contact_email=_CONTACT_EMAIL
                    )

                    if isbn in _KNOWN_NO_CROSSREF_CHAPTERS:
                        self.assertEqual(
                            fetched, [],
                            f"{isbn}: expected zero Crossref book-chapter records (known gap, see module "
                            f"docstring) but got {len(fetched)} -- move this book's ground truth to the "
                            f"crossref_gt reconciliation path (evaluation/CLAUDE.md)",
                        )
                        continue

                    fetched_titles = [c.title for c in fetched]
                    misses = [
                        c["title"] for c in expected
                        if max(
                            (fuzz.partial_ratio(c["title"], t) for t in fetched_titles), default=0.0
                        ) < _TITLE_MATCH_THRESHOLD
                    ]
                    recall = (len(expected) - len(misses)) / len(expected) if expected else 0.0
                    print(f"open-access/{isbn}: recall={recall:.2f} ({len(expected) - len(misses)}/{len(expected)})")
                    self.assertEqual(
                        misses, [],
                        f"{isbn}: Crossref lookup did not recover {len(misses)}/{len(expected)} expected "
                        f"chapter(s): {misses}",
                    )


if __name__ == "__main__":
    unittest.main()
