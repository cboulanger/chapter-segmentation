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
