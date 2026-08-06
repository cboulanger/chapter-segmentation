#!/usr/bin/env python3
"""Runs the chapter-segmentation evaluation set through
analyze_attachment_with_llm_fallback instead of the pure-heuristic
analyze_attachment, and prints the same precision/recall table format
test_segmentation_accuracy.py already uses, plus per-book fallback-usage
counts.

Requires the `llm-eval` extra (openai) and a real, working LLM endpoint --
costs a paid API call per book. Not a pytest test, run manually:

    OPENAI_API_KEY=... uv run python evaluation/scripts/evaluate_chapter_segmentation_llm_fallback.py

Pass --base-url to point at any OpenAI-compatible endpoint (e.g. KISSKI)
instead of api.openai.com, and --model to pick the model:

    OPENAI_API_KEY=... uv run python evaluation/scripts/evaluate_chapter_segmentation_llm_fallback.py \\
      --base-url https://chat-ai.academiccloud.de/v1 --model meta-llama-3.1-8b-instruct
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.segmentation import analyze_attachment_with_llm_fallback
from evaluation.harness import analysis_pages_for, available_books


class _OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) backed by
    any OpenAI-compatible chat completions endpoint. Deliberately does not
    replicate zotero-rag's own multi-provider preset/model-rotation
    machinery -- that lives with the Zotero integration, not this
    standalone evaluation script.
    """

    def __init__(self, model: str, base_url: Optional[str] = None):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(base_url=base_url, api_key=os.environ["OPENAI_API_KEY"])
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


async def _main(model: str, base_url: Optional[str]) -> int:
    pairs = available_books()
    if not pairs:
        print("No evaluation PDFs present -- run: uv run python evaluation/scripts/fetch_evaluation_pdfs.py")
        return 1

    llm_client = _OpenAICompatibleLLMClient(model=model, base_url=base_url)

    for pdf_path, expected_path, _book in pairs:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
        pages = analysis_pages_for(pdf_path.read_bytes())
        if pages is None:
            print(f"{pdf_path.name}: SKIPPED (needs OCR — populate the cache with: "
                  f"uv run python evaluation/scripts/ocr_evaluation_pdfs.py)")
            continue
        result = await analyze_attachment_with_llm_fallback(pages, llm_client)

        expected_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in expected}
        found_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in result["chapters"]}
        true_positives = expected_ranges & found_ranges

        precision = len(true_positives) / len(found_ranges) if found_ranges else 0.0
        recall = len(true_positives) / len(expected_ranges) if expected_ranges else 0.0
        diag = result["diagnostics"]
        print(
            f"{pdf_path.name}: precision={precision:.2f} recall={recall:.2f} "
            f"({len(true_positives)}/{len(found_ranges)} found, {len(true_positives)}/{len(expected_ranges)} expected) "
            f"llm_toc_extraction_used={diag.get('llm_toc_extraction_used')} "
            f"llm_disambiguation_used={diag.get('llm_disambiguation_used')}"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(model=args.model, base_url=args.base_url)))
