#!/usr/bin/env python3
"""Generates a prose-free static results page from the committed
public-cache corpus -- see design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md.
Runs the heuristic and outline strategies live (no network/API calls); if
evaluation/llm-cache/ has cached LLM results, folds in the single
best-performing cached model too. Also regenerates public/llm/index.html,
a full breakdown of every cached LLM model. No LLM call anywhere in this
path; a plain f-string template, no templating-engine dependency.

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
    LLM_CACHE_DIR,
    available_public_books,
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


def _load_llm_cache(manifest_key: str) -> dict:
    cache_path = LLM_CACHE_DIR / f"{manifest_key}.json"
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8")).get("models", {})


def _best_llm_model(books: list[tuple[str, list[dict]]]) -> str | None:
    """books: [(manifest_key, expected_chapters)]. Picks the cached model
    with the highest micro-F1 aggregated across every book that has a
    cache entry for it (a model with partial corpus coverage is still
    scored, on however many books it has -- see design spec's "LLM
    results cache"), ties broken by lower total time. Returns None if no
    book has any cached LLM result at all."""
    per_model_aggregate: dict[str, MicroAggregate] = {}
    for manifest_key, expected in books:
        for model_id, entry in _load_llm_cache(manifest_key).items():
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


def generate(out_dir: Path) -> None:
    books = available_public_books()
    expected_by_key = {
        key: json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
        for key, expected_path, _book in books
    }

    best_llm_model = _best_llm_model(list(expected_by_key.items()))
    llm_strategy_name = f"LLM ({best_llm_model})" if best_llm_model else None
    strategy_names = [HEURISTIC, OUTLINE] + ([llm_strategy_name] if llm_strategy_name else [])

    per_document: dict[str, dict] = {}
    heuristic_agg, outline_agg, llm_agg = MicroAggregate(), MicroAggregate(), MicroAggregate()
    heuristic_citation_agg, outline_citation_agg, llm_citation_agg = (
        CitationPageAggregate(), CitationPageAggregate(), CitationPageAggregate(),
    )

    for manifest_key, _expected_path, _book in books:
        pages = public_pages_for(manifest_key)
        expected = expected_by_key[manifest_key]
        cells: dict = {}

        start = time.perf_counter()
        heuristic_result = analyze_attachment(pages)
        heuristic_elapsed = time.perf_counter() - start
        heuristic_metrics = precision_recall_f1(expected, heuristic_result["chapters"])
        heuristic_agg.add(heuristic_metrics, heuristic_elapsed)
        heuristic_citation_agg.add(citation_pages_metrics(expected, heuristic_result["chapters"]))
        cells[HEURISTIC] = (heuristic_metrics, heuristic_elapsed)

        outline_candidates = public_outline_candidates_for(manifest_key)
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
            llm_entry = _load_llm_cache(manifest_key).get(best_llm_model)
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

    description = """<p>Each book has a hand-verified <code>*.expected.json</code> ground truth (real
chapter boundaries as exact PDF page ranges). Each strategy below is run
independently against the same pages -- no pipeline merge/fallback logic
is involved, so this reflects each strategy's own standalone accuracy, not
a production routing decision. A match requires the exact same page range
-- no partial credit. For per-book root-cause notes, see
<a href="https://github.com/cboulanger/chapter-segmentation/blob/main/evaluation/RESULTS.md">RESULTS.md</a>.
The full breakdown of every LLM model ever evaluated (not just the best)
is at <a href="llm/index.html">llm/index.html</a>.</p>"""

    html = render_strategy_tables(
        title="chapter-segmentation: public-cache corpus results",
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

    _generate_llm_detail_page(out_dir, list(expected_by_key.items()))


def _generate_llm_detail_page(out_dir: Path, books: list[tuple[str, list[dict]]]) -> None:
    model_ids: set[str] = set()
    for manifest_key, _expected in books:
        model_ids.update(_load_llm_cache(manifest_key).keys())

    if not model_ids:
        html = (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<title>chapter-segmentation LLM results</title></head><body>"
            "<h1>chapter-segmentation: LLM strategy results</h1>"
            "<p>No cached LLM results yet -- run "
            "<code>evaluation/refresh_llm_cache.py</code>.</p></body></html>"
        )
    else:
        per_document: dict[str, dict] = {}
        aggregates_acc = {model_id: MicroAggregate() for model_id in model_ids}
        citation_aggregates_acc = {model_id: CitationPageAggregate() for model_id in model_ids}
        for manifest_key, expected in books:
            cache = _load_llm_cache(manifest_key)
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
            title="chapter-segmentation: LLM strategy results (all cached models)",
            description_html=(
                "<p>Every KISSKI model ever evaluated by "
                "<code>evaluation/refresh_llm_cache.py</code>, run standalone via "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="public")
    args = parser.parse_args()
    generate(Path(args.out))
