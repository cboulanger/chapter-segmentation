#!/usr/bin/env python3
"""Generates a prose-free static results page from the committed
public-cache corpus -- see design spec section 11 (in the zotero-rag repo
this was extracted from). No LLM call anywhere in this path; a plain
f-string template, no templating-engine dependency.

    uv run python evaluation/generate_report.py --out public/
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapter_segmentation.segmentation import analyze_attachment
from evaluation.harness import available_public_books, public_pages_for


def compute_precision_recall(expected: list[dict], found: list[dict]) -> tuple[float, float, int, int, int]:
    """Returns (precision, recall, true_positives, found_count, expected_count)."""
    expected_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in expected}
    found_ranges = {(c["pdf_start_index"], c["pdf_end_index"]) for c in found}
    true_positives = expected_ranges & found_ranges
    precision = len(true_positives) / len(found_ranges) if found_ranges else 0.0
    recall = len(true_positives) / len(expected_ranges) if expected_ranges else 0.0
    return precision, recall, len(true_positives), len(found_ranges), len(expected_ranges)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _row(manifest_key: str, precision: float, recall: float, tp: int, found: int, expected: int) -> str:
    return (
        f"<tr><td>{manifest_key}</td><td>{precision:.2f}</td><td>{recall:.2f}</td>"
        f"<td>{tp}/{found} found, {tp}/{expected} expected</td></tr>"
    )


def generate(out_dir: Path) -> None:
    import json

    rows: list[str] = []
    total_tp = total_found = total_expected = 0

    for manifest_key, expected_path, _book in available_public_books():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))["chapters"]
        pages = public_pages_for(manifest_key)
        result = analyze_attachment(pages)
        precision, recall, tp, found, exp = compute_precision_recall(expected, result["chapters"])
        rows.append(_row(manifest_key, precision, recall, tp, found, exp))
        total_tp += tp
        total_found += found
        total_expected += exp

    micro_precision = total_tp / total_found if total_found else 0.0
    micro_recall = total_tp / total_expected if total_expected else 0.0

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>chapter-segmentation results</title>
<style>table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}</style>
</head><body>
<h1>chapter-segmentation: public-cache corpus results</h1>
<p>Each book has a hand-verified <code>*.expected.json</code> ground truth (real chapter
boundaries as exact PDF page ranges). "Found" is what <code>analyze_attachment()</code>
detected on the same pages. A match requires the exact same page range -- no partial
credit. Precision = correct / found; Recall = correct / expected.
For per-book root-cause notes (why a given score is what it is), see
<a href="https://github.com/cboulanger/chapter-segmentation/blob/main/evaluation/RESULTS.md">RESULTS.md</a>.</p>
<table>
<tr><th>Book</th><th>Precision</th><th>Recall</th><th>Found / Expected</th></tr>
{"".join(rows)}
<tr><th>Aggregate (micro)</th><th>{micro_precision:.2f}</th><th>{micro_recall:.2f}</th><th>{total_tp}/{total_found} found, {total_tp}/{total_expected} expected</th></tr>
</table>
<p>Generated {datetime.now(timezone.utc).isoformat()} from commit {_git_sha()}.</p>
</body></html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="public")
    args = parser.parse_args()
    generate(Path(args.out))
