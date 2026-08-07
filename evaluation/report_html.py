"""Shared HTML table rendering for generate_report.py's main report and
its LLM detail page -- one renderer so both pages look and behave
identically. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"Metrics and rendering (shared code)".
"""

from evaluation.metrics import Metrics

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


def render_strategy_tables(
    title: str,
    description_html: str,
    strategy_names: list[str],
    per_document: dict[str, dict[str, _TableCell]],
    aggregates: dict[str, Metrics],
    aggregate_times: dict[str, float],
) -> str:
    """
    strategy_names: column order for the per-document table.
    per_document: {document_key: {strategy_name: (Metrics, elapsed_seconds) or None}}.
        None (or a missing key) means the strategy produced no result for
        that document -- rendered as "N/A".
    aggregates: {strategy_name: Metrics} -- micro-aggregate across every
        document that strategy actually ran on.
    aggregate_times: {strategy_name: total_elapsed_seconds}.
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
            is_best = cell is not None and best_f1 is not None and cell[0].f1 == best_f1
            row_cells.append(_cell_html(cell, is_best))
        doc_rows.append(f"<tr><td>{doc_key}</td>{''.join(row_cells)}</tr>")

    ranked_strategies = sorted(aggregates, key=lambda s: aggregates[s].f1, reverse=True)
    agg_rows = []
    for strategy in ranked_strategies:
        m = aggregates[strategy]
        t = aggregate_times.get(strategy, 0.0)
        agg_rows.append(
            f"<tr><td>{strategy}</td><td>{m.precision:.2f}</td><td>{m.recall:.2f}</td>"
            f"<td>{m.f1:.2f}</td><td>{m.true_positives}/{m.found_count} found, "
            f"{m.true_positives}/{m.expected_count} expected</td><td>{t:.2f}s</td></tr>"
        )

    doc_header = "".join(f"<th>{s}</th>" for s in strategy_names)
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
<h2>Per strategy (aggregate, ordered by F1)</h2>
<table>
<tr><th>Strategy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Found / Expected</th><th>Total time</th></tr>
{"".join(agg_rows)}
</table>
</body></html>
"""
