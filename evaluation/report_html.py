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
