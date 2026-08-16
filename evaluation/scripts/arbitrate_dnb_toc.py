"""Surfaces dnb-toc-only books whose two vision-model TOC extractions
didn't clear generate_dnb_toc_ground_truth.py's agreement gate, so a
Claude Code session can arbitrate the conflict directly -- see design
spec docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md.
This script only REPORTS and records rejections; it never decides. The
arbitrator reads a book's report, opens the PDF's actual TOC pages via
the Read tool when the text alone doesn't settle it, then either writes
evaluation/corpus/dnb-toc-only/<key>.expected.json directly (same schema
as a passing book, "verified": true) or runs this script's `reject`
subcommand to permanently record the book as unrecoverable.

    uv run python evaluation/scripts/arbitrate_dnb_toc.py
    uv run python evaluation/scripts/arbitrate_dnb_toc.py reject 9783515114868 "both models hallucinate on this scan"
"""

import argparse
import json
from datetime import date
from pathlib import Path

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import diff_toc_entries
from evaluation.dnb_toc_vision import load_cached_llm_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key

_CORPUS_NAME = "dnb-toc-only"


def _rejected_path(cdir: Path) -> Path:
    return cdir / "arbitration-rejected.json"


def _load_rejected_keys(cdir: Path) -> set[str]:
    path = _rejected_path(cdir)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["key"] for entry in data["rejected"]}


def _cached_book_keys(cache_directory: Path) -> list[str]:
    """Every distinct book key with at least one <key>.<model>.json file
    in cache_directory, sorted for stable output."""
    return sorted({p.name.split(".", 1)[0] for p in cache_directory.glob("*.json")})


def _cached_models_for_book(cache_directory: Path, key: str) -> dict[str, list[TocEntry]]:
    """Every model's cached entries for one book key, keyed by model id
    (the cache filename's middle segment, <key>.<model>.json -- sliced
    rather than split on ".", since a model id can itself contain a dot,
    e.g. "qwen3.6-27b")."""
    result: dict[str, list[TocEntry]] = {}
    for path in sorted(cache_directory.glob(f"{key}.*.json")):
        model = path.name[len(key) + 1: -len(".json")]
        entries = load_cached_llm_entries(cache_directory, key, model)
        if entries is not None:
            result[model] = entries
    return result


def books_needing_arbitration(cdir: Path, cache_directory: Path) -> list[str]:
    """Book keys with cached model output, no .expected.json yet, and not
    already permanently rejected."""
    rejected = _load_rejected_keys(cdir)
    needing = []
    for key in _cached_book_keys(cache_directory):
        if key in rejected:
            continue
        if (cdir / f"{key}.expected.json").exists():
            continue
        needing.append(key)
    return needing


def _format_entry(entry: TocEntry) -> str:
    page = entry.printed_page_number if entry.printed_page_number != -1 else "?"
    return f"    p.{page!s:>4}  {entry.title}"


def format_book_report(key: str, title: str, pdf_path: Path, models_to_entries: dict[str, list[TocEntry]]) -> str:
    """Human-readable diff for one book -- the actual disagreement, ready
    for Claude (or a human) to arbitrate. Handles the normal two-model
    case, the single-surviving-model case (the other model's response
    was empty/malformed), and defensively falls back to a plain per-model
    listing for any other count."""
    lines = [f"=== {key} -- {title} ===", f"PDF: {pdf_path}"]
    model_names = sorted(models_to_entries)
    if len(model_names) == 1:
        model = model_names[0]
        entries = models_to_entries[model]
        lines.append(f"Only {model} returned usable output ({len(entries)} entries) -- verify directly against the page images:")
        for entry in entries:
            lines.append(_format_entry(entry))
        return "\n".join(lines)
    if len(model_names) != 2:
        lines.append(f"Expected 1 or 2 cached models, found {len(model_names)}: {model_names} -- review each list directly:")
        for model in model_names:
            lines.append(f"  -- {model} ({len(models_to_entries[model])} entries) --")
            for entry in models_to_entries[model]:
                lines.append(_format_entry(entry))
        return "\n".join(lines)
    model_a, model_b = model_names
    entries_a, entries_b = models_to_entries[model_a], models_to_entries[model_b]
    matched, only_a, only_b = diff_toc_entries(entries_a, entries_b)
    rate = len(matched) / max(len(entries_a), len(entries_b))
    lines.append(f"{model_a}: {len(entries_a)} entries, {model_b}: {len(entries_b)} entries -- {len(matched)} matched, rate={rate:.2f}")
    if only_a:
        lines.append(f"  Only in {model_a}:")
        for entry in only_a:
            lines.append(_format_entry(entry))
    if only_b:
        lines.append(f"  Only in {model_b}:")
        for entry in only_b:
            lines.append(_format_entry(entry))
    return "\n".join(lines)


def _list(cdir: Path, cache_directory: Path) -> int:
    needing = books_needing_arbitration(cdir, cache_directory)
    if not needing:
        print("No books currently need arbitration.")
        return 0
    titles = {manifest_key(book): book.get("title", "") for book in load_manifest_books(_CORPUS_NAME)}
    for key in needing:
        models_to_entries = _cached_models_for_book(cache_directory, key)
        print(format_book_report(key, titles.get(key, ""), cdir / f"{key}.pdf", models_to_entries))
        print()
    return 0


def reject_book(cdir: Path, key: str, reason: str, today=date.today) -> int:
    """Permanently records key as unrecoverable so future arbitration
    passes never resurface it. Errors (returns 1) rather than silently
    overwriting if key is already rejected."""
    path = _rejected_path(cdir)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rejected": []}
    if any(entry["key"] == key for entry in data["rejected"]):
        print(f"{key} is already marked rejected -- not overwriting.")
        return 1
    data["rejected"].append({"key": key, "reason": reason, "rejected_at": today().isoformat()})
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="List books needing arbitration (default)")
    reject_parser = subparsers.add_parser("reject", help="Permanently mark a book as unrecoverable")
    reject_parser.add_argument("key")
    reject_parser.add_argument("reason")
    args = parser.parse_args()

    cdir = corpus_dir(_CORPUS_NAME)
    if args.command == "reject":
        return reject_book(cdir, args.key, args.reason)
    return _list(cdir, llm_cache_dir(_CORPUS_NAME))


if __name__ == "__main__":
    raise SystemExit(main())
