#!/usr/bin/env python3
"""One-time reconciliation: migrates evaluation/corpus/dnb-toc-only/manifest.json
from its original schema (a "lobid_record" field embedding the full
lobid-resources record verbatim) to the corrected one (a "lobid_url"
field plus a separate <key>.lobid.json side file) -- see
docs/superpowers/plans/2026-08-15-dnb-toc-corpus-corrections.md Task 2.

While rewriting each entry, also drops any book whose lobid_record isn't
actually EditedVolume-typed (see Task 1 of that plan -- the original
_record_matches filter was too broad and let single-author/thesis/
textbook records in). A dropped book's PDF and any already-written
.lobid.json are deleted; its manifest entry is removed entirely.

Usage:
    uv run python evaluation/scripts/migrate_dnb_toc_lobid_storage.py --dry-run
    uv run python evaluation/scripts/migrate_dnb_toc_lobid_storage.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir
from evaluation.scripts.fetch_dnb_toc_corpus import _record_api_url, _record_matches

_CORPUS_NAME = "dnb-toc-only"


def migrate(cdir: Path, dry_run: bool) -> None:
    manifest_path = cdir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    kept = []
    purged = []
    for book in data["books"]:
        key = Path(book["filename"]).stem
        record = book.get("lobid_record")
        if record is None:
            # Already migrated (no lobid_record field) -- leave as-is.
            kept.append(book)
            continue

        if not _record_matches(record):
            purged.append((key, book.get("title", "")))
            if not dry_run:
                (cdir / book["filename"]).unlink(missing_ok=True)
                (cdir / f"{key}.lobid.json").unlink(missing_ok=True)
            continue

        new_book = {k: v for k, v in book.items() if k != "lobid_record"}
        new_book["lobid_url"] = _record_api_url(record)
        kept.append(new_book)
        if not dry_run:
            (cdir / f"{key}.lobid.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            )

    print(f"{'[DRY RUN] ' if dry_run else ''}Kept {len(kept)} book(s), purged {len(purged)}:")
    for key, title in purged:
        print(f"  - {key}: {title}")

    if not dry_run:
        data["books"] = kept
        # Write atomically (temp file + rename) so a process killed mid-write
        # (SIGKILL, OOM, power loss) can never leave manifest.json truncated
        # or corrupt -- by this point purged books' PDFs are already deleted
        # from disk, so a corrupted manifest here would be unrecoverable.
        # Same pattern as _upsert_cache in evaluation/refresh_llm_cache.py
        # (see commit 8a59d90).
        tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        os.replace(tmp_path, manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing or deleting anything")
    args = parser.parse_args()
    migrate(corpus_dir(_CORPUS_NAME), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
