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
