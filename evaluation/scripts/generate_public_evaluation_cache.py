#!/usr/bin/env python3
"""Generate each corpus's public-cache/ -- a git-trackable corpus safe to
commit and distribute -- plus, per book, a resolved outline-strategy
candidate snapshot (<key>.outline.json -- titles/authors/page indices
only, no prose) so the outline strategy is also testable without the real
PDF -- see evaluation/README.md for the workflow and
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
for the outline-snapshot rationale.

A book's manifest "oa" flag decides how its page text is cached:
open-access books (oa: true) are cached VERBATIM -- the PDF itself is
already legally redistributable (that's what oa: true means), so the
extracted text, a strict subset of that same content, needs no redaction.
Every other book is redacted (real navigational/bibliographic text kept
verbatim, chapter prose replaced with random real words in the book's own
language) since its PDF cannot be redistributed.

Run by a maintainer who has the real books locally; not something a
contributor without PDFs needs to run.

    uv run python evaluation/scripts/generate_public_evaluation_cache.py [--book <manifest-key>] [--corpus <name>] [--no-verify] [--skip-redaction]

--skip-redaction caches a non-OA book's REAL page text verbatim, with no
redaction attempted at all -- for active-development situations where a
working cache is wanted now (so downstream tooling like
evaluation/refresh_llm_cache.py has something to read) but redacting a
freshly-added batch of books isn't worth blocking on yet. This writes real
copyrighted prose into what is normally the git-tracked, safe-to-publish
public-cache/ directory -- the caller is responsible for NOT committing
those specific files (see the printed WARNING for exactly which ones, and
`evaluation/CLAUDE.md`'s "redaction_overrides.json" section). Every such
entry is marked `"needs_redaction": true` so a later real run of this
script without the flag (which always re-redacts a book it touches) is
the way back to a publishable cache.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chapter_segmentation.evidence.outline_strategy import extract_outline_candidates
from chapter_segmentation.ocr import detect_language
from chapter_segmentation.segmentation import (
    analyze_attachment,
    extract_page_texts_for_analysis,
    pages_need_ocr,
)
from evaluation.harness import (
    analysis_pages_for,
    available_books,
    corpus_dir,
    list_corpora,
    outline_candidate_to_dict,
    public_cache_dir,
)
from evaluation.redaction.redact import redact_book_until_stable

CIPHER_VERSION = 3  # bumped: cache entries now also carry a "verified" flag
# (bumped previously: cache entries carry a "redacted" flag, and oa:true
# books are cached verbatim instead of redacted -- see module docstring)


def _verify(real_pages: list[str], redacted_pages: list[str]) -> list[str]:
    """Human-readable diff lines, empty when the two page sets produce
    identical detected chapter boundaries."""
    real_result = analyze_attachment(real_pages)
    redacted_result = analyze_attachment(redacted_pages)
    real_boundaries = {(c["pdf_start_index"], c["pdf_end_index"]) for c in real_result["chapters"]}
    redacted_boundaries = {(c["pdf_start_index"], c["pdf_end_index"]) for c in redacted_result["chapters"]}
    if real_boundaries == redacted_boundaries:
        return []
    return [
        f"  real boundaries:     {sorted(real_boundaries)}",
        f"  redacted boundaries: {sorted(redacted_boundaries)}",
    ]


def _load_redaction_overrides(corpus: str) -> dict[str, list[int]]:
    """Reads evaluation/corpus/<corpus>/redaction_overrides.json -- a
    committed, hand-maintained map of manifest key -> extra page indices to
    always keep verbatim during redaction. Exists for residual drift
    `redact_book_until_stable` cannot resolve on its own (a page that is
    neither the start nor end of any mismatched chapter range, so nothing
    in the automatic retry loop ever proposes forcing it -- see that
    function's docstring for the `9781841136400` case that motivated this).
    Diagnose the exact page(s) by hand (bisect the mismatched range) rather
    than reaching for this file first -- it is a last resort for the
    handful of books where the automatic loop provably cannot converge
    without forcing an unacceptably large fraction of the book verbatim."""
    path = corpus_dir(corpus) / "redaction_overrides.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", help="Only regenerate this manifest key (filename stem)")
    parser.add_argument("--corpus", help="Only regenerate this corpus (default: every corpus under evaluation/corpus/)")
    parser.add_argument("--no-verify", action="store_true", help="Skip the exact-boundary-match check")
    parser.add_argument(
        "--skip-redaction", action="store_true",
        help="Cache non-OA books' REAL text verbatim, unredacted -- see module docstring. "
        "Do NOT commit the resulting public-cache/ files.",
    )
    args = parser.parse_args()

    failures = 0
    unverified = 0
    unredacted_written: list[str] = []
    corpora = [args.corpus] if args.corpus else list_corpora()
    for corpus in corpora:
        cache_dir = public_cache_dir(corpus)
        cache_dir.mkdir(parents=True, exist_ok=True)
        redaction_overrides = _load_redaction_overrides(corpus)
        for pdf_path, _expected_path, book in available_books(corpus):
            manifest_key = Path(book["filename"]).stem
            if args.book and manifest_key != args.book:
                continue
            file_bytes = pdf_path.read_bytes()
            real_pages = analysis_pages_for(corpus, file_bytes)
            if real_pages is None:
                print(f"{corpus}/{manifest_key}: SKIPPED (needs OCR -- run scripts/ocr_evaluation_pdfs.py first)")
                continue
            try:
                raw_pages, _layout_used = extract_page_texts_for_analysis(file_bytes)
                source = "ocr" if pages_need_ocr(raw_pages) else "extracted"
                needs_redaction = False
                if book.get("oa", False):
                    # The PDF itself is already legally redistributable, so
                    # the extracted text needs no redaction -- this also
                    # sidesteps the whole class of redaction-induced parity
                    # drift documented in CLAUDE.md's "Known failure modes"
                    # (a redacted page coincidentally gaining or losing a
                    # fuzzy-match it didn't have on the real text).
                    cache_pages = real_pages
                    redacted = False
                    verified = True
                elif args.skip_redaction:
                    cache_pages = real_pages
                    redacted = False
                    verified = None
                    needs_redaction = True
                    unredacted_written.append(f"{corpus}/{manifest_key}.pages.json")
                    print(f"{corpus}/{manifest_key}: WARNING -- cached REAL text verbatim (--skip-redaction), "
                          f"\"needs_redaction\": true. DO NOT COMMIT this file.")
                else:
                    language = detect_language(book.get("language"), book.get("title", ""))
                    extra_forced = frozenset(redaction_overrides.get(manifest_key, []))
                    cache_pages, extra_preserved = redact_book_until_stable(
                        real_pages, detected_language=language, book_salt=manifest_key,
                        extra_forced=extra_forced,
                    )
                    redacted = True
                    verified = True
                    if extra_preserved:
                        print(f"{corpus}/{manifest_key}: self-corrected -- forced {len(extra_preserved)} extra page(s) "
                              f"fully verbatim to resolve a redaction-induced boundary drift: {sorted(extra_preserved)}")
                    if not args.no_verify:
                        # Defense in depth: redact_book_until_stable already verifies
                        # internally on every attempt, so this only fires if
                        # max_attempts was exhausted without full convergence. Rather
                        # than drop the book entirely, cache it anyway with
                        # "verified": false and a loud warning -- test_public_
                        # evaluation_cache_parity.py only *reports* precision/recall
                        # from whatever's in the cache (it's not gated), so an
                        # imperfect cache still gives contributors without the PDF a
                        # usable (if slightly noisy for this one book) signal, which
                        # beats having no cache -- and therefore no llm-cache either,
                        # since that depends on a public-cache entry existing --
                        # at all. The warning is the flag for a human to look closer
                        # (a smaller `redaction_overrides.json` entry, or accepting
                        # the noise) at their own pace, not a build-blocking failure.
                        diff = _verify(real_pages, cache_pages)
                        if diff:
                            verified = False
                            unverified += 1
                            print(f"{corpus}/{manifest_key}: WARNING -- redaction changed detected chapter boundaries "
                                  f"even after self-correction; caching anyway with \"verified\": false. Needs manual "
                                  f"review (see evaluation/CLAUDE.md's redaction_overrides.json section).")
                            print("\n".join(diff))
            except Exception as exc:
                # One book's failure must not strand the rest of the batch --
                # same catch-log-continue shape as scripts/ocr_evaluation_pdfs.py.
                print(f"{corpus}/{manifest_key}: FAILED ({exc}) -- skipping")
                failures += 1
                continue
            cache_path = cache_dir / f"{manifest_key}.pages.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "cipher_version": CIPHER_VERSION, "source": source, "redacted": redacted,
                        "verified": verified, "needs_redaction": needs_redaction, "pages": cache_pages,
                    },
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            verbatim_note = " (verbatim, needs_redaction)" if needs_redaction else (" (verbatim, oa)" if not redacted else "")
            print(f"{corpus}/{manifest_key}: OK, wrote {cache_path}{verbatim_note}")
            outline_candidates = extract_outline_candidates(file_bytes)
            outline_cache_path = cache_dir / f"{manifest_key}.outline.json"
            outline_cache_path.write_text(
                json.dumps(
                    {"candidates": [outline_candidate_to_dict(c) for c in outline_candidates]},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"{corpus}/{manifest_key}: wrote {len(outline_candidates)} outline candidate(s) to {outline_cache_path}")
    if unverified:
        print(f"{unverified} book(s) cached with \"verified\": false -- needs manual review, see WARNINGs above")
    if unredacted_written:
        print(f"\n{len(unredacted_written)} file(s) written UNREDACTED (--skip-redaction) -- DO NOT COMMIT these:")
        for path in unredacted_written:
            print(f"  evaluation/corpus/{path}")
    if failures:
        print(f"{failures} book(s) failed -- see above")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
