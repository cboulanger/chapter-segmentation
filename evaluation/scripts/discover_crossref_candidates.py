#!/usr/bin/env python3
"""Discovers new open-access book candidates (monographs and edited
volumes) for evaluation/crossref_gt/manifest.json, seeded from Crossref
publishers already represented there.

Resolves each candidate's direct PDF URL from three independent sources,
tried in order (Crossref's own registered link, then Unpaywall, then
OpenAlex), and prioritizes appending candidates in currently-underrepresented
languages first so automated growth doesn't quietly re-converge on
English-language volumes just because they're easiest to find.

This script only appends to manifest.json -- run
fetch_crossref_gt_corpus.py afterwards to actually download the new
entries' PDFs and Crossref chapter metadata, then
build_crossref_gt_ground_truth.py to reconcile and (if confirmed + novel)
migrate them into evaluation/corpus/pending/.

Usage:
    uv run python evaluation/scripts/discover_crossref_candidates.py
    uv run python evaluation/scripts/discover_crossref_candidates.py --dry-run
    uv run python evaluation/scripts/discover_crossref_candidates.py --max-per-language 3
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)
from evaluation.scripts.fetch_crossref_gt_corpus import (
    _DEFAULT_CONTACT_EMAIL,
    _UNPAYWALL_BASE_URL,
    _item_license_url,
    _unpaywall_license_url,
)

_CROSSREF_DIR = Path(__file__).resolve().parent.parent / "crossref_gt"
_OPEN_ACCESS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "open-access"

_OPENALEX_BASE_URL = "https://api.openalex.org/works"

_WORK_TYPES = ["monograph", "edited-book"]
_DEFAULT_MAX_PER_LANGUAGE = 5
_DEFAULT_MAX_RESULTS_PER_QUERY = 300

# Real Crossref member IDs, looked up live against
# https://api.crossref.org/members (by publisher name) and
# https://api.crossref.org/prefixes/<prefix> (by cross-checking against
# each publisher's DOI prefix already present in manifest.json) --
# not guessed. "Presses universitaires de Rennes" and "Africae" turned out
# to be OpenEdition imprints sharing one Crossref member, not separate
# members -- consolidated into the single OpenEdition entry below.
_SEED_PUBLISHERS = [
    {"member_id": "4923", "publisher": "Open Book Publishers", "default_language": "en"},
    {"member_id": "5471", "publisher": "transcript Verlag", "default_language": "de"},
    {"member_id": "5433", "publisher": "UCL Press", "default_language": "en"},
    {"member_id": "5869", "publisher": "Athabasca University Press", "default_language": "en"},
    {"member_id": "297", "publisher": "Springer Science and Business Media LLC", "default_language": "en"},
    {"member_id": "2399", "publisher": "OpenEdition", "default_language": "fr"},
]


def _item_isbn(item: dict) -> Optional[str]:
    isbns = item.get("ISBN") or []
    return isbns[0] if isbns else None


def _item_title(item: dict) -> Optional[str]:
    titles = item.get("title") or []
    return titles[0] if titles else None


def _crossref_link_pdf_url(item: dict) -> Optional[str]:
    """The first link entry that looks like a real full-text PDF resource.
    content-type "application/pdf" is the correct MIME type; "unspecified"
    is included too because several publishers (confirmed against a live
    Open Book Publishers record) register their actual PDF link that way
    instead."""
    for link in item.get("link") or []:
        if link.get("content-type") in ("application/pdf", "unspecified"):
            return link.get("URL")
    return None


def _unpaywall_pdf_url(doi: Optional[str], client: httpx.Client, contact_email: Optional[str]) -> Optional[str]:
    """Fallback PDF URL lookup via Unpaywall, tried only when Crossref's
    own link array has none. Never raises -- one bad lookup must not abort
    the batch."""
    if not doi:
        return None
    email = contact_email or _DEFAULT_CONTACT_EMAIL
    try:
        response = client.get(f"{_UNPAYWALL_BASE_URL}/{doi}", params={"email": email}, timeout=10.0)
        response.raise_for_status()
        location = response.json().get("best_oa_location")
    except Exception as exc:
        print(f"  [warn] Unpaywall URL lookup failed for {doi}: {exc}")
        return None
    return location.get("url_for_pdf") if location else None


def _openalex_pdf_url(doi: Optional[str], client: httpx.Client) -> Optional[str]:
    """Second fallback PDF URL lookup via OpenAlex, tried only when
    neither Crossref nor Unpaywall has one. Never raises."""
    if not doi:
        return None
    try:
        response = client.get(f"{_OPENALEX_BASE_URL}/doi:{doi}", timeout=10.0)
        response.raise_for_status()
        location = response.json().get("best_oa_location")
    except Exception as exc:
        print(f"  [warn] OpenAlex URL lookup failed for {doi}: {exc}")
        return None
    return location.get("pdf_url") if location else None


def resolve_download_url(
    item: dict, doi: Optional[str], client: httpx.Client, contact_email: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Tries, in order, Crossref's own link array, then Unpaywall, then
    OpenAlex. Returns (url, source) -- source is "crossref", "unpaywall",
    "openalex", or None if none of the three has one."""
    url = _crossref_link_pdf_url(item)
    if url:
        return url, "crossref"
    url = _unpaywall_pdf_url(doi, client, contact_email)
    if url:
        return url, "unpaywall"
    url = _openalex_pdf_url(doi, client)
    if url:
        return url, "openalex"
    return None, None


def _is_new_candidate(
    isbn: Optional[str], doi: Optional[str], existing_isbns: set[str], existing_dois: set[str]
) -> bool:
    if not isbn or isbn in existing_isbns:
        return False
    if doi and doi in existing_dois:
        return False
    return True


def _language_counts(*manifest_paths: Path) -> Counter:
    counts: Counter = Counter()
    for path in manifest_paths:
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for book in manifest["books"]:
            language = book.get("language")
            if language:
                counts[language] += 1
    return counts


def _language_priority(counts: Counter, languages: set[str]) -> list[str]:
    """Ranks languages by ascending current count (most underrepresented
    first); a language with no entries yet ranks ahead of any with at
    least one."""
    return sorted(languages, key=lambda language: counts.get(language, 0))


def _select_candidates(
    candidates_by_language: dict[str, list[dict]], priority: list[str], max_per_language: int
) -> list[dict]:
    selected = []
    for language in priority:
        selected.extend(candidates_by_language.get(language, [])[:max_per_language])
    return selected


def _crossref_publisher_works(
    member_id: str,
    work_type: str,
    client: httpx.Client,
    contact_email: Optional[str],
    rows: int = 100,
    max_results: int = _DEFAULT_MAX_RESULTS_PER_QUERY,
) -> list[dict]:
    """Paginates https://api.crossref.org/works?filter=member:<id>,type:<work_type>
    via offset, stopping once a page returns fewer than `rows` items or
    `max_results` total items have been collected. Never raises -- a
    network/HTTP/JSON failure for one page is logged and treated as "no
    more pages" rather than aborting the whole discovery run. "language"
    is deliberately not in the select list -- Crossref rejects it as an
    invalid select field for the /works route (confirmed live), and in
    practice book-type records essentially never carry it anyway."""
    items: list[dict] = []
    offset = 0
    while len(items) < max_results:
        params: dict[str, str | int] = {
            "filter": f"member:{member_id},type:{work_type}",
            "select": "DOI,ISBN,title,link,publisher,type,license",
            "rows": rows,
            "offset": offset,
        }
        if contact_email:
            params["mailto"] = contact_email

        response = None
        for _attempt in range(_MAX_RETRIES):
            try:
                response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
            except httpx.HTTPError as exc:
                print(f"  [warn] network error fetching member {member_id}/{work_type}: {exc}")
                return items
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
            time.sleep(delay)
        else:
            print(f"  [warn] exhausted retries (429) fetching member {member_id}/{work_type}")
            return items

        try:
            response.raise_for_status()
            page_items = response.json()["message"]["items"]
        except Exception as exc:
            print(f"  [warn] bad Crossref response for member {member_id}/{work_type}: {exc}")
            return items

        items.extend(page_items)
        if len(page_items) < rows:
            break
        offset += rows
    return items[:max_results]


def _build_candidate(
    item: dict, seed: dict, download_url: str, license_url: Optional[str], license_source: Optional[str]
) -> dict:
    """Assembles one manifest-shaped candidate entry from a Crossref work
    item, its seed publisher, and its already-resolved download URL/license."""
    return {
        "isbn": _item_isbn(item),
        "title": _item_title(item),
        "doi": item.get("DOI"),
        "domain": None,
        "language": item.get("language") or seed["default_language"],
        "publisher": seed["publisher"],
        "download_url": download_url,
        "license": license_url,
        "license_source": license_source,
        "discovery_source": "auto",
    }


def discover(max_per_language: int, dry_run: bool, contact_email: Optional[str]) -> int:
    manifest_path = _CROSSREF_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_isbns = {book["isbn"] for book in manifest["books"]}
    existing_dois = {book["doi"] for book in manifest["books"] if book.get("doi")}

    lang_counts = _language_counts(manifest_path, _OPEN_ACCESS_DIR / "manifest.json")

    candidates_by_language: dict[str, list[dict]] = {}
    with httpx.Client(follow_redirects=True) as client:
        for seed in _SEED_PUBLISHERS:
            for work_type in _WORK_TYPES:
                items = _crossref_publisher_works(seed["member_id"], work_type, client, contact_email)
                for item in items:
                    isbn = _item_isbn(item)
                    doi = item.get("DOI")
                    if not _is_new_candidate(isbn, doi, existing_isbns, existing_dois):
                        continue
                    download_url, url_source = resolve_download_url(item, doi, client, contact_email)
                    if not download_url:
                        print(f"  [skip] {isbn}: no download URL found (Crossref/Unpaywall/OpenAlex)")
                        continue
                    license_url = _item_license_url(item)
                    license_source = "crossref" if license_url else None
                    if license_url is None:
                        license_url = _unpaywall_license_url(doi, client, contact_email)
                        license_source = "unpaywall" if license_url else None
                    candidate = _build_candidate(item, seed, download_url, license_url, license_source)
                    candidates_by_language.setdefault(candidate["language"], []).append(candidate)
                    existing_isbns.add(isbn)
                    if doi:
                        existing_dois.add(doi)

    priority = _language_priority(lang_counts, set(candidates_by_language))
    added = _select_candidates(candidates_by_language, priority, max_per_language)

    if added and not dry_run:
        manifest["books"].extend(added)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'Would add' if dry_run else 'Added'} {len(added)} new candidate(s):")
    for candidate in added:
        print(f"  - {candidate['isbn']} ({candidate['language']}): {candidate['title']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would be added without writing manifest.json")
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=_DEFAULT_MAX_PER_LANGUAGE,
        help=(
            "Cap on how many new candidates of one language get appended per run, "
            f"so one prolific publisher's catalog can't monopolize a run. Default: {_DEFAULT_MAX_PER_LANGUAGE}."
        ),
    )
    parser.add_argument("--contact-email", default=_DEFAULT_CONTACT_EMAIL, help="Crossref/Unpaywall polite-pool contact email")
    args = parser.parse_args()
    return discover(args.max_per_language, args.dry_run, args.contact_email)


if __name__ == "__main__":
    raise SystemExit(main())
