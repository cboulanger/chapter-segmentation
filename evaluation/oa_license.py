"""Shared Crossref/Unpaywall open-access license lookup.

Single home for the license-resolution logic evaluation/scripts/
fetch_crossref_gt_corpus.py, discover_crossref_candidates.py, and
promote_pending_book.py all need. Previously duplicated as private
functions inside fetch_crossref_gt_corpus.py, with discover_crossref_candidates.py
reaching across the module boundary to import its private names directly
-- see docs/superpowers/specs/2026-08-12-ground-truth-pipeline-hardening-design.md.
Lives under evaluation/ (not evaluation/scripts/) for the same reason
harness.py does: evaluation/scripts/*.py must not depend on each other's
internals any more than they depend on the test tree.
"""

import time
from collections import Counter
from typing import Optional

import httpx

from chapter_segmentation.evidence.crossref_strategy import (
    _CROSSREF_BASE_URL,
    _DEFAULT_RETRY_DELAY_SECONDS,
    _MAX_RETRIES,
)

DEFAULT_CONTACT_EMAIL = "boulanger@lhlt.mpg.de"
UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"

# Unpaywall reports a short SPDX-ish code (e.g. "cc-by-nc"), not a URL --
# mapped to the CC 4.0 URL, since every book this mapping has been
# checked against so far is a 2020s publication and 4.0 is the only
# version any of its publishers use for new titles (found empirically:
# every Crossref-registered license in this corpus that DOES carry an
# explicit version is 4.0).
_UNPAYWALL_LICENSE_URLS = {
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc-by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "cc-by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "pd": "https://creativecommons.org/publicdomain/mark/1.0/",
}


def crossref_book_chapter_items(isbn: str, client: httpx.Client, contact_email: Optional[str]) -> list[dict]:
    """GET .../works?filter=isbn:{isbn}, returning raw type=="book-chapter"
    items. Any network/HTTP/JSON failure is printed and treated as an
    empty result -- never raises, so one bad book never aborts a batch."""
    params: dict[str, str | int] = {
        "filter": f"isbn:{isbn}",
        "select": "DOI,title,subtitle,author,page,type,container-title,published,ISBN,license",
        "rows": 100,
    }
    if contact_email:
        params["mailto"] = contact_email

    response = None
    for _attempt in range(_MAX_RETRIES):
        try:
            response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
        except httpx.HTTPError as exc:
            print(f"  [warn] network error fetching Crossref metadata for {isbn}: {exc}")
            return []
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
        time.sleep(delay)
    else:
        print(f"  [warn] exhausted retries (429) fetching Crossref metadata for {isbn}")
        return []

    try:
        response.raise_for_status()
        items = response.json()["message"]["items"]
    except Exception as exc:
        print(f"  [warn] bad Crossref response for {isbn}: {exc}")
        return []

    return [item for item in items if item.get("type") == "book-chapter"]


def item_license_url(item: dict) -> Optional[str]:
    """The registered OA license URL for one Crossref item, preferring the
    version-of-record entry (content-version=="vor", delay-in-days==0) --
    the license that actually applies to the publicly available PDF, not
    an embargoed accepted-manuscript variant Crossref may register
    alongside it."""
    licenses = item.get("license") or []
    for entry in licenses:
        if entry.get("content-version") == "vor" and entry.get("delay-in-days", 0) == 0:
            return entry.get("URL")
    return licenses[0].get("URL") if licenses else None


def book_license_url(raw_items: list[dict]) -> Optional[str]:
    """The book's OA license URL, by majority vote across its chapters'
    individually-registered licenses -- in practice unanimous, since
    Crossref licenses are registered once per book and inherited by every
    chapter, but a vote is cheap insurance against one mis-registered
    chapter. None if no chapter has a registered license at all."""
    urls = [url for item in raw_items if (url := item_license_url(item)) is not None]
    if not urls:
        return None
    return Counter(urls).most_common(1)[0][0]


def unpaywall_license_url(doi: Optional[str], client: httpx.Client, contact_email: Optional[str]) -> Optional[str]:
    """Fallback license lookup via Unpaywall, tried only when Crossref has
    no license registered for a book. Crossref's license field is
    self-reported by the publisher at metadata-deposit time and often left
    blank; Unpaywall aggregates OA status from institutional repositories
    and publisher landing pages instead, so it is NOT the same data.
    Returns None if there's no DOI, no Unpaywall OA record, no license, or
    the request fails -- never raises, so one bad lookup never aborts a
    batch."""
    if not doi:
        return None
    email = contact_email or DEFAULT_CONTACT_EMAIL
    try:
        response = client.get(f"{UNPAYWALL_BASE_URL}/{doi}", params={"email": email}, timeout=10.0)
        response.raise_for_status()
        location = response.json().get("best_oa_location")
    except Exception as exc:
        print(f"  [warn] Unpaywall lookup failed for {doi}: {exc}")
        return None
    code = location.get("license") if location else None
    return _UNPAYWALL_LICENSE_URLS.get(code)


def resolve_license(
    isbn: str, doi: Optional[str], client: httpx.Client, contact_email: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """The one call most callers need: fetches isbn's Crossref
    book-chapter items, tries book_license_url on them, falls back to
    unpaywall_license_url(doi, ...) if that's None. Returns (license_url,
    license_source) with license_source in {"crossref", "unpaywall",
    None}. Never raises."""
    raw_items = crossref_book_chapter_items(isbn, client, contact_email)
    license_url = book_license_url(raw_items)
    if license_url is not None:
        return license_url, "crossref"
    license_url = unpaywall_license_url(doi, client, contact_email)
    return license_url, ("unpaywall" if license_url else None)
