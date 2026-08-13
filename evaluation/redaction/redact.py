"""Turns a RegionMap (scripts/evaluation_redaction/region_classification.py)
into per-character preserve/redact decisions and applies word-level
redaction -- see docs/superpowers/specs/
2026-08-05-evaluation-corpus-redaction-design.md sections 3.1, 5."""

import hashlib
import re
from dataclasses import replace

from chapter_segmentation.segmentation import (
    _LISTING_PAGE_BODY_WINDOW,
    _locate_toc_entries,
    _normalize_header_line,
    _PAGE_NUMBER_TOKEN_RE,
    analyze_attachment,
    find_toc_candidates,
)
from evaluation.redaction.region_classification import RegionMap, _header_stripped_offset, classify_regions
from evaluation.redaction.wordlists import pick_word, build_word_pool

_TOKEN_RE = re.compile(r"(?P<word>\w+)|(?P<other>\W+)", re.UNICODE)

# Beyond _LISTING_PAGE_BODY_WINDOW characters past the header-stripped start
# of a page, no production heuristic reads content anymore (see
# chapter_segmentation._secondary_listing_pages -- the largest fixed window
# any heuristic uses; locate_chapter_start_candidates' 200-char heading
# window is smaller and already handled verbatim via heading_windows). Only
# the page's TOTAL length still matters past that point, for the trailing-
# blank-page trim in chapter_segmentation._chapters_from_located -- so word
# tokens past the cutoff get length-preserving filler instead of a real
# pool word: same effect on every measured heuristic, without generating
# realistic-looking prose nothing ever reads. "q" is not a roman-numeral
# letter (chapter_segmentation._PAGE_NUMBER_TOKEN_RE's [ivxlcdm]) or a
# digit, so a filler run can never be mistaken for a page number.
_FILLER_CHAR = "q"


def build_preserve_mask(text: str, page_index: int, regions: RegionMap) -> list[bool]:
    """True at every character index in `text` that must survive redaction
    unchanged: this page's chapter-heading window (if any) and every line
    on this page whose normalized form is a recognized running header."""
    if page_index in regions.full_pages:
        return [True] * len(text)
    mask = [False] * len(text)
    window = regions.heading_windows.get(page_index)
    if window is not None:
        start, end = window
        for i in range(max(start, 0), min(end, len(text))):
            mask[i] = True
    pos = 0
    for line in text.splitlines(keepends=True):
        if _normalize_header_line(line.rstrip("\r\n")) in regions.header_lines:
            for i in range(pos, pos + len(line)):
                mask[i] = True
        pos += len(line)
    return mask


def _match_case(word: str, original: str) -> str:
    if original.isupper() and len(original) > 1:
        return word.upper()
    if original[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def redact_page(text: str, page_index: int, regions: RegionMap, pool: dict[int, list[str]], book_salt: str) -> str:
    """Replaces every word token outside `regions`' preserved spans with a
    deterministic, same-length (or nearest-length) real word from `pool`,
    up to `_LISTING_PAGE_BODY_WINDOW` characters past the header-stripped
    start of the page; word tokens beyond that get length-preserving
    filler instead (see `_FILLER_CHAR`). Digits, roman-numeral-shaped
    tokens, and preserved spans pass through unchanged everywhere."""
    mask = build_preserve_mask(text, page_index, regions)
    real_content_cutoff = _header_stripped_offset(text, regions.header_lines) + _LISTING_PAGE_BODY_WINDOW
    out: list[str] = []
    pos = 0
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        start = pos
        pos += len(token)
        if match.group("word") is None:
            out.append(token)  # whitespace/punctuation run
            continue
        if any(mask[start:pos]):
            # Preserve the WHOLE token even if it only partially overlaps a
            # preserved span (e.g. a heading window ending mid-word):
            # locate_chapter_start_candidates reads a raw character slice,
            # not a word-aligned one, so redacting the out-of-window tail
            # of a straddling word corrupts characters that ARE inside the
            # window too. A few extra characters of spillover past the
            # nominal boundary is harmless (same "more inclusive than
            # strictly necessary" tradeoff as the heading-window pooling in
            # region_classification.py) -- found empirically: a real book's
            # chapter-title word straddled its window right at the 200-char
            # boundary, and redacting the whole word turned a borderline
            # (80.46) fuzzy match into a miss, changing a detected chapter
            # boundary after redaction.
            out.append(token)
            continue
        if token.isdigit() or _PAGE_NUMBER_TOKEN_RE.match(token):
            out.append(token)  # page-number-shaped token, never prose
            continue
        if start >= real_content_cutoff:
            out.append(_FILLER_CHAR * len(token))  # past every heuristic's read window
            continue
        seed = int.from_bytes(
            hashlib.sha256(f"{book_salt}:{page_index}:{start}:{token.casefold()}".encode("utf-8")).digest()[:8],
            "big",
        )
        out.append(_match_case(pick_word(pool, len(token), seed), token))
    return "".join(out)


def redact_book(pages: list[str], detected_language: str, book_salt: str) -> list[str]:
    """Full per-book redaction pipeline: classify regions once, build the
    language's word pool once, then redact every page."""
    regions = classify_regions(pages)
    pool = build_word_pool(detected_language)
    return [redact_page(text, index, regions, pool, book_salt) for index, text in enumerate(pages)]


def _entry_key(entry) -> tuple:
    """Identifies a TocEntry across two separate _locate_toc_entries calls
    on the SAME original entry list, even though _locate_toc_entries
    replaces `.title` with whichever variant won -- `.title_variants` is
    never mutated, so combined with the entry's fixed source position it's
    a stable key regardless of which reading ends up winning."""
    return (entry.source_page_index, entry.printed_page_number, entry.title_variants)


def _drifted_pages(real_map: dict, redacted_map: dict) -> set[int]:
    """Every page index involved in a TOC entry whose located page (keyed
    identically in both maps) differs between the two -- the page that
    lost its match, the page that spuriously gained one, or both when an
    entry simply relocated. Powers redact_book_until_stable's retry loop:
    forcing these pages fully verbatim and re-redacting eliminates
    whatever redacted-text coincidence caused the drift."""
    drifted: set[int] = set()
    for key in set(real_map) | set(redacted_map):
        real_index = real_map.get(key)
        redacted_index = redacted_map.get(key)
        if real_index != redacted_index:
            if real_index is not None:
                drifted.add(real_index)
            if redacted_index is not None:
                drifted.add(redacted_index)
    return drifted


def _boundary_drifted_pages(real_pages: list[str], redacted_pages: list[str]) -> set[int]:
    """Every page index that is a start or end of a chapter range
    `analyze_attachment` finds on one side but not the other -- catches
    drift `_drifted_pages`' TOC-entry-location comparison misses (e.g. a
    heading pattern with no TOC entry driving it at all, or a merge/split
    that changes which ranges exist without any single TOC entry's located
    page moving). This is the same comparison `generate_public_evaluation_cache.py`'s
    `_verify` makes, so folding it into the retry loop here means a book
    that would otherwise reach `_verify` and fail gets a chance to
    self-correct first, the same as an entry-location drift already did."""
    real_boundaries = {(c["pdf_start_index"], c["pdf_end_index"]) for c in analyze_attachment(real_pages)["chapters"]}
    redacted_boundaries = {
        (c["pdf_start_index"], c["pdf_end_index"]) for c in analyze_attachment(redacted_pages)["chapters"]
    }
    drifted: set[int] = set()
    for start, end in real_boundaries ^ redacted_boundaries:
        drifted.add(start)
        drifted.add(end)
    return drifted


def redact_book_until_stable(
    pages: list[str],
    detected_language: str,
    book_salt: str,
    max_attempts: int = 15,
    extra_forced: frozenset[int] = frozenset(),
) -> tuple[list[str], frozenset[int]]:
    """redact_book, but self-correcting: after redacting, checks whether
    redaction changed any TOC entry's located page
    (chapter_segmentation._locate_toc_entries -- the same lookup
    classify_regions's heading-window computation is built on) OR changed
    any chapter boundary `analyze_attachment` detects (`_boundary_drifted_pages`
    above -- a broader, more expensive check added after the entry-location
    check alone was found to still let real drift through to
    `generate_public_evaluation_cache.py`'s `_verify` step, 2026-08-13: a
    batch of newly-promoted books included one, `9783428038275`, whose
    redacted text spuriously gained two chapter boundaries `analyze_attachment`
    never found on the real text at all, with no TOC entry involved on
    either side for the entry-location check to catch). When either check
    finds drift, every page involved is forced fully verbatim and redaction
    retries, looping until both are stable or `max_attempts` is exhausted.
    The entry-location check directly neutralizes the "incidental
    fuzzy-match false positive" risk documented in docs/superpowers/specs/
    2026-08-05-evaluation-corpus-redaction-design.md section 10.3 -- found
    empirically: a book with several letter-spaced ("Sperrdruck") chapter
    titles had a short, generic title variant that was correctly ambiguous
    (two competing real-text candidates) but became spuriously uncontested
    after redaction, because the redacted text on one of its two
    competitors' pages happened to drop just enough to lose its
    fuzzy-match status -- a book-specific typographic risk that retrying
    with a different book_salt alone does not fix (verified empirically:
    the same drift reproduced across 6 different salts), since it's driven
    by the text's structural shape (word lengths, spacing), not by which
    specific real word fills each slot.

    `extra_forced` seeds pages to keep verbatim from the very first attempt,
    on top of whatever this loop finds on its own -- for a book whose
    residual drift comes from a page that is neither the start nor end of
    any mismatched chapter range (so `_boundary_drifted_pages` never
    proposes it, and the same false split reproduces every attempt no
    matter how high `max_attempts` goes -- found manually for
    `9781841136400`, 2026-08-13: a real, single chapter spanning pages
    451-456 was spuriously split at pages 452-455 after redaction, and the
    loop kept re-forcing only the mismatch's 451/452/455/456 endpoints
    forever without ever touching 453/454, the pages actually causing the
    false split). See `evaluation/corpus/<corpus>/redaction_overrides.json`.

    Returns (redacted_pages, extra_preserved_page_indices) -- the latter
    empty when the first attempt was already stable and `extra_forced` was
    empty. A book whose drift is scattered across a large fraction of its
    pages may still exhaust `max_attempts` without stabilizing -- forcing
    that much of a non-open-access book fully verbatim would defeat
    redaction's purpose (leaking most of its actual prose into a committed
    cache file), so this deliberately does not keep escalating forever; the
    caller's `_verify` step is the backstop that refuses to cache an
    unstable result."""
    regions = classify_regions(pages)
    regions = replace(regions, full_pages=regions.full_pages | extra_forced)
    pool = build_word_pool(detected_language)
    toc_entries = find_toc_candidates(pages)
    toc_indices = {e.source_page_index for e in toc_entries}
    located_real, _unlocated_real, _non_content_real = _locate_toc_entries(
        pages, toc_entries, exclude_indices=toc_indices,
    )
    real_map = {_entry_key(entry): match.index for entry, match in located_real}

    extra_preserved: frozenset[int] = extra_forced
    redacted = pages
    for _attempt in range(max_attempts):
        redacted = [redact_page(text, index, regions, pool, book_salt) for index, text in enumerate(pages)]
        located_redacted, _unlocated_redacted, _non_content_redacted = _locate_toc_entries(
            redacted, toc_entries, exclude_indices=toc_indices,
        )
        redacted_map = {_entry_key(entry): match.index for entry, match in located_redacted}
        drifted = _drifted_pages(real_map, redacted_map) | _boundary_drifted_pages(pages, redacted)
        if not drifted:
            break
        extra_preserved = extra_preserved | drifted
        regions = replace(regions, full_pages=regions.full_pages | drifted)
    return redacted, extra_preserved
