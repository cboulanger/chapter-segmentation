"""Per-page ground-truth labels (toc / chapter_first / other) derived from
an .expected.json-shaped dict. Used by
evaluation/scripts/layout_corpus.py's build_feature_table (in turn used by
build_crossref_gt_ground_truth.py's novelty check)."""

LABEL_TOC = "toc"
LABEL_CHAPTER_FIRST = "chapter_first"
LABEL_OTHER = "other"


def page_labels(expected: dict, total_pages: int) -> list[str] | None:
    """Returns a total_pages-length list of per-page labels, or None if this
    book has no usable "toc" field yet -- the key being entirely absent
    means "not yet retrofitted / flagged for manual review", which is
    different from an explicit "toc": null ("confirmed, no TOC page
    exists"), a book that IS usable and simply contributes no
    toc-labeled pages."""
    if "toc" not in expected:
        return None

    labels = [LABEL_OTHER] * total_pages
    toc = expected["toc"]
    if toc is not None:
        for index in range(toc["toc_start_index"], toc["toc_end_index"] + 1):
            labels[index] = LABEL_TOC
    for chapter in expected["chapters"]:
        labels[chapter["pdf_start_index"]] = LABEL_CHAPTER_FIRST
    return labels
