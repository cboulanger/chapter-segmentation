#!/usr/bin/env python3
"""Pilot: leave-one-book-out evaluation of a layout-geometry TOC/
chapter-first-page classifier. See
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md.

Manual run, not part of `uv run pytest` -- same convention as
evaluation/scripts/fetch_evaluation_pdfs.py.

Usage:
    uv run python evaluation/scripts/evaluate_layout_toc_classifier.py
    uv run python evaluation/scripts/evaluate_layout_toc_classifier.py --pdfalto-bin /path/to/pdfalto
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_RECALL_TARGET = 0.90  # threshold picked per fold to hit this recall on training pages


def select_threshold(
    train_probs: list[float], train_labels: list[bool], recall_target: float
) -> float:
    """Picks the highest probability threshold that still achieves at least
    recall_target on the training positives -- a lower threshold always
    yields recall >= a higher one, so the highest satisfying threshold is
    the most precise choice that still clears the bar. Returns 1.0 (accept
    nothing) if there are no positives to calibrate against."""
    if not 0.0 < recall_target <= 1.0:
        raise ValueError(f"recall_target must be in (0.0, 1.0], got {recall_target!r}")
    if len(train_probs) != len(train_labels):
        raise ValueError(
            f"train_probs and train_labels must be the same length, "
            f"got {len(train_probs)} and {len(train_labels)}"
        )
    positive_probs = sorted(
        (p for p, is_positive in zip(train_probs, train_labels) if is_positive), reverse=True
    )
    if not positive_probs:
        return 1.0
    # ceil, not round: "at least recall_target" must never be undershot,
    # e.g. 6 positives at 0.90 -> ceil(5.4) = 6, not round(5.4) = 5.
    n_needed = max(1, math.ceil(recall_target * len(positive_probs)))
    return positive_probs[n_needed - 1]
