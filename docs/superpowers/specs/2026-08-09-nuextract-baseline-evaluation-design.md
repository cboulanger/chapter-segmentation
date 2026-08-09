# NuExtract-1.5-tiny zero-shot TOC-extraction baseline

Status: approved for planning
Date: 2026-08-09

## Problem

The pipeline's only fallback for irregular TOC layouts today is
`llm_extract_toc_entries` (segmentation.py:586), which calls whatever
`LLMClient` (llm.py:13) the caller supplies -- in practice a cloud provider,
since no local `LLMClient` implementation exists in this repo or in its
current consumers. That has two costs: real money/latency per call, and a
hard dependency on network access and an API key, which blocks fully local
or offline use of the package (e.g. inside a Zotero plugin with no LLM
subscription configured).

[NuExtract](https://huggingface.co/numind/NuExtract-1.5-tiny) is a small
(0.5B, MIT-licensed, fine-tune of Qwen2.5-0.5B) model purpose-built for
exactly this shape of task: given a JSON template and source text, it fills
the template with values copied verbatim from the text. NuMind's own model
card states it is "intended to be fine-tuned on a specific task (>= 30
examples)" -- close to this project's own ~50-book ground-truth corpus size
-- but it also claims usable zero-shot performance out of the box. Before
committing to any fine-tuning work, or to building the general pluggable
strategy architecture (see the companion spec,
`2026-08-09-pluggable-toc-strategy-cascade-design.md`), we need a real
number: how well does NuExtract-1.5-tiny extract TOC entries from this
project's actual ground-truth books, zero-shot, with no training at all?

Goal: produce a precision/recall baseline, comparable in spirit to the
existing per-strategy numbers in `evaluation/RESULTS.md`, that is the go/no-go
signal for further NuExtract investment. This is a standalone measurement
spike -- it does not touch `segmentation.py`, `cli.py`, or any production
strategy code.

## Scope

**Script**: `evaluation/scripts/evaluate_nuextract_baseline.py`, run manually
(not part of `uv run pytest`, same convention as `fetch_evaluation_pdfs.py`).

**Page selection**: reuse `_llm_scan_indices` (segmentation.py:506)
unchanged, via a direct import, so the exact same page range the existing
cloud-LLM fallback sees is what NuExtract sees -- an apples-to-apples input,
even though the *scoring* below is not apples-to-apples with the full
pipeline (see "Decision criteria").

**Template**: a NuExtract JSON template mirroring `TocEntry`'s fields:

```json
{
  "chapters": [
    {"title": "", "authors": [""], "printed_page_number": ""}
  ]
}
```

Formatted into NuExtract's documented input convention (template block +
text block, per its model card) -- not the free-form
`_LLM_TOC_EXTRACTION_PROMPT` (segmentation.py:488) used for the cloud LLM,
since NuExtract does not follow chat-style instructions the same way.

**Serving**: a local Ollama server running NuExtract-1.5-tiny. If no
Ollama-published tag exists for this model, convert the HF checkpoint to
GGUF via `llama.cpp`'s conversion script once and `ollama create` a local
tag from it -- a one-time setup step documented in the script's docstring
and in `evaluation/README.md`, not automated (no network fetch of arbitrary
model conversions belongs in a test/eval script).

**Scoring**: for each book in `open-access/` and `copyrighted-scans/` with
ground truth, run NuExtract over its scan-index pages, parse the returned
JSON into a list of `(title, printed_page_number)` pairs, and match against
`expected.json`'s chapters using the same fuzzy title-match convention
`fusion.py`'s `_align` already uses (`rapidfuzz.fuzz.token_sort_ratio`,
threshold 70) plus an exact `printed_page_number` (parsed from
`citation_pages`'s first number) match. Report precision/recall per corpus
and in aggregate, same table shape as `RESULTS.md`.

This is deliberately a narrower metric than full chapter-boundary accuracy
-- it scores only the TOC-*listing* extraction step NuExtract would
replace, not page-localization, since this spike does not run
`_locate_toc_entries`/`_chapters_from_located` at all.

## Non-goals

- No fine-tuning.
- No production wiring into `segmentation.py` or a `TocExtractionStrategy`
  implementation -- that is explicitly the companion spec's job, once this
  spike's numbers justify it.
- No multimodal/image-input variant (NuExtract-2.0) -- text-only
  NuExtract-1.5-tiny only, per the approved scope.
- No pytest/CI integration -- this is a manual, one-off measurement, not a
  regression-guarded suite.

## Decision criteria

A "promising" result is one where NuExtract's entry-level precision/recall
is in the neighborhood of the existing heuristic pipeline's aggregate
(0.91/0.91 per `RESULTS.md`) on `open-access/`, or meaningfully better than
0 on the `copyrighted-scans/` cases the heuristic and outline/crossref/
zotero-catalog strategies already fail on -- since that is the gap a local
LLM-shaped strategy exists to fill. A weak result does not kill the idea
outright (fine-tuning may still close the gap) but means the companion
spec's production wiring should wait until a fine-tuned or larger NuExtract
variant is validated the same way, rather than shipping a zero-shot strategy
that regresses accuracy relative to today's cloud-LLM fallback.

## Out of scope

- Deciding whether to *also* fine-tune NuExtract -- that is a follow-up
  decision made after reading this spike's output, not part of this spec.
- NuExtract-2.0 (multimodal) evaluation -- a candidate follow-up spike with
  the same shape, once the text-only baseline is measured.
- Extending `crossref_gt` to 100+ books -- orthogonal, already possible
  independently of this spike; not required to produce a first baseline
  number on the existing ~50-book corpus.
