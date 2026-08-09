# Pluggable, cost-ordered TOC-extraction strategy cascade

Status: approved for planning
Date: 2026-08-09

## Problem

`analyze_attachment_with_strategies` (segmentation.py:1652) accepts a single
`llm_client: Optional[LLMClient]` and hardcodes exactly one fallback path
when the deterministic pipeline (outline + Crossref + Zotero-catalog fusion)
comes up empty: `analyze_attachment_with_llm_fallback`, which calls
whichever cloud provider the caller's one `LLMClient` implementation talks
to. There is no way to try more than one LLM-shaped strategy, and no way to
prefer a free/local/instant option before an expensive/networked one.

The companion spec (`2026-08-09-nuextract-baseline-evaluation-design.md`)
is about to produce a second, structurally different TOC-extraction
strategy -- a local NuExtract model, served via Ollama, driven by a JSON
template rather than a free-form chat prompt. A future third
(NuExtract-2.0, multimodal, image input) and a fourth (a fine-tuned variant
of either) are expected to follow. None of these share `LLMClient`'s
`generate(prompt: str) -> str` calling convention once images enter the
picture, and forcing them to pretend they do (base64-encoding an image into
a prompt string) would be a worse fit than giving each strategy its own
internal calling convention behind one uniform contract.

Goal: generalize the single hardcoded LLM-fallback slot into an ordered,
freely composable list of `TocExtractionStrategy` implementations, tried in
the caller-supplied order (cheapest/fastest first), falling through to the
next only when the current one produces nothing usable -- mirroring the
`if not merged: ...` fallthrough already used one level up
(segmentation.py:1717) for the deterministic-pipeline-to-LLM handoff, now
generalized from a binary choice to an N-ary chain.

## The `TocExtractionStrategy` Protocol

New in `evidence/types.py`, sibling to the existing `StructureStrategy`/
`MetadataStrategy`:

```python
class TocExtractionStrategy(Protocol):
    def applicable(self, pages: list[str]) -> bool: ...
    async def extract(self, pages: list[str]) -> list[TocEntry]: ...
```

`applicable` is a cheap pre-check (e.g. "does `_llm_scan_indices` find any
candidate pages at all") that lets the cascade skip a strategy without
paying its call cost. `extract` returns `[]` for "found nothing" -- the
cascade's only fallthrough signal (see "Cascade trigger" below); a strategy
must never raise for an ordinary "no TOC found" outcome, only for a genuine
failure (network error, malformed response after retries), which the
cascade catches and treats identically to an empty result.

Each concrete implementation owns its full internal calling convention:

- **`PromptedLLMStrategy`** (new name for today's behavior) -- wraps the
  existing `llm_extract_toc_entries` + `LLMClient` flow unchanged. Preserves
  today's cloud-LLM behavior exactly; existing `LLMClient` implementations
  need no changes.
- **`NuExtractLocalStrategy`** -- the companion spec's NuExtract-1.5-tiny
  integration, once its baseline numbers justify shipping it: template
  construction, Ollama HTTP call, JSON-to-`TocEntry` parsing, all internal.
- Future: a multimodal NuExtract strategy, or a fine-tuned variant of either
  -- same Protocol, no changes needed elsewhere.

`LLMClient` (llm.py) is unchanged and not deprecated -- it remains exactly
what `PromptedLLMStrategy` needs it to be. It is not generalized to carry
images or templates; that responsibility moves up to the
`TocExtractionStrategy` level, which is where it belongs since different
strategies genuinely need different shapes there.

## Cascade mechanics

`analyze_attachment_with_strategies`'s signature changes:

```python
async def analyze_attachment_with_strategies(
    pages: list[str],
    file_bytes: bytes,
    context: BookContext,
    zotero_catalog_strategy: MetadataStrategy,
    crossref_strategy: Optional[MetadataStrategy] = None,
    *,
    toc_strategies: list[TocExtractionStrategy] = (),
) -> dict:
```

replacing the `llm_client: Optional[LLMClient] = None` parameter. Callers
that want today's single-cloud-LLM behavior pass
`toc_strategies=[PromptedLLMStrategy(llm_client)]`; callers that want the
new cost-ordered cascade pass e.g.
`toc_strategies=[NuExtractLocalStrategy(...), PromptedLLMStrategy(llm_client)]`.
**The list's order *is* the cost policy** -- no `cost_tier` enum or other
metadata is added to the Protocol; a caller that wants free/local before
paid/cloud simply lists them in that order. This is deliberately the
simplest thing that could work: nothing downstream ever needs to introspect
or compare cost, only iterate a list, so a declared cost field would be
unused complexity.

The existing empty-merge fallthrough (segmentation.py:1717-1721):

```python
if not merged:
    if llm_client is not None:
        result = await analyze_attachment_with_llm_fallback(pages, llm_client)
    else:
        result = analyze_attachment(pages)
```

becomes a loop over `toc_strategies`, stopping at the first strategy whose
`extract()` returns a non-empty list:

```python
if not merged:
    result = None
    for strategy in toc_strategies:
        if not strategy.applicable(pages):
            continue
        try:
            entries = await strategy.extract(pages)
        except Exception:
            logger.warning("toc strategy %r failed", strategy, exc_info=True)
            continue
        if entries:
            result = await _finish_with_toc_entries(
                pages, entries, disambiguation_llm_client, strategy_name=type(strategy).__name__,
            )
            break
    if result is None:
        result = analyze_attachment(pages)
```

`_finish_with_toc_entries` is a small extraction of
`analyze_attachment_with_llm_fallback`'s existing post-TOC-extraction body
(locate, disambiguate-unlocated-entries via `llm_disambiguate_chapter_start`,
`_chapters_from_located`) -- unchanged logic, just factored out so any
strategy in the loop can reach it, not only the one cloud-LLM path that
owned it today. `disambiguation_llm_client` is resolved once before the
loop (see "Chapter-start disambiguation stays as-is" below) and passed
through unchanged on every iteration -- it does not vary per winning
strategy.

## Chapter-start disambiguation stays as-is

`llm_disambiguate_chapter_start` (segmentation.py:881) -- picking which of
several candidate pages is a chapter's true start -- is a different task
shape (multiple-choice ranking over page snippets, not schema-guided
extraction) that NuExtract is not a natural fit for. It stays hardcoded to
`LLMClient` and is called once, after the winning `TocExtractionStrategy`
has produced entries, exactly as `analyze_attachment_with_llm_fallback` does
today. `_finish_with_toc_entries` takes an `Optional[LLMClient]` for this
step alone; the cascade constructing it passes whichever `LLMClient` the
`PromptedLLMStrategy` (if any) in `toc_strategies` already has, or `None` if
none is present (ambiguous entries are then left unlocated, same as today's
`llm_client=None` behavior). Generalizing disambiguation to a pluggable
cascade of its own is explicitly out of scope (see below) -- it can be
revisited once a non-chat-LLM disambiguation strategy actually exists to
justify it.

## Diagnostics

`diagnostics["strategies_used"]` (already present, segmentation.py:1723)
gains the winning `TocExtractionStrategy`'s class name (or `None` if every
strategy came up empty and pure heuristic ran), so a caller can see which
tier of the cascade actually produced a result -- useful both for debugging
and for the same kind of per-strategy accuracy reporting `RESULTS.md`
already does for the deterministic strategies.

## Migration

- `analyze_attachment_with_llm_fallback` and `analyze_attachment_llm_only`
  (segmentation.py:1456, 1529) -- kept as thin wrappers for their own
  direct callers/tests (`analyze_attachment_llm_only` in particular exists
  specifically to measure one strategy's standalone accuracy, which the
  cascade's `applicable`/`extract` split already supports per-strategy
  without needing the wrapper, but no existing test depends on removing it
  and it costs nothing to keep). Their bodies now delegate to
  `PromptedLLMStrategy` + `_finish_with_toc_entries` internally rather than
  duplicating the extraction/disambiguation logic.
- `tests/test_segmentation_strategies.py` -- every
  `analyze_attachment_with_strategies(..., llm_client=llm)` call site
  updates to `toc_strategies=[PromptedLLMStrategy(llm)]` (or `[]` where the
  test currently passes no client at all). No back-compat shim for the old
  keyword -- there is no external production caller of this function today
  (grepped; only this test module and `test_segmentation.py` call the
  LLM-fallback entry points, and neither `cli.py` nor the `zotero-rag`
  working directory currently wires an `LLMClient` at all), so a clean
  rename is safe.
- `tests/test_segmentation.py` -- `llm_disambiguate_chapter_start` call
  sites are unchanged (still takes `LLMClient` directly, per "stays as-is"
  above); `analyze_attachment_with_llm_fallback`/`analyze_attachment_llm_only`
  call sites are unchanged in shape (still take a bare `LLMClient`), since
  those wrapper functions keep their existing signatures.
- New unit tests for the cascade loop itself, using fake
  `TocExtractionStrategy` instances (`applicable` always `True`, `extract`
  returning either `[]` or a fixed list, or raising) to verify: ordering is
  respected, first non-empty result wins and later strategies are never
  called, an `applicable() == False` strategy is skipped without being
  called, a raising strategy is caught and treated as empty, and an
  all-empty cascade falls through to `analyze_attachment` exactly as
  `llm_client=None` does today.

## Out of scope

- Any change to `llm_disambiguate_chapter_start`'s own pluggability (see
  above) -- it remains a single hardcoded `LLMClient` call.
- A `cost_tier`/priority field on the Protocol -- ordering is entirely the
  caller's responsibility via list order, per "Cascade mechanics" above.
- Concurrent/parallel strategy execution (e.g. racing two local strategies)
  -- strictly sequential, matching the existing single-fallback model's
  semantics and this spec's "cheap first, only pay for the next tier if the
  cheap one truly found nothing" goal, which a race would undermine.
- Implementing `NuExtractLocalStrategy` itself -- this spec defines the
  Protocol it must satisfy; the companion spec's baseline spike is the
  prerequisite that decides whether it gets built at all.
- Wiring any of this into `cli.py` or an external consumer -- no such
  wiring exists today for the single-`LLMClient` case either.
