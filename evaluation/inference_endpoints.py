"""Provider-agnostic OpenAI-compatible inference endpoints -- lets
generate_dnb_toc_ground_truth.py and refresh_llm_cache.py target KISSKI,
an MPCDF LLM Inference Service session (https://llm.mpcdf.mpg.de), or any
other OpenAI-compatible chat completions endpoint, selected per invocation
via a repeatable --endpoint ALIAS CLI flag. See design spec
docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md.

Deliberately NOT a Provider class hierarchy: KISSKI's real discovery/
demand-aware model selection (evaluation/kisski.py) has no equivalent for
a self-deployed MPCDF session -- no shared pool, no discovery endpoint,
you pick and deploy exactly one model yourself. This module stays a flat
(base_url, api_key, model_id) resolver; evaluation/kisski.py is untouched
and remains the default path when no --endpoint is given.
"""

import os
from dataclasses import dataclass

from openai import AsyncOpenAI

DEFAULT_TIMEOUT = 90.0


@dataclass(frozen=True)
class ModelEndpoint:
    """One ready-to-call (client, model_id) pair. `label` is the alias
    (or "kisski" for the auto-selected default path) -- used only for
    log/print output, never to branch behavior."""

    label: str
    model_id: str
    client: AsyncOpenAI


def resolve_endpoint_from_env(alias: str, *, timeout: float = DEFAULT_TIMEOUT) -> ModelEndpoint:
    """Builds a ModelEndpoint from `<alias>_BASE_URL`, `<alias>_API_KEY`,
    `<alias>_MODEL` environment variables. Raises ValueError naming
    exactly which variable(s) are missing -- this is meant to be diagnosed
    by a human setting up an MPCDF (or other) session, not by reading a
    bare KeyError traceback."""
    var_names = (f"{alias}_BASE_URL", f"{alias}_API_KEY", f"{alias}_MODEL")
    missing = [var for var in var_names if var not in os.environ]
    if missing:
        raise ValueError(
            f"--endpoint {alias} requires environment variables "
            f"{alias}_BASE_URL, {alias}_API_KEY, {alias}_MODEL to be set -- "
            f"missing: {', '.join(missing)}"
        )
    base_url, api_key, model_id = (os.environ[var] for var in var_names)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return ModelEndpoint(label=alias, model_id=model_id, client=client)
