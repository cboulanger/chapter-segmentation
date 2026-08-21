"""Provider-agnostic OpenAI-compatible inference endpoints -- lets
generate_dnb_toc_ground_truth.py and refresh_llm_cache.py target KISSKI,
an MPCDF LLM Inference Service session (https://llm.mpcdf.mpg.de), or any
other OpenAI-compatible chat completions endpoint, selected per invocation
via either a repeatable --endpoint ALIAS CLI flag (env-var-backed) or an
explicit --config-file PATH flag (pasted-session-table-backed). See design
spec docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md.

Deliberately NOT a Provider class hierarchy: KISSKI's real discovery/
demand-aware model selection (evaluation/kisski.py) has no equivalent for
a self-deployed MPCDF session -- no shared pool, no discovery endpoint,
you pick and deploy exactly one model yourself. This module stays a flat
(base_url, api_key, model_id) resolver; evaluation/kisski.py is untouched
and remains the default path when neither --endpoint nor --config-file is
given.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from openai import AsyncOpenAI

DEFAULT_TIMEOUT = 90.0

DEFAULT_SESSIONS_FILENAME = ".mpcdf-sessions.txt"

_MODEL_ARG_RE = re.compile(r"--model[= ](\S+)")


@dataclass(frozen=True)
class ModelEndpoint:
    """One ready-to-call (client, model_id) pair. `label` is the alias
    (or "kisski" for the auto-selected default path, or "config:A"/"config:B"/...
    for a --config-file-selected session) -- used only for log/print
    output, never to branch behavior."""

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


def _parse_session_block(block: str) -> dict[str, str]:
    """Parses one pasted MPCDF dashboard session table (tab-separated
    `field<TAB>value` lines, exactly as copied from the UI, no reformatting)
    into a dict."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if "\t" not in line:
            continue
        key, _, value = line.partition("\t")
        key = key.strip()
        if key:
            fields[key] = value.strip()
    return fields


def load_mpcdf_sessions(path: Path) -> list[dict[str, str]]:
    """Parses a gitignored MPCDF session-table config file (see
    `evaluation/hpc/llm-mpcdf.md`) into a list of {base_url, api_key, model}
    dicts, one per pasted table, in file order. A block missing its `url`
    or `key` field, or with no `--model=` in `framework_args`, is skipped.
    Returns [] if the file doesn't exist."""
    if not path.exists():
        return []
    blocks = [b for b in re.split(r"\n\s*\n", path.read_text().strip()) if b.strip()]
    sessions = []
    for block in blocks:
        fields = _parse_session_block(block)
        url = fields.get("url", "").strip()
        api_key = fields.get("key", "").strip()
        model_match = _MODEL_ARG_RE.search(fields.get("framework_args", ""))
        if not (url and api_key and model_match):
            continue
        if not url.rstrip("/").endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        sessions.append({"base_url": url, "api_key": api_key, "model": model_match.group(1)})
    return sessions


def resolve_endpoints_from_config_file(path: Path, *, timeout: float = DEFAULT_TIMEOUT) -> list[ModelEndpoint]:
    """Builds one ModelEndpoint per pasted session table in `path` (see
    load_mpcdf_sessions), in file order, labeled "config:A", "config:B", ...
    Raises ValueError naming the path if it doesn't exist or contains no
    usable session block -- meant for a human who just pasted (or forgot
    to paste, or mistyped the path of) a session table to diagnose
    directly, not a bare empty-list surprise downstream."""
    sessions = load_mpcdf_sessions(path)
    if not sessions:
        raise ValueError(
            f"--config-file {path} has no usable pasted session table -- "
            f"see evaluation/hpc/llm-mpcdf.md for the expected format"
        )
    endpoints = []
    for i, session in enumerate(sessions):
        label = f"config:{chr(ord('A') + i)}"
        client = AsyncOpenAI(base_url=session["base_url"], api_key=session["api_key"], timeout=timeout)
        endpoints.append(ModelEndpoint(label=label, model_id=session["model"], client=client))
    return endpoints


class OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) wrapping
    an already-built AsyncOpenAI client -- callers construct the client
    themselves (KISSKI's own base_url/api_key, or a ModelEndpoint's client
    resolved above), so this class has no provider-specific knowledge at
    all. Relocated here (2026-08-20, see design spec
    docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
    section 2) from evaluation/refresh_llm_cache.py, its original home --
    a pure move, not a behavior change: evaluation/dnb_toc_ocr.py's
    text_extract_toc_entries needs the exact same LLMClient-shaped
    adapter, and importing it from refresh_llm_cache.py (a script module)
    would be backwards for a library module to depend on."""

    def __init__(self, model: str, client: AsyncOpenAI):
        self._client = client
        self._model = model

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float,
        is_valid: Optional[Callable[[str], bool]] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
