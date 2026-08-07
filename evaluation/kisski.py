"""KISSKI (Academic Cloud) model listing and selection -- mirrors
zotero-rag's backend/utils/kisski.py fetch_kisski_rag_models and
backend/dependencies.py _resolve_auto_select_candidates, reimplemented
standalone since this repo has no dependency on zotero-rag's backend
package. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"LLM cache refresh".
"""

from dataclasses import dataclass

import httpx

DEFAULT_KISSKI_BASE_URL = "https://chat-ai.academiccloud.de/v1"


@dataclass(frozen=True)
class KisskiModel:
    id: str
    name: str
    demand: int

    @property
    def availability(self) -> str:
        if self.demand == 0:
            return "available"
        if self.demand <= 5:
            return "busy"
        return "very busy"


def fetch_kisski_models(base_url: str, api_key: str, timeout: float = 5.0) -> list[KisskiModel]:
    """POST {base_url}/models -- same request shape as zotero-rag's
    fetch_kisski_rag_models. Raises on a network/HTTP error; the caller
    decides how to handle that."""
    url = base_url.rstrip("/") + "/models"
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    entries = response.json().get("data", [])
    return [
        KisskiModel(id=entry["id"], name=entry.get("name") or entry["id"], demand=int(entry.get("demand", 0)))
        for entry in entries
    ]


def select_top5(models: list[KisskiModel]) -> list[KisskiModel]:
    """On-demand-refresh selection: every non-'very busy' model, ascending
    by demand (so 'available' models sort before 'busy' ones), capped at
    5."""
    eligible = [m for m in models if m.availability != "very busy"]
    return sorted(eligible, key=lambda m: m.demand)[:5]


def select_gap_fill(models: list[KisskiModel], covered_model_ids: set[str], limit: int = 5) -> list[KisskiModel]:
    """Nightly-refresh selection: from non-'very busy' models not already
    in `covered_model_ids` (a model id counted as covered only once it has
    a cache entry for EVERY book in the current corpus -- see
    refresh_llm_cache.py's _fully_covered_model_ids), take up to `limit`
    in ascending-demand order."""
    eligible = [m for m in models if m.availability != "very busy" and m.id not in covered_model_ids]
    return sorted(eligible, key=lambda m: m.demand)[:limit]
