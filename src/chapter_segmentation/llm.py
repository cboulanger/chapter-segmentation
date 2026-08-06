"""Minimal LLM-client interface the segmentation engine's optional LLM
fallback depends on -- see design spec
docs/superpowers/specs/2026-08-06-chapter-segmentation-extraction-design.md
section 5 (in the zotero-rag repo this was extracted from).

Any object exposing this single async method works -- structural typing
means callers never need to subclass this Protocol explicitly.
"""

from typing import Callable, Optional, Protocol


class LLMClient(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        is_valid: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Return the model's raw text completion for `prompt`.

        `is_valid`, when given, lets an implementation retry against a
        different model/provider if the first response doesn't satisfy it
        (e.g. isn't parseable as the JSON the caller asked for) -- entirely
        optional to honor; a minimal implementation may ignore it.
        """
        ...
