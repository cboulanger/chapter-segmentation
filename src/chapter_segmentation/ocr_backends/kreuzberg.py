"""Kreuzberg-sidecar-backed OcrBackend. Requires the `kreuzberg` optional
extra (httpx). See chapter_segmentation.ocr.OcrBackend.
"""

import io
import json
from typing import Optional

import httpx
from pypdf import PdfReader

from chapter_segmentation.ocr import slice_single_page_pdf


class KreuzbergOcrBackend:
    """Calls a running Kreuzberg sidecar's HTTP API, one request per page --
    guarantees a clean 1:1 page-index<->text mapping, matching pypdf's own
    indexing used throughout the segmentation engine.
    """

    def __init__(self, kreuzberg_url: str = "http://localhost:8100", timeout: float = 120.0):
        self._kreuzberg_url = kreuzberg_url.rstrip("/")
        self._timeout = timeout

    async def ocr_pdf_pages(self, content: bytes, *, language: Optional[str] = None) -> list[str]:
        reader = PdfReader(io.BytesIO(content))
        total_pages = len(reader.pages)
        config = {"force_ocr": True}
        if language:
            config["ocr"] = {"language": language}

        page_texts: list[str] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for page_index in range(total_pages):
                single_page_bytes = slice_single_page_pdf(content, page_index)
                response = await client.post(
                    f"{self._kreuzberg_url}/extract",
                    files={"files": ("page.pdf", single_page_bytes, "application/pdf")},
                    data={"config": json.dumps(config)},
                )
                response.raise_for_status()
                results = response.json()
                chunks = (results[0].get("chunks") or []) if results else []
                page_texts.append(" ".join(c.get("content") or "" for c in chunks))
        return page_texts
