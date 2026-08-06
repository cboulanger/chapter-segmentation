"""Local-binary OcrBackend: renders PDF pages with pymupdf and recognizes
text with the system `tesseract` binary via pytesseract. No daemon, no
network, no container -- see chapter_segmentation.ocr.OcrBackend and
design spec section 4 (in the zotero-rag repo this was extracted from) for
why this exists alongside KreuzbergOcrBackend.

Requires the `tesseract` extra (pytesseract, pymupdf, pillow) AND the
system `tesseract` binary + language data on PATH:
    apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-spa   # Debian/Ubuntu
    brew install tesseract tesseract-lang                                                  # macOS
"""

import asyncio
import shutil
from typing import Optional


_INSTALL_HINT = (
    "tesseract binary not found on PATH. Install it: "
    "apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-spa (Debian/Ubuntu), "
    "brew install tesseract tesseract-lang (macOS), "
    "or see https://tesseract-ocr.github.io/tessdoc/Installation.html"
)


class TesseractOcrBackend:
    """Renders each page at `dpi` and OCRs it with pytesseract/tesseract."""

    def __init__(self, dpi: int = 300):
        if shutil.which("tesseract") is None:
            raise RuntimeError(_INSTALL_HINT)
        self._dpi = dpi

    async def ocr_pdf_pages(self, content: bytes, *, language: Optional[str] = None) -> list[str]:
        return await asyncio.to_thread(self._ocr_sync, content, language or "eng")

    def _ocr_sync(self, content: bytes, language: str) -> list[str]:
        import fitz  # pymupdf
        import pytesseract
        from PIL import Image

        doc = fitz.open(stream=content, filetype="pdf")
        try:
            pages: list[str] = []
            for page in doc:
                pix = page.get_pixmap(dpi=self._dpi)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(pytesseract.image_to_string(image, lang=language))
            return pages
        finally:
            doc.close()
