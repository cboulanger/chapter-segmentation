import io
import unittest
from unittest.mock import AsyncMock, patch

from pypdf import PdfWriter

from chapter_segmentation.ocr_backends.kreuzberg import KreuzbergOcrBackend


def _two_page_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestKreuzbergOcrBackend(unittest.IsolatedAsyncioTestCase):
    async def test_sends_one_request_per_page_and_joins_chunks(self):
        pdf_bytes = _two_page_pdf_bytes()
        backend = KreuzbergOcrBackend(kreuzberg_url="http://fake:8100")

        responses = [
            AsyncMock(status_code=200, json=lambda: [{"chunks": [{"content": "page one"}]}]),
            AsyncMock(status_code=200, json=lambda: [{"chunks": [{"content": "page two"}]}]),
        ]
        for r in responses:
            r.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.post", AsyncMock(side_effect=responses)) as mock_post:
            pages = await backend.ocr_pdf_pages(pdf_bytes, language="eng")

        self.assertEqual(pages, ["page one", "page two"])
        self.assertEqual(mock_post.await_count, 2)
        first_call_kwargs = mock_post.await_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["data"], {"config": '{"force_ocr": true, "ocr": {"language": "eng"}}'})


if __name__ == "__main__":
    unittest.main()
