import tempfile
import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import pymupdf

    from rag_pdf_ingest.config import Settings
    from rag_pdf_ingest.gemini_client import GeminiResponseError
    from rag_pdf_ingest.pipeline import ingest_pdf

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@unittest.skipUnless(PIPELINE_AVAILABLE, "pipeline dependencies are not installed")
class PipelineRetryTests(unittest.TestCase):
    def test_opt_in_native_router_skips_only_pages_that_pass_every_gate(self):
        environment = patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        class RouterConverter:
            calls: list[int] = []

            def __init__(self, settings):
                self.settings = settings

            def convert_page(self, page):
                self.__class__.calls.append(page.page_number)
                return SimpleNamespace(
                    markdown=f"# Visual page {page.page_number}",
                    backend="gemini",
                    model="gemini-test",
                    page_ir=None,
                    structured_data=None,
                    diagnostics={},
                    warnings=(),
                    response_id=f"response-{page.page_number}",
                    model_version="test-model",
                    usage_metadata={},
                    finish_reason="STOP",
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir()
            (root / "prompts" / "page_to_markdown.txt").write_text(
                "Transcribe the page.", encoding="utf-8"
            )
            (root / ".env").write_text(
                "GEMINI_API_KEY=test-key\n"
                "GEMINI_REQUESTS_PER_DAY=0\n"
                "GEMINI_MIN_REQUEST_INTERVAL_SECONDS=0\n"
                "NATIVE_PAGE_ROUTER_ENABLED=true\n",
                encoding="utf-8",
            )
            settings = Settings.load(root)
            pdf_path = settings.inbox_dir / "Router.pdf"
            document = pymupdf.open()
            text_page = document.new_page(width=400, height=500)
            text_page.insert_text(
                (72, 90), "Clean native paragraph with exact strength data.", fontsize=11
            )
            visual_page = document.new_page(width=400, height=500)
            pixmap = pymupdf.Pixmap(
                pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False
            )
            pixmap.clear_with(0xCCCCCC)
            visual_page.insert_image((30, 30, 370, 470), pixmap=pixmap)
            document.save(pdf_path)
            document.close()

            RouterConverter.calls = []
            with patch("rag_pdf_ingest.pipeline.GeminiPageConverter", RouterConverter):
                manifest = ingest_pdf(pdf_path, settings)

            self.assertEqual(RouterConverter.calls, [2])
            self.assertEqual(manifest["native_routed_pages"], [1])
            self.assertEqual(
                manifest["native_router_summary"],
                {
                    "page_count": 2,
                    "eligible_if_enabled": 1,
                    "native_routed": 1,
                    "llm_invoked": 1,
                    "failed_gate_counts": {
                        "native_text_empty": 1,
                        "native_text_quality_gate_failed": 1,
                        "needs_visual_parser": 1,
                    },
                    "native_text_quality_reason_counts": {"native_text_empty": 1},
                },
            )
            metadata_dir = settings.metadata_dir / manifest["book_id"]
            page_one = json.loads(
                (metadata_dir / "page-0001.json").read_text(encoding="utf-8")
            )
            page_two = json.loads(
                (metadata_dir / "page-0002.json").read_text(encoding="utf-8")
            )
            self.assertEqual(page_one["conversion_route"], "native_structured")
            self.assertEqual(page_one["parser_backend"], "pymupdf_native")
            self.assertFalse(page_one["llm_called"])
            self.assertEqual(page_one["finish_reason"], "NOT_CALLED")
            self.assertEqual(page_two["conversion_route"], "visual_llm")
            self.assertTrue(page_two["llm_called"])
            chunks = [
                json.loads(line)
                for line in (
                    settings.chunks_dir / manifest["book_id"] / "chunks.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(1 in chunk["native_structured_pages"] for chunk in chunks))
            self.assertTrue(any(chunk["has_native_structured"] for chunk in chunks))

    def test_recitation_fallback_and_hash_based_single_page_retry(self):
        class FakeConverter:
            recite_page_two = True
            calls: list[int] = []

            def __init__(self, settings):
                self.settings = settings

            def convert(self, image_bytes, native_text, page_number, image_mime_type="image/png"):
                self.__class__.calls.append(page_number)
                if page_number == 2 and self.__class__.recite_page_two:
                    raise GeminiResponseError(
                        "Gemini stopped with finish_reason=RECITATION",
                        finish_reason="RECITATION",
                        response_snapshot={"candidates": [{"finish_reason": "RECITATION"}]},
                    )
                return SimpleNamespace(
                    markdown=f"# Page {page_number}\n\n{native_text}",
                    response_id=f"response-{page_number}",
                    model_version="test-model",
                    usage_metadata={},
                    finish_reason="STOP",
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir()
            (root / "prompts" / "page_to_markdown.txt").write_text(
                "Transcribe the page.", encoding="utf-8"
            )
            (root / ".env").write_text(
                "GEMINI_API_KEY=test-key\n"
                "GEMINI_REQUESTS_PER_DAY=0\n"
                "GEMINI_MIN_REQUEST_INTERVAL_SECONDS=0\n"
                "NATIVE_PAGE_ROUTER_ENABLED=false\n",
                encoding="utf-8",
            )
            settings = Settings.load(root)
            pdf_path = settings.inbox_dir / "Book.pdf"
            document = pymupdf.open()
            for page_number in (1, 2):
                page = document.new_page()
                page.insert_text((72, 72), f"Native text for page {page_number}.")
            document.save(pdf_path)
            document.close()

            FakeConverter.calls = []
            with patch("rag_pdf_ingest.pipeline.GeminiPageConverter", FakeConverter):
                first_manifest = ingest_pdf(pdf_path, settings)
            canonical_book_id = first_manifest["book_id"]
            self.assertEqual(first_manifest["status"], "partial")
            self.assertEqual(first_manifest["fallback_pages"], [2])
            self.assertEqual(FakeConverter.calls, [1, 2])
            first_page_metadata = json.loads(
                (settings.metadata_dir / canonical_book_id / "page-0001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(first_page_metadata["page_image_mime_type"], "image/jpeg")
            self.assertTrue(first_page_metadata["page_image_path"].endswith(".jpg"))
            self.assertTrue(
                (settings.root / first_page_metadata["page_image_path"]).exists()
            )
            fallback_markdown = (
                settings.md_dir / canonical_book_id / "page-0002.md"
            ).read_text(encoding="utf-8")
            self.assertIn('status: "native_text_fallback"', fallback_markdown)
            self.assertIn("Gemini finish reason: `RECITATION`", fallback_markdown)
            self.assertIn("Native text for page 2.", fallback_markdown)

            processed_pdf = settings.processed_dir / f"{canonical_book_id}.pdf"
            FakeConverter.recite_page_two = False
            FakeConverter.calls = []
            with patch("rag_pdf_ingest.pipeline.GeminiPageConverter", FakeConverter):
                second_manifest = ingest_pdf(
                    processed_pdf,
                    settings,
                    retry_pages={2},
                    retry_fallback_pages=True,
                    retry_existing=True,
                )
            self.assertEqual(second_manifest["book_id"], canonical_book_id)
            self.assertEqual(second_manifest["source_file"], "Book.pdf")
            self.assertEqual(second_manifest["status"], "complete")
            self.assertEqual(FakeConverter.calls, [2])


if __name__ == "__main__":
    unittest.main()
