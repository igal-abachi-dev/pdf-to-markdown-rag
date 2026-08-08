import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from rag_pdf_ingest.config import Settings
    from rag_pdf_ingest.gemini_client import (
        GeminiResult,
        GeminiPageConverter,
        GeminiResponseError,
        _is_daily_quota_error,
        _retry_delay_seconds,
    )
    from rag_pdf_ingest.page_converter import PageInput, PageResult

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@unittest.skipUnless(GEMINI_AVAILABLE, "google-genai is not installed")
class GeminiClientTests(unittest.TestCase):
    def test_automatic_function_calling_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir()
            (root / "prompts" / "page_to_markdown.txt").write_text(
                "Transcribe.", encoding="utf-8"
            )
            (root / ".env").write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")
            converter = GeminiPageConverter(Settings.load(root))
            try:
                config = converter._config()
                self.assertTrue(config.automatic_function_calling.disable)
            finally:
                converter.close()

    def test_error_keeps_response_snapshot(self):
        snapshot = {"candidates": [{"finish_reason": "RECITATION"}]}
        error = GeminiResponseError(
            "recitation",
            finish_reason="RECITATION",
            response_snapshot=snapshot,
        )
        self.assertEqual(error.finish_reason, "RECITATION")
        self.assertEqual(error.response_snapshot, snapshot)

    def test_daily_quota_and_retry_delay_are_detected(self):
        snapshot = {
            "details": [
                {
                    "quotaId": "GenerateContentRequestsPerDay-FreeTier",
                    "retryDelay": "3600.5s",
                }
            ]
        }
        self.assertTrue(_is_daily_quota_error(snapshot))
        self.assertEqual(_retry_delay_seconds(snapshot), 3600.5)

    def test_neutral_page_contract_preserves_backend_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir()
            (root / "prompts" / "page_to_markdown.txt").write_text("Transcribe.")
            (root / ".env").write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")
            converter = GeminiPageConverter(Settings.load(root))
            expected = GeminiResult(
                markdown="# Page",
                response_id="response",
                model_version="version",
                usage_metadata={"output_tokens": 2},
                finish_reason="STOP",
                model="gemini-test",
                diagnostics={"native_reference_supplied": True},
            )
            page = PageInput(
                image_bytes=b"jpeg",
                image_mime_type="image/jpeg",
                page_number=7,
                native_text="Exact text",
            )
            try:
                with patch.object(converter, "convert", return_value=expected) as convert:
                    result = converter.convert_page(page)
                self.assertIsInstance(result, PageResult)
                self.assertEqual(result.backend, "gemini")
                self.assertEqual(result.model, "gemini-test")
                self.assertEqual(result.diagnostics["native_reference_supplied"], True)
                convert.assert_called_once_with(
                    b"jpeg", "Exact text", 7, image_mime_type="image/jpeg"
                )
            finally:
                converter.close()


if __name__ == "__main__":
    unittest.main()
