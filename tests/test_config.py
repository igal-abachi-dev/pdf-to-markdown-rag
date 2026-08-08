import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_pdf_ingest.config import Settings


class ConfigTests(unittest.TestCase):
    def test_native_page_router_is_opt_in(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.dict(os.environ, {"NATIVE_PAGE_ROUTER_ENABLED": "false"}):
                settings = Settings.load(root)
                self.assertFalse(settings.native_page_router_enabled)
                (root / ".env").write_text(
                    "NATIVE_PAGE_ROUTER_ENABLED=true\n", encoding="utf-8"
                )
                enabled = Settings.load(root)
                self.assertTrue(enabled.native_page_router_enabled)

    def test_project_env_overrides_stale_shell_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text(
                "GEMINI_MODEL=project-model\nGEMINI_API_KEY=project-key\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"GEMINI_MODEL": "stale-shell-model", "GEMINI_API_KEY": "stale-key"},
            ):
                settings = Settings.load(root)
                self.assertEqual(settings.gemini_model, "project-model")
                self.assertEqual(settings.gemini_api_key, "project-key")
                self.assertEqual(settings.page_image_format, "JPEG")
                self.assertEqual(settings.page_jpeg_quality, 90)


if __name__ == "__main__":
    unittest.main()
