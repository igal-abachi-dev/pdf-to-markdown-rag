import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_pdf_ingest.cli import _parse_pages, main


class CliTests(unittest.TestCase):
    def test_retry_page_ranges(self):
        self.assertEqual(_parse_pages("4,7-9"), {4, 7, 8, 9})

    def test_invalid_chunk_settings_return_clean_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text(
                "CHUNK_TARGET_TOKENS=901\nCHUNK_MAX_TOKENS=900\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(stderr):
                result = main(["--root", str(root), "doctor"])
            output = stderr.getvalue()
            self.assertEqual(result, 1)
            self.assertIn("ERROR: CHUNK_TARGET_TOKENS cannot exceed", output)
            self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
