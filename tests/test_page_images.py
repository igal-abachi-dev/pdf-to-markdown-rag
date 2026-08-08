import json
import os
import tempfile
import unittest
from pathlib import Path

import pymupdf

from rag_pdf_ingest.config import Settings
from rag_pdf_ingest.page_images import convert_page_images_to_jpeg


class PageImageTests(unittest.TestCase):
    def _make_page_png(self, settings: Settings, book_id: str) -> Path:
        book_pages = settings.pages_dir / book_id
        book_metadata = settings.metadata_dir / book_id
        book_pages.mkdir(parents=True, exist_ok=True)
        book_metadata.mkdir(parents=True, exist_ok=True)
        document = pymupdf.open()
        page = document.new_page(width=400, height=600)
        page.insert_text((40, 60), "Squat biomechanics: 5 x 5", fontsize=18)
        pixmap = page.get_pixmap(dpi=180, alpha=False)
        png_path = book_pages / "page-0001.png"
        png_path.write_bytes(pixmap.tobytes("png"))
        document.close()
        metadata = {
            "book_id": book_id,
            "pdf_page_number": 1,
            "page_image_path": f"pages/{book_id}/page-0001.png",
            "page_image_mime_type": "image/png",
            "page_image_bytes": png_path.stat().st_size,
        }
        (book_metadata / "page-0001.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        (book_metadata / "pages.jsonl").write_text(
            json.dumps(metadata) + "\n", encoding="utf-8"
        )
        return png_path

    def test_conversion_replaces_png_and_updates_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings.load(Path(temp_dir))
            book_id = "book-123"
            png_path = self._make_page_png(settings, book_id)
            original_size = png_path.stat().st_size

            result = convert_page_images_to_jpeg(
                settings, book_ids={book_id}, quality=90
            )

            jpeg_path = png_path.with_suffix(".jpg")
            self.assertEqual(result["converted"], 1)
            self.assertFalse(result["errors"])
            self.assertFalse(png_path.exists())
            self.assertTrue(jpeg_path.exists())
            pixmap = pymupdf.Pixmap(str(jpeg_path))
            self.assertGreater(pixmap.width, 0)
            metadata = json.loads(
                (settings.metadata_dir / book_id / "page-0001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["page_image_mime_type"], "image/jpeg")
            self.assertTrue(metadata["page_image_path"].endswith(".jpg"))
            self.assertEqual(metadata["page_image_original_bytes"], original_size)
            self.assertEqual(metadata["page_image_jpeg_quality"], 90)
            combined = json.loads(
                (settings.metadata_dir / book_id / "pages.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(combined["page_image_path"], metadata["page_image_path"])

    def test_active_book_lock_is_respected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings.load(Path(temp_dir))
            book_id = "book-locked"
            png_path = self._make_page_png(settings, book_id)
            lock_path = settings.state_dir / "locks" / f"{book_id}.lock"
            lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

            result = convert_page_images_to_jpeg(settings, book_ids={book_id})

            self.assertEqual(result["converted"], 0)
            self.assertEqual(result["skipped_locked"], [book_id])
            self.assertTrue(png_path.exists())
            self.assertTrue(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
