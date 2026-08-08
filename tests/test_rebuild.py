import json
import os
import tempfile
import unittest
from pathlib import Path

import pymupdf

from rag_pdf_ingest.config import Settings
from rag_pdf_ingest.pipeline import rebuild_book


class RebuildBookTests(unittest.TestCase):
    def test_active_ingestion_lock_blocks_entire_rebuild_transaction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings.load(root)
            book_id = "locked-book"
            metadata_dir = settings.metadata_dir / book_id
            metadata_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = metadata_dir / "manifest.json"
            original = {
                "book_id": book_id,
                "source_file": "Locked.pdf",
                "source_sha256": "d" * 64,
                "page_count": 1,
                "status": "partial",
            }
            manifest_path.write_text(json.dumps(original), encoding="utf-8")
            lock_path = settings.state_dir / "locks" / f"{book_id}.lock"
            lock_path.write_text(
                json.dumps({"pid": os.getpid(), "source_file": "Locked.pdf"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "already being processed"):
                rebuild_book(
                    settings,
                    book_id,
                    refresh_pdf_facts=True,
                    refresh_native_fallbacks=True,
                )
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")), original
            )
            self.assertTrue(lock_path.exists())

    def test_refresh_pdf_facts_and_native_fallbacks_is_offline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings.load(root)
            book_id = "refresh-book"
            for directory in (
                settings.raw_dir,
                settings.md_dir,
                settings.metadata_dir,
            ):
                (directory / book_id).mkdir(parents=True, exist_ok=True)
            processed_pdf = settings.processed_dir / f"{book_id}.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Structured fallback heading", fontsize=16)
            page.insert_text((72, 105), "Exact native fallback text.", fontsize=11)
            document.set_metadata({"title": "PDF Metadata Title", "author": "PDF Author"})
            document.set_toc([[1, "Structured fallback heading", 1]])
            document.set_page_labels(
                [{"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 1}]
            )
            document.save(processed_pdf)
            document.close()
            (settings.raw_dir / book_id / "page-0001.txt").write_text("Old raw\n")
            (settings.md_dir / book_id / "page-0001.md").write_text(
                "---\nstatus: native_text_fallback\n---\n\nOld fallback.\n",
                encoding="utf-8",
            )
            (settings.metadata_dir / book_id / "page-0001.json").write_text(
                json.dumps(
                    {
                        "pdf_page_number": 1,
                        "status": "native_text_fallback",
                        "citation": "Old.pdf, PDF p. 1",
                        "parser_model": "gemini-test",
                        "finish_reason": "RECITATION",
                        "error": "recitation",
                    }
                ),
                encoding="utf-8",
            )
            (settings.metadata_dir / book_id / "manifest.json").write_text(
                json.dumps(
                    {
                        "book_id": book_id,
                        "source_file": "Original Filename.pdf",
                        "source_sha256": "c" * 64,
                        "page_count": 1,
                        "status": "partial",
                        "failed_pages": [],
                        "fallback_pages": [1],
                        "processed_pdf": f"processed/{book_id}.pdf",
                        "license": {"status": "authorized"},
                    }
                ),
                encoding="utf-8",
            )

            rebuilt = rebuild_book(
                settings,
                book_id,
                refresh_pdf_facts=True,
                refresh_native_fallbacks=True,
            )
            self.assertEqual(rebuilt["bibliography"]["title"], "PDF Metadata Title")
            self.assertEqual(rebuilt["bibliography"]["title_source"], "pdf_metadata")
            page_metadata = json.loads(
                (settings.metadata_dir / book_id / "page-0001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(page_metadata["printed_page_label"], "A-1")
            self.assertEqual(page_metadata["outline_path"], ["Structured fallback heading"])
            self.assertEqual(page_metadata["fallback_source"], "pymupdf_structured_v1")
            self.assertTrue(
                (settings.raw_dir / book_id / "page-0001.ir.json").exists()
            )
            self.assertTrue(
                (settings.raw_dir / book_id / "page-0001.native.md").exists()
            )
            page_markdown = (
                settings.md_dir / book_id / "page-0001.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Exact native fallback text.", page_markdown)
            self.assertIn("printed p. A-1", page_metadata["citation"])

    def test_rebuild_attaches_license_and_quality_without_pdf_or_gemini(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings.load(root)
            book_id = "book-abc"
            for directory in (settings.raw_dir, settings.md_dir, settings.metadata_dir):
                (directory / book_id).mkdir(parents=True, exist_ok=True)
            statuses = ("complete", "native_text_fallback")
            for page_number, status in enumerate(statuses, 1):
                stem = f"page-{page_number:04d}"
                (settings.raw_dir / book_id / f"{stem}.txt").write_text(
                    f"Native {page_number}\n", encoding="utf-8"
                )
                (settings.md_dir / book_id / f"{stem}.md").write_text(
                    f"---\nstatus: {status}\n---\n\nPage {page_number} body.\n",
                    encoding="utf-8",
                )
                (settings.metadata_dir / book_id / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "pdf_page_number": page_number,
                            "status": status,
                            "citation": f"Book.pdf, PDF p. {page_number}",
                        }
                    ),
                    encoding="utf-8",
                )
            manifest = {
                "book_id": book_id,
                "source_file": "Book.pdf",
                "source_sha256": "a" * 64,
                "page_count": 2,
                "status": "partial",
                "failed_pages": [],
                "fallback_pages": [2],
                "license": {"status": "unspecified"},
            }
            (settings.metadata_dir / book_id / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            license_path = root / "Book.license.json"
            license_path.write_text(
                json.dumps(
                    {
                        "status": "licensed",
                        "scope": "private hosted retrieval",
                        "bibliography": {
                            "title": "Canonical Book Title",
                            "authors": ["Ada Author"],
                            "edition": "2nd",
                            "publication_year": 2025,
                            "language": "en",
                        },
                    }
                ),
                encoding="utf-8",
            )

            rebuilt = rebuild_book(settings, book_id, license_sidecar=license_path)
            self.assertEqual(rebuilt["license"]["status"], "licensed")
            self.assertEqual(rebuilt["bibliography"]["title"], "Canonical Book Title")
            self.assertEqual(rebuilt["bibliography"]["title_source"], "sidecar")
            records = [
                json.loads(line)
                for line in (settings.chunks_dir / book_id / "chunks.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(records[0]["has_native_text_fallback"])
            self.assertEqual(records[0]["fallback_pages"], [2])
            self.assertEqual(records[0]["license_status"], "licensed")
            self.assertEqual(records[0]["book_title"], "Canonical Book Title")
            self.assertEqual(records[0]["authors"], ["Ada Author"])
            self.assertIn("Ada Author, Canonical Book Title, 2nd ed., 2025", records[0]["citation"])
            page_metadata = json.loads(
                (settings.metadata_dir / book_id / "page-0001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(page_metadata["book_title"], "Canonical Book Title")
            self.assertIn("book_title: \"Canonical Book Title\"", (
                settings.md_dir / book_id / "page-0001.md"
            ).read_text(encoding="utf-8"))

    def test_rebuild_uses_filename_stem_as_display_title_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings.load(root)
            book_id = "fallback-book"
            for directory in (settings.raw_dir, settings.md_dir, settings.metadata_dir):
                (directory / book_id).mkdir(parents=True, exist_ok=True)
            (settings.raw_dir / book_id / "page-0001.txt").write_text("Text\n")
            (settings.md_dir / book_id / "page-0001.md").write_text(
                "---\nstatus: complete\n---\n\nBody.\n", encoding="utf-8"
            )
            (settings.metadata_dir / book_id / "page-0001.json").write_text(
                json.dumps(
                    {
                        "pdf_page_number": 1,
                        "status": "complete",
                        "citation": "Useful Book - Copy.pdf, PDF p. 1",
                    }
                ),
                encoding="utf-8",
            )
            (settings.metadata_dir / book_id / "manifest.json").write_text(
                json.dumps(
                    {
                        "book_id": book_id,
                        "source_file": "Useful Book - Copy.pdf",
                        "source_sha256": "b" * 64,
                        "page_count": 1,
                        "status": "complete",
                        "failed_pages": [],
                        "fallback_pages": [],
                        "license": {"status": "authorized"},
                        "bibliography": {
                            "title": "Old Manual Title",
                            "title_source": "sidecar",
                            "authors": ["Old Author"],
                            "edition": None,
                            "publication_year": None,
                            "language": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            replacement_license = root / "fallback.license.json"
            replacement_license.write_text(
                json.dumps({"status": "authorized"}), encoding="utf-8"
            )

            rebuilt = rebuild_book(
                settings, book_id, license_sidecar=replacement_license
            )
            self.assertEqual(rebuilt["bibliography"]["title"], "Useful Book - Copy")
            self.assertEqual(rebuilt["bibliography"]["title_source"], "source_file")
            self.assertEqual(rebuilt["source_file"], "Useful Book - Copy.pdf")


if __name__ == "__main__":
    unittest.main()
