import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from rag_pdf_ingest.config import Settings
from rag_pdf_ingest.pdf_facts import (
    build_document_profile,
    extract_document_facts,
    native_text_quality,
    page_facts,
)
from rag_pdf_ingest.pipeline import _text_blocks, ingest_pdf


class PdfFactsTests(unittest.TestCase):
    def test_metadata_outline_labels_and_rotation_are_explicit(self):
        document = pymupdf.open()
        for _ in range(3):
            document.new_page()
        document.set_metadata(
            {
                "title": "Canonical PDF Title",
                "author": "Ada Author",
                "creator": "Fixture",
                "producer": "PyMuPDF",
            }
        )
        document.set_toc([[1, "Front Matter", 1], [1, "Chapter One", 3]])
        document.set_page_labels(
            [
                {"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
                {"startpage": 2, "prefix": "", "style": "D", "firstpagenum": 1},
            ]
        )
        document[1].set_rotation(90)

        facts, xmp = extract_document_facts(document)
        self.assertEqual(facts["pdf_metadata"]["title"], "Canonical PDF Title")
        self.assertEqual(facts["pdf_metadata"]["author"], "Ada Author")
        self.assertEqual([page["printed_page_label"] for page in facts["pages"]], ["i", "ii", "1"])
        self.assertEqual(facts["pages"][1]["rotation_degrees"], 90)
        self.assertEqual(facts["pages"][1]["coordinate_space"], "mupdf_unrotated")
        self.assertEqual(facts["pages"][1]["outline_path"], ["Front Matter"])
        self.assertEqual(facts["pages"][2]["outline_path"], ["Chapter One"])
        self.assertEqual(xmp, "")
        document.close()

    def test_all_rotation_values_keep_explicit_coordinate_spaces(self):
        document = pymupdf.open()
        for rotation in (0, 90, 180, 270):
            page = document.new_page(width=400, height=600)
            page.set_rotation(rotation)
        facts, _ = extract_document_facts(document)
        self.assertEqual(
            [page["rotation_degrees"] for page in facts["pages"]],
            [0, 90, 180, 270],
        )
        self.assertTrue(
            all(page["coordinate_space"] == "mupdf_unrotated" for page in facts["pages"])
        )
        self.assertTrue(
            all(page["render_coordinate_space"] == "mupdf_rotated_page" for page in facts["pages"])
        )
        document.close()

    def test_password_failure_happens_before_converter_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir()
            (root / "prompts" / "page_to_markdown.txt").write_text("Transcribe.")
            (root / ".env").write_text("GEMINI_API_KEY=test\n", encoding="utf-8")
            settings = Settings.load(root)
            pdf_path = settings.inbox_dir / "Protected.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Protected text")
            document.save(
                pdf_path,
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner-secret",
                user_pw="user-secret",
            )
            document.close()

            with patch("rag_pdf_ingest.pipeline.GeminiPageConverter") as converter:
                with self.assertRaisesRegex(RuntimeError, "requires a password"):
                    ingest_pdf(pdf_path, settings)
            converter.assert_not_called()

    def test_document_profile_detects_repeated_marginalia(self):
        document = pymupdf.open()
        for index in range(4):
            page = document.new_page(width=500, height=700)
            page.insert_text((50, 25), "Training Manual", fontsize=8)
            page.insert_text((50, 100), f"Unique body paragraph {index}.", fontsize=11)
        profile = build_document_profile(document)
        self.assertEqual(profile["body_font_size"], 11.0)
        self.assertIn("training manual", profile["repeated_marginalia"])
        document.close()

    def test_replacement_glyphs_make_native_reference_unusable(self):
        quality = native_text_quality("Valid text " + "\ufffd" * 5, [], 1000)
        self.assertFalse(quality["usable_as_reference"])
        self.assertIn("native_text_invalid_characters", quality["reason_codes"])
        self.assertIn("excessive_space_ratio", quality)
        self.assertIn("excessive_newline_ratio", quality)


if __name__ == "__main__":
    unittest.main()
