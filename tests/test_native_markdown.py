import unittest

import pymupdf

from rag_pdf_ingest.native_markdown import (
    NATIVE_PROCESSOR_CONFIG_SHA256,
    render_debug_overlay,
    render_native_page,
)
from rag_pdf_ingest.pdf_facts import (
    build_document_profile,
    extract_document_facts,
    page_facts,
)
from rag_pdf_ingest.pipeline import _text_blocks


class NativeMarkdownTests(unittest.TestCase):
    def test_merged_cells_render_as_atomic_html_table(self):
        document = pymupdf.open()
        page = document.new_page(width=400, height=400)
        page.draw_rect((50, 50, 350, 200))
        for y in (90, 130, 165):
            page.draw_line((50, y), (350, y))
        page.draw_line((200, 90), (200, 200))
        values = [
            ((130, 75), "Merged Header"),
            ((80, 115), "A"),
            ((230, 115), "B"),
            ((80, 150), "C"),
            ((230, 150), "D"),
            ((80, 190), "E"),
            ((230, 190), "F"),
        ]
        for point, value in values:
            page.insert_text(point, value, fontsize=9)
        native_text = page.get_text("text", sort=True).strip()
        result = render_native_page(
            page,
            native_text,
            _text_blocks(page),
            {"pdf_page_number": 1, "outline_path": []},
            {"body_font_size": 9.0, "heading_font_sizes": [], "repeated_marginalia": []},
        )
        self.assertIn('<th colspan="2">Merged Header</th>', result.markdown)
        table_blocks = [block for block in result.page_ir["blocks"] if block["type"] == "table"]
        self.assertEqual(len(table_blocks), 1)
        self.assertEqual(table_blocks[0]["structural_block_type"], "html_table")
        self.assertEqual(result.diagnostics["rejected_tables"], [])
        document.close()

    def _document(self):
        document = pymupdf.open()
        for index in range(3):
            page = document.new_page(width=500, height=700)
            page.insert_text((50, 25), "Training Manual", fontsize=8)
            page.insert_text((50, 670), str(index + 1), fontsize=8)
            if index == 0:
                page.insert_text(
                    (50, 75), "Squat Mechanics", fontsize=18, fontname="hebo"
                )
                page.insert_text(
                    (50, 110), "Exact biomechanics text.", fontsize=11
                )
                xs = [50, 200, 350]
                ys = [180, 220, 260, 300]
                for x in xs:
                    page.draw_line((x, ys[0]), (x, ys[-1]))
                for y in ys:
                    page.draw_line((xs[0], y), (xs[-1], y))
                rows = [
                    ("Exercise", "Load"),
                    ("Squat", "100 kg"),
                    ("Press", "50 kg"),
                ]
                for row_index, row in enumerate(rows):
                    for column_index, value in enumerate(row):
                        page.insert_text(
                            (xs[column_index] + 5, ys[row_index] + 15),
                            value,
                            fontsize=9,
                        )
            else:
                page.insert_text(
                    (50, 75), f"Body page {index + 1} training material.", fontsize=11
                )
        document.set_toc([[1, "Squat Mechanics", 1]])
        return document

    def test_structured_native_render_is_deterministic_and_auditable(self):
        document = self._document()
        facts, _ = extract_document_facts(document)
        profile = build_document_profile(document)
        page = document[0]
        native_text = page.get_text("text", sort=True).strip()
        blocks = _text_blocks(page)

        first = render_native_page(
            page, native_text, blocks, page_facts(facts, 0), profile
        )
        second = render_native_page(
            page, native_text, blocks, page_facts(facts, 0), profile
        )
        self.assertEqual(first.markdown, second.markdown)
        self.assertEqual(first.page_ir, second.page_ir)
        self.assertIn("# **Squat Mechanics**", first.markdown)
        self.assertIn("| Exercise | Load |", first.markdown)
        self.assertNotIn("Training Manual", first.markdown)
        self.assertEqual(first.diagnostics["render_mode"], "structured")
        self.assertEqual(first.diagnostics["rendered_text_coverage"], 1.0)
        self.assertEqual(first.diagnostics["table_count"], 1)
        self.assertEqual(
            first.diagnostics["native_processor_config_sha256"],
            NATIVE_PROCESSOR_CONFIG_SHA256,
        )
        self.assertGreater(first.diagnostics["page_complexity"]["drawing_count"], 0)
        self.assertEqual(first.diagnostics["page_complexity"]["table_candidates"], 1)
        ignored = [
            block for block in first.page_ir["blocks"] if block["ignored_for_output"]
        ]
        self.assertTrue(
            any(block["ignore_reason"] == "repeated_running_header_or_footer" for block in ignored)
        )
        self.assertTrue(
            any(block["ignore_reason"] == "represented_by_structured_table" for block in ignored)
        )
        overlay = render_debug_overlay(page, first.page_ir)
        self.assertTrue(overlay.startswith(b"\xff\xd8"))
        document.close()

    def test_two_column_order_and_superscript_are_structured(self):
        document = pymupdf.open()
        page = document.new_page(width=500, height=700)

        def block(text, bbox, flags=0):
            return {
                "bbox": bbox,
                "number": 0,
                "lines": [
                    {
                        "bbox": bbox,
                        "dir": [1, 0],
                        "spans": [
                            {
                                "text": text,
                                "size": 11,
                                "flags": flags,
                                "bbox": bbox,
                            }
                        ],
                    }
                ],
            }

        blocks = [
            block("Left one", [40, 80, 210, 100]),
            block("Right one", [290, 80, 460, 100]),
            block("Left two", [40, 120, 210, 140]),
            block("Right two", [290, 120, 460, 140]),
            block("2", [40, 170, 50, 185], pymupdf.TEXT_FONT_SUPERSCRIPT),
        ]
        result = render_native_page(
            page,
            "Left one\nRight one\nLeft two\nRight two\n2",
            blocks,
            {"pdf_page_number": 1, "outline_path": []},
            {"body_font_size": 11.0, "heading_font_sizes": [], "repeated_marginalia": []},
        )
        self.assertEqual(result.diagnostics["column_count"], 2)
        self.assertLess(result.markdown.index("Left two"), result.markdown.index("Right one"))
        self.assertIn("<sup>2</sup>", result.markdown)
        document.close()

    def test_image_inventory_flags_visual_review_and_classifies_caption(self):
        document = pymupdf.open()
        page = document.new_page(width=400, height=500)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        pixmap.clear_with(0xDDDDDD)
        page.insert_image((50, 50, 350, 250), pixmap=pixmap)
        page.insert_text((70, 270), "Figure 1. Squat position", fontsize=9)
        native_text = page.get_text("text", sort=True).strip()
        result = render_native_page(
            page,
            native_text,
            _text_blocks(page),
            {"pdf_page_number": 1, "outline_path": []},
            {"body_font_size": 11.0, "heading_font_sizes": [], "repeated_marginalia": []},
        )
        types = [block["type"] for block in result.page_ir["blocks"]]
        self.assertIn("figure", types)
        self.assertIn("caption", types)
        self.assertTrue(result.diagnostics["needs_visual_parser"])
        self.assertIn("visual_content_requires_description", result.diagnostics["reason_codes"])
        self.assertGreater(result.diagnostics["page_complexity"]["image_density"], 0)
        document.close()

    def test_hebrew_text_is_not_latin_dehyphenated(self):
        document = pymupdf.open()
        page = document.new_page()
        blocks = [
            {
                "bbox": [72, 72, 300, 110],
                "number": 0,
                "lines": [
                    {
                        "bbox": [72, 72, 300, 85],
                        "dir": [1, 0],
                        "spans": [
                            {"text": "שלום-", "size": 11, "flags": 0, "bbox": [72, 72, 120, 85]}
                        ],
                    },
                    {
                        "bbox": [72, 90, 300, 103],
                        "dir": [1, 0],
                        "spans": [
                            {"text": "עולם", "size": 11, "flags": 0, "bbox": [72, 90, 120, 103]}
                        ],
                    },
                ],
            }
        ]
        result = render_native_page(
            page,
            "שלום-\nעולם",
            blocks,
            {"pdf_page_number": 1, "outline_path": []},
            {"body_font_size": 11.0, "heading_font_sizes": [], "repeated_marginalia": []},
        )
        self.assertIn("שלום- עולם", result.markdown)
        document.close()


if __name__ == "__main__":
    unittest.main()
