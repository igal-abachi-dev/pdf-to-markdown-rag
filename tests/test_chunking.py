import json
import tempfile
import unittest
from pathlib import Path

from rag_pdf_ingest.chunking import (
    Block,
    _blocks_from_pages,
    _make_chunks,
    _split_long_text,
    approximate_tokens,
    build_chunks,
)


class ChunkingTests(unittest.TestCase):
    def test_printed_labels_outline_and_visual_review_propagate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_chunks(
                pages=[(18, "Visible page body.")],
                output_dir=output,
                source_name="Book.pdf",
                book_id="book-123",
                target_tokens=200,
                max_tokens=300,
                overlap_tokens=20,
                page_quality={18: "native_text_fallback"},
                page_context={
                    18: {
                        "printed_page_label": "3",
                        "outline_path": ["Chapter 2", "The Squat"],
                        "outline_section": "The Squat",
                        "needs_visual_parser": True,
                        "reason_codes": ["visual_content_requires_description"],
                    }
                },
            )
            record = json.loads((output / "chunks.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["printed_page_start"], "3")
            self.assertEqual(record["outline_path"], ["Chapter 2", "The Squat"])
            self.assertEqual(record["section"], "The Squat")
            self.assertEqual(record["visual_review_pages"], [18])
            self.assertIn("printed p. 3", record["citation"])

    def test_chunks_respect_maximum_before_page_markers(self):
        blocks = [Block("word " * 60, page, "Section") for page in range(1, 7)]
        chunks = _make_chunks(blocks, target_tokens=120, max_tokens=150, overlap_tokens=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(
                sum(approximate_tokens(block.text) for block in chunk), 150
            )

    def test_export_has_obsidian_metadata_and_each_page_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            result = build_chunks(
                pages=[
                    (1, "# Movement\n\nFirst page text."),
                    (2, "Second page text with 10 kg."),
                ],
                output_dir=output,
                source_name="Biomechanics.pdf",
                book_id="biomechanics-123",
                target_tokens=200,
                max_tokens=300,
                overlap_tokens=20,
                page_quality={1: "complete", 2: "native_text_fallback"},
            )
            self.assertEqual(result["chunk_count"], 1)
            chunk_path = next(output.glob("chunk-*.md"))
            markdown = chunk_path.read_text(encoding="utf-8")
            self.assertIn("page_start: 1", markdown)
            self.assertIn("page_end: 2", markdown)
            self.assertIn("[p. 1]", markdown)
            self.assertIn("[p. 2]", markdown)
            self.assertIn("[[md/biomechanics-123/page-0001", markdown)
            record = json.loads((output / "chunks.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["page_start"], 1)
            self.assertEqual(record["page_end"], 2)
            self.assertEqual(record["schema_version"], 5)
            self.assertEqual(record["book_title"], "Biomechanics")
            self.assertEqual(record["title_source"], "source_file")
            self.assertEqual(record["fallback_pages"], [2])
            self.assertTrue(record["has_native_text_fallback"])
            self.assertEqual(
                record["source_quality"], "mixed_gemini_and_native_text_fallback"
            )
            self.assertIsNone(record["structural_block_type"])
            self.assertEqual(record["structural_block_types"], [])
            self.assertFalse(record["oversized"])
            self.assertIsNone(record["oversized_reason"])

    def test_structural_blocks_are_extracted_atomically(self):
        html_table = (
            '<table class="loads">\n'
            '  <tr><th rowspan="2">Exercise</th><th>Load</th></tr>\n\n'
            '  <tr><td>100 kg</td></tr>\n'
            '</table>'
        )
        fenced = "```text\nfirst line\n\nsecond line\n```"
        markdown_table = "| Exercise | Load |\n|---|---:|\n| Squat | 100 kg |"
        source = (
            "# Programming\n\nBefore.\n\n"
            + html_table
            + "\n\n"
            + fenced
            + "\n\n"
            + markdown_table
            + "\n\nAfter."
        )

        blocks = _blocks_from_pages([(7, source)])
        structural = [
            (block.structural_block_type, block.text)
            for block in blocks
            if block.structural_block_type
        ]
        self.assertEqual(
            [structural_type for structural_type, _ in structural],
            ["html_table", "fenced_block", "markdown_table"],
        )
        self.assertEqual(structural[0][1], html_table)
        self.assertEqual(structural[1][1], fenced)
        self.assertEqual(structural[2][1], markdown_table)
        self.assertIn('rowspan="2"', structural[0][1])

    def test_oversized_html_table_is_preserved_and_explained(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            table = "<table>\n" + "\n\n".join(
                f"<tr><td>Exercise {index}</td><td>{index} kg</td></tr>"
                for index in range(100)
            ) + "\n</table>"
            build_chunks(
                pages=[(1, table)],
                output_dir=output,
                source_name="Tables.pdf",
                book_id="tables-123",
                target_tokens=40,
                max_tokens=50,
                overlap_tokens=5,
            )
            record = json.loads((output / "chunks.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["body"].count("<table>"), 1)
            self.assertEqual(record["body"].count("</table>"), 1)
            self.assertIn(table, record["body"])
            self.assertEqual(record["structural_block_type"], "html_table")
            self.assertEqual(record["structural_block_types"], ["html_table"])
            self.assertTrue(record["oversized"])
            self.assertEqual(
                record["oversized_reason"],
                "atomic_html_table_exceeds_chunk_max",
            )

    def test_oversized_markdown_table_is_not_corrupted(self):
        table = "| Exercise | Load |\n|---|---|\n" + "\n".join(
            f"| Squat {index} | {index} kg |" for index in range(100)
        )
        self.assertEqual(_split_long_text(table, max_tokens=50), [table])

    def test_failed_page_gap_is_not_hidden_inside_range(self):
        blocks = [
            Block("Page one.", 1, "Section"),
            Block("Page three.", 3, "Section"),
        ]
        chunks = _make_chunks(blocks, target_tokens=200, max_tokens=300, overlap_tokens=20)
        self.assertEqual([[block.page for block in chunk] for chunk in chunks], [[1], [3]])

    def test_content_addressed_id_is_stable_when_chunk_index_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "source_name": "Book.pdf",
                "book_id": "book-123",
                "target_tokens": 200,
                "max_tokens": 300,
                "overlap_tokens": 20,
            }
            build_chunks(pages=[(3, "Stable later content.")], output_dir=root / "a", **common)
            build_chunks(
                pages=[(1, "Earlier content."), (3, "Stable later content.")],
                output_dir=root / "b",
                **common,
            )
            first_records = [
                json.loads(line)
                for line in (root / "a" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            second_records = [
                json.loads(line)
                for line in (root / "b" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            later = next(record for record in second_records if record["page_start"] == 3)
            self.assertEqual(first_records[0]["chunk_id"], later["chunk_id"])
            self.assertNotEqual(first_records[0]["chunk_index"], later["chunk_index"])

    def test_content_addressed_id_is_stable_when_bibliography_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "pages": [(1, "Stable content.")],
                "source_name": "Filename - Copy.pdf",
                "book_id": "book-123",
                "target_tokens": 200,
                "max_tokens": 300,
                "overlap_tokens": 20,
            }
            build_chunks(output_dir=root / "fallback", **common)
            build_chunks(
                output_dir=root / "explicit",
                bibliography={
                    "title": "Canonical Title",
                    "title_source": "sidecar",
                    "authors": ["Author Name"],
                    "edition": "3rd",
                    "publication_year": "2024",
                    "language": "en",
                },
                **common,
            )
            fallback = json.loads((root / "fallback" / "chunks.jsonl").read_text())
            explicit = json.loads((root / "explicit" / "chunks.jsonl").read_text())
            self.assertEqual(fallback["chunk_id"], explicit["chunk_id"])
            self.assertEqual(explicit["book_title"], "Canonical Title")
            self.assertIn("Author Name, Canonical Title, 3rd ed., 2024", explicit["citation"])

    def test_zero_length_block_is_ignored(self):
        chunks = _make_chunks(
            [Block("", 1, ""), Block("Useful text.", 1, "")],
            target_tokens=100,
            max_tokens=200,
            overlap_tokens=20,
        )
        self.assertEqual([[block.text for block in chunk] for chunk in chunks], [["Useful text."]])


if __name__ == "__main__":
    unittest.main()
