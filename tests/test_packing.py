import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rag_pdf_ingest.chunking import build_chunks
from rag_pdf_ingest.config import Settings
from rag_pdf_ingest.packing import (
    DeterministicFakeEmbedder,
    EmbeddingDocument,
    GeminiEmbedder,
    fts_match_query,
    pack_corpus,
    preflight_corpus,
)


class PackingTests(unittest.TestCase):
    def _corpus(self, root: Path) -> Settings:
        settings = Settings.load(root)
        book_id = "test-book-123"
        (settings.md_dir / book_id).mkdir(parents=True, exist_ok=True)
        (settings.metadata_dir / book_id).mkdir(parents=True, exist_ok=True)
        (settings.raw_dir / book_id).mkdir(parents=True, exist_ok=True)
        for page_number, body in ((1, "# Squat\n\nUse a 5×5 protocol."), (2, "Protein: 1.6 g/kg.")):
            (settings.md_dir / book_id / f"page-{page_number:04d}.md").write_text(
                f"---\nstatus: complete\n---\n\n{body}\n", encoding="utf-8"
            )
            (settings.raw_dir / book_id / f"page-{page_number:04d}.ir.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (
                settings.raw_dir
                / book_id
                / f"page-{page_number:04d}.native-diagnostics.json"
            ).write_text("{}\n", encoding="utf-8")
        pages = [
            {
                "schema_version": 3,
                "pdf_page_number": page_number,
                "printed_page_label": None,
                "status": "complete",
                "citation": f"Test.pdf, PDF p. {page_number}",
                "outline_path": [],
                "outline_section": None,
                "needs_visual_parser": False,
                "reason_codes": [],
                "native_ir_path": f"raw/test-book-123/page-{page_number:04d}.ir.json",
                "native_diagnostics_path": f"raw/test-book-123/page-{page_number:04d}.native-diagnostics.json",
                "native_processor_config_sha256": "f" * 64,
                "native_text_quality": {"usable_as_reference": True},
                "page_complexity": {"table_candidates": 0},
                "parser_backend": "gemini",
                "parser_model": "gemini-test",
                "conversion_route": "visual_llm",
                "llm_called": True,
                "native_text_characters": 20,
                "markdown_sha256": str(page_number) * 64,
            }
            for page_number in (1, 2)
        ]
        (settings.metadata_dir / book_id / "pages.jsonl").write_text(
            "".join(json.dumps(page) + "\n" for page in pages), encoding="utf-8"
        )
        build_chunks(
            pages=[
                (1, "# Squat\n\nUse a 5×5 protocol."),
                (2, "Protein: 1.6 g/kg."),
            ],
            output_dir=settings.chunks_dir / book_id,
            source_name="Test.pdf",
            book_id=book_id,
            target_tokens=100,
            max_tokens=200,
            overlap_tokens=10,
            page_quality={1: "complete", 2: "complete"},
            bibliography={
                "title": "Starting Strength",
                "title_source": "sidecar",
                "authors": ["Mark Rippetoe"],
                "authors_source": "sidecar",
                "edition": "3rd",
                "publication_year": "2011",
                "language": "en",
            },
        )
        manifest = {
            "schema_version": 3,
            "book_id": book_id,
            "source_file": "Test.pdf",
            "source_sha256": "a" * 64,
            "page_count": 2,
            "status": "complete",
            "failed_pages": [],
            "fallback_pages": [],
            "license": {"status": "authorized"},
            "bibliography": {
                "title": "Starting Strength",
                "title_source": "sidecar",
                "authors": ["Mark Rippetoe"],
                "authors_source": "sidecar",
                "edition": "3rd",
                "publication_year": "2011",
                "language": "en",
            },
        }
        (settings.metadata_dir / book_id / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return settings

    def test_preflight_and_fake_pack_create_auditable_hybrid_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._corpus(root)
            report = preflight_corpus(settings)
            self.assertFalse(report.errors)
            output = settings.corpus_dir / "test.sqlite3"
            result = pack_corpus(
                settings,
                report,
                output_path=output,
                embedder=DeterministicFakeEmbedder(16),
                batch_size=2,
            )
            self.assertEqual(result["cache_hits"], 0)
            self.assertEqual(result["cache_misses"], report.chunk_count)
            self.assertEqual(result["schema_version"], 5)
            self.assertEqual(result["artifact_mode"], "development")
            self.assertFalse(result["semantic_vectors"])
            self.assertFalse(result["production_ready"])
            with closing(sqlite3.connect(output)) as connection:
                chunk_count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
                page_count = connection.execute("SELECT count(*) FROM pages").fetchone()[0]
                vector = connection.execute(
                    "SELECT row_count, dimensions, length(vectors_f32le) FROM embedding_matrix"
                ).fetchone()
                fts_hits = connection.execute(
                    "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                    (fts_match_query("5×5 squat"),),
                ).fetchone()[0]
                filename_hits = connection.execute(
                    "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                    (fts_match_query("Test.pdf"),),
                ).fetchone()[0]
                weighted_score = connection.execute(
                    "SELECT bm25(chunks_fts, 10.0, 3.0) FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? LIMIT 1",
                    (fts_match_query("squat protocol"),),
                ).fetchone()
                meta = dict(connection.execute("SELECT key, value FROM meta"))
                markdown = connection.execute(
                    "SELECT markdown FROM pages WHERE page_number = 1"
                ).fetchone()[0]
                book_identity = connection.execute(
                    "SELECT title, title_source, authors_json, edition, "
                    "publication_year, language FROM books"
                ).fetchone()
                chunk_identity = connection.execute(
                    "SELECT book_title, title_source, authors_json FROM chunks LIMIT 1"
                ).fetchone()
                structure = connection.execute(
                    "SELECT structural_block_type, structural_block_types_json, "
                    "oversized, oversized_reason FROM chunks LIMIT 1"
                ).fetchone()
                page_route = connection.execute(
                    "SELECT parser_backend, parser_model, conversion_route, llm_called "
                    "FROM pages WHERE page_number = 1"
                ).fetchone()
                chunk_route = connection.execute(
                    "SELECT native_structured_pages_json, has_native_structured "
                    "FROM chunks LIMIT 1"
                ).fetchone()
            self.assertEqual(chunk_count, report.chunk_count)
            self.assertEqual(page_count, 2)
            self.assertEqual(vector, (report.chunk_count, 16, report.chunk_count * 16 * 4))
            self.assertGreater(fts_hits, 0)
            self.assertEqual(filename_hits, 0)
            self.assertIsNotNone(weighted_score)
            self.assertEqual(meta["fts_columns"], "body,section")
            self.assertEqual(meta["fts_bm25_weights"], "10.0,3.0")
            self.assertEqual(meta["production_ready"], "false")
            self.assertEqual(meta["schema_version"], "5")
            self.assertEqual(
                book_identity,
                ("Starting Strength", "sidecar", '["Mark Rippetoe"]', "3rd", "2011", "en"),
            )
            self.assertEqual(
                chunk_identity,
                ("Starting Strength", "sidecar", '["Mark Rippetoe"]'),
            )
            self.assertEqual(structure, (None, "[]", 0, None))
            self.assertEqual(page_route, ("gemini", "gemini-test", "visual_llm", 1))
            self.assertEqual(chunk_route, ("[]", 0))
            self.assertIn("5×5 protocol", markdown)

            second = pack_corpus(
                settings,
                report,
                output_path=output,
                embedder=DeterministicFakeEmbedder(16),
                batch_size=2,
            )
            self.assertEqual(second["cache_hits"], report.chunk_count)
            self.assertEqual(second["cache_misses"], 0)

            # Bibliography is part of the exact embedding input but not the
            # stable content-addressed chunk ID. A title correction must reuse
            # IDs while invalidating the corresponding vector cache entries.
            book_id = "test-book-123"
            manifest_path = settings.metadata_dir / book_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["bibliography"]["title"] = "Starting Strength: Basic Barbell Training"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            chunks_path = settings.chunks_dir / book_id / "chunks.jsonl"
            changed_chunks = [
                json.loads(line)
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
            ]
            original_ids = [chunk["chunk_id"] for chunk in changed_chunks]
            for chunk in changed_chunks:
                chunk["book_title"] = manifest["bibliography"]["title"]
            chunks_path.write_text(
                "".join(json.dumps(chunk) + "\n" for chunk in changed_chunks),
                encoding="utf-8",
            )
            changed_report = preflight_corpus(settings)
            self.assertFalse(changed_report.errors)
            changed = pack_corpus(
                settings,
                changed_report,
                output_path=output,
                embedder=DeterministicFakeEmbedder(16),
                batch_size=2,
            )
            self.assertEqual(
                [chunk["chunk_id"] for chunk in changed_report.books[0].chunks],
                original_ids,
            )
            self.assertEqual(changed["cache_hits"], 0)
            self.assertEqual(changed["cache_misses"], changed_report.chunk_count)

    def test_shared_fts_query_fixtures(self):
        fixtures = json.loads(
            (Path(__file__).parents[1] / "fixtures" / "fts-query-normalization.json").read_text(
                encoding="utf-8"
            )
        )
        for fixture in fixtures:
            self.assertEqual(fts_match_query(fixture["input"]), fixture["output"])

    def test_preflight_rejects_atomic_table_above_embedding_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._corpus(root)
            book_id = "test-book-123"
            manifest = json.loads(
                (settings.metadata_dir / book_id / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            huge_table = "<table>\n" + "\n".join(
                f"<tr><td>Exercise repetition {index}</td><td>{index} kilograms</td></tr>"
                for index in range(3000)
            ) + "\n</table>"
            build_chunks(
                pages=[(1, huge_table)],
                output_dir=settings.chunks_dir / book_id,
                source_name="Test.pdf",
                book_id=book_id,
                target_tokens=750,
                max_tokens=900,
                overlap_tokens=120,
                page_quality={1: "complete"},
                bibliography=manifest["bibliography"],
            )

            report = preflight_corpus(settings)
            self.assertTrue(
                any("8,192-token input limit" in error for error in report.errors),
                report.errors,
            )

    def test_gemini_embedder_keeps_each_document_as_a_content(self):
        class Part:
            @staticmethod
            def from_text(*, text):
                return {"text": text}

        class Content:
            def __init__(self, *, role, parts):
                self.role = role
                self.parts = parts

        class Config:
            def __init__(self, **values):
                self.values = values

        Types = type(
            "Types",
            (),
            {"Part": Part, "Content": Content, "EmbedContentConfig": Config},
        )

        class Embedding:
            def __init__(self, values):
                self.values = values

        calls = []

        class Models:
            def embed_content(self, **kwargs):
                calls.append(kwargs)
                return type(
                    "Response",
                    (),
                    {"embeddings": [Embedding([1.0, 0.0]), Embedding([0.0, 1.0])]},
                )()

        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        embedder._types = Types
        embedder._client = type("Client", (), {"models": Models()})()
        embedder.model = "gemini-embedding-2"
        embedder.dimensions = 2
        embedder.max_retries = 1
        documents = [
            EmbeddingDocument("a", "first", "1" * 64),
            EmbeddingDocument("b", "second", "2" * 64),
        ]
        vectors = embedder.embed(documents)
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(len(calls[0]["contents"]), 2)
        self.assertTrue(all(isinstance(item, Content) for item in calls[0]["contents"]))
        self.assertNotIn("auto_truncate", calls[0]["config"].values)

    def test_gemini_configuration_errors_are_not_retried(self):
        class Part:
            @staticmethod
            def from_text(*, text):
                return {"text": text}

        class Content:
            def __init__(self, **values):
                self.values = values

        class Config:
            def __init__(self, **values):
                self.values = values

        types = type(
            "Types",
            (),
            {"Part": Part, "Content": Content, "EmbedContentConfig": Config},
        )
        calls = []

        class Models:
            def embed_content(self, **kwargs):
                calls.append(kwargs)
                raise ValueError("unsupported configuration")

        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        embedder._types = types
        embedder._client = type("Client", (), {"models": Models()})()
        embedder.model = "gemini-embedding-2"
        embedder.dimensions = 2
        embedder.max_retries = 5
        with self.assertRaisesRegex(ValueError, "unsupported configuration"):
            embedder.embed([EmbeddingDocument("a", "first", "1" * 64)])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
