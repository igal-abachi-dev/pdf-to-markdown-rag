from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .bibliography import resolve_bibliography
from .chunking import approximate_tokens, chunk_content_sha256
from .config import Settings
from .provenance import DEFAULT_LICENSE_STATUS
from .utils import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_text,
    utc_now,
)


EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
EMBEDDING_INPUT_TOKEN_LIMIT = 8192
LICENSED_STATUSES = {
    "licensed",
    "authorized",
    "owned",
    "public_domain",
    "public-domain",
}
FTS_MAX_QUERY_TERMS = 8
FTS_IGNORED_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "body",
    "book",
    "by",
    "citation",
    "file",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "page",
    "path",
    "pdf",
    "source",
    "that",
    "the",
    "this",
    "title",
    "to",
    "was",
    "were",
    "with",
    "את",
    "של",
    "על",
    "עם",
    "זה",
    "זו",
    "או",
}


@dataclass(frozen=True)
class CorpusBook:
    manifest_path: Path
    manifest: dict[str, Any]
    chunks: tuple[dict[str, Any], ...]
    pages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EmbeddingDocument:
    chunk_id: str
    text: str
    embedding_input_sha256: str


@dataclass(frozen=True)
class PreflightReport:
    books: tuple[CorpusBook, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def chunk_count(self) -> int:
        return sum(len(book.chunks) for book in self.books)

    @property
    def page_count(self) -> int:
        return sum(len(book.pages) for book in self.books)


class Embedder(Protocol):
    model: str
    dimensions: int
    task_type: str
    semantic_vectors: bool

    def embed(self, documents: Sequence[EmbeddingDocument]) -> list[list[float]]: ...

    def close(self) -> None: ...


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _conservative_token_estimate(text: str) -> int:
    # Gemini Developer API does not expose the Enterprise-only auto_truncate
    # control. This estimate is deliberately conservative for non-Latin scripts
    # and catches pathological structural blocks before an embedding request.
    non_ascii = sum(1 for char in text if ord(char) > 127 and not char.isspace())
    return max(
        approximate_tokens(text),
        math.ceil(len(text.encode("utf-8")) / 4),
        non_ascii,
    )


def build_embedding_text(chunk: dict[str, Any]) -> str:
    title = str(chunk["book_title"])
    section = str(chunk.get("section") or "")
    lines = [f"Book: {title}"]
    authors = [str(author) for author in chunk.get("authors", []) if str(author).strip()]
    if authors:
        lines.append(f"Authors: {', '.join(authors)}")
    if chunk.get("edition"):
        lines.append(f"Edition: {chunk['edition']}")
    if chunk.get("publication_year"):
        lines.append(f"Publication year: {chunk['publication_year']}")
    if chunk.get("language"):
        lines.append(f"Language: {chunk['language']}")
    lines.append(f"Section: {section}")
    return ("\n".join(lines) + f"\n\n{chunk['body']}").strip()


def fts_query_terms(raw: str, *, max_terms: int = FTS_MAX_QUERY_TERMS) -> list[str]:
    """Convert user text into a safe, minimal FTS5 OR query.

    This is intentionally bounded. It accepts user-question text, not serialized
    assessment objects or chunk JSON. Multiplication forms are compacted, common
    prose/schema words are removed, and standalone digits are omitted because
    they produce nearly corpus-wide matches.
    """
    if max_terms <= 0:
        return []
    normalized = unicodedata.normalize("NFKC", raw).casefold().replace("×", "x")
    normalized = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1x\2", normalized)
    normalized = re.sub(r"\brpe\s*[-:]?\s*(\d+)\b", r"rpe\1", normalized)
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    selected: list[str] = []
    for token in tokens:
        if token in FTS_IGNORED_TERMS or (len(token) == 1 and token.isdigit()):
            continue
        if token not in selected:
            selected.append(token)
        if len(selected) >= max_terms:
            break
    return selected


def fts_match_query(raw: str, *, max_terms: int = FTS_MAX_QUERY_TERMS) -> str | None:
    terms = fts_query_terms(raw, max_terms=max_terms)
    if not terms:
        return None
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def preflight_corpus(
    settings: Settings,
    *,
    book_ids: set[str] | None = None,
    include_unspecified: bool = False,
) -> PreflightReport:
    errors: list[str] = []
    warnings: list[str] = []
    books: list[CorpusBook] = []
    seen_chunk_ids: set[str] = set()
    manifests = sorted(settings.metadata_dir.glob("*/manifest.json"))
    found_ids: set[str] = set()

    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read manifest {manifest_path}: {exc}")
            continue
        book_id = str(manifest.get("book_id") or manifest_path.parent.name)
        if book_ids is not None and book_id not in book_ids:
            continue
        found_ids.add(book_id)
        status = str(manifest.get("status", "unknown"))
        if status not in {"complete", "partial"}:
            warnings.append(f"{book_id}: skipped non-final status {status}")
            continue
        manifest_bibliography = resolve_bibliography(
            manifest.get("license"),
            str(manifest.get("source_file") or f"{book_id}.pdf"),
            existing=manifest.get("bibliography"),
        )
        if manifest.get("bibliography") != manifest_bibliography:
            errors.append(
                f"{book_id}: manifest bibliography is missing or stale; "
                f"run rebuild-book {book_id}"
            )
            continue

        license_status = str(
            manifest.get("license", {}).get("status", DEFAULT_LICENSE_STATUS)
        ).strip()
        if license_status not in LICENSED_STATUSES:
            if not (include_unspecified and license_status == "unspecified"):
                errors.append(
                    f"{book_id}: license status {license_status!r} is not packable; "
                    "record a licensed/authorized/owned/public_domain grant or use "
                    "--include-unspecified for local development only"
                )
                continue
            warnings.append(f"{book_id}: including unspecified license for development")

        failed_pages = [int(value) for value in manifest.get("failed_pages", [])]
        if failed_pages:
            errors.append(
                f"{book_id}: failed pages are retrieval holes: {failed_pages}; repair them first"
            )
            continue
        fallback_pages = [int(value) for value in manifest.get("fallback_pages", [])]
        if fallback_pages:
            warnings.append(
                f"{book_id}: native-text fallback pages require visual-layout review: "
                f"{fallback_pages}"
            )

        chunks_path = settings.chunks_dir / book_id / "chunks.jsonl"
        pages_path = settings.metadata_dir / book_id / "pages.jsonl"
        if not chunks_path.exists() or not pages_path.exists():
            errors.append(
                f"{book_id}: missing chunks.jsonl or pages.jsonl; run rebuild-book {book_id}"
            )
            continue
        try:
            chunks = _read_jsonl(chunks_path)
            pages = _read_jsonl(pages_path)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if not chunks:
            errors.append(f"{book_id}: chunks.jsonl is empty")
            continue
        if len(pages) != int(manifest.get("page_count", 0)):
            errors.append(
                f"{book_id}: pages.jsonl has {len(pages)} records but manifest expects "
                f"{manifest.get('page_count')}"
            )
            continue

        book_errors = False
        for page in pages:
            page_number = int(page.get("pdf_page_number", 0) or 0)
            if page.get("schema_version") != 3:
                errors.append(
                    f"{book_id}: page {page_number} uses schema "
                    f"{page.get('schema_version')!r}; run rebuild-book {book_id}"
                )
                book_errors = True
                continue
            required_page_fields = (
                "printed_page_label",
                "outline_path",
                "outline_section",
                "needs_visual_parser",
                "reason_codes",
                "native_ir_path",
                "native_diagnostics_path",
                "native_processor_config_sha256",
                "native_text_quality",
                "page_complexity",
                "parser_backend",
                "parser_model",
                "conversion_route",
                "llm_called",
            )
            missing_page_fields = [
                field for field in required_page_fields if field not in page
            ]
            if missing_page_fields:
                errors.append(
                    f"{book_id}: page {page_number} lacks {missing_page_fields}; "
                    f"run rebuild-book {book_id} --refresh-pdf-facts"
                )
                book_errors = True
                continue
            for artifact_field in ("native_ir_path", "native_diagnostics_path"):
                artifact_value = page.get(artifact_field)
                artifact_path = (
                    settings.root / str(artifact_value)
                    if artifact_value
                    else None
                )
                if artifact_path is None or not artifact_path.is_file():
                    errors.append(
                        f"{book_id}: page {page_number} has missing {artifact_field} artifact; "
                        f"run rebuild-book {book_id} --refresh-native-fallbacks"
                    )
                    book_errors = True
            config_hash = str(page.get("native_processor_config_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
                errors.append(
                    f"{book_id}: page {page_number} has an invalid native processor config hash"
                )
                book_errors = True
            if not isinstance(page.get("native_text_quality"), dict) or not isinstance(
                page.get("page_complexity"), dict
            ):
                errors.append(
                    f"{book_id}: page {page_number} has invalid native quality/complexity metadata"
                )
                book_errors = True
            conversion_route = str(page.get("conversion_route") or "")
            allowed_routes = {
                "native_structured",
                "visual_llm",
                "native_fallback_after_llm",
                "visual_llm_failed",
                "visual_render_failed_before_llm",
            }
            if conversion_route not in allowed_routes:
                errors.append(
                    f"{book_id}: page {page_number} has invalid conversion_route "
                    f"{conversion_route!r}"
                )
                book_errors = True
            if conversion_route == "native_structured" and (
                bool(page.get("llm_called"))
                or bool(page.get("needs_visual_parser"))
                or page.get("parser_backend") != "pymupdf_native"
            ):
                errors.append(
                    f"{book_id}: page {page_number} violates native routing invariants"
                )
                book_errors = True

        page_statuses = {
            int(page.get("pdf_page_number", 0)): str(page.get("status", "unknown"))
            for page in pages
        }
        actual_failed = sorted(
            page for page, page_status in page_statuses.items() if page_status == "failed"
        )
        if actual_failed:
            errors.append(f"{book_id}: pages.jsonl contains failed pages: {actual_failed}")
            continue
        for page_number in range(1, int(manifest["page_count"]) + 1):
            markdown_path = settings.md_dir / book_id / f"page-{page_number:04d}.md"
            if not markdown_path.exists():
                errors.append(f"{book_id}: missing exact page Markdown {markdown_path.name}")
                book_errors = True

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk.get("schema_version") != 5:
                errors.append(
                    f"{book_id}: chunk {chunk_id or '<unknown>'} uses schema "
                    f"{chunk.get('schema_version')!r}; run rebuild-book {book_id}"
                )
                book_errors = True
            required = (
                "chunk_id",
                "book_id",
                "source_file",
                "book_title",
                "title_source",
                "authors",
                "authors_source",
                "edition",
                "publication_year",
                "language",
                "page_start",
                "page_end",
                "printed_page_start",
                "printed_page_end",
                "citation",
                "body",
                "content_sha256",
                "source_quality",
                "included_pages",
                "outline_section",
                "markdown_section",
                "outline_path",
                "visual_review_pages",
                "native_structured_pages",
                "has_native_structured",
                "reason_codes",
                "structural_block_type",
                "structural_block_types",
                "oversized",
                "oversized_reason",
            )
            missing = [field for field in required if field not in chunk]
            if missing:
                errors.append(
                    f"{book_id}: chunk {chunk_id or '<unknown>'} lacks {missing}; "
                    f"run rebuild-book {book_id}"
                )
                book_errors = True
                continue
            if chunk_id in seen_chunk_ids:
                errors.append(f"Duplicate chunk_id across corpus: {chunk_id}")
                book_errors = True
            seen_chunk_ids.add(chunk_id)
            if str(chunk.get("book_id")) != book_id:
                errors.append(f"{book_id}: chunk belongs to {chunk.get('book_id')}")
                book_errors = True
            chunk_bibliography = {
                "title": chunk.get("book_title"),
                "title_source": chunk.get("title_source"),
                "authors": chunk.get("authors"),
                "authors_source": chunk.get("authors_source"),
                "edition": chunk.get("edition"),
                "publication_year": chunk.get("publication_year"),
                "language": chunk.get("language"),
            }
            if chunk_bibliography != manifest_bibliography:
                errors.append(
                    f"{book_id}: chunk bibliography differs from manifest for "
                    f"{chunk_id}; run rebuild-book {book_id}"
                )
                book_errors = True
            body = str(chunk.get("body") or "").strip()
            if not body:
                errors.append(f"{book_id}: empty body for {chunk_id}")
                book_errors = True
                continue
            expected_id = f"{book_id}-sha256-{chunk['content_sha256']}"
            if chunk_id != expected_id:
                errors.append(f"{book_id}: content-address mismatch for {chunk_id}")
                book_errors = True
            recomputed_hash = chunk_content_sha256(
                book_id=book_id,
                page_start=int(chunk["page_start"]),
                page_end=int(chunk["page_end"]),
                section=str(chunk.get("section") or ""),
                body=body,
            )
            if recomputed_hash != str(chunk["content_sha256"]):
                errors.append(f"{book_id}: body/hash mismatch for {chunk_id}")
                book_errors = True
            if str(chunk.get("license_status")) != license_status:
                errors.append(
                    f"{book_id}: chunk license {chunk.get('license_status')!r} differs "
                    f"from manifest {license_status!r}; run rebuild-book {book_id}"
                )
                book_errors = True
            allowed_structural_types = {
                None,
                "html_table",
                "markdown_table",
                "fenced_block",
                "mixed",
            }
            if chunk.get("structural_block_type") not in allowed_structural_types:
                errors.append(
                    f"{book_id}: invalid structural_block_type for {chunk_id}: "
                    f"{chunk.get('structural_block_type')!r}"
                )
                book_errors = True
            structural_types = chunk.get("structural_block_types")
            if not isinstance(structural_types, list) or any(
                value not in allowed_structural_types - {None, "mixed"}
                for value in structural_types
            ):
                errors.append(
                    f"{book_id}: invalid structural_block_types for {chunk_id}: "
                    f"{structural_types!r}"
                )
                book_errors = True
            else:
                expected_structural_type = (
                    None
                    if not structural_types
                    else structural_types[0]
                    if len(structural_types) == 1
                    else "mixed"
                )
                if chunk.get("structural_block_type") != expected_structural_type:
                    errors.append(
                        f"{book_id}: structural metadata is inconsistent for {chunk_id}"
                    )
                    book_errors = True
            oversized = bool(chunk.get("oversized"))
            oversized_reason = chunk.get("oversized_reason")
            if oversized != bool(oversized_reason):
                errors.append(
                    f"{book_id}: {chunk_id} must record oversized_reason exactly "
                    "when oversized is true; run rebuild-book"
                )
                book_errors = True
            estimate = _conservative_token_estimate(build_embedding_text(chunk))
            if estimate > EMBEDDING_INPUT_TOKEN_LIMIT:
                errors.append(
                    f"{book_id}: {chunk_id} is estimated at {estimate} tokens, above "
                    f"Gemini Embedding 2's {EMBEDDING_INPUT_TOKEN_LIMIT:,}-token input limit"
                )
                book_errors = True
            chunk_file = settings.chunks_dir / book_id / str(chunk.get("path", ""))
            if not chunk_file.exists():
                errors.append(f"{book_id}: missing chunk artifact {chunk_file.name}")
                book_errors = True
        if not book_errors:
            books.append(
                CorpusBook(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    chunks=tuple(chunks),
                    pages=tuple(pages),
                )
            )

    if book_ids is not None:
        for missing_id in sorted(book_ids - found_ids):
            errors.append(f"Unknown book_id: {missing_id}")
    if not books and not errors:
        errors.append("No finalized books are available to pack")
    return PreflightReport(tuple(books), tuple(errors), tuple(warnings))


class DeterministicFakeEmbedder:
    model = "deterministic-fake-embedding-v1"
    task_type = EMBEDDING_TASK_TYPE
    semantic_vectors = False

    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions

    def embed(self, documents: Sequence[EmbeddingDocument]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for document in documents:
            seed = hashlib.sha256(document.text.encode("utf-8")).digest()
            raw: list[float] = []
            counter = 0
            while len(raw) < self.dimensions:
                digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
                raw.extend((byte - 127.5) / 127.5 for byte in digest)
                counter += 1
            raw = raw[: self.dimensions]
            norm = math.sqrt(sum(value * value for value in raw)) or 1.0
            vectors.append([value / norm for value in raw])
        return vectors

    def close(self) -> None:
        return None


class PermanentEmbeddingError(RuntimeError):
    """A response-shape/configuration failure that must never consume retries."""


def _embedding_status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(408|409|429|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable_embedding_error(exc: Exception) -> bool:
    if isinstance(exc, (ValueError, TypeError, PermanentEmbeddingError)):
        return False
    status_code = _embedding_status_code(exc)
    if status_code is not None:
        return status_code in {408, 409, 429} or status_code >= 500
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


class GeminiEmbedder:
    task_type = EMBEDDING_TASK_TYPE
    semantic_vectors = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        max_retries: int = 5,
    ):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for real corpus embeddings")
        from google import genai

        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries

    def embed(self, documents: Sequence[EmbeddingDocument]) -> list[list[float]]:
        if not documents:
            return []
        contents = [
            self._types.Content(
                role="user",
                parts=[self._types.Part.from_text(text=document.text)],
            )
            for document in documents
        ]
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=self._types.EmbedContentConfig(
                        task_type=self.task_type,
                        output_dimensionality=self.dimensions,
                    ),
                )
                embeddings = list(response.embeddings or [])
                if len(embeddings) != len(documents):
                    raise PermanentEmbeddingError(
                        "Embedding response cardinality mismatch: requested "
                        f"{len(documents)}, received {len(embeddings)}. Each input must "
                        "remain a separate Content object."
                    )
                vectors = [list(embedding.values or []) for embedding in embeddings]
                for vector in vectors:
                    if len(vector) != self.dimensions:
                        raise PermanentEmbeddingError(
                            f"Embedding dimension mismatch: expected {self.dimensions}, "
                            f"received {len(vector)}"
                        )
                return vectors
            except PermanentEmbeddingError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries or not _is_retryable_embedding_error(exc):
                    break
                time.sleep(min(2 ** (attempt - 1), 30))
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class EmbeddingCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(
        *, model: str, dimensions: int, task_type: str, embedding_input_sha256: str
    ) -> str:
        canonical = json.dumps(
            {
                "model": model,
                "dimensions": dimensions,
                "task_type": task_type,
                # Keep the serialized field name for compatibility with cache
                # entries written before the internal name was clarified.
                "content_sha256": embedding_input_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256_text(canonical)

    def load(self, key: str, dimensions: int) -> list[float] | None:
        path = self.root / f"{key}.f32"
        metadata_path = self.root / f"{key}.json"
        if not path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data = path.read_bytes()
            if metadata.get("sha256") != hashlib.sha256(data).hexdigest():
                return None
            if len(data) != dimensions * 4:
                return None
            return list(struct.unpack(f"<{dimensions}f", data))
        except (OSError, ValueError, json.JSONDecodeError, struct.error):
            return None

    def store(
        self,
        key: str,
        vector: Sequence[float],
        *,
        document: EmbeddingDocument,
        embedder: Embedder,
    ) -> None:
        if len(vector) != embedder.dimensions:
            raise ValueError("Cannot cache vector with wrong dimensions")
        data = struct.pack(f"<{embedder.dimensions}f", *vector)
        atomic_write_bytes(self.root / f"{key}.f32", data)
        atomic_write_json(
            self.root / f"{key}.json",
            {
                "schema_version": 2,
                "cache_key": key,
                "chunk_id": document.chunk_id,
                "model": embedder.model,
                "dimensions": embedder.dimensions,
                "task_type": embedder.task_type,
                "embedding_input_sha256": document.embedding_input_sha256,
                "sha256": hashlib.sha256(data).hexdigest(),
                "created_at": utc_now(),
            },
        )


def _embed_with_cache(
    documents: Sequence[EmbeddingDocument],
    *,
    embedder: Embedder,
    cache: EmbeddingCache,
    batch_size: int,
) -> tuple[list[list[float]], list[str], int, int]:
    vectors: list[list[float] | None] = [None] * len(documents)
    keys: list[str] = []
    misses: list[int] = []
    hits = 0
    for index, document in enumerate(documents):
        key = cache.key(
            model=embedder.model,
            dimensions=embedder.dimensions,
            task_type=embedder.task_type,
            embedding_input_sha256=document.embedding_input_sha256,
        )
        keys.append(key)
        cached = cache.load(key, embedder.dimensions)
        if cached is None:
            misses.append(index)
        else:
            vectors[index] = cached
            hits += 1

    for start in range(0, len(misses), batch_size):
        indexes = misses[start : start + batch_size]
        batch = [documents[index] for index in indexes]
        results = embedder.embed(batch)
        if len(results) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(results)} vectors for {len(batch)} documents"
            )
        for index, document, vector in zip(indexes, batch, results, strict=True):
            if len(vector) != embedder.dimensions:
                raise RuntimeError(
                    f"Wrong vector size for {document.chunk_id}: {len(vector)}"
                )
            vectors[index] = vector
            cache.store(keys[index], vector, document=document, embedder=embedder)

    if any(vector is None for vector in vectors):
        raise RuntimeError("Internal error: one or more embeddings were not populated")
    return [list(vector) for vector in vectors if vector is not None], keys, hits, len(misses)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE books (
          book_id TEXT PRIMARY KEY,
          source_file TEXT NOT NULL,
          title TEXT NOT NULL,
          title_source TEXT NOT NULL,
          authors_json TEXT NOT NULL,
          authors_source TEXT,
          edition TEXT,
          publication_year TEXT,
          language TEXT,
          source_sha256 TEXT NOT NULL,
          page_count INTEGER NOT NULL,
          status TEXT NOT NULL,
          license_status TEXT NOT NULL,
          fallback_pages_json TEXT NOT NULL,
          manifest_json TEXT NOT NULL
        );
        CREATE TABLE pages (
          book_id TEXT NOT NULL REFERENCES books(book_id),
          page_number INTEGER NOT NULL,
          printed_page_label TEXT,
          status TEXT NOT NULL,
          citation TEXT NOT NULL,
          outline_path_json TEXT NOT NULL,
          outline_section TEXT,
          needs_visual_parser INTEGER NOT NULL,
          reason_codes_json TEXT NOT NULL,
          native_ir_path TEXT,
          native_diagnostics_path TEXT,
          native_processor_config_sha256 TEXT,
          native_text_quality_json TEXT NOT NULL,
          page_complexity_json TEXT NOT NULL,
          parser_backend TEXT NOT NULL,
          parser_model TEXT NOT NULL,
          conversion_route TEXT NOT NULL,
          llm_called INTEGER NOT NULL,
          markdown TEXT NOT NULL,
          native_text_characters INTEGER NOT NULL,
          markdown_sha256 TEXT,
          PRIMARY KEY (book_id, page_number)
        );
        CREATE TABLE chunks (
          chunk_id TEXT PRIMARY KEY,
          book_id TEXT NOT NULL REFERENCES books(book_id),
          chunk_index INTEGER NOT NULL,
          source_file TEXT NOT NULL,
          book_title TEXT NOT NULL,
          title_source TEXT NOT NULL,
          authors_json TEXT NOT NULL,
          authors_source TEXT,
          edition TEXT,
          publication_year TEXT,
          language TEXT,
          page_start INTEGER NOT NULL,
          page_end INTEGER NOT NULL,
          printed_page_start TEXT,
          printed_page_end TEXT,
          included_pages_json TEXT NOT NULL,
          section TEXT NOT NULL,
          outline_section TEXT,
          markdown_section TEXT,
          outline_path_json TEXT NOT NULL,
          citation TEXT NOT NULL,
          license_status TEXT NOT NULL,
          source_quality TEXT NOT NULL,
          structural_block_type TEXT,
          structural_block_types_json TEXT NOT NULL,
          fallback_pages_json TEXT NOT NULL,
          native_structured_pages_json TEXT NOT NULL,
          has_native_structured INTEGER NOT NULL,
          visual_review_pages_json TEXT NOT NULL,
          reason_codes_json TEXT NOT NULL,
          body TEXT NOT NULL,
          approximate_tokens INTEGER NOT NULL,
          oversized INTEGER NOT NULL,
          oversized_reason TEXT,
          content_sha256 TEXT NOT NULL,
          UNIQUE (book_id, chunk_index)
        );
        CREATE INDEX chunks_book_pages ON chunks(book_id, page_start, page_end);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
          body,
          section,
          content='chunks',
          content_rowid='rowid',
          tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE embedding_rows (
          vector_index INTEGER PRIMARY KEY,
          chunk_id TEXT NOT NULL UNIQUE REFERENCES chunks(chunk_id),
          cache_key TEXT NOT NULL
        );
        CREATE TABLE embedding_matrix (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          model TEXT NOT NULL,
          dimensions INTEGER NOT NULL,
          task_type TEXT NOT NULL,
          row_count INTEGER NOT NULL,
          ordered_chunk_ids_sha256 TEXT NOT NULL,
          blob_sha256 TEXT NOT NULL,
          vectors_f32le BLOB NOT NULL
        );
        """
    )


def pack_corpus(
    settings: Settings,
    report: PreflightReport,
    *,
    output_path: Path,
    embedder: Embedder,
    batch_size: int,
) -> dict[str, Any]:
    if report.errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(report.errors))
    if not report.books:
        raise RuntimeError("Preflight selected no books")
    if embedder.semantic_vectors and embedder.model.startswith("deterministic-fake"):
        raise RuntimeError("A fake embedding model cannot be marked semantic")

    chunks = sorted(
        (chunk for book in report.books for chunk in book.chunks),
        key=lambda chunk: (str(chunk["book_id"]), int(chunk["chunk_index"])),
    )
    documents: list[EmbeddingDocument] = []
    for chunk in chunks:
        embedding_text = build_embedding_text(chunk)
        documents.append(
            EmbeddingDocument(
                chunk_id=str(chunk["chunk_id"]),
                text=embedding_text,
                embedding_input_sha256=sha256_text(embedding_text),
            )
        )
    cache = EmbeddingCache(settings.embedding_cache_dir)
    vectors, cache_keys, cache_hits, cache_misses = _embed_with_cache(
        documents,
        embedder=embedder,
        cache=cache,
        batch_size=batch_size,
    )
    vector_blob = b"".join(
        struct.pack(f"<{embedder.dimensions}f", *vector) for vector in vectors
    )
    expected_bytes = len(chunks) * embedder.dimensions * 4
    if len(vector_blob) != expected_bytes:
        raise RuntimeError(
            f"Vector matrix is {len(vector_blob)} bytes; expected {expected_bytes}"
        )
    ordered_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    ordered_ids_sha256 = sha256_text("\n".join(ordered_ids) + "\n")
    vector_blob_sha256 = hashlib.sha256(vector_blob).hexdigest()
    corpus_sha256 = sha256_text(
        json.dumps(
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "content_sha256": chunk["content_sha256"],
                    "embedding_input_sha256": document.embedding_input_sha256,
                }
                for chunk, document in zip(chunks, documents, strict=True)
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    artifact_mode = "production" if embedder.semantic_vectors else "development"
    production_ready = bool(embedder.semantic_vectors)
    built_at = utc_now()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary)
    try:
        _create_schema(connection)
        for book in report.books:
            manifest = book.manifest
            book_id = str(manifest["book_id"])
            bibliography = manifest["bibliography"]
            license_status = str(
                manifest.get("license", {}).get("status", DEFAULT_LICENSE_STATUS)
            )
            connection.execute(
                "INSERT INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    book_id,
                    str(manifest["source_file"]),
                    str(bibliography["title"]),
                    str(bibliography["title_source"]),
                    json.dumps(bibliography.get("authors", []), ensure_ascii=False),
                    bibliography.get("authors_source"),
                    bibliography.get("edition"),
                    bibliography.get("publication_year"),
                    bibliography.get("language"),
                    str(manifest["source_sha256"]),
                    int(manifest["page_count"]),
                    str(manifest["status"]),
                    license_status,
                    json.dumps(manifest.get("fallback_pages", [])),
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                ),
            )
            for page in book.pages:
                page_number = int(page["pdf_page_number"])
                markdown_path = settings.md_dir / book_id / f"page-{page_number:04d}.md"
                connection.execute(
                    "INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        book_id,
                        page_number,
                        page.get("printed_page_label"),
                        str(page.get("status", "unknown")),
                        str(page["citation"]),
                        json.dumps(page.get("outline_path", []), ensure_ascii=False),
                        page.get("outline_section"),
                        int(bool(page.get("needs_visual_parser"))),
                        json.dumps(page.get("reason_codes", []), ensure_ascii=False),
                        page.get("native_ir_path"),
                        page.get("native_diagnostics_path"),
                        page.get("native_processor_config_sha256"),
                        json.dumps(page.get("native_text_quality", {}), ensure_ascii=False),
                        json.dumps(page.get("page_complexity", {}), ensure_ascii=False),
                        str(page.get("parser_backend") or "unknown"),
                        str(page.get("parser_model") or "unknown"),
                        str(page.get("conversion_route") or "unknown"),
                        int(bool(page.get("llm_called"))),
                        markdown_path.read_text(encoding="utf-8"),
                        int(page.get("native_text_characters", 0)),
                        page.get("markdown_sha256"),
                    ),
                )
        for chunk in chunks:
            connection.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(chunk["chunk_id"]),
                    str(chunk["book_id"]),
                    int(chunk["chunk_index"]),
                    str(chunk["source_file"]),
                    str(chunk["book_title"]),
                    str(chunk["title_source"]),
                    json.dumps(chunk.get("authors", []), ensure_ascii=False),
                    chunk.get("authors_source"),
                    chunk.get("edition"),
                    chunk.get("publication_year"),
                    chunk.get("language"),
                    int(chunk["page_start"]),
                    int(chunk["page_end"]),
                    chunk.get("printed_page_start"),
                    chunk.get("printed_page_end"),
                    json.dumps(chunk["included_pages"]),
                    str(chunk.get("section") or ""),
                    chunk.get("outline_section"),
                    chunk.get("markdown_section"),
                    json.dumps(chunk.get("outline_path", []), ensure_ascii=False),
                    str(chunk["citation"]),
                    str(chunk["license_status"]),
                    str(chunk["source_quality"]),
                    chunk.get("structural_block_type"),
                    json.dumps(chunk.get("structural_block_types", [])),
                    json.dumps(chunk.get("fallback_pages", [])),
                    json.dumps(chunk.get("native_structured_pages", [])),
                    int(bool(chunk.get("has_native_structured"))),
                    json.dumps(chunk.get("visual_review_pages", [])),
                    json.dumps(chunk.get("reason_codes", []), ensure_ascii=False),
                    str(chunk["body"]),
                    int(chunk["approximate_tokens"]),
                    int(bool(chunk.get("oversized"))),
                    chunk.get("oversized_reason"),
                    str(chunk["content_sha256"]),
                ),
            )
        # External-content FTS5 tables do not populate themselves.
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        connection.executemany(
            "INSERT INTO embedding_rows VALUES (?, ?, ?)",
            [
                (index, chunk["chunk_id"], cache_keys[index])
                for index, chunk in enumerate(chunks)
            ],
        )
        connection.execute(
            "INSERT INTO embedding_matrix VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
            (
                embedder.model,
                embedder.dimensions,
                embedder.task_type,
                len(chunks),
                ordered_ids_sha256,
                vector_blob_sha256,
                vector_blob,
            ),
        )
        metadata = {
            "schema_version": "5",
            "built_at": built_at,
            "corpus_sha256": corpus_sha256,
            "book_count": str(len(report.books)),
            "page_count": str(report.page_count),
            "chunk_count": str(len(chunks)),
            "embedding_model": embedder.model,
            "embedding_dimensions": str(embedder.dimensions),
            "embedding_task_type": embedder.task_type,
            "artifact_mode": artifact_mode,
            "semantic_vectors": str(embedder.semantic_vectors).lower(),
            "production_ready": str(production_ready).lower(),
            "ordered_chunk_ids_sha256": ordered_ids_sha256,
            "vector_blob_sha256": vector_blob_sha256,
            "fts_tokenizer": "unicode61 remove_diacritics 2",
            "fts_columns": "body,section",
            "fts_bm25_weights": "10.0,3.0",
        }
        connection.executemany("INSERT INTO meta VALUES (?, ?)", metadata.items())
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        fts_term = fts_match_query(str(chunks[0]["body"]))
        if fts_term:
            fts_hit = connection.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
                (fts_term,),
            ).fetchone()
            if fts_hit is None:
                raise RuntimeError("FTS5 verification query returned no content hit")
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    os.replace(temporary, output_path)
    database_sha256 = sha256_file(output_path)
    result = {
        "schema_version": 5,
        "path": str(output_path),
        "sha256": database_sha256,
        "built_at": built_at,
        "corpus_sha256": corpus_sha256,
        "book_count": len(report.books),
        "page_count": report.page_count,
        "chunk_count": len(chunks),
        "embedding_model": embedder.model,
        "embedding_dimensions": embedder.dimensions,
        "embedding_task_type": embedder.task_type,
        "artifact_mode": artifact_mode,
        "semantic_vectors": embedder.semantic_vectors,
        "production_ready": production_ready,
        "ordered_chunk_ids_sha256": ordered_ids_sha256,
        "vector_blob_sha256": vector_blob_sha256,
        "vector_bytes": len(vector_blob),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "warnings": list(report.warnings),
    }
    atomic_write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), result)
    return result
