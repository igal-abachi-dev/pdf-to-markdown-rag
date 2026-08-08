from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .config import Settings
from .chunking import build_chunks
from .bibliography import format_pdf_citation, resolve_bibliography
from .gemini_client import GeminiPageConverter
from .native_markdown import (
    NATIVE_EXTRACTOR_VERSION,
    NATIVE_PROCESSOR_CONFIG_SHA256,
    NativeRenderResult,
    render_debug_overlay,
    render_native_page,
)
from .pdf_facts import (
    build_document_profile,
    extract_document_facts,
    page_facts as document_page_facts,
)
from .page_converter import PageInput, PageResult
from .provenance import (
    DEFAULT_LICENSE_STATUS,
    load_license_file,
    load_license_metadata,
    normalize_license_info,
)
from .utils import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    relative_posix,
    safe_slug,
    sha256_file,
    sha256_text,
    utc_now,
    without_front_matter,
    yaml_string,
)


LOGGER = logging.getLogger(__name__)
MAX_INLINE_IMAGE_BYTES = 14 * 1024 * 1024


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # Access denied means the process exists but is owned by another account.
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _acquire_lock(lock_path: Path, source_name: str):
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_pid = int(lock_data.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            lock_pid = 0
        if lock_pid and _process_exists(lock_pid):
            raise RuntimeError(
                f"PDF is already being processed by PID {lock_pid}: {source_name}"
            )
        LOGGER.warning(
            "Removing stale lock for %s (PID %s is not running)", source_name, lock_pid
        )
        lock_path.unlink(missing_ok=True)

    try:
        handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"PDF is already being processed: {source_name}") from exc
    json.dump(
        {"pid": os.getpid(), "started_at": utc_now(), "source_file": source_name}, handle
    )
    handle.write("\n")
    handle.close()


@dataclass(frozen=True)
class BookPaths:
    book_id: str
    raw: Path
    md: Path
    metadata: Path
    pages: Path
    chunks: Path
    debug: Path
    manifest: Path


def _book_paths(settings: Settings, book_id: str) -> BookPaths:
    raw = settings.raw_dir / book_id
    md = settings.md_dir / book_id
    metadata = settings.metadata_dir / book_id
    pages = settings.pages_dir / book_id
    chunks = settings.chunks_dir / book_id
    debug = settings.debug_dir / book_id
    for path in (raw, md, metadata, pages, chunks, debug):
        path.mkdir(parents=True, exist_ok=True)
    return BookPaths(
        book_id=book_id,
        raw=raw,
        md=md,
        metadata=metadata,
        pages=pages,
        chunks=chunks,
        debug=debug,
        manifest=metadata / "manifest.json",
    )


def _find_existing_manifest_by_hash(
    settings: Settings, source_sha256: str
) -> tuple[dict[str, Any] | None, Path | None]:
    matches: list[tuple[dict[str, Any], Path]] = []
    for manifest_path in settings.metadata_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("source_sha256") == source_sha256 and manifest.get("book_id"):
            matches.append((manifest, manifest_path))
    if not matches:
        return None, None
    # Older retry behavior could append the hash suffix repeatedly. The shortest
    # ID is the original canonical book identity.
    return min(matches, key=lambda item: (len(str(item[0]["book_id"])), str(item[0]["book_id"])))


def _text_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    page_dict = page.get_text("dict", sort=True)
    result: list[dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[dict[str, Any]] = []
        for line in block.get("lines", []):
            spans = [
                {
                    "text": span.get("text", ""),
                    "bbox": span.get("bbox"),
                    "font": span.get("font"),
                    "size": span.get("size"),
                    "flags": span.get("flags"),
                    "color": span.get("color"),
                    "alpha": span.get("alpha", 255),
                    "char_flags": span.get("char_flags"),
                    "origin": span.get("origin"),
                }
                for span in line.get("spans", [])
            ]
            lines.append({"bbox": line.get("bbox"), "dir": line.get("dir"), "spans": spans})
        result.append({"bbox": block.get("bbox"), "number": block.get("number"), "lines": lines})
    return result


def _native_route_decision(
    settings: Settings,
    native_result: NativeRenderResult,
    native_text: str,
    extraction_error: str | None,
) -> dict[str, Any]:
    """Return an auditable, conservative decision for the opt-in native route."""
    failed_gates: list[str] = []
    diagnostics = native_result.diagnostics
    quality = diagnostics.get("native_text_quality") or {}
    if not settings.native_page_router_enabled:
        failed_gates.append("router_disabled")
    if extraction_error:
        failed_gates.append("native_extraction_error")
    if not native_text.strip():
        failed_gates.append("native_text_empty")
    if diagnostics.get("render_mode") != "structured":
        failed_gates.append("native_render_not_structured")
    if not quality.get("usable_as_reference", False):
        failed_gates.append("native_text_quality_gate_failed")
    if diagnostics.get("needs_visual_parser"):
        failed_gates.append("needs_visual_parser")
    eligibility_failures = [
        gate for gate in failed_gates if gate != "router_disabled"
    ]
    return {
        "enabled": settings.native_page_router_enabled,
        "eligible_if_enabled": not eligibility_failures,
        "route_native": settings.native_page_router_enabled
        and not eligibility_failures,
        "failed_gates": failed_gates,
        "render_mode": diagnostics.get("render_mode"),
        "rendered_text_coverage": diagnostics.get("rendered_text_coverage"),
        "native_text_quality_reason_codes": quality.get("reason_codes", []),
        "visual_reason_codes": diagnostics.get("reason_codes", []),
    }


def _native_router_summary(page_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    gate_counts: dict[str, int] = {}
    quality_reason_counts: dict[str, int] = {}
    for page in page_metadata:
        router = page.get("native_router") or {}
        for gate in router.get("failed_gates") or []:
            if gate == "router_disabled":
                continue
            gate_counts[str(gate)] = gate_counts.get(str(gate), 0) + 1
        for reason in router.get("native_text_quality_reason_codes") or []:
            quality_reason_counts[str(reason)] = (
                quality_reason_counts.get(str(reason), 0) + 1
            )
    return {
        "page_count": len(page_metadata),
        "eligible_if_enabled": sum(
            bool((page.get("native_router") or {}).get("eligible_if_enabled"))
            for page in page_metadata
        ),
        "native_routed": sum(
            page.get("conversion_route") == "native_structured"
            for page in page_metadata
        ),
        "llm_invoked": sum(bool(page.get("llm_called")) for page in page_metadata),
        "failed_gate_counts": dict(sorted(gate_counts.items())),
        "native_text_quality_reason_counts": dict(
            sorted(quality_reason_counts.items())
        ),
    }


def _native_blocks_to_markdown(blocks: list[dict[str, Any]], native_text: str) -> str:
    paragraphs: list[str] = []
    for block in blocks:
        lines: list[str] = []
        for line in block.get("lines", []):
            text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
            text = " ".join(text.split())
            if text:
                lines.append(text)
        if not lines:
            continue
        paragraph = lines[0]
        for line in lines[1:]:
            paragraph += line if paragraph.endswith("-") else f" {line}"
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip() or native_text.strip()


def _page_front_matter(
    source_name: str,
    book_id: str,
    page_number: int,
    page_count: int,
    citation: str,
    model: str,
    status: str,
    bibliography: dict[str, Any],
    page_facts: dict[str, Any] | None = None,
    *,
    parser_backend: str | None = None,
    conversion_route: str | None = None,
    llm_called: bool | None = None,
) -> str:
    page_facts = page_facts or {}
    printed_label = page_facts.get("printed_page_label")
    outline_path = page_facts.get("outline_path") or []
    return (
        "---\n"
        f"source_file: {yaml_string(source_name)}\n"
        f"book_title: {yaml_string(str(bibliography['title']))}\n"
        f"title_source: {yaml_string(str(bibliography['title_source']))}\n"
        f"authors: {json.dumps(bibliography.get('authors', []), ensure_ascii=False)}\n"
        f"authors_source: {yaml_string(str(bibliography.get('authors_source') or ''))}\n"
        f"edition: {yaml_string(str(bibliography.get('edition') or ''))}\n"
        f"publication_year: {yaml_string(str(bibliography.get('publication_year') or ''))}\n"
        f"language: {yaml_string(str(bibliography.get('language') or ''))}\n"
        f"book_id: {yaml_string(book_id)}\n"
        f"pdf_page_number: {page_number}\n"
        f"printed_page_label: {yaml_string(str(printed_label or ''))}\n"
        f"pdf_page_count: {page_count}\n"
        f"outline_path: {json.dumps(outline_path, ensure_ascii=False)}\n"
        f"outline_section: {yaml_string(str(outline_path[-1] if outline_path else ''))}\n"
        f"citation: {yaml_string(citation)}\n"
        f"parser_model: {yaml_string(model)}\n"
        f"parser_backend: {yaml_string(str(parser_backend or 'unknown'))}\n"
        f"conversion_route: {yaml_string(str(conversion_route or 'unknown'))}\n"
        f"llm_called: {str(bool(llm_called)).lower() if llm_called is not None else 'null'}\n"
        f"status: {yaml_string(status)}\n"
        "tags:\n"
        "  - pdf-page\n"
        "  - rag-source\n"
        "---\n\n"
    )


def _write_combined_outputs(
    paths: BookPaths,
    source_name: str,
    page_count: int,
    settings: Settings,
    license_info: dict[str, Any],
    bibliography: dict[str, Any],
) -> dict[str, Any]:
    raw_parts: list[str] = []
    md_parts: list[str] = []
    chunk_pages: list[tuple[int, str]] = []
    page_metadata: list[dict[str, Any]] = []
    for page_number in range(1, page_count + 1):
        name = f"page-{page_number:04d}"
        raw_text = (paths.raw / f"{name}.txt").read_text(encoding="utf-8")
        markdown = (paths.md / f"{name}.md").read_text(encoding="utf-8")
        metadata = json.loads((paths.metadata / f"{name}.json").read_text(encoding="utf-8"))
        raw_parts.append(f"===== PDF PAGE {page_number} =====\n{raw_text.rstrip()}\n")
        body = without_front_matter(markdown).strip()
        printed_label = metadata.get("printed_page_label")
        printed_marker = (
            f", printed page {printed_label}"
            if printed_label and str(printed_label) != str(page_number)
            else ""
        )
        marker = f"[Source: {source_name}, PDF page {page_number}{printed_marker}]"
        page_entry = (
            f"<!-- PAGE_START source={json.dumps(source_name, ensure_ascii=False)} "
            f"pdf_page={page_number} -->\n\n"
            f"{marker}\n\n"
            f"> [!cite] {metadata['citation']}\n"
            f"> Obsidian page note: [[md/{paths.book_id}/page-{page_number:04d}|PDF p. {page_number}]] "
            f"^pdf-page-{page_number:04d}\n\n"
            f"{body}\n\n"
            f"<!-- PAGE_END pdf_page={page_number} -->\n"
        )
        md_parts.append(page_entry)
        if metadata.get("status") in {"complete", "native_text_fallback"}:
            chunk_pages.append((page_number, body))
        page_metadata.append(metadata)

    atomic_write_text(paths.raw / "book.txt", "\n".join(raw_parts))
    failed_page_numbers = [
        item["pdf_page_number"] for item in page_metadata if item.get("status") == "failed"
    ]
    fallback_page_numbers = [
        item["pdf_page_number"]
        for item in page_metadata
        if item.get("status") == "native_text_fallback"
    ]
    visual_review_page_numbers = [
        item["pdf_page_number"]
        for item in page_metadata
        if item.get("needs_visual_parser")
    ]
    native_routed_page_numbers = [
        item["pdf_page_number"]
        for item in page_metadata
        if item.get("conversion_route") == "native_structured"
    ]
    book_front_matter = (
        "---\n"
        f"title: {yaml_string(str(bibliography['title']))}\n"
        f"source_file: {yaml_string(source_name)}\n"
        f"title_source: {yaml_string(str(bibliography['title_source']))}\n"
        f"authors: {json.dumps(bibliography.get('authors', []), ensure_ascii=False)}\n"
        f"authors_source: {yaml_string(str(bibliography.get('authors_source') or ''))}\n"
        f"edition: {yaml_string(str(bibliography.get('edition') or ''))}\n"
        f"publication_year: {yaml_string(str(bibliography.get('publication_year') or ''))}\n"
        f"language: {yaml_string(str(bibliography.get('language') or ''))}\n"
        f"book_id: {yaml_string(paths.book_id)}\n"
        f"pdf_page_count: {page_count}\n"
        f"status: {yaml_string('partial' if failed_page_numbers or fallback_page_numbers else 'complete')}\n"
        f"failed_pages: {json.dumps(failed_page_numbers)}\n"
        f"fallback_pages: {json.dumps(fallback_page_numbers)}\n"
        f"native_routed_pages: {json.dumps(native_routed_page_numbers)}\n"
        f"visual_review_pages: {json.dumps(visual_review_page_numbers)}\n"
        f"license_status: {yaml_string(str(license_info.get('status', DEFAULT_LICENSE_STATUS)))}\n"
        "tags:\n"
        "  - pdf-book\n"
        "  - rag-source\n"
        "---\n\n"
    )
    atomic_write_text(paths.md / "book.md", book_front_matter + "\n\n".join(md_parts))
    atomic_write_text(
        paths.metadata / "pages.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in page_metadata),
    )
    chunk_summary = build_chunks(
        pages=chunk_pages,
        output_dir=paths.chunks,
        source_name=source_name,
        book_id=paths.book_id,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        license_status=str(license_info.get("status", DEFAULT_LICENSE_STATUS)),
        page_quality={
            int(item["pdf_page_number"]): str(item.get("status", "unknown"))
            for item in page_metadata
        },
        bibliography=bibliography,
        page_context={
            int(item["pdf_page_number"]): {
                "printed_page_label": item.get("printed_page_label"),
                "outline_path": item.get("outline_path") or [],
                "outline_section": item.get("outline_section"),
                "needs_visual_parser": bool(item.get("needs_visual_parser")),
                "reason_codes": item.get("reason_codes") or [],
                "conversion_route": item.get("conversion_route"),
                "parser_backend": item.get("parser_backend"),
                "llm_called": item.get("llm_called"),
            }
            for item in page_metadata
        },
    )
    chunk_summary["native_router_summary"] = _native_router_summary(page_metadata)
    return chunk_summary


def _refresh_page_bibliography(
    paths: BookPaths,
    source_name: str,
    page_count: int,
    bibliography: dict[str, Any],
    document_facts: dict[str, Any] | None = None,
) -> None:
    """Refresh page identity/citations without changing the transcribed body."""
    for page_number in range(1, page_count + 1):
        stem = f"page-{page_number:04d}"
        markdown_path = paths.md / f"{stem}.md"
        metadata_path = paths.metadata / f"{stem}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        old_citation = str(metadata.get("citation") or "")
        facts = (
            document_page_facts(document_facts, page_number - 1)
            if document_facts
            else {
                "printed_page_label": metadata.get("printed_page_label"),
                "outline_path": metadata.get("outline_path") or [],
            }
        )
        citation = format_pdf_citation(
            bibliography,
            page_number,
            printed_page_start=facts.get("printed_page_label"),
        )
        body = without_front_matter(markdown_path.read_text(encoding="utf-8")).strip()
        if old_citation and old_citation != citation:
            body = body.replace(old_citation, citation)
        conversion_route = metadata.get("conversion_route")
        if not conversion_route:
            conversion_route = (
                "native_fallback_after_llm"
                if metadata.get("status") == "native_text_fallback"
                else "visual_llm"
            )
        parser_backend = metadata.get("parser_backend") or (
            "pymupdf_native"
            if conversion_route == "native_structured"
            else "gemini"
        )
        metadata.update(
            {
                "schema_version": 3,
                "citation": citation,
                "bibliography": bibliography,
                "book_title": bibliography["title"],
                "title_source": bibliography["title_source"],
                "authors": bibliography.get("authors", []),
                "authors_source": bibliography.get("authors_source"),
                "edition": bibliography.get("edition"),
                "publication_year": bibliography.get("publication_year"),
                "language": bibliography.get("language"),
                "printed_page_label": facts.get("printed_page_label"),
                "outline_path": facts.get("outline_path") or [],
                "outline_section": (facts.get("outline_path") or [None])[-1],
                "rotation_degrees": facts.get("rotation_degrees", metadata.get("rotation_degrees", 0)),
                "coordinate_space": facts.get("coordinate_space", "mupdf_unrotated"),
                "needs_visual_parser": bool(metadata.get("needs_visual_parser", False)),
                "reason_codes": metadata.get("reason_codes") or [],
                "conversion_route": conversion_route,
                "llm_called": bool(
                    metadata.get(
                        "llm_called", conversion_route != "native_structured"
                    )
                ),
                "parser_backend": parser_backend,
                "parser_model": metadata.get("parser_model")
                or (
                    f"pymupdf-structured-v{NATIVE_EXTRACTOR_VERSION}"
                    if parser_backend == "pymupdf_native"
                    else "unknown"
                ),
            }
        )
        front_matter = _page_front_matter(
            source_name,
            paths.book_id,
            page_number,
            page_count,
            citation,
            str(metadata.get("parser_model") or "unknown"),
            str(metadata.get("status") or "unknown"),
            bibliography,
            facts,
            parser_backend=str(metadata.get("parser_backend") or "unknown"),
            conversion_route=str(metadata.get("conversion_route") or "unknown"),
            llm_called=metadata.get("llm_called"),
        )
        atomic_write_text(markdown_path, front_matter + body + "\n")
        atomic_write_json(metadata_path, metadata)


def rebuild_book(
    settings: Settings,
    book_id: str,
    *,
    license_sidecar: Path | None = None,
    refresh_pdf_facts: bool = False,
    refresh_native_fallbacks: bool = False,
) -> dict[str, Any]:
    """Rebuild derived artifacts while holding the book lock end to end."""
    paths = _book_paths(settings, book_id)
    if not paths.manifest.exists():
        raise FileNotFoundError(f"Unknown book_id: {book_id}")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    source_name = str(manifest.get("source_file") or book_id)
    lock_path = settings.state_dir / "locks" / f"{book_id}.lock"
    _acquire_lock(lock_path, source_name)
    try:
        return _rebuild_book_locked(
            settings,
            book_id,
            license_sidecar=license_sidecar,
            refresh_pdf_facts=refresh_pdf_facts,
            refresh_native_fallbacks=refresh_native_fallbacks,
        )
    finally:
        lock_path.unlink(missing_ok=True)


def _rebuild_book_locked(
    settings: Settings,
    book_id: str,
    *,
    license_sidecar: Path | None = None,
    refresh_pdf_facts: bool = False,
    refresh_native_fallbacks: bool = False,
) -> dict[str, Any]:
    """Rebuild combined Markdown and chunks from saved pages without Gemini calls."""
    paths = _book_paths(settings, book_id)
    if not paths.manifest.exists():
        raise FileNotFoundError(f"Unknown book_id: {book_id}")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "partial"}:
        raise RuntimeError(
            f"Book must be complete or partial before rebuild; current status is "
            f"{manifest.get('status', 'unknown')}"
        )
    page_count = int(manifest.get("page_count", 0))
    if page_count <= 0:
        raise RuntimeError(f"Manifest has no valid page_count: {paths.manifest}")

    if license_sidecar is not None:
        license_sidecar = license_sidecar.resolve()
        license_info, license_provenance = load_license_file(license_sidecar)
    else:
        saved_license_path = paths.metadata / "license.json"
        if saved_license_path.exists():
            saved = json.loads(saved_license_path.read_text(encoding="utf-8"))
            license_info = normalize_license_info(saved.get("license"))
            license_provenance = saved.get(
                "provenance", {"sidecar_file": None, "sidecar_sha256": None}
            )
        else:
            license_info = normalize_license_info(manifest.get("license"))
            license_provenance = manifest.get(
                "license_provenance",
                {"sidecar_file": None, "sidecar_sha256": None},
            )

    source_name = str(manifest["source_file"])
    document_facts: dict[str, Any] | None = None
    document_facts_path = paths.metadata / "document.json"
    if document_facts_path.exists():
        document_facts = json.loads(document_facts_path.read_text(encoding="utf-8"))

    source_pdf: Path | None = None
    if refresh_pdf_facts or refresh_native_fallbacks:
        candidates: list[Path] = []
        if manifest.get("processed_pdf"):
            candidates.append(settings.root / str(manifest["processed_pdf"]))
        candidates.extend(
            [
                settings.processed_dir / f"{book_id}.pdf",
                settings.inbox_dir / source_name,
            ]
        )
        source_pdf = next((candidate for candidate in candidates if candidate.exists()), None)
        if source_pdf is None:
            raise FileNotFoundError(
                "Cannot refresh PDF-derived artifacts; the processed or inbox PDF is missing"
            )

        with fitz.open(source_pdf) as document:
            if document.needs_pass:
                raise RuntimeError(
                    "PDF requires a password; cannot refresh document facts or native fallbacks"
                )
            if document.page_count != page_count:
                raise RuntimeError(
                    f"PDF now has {document.page_count} pages but the manifest expects {page_count}"
                )
            document_facts, xmp_metadata = extract_document_facts(document)
            document_profile = build_document_profile(document)
            atomic_write_json(document_facts_path, document_facts)
            atomic_write_json(paths.metadata / "document-profile.json", document_profile)
            if xmp_metadata:
                atomic_write_text(paths.metadata / "document.xmp.xml", xmp_metadata)
            else:
                (paths.metadata / "document.xmp.xml").unlink(missing_ok=True)

            if refresh_native_fallbacks:
                for page_index in range(page_count):
                    page_number = page_index + 1
                    stem = f"page-{page_number:04d}"
                    page = document.load_page(page_index)
                    native_text = page.get_text("text", sort=True).strip()
                    blocks = _text_blocks(page)
                    facts = document_page_facts(document_facts, page_index)
                    try:
                        native_result = render_native_page(
                            page, native_text, blocks, facts, document_profile
                        )
                    except Exception as exc:
                        LOGGER.exception(
                            "Structured native refresh failed; retaining exact flat text: %s page %s",
                            source_name,
                            page_number,
                        )
                        fallback_body = _native_blocks_to_markdown(blocks, native_text)
                        diagnostics = {
                            "schema_version": 1,
                            "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
                            "native_processor_config_sha256": NATIVE_PROCESSOR_CONFIG_SHA256,
                            "render_mode": "flat_exception_fallback",
                            "needs_visual_parser": True,
                            "reason_codes": ["native_structured_render_error"],
                            "warnings": [f"{type(exc).__name__}: {str(exc)[:1000]}"],
                        }
                        native_result = NativeRenderResult(
                            markdown=fallback_body,
                            page_ir={
                                "schema_version": 1,
                                "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
                                "native_processor_config_sha256": NATIVE_PROCESSOR_CONFIG_SHA256,
                                "pdf_page_number": page_number,
                                "printed_page_label": facts.get("printed_page_label"),
                                "coordinate_space": "mupdf_unrotated",
                                "outline_path": facts.get("outline_path") or [],
                                "outline_section": (facts.get("outline_path") or [None])[-1],
                                "blocks": [],
                                "diagnostics": diagnostics,
                            },
                            diagnostics=diagnostics,
                        )
                    raw_path = paths.raw / f"{stem}.txt"
                    blocks_path = paths.raw / f"{stem}.blocks.json"
                    native_md_path = paths.raw / f"{stem}.native.md"
                    native_ir_path = paths.raw / f"{stem}.ir.json"
                    native_diagnostics_path = (
                        paths.raw / f"{stem}.native-diagnostics.json"
                    )
                    atomic_write_text(
                        raw_path, native_text + ("\n" if native_text else "")
                    )
                    atomic_write_json(
                        blocks_path,
                        {
                            "schema_version": 2,
                            "pdf_page_number": page_number,
                            "width_points": page.rect.width,
                            "height_points": page.rect.height,
                            "coordinate_space": "mupdf_unrotated",
                            "page_facts": facts,
                            "blocks": blocks,
                            "extraction_error": None,
                        },
                    )
                    atomic_write_text(
                        native_md_path,
                        native_result.markdown
                        + ("\n" if native_result.markdown else ""),
                    )
                    atomic_write_json(native_ir_path, native_result.page_ir)
                    atomic_write_json(
                        native_diagnostics_path, native_result.diagnostics
                    )
                    debug_overlay_path = paths.debug / f"{stem}-layout.jpg"
                    if (
                        settings.native_debug_overlays
                        and native_result.diagnostics.get("needs_visual_parser")
                    ):
                        atomic_write_bytes(
                            debug_overlay_path,
                            render_debug_overlay(page, native_result.page_ir),
                        )
                    metadata_path = paths.metadata / f"{stem}.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata.update(
                        {
                            "schema_version": 3,
                            "native_text_present": bool(native_text),
                            "native_text_characters": len(native_text),
                            "native_text_path": relative_posix(raw_path, settings.root),
                            "native_blocks_path": relative_posix(
                                blocks_path, settings.root
                            ),
                            "native_markdown_path": relative_posix(
                                native_md_path, settings.root
                            ),
                            "native_ir_path": relative_posix(
                                native_ir_path, settings.root
                            ),
                            "native_diagnostics_path": relative_posix(
                                native_diagnostics_path, settings.root
                            ),
                            "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
                            "native_processor_config_sha256": native_result.diagnostics.get(
                                "native_processor_config_sha256"
                            ),
                            "native_render_mode": native_result.diagnostics.get(
                                "render_mode"
                            ),
                            "native_text_quality": native_result.diagnostics.get(
                                "native_text_quality", {}
                            ),
                            "page_complexity": native_result.diagnostics.get(
                                "page_complexity", {}
                            ),
                            "needs_visual_parser": bool(
                                native_result.diagnostics.get("needs_visual_parser")
                            ),
                            "reason_codes": native_result.diagnostics.get(
                                "reason_codes", []
                            ),
                            "native_diagnostics": native_result.diagnostics,
                            "native_router": _native_route_decision(
                                settings, native_result, native_text, None
                            ),
                            "native_debug_overlay_path": (
                                relative_posix(debug_overlay_path, settings.root)
                                if settings.native_debug_overlays
                                and native_result.diagnostics.get(
                                    "needs_visual_parser"
                                )
                                else None
                            ),
                        }
                    )
                    if metadata.get("status") == "native_text_fallback":
                        citation = format_pdf_citation(
                            resolve_bibliography(
                                license_info,
                                source_name,
                                existing=manifest.get("bibliography"),
                                pdf_metadata=document_facts.get("pdf_metadata"),
                            ),
                            page_number,
                            printed_page_start=facts.get("printed_page_label"),
                        )
                        front_matter = _page_front_matter(
                            source_name,
                            book_id,
                            page_number,
                            page_count,
                            citation,
                            str(metadata.get("parser_model") or "unknown"),
                            "native_text_fallback",
                            resolve_bibliography(
                                license_info,
                                source_name,
                                existing=manifest.get("bibliography"),
                                pdf_metadata=document_facts.get("pdf_metadata"),
                            ),
                            facts,
                            parser_backend=str(
                                metadata.get("parser_backend") or "gemini"
                            ),
                            conversion_route=str(
                                metadata.get("conversion_route")
                                or "native_fallback_after_llm"
                            ),
                            llm_called=metadata.get("llm_called", True),
                        )
                        body = (
                            front_matter
                            + f"> [!warning] Native-text fallback for PDF page {page_number}\n"
                            + f"> Gemini finish reason: `{metadata.get('finish_reason') or 'RECITATION'}`\n"
                            + f"> API message: {metadata.get('error') or 'Not available'}\n"
                            + f"> Full response details: `metadata/{book_id}/{stem}.json`\n"
                            + "> Text is searchable and citable, but visual layout and image descriptions require review.\n\n"
                            + native_result.markdown
                            + "\n"
                        )
                        atomic_write_text(paths.md / f"{stem}.md", body)
                        metadata["fallback_source"] = "pymupdf_structured_v1"
                        metadata["layout_verified"] = False
                        metadata["markdown_sha256"] = sha256_text(
                            native_result.markdown
                        )
                    atomic_write_json(metadata_path, metadata)

    bibliography = resolve_bibliography(
        license_info,
        source_name,
        existing=None if license_sidecar is not None else manifest.get("bibliography"),
        pdf_metadata=(document_facts or {}).get("pdf_metadata"),
    )

    missing: list[str] = []
    for page_number in range(1, page_count + 1):
        stem = f"page-{page_number:04d}"
        for required in (
            paths.raw / f"{stem}.txt",
            paths.md / f"{stem}.md",
            paths.metadata / f"{stem}.json",
        ):
            if not required.exists():
                missing.append(relative_posix(required, settings.root))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise RuntimeError(f"Cannot rebuild; missing page artifacts: {preview}{suffix}")

    lock_path = settings.state_dir / "locks" / f"{book_id}.lock"
    try:
        _refresh_page_bibliography(
            paths,
            source_name,
            page_count,
            bibliography,
            document_facts=document_facts,
        )
        atomic_write_json(
            paths.metadata / "license.json",
            {
                "license": license_info,
                "provenance": license_provenance,
                "bibliography": bibliography,
            },
        )
        chunk_summary = _write_combined_outputs(
            paths,
            source_name,
            page_count,
            settings,
            license_info,
            bibliography,
        )
        page_records = [
            json.loads(
                (paths.metadata / f"page-{page_number:04d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for page_number in range(1, page_count + 1)
        ]
        failed_pages = [
            int(record["pdf_page_number"])
            for record in page_records
            if record.get("status") == "failed"
        ]
        fallback_pages = [
            int(record["pdf_page_number"])
            for record in page_records
            if record.get("status") == "native_text_fallback"
        ]
        native_routed_pages = [
            int(record["pdf_page_number"])
            for record in page_records
            if record.get("conversion_route") == "native_structured"
        ]
        visual_review_pages = [
            int(record["pdf_page_number"])
            for record in page_records
            if record.get("needs_visual_parser")
        ]
        successful_pages = sum(
            record.get("status") == "complete" for record in page_records
        )
        manifest["license"] = license_info
        manifest["license_provenance"] = license_provenance
        manifest["bibliography"] = bibliography
        manifest["schema_version"] = 3
        manifest["status"] = "partial" if failed_pages or fallback_pages else "complete"
        manifest["completed_pages"] = page_count
        manifest["successful_pages"] = successful_pages
        manifest["failed_pages"] = failed_pages
        manifest["fallback_pages"] = fallback_pages
        manifest["native_routed_pages"] = native_routed_pages
        manifest["visual_review_pages"] = visual_review_pages
        manifest["native_router_summary"] = chunk_summary["native_router_summary"]
        manifest["rebuilt_at"] = utc_now()
        if document_facts is not None:
            manifest["document_facts"] = {
                "path": relative_posix(document_facts_path, settings.root),
                "schema_version": document_facts.get("schema_version"),
                "outline_entries": len(document_facts.get("outline") or []),
                "page_label_rules": len(document_facts.get("page_label_rules") or []),
            }
        manifest.setdefault("outputs", {})["chunk_count"] = chunk_summary["chunk_count"]
        manifest["outputs"].update(
            {
                "raw_book": relative_posix(paths.raw / "book.txt", settings.root),
                "markdown_book": relative_posix(paths.md / "book.md", settings.root),
                "pages_metadata": relative_posix(
                    paths.metadata / "pages.jsonl", settings.root
                ),
                "chunks_directory": relative_posix(paths.chunks, settings.root),
                "chunks_metadata": relative_posix(
                    paths.chunks / "chunks.jsonl", settings.root
                ),
            }
        )
        atomic_write_json(paths.manifest, manifest)
        return manifest
    finally:
        lock_path.unlink(missing_ok=True)


def _manifest_base(
    pdf_path: Path,
    source_name: str,
    source_sha256: str,
    book_id: str,
    page_count: int,
    settings: Settings,
    license_info: dict[str, Any],
    license_provenance: dict[str, Any],
    bibliography: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "book_id": book_id,
        "source_file": source_name,
        "source_sha256": source_sha256,
        "source_size_bytes": pdf_path.stat().st_size,
        "page_count": page_count,
        "model": settings.gemini_model,
        "thinking_level": settings.thinking_level,
        "render_dpi": settings.render_dpi,
        "native_page_router_enabled": settings.native_page_router_enabled,
        "request_governor": {
            "max_requests_per_rolling_24_hours": settings.requests_per_day,
            "min_interval_seconds": settings.min_request_interval_seconds,
            "pause_on_rate_limit": settings.pause_on_rate_limit,
        },
        "license": license_info,
        "license_provenance": license_provenance,
        "bibliography": bibliography,
        "prompt_sha256": sha256_text(settings.prompt_path.read_text(encoding="utf-8-sig")),
    }


def ingest_pdf(
    pdf_path: Path,
    settings: Settings,
    *,
    retry_pages: set[int] | None = None,
    retry_fallback_pages: bool = False,
    retry_existing: bool = False,
) -> dict[str, Any]:
    settings.validate_for_ingestion()
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    source_sha256 = sha256_file(pdf_path)
    existing, _ = _find_existing_manifest_by_hash(settings, source_sha256)
    if retry_existing and existing is None:
        raise RuntimeError(
            "No existing ingestion job matches this PDF. Use 'ingest' for a new book."
        )
    if existing is not None:
        book_id = str(existing["book_id"])
        source_name = str(existing.get("source_file") or pdf_path.name)
        if pdf_path.stem != Path(source_name).stem:
            LOGGER.info(
                "Resolved renamed/duplicate PDF %s to canonical book %s (%s)",
                pdf_path.name,
                book_id,
                source_name,
            )
    else:
        book_id = f"{safe_slug(pdf_path.stem)}-{source_sha256[:10]}"
        source_name = pdf_path.name
    paths = _book_paths(settings, book_id)
    lock_path = settings.state_dir / "locks" / f"{book_id}.lock"
    license_info, license_provenance, license_sidecar = load_license_metadata(pdf_path)
    if license_sidecar is None and existing is not None and existing.get("license"):
        license_info = normalize_license_info(existing["license"])
        license_provenance = existing.get("license_provenance", license_provenance)
    bibliography = resolve_bibliography(
        license_info,
        source_name,
        existing=(
            existing.get("bibliography")
            if existing is not None and license_sidecar is None
            else None
        ),
    )
    atomic_write_json(
        paths.metadata / "license.json",
        {
            "license": license_info,
            "provenance": license_provenance,
            "bibliography": bibliography,
        },
    )

    if paths.manifest.exists():
        existing = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and retry_pages is None:
            identity_changed = existing.get("bibliography") != bibliography
            existing["license"] = license_info
            existing["license_provenance"] = license_provenance
            existing["bibliography"] = bibliography
            atomic_write_json(paths.manifest, existing)
            if identity_changed:
                LOGGER.info("Bibliography changed; rebuilding derived artifacts for %s", book_id)
                return rebuild_book(settings, book_id)
            LOGGER.info("Already complete: %s (%s)", source_name, book_id)
            return existing

    converter: GeminiPageConverter | None = None
    lock_acquired = False
    try:
        _acquire_lock(lock_path, source_name)
        lock_acquired = True
        with fitz.open(pdf_path) as document:
            if document.needs_pass:
                raise RuntimeError(
                    "PDF requires a password before its pages or metadata can be read; "
                    "password-protected PDFs are not supported"
                )
            page_count = document.page_count
            document_facts, xmp_metadata = extract_document_facts(document)
            document_profile = build_document_profile(document)
            bibliography = resolve_bibliography(
                license_info,
                source_name,
                existing=(
                    existing.get("bibliography")
                    if existing is not None and license_sidecar is None
                    else None
                ),
                pdf_metadata=document_facts.get("pdf_metadata"),
            )
            atomic_write_json(paths.metadata / "document.json", document_facts)
            if xmp_metadata:
                atomic_write_text(paths.metadata / "document.xmp.xml", xmp_metadata)
            else:
                (paths.metadata / "document.xmp.xml").unlink(missing_ok=True)
            atomic_write_json(paths.metadata / "document-profile.json", document_profile)
            atomic_write_json(
                paths.metadata / "license.json",
                {
                    "license": license_info,
                    "provenance": license_provenance,
                    "bibliography": bibliography,
                },
            )
            converter = GeminiPageConverter(settings)
            if retry_pages is not None:
                out_of_range = sorted(page for page in retry_pages if page > page_count)
                if out_of_range:
                    raise ValueError(
                        f"Retry pages exceed the {page_count}-page PDF: {out_of_range}"
                    )
            manifest = _manifest_base(
                pdf_path,
                source_name,
                source_sha256,
                book_id,
                page_count,
                settings,
                license_info,
                license_provenance,
                bibliography,
            )
            manifest["document_facts"] = {
                "path": relative_posix(paths.metadata / "document.json", settings.root),
                "schema_version": document_facts.get("schema_version"),
                "outline_entries": len(document_facts.get("outline") or []),
                "page_label_rules": len(document_facts.get("page_label_rules") or []),
                "is_repaired": document_facts.get("is_repaired"),
                "permissions": document_facts.get("permissions"),
            }
            copy_allowed = (
                document_facts.get("permissions", {})
                .get("decoded", {})
                .get("copy")
            )
            if copy_allowed is False:
                LOGGER.warning(
                    "PDF permission flags disallow copying for %s; recording the flag "
                    "separately from the license sidecar and continuing",
                    source_name,
                )
            manifest.update(
                {
                    "status": "processing",
                    "started_at": utc_now(),
                    "completed_pages": 0,
                    "successful_pages": 0,
                    "failed_pages": [],
                    "fallback_pages": [],
                    "native_routed_pages": [],
                    "visual_review_pages": [],
                }
            )
            if existing is not None:
                for preserved_key in ("processed_pdf", "processed_license_sidecar"):
                    if existing.get(preserved_key):
                        manifest[preserved_key] = existing[preserved_key]
            atomic_write_json(paths.manifest, manifest)

            successful_pages = 0
            failed_pages: list[int] = []
            fallback_pages: list[int] = []
            native_routed_pages: list[int] = []
            visual_review_pages: list[int] = []
            consecutive_page_failures = 0

            for page_index in range(page_count):
                page_number = page_index + 1
                page_stem = f"page-{page_number:04d}"
                page_md_path = paths.md / f"{page_stem}.md"
                page_meta_path = paths.metadata / f"{page_stem}.json"

                if page_md_path.exists() and page_meta_path.exists():
                    page_metadata = json.loads(page_meta_path.read_text(encoding="utf-8"))
                    recorded_status = page_metadata.get("status")
                    should_retry_selected = (
                        retry_pages is None or page_number in retry_pages
                    )
                    if recorded_status == "complete":
                        if page_metadata.get("conversion_route") == "native_structured":
                            native_routed_pages.append(page_number)
                        if page_metadata.get("needs_visual_parser"):
                            visual_review_pages.append(page_number)
                        successful_pages += 1
                        consecutive_page_failures = 0
                        LOGGER.info(
                            "Resume: skipping successful page %s/%s", page_number, page_count
                        )
                        manifest["completed_pages"] = page_number
                        manifest["successful_pages"] = successful_pages
                        manifest["failed_pages"] = failed_pages
                        manifest["fallback_pages"] = fallback_pages
                        manifest["native_routed_pages"] = native_routed_pages
                        manifest["visual_review_pages"] = visual_review_pages
                        atomic_write_json(paths.manifest, manifest)
                        continue
                    if recorded_status == "native_text_fallback" and not (
                        retry_fallback_pages and should_retry_selected
                    ):
                        fallback_pages.append(page_number)
                        if page_metadata.get("needs_visual_parser"):
                            visual_review_pages.append(page_number)
                        consecutive_page_failures = 0
                        LOGGER.info(
                            "Resume: keeping native-text fallback page %s/%s",
                            page_number,
                            page_count,
                        )
                        manifest["completed_pages"] = page_number
                        manifest["successful_pages"] = successful_pages
                        manifest["failed_pages"] = failed_pages
                        manifest["fallback_pages"] = fallback_pages
                        manifest["visual_review_pages"] = visual_review_pages
                        atomic_write_json(paths.manifest, manifest)
                        continue
                    if not should_retry_selected:
                        failed_pages.append(page_number)
                        LOGGER.info(
                            "Retry selection: leaving failed page %s/%s unchanged",
                            page_number,
                            page_count,
                        )
                        manifest["completed_pages"] = page_number
                        manifest["successful_pages"] = successful_pages
                        manifest["failed_pages"] = failed_pages
                        manifest["fallback_pages"] = fallback_pages
                        manifest["visual_review_pages"] = visual_review_pages
                        atomic_write_json(paths.manifest, manifest)
                        continue
                    LOGGER.info("Resume: retrying failed page %s/%s", page_number, page_count)

                page = document.load_page(page_index)
                facts = document_page_facts(document_facts, page_index)
                extraction_error: str | None = None
                try:
                    native_text = page.get_text("text", sort=True).strip()
                    blocks = _text_blocks(page)
                except Exception as exc:
                    LOGGER.exception(
                        "Native extraction failed; visual conversion will continue: %s page %s",
                        source_name,
                        page_number,
                    )
                    native_text = ""
                    blocks = []
                    extraction_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
                raw_path = paths.raw / f"{page_stem}.txt"
                blocks_path = paths.raw / f"{page_stem}.blocks.json"
                native_md_path = paths.raw / f"{page_stem}.native.md"
                native_ir_path = paths.raw / f"{page_stem}.ir.json"
                native_diagnostics_path = paths.raw / f"{page_stem}.native-diagnostics.json"
                parser_ir_path = paths.raw / f"{page_stem}.parser.ir.json"
                parser_structured_path = paths.raw / f"{page_stem}.parser.json"
                debug_overlay_path = paths.debug / f"{page_stem}-layout.jpg"
                image_path = paths.pages / (
                    f"{page_stem}.jpg"
                    if settings.page_image_format == "JPEG"
                    else f"{page_stem}.png"
                )

                atomic_write_text(raw_path, native_text + ("\n" if native_text else ""))
                try:
                    native_result = render_native_page(
                        page,
                        native_text,
                        blocks,
                        facts,
                        document_profile,
                    )
                except Exception as exc:
                    LOGGER.exception(
                        "Structured native rendering failed; retaining flat exact text: %s page %s",
                        source_name,
                        page_number,
                    )
                    fallback_body = _native_blocks_to_markdown(blocks, native_text)
                    diagnostics = {
                        "schema_version": 1,
                        "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
                        "native_processor_config_sha256": NATIVE_PROCESSOR_CONFIG_SHA256,
                        "render_mode": "flat_exception_fallback",
                        "needs_visual_parser": True,
                        "reason_codes": ["native_structured_render_error"],
                        "warnings": [f"{type(exc).__name__}: {str(exc)[:1000]}"],
                    }
                    native_result = NativeRenderResult(
                        markdown=fallback_body,
                        page_ir={
                            "schema_version": 1,
                            "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
                            "native_processor_config_sha256": NATIVE_PROCESSOR_CONFIG_SHA256,
                            "pdf_page_number": page_number,
                            "printed_page_label": facts.get("printed_page_label"),
                            "coordinate_space": "mupdf_unrotated",
                            "outline_path": facts.get("outline_path") or [],
                            "outline_section": (facts.get("outline_path") or [None])[-1],
                            "blocks": [],
                            "diagnostics": diagnostics,
                        },
                        diagnostics=diagnostics,
                    )
                atomic_write_text(
                    native_md_path,
                    native_result.markdown + ("\n" if native_result.markdown else ""),
                )
                atomic_write_json(native_ir_path, native_result.page_ir)
                atomic_write_json(native_diagnostics_path, native_result.diagnostics)
                if (
                    settings.native_debug_overlays
                    and native_result.diagnostics.get("needs_visual_parser")
                ):
                    atomic_write_bytes(
                        debug_overlay_path,
                        render_debug_overlay(page, native_result.page_ir),
                    )
                atomic_write_json(
                    blocks_path,
                    {
                        "schema_version": 2,
                        "pdf_page_number": page_number,
                        "width_points": page.rect.width,
                        "height_points": page.rect.height,
                        "coordinate_space": "mupdf_unrotated",
                        "page_facts": facts,
                        "blocks": blocks,
                        "extraction_error": extraction_error,
                    },
                )

                render_error: Exception | None = None
                try:
                    pixmap = page.get_pixmap(dpi=settings.render_dpi, alpha=False)
                    if settings.page_image_format == "JPEG":
                        image_bytes = pixmap.tobytes(
                            "jpeg", jpg_quality=settings.page_jpeg_quality
                        )
                        image_mime_type = "image/jpeg"
                    else:
                        image_bytes = pixmap.tobytes("png")
                        image_mime_type = "image/png"
                    if len(image_bytes) > MAX_INLINE_IMAGE_BYTES:
                        if image_mime_type == "image/png":
                            LOGGER.warning(
                                "PNG for %s page %s is %.1f MB; using JPEG for inline limits",
                                source_name,
                                page_number,
                                len(image_bytes) / (1024 * 1024),
                            )
                            image_bytes = pixmap.tobytes(
                                "jpeg", jpg_quality=settings.page_jpeg_quality
                            )
                            image_mime_type = "image/jpeg"
                            image_path = paths.pages / f"{page_stem}.jpg"
                        elif settings.page_jpeg_quality > 80:
                            LOGGER.warning(
                                "JPEG for %s page %s is %.1f MB; retrying at quality 80",
                                source_name,
                                page_number,
                                len(image_bytes) / (1024 * 1024),
                            )
                            image_bytes = pixmap.tobytes("jpeg", jpg_quality=80)
                    if len(image_bytes) > MAX_INLINE_IMAGE_BYTES:
                        raise RuntimeError(
                            f"Rendered page remains too large for inline upload: "
                            f"{len(image_bytes) / (1024 * 1024):.1f} MB"
                        )
                    if settings.keep_page_images:
                        atomic_write_bytes(image_path, image_bytes)
                except Exception as exc:
                    image_bytes = b""
                    image_mime_type = "image/png"
                    render_error = exc

                citation = format_pdf_citation(
                    bibliography,
                    page_number,
                    printed_page_start=facts.get("printed_page_label"),
                )
                common_metadata = {
                    "schema_version": 3,
                    "book_id": book_id,
                    "source_file": source_name,
                    "bibliography": bibliography,
                    "book_title": bibliography["title"],
                    "title_source": bibliography["title_source"],
                    "authors": bibliography.get("authors", []),
                    "authors_source": bibliography.get("authors_source"),
                    "edition": bibliography.get("edition"),
                    "publication_year": bibliography.get("publication_year"),
                    "language": bibliography.get("language"),
                    "source_sha256": source_sha256,
                    "pdf_page_index": page_index,
                    "pdf_page_number": page_number,
                    "pdf_page_count": page_count,
                    "printed_page_label": facts.get("printed_page_label"),
                    "outline_path": facts.get("outline_path") or [],
                    "outline_section": (facts.get("outline_path") or [None])[-1],
                    "citation": citation,
                    "page_width_points": page.rect.width,
                    "page_height_points": page.rect.height,
                    "rotation_degrees": facts.get("rotation_degrees", 0),
                    "coordinate_space": facts.get("coordinate_space", "mupdf_unrotated"),
                    "native_text_present": bool(native_text),
                    "native_text_characters": len(native_text),
                    "native_extraction_error": extraction_error,
                    "native_text_path": relative_posix(raw_path, settings.root),
                    "native_blocks_path": relative_posix(blocks_path, settings.root),
                    "native_markdown_path": relative_posix(native_md_path, settings.root),
                    "native_ir_path": relative_posix(native_ir_path, settings.root),
                    "native_diagnostics_path": relative_posix(
                        native_diagnostics_path, settings.root
                    ),
                    "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
                    "native_processor_config_sha256": native_result.diagnostics.get(
                        "native_processor_config_sha256"
                    ),
                    "native_render_mode": native_result.diagnostics.get("render_mode"),
                    "native_text_quality": native_result.diagnostics.get(
                        "native_text_quality", {}
                    ),
                    "page_complexity": native_result.diagnostics.get(
                        "page_complexity", {}
                    ),
                    "needs_visual_parser": bool(
                        native_result.diagnostics.get("needs_visual_parser")
                    ),
                    "reason_codes": native_result.diagnostics.get("reason_codes", []),
                    "document_facts_path": relative_posix(
                        paths.metadata / "document.json", settings.root
                    ),
                    "native_debug_overlay_path": (
                        relative_posix(debug_overlay_path, settings.root)
                        if settings.native_debug_overlays
                        and native_result.diagnostics.get("needs_visual_parser")
                        else None
                    ),
                    "markdown_path": relative_posix(page_md_path, settings.root),
                    "page_image_path": (
                        relative_posix(image_path, settings.root)
                        if settings.keep_page_images and render_error is None
                        else None
                    ),
                    "page_image_mime_type": (
                        image_mime_type if render_error is None else None
                    ),
                    "page_image_bytes": len(image_bytes) if render_error is None else None,
                    "parser_model": settings.gemini_model,
                    "thinking_level": settings.thinking_level,
                    "parser_backend": "gemini",
                }
                native_route_decision = _native_route_decision(
                    settings, native_result, native_text, extraction_error
                )
                common_metadata["native_router"] = native_route_decision
                llm_call_attempted = False

                try:
                    if native_route_decision["route_native"]:
                        LOGGER.info(
                            "Native router: %s page %s/%s (no LLM request)",
                            source_name,
                            page_number,
                            page_count,
                        )
                        result = PageResult(
                            markdown=native_result.markdown,
                            backend="pymupdf_native",
                            model=f"pymupdf-structured-v{NATIVE_EXTRACTOR_VERSION}",
                            diagnostics=native_result.diagnostics,
                            warnings=tuple(
                                native_result.diagnostics.get("warnings") or []
                            ),
                            model_version=NATIVE_EXTRACTOR_VERSION,
                            usage_metadata=None,
                            finish_reason="NOT_CALLED",
                        )
                    else:
                        if render_error is not None:
                            raise RuntimeError(
                                f"Page rendering failed: {type(render_error).__name__}: {render_error}"
                            ) from render_error
                        LOGGER.info(
                            "Gemini: %s page %s/%s",
                            source_name,
                            page_number,
                            page_count,
                        )
                        reference_text = (
                            native_text
                            if native_result.diagnostics.get(
                                "native_text_quality", {}
                            ).get("usable_as_reference", True)
                            else ""
                        )
                        page_input = PageInput(
                            image_bytes=image_bytes,
                            image_mime_type=image_mime_type,
                            page_number=page_number,
                            native_text=reference_text,
                            native_blocks=blocks,
                            page_facts=facts,
                            document_facts=document_facts,
                        )
                        llm_call_attempted = True
                        if hasattr(converter, "convert_page"):
                            result = converter.convert_page(page_input)
                        else:  # Backward-compatible test/custom converter contract.
                            result = converter.convert(
                                image_bytes,
                                reference_text,
                                page_number,
                                image_mime_type=image_mime_type,
                            )
                    result_backend = str(getattr(result, "backend", None) or "gemini")
                    result_model = str(
                        getattr(result, "model", None) or settings.gemini_model
                    )
                    result_page_ir = getattr(result, "page_ir", None)
                    result_structured = getattr(result, "structured_data", None)
                    if result_page_ir is not None:
                        atomic_write_json(parser_ir_path, result_page_ir)
                    else:
                        parser_ir_path.unlink(missing_ok=True)
                    if result_structured is not None:
                        atomic_write_json(parser_structured_path, result_structured)
                    else:
                        parser_structured_path.unlink(missing_ok=True)
                    front_matter = _page_front_matter(
                        source_name,
                        book_id,
                        page_number,
                        page_count,
                        citation,
                        result_model,
                        "complete",
                        bibliography,
                        facts,
                        parser_backend=result_backend,
                        conversion_route=(
                            "native_structured"
                            if native_route_decision["route_native"]
                            else "visual_llm"
                        ),
                        llm_called=llm_call_attempted,
                    )
                    page_markdown = front_matter + result.markdown.strip() + "\n"
                    metadata = {
                        **common_metadata,
                        "status": "complete",
                        "finish_reason": result.finish_reason,
                        "response_id": result.response_id,
                        "model_version": result.model_version,
                        "usage_metadata": result.usage_metadata,
                        "parser_backend": result_backend,
                        "parser_model": result_model,
                        "parser_ir_path": (
                            relative_posix(parser_ir_path, settings.root)
                            if result_page_ir is not None
                            else None
                        ),
                        "parser_structured_path": (
                            relative_posix(parser_structured_path, settings.root)
                            if result_structured is not None
                            else None
                        ),
                        "parser_diagnostics": getattr(result, "diagnostics", {}) or {},
                        "parser_warnings": list(getattr(result, "warnings", ()) or ()),
                        "conversion_route": (
                            "native_structured"
                            if native_route_decision["route_native"]
                            else "visual_llm"
                        ),
                        "llm_called": llm_call_attempted,
                        "layout_verified": False,
                        "thinking_level": (
                            None
                            if native_route_decision["route_native"]
                            else settings.thinking_level
                        ),
                        "markdown_sha256": sha256_text(result.markdown),
                        "completed_at": utc_now(),
                    }
                    successful_pages += 1
                    consecutive_page_failures = 0
                except Exception as exc:
                    finish_reason = getattr(exc, "finish_reason", None)
                    response_snapshot = getattr(exc, "response_snapshot", None)
                    error_message = " ".join(str(exc).split())[:1000]
                    if finish_reason == "RECITATION" and native_text:
                        LOGGER.warning(
                            "Gemini RECITATION on %s page %s/%s; using native text fallback",
                            source_name,
                            page_number,
                            page_count,
                        )
                        fallback_body = native_result.markdown
                        front_matter = _page_front_matter(
                            source_name,
                            book_id,
                            page_number,
                            page_count,
                            citation,
                            settings.gemini_model,
                            "native_text_fallback",
                            bibliography,
                            facts,
                            parser_backend="gemini",
                            conversion_route="native_fallback_after_llm",
                            llm_called=llm_call_attempted,
                        )
                        page_markdown = (
                            front_matter
                            + f"> [!warning] Native-text fallback for PDF page {page_number}\n"
                            + f"> Gemini finish reason: `{finish_reason}`\n"
                            + f"> API message: {error_message}\n"
                            + f"> Full response details: `metadata/{book_id}/{page_stem}.json`\n"
                            + "> Text is searchable and citable, but visual layout and image descriptions require review.\n\n"
                            + fallback_body
                            + "\n"
                        )
                        metadata = {
                            **common_metadata,
                            "status": "native_text_fallback",
                            "finish_reason": finish_reason,
                            "fallback_source": "pymupdf_structured_v1",
                            "layout_verified": False,
                            "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
                            "native_diagnostics": native_result.diagnostics,
                            "conversion_route": "native_fallback_after_llm",
                            "llm_called": llm_call_attempted,
                            "error_type": type(exc).__name__,
                            "error": error_message,
                            "gemini_error_response": response_snapshot,
                            "markdown_sha256": sha256_text(fallback_body),
                            "completed_at": utc_now(),
                        }
                        fallback_pages.append(page_number)
                        consecutive_page_failures = 0
                    else:
                        LOGGER.exception(
                            "Page failed but book will continue: %s page %s/%s",
                            source_name,
                            page_number,
                            page_count,
                        )
                        front_matter = _page_front_matter(
                            source_name,
                            book_id,
                            page_number,
                            page_count,
                            citation,
                            settings.gemini_model,
                            "failed",
                            bibliography,
                            facts,
                            parser_backend="gemini",
                            conversion_route=(
                                "visual_llm_failed"
                                if llm_call_attempted
                                else "visual_render_failed_before_llm"
                            ),
                            llm_called=llm_call_attempted,
                        )
                        reason_detail = finish_reason or "not provided"
                        page_markdown = (
                            front_matter
                            + "# Page transcription unavailable\n\n"
                            + f"> [!failure] Page conversion failed for PDF page {page_number}\n"
                            + f"> Source: {citation}\n"
                            + f"> Error type: `{type(exc).__name__}`\n"
                            + f"> Finish reason: `{reason_detail}`\n"
                            + f"> API message: {error_message}\n"
                            + f"> Full response details: `metadata/{book_id}/{page_stem}.json`\n\n"
                            + "Native extraction artifacts remain available; the rendered image is retained when configured.\n"
                        )
                        metadata = {
                            **common_metadata,
                            "status": "failed",
                            "finish_reason": finish_reason,
                            "error_type": type(exc).__name__,
                            "error": error_message,
                            "gemini_error_response": response_snapshot,
                            "conversion_route": (
                                "visual_llm_failed"
                                if llm_call_attempted
                                else "visual_render_failed_before_llm"
                            ),
                            "llm_called": llm_call_attempted,
                            "failed_at": utc_now(),
                        }
                        failed_pages.append(page_number)
                        consecutive_page_failures += 1

                atomic_write_text(page_md_path, page_markdown)
                atomic_write_json(page_meta_path, metadata)
                if (
                    metadata.get("conversion_route") == "native_structured"
                    and page_number not in native_routed_pages
                ):
                    native_routed_pages.append(page_number)
                if metadata.get("needs_visual_parser") and page_number not in visual_review_pages:
                    visual_review_pages.append(page_number)
                if settings.keep_page_images and metadata.get("page_image_path"):
                    obsolete_suffix = ".png" if image_path.suffix.lower() == ".jpg" else ".jpg"
                    (paths.pages / f"{page_stem}{obsolete_suffix}").unlink(missing_ok=True)
                manifest["completed_pages"] = page_number
                manifest["successful_pages"] = successful_pages
                manifest["failed_pages"] = failed_pages
                manifest["fallback_pages"] = fallback_pages
                manifest["native_routed_pages"] = native_routed_pages
                manifest["visual_review_pages"] = visual_review_pages
                manifest["updated_at"] = utc_now()
                atomic_write_json(paths.manifest, manifest)
                if (
                    consecutive_page_failures
                    >= settings.max_consecutive_page_failures
                ):
                    raise RuntimeError(
                        f"Stopped after {consecutive_page_failures} consecutive page failures; "
                        "this usually indicates a systemic API, model, or configuration problem"
                    )

            chunk_summary = _write_combined_outputs(
                paths, source_name, page_count, settings, license_info, bibliography
            )
            final_status = "partial" if failed_pages or fallback_pages else "complete"
            manifest.update(
                {
                    "status": final_status,
                    "completed_pages": page_count,
                    "successful_pages": successful_pages,
                    "failed_pages": failed_pages,
                    "fallback_pages": fallback_pages,
                    "native_routed_pages": native_routed_pages,
                    "visual_review_pages": visual_review_pages,
                    "native_router_summary": chunk_summary["native_router_summary"],
                    "completed_at": utc_now(),
                    "outputs": {
                        "raw_book": relative_posix(paths.raw / "book.txt", settings.root),
                        "markdown_book": relative_posix(paths.md / "book.md", settings.root),
                        "pages_metadata": relative_posix(
                            paths.metadata / "pages.jsonl", settings.root
                        ),
                        "chunks_directory": relative_posix(paths.chunks, settings.root),
                        "chunks_metadata": relative_posix(
                            paths.chunks / "chunks.jsonl", settings.root
                        ),
                        "document_facts": relative_posix(
                            paths.metadata / "document.json", settings.root
                        ),
                        "document_profile": relative_posix(
                            paths.metadata / "document-profile.json", settings.root
                        ),
                        "chunk_count": chunk_summary["chunk_count"],
                    },
                }
            )
            atomic_write_json(paths.manifest, manifest)

        if settings.move_completed_pdfs and pdf_path.parent == settings.inbox_dir.resolve():
            destination = settings.processed_dir / f"{book_id}.pdf"
            if not destination.exists():
                shutil.move(str(pdf_path), str(destination))
            manifest["processed_pdf"] = relative_posix(destination, settings.root)
            if license_sidecar is not None and license_sidecar.parent == settings.inbox_dir.resolve():
                license_destination = settings.processed_dir / f"{book_id}.license.json"
                if not license_destination.exists():
                    shutil.move(str(license_sidecar), str(license_destination))
                manifest["processed_license_sidecar"] = relative_posix(
                    license_destination, settings.root
                )
            atomic_write_json(paths.manifest, manifest)

        LOGGER.info("%s: %s -> %s", manifest["status"].title(), source_name, book_id)
        return manifest
    except Exception as exc:
        if not lock_acquired:
            raise
        error_manifest: dict[str, Any]
        if paths.manifest.exists():
            error_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        else:
            error_manifest = {
                "schema_version": 1,
                "book_id": book_id,
                "source_file": source_name,
                "source_sha256": source_sha256,
            }
        error_manifest.update({"status": "failed", "failed_at": utc_now(), "error": str(exc)})
        atomic_write_json(paths.manifest, error_manifest)
        raise
    finally:
        if converter is not None:
            try:
                converter.close()
            except Exception:
                LOGGER.debug("Could not close Gemini client", exc_info=True)
        if lock_acquired:
            lock_path.unlink(missing_ok=True)


def _is_stable(path: Path, stable_seconds: int) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    return stat.st_size > 0 and (time.time() - stat.st_mtime) >= stable_seconds


def scan_once(settings: Settings) -> tuple[int, int]:
    completed = 0
    failed = 0
    for pdf_path in sorted(settings.inbox_dir.glob("*.pdf")):
        if not _is_stable(pdf_path, settings.file_stable_seconds):
            LOGGER.info("Waiting for file copy to finish: %s", pdf_path.name)
            continue
        try:
            ingest_pdf(pdf_path, settings)
            completed += 1
        except Exception:
            failed += 1
            LOGGER.exception("Failed: %s", pdf_path.name)
            try:
                digest = sha256_file(pdf_path)[:10]
                destination = settings.failed_dir / f"{safe_slug(pdf_path.stem)}-{digest}.pdf"
                if not destination.exists():
                    shutil.move(str(pdf_path), str(destination))
            except Exception:
                LOGGER.exception("Could not move failed PDF: %s", pdf_path)
    return completed, failed


def watch(settings: Settings) -> None:
    LOGGER.info("Watching %s every %ss. Press Ctrl+C to stop.", settings.inbox_dir, settings.poll_seconds)
    while True:
        scan_once(settings)
        time.sleep(settings.poll_seconds)
