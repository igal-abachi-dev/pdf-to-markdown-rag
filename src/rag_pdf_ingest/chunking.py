from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bibliography import format_pdf_citation, resolve_bibliography
from .utils import atomic_write_text, sha256_text, utc_now, yaml_string


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
FENCE_OPEN_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
MARKDOWN_TABLE_DELIMITER_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$"
)


@dataclass(frozen=True)
class Block:
    text: str
    page: int
    section: str
    structural_block_type: str | None = None


def approximate_tokens(text: str) -> int:
    """Cheap multilingual token estimate used only for local chunk sizing."""
    return len(TOKEN_RE.findall(text))


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    if approximate_tokens(text) <= max_tokens:
        return [text]
    stripped = text.lstrip()
    lines = text.splitlines()
    is_markdown_table = len(lines) >= 2 and bool(
        re.match(r"^\s*\|?\s*:?-{3,}", lines[1])
    )
    if is_markdown_table or stripped.startswith("<table") or stripped.startswith("```"):
        # An oversized structural block is safer than a malformed table/code block.
        return [text]
    if len(lines) > 1:
        pieces: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0
        for line in lines:
            line_tokens = approximate_tokens(line)
            if current_lines and current_tokens + line_tokens > max_tokens:
                pieces.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens
        if current_lines:
            pieces.append("\n".join(current_lines))
        return pieces
    sentences = re.split(r"(?<=[.!?\u05c3\u3002\uff01\uff1f])\s+", text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = approximate_tokens(sentence)
        if sentence_tokens > max_tokens:
            words = sentence.split()
            start = 0
            while start < len(words):
                take: list[str] = []
                count = 0
                while start < len(words):
                    word_tokens = approximate_tokens(words[start])
                    if take and count + word_tokens > max_tokens:
                        break
                    take.append(words[start])
                    count += word_tokens
                    start += 1
                if current:
                    pieces.append(" ".join(current))
                    current = []
                    current_tokens = 0
                pieces.append(" ".join(take))
            continue
        if current and current_tokens + sentence_tokens > max_tokens:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        pieces.append(" ".join(current))
    return [piece for piece in pieces if piece.strip()]


def _fenced_block_spans(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        opening = FENCE_OPEN_RE.match(lines[index].rstrip("\r\n"))
        if opening is None:
            index += 1
            continue
        fence = opening.group("fence")
        closing_re = re.compile(
            rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$"
        )
        end_index = index + 1
        while end_index < len(lines):
            if closing_re.match(lines[end_index].rstrip("\r\n")):
                end_index += 1
                break
            end_index += 1
        start = offsets[index]
        end = offsets[end_index] if end_index < len(offsets) else len(text)
        spans.append((start, end, "fenced_block"))
        index = end_index
    return spans


def _protected_structural_spans(text: str) -> list[tuple[int, int, str]]:
    # Fences win over HTML detection so a literal <table> example inside a
    # code fence remains one fenced block rather than becoming nested spans.
    spans = _fenced_block_spans(text)
    for match in HTML_TABLE_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end, _ in spans):
            continue
        spans.append((match.start(), match.end(), "html_table"))
    return sorted(spans, key=lambda item: item[0])


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index].rstrip("\r\n")
    delimiter = lines[index + 1].rstrip("\r\n")
    return "|" in header and bool(MARKDOWN_TABLE_DELIMITER_RE.match(delimiter))


def _plain_segments(text: str) -> list[tuple[str, str | None]]:
    """Split prose while keeping complete Markdown tables atomic."""
    lines = text.splitlines(keepends=True)
    segments: list[tuple[str, str | None]] = []
    plain_lines: list[str] = []

    def flush_plain() -> None:
        if not plain_lines:
            return
        plain = "".join(plain_lines)
        plain_lines.clear()
        for paragraph in re.split(r"(?:\r?\n){2,}", plain.strip()):
            if paragraph.strip():
                segments.append((paragraph.strip(), None))

    index = 0
    while index < len(lines):
        if not _is_markdown_table_start(lines, index):
            plain_lines.append(lines[index])
            index += 1
            continue
        flush_plain()
        end = index + 2
        while end < len(lines):
            row = lines[end].rstrip("\r\n")
            if not row.strip() or "|" not in row:
                break
            end += 1
        table = "".join(lines[index:end]).strip()
        segments.append((table, "markdown_table"))
        index = end
    flush_plain()
    return segments


def _page_segments(markdown: str) -> list[tuple[str, str | None]]:
    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for start, end, structural_type in _protected_structural_spans(markdown):
        if start > cursor:
            segments.extend(_plain_segments(markdown[cursor:start]))
        structural_text = markdown[start:end].strip()
        if structural_text:
            segments.append((structural_text, structural_type))
        cursor = end
    if cursor < len(markdown):
        segments.extend(_plain_segments(markdown[cursor:]))
    return segments


def _blocks_from_pages(pages: list[tuple[int, str]]) -> list[Block]:
    blocks: list[Block] = []
    section = ""
    for page_number, markdown in pages:
        for segment, structural_type in _page_segments(markdown.strip()):
            segment = segment.strip()
            if not segment:
                continue
            heading = HEADING_RE.match(segment.splitlines()[0])
            if heading:
                section = heading.group(2).strip()
            blocks.append(Block(segment, page_number, section, structural_type))
    return blocks


def _tail_overlap(blocks: list[Block], overlap_tokens: int) -> list[Block]:
    if overlap_tokens <= 0:
        return []
    tail: list[Block] = []
    count = 0
    for block in reversed(blocks):
        tail.insert(0, block)
        count += approximate_tokens(block.text)
        if count >= overlap_tokens:
            break
    return tail


def _make_chunks(
    blocks: list[Block], target_tokens: int, max_tokens: int, overlap_tokens: int
) -> list[list[Block]]:
    expanded: list[Block] = []
    for block in blocks:
        if not block.text.strip():
            continue
        pieces = (
            [block.text]
            if block.structural_block_type is not None
            else _split_long_text(block.text, max_tokens)
        )
        for piece in pieces:
            if piece.strip():
                expanded.append(
                    Block(
                        piece,
                        block.page,
                        block.section,
                        block.structural_block_type,
                    )
                )

    chunks: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0
    for block in expanded:
        block_tokens = approximate_tokens(block.text)
        first_line = next(iter(block.text.splitlines()), "")
        starts_heading = bool(HEADING_RE.match(first_line))
        page_gap = bool(current and block.page > current[-1].page + 1)
        should_break = bool(
            current
            and (
                page_gap
                or current_tokens + block_tokens > max_tokens
                or current_tokens >= target_tokens
                or (starts_heading and current_tokens >= target_tokens // 2)
            )
        )
        if should_break:
            chunks.append(current)
            current = [] if page_gap else _tail_overlap(current, overlap_tokens)
            current_tokens = sum(approximate_tokens(item.text) for item in current)
            while current and current_tokens + block_tokens > max_tokens:
                removed = current.pop(0)
                current_tokens -= approximate_tokens(removed.text)
        current.append(block)
        current_tokens += block_tokens
    if current:
        chunks.append(current)
    return chunks


def _render_chunk_body(blocks: list[Block]) -> str:
    parts: list[str] = []
    current_page: int | None = None
    for block in blocks:
        if block.page != current_page:
            parts.append(f"[p. {block.page}]")
            current_page = block.page
        parts.append(block.text)
    return "\n\n".join(parts).strip()


def _structural_metadata(
    blocks: list[Block], *, body_tokens: int, max_tokens: int
) -> tuple[str | None, list[str], bool, str | None]:
    structural_types = list(
        dict.fromkeys(
            block.structural_block_type
            for block in blocks
            if block.structural_block_type is not None
        )
    )
    structural_block_type = (
        None
        if not structural_types
        else structural_types[0]
        if len(structural_types) == 1
        else "mixed"
    )
    oversized = body_tokens > max_tokens
    oversized_reason: str | None = None
    if oversized:
        oversized_atomic_types = list(
            dict.fromkeys(
                block.structural_block_type
                for block in blocks
                if block.structural_block_type is not None
                and approximate_tokens(block.text) > max_tokens
            )
        )
        if len(oversized_atomic_types) == 1:
            oversized_reason = (
                f"atomic_{oversized_atomic_types[0]}_exceeds_chunk_max"
            )
        elif len(oversized_atomic_types) > 1:
            oversized_reason = "multiple_atomic_structural_blocks_exceed_chunk_max"
        else:
            # _make_chunks sizes prose before rendering. Visible [p. N] markers
            # can therefore push an otherwise compliant chunk just over max.
            oversized_reason = "rendered_page_markers_exceed_chunk_max"
    return structural_block_type, structural_types, oversized, oversized_reason


def chunk_content_sha256(
    *,
    book_id: str,
    page_start: int,
    page_end: int,
    section: str,
    body: str,
) -> str:
    canonical_content = json.dumps(
        {
            "schema": "rag-chunk-content-v1",
            "book_id": book_id,
            "page_start": page_start,
            "page_end": page_end,
            "section": section,
            "body": body,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical_content)


def build_chunks(
    *,
    pages: list[tuple[int, str]],
    output_dir: Path,
    source_name: str,
    book_id: str,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    license_status: str = "authorized",
    page_quality: dict[int, str] | None = None,
    bibliography: dict[str, Any] | None = None,
    page_context: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_chunk in output_dir.glob("chunk-*.md"):
        old_chunk.unlink()

    chunks = _make_chunks(
        _blocks_from_pages(pages),
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    records: list[dict[str, Any]] = []
    page_quality = page_quality or {}
    page_context = page_context or {}
    bibliography = dict(bibliography or resolve_bibliography({}, source_name))
    bibliography.setdefault(
        "authors_source",
        bibliography.get("title_source") if bibliography.get("authors") else None,
    )
    for index, chunk_blocks in enumerate(chunks, start=1):
        page_start = min(block.page for block in chunk_blocks)
        page_end = max(block.page for block in chunk_blocks)
        markdown_section = next(
            (block.section for block in reversed(chunk_blocks) if block.section), ""
        )
        outline_path = next(
            (
                list(page_context[page].get("outline_path") or [])
                for page in sorted({block.page for block in chunk_blocks}, reverse=True)
                if page_context.get(page, {}).get("outline_path")
            ),
            [],
        )
        outline_section = str(outline_path[-1]) if outline_path else ""
        section = outline_section or markdown_section
        printed_page_start = page_context.get(page_start, {}).get("printed_page_label")
        printed_page_end = page_context.get(page_end, {}).get("printed_page_label")
        citation = format_pdf_citation(
            bibliography,
            page_start,
            page_end,
            printed_page_start=printed_page_start,
            printed_page_end=printed_page_end,
        )
        body = _render_chunk_body(chunk_blocks)
        body_tokens = approximate_tokens(body)
        (
            structural_block_type,
            structural_block_types,
            oversized,
            oversized_reason,
        ) = _structural_metadata(
            chunk_blocks, body_tokens=body_tokens, max_tokens=max_tokens
        )
        included_pages = sorted({block.page for block in chunk_blocks})
        fallback_pages = [
            page
            for page in included_pages
            if page_quality.get(page) == "native_text_fallback"
        ]
        native_structured_pages = [
            page
            for page in included_pages
            if page_context.get(page, {}).get("conversion_route")
            == "native_structured"
        ]
        visual_review_pages = [
            page
            for page in included_pages
            if page_context.get(page, {}).get("needs_visual_parser")
        ]
        reason_codes = list(
            dict.fromkeys(
                str(reason)
                for page in included_pages
                for reason in page_context.get(page, {}).get("reason_codes", [])
            )
        )
        if len(fallback_pages) == len(included_pages):
            source_quality = "native_text_fallback"
        elif len(native_structured_pages) == len(included_pages):
            source_quality = "native_structured"
        elif not fallback_pages and not native_structured_pages:
            source_quality = "gemini_visual"
        elif fallback_pages and not native_structured_pages:
            source_quality = "mixed_gemini_and_native_text_fallback"
        elif native_structured_pages and not fallback_pages:
            source_quality = "mixed_gemini_and_native_structured"
        else:
            source_quality = "mixed_gemini_native_structured_and_native_text_fallback"
        content_sha256 = chunk_content_sha256(
            book_id=book_id,
            page_start=page_start,
            page_end=page_end,
            section=section,
            body=body,
        )
        chunk_id = f"{book_id}-sha256-{content_sha256}"
        filename = (
            f"chunk-{content_sha256[:20]}-pp-{page_start:04d}-{page_end:04d}.md"
        )
        source_marker = f"[Source: {source_name}, PDF pages {page_start}-{page_end}]"
        front_matter = (
            "---\n"
            f"title: {yaml_string(citation)}\n"
            f"book_id: {yaml_string(book_id)}\n"
            f"source_file: {yaml_string(source_name)}\n"
            f"book_title: {yaml_string(str(bibliography['title']))}\n"
            f"title_source: {yaml_string(str(bibliography['title_source']))}\n"
            f"authors: {json.dumps(bibliography.get('authors', []), ensure_ascii=False)}\n"
            f"authors_source: {yaml_string(str(bibliography.get('authors_source') or ''))}\n"
            f"edition: {yaml_string(str(bibliography.get('edition') or ''))}\n"
            f"publication_year: {yaml_string(str(bibliography.get('publication_year') or ''))}\n"
            f"language: {yaml_string(str(bibliography.get('language') or ''))}\n"
            f"chunk_id: {yaml_string(chunk_id)}\n"
            f"page_start: {page_start}\n"
            f"page_end: {page_end}\n"
            f"printed_page_start: {yaml_string(str(printed_page_start or ''))}\n"
            f"printed_page_end: {yaml_string(str(printed_page_end or ''))}\n"
            f"section: {yaml_string(section)}\n"
            f"outline_section: {yaml_string(outline_section)}\n"
            f"markdown_section: {yaml_string(markdown_section)}\n"
            f"outline_path: {json.dumps(outline_path, ensure_ascii=False)}\n"
            f"citation: {yaml_string(citation)}\n"
            f"license_status: {yaml_string(license_status)}\n"
            f"source_quality: {yaml_string(source_quality)}\n"
            f"fallback_pages: {json.dumps(fallback_pages)}\n"
            f"native_structured_pages: {json.dumps(native_structured_pages)}\n"
            f"visual_review_pages: {json.dumps(visual_review_pages)}\n"
            f"reason_codes: {json.dumps(reason_codes, ensure_ascii=False)}\n"
            f"structural_block_type: {yaml_string(structural_block_type or '')}\n"
            f"structural_block_types: {json.dumps(structural_block_types)}\n"
            f"oversized: {str(oversized).lower()}\n"
            f"oversized_reason: {yaml_string(oversized_reason or '')}\n"
            "tags:\n"
            "  - rag-chunk\n"
            "  - pdf-book\n"
            "---\n\n"
        )
        callout = (
            f"> [!cite] {citation}\n"
            f"> Exact pages: [[md/{book_id}/page-{page_start:04d}|PDF p. {page_start}]]"
        )
        if page_end != page_start:
            callout += f" through [[md/{book_id}/page-{page_end:04d}|PDF p. {page_end}]]"
        markdown = f"{front_matter}{source_marker}\n\n{callout}\n\n{body}\n"
        path = output_dir / filename
        atomic_write_text(path, markdown)
        records.append(
            {
                "schema_version": 5,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "book_id": book_id,
                "source_file": source_name,
                "book_title": bibliography["title"],
                "title_source": bibliography["title_source"],
                "authors": bibliography.get("authors", []),
                "authors_source": bibliography.get("authors_source"),
                "edition": bibliography.get("edition"),
                "publication_year": bibliography.get("publication_year"),
                "language": bibliography.get("language"),
                "page_start": page_start,
                "page_end": page_end,
                "printed_page_start": printed_page_start,
                "printed_page_end": printed_page_end,
                "section": section,
                "outline_section": outline_section,
                "markdown_section": markdown_section,
                "outline_path": outline_path,
                "citation": citation,
                "license_status": license_status,
                "included_pages": included_pages,
                "fallback_pages": fallback_pages,
                "native_structured_pages": native_structured_pages,
                "visual_review_pages": visual_review_pages,
                "needs_visual_review": bool(visual_review_pages),
                "reason_codes": reason_codes,
                "has_native_text_fallback": bool(fallback_pages),
                "has_native_structured": bool(native_structured_pages),
                "source_quality": source_quality,
                "structural_block_type": structural_block_type,
                "structural_block_types": structural_block_types,
                "body": body,
                "approximate_tokens": body_tokens,
                "oversized": oversized,
                "oversized_reason": oversized_reason,
                "overlap_target_tokens": overlap_tokens,
                "path": path.name,
                "content_sha256": content_sha256,
                "sha256": content_sha256,
                "artifact_sha256": sha256_text(markdown),
                "created_at": utc_now(),
            }
        )

    atomic_write_text(
        output_dir / "chunks.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    return {
        "chunk_count": len(records),
        "chunks_metadata": str(output_dir / "chunks.jsonl"),
    }
