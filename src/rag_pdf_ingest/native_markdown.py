from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymupdf as fitz

from .pdf_facts import native_text_quality
from .utils import sha256_text


NATIVE_EXTRACTOR_VERSION = "1"
NATIVE_PROCESSOR_CONFIG = {
    "coverage_gate": 0.98,
    "heading_size_ratio": 1.08,
    "image_area_threshold": 0.01,
    "marginalia_top_ratio": 0.10,
    "marginalia_bottom_ratio": 0.87,
    "table_strategy": "pymupdf_find_tables_confidence_gated",
}
NATIVE_PROCESSOR_CONFIG_SHA256 = sha256_text(
    json.dumps(NATIVE_PROCESSOR_CONFIG, sort_keys=True, separators=(",", ":"))
)
if hasattr(fitz, "no_recommend_layout"):
    # This pipeline intentionally uses the built-in deterministic table finder.
    fitz.no_recommend_layout()
LIST_RE = re.compile(r"^\s*(?:[-*+•◦▪‣]|\d+[.)])\s+")
TOKEN_RE = re.compile(r"[\w\u0590-\u05ff\u0600-\u06ff]+", re.UNICODE)


@dataclass(frozen=True)
class NativeRenderResult:
    markdown: str
    page_ir: dict[str, Any]
    diagnostics: dict[str, Any]


def render_debug_overlay(
    page: fitz.Page, page_ir: dict[str, Any], *, dpi: int = 110, quality: int = 82
) -> bytes:
    """Render a non-mutating JPEG showing IR boxes and reading order."""
    colors = {
        "section_header": (0.85, 0.15, 0.15),
        "table": (0.1, 0.45, 0.9),
        "figure": (0.65, 0.2, 0.8),
        "footnote": (0.9, 0.55, 0.05),
        "page_header": (0.4, 0.4, 0.4),
        "page_footer": (0.4, 0.4, 0.4),
    }
    debug_document = fitz.open()
    try:
        debug_document.insert_pdf(
            page.parent, from_page=page.number, to_page=page.number
        )
        debug_page = debug_document[0]
        for block in page_ir.get("blocks", []):
            rect = fitz.Rect(block.get("bbox") or (0, 0, 0, 0))
            if rect.is_empty:
                continue
            block_type = str(block.get("type") or "unknown")
            color = colors.get(block_type, (0.1, 0.65, 0.25))
            debug_page.draw_rect(rect, color=color, width=0.8, overlay=True)
            label = f"{block.get('order', '?')}:{block_type}"
            point = fitz.Point(rect.x0, max(6.0, rect.y0 - 1.5))
            debug_page.insert_text(
                point, label, fontsize=5.5, color=color, overlay=True
            )
        pixmap = debug_page.get_pixmap(dpi=dpi, alpha=False)
        return pixmap.tobytes("jpeg", jpg_quality=quality)
    finally:
        debug_document.close()


def _bbox(value: object) -> list[float]:
    try:
        rect = fitz.Rect(value)
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]
    return [round(float(item), 3) for item in rect]


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_ratio(inner: list[float], outer: list[float]) -> float:
    intersection = [
        max(inner[0], outer[0]),
        max(inner[1], outer[1]),
        min(inner[2], outer[2]),
        min(inner[3], outer[3]),
    ]
    return _area(intersection) / max(1.0, _area(inner))


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _token_coverage(expected: str, observed: str) -> tuple[float, list[str]]:
    expected_counts = Counter(_tokens(expected))
    observed_counts = Counter(_tokens(observed))
    total = sum(expected_counts.values())
    if not total:
        return 1.0, []
    matched = sum(min(count, observed_counts[token]) for token, count in expected_counts.items())
    missing: list[str] = []
    for token, count in expected_counts.items():
        missing.extend([token] * max(0, count - observed_counts[token]))
        if len(missing) >= 30:
            break
    return matched / total, missing[:30]


def _plain_line(line: dict[str, Any]) -> str:
    return "".join(str(span.get("text", "")) for span in line.get("spans", []))


def _styled_span(span: dict[str, Any]) -> str:
    text = str(span.get("text", ""))
    if not text or text.isspace():
        return text
    flags = int(span.get("flags", 0) or 0)
    value = text
    superscript = bool(flags & int(getattr(fitz, "TEXT_FONT_SUPERSCRIPT", 1)))
    if flags & int(getattr(fitz, "TEXT_FONT_MONOSPACED", 8)):
        value = f"`{value.replace('`', 'ˋ')}`"
    else:
        if flags & int(getattr(fitz, "TEXT_FONT_BOLD", 16)):
            value = f"**{value}**"
        if flags & int(getattr(fitz, "TEXT_FONT_ITALIC", 2)):
            value = f"*{value}*"
    if superscript:
        value = f"<sup>{value}</sup>"
    return value


def _vector_page_signals(
    page: fitz.Page, page_width: float, page_height: float, page_area: float
) -> dict[str, Any]:
    drawing_count = 0
    approximate_area = 0.0
    decorative_rules: list[dict[str, Any]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for drawing in drawings:
        drawing_count += 1
        bbox = _bbox(drawing.get("rect"))
        approximate_area += _area(bbox)
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        if (height <= 2.0 and width >= page_width * 0.2) or (
            width <= 2.0 and height >= page_height * 0.2
        ):
            decorative_rules.append(
                {"bbox": bbox, "reason": "thin_vector_rule_not_text_content"}
            )
    return {
        "drawing_count": drawing_count,
        "vector_density": round(min(1.0, approximate_area / max(1.0, page_area)), 6),
        "decorative_rule_count": len(decorative_rules),
        "ignored_vector_artifacts": decorative_rules,
    }


def _styled_line(line: dict[str, Any]) -> str:
    return "".join(_styled_span(span) for span in line.get("spans", [])).strip()


def _block_text(block: dict[str, Any]) -> str:
    return "\n".join(_plain_line(line).strip() for line in block.get("lines", [])).strip()


def _is_rtl(text: str) -> bool:
    rtl = sum("\u0590" <= char <= "\u08ff" for char in text)
    latin = sum(("a" <= char.casefold() <= "z") for char in text)
    return rtl > latin


def _paragraph_markdown(block: dict[str, Any], *, rtl: bool) -> str:
    lines = [_styled_line(line) for line in block.get("lines", [])]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    result = lines[0]
    for line in lines[1:]:
        if not rtl and result.endswith("-") and line[:1].islower():
            result = result[:-1] + line
        else:
            result += " " + line
    return " ".join(result.split())


def _normalized_heading(text: str) -> str:
    return " ".join(_tokens(text))


def _heading_level(
    text: str,
    max_size: float,
    profile: dict[str, Any],
    outline_path: list[str],
) -> tuple[int | None, bool]:
    normalized = _normalized_heading(text)
    outline_match = any(
        normalized
        and (
            normalized == _normalized_heading(title)
            or (
                len(normalized) >= 5
                and normalized in _normalized_heading(title)
            )
        )
        for title in outline_path
    )
    sizes = [float(size) for size in profile.get("heading_font_sizes", [])]
    if outline_match:
        return min(6, max(1, len(outline_path))), True
    if len(text) > 240 or not sizes:
        return None, False
    for index, size in enumerate(sizes, start=1):
        if max_size >= size - 0.15:
            return min(6, index), False
    return None, False


def _stable_block_id(page_number: int, block_type: str, bbox: list[float], text: str) -> str:
    canonical = json.dumps(
        {
            "page": page_number,
            "type": block_type,
            "bbox": [round(value, 1) for value in bbox],
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"blk-{sha256_text(canonical)[:24]}"


def _edge_key(text: str) -> str:
    value = " ".join(text.split()).casefold()
    value = re.sub(r"^\W*\d+\W*|\W*\d+\W*$", "", value)
    return " ".join(value.split())


def _table_markdown(rows: list[list[Any]]) -> str | None:
    if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
        return None
    width = max(len(row) for row in rows)
    normalized = [list(row) + [""] * (width - len(row)) for row in rows]
    if any(value is None for row in normalized for value in row):
        return None

    def cell(value: object) -> str:
        return " ".join(str(value or "").split()).replace("|", "\\|")

    rendered = ["| " + " | ".join(cell(value) for value in normalized[0]) + " |"]
    rendered.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in normalized[1:]:
        rendered.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(rendered)


def _merged_table_html(table: Any, rows: list[list[Any]]) -> str | None:
    try:
        table_rows = list(table.rows)
        row_count = int(table.row_count)
        col_count = int(table.col_count)
    except Exception:
        return None
    if row_count < 1 or col_count < 2 or len(table_rows) != row_count:
        return None
    rects = [
        _bbox(cell)
        for row in table_rows
        for cell in row.cells
        if cell is not None
    ]
    x_bounds = sorted({round(rect[0], 2) for rect in rects} | {round(rect[2], 2) for rect in rects})
    y_bounds = sorted({round(rect[1], 2) for rect in rects} | {round(rect[3], 2) for rect in rects})
    if len(x_bounds) != col_count + 1 or len(y_bounds) != row_count + 1:
        return None

    def index_of(bounds: list[float], value: float) -> int | None:
        nearest = min(range(len(bounds)), key=lambda idx: abs(bounds[idx] - value))
        return nearest if abs(bounds[nearest] - value) <= 1.0 else None

    occupied: set[tuple[int, int]] = set()
    emitted: set[tuple[float, float, float, float]] = set()
    output = ["<table>"]
    for row_index, row in enumerate(table_rows):
        output.append("  <tr>")
        for col_index, cell_rect in enumerate(row.cells):
            if cell_rect is None:
                continue
            rect = tuple(_bbox(cell_rect))
            if rect in emitted:
                continue
            emitted.add(rect)
            x0 = index_of(x_bounds, rect[0])
            x1 = index_of(x_bounds, rect[2])
            y0 = index_of(y_bounds, rect[1])
            y1 = index_of(y_bounds, rect[3])
            if None in {x0, x1, y0, y1} or x1 <= x0 or y1 <= y0:
                return None
            slots = {
                (r, c)
                for r in range(int(y0), int(y1))
                for c in range(int(x0), int(x1))
            }
            if occupied & slots:
                return None
            occupied |= slots
            text = ""
            if row_index < len(rows) and col_index < len(rows[row_index]):
                text = " ".join(str(rows[row_index][col_index] or "").split())
            attrs: list[str] = []
            if int(x1) - int(x0) > 1:
                attrs.append(f'colspan="{int(x1) - int(x0)}"')
            if int(y1) - int(y0) > 1:
                attrs.append(f'rowspan="{int(y1) - int(y0)}"')
            tag = "th" if row_index == 0 else "td"
            attr_text = (" " + " ".join(attrs)) if attrs else ""
            output.append(f"    <{tag}{attr_text}>{html.escape(text)}</{tag}>")
        output.append("  </tr>")
    if len(occupied) != row_count * col_count:
        return None
    output.append("</table>")
    return "\n".join(output)


def _extract_tables(page: fitz.Page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        tables = list(page.find_tables().tables)
    except Exception as exc:
        return [], [{"reason": "table_detection_error", "error": f"{type(exc).__name__}: {exc}"}]
    for index, table in enumerate(tables):
        bbox = _bbox(table.bbox)
        try:
            rows = table.extract() or []
        except Exception as exc:
            rejected.append(
                {"index": index, "bbox": bbox, "reason": "table_extract_error", "error": str(exc)}
            )
            continue
        markdown = _table_markdown(rows)
        structural_type = "markdown_table"
        confidence = 0.95
        if markdown is None:
            markdown = _merged_table_html(table, rows)
            structural_type = "html_table"
            confidence = 0.85
        if markdown is None:
            rejected.append(
                {"index": index, "bbox": bbox, "reason": "low_confidence_table_structure"}
            )
            continue
        accepted.append(
            {
                "index": index,
                "bbox": bbox,
                "markdown": markdown,
                "plain_text": " ".join(
                    " ".join(str(value or "") for value in row) for row in rows
                ),
                "structural_block_type": structural_type,
                "confidence": confidence,
            }
        )
    return accepted, rejected


def _column_order(blocks: list[dict[str, Any]], page_width: float, rtl: bool) -> tuple[list[dict[str, Any]], int, bool]:
    visible = [block for block in blocks if not block.get("ignored_for_output")]
    narrow = [block for block in visible if (block["bbox"][2] - block["bbox"][0]) < page_width * 0.58]
    midpoint = page_width / 2
    left = [block for block in narrow if block["bbox"][2] <= midpoint * 1.08]
    right = [block for block in narrow if block["bbox"][0] >= midpoint * 0.92]
    if len(left) < 2 or len(right) < 2:
        return blocks, 1, False
    narrow_top = min(block["bbox"][1] for block in narrow)
    narrow_bottom = max(block["bbox"][3] for block in narrow)
    interior_wide = [
        block
        for block in visible
        if block not in narrow
        and block["bbox"][1] > narrow_top
        and block["bbox"][3] < narrow_bottom
    ]
    if interior_wide:
        return blocks, 2, True
    prefix = [block for block in blocks if block not in narrow and block["bbox"][3] <= narrow_top]
    suffix = [block for block in blocks if block not in narrow and block["bbox"][1] >= narrow_bottom]
    middle_other = [block for block in blocks if block not in prefix and block not in suffix and block not in narrow]
    if middle_other:
        return blocks, 2, True
    first, second = (right, left) if rtl else (left, right)
    ordered = (
        sorted(prefix, key=lambda block: (block["bbox"][1], block["bbox"][0]))
        + sorted(first, key=lambda block: (block["bbox"][1], block["bbox"][0]))
        + sorted(second, key=lambda block: (block["bbox"][1], block["bbox"][0]))
        + sorted(suffix, key=lambda block: (block["bbox"][1], block["bbox"][0]))
    )
    return ordered, 2, False


def render_native_page(
    page: fitz.Page,
    native_text: str,
    text_blocks: list[dict[str, Any]],
    page_facts: dict[str, Any],
    document_profile: dict[str, Any],
) -> NativeRenderResult:
    page_number = int(page_facts.get("pdf_page_number", page.number + 1))
    outline_path = [str(value) for value in page_facts.get("outline_path", [])]
    page_width = max(1.0, float(page.rect.width))
    page_height = max(1.0, float(page.rect.height))
    page_area = page_width * page_height
    quality = native_text_quality(native_text, text_blocks, page_area)
    accepted_tables, rejected_tables = _extract_tables(page)
    repeated = set(document_profile.get("repeated_marginalia", []))
    body_size = float(document_profile.get("body_font_size", 0.0) or 0.0)
    rtl = _is_rtl(native_text)
    ir_blocks: list[dict[str, Any]] = []
    ignored_text: list[str] = []

    for source_order, block in enumerate(text_blocks):
        text = _block_text(block)
        if not text:
            continue
        bbox = _bbox(block.get("bbox"))
        max_size = max(
            (
                float(span.get("size", 0.0) or 0.0)
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ),
            default=0.0,
        )
        flags = [
            int(span.get("flags", 0) or 0)
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if str(span.get("text", "")).strip()
        ]
        heading_level, outline_match = _heading_level(
            text, max_size, document_profile, outline_path
        )
        block_type = "text"
        if heading_level is not None:
            block_type = "section_header"
        elif LIST_RE.match(text):
            block_type = "list_item"
        elif flags and sum(bool(flag & int(getattr(fitz, "TEXT_FONT_MONOSPACED", 8))) for flag in flags) / len(flags) > 0.7:
            block_type = "code"
        elif bbox[1] >= page_height * 0.78 and body_size and max_size < body_size * 0.88:
            block_type = "footnote"

        edge = bbox[3] <= page_height * 0.1 or bbox[1] >= page_height * 0.87
        repeated_match = edge and _edge_key(text) in repeated
        isolated_page_number = edge and bool(
            re.fullmatch(r"\s*(?:\d+|[ivxlcdm]+)\s*", text, re.IGNORECASE)
        )
        ignored = repeated_match or isolated_page_number
        ignore_reason = (
            "repeated_running_header_or_footer"
            if repeated_match
            else "isolated_printed_page_number"
            if isolated_page_number
            else None
        )
        if ignored:
            ignored_text.append(text)
            block_type = "page_header" if bbox[3] <= page_height * 0.1 else "page_footer"

        markdown = _paragraph_markdown(block, rtl=rtl)
        if block_type == "section_header" and markdown:
            markdown = f"{'#' * int(heading_level or 2)} {markdown}"
        elif block_type == "list_item" and markdown and not markdown.startswith(("- ", "* ", "+ ")):
            markdown = LIST_RE.sub("- ", markdown, count=1)
        elif block_type == "code" and markdown:
            markdown = f"```text\n{text}\n```"

        ir_blocks.append(
            {
                "block_id": _stable_block_id(page_number, block_type, bbox, text),
                "type": block_type,
                "bbox": bbox,
                "source_order": source_order,
                "order": source_order,
                "text": text,
                "markdown": markdown,
                "html": None,
                "structural_block_type": "fenced_block" if block_type == "code" else None,
                "extraction_method": "pymupdf_native",
                "confidence": 0.98 if outline_match else 0.85,
                "heading_level": heading_level,
                "outline_match": outline_match,
                "ignored_for_output": ignored,
                "ignore_reason": ignore_reason,
                "warnings": [],
            }
        )

    for table in accepted_tables:
        table_tokens = Counter(_tokens(table["plain_text"]))
        contained = [
            block
            for block in ir_blocks
            if not block["ignored_for_output"]
            and _intersection_ratio(block["bbox"], table["bbox"]) >= 0.95
        ]
        contained_tokens = Counter(
            token for block in contained for token in _tokens(block["text"])
        )
        accounted = all(table_tokens[token] >= count for token, count in contained_tokens.items())
        if not accounted:
            rejected_tables.append(
                {"index": table["index"], "bbox": table["bbox"], "reason": "table_text_mismatch"}
            )
            continue
        for block in contained:
            block["ignored_for_output"] = True
            block["ignore_reason"] = "represented_by_structured_table"
            ignored_text.append(block["text"])
        ir_blocks.append(
            {
                "block_id": _stable_block_id(page_number, "table", table["bbox"], table["plain_text"]),
                "type": "table",
                "bbox": table["bbox"],
                "source_order": min((block["source_order"] for block in contained), default=10_000 + table["index"]),
                "order": 0,
                "text": table["plain_text"],
                "markdown": table["markdown"],
                "html": table["markdown"] if table["structural_block_type"] == "html_table" else None,
                "structural_block_type": table["structural_block_type"],
                "extraction_method": "pymupdf_native_table",
                "confidence": table["confidence"],
                "table_extraction_method": "pymupdf_native",
                "table_confidence": table["confidence"],
                "table_warnings": [],
                "ignored_for_output": False,
                "ignore_reason": None,
                "warnings": [],
            }
        )

    image_density = 0.0
    significant_images: list[list[float]] = []
    try:
        for image in page.get_image_info(xrefs=True):
            bbox = _bbox(image.get("bbox"))
            ratio = _area(bbox) / page_area
            image_density += ratio
            if ratio >= 0.01 and bbox[2] - bbox[0] >= 12 and bbox[3] - bbox[1] >= 12:
                significant_images.append(bbox)
    except Exception:
        pass
    for index, bbox in enumerate(significant_images):
        description = (
            f"[Image: Figure or illustration present on PDF page {page_number}; "
            "semantic description unavailable in native fallback.]"
        )
        ir_blocks.append(
            {
                "block_id": _stable_block_id(page_number, "figure", bbox, description),
                "type": "figure",
                "bbox": bbox,
                "source_order": 20_000 + index,
                "order": 0,
                "text": "",
                "markdown": description,
                "html": None,
                "structural_block_type": None,
                "extraction_method": "pymupdf_image_inventory",
                "confidence": 0.9,
                "ignored_for_output": False,
                "ignore_reason": None,
                "warnings": ["semantic_image_description_unavailable"],
            }
        )

    # Preserve captions as text, but classify their relationship for auditability.
    for block in ir_blocks:
        if block["type"] != "text" or len(block["text"]) > 300:
            continue
        for image_bbox in significant_images:
            horizontal_overlap = max(
                0.0,
                min(block["bbox"][2], image_bbox[2])
                - max(block["bbox"][0], image_bbox[0]),
            )
            reference_width = max(1.0, min(
                block["bbox"][2] - block["bbox"][0],
                image_bbox[2] - image_bbox[0],
            ))
            vertical_gap = block["bbox"][1] - image_bbox[3]
            if horizontal_overlap / reference_width >= 0.4 and 0 <= vertical_gap <= page_height * 0.04:
                block["type"] = "caption"
                block["confidence"] = 0.8
                block["block_id"] = _stable_block_id(
                    page_number, "caption", block["bbox"], block["text"]
                )
                break

    ir_blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0], block["source_order"]))
    ordered, column_count, ambiguous_columns = _column_order(ir_blocks, page_width, rtl)
    footnotes = [block for block in ordered if block["type"] == "footnote"]
    ordered = [block for block in ordered if block["type"] != "footnote"] + footnotes
    for order, block in enumerate(ordered):
        block["order"] = order

    rendered_parts = [
        str(block.get("markdown") or "").strip()
        for block in ordered
        if not block.get("ignored_for_output") and str(block.get("markdown") or "").strip()
    ]
    markdown = "\n\n".join(rendered_parts).strip()
    nonignored_source = "\n".join(
        block["text"]
        for block in ordered
        if not block.get("ignored_for_output") and block["type"] != "figure"
    )
    coverage, missing = _token_coverage(nonignored_source, markdown)
    render_mode = "structured"
    warnings: list[str] = []
    if coverage < 0.98 and native_text.strip():
        markdown = native_text.strip()
        render_mode = "flat_safety_fallback"
        warnings.append("structured_render_failed_text_coverage_gate")
    reason_codes = list(quality["reason_codes"])
    if rejected_tables:
        reason_codes.append("low_confidence_table_structure")
    if significant_images:
        reason_codes.append("visual_content_requires_description")
    if ambiguous_columns:
        reason_codes.append("ambiguous_multi_column_order")
    reason_codes = list(dict.fromkeys(reason_codes))
    vector_signals = _vector_page_signals(page, page_width, page_height, page_area)
    page_complexity = {
        "text_density": quality["text_density"],
        "image_density": round(image_density, 6),
        "vector_density": vector_signals["vector_density"],
        "table_candidates": len(accepted_tables) + len(rejected_tables),
        "drawing_count": vector_signals["drawing_count"],
        "decorative_rule_count": vector_signals["decorative_rule_count"],
    }

    diagnostics = {
        "schema_version": 1,
        "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
        "native_processor_config_sha256": NATIVE_PROCESSOR_CONFIG_SHA256,
        "render_mode": render_mode,
        "rendered_text_coverage": round(coverage, 6),
        "unmatched_tokens": missing,
        "ignored_text_blocks": len([b for b in ordered if b.get("ignored_for_output")]),
        "column_count": column_count,
        "ambiguous_column_order": ambiguous_columns,
        "table_count": len([b for b in ordered if b["type"] == "table"]),
        "rejected_tables": rejected_tables,
        "significant_image_count": len(significant_images),
        "image_density": round(image_density, 6),
        "native_text_quality": quality,
        "page_complexity": page_complexity,
        "ignored_vector_artifacts": vector_signals["ignored_vector_artifacts"],
        "needs_visual_parser": bool(reason_codes),
        "reason_codes": reason_codes,
        "warnings": warnings,
    }
    page_ir = {
        "schema_version": 1,
        "native_extractor_version": NATIVE_EXTRACTOR_VERSION,
        "native_processor_config_sha256": NATIVE_PROCESSOR_CONFIG_SHA256,
        "pdf_page_number": page_number,
        "printed_page_label": page_facts.get("printed_page_label"),
        "coordinate_space": "mupdf_unrotated",
        "outline_path": outline_path,
        "outline_section": outline_path[-1] if outline_path else None,
        "blocks": ordered,
        "diagnostics": diagnostics,
    }
    return NativeRenderResult(markdown=markdown, page_ir=page_ir, diagnostics=diagnostics)
