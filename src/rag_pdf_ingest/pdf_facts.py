from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pymupdf as fitz

from .utils import json_safe, sha256_text


DOCUMENT_FACTS_SCHEMA_VERSION = 1


def _rect(value: object) -> list[float] | None:
    try:
        rect = fitz.Rect(value)
    except Exception:
        return None
    return [round(float(item), 4) for item in rect]


def _clean_metadata(metadata: object) -> dict[str, str | None]:
    if not isinstance(metadata, dict):
        return {}
    result: dict[str, str | None] = {}
    for key, value in metadata.items():
        if value is None:
            result[str(key)] = None
            continue
        text = str(value).strip()
        result[str(key)] = None if text.casefold() in {"", "none", "null"} else text
    return result


def _permission_facts(document: fitz.Document) -> dict[str, Any]:
    raw = int(getattr(document, "permissions", 0) or 0)
    constants = {
        "print": "PDF_PERM_PRINT",
        "modify": "PDF_PERM_MODIFY",
        "copy": "PDF_PERM_COPY",
        "annotate": "PDF_PERM_ANNOTATE",
        "form": "PDF_PERM_FORM",
        "accessibility": "PDF_PERM_ACCESSIBILITY",
        "assemble": "PDF_PERM_ASSEMBLE",
        "print_high_quality": "PDF_PERM_PRINT_HQ",
    }
    decoded: dict[str, bool | None] = {}
    for name, constant_name in constants.items():
        flag = getattr(fitz, constant_name, None)
        decoded[name] = None if flag is None else bool(raw & int(flag))
    return {"raw": raw, "decoded": decoded}


def _normalize_outline(document: fitz.Document) -> list[dict[str, Any]]:
    try:
        raw_toc = document.get_toc(simple=False) or []
    except Exception:
        return []
    stack: dict[int, str] = {}
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_toc):
        if len(item) < 3:
            continue
        try:
            level = max(1, int(item[0]))
            page_number = int(item[2])
        except (TypeError, ValueError):
            continue
        title = " ".join(str(item[1]).split()).strip()
        if not title:
            continue
        for old_level in [key for key in stack if key >= level]:
            del stack[old_level]
        stack[level] = title
        path = [stack[key] for key in sorted(stack) if key <= level]
        result.append(
            {
                "index": index,
                "level": level,
                "title": title,
                "pdf_page_number": page_number if page_number > 0 else None,
                "path": path,
                "destination": json_safe(item[3]) if len(item) > 3 else None,
            }
        )
    return result


def outline_path_for_page(
    outline: list[dict[str, Any]], page_number: int
) -> list[str]:
    candidates = [
        item
        for item in outline
        if isinstance(item.get("pdf_page_number"), int)
        and 0 < int(item["pdf_page_number"]) <= page_number
    ]
    if not candidates:
        return []
    selected = max(
        candidates,
        key=lambda item: (
            int(item["pdf_page_number"]),
            int(item.get("index", 0)),
            int(item.get("level", 0)),
        ),
    )
    return [str(value) for value in selected.get("path", [])]


def extract_document_facts(document: fitz.Document) -> tuple[dict[str, Any], str]:
    metadata = _clean_metadata(getattr(document, "metadata", None))
    try:
        xmp = document.get_xml_metadata() or ""
    except Exception:
        xmp = ""
    try:
        page_label_rules = json_safe(document.get_page_labels() or [])
    except Exception:
        page_label_rules = []
    try:
        signature_flags = int(document.get_sigflags())
    except Exception:
        signature_flags = None

    outline = _normalize_outline(document)
    pages: list[dict[str, Any]] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        try:
            label = str(page.get_label() or "").strip() or None
        except Exception:
            label = None
        pages.append(
            {
                "pdf_page_index": page_index,
                "pdf_page_number": page_index + 1,
                "printed_page_label": label,
                "rotation_degrees": int(page.rotation or 0),
                "coordinate_space": "mupdf_unrotated",
                "render_coordinate_space": "mupdf_rotated_page",
                "rect": _rect(page.rect),
                "mediabox": _rect(page.mediabox),
                "cropbox": _rect(page.cropbox),
                "outline_path": outline_path_for_page(outline, page_index + 1),
            }
        )

    facts = {
        "schema_version": DOCUMENT_FACTS_SCHEMA_VERSION,
        "format": metadata.get("format"),
        "pdf_metadata": metadata,
        "xmp_present": bool(xmp.strip()),
        "xmp_sha256": sha256_text(xmp) if xmp else None,
        "encryption": {
            "needs_password": bool(document.needs_pass),
            "is_encrypted": bool(document.is_encrypted),
            "method": metadata.get("encryption"),
        },
        "permissions": _permission_facts(document),
        "is_repaired": bool(getattr(document, "is_repaired", False)),
        "version_count": int(getattr(document, "version_count", 0) or 0),
        "signature_flags": signature_flags,
        "page_count": int(document.page_count),
        "page_label_rules": page_label_rules,
        "outline": outline,
        "pages": pages,
    }
    return facts, xmp


def page_facts(document_facts: dict[str, Any], page_index: int) -> dict[str, Any]:
    pages = document_facts.get("pages") or []
    if 0 <= page_index < len(pages) and isinstance(pages[page_index], dict):
        return dict(pages[page_index])
    return {
        "pdf_page_index": page_index,
        "pdf_page_number": page_index + 1,
        "printed_page_label": None,
        "rotation_degrees": 0,
        "coordinate_space": "mupdf_unrotated",
        "render_coordinate_space": "mupdf_rotated_page",
        "outline_path": [],
    }


def native_text_quality(text: str, blocks: list[dict[str, Any]], page_area: float) -> dict[str, Any]:
    length = max(1, len(text))
    nonspace = [char for char in text if not char.isspace()]
    nonspace_count = max(1, len(nonspace))
    invalid = sum(char == "\ufffd" for char in text)
    alphanumeric = sum(char.isalnum() for char in nonspace)
    spaces = sum(char.isspace() and char not in "\r\n" for char in text)
    newlines = text.count("\n")
    span_area = 0.0
    span_count = 0
    invisible_ocr_spans = 0
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                if bbox and len(bbox) == 4:
                    span_area += max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                        0.0, float(bbox[3]) - float(bbox[1])
                    )
                span_count += 1
                if int(span.get("alpha", 255) or 0) == 0:
                    invisible_ocr_spans += 1
    invalid_ratio = invalid / length
    alphanumeric_ratio = alphanumeric / nonspace_count
    reasons: list[str] = []
    if not text.strip():
        reasons.append("native_text_empty")
    if invalid_ratio > 0.03:
        reasons.append("native_text_invalid_characters")
    if text.strip() and alphanumeric_ratio < 0.3:
        reasons.append("native_text_low_alphanumeric_ratio")
    if spaces / length > 0.7:
        reasons.append("native_text_excessive_spaces")
    if newlines / length > 0.6:
        reasons.append("native_text_excessive_newlines")
    return {
        "characters": len(text),
        "span_count": span_count,
        "invalid_character_ratio": round(invalid_ratio, 6),
        "alphanumeric_ratio": round(alphanumeric_ratio, 6),
        "space_ratio": round(spaces / length, 6),
        "newline_ratio": round(newlines / length, 6),
        "excessive_space_ratio": round(spaces / length, 6),
        "excessive_newline_ratio": round(newlines / length, 6),
        "text_density": round(span_area / max(1.0, page_area), 6),
        "invisible_ocr_spans": invisible_ocr_spans,
        "usable_as_reference": not reasons,
        "reason_codes": reasons,
    }


def build_document_profile(document: fitz.Document) -> dict[str, Any]:
    size_counts: Counter[float] = Counter()
    edge_counts: Counter[str] = Counter()
    page_candidates: list[list[tuple[str, float, float]]] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        try:
            page_dict = page.get_text("dict", sort=True)
        except Exception:
            page_candidates.append([])
            continue
        page_height = max(1.0, float(page.rect.height))
        candidates: list[tuple[str, float, float]] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_text = "".join(
                str(span.get("text", ""))
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
            normalized = " ".join(block_text.split()).strip()
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).strip()
                    if text:
                        size = round(float(span.get("size", 0.0) or 0.0), 1)
                        size_counts[size] += max(1, len(text))
            bbox = block.get("bbox") or (0, 0, 0, 0)
            if not normalized or len(normalized) > 150:
                continue
            y0 = float(bbox[1]) / page_height
            y1 = float(bbox[3]) / page_height
            if y1 <= 0.1 or y0 >= 0.87:
                key = re.sub(r"^\W*\d+\W*|\W*\d+\W*$", "", normalized.casefold())
                key = " ".join(key.split())
                if key:
                    edge_counts[key] += 1
                    candidates.append((key, y0, y1))
        page_candidates.append(candidates)
    minimum = max(3, math.ceil(max(1, document.page_count) * 0.2))
    repeated = sorted(key for key, count in edge_counts.items() if count >= minimum)
    body_size = size_counts.most_common(1)[0][0] if size_counts else 0.0
    heading_sizes = sorted(
        (size for size in size_counts if size > body_size * 1.08), reverse=True
    )[:6]
    return {
        "schema_version": 1,
        "body_font_size": body_size,
        "heading_font_sizes": heading_sizes,
        "font_size_character_counts": {
            f"{size:g}": count for size, count in sorted(size_counts.items())
        },
        "repeated_marginalia": repeated,
        "repeated_marginalia_minimum_pages": minimum,
    }
