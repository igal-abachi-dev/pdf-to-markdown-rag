from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _authors(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        text = _optional_text(item)
        if text and text not in result:
            result.append(text)
    return result


def _valid_pdf_text(value: Any, *, title: bool = False) -> str | None:
    text = _optional_text(value)
    if not text:
        return None
    folded = text.casefold().strip(" ._-")
    if folded in {"untitled", "unknown", "document", "none", "null"}:
        return None
    if title and re.match(r"^(microsoft word|libreoffice writer)\s*[-:]", folded):
        return None
    return text


def resolve_bibliography(
    license_info: dict[str, Any] | None,
    source_file: str,
    *,
    existing: dict[str, Any] | None = None,
    pdf_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve display metadata: sidecar, PDF metadata, then filename stem."""
    license_info = license_info or {}
    nested = license_info.get("bibliography")
    nested = nested if isinstance(nested, dict) else {}
    existing = existing or {}
    pdf_metadata = pdf_metadata or {}

    explicit_title = _optional_text(nested.get("title") or license_info.get("title"))
    preserved_sidecar_title = (
        _optional_text(existing.get("title"))
        if existing.get("title_source") == "sidecar"
        else None
    )
    pdf_title = _valid_pdf_text(pdf_metadata.get("title"), title=True)
    preserved_pdf_title = (
        _optional_text(existing.get("title"))
        if existing.get("title_source") == "pdf_metadata"
        else None
    )
    title = (
        explicit_title
        or preserved_sidecar_title
        or pdf_title
        or preserved_pdf_title
        or Path(source_file).stem
    )
    title_source = (
        "sidecar"
        if explicit_title or preserved_sidecar_title
        else "pdf_metadata"
        if pdf_title or preserved_pdf_title
        else "source_file"
    )

    explicit_authors = _authors(
        nested.get("authors")
        or nested.get("author")
        or license_info.get("authors")
        or license_info.get("author")
    )
    preserved_sidecar_authors = (
        _authors(existing.get("authors"))
        if existing.get("authors_source") == "sidecar"
        or (existing.get("title_source") == "sidecar" and existing.get("authors"))
        else []
    )
    pdf_authors = _authors(_valid_pdf_text(pdf_metadata.get("author")))
    preserved_pdf_authors = (
        _authors(existing.get("authors"))
        if existing.get("authors_source") == "pdf_metadata"
        else []
    )
    authors = (
        explicit_authors
        or preserved_sidecar_authors
        or pdf_authors
        or preserved_pdf_authors
    )
    authors_source = (
        "sidecar"
        if explicit_authors or preserved_sidecar_authors
        else "pdf_metadata"
        if pdf_authors or preserved_pdf_authors
        else None
    )
    edition = _optional_text(
        nested.get("edition") or license_info.get("edition") or existing.get("edition")
    )
    publication_year = _optional_text(
        nested.get("publication_year")
        or nested.get("year")
        or license_info.get("publication_year")
        or license_info.get("year")
        or existing.get("publication_year")
    )
    language = _optional_text(
        nested.get("language")
        or license_info.get("language")
        or existing.get("language")
    )
    return {
        "title": title,
        "title_source": title_source,
        "authors": authors,
        "authors_source": authors_source,
        "edition": edition,
        "publication_year": publication_year,
        "language": language,
    }


def format_book_label(bibliography: dict[str, Any]) -> str:
    parts: list[str] = []
    authors = _authors(bibliography.get("authors"))
    if authors:
        parts.append(", ".join(authors))
    parts.append(str(bibliography["title"]))
    edition = _optional_text(bibliography.get("edition"))
    if edition:
        edition_lower = edition.casefold()
        parts.append(
            edition
            if "edition" in edition_lower or "ed." in edition_lower
            else f"{edition} ed."
        )
    year = _optional_text(bibliography.get("publication_year"))
    if year:
        parts.append(year)
    return ", ".join(parts)


def format_pdf_citation(
    bibliography: dict[str, Any],
    page_start: int,
    page_end: int | None = None,
    *,
    printed_page_start: str | None = None,
    printed_page_end: str | None = None,
) -> str:
    page_end = page_start if page_end is None else page_end
    page_text = (
        f"PDF p. {page_start}"
        if page_start == page_end
        else f"PDF pp. {page_start}-{page_end}"
    )
    printed_start = _optional_text(printed_page_start)
    printed_end = _optional_text(printed_page_end)
    if printed_start and not (
        page_start == page_end and printed_start == str(page_start)
    ):
        if page_start == page_end or not printed_end or printed_end == printed_start:
            page_text += f" (printed p. {printed_start})"
        else:
            page_text += f" (printed pp. {printed_start}-{printed_end})"
    return f"{format_book_label(bibliography)}, {page_text}"
