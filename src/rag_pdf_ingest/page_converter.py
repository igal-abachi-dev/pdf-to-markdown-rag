from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class PageInput:
    """Provider-neutral page payload used by current and future converters."""

    image_bytes: bytes
    image_mime_type: str
    page_number: int
    native_text: str
    native_blocks: list[dict[str, Any]] = field(default_factory=list)
    page_facts: dict[str, Any] = field(default_factory=dict)
    document_facts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PageResult:
    markdown: str
    backend: str
    model: str
    page_ir: dict[str, Any] | None = None
    structured_data: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    response_id: str | None = None
    model_version: str | None = None
    usage_metadata: object = None
    finish_reason: str = "STOP"


class PageConverter(Protocol):
    def convert_page(self, page: PageInput) -> PageResult: ...

    def close(self) -> None: ...
