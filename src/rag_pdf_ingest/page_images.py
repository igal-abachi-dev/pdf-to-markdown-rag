from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .config import Settings
from .pipeline import _acquire_lock
from .utils import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    relative_posix,
    utc_now,
)


LOGGER = logging.getLogger(__name__)


def _sync_combined_page_metadata(settings: Settings, book_id: str) -> None:
    combined_path = settings.metadata_dir / book_id / "pages.jsonl"
    if not combined_path.exists():
        return
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((settings.metadata_dir / book_id).glob("page-*.json"))
    ]
    atomic_write_text(
        combined_path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def convert_page_images_to_jpeg(
    settings: Settings,
    *,
    book_ids: set[str] | None = None,
    quality: int | None = None,
    keep_png: bool = False,
) -> dict[str, Any]:
    """Convert stored page PNGs to JPEG and update page metadata atomically."""
    jpeg_quality = quality if quality is not None else settings.page_jpeg_quality
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")

    converted = 0
    skipped_locked: list[str] = []
    skipped_missing_metadata: list[str] = []
    errors: list[str] = []
    bytes_before = 0
    bytes_after = 0
    found_ids: set[str] = set()

    for book_dir in sorted(path for path in settings.pages_dir.iterdir() if path.is_dir()):
        book_id = book_dir.name
        if book_ids is not None and book_id not in book_ids:
            continue
        found_ids.add(book_id)
        png_paths = sorted(book_dir.glob("page-*.png"))
        if not png_paths:
            _sync_combined_page_metadata(settings, book_id)
            continue

        lock_path = settings.state_dir / "locks" / f"{book_id}.lock"
        try:
            _acquire_lock(lock_path, f"stored page images for {book_id}")
        except RuntimeError as exc:
            skipped_locked.append(book_id)
            LOGGER.warning("Skipping locked book %s: %s", book_id, exc)
            continue

        try:
            for png_path in png_paths:
                metadata_path = settings.metadata_dir / book_id / f"{png_path.stem}.json"
                if not metadata_path.exists():
                    skipped_missing_metadata.append(
                        f"{book_id}/{png_path.name}"
                    )
                    continue
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    original_bytes = png_path.stat().st_size
                    pixmap = fitz.Pixmap(str(png_path))
                    jpeg_bytes = pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
                    verification = fitz.Pixmap(jpeg_bytes)
                    if verification.width != pixmap.width or verification.height != pixmap.height:
                        raise RuntimeError("JPEG dimensions differ from the source PNG")

                    jpeg_path = png_path.with_suffix(".jpg")
                    atomic_write_bytes(jpeg_path, jpeg_bytes)
                    metadata.update(
                        {
                            "page_image_path": relative_posix(jpeg_path, settings.root),
                            "page_image_mime_type": "image/jpeg",
                            "page_image_bytes": len(jpeg_bytes),
                            "page_image_jpeg_quality": jpeg_quality,
                            "page_image_original_format": "image/png",
                            "page_image_original_bytes": original_bytes,
                            "page_image_storage_converted_at": utc_now(),
                        }
                    )
                    atomic_write_json(metadata_path, metadata)
                    if not keep_png:
                        png_path.unlink()
                    converted += 1
                    bytes_before += original_bytes
                    bytes_after += len(jpeg_bytes) + (original_bytes if keep_png else 0)
                except Exception as exc:
                    errors.append(
                        f"{book_id}/{png_path.name}: {type(exc).__name__}: {exc}"
                    )
                    LOGGER.exception("Could not convert %s", png_path)
            _sync_combined_page_metadata(settings, book_id)
        finally:
            lock_path.unlink(missing_ok=True)

    if book_ids is not None:
        errors.extend(f"Unknown book_id: {book_id}" for book_id in sorted(book_ids - found_ids))
    return {
        "converted": converted,
        "skipped_locked": skipped_locked,
        "skipped_missing_metadata": skipped_missing_metadata,
        "errors": errors,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "bytes_saved": bytes_before - bytes_after,
        "jpeg_quality": jpeg_quality,
        "kept_png": keep_png,
    }
