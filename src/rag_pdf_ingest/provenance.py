from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import sha256_file


DEFAULT_LICENSE_STATUS = "authorized"


def normalize_license_info(value: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize legacy/missing records to the user's authorized-only policy."""
    license_info = {"status": DEFAULT_LICENSE_STATUS, **(value or {})}
    status = license_info.get("status")
    if not isinstance(status, str) or not status.strip():
        raise RuntimeError("License status must be a non-empty string")
    status = status.strip()
    if status == "unspecified":
        status = DEFAULT_LICENSE_STATUS
    license_info["status"] = status
    return license_info


def load_license_file(
    sidecar: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate a license record without requiring the source PDF."""
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid license sidecar {sidecar.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"License sidecar must contain a JSON object: {sidecar.name}")
    try:
        license_info = normalize_license_info(value)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}: {sidecar.name}") from exc
    provenance = {
        "sidecar_file": sidecar.name,
        "sidecar_sha256": sha256_file(sidecar),
    }
    return license_info, provenance


def load_license_metadata(
    pdf_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    candidates = [
        pdf_path.with_suffix(".license.json"),
        pdf_path.with_name(f"{pdf_path.name}.license.json"),
    ]
    sidecar = next((candidate for candidate in candidates if candidate.exists()), None)
    if sidecar is None:
        return {"status": DEFAULT_LICENSE_STATUS}, {
            "sidecar_file": None,
            "sidecar_sha256": None,
        }, None

    license_info, provenance = load_license_file(sidecar)
    return license_info, provenance, sidecar
