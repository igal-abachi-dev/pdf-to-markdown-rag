from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        # This project is configured by its local .env. It is deliberately
        # authoritative so a stale value inherited from a shell cannot win.
        os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_float(name: str, default: float, minimum: float = 0) -> float:
    value = float(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    root: Path
    inbox_dir: Path
    raw_dir: Path
    md_dir: Path
    metadata_dir: Path
    pages_dir: Path
    chunks_dir: Path
    processed_dir: Path
    failed_dir: Path
    logs_dir: Path
    debug_dir: Path
    state_dir: Path
    prompt_path: Path
    gemini_api_key: str
    gemini_model: str
    thinking_level: str
    media_resolution: str
    max_output_tokens: int
    max_retries: int
    requests_per_day: int
    min_request_interval_seconds: float
    pause_on_rate_limit: bool
    rate_limit_retry_seconds: int
    max_consecutive_page_failures: int
    render_dpi: int
    poll_seconds: int
    file_stable_seconds: int
    move_completed_pdfs: bool
    keep_page_images: bool
    native_debug_overlays: bool
    native_page_router_enabled: bool
    page_image_format: str
    page_jpeg_quality: int
    chunk_target_tokens: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    embedding_model: str
    embedding_dimensions: int
    embedding_batch_size: int
    embedding_cache_dir: Path
    corpus_dir: Path

    @classmethod
    def load(cls, root: Path | None = None) -> "Settings":
        if root is None:
            env_root = os.getenv("RAG_ROOT")
            root = Path(env_root) if env_root else Path(__file__).resolve().parents[2]
        root = root.resolve()
        _load_env_file(root / ".env")

        settings = cls(
            root=root,
            inbox_dir=root / "inbox",
            raw_dir=root / "raw",
            md_dir=root / "md",
            metadata_dir=root / "metadata",
            pages_dir=root / "pages",
            chunks_dir=root / "chunks",
            processed_dir=root / "processed",
            failed_dir=root / "failed",
            logs_dir=root / "logs",
            debug_dir=root / "debug",
            state_dir=root / ".state",
            prompt_path=root / "prompts" / "page_to_markdown.txt",
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
            thinking_level=os.getenv("GEMINI_THINKING_LEVEL", "HIGH").strip().upper(),
            media_resolution=os.getenv("GEMINI_MEDIA_RESOLUTION", "HIGH").strip().upper(),
            max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 65536, 1024),
            max_retries=_env_int("GEMINI_MAX_RETRIES", 5, 1),
            requests_per_day=_env_int("GEMINI_REQUESTS_PER_DAY", 18, 0),
            min_request_interval_seconds=_env_float(
                "GEMINI_MIN_REQUEST_INTERVAL_SECONDS", 60.0, 0
            ),
            pause_on_rate_limit=_env_bool("GEMINI_PAUSE_ON_RATE_LIMIT", True),
            rate_limit_retry_seconds=_env_int(
                "GEMINI_RATE_LIMIT_RETRY_SECONDS", 300, 1
            ),
            max_consecutive_page_failures=_env_int(
                "MAX_CONSECUTIVE_PAGE_FAILURES", 5, 1
            ),
            render_dpi=_env_int("RENDER_DPI", 240, 72),
            poll_seconds=_env_int("POLL_SECONDS", 5, 1),
            file_stable_seconds=_env_int("FILE_STABLE_SECONDS", 8, 1),
            move_completed_pdfs=_env_bool("MOVE_COMPLETED_PDFS", True),
            keep_page_images=_env_bool("KEEP_PAGE_IMAGES", True),
            native_debug_overlays=_env_bool("NATIVE_DEBUG_OVERLAYS", False),
            native_page_router_enabled=_env_bool(
                "NATIVE_PAGE_ROUTER_ENABLED", False
            ),
            page_image_format=os.getenv("PAGE_IMAGE_FORMAT", "JPEG").strip().upper(),
            page_jpeg_quality=_env_int("PAGE_JPEG_QUALITY", 90, 1),
            chunk_target_tokens=_env_int("CHUNK_TARGET_TOKENS", 750, 100),
            chunk_max_tokens=_env_int("CHUNK_MAX_TOKENS", 900, 100),
            chunk_overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 120, 0),
            embedding_model=os.getenv(
                "GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"
            ).strip(),
            embedding_dimensions=_env_int("GEMINI_EMBEDDING_DIMENSIONS", 768, 1),
            embedding_batch_size=_env_int("GEMINI_EMBEDDING_BATCH_SIZE", 32, 1),
            embedding_cache_dir=root / ".state" / "embedding-cache",
            corpus_dir=root / "artifacts",
        )
        if settings.chunk_target_tokens > settings.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS cannot exceed CHUNK_MAX_TOKENS")
        if settings.chunk_overlap_tokens >= settings.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS")
        if settings.page_image_format == "JPG":
            object.__setattr__(settings, "page_image_format", "JPEG")
        if settings.page_image_format not in {"JPEG", "PNG"}:
            raise ValueError("PAGE_IMAGE_FORMAT must be JPEG, JPG, or PNG")
        if settings.page_jpeg_quality > 100:
            raise ValueError("PAGE_JPEG_QUALITY cannot exceed 100")
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        for path in (
            self.inbox_dir,
            self.raw_dir,
            self.md_dir,
            self.metadata_dir,
            self.pages_dir,
            self.chunks_dir,
            self.processed_dir,
            self.failed_dir,
            self.logs_dir,
            self.debug_dir,
            self.state_dir / "locks",
            self.embedding_cache_dir,
            self.corpus_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_for_ingestion(self) -> None:
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and add the key."
            )
        if not self.prompt_path.exists():
            raise RuntimeError(f"Prompt file does not exist: {self.prompt_path}")
        if self.thinking_level not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
            raise RuntimeError("GEMINI_THINKING_LEVEL must be MINIMAL, LOW, MEDIUM, or HIGH")
        if self.media_resolution not in {"LOW", "MEDIUM", "HIGH"}:
            raise RuntimeError("GEMINI_MEDIA_RESOLUTION must be LOW, MEDIUM, or HIGH")
