from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from .config import Settings
from .page_converter import PageInput, PageResult
from .rate_limit import PersistentRequestLimiter
from .utils import json_safe, strip_markdown_fence


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiResult:
    markdown: str
    response_id: str | None
    model_version: str | None
    usage_metadata: object
    finish_reason: str
    backend: str = "gemini"
    model: str = ""
    structured_data: dict[str, object] | None = None
    diagnostics: dict[str, object] | None = None
    warnings: tuple[str, ...] = ()


class GeminiResponseError(RuntimeError):
    """A completed Gemini response that is unsafe to persist as a successful page."""

    def __init__(
        self,
        message: str,
        finish_reason: str = "UNKNOWN",
        response_snapshot: object = None,
    ):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.response_snapshot = response_snapshot


def _finish_reason_name(value: object) -> str:
    if value is None:
        return "UNKNOWN"
    enum_value = getattr(value, "value", value)
    return str(enum_value).rsplit(".", 1)[-1]


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, GeminiResponseError):
        return False
    if isinstance(exc, errors.APIError):
        code = int(getattr(exc, "code", 0) or 0)
        return code in {408, 409, 429} or 500 <= code < 600
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    transport_name = type(exc).__name__.lower()
    return "timeout" in transport_name or transport_name in {
        "connecterror",
        "connectionerror",
        "networkerror",
        "readerror",
        "remoteprotocolerror",
    }


def _api_error_code(exc: Exception) -> int:
    if not isinstance(exc, errors.APIError):
        return 0
    try:
        return int(getattr(exc, "code", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _api_error_snapshot(exc: Exception) -> dict[str, object]:
    return {
        "code": _api_error_code(exc),
        "status": getattr(exc, "status", None),
        "message": getattr(exc, "message", None) or str(exc),
        "details": json_safe(getattr(exc, "details", None)),
    }


def _is_daily_quota_error(snapshot: dict[str, object]) -> bool:
    serialized = json.dumps(snapshot, ensure_ascii=False, default=str).lower()
    daily_markers = (
        "perday",
        "per_day",
        "per day",
        "requestsperday",
        "requests_per_day",
        "daily",
        "rpd",
    )
    return any(marker in serialized for marker in daily_markers)


def _retry_delay_seconds(snapshot: dict[str, object]) -> float | None:
    def walk(value: object) -> float | None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower().replace("_", "") == "retrydelay":
                    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*", str(item))
                    if match:
                        return float(match.group(1))
                nested = walk(item)
                if nested is not None:
                    return nested
        elif isinstance(value, list):
            for item in value:
                nested = walk(item)
                if nested is not None:
                    return nested
        return None

    return walk(snapshot)


class GeminiPageConverter:
    def __init__(self, settings: Settings):
        settings.validate_for_ingestion()
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.base_prompt = settings.prompt_path.read_text(encoding="utf-8-sig").strip()
        self.request_limiter = PersistentRequestLimiter(
            settings.state_dir / "gemini_request_limit.json",
            max_requests_per_day=settings.requests_per_day,
            min_interval_seconds=settings.min_request_interval_seconds,
        )

    def _config(self) -> types.GenerateContentConfig:
        thinking_level = getattr(types.ThinkingLevel, self.settings.thinking_level)
        resolution_name = f"MEDIA_RESOLUTION_{self.settings.media_resolution}"
        media_resolution = getattr(types.MediaResolution, resolution_name)
        return types.GenerateContentConfig(
            max_output_tokens=self.settings.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            media_resolution=media_resolution,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def convert(
        self,
        image_bytes: bytes,
        native_text: str,
        page_number: int,
        image_mime_type: str = "image/png",
    ) -> GeminiResult:
        reference = re.sub(
            r"</?NATIVE_TEXT_LAYER_REFERENCE>", "", native_text, flags=re.IGNORECASE
        ).strip()
        prompt = self.base_prompt
        if reference:
            prompt += (
                f"\n\nPDF page number: {page_number}\n"
                "<NATIVE_TEXT_LAYER_REFERENCE>\n"
                f"{reference}\n"
                "</NATIVE_TEXT_LAYER_REFERENCE>"
            )
        else:
            prompt += f"\n\nPDF page number: {page_number}. No usable native text layer is available."

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type),
            types.Part.from_text(text=prompt),
        ]

        last_error: Exception | None = None
        transient_failures = 0
        consecutive_rate_limits = 0
        while True:
            try:
                self.request_limiter.before_request()
                response = self.client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=contents,
                    config=self._config(),
                )
                candidates = response.candidates or []
                if not candidates:
                    raise GeminiResponseError(
                        "Gemini returned no response candidates",
                        response_snapshot=json_safe(response),
                    )
                candidate = candidates[0]
                finish_reason = _finish_reason_name(candidate.finish_reason)
                if finish_reason != "STOP":
                    finish_message = getattr(candidate, "finish_message", None)
                    detail = f": {finish_message}" if finish_message else ""
                    raise GeminiResponseError(
                        f"Gemini stopped with finish_reason={finish_reason}{detail}",
                        finish_reason=finish_reason,
                        response_snapshot=json_safe(response),
                    )
                markdown = strip_markdown_fence(response.text or "")
                if not markdown:
                    raise GeminiResponseError(
                        "Gemini returned an empty Markdown response",
                        finish_reason=finish_reason,
                        response_snapshot=json_safe(response),
                    )
                return GeminiResult(
                    markdown=markdown,
                    response_id=getattr(response, "response_id", None),
                    model_version=getattr(response, "model_version", None),
                    usage_metadata=json_safe(getattr(response, "usage_metadata", None)),
                    finish_reason=finish_reason,
                    model=self.settings.gemini_model,
                    diagnostics={"native_reference_supplied": bool(reference)},
                )
            except Exception as exc:  # SDK error classes vary by transport/version.
                last_error = exc
                if _api_error_code(exc) == 429 and self.settings.pause_on_rate_limit:
                    consecutive_rate_limits += 1
                    snapshot = _api_error_snapshot(exc)
                    snapshot_text = json.dumps(snapshot, ensure_ascii=False, default=str)
                    LOGGER.warning(
                        "Gemini 429 on page %s (attempt %s): %s",
                        page_number,
                        consecutive_rate_limits,
                        snapshot_text[:8000],
                    )
                    retry_delay = _retry_delay_seconds(snapshot)
                    if _is_daily_quota_error(snapshot) or consecutive_rate_limits >= 3:
                        delay = self.request_limiter.seconds_until_daily_reset()
                        LOGGER.warning(
                            "Daily quota is exhausted or 429 persisted; pausing %.0fs until "
                            "the Gemini RPD reset at %s without failing page %s",
                            delay,
                            self.request_limiter.next_daily_reset_iso(),
                            page_number,
                        )
                    else:
                        delay = max(
                            float(self.settings.rate_limit_retry_seconds),
                            retry_delay or 0.0,
                        )
                        LOGGER.warning(
                            "Temporary Gemini rate limit; pausing %.0fs without failing page %s",
                            delay,
                            page_number,
                        )
                    time.sleep(max(1.0, delay))
                    continue
                transient_failures += 1
                if (
                    transient_failures >= self.settings.max_retries
                    or not _is_retryable(exc)
                ):
                    break
                delay = min(60.0, (2 ** (transient_failures - 1)) + random.random())
                LOGGER.warning(
                    "Gemini page %s attempt %s/%s failed: %s. Retrying in %.1fs",
                    page_number,
                    transient_failures,
                    self.settings.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        if last_error is not None and not _is_retryable(last_error):
            raise last_error
        attempts = self.settings.max_retries
        raise RuntimeError(f"Gemini failed for page {page_number} after {attempts} attempt(s)") from last_error

    def convert_page(self, page: PageInput) -> PageResult:
        result = self.convert(
            page.image_bytes,
            page.native_text,
            page.page_number,
            image_mime_type=page.image_mime_type,
        )
        return PageResult(
            markdown=result.markdown,
            backend=result.backend,
            model=result.model,
            page_ir=None,
            structured_data=result.structured_data,
            diagnostics=result.diagnostics or {},
            warnings=result.warnings,
            response_id=result.response_id,
            model_version=result.model_version,
            usage_metadata=result.usage_metadata,
            finish_reason=result.finish_reason,
        )

    def close(self) -> None:
        self.client.close()
