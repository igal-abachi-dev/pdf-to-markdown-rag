from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, time as datetime_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .utils import atomic_write_json


LOGGER = logging.getLogger(__name__)
DAY_SECONDS = 24 * 60 * 60
PACIFIC_TIME_ZONE = ZoneInfo("America/Los_Angeles")


def pacific_daily_window(now: float) -> tuple[float, float]:
    """Return the current Gemini RPD window, which resets at midnight Pacific."""
    local_now = datetime.fromtimestamp(now, PACIFIC_TIME_ZONE)
    start = datetime.combine(local_now.date(), datetime_time.min, PACIFIC_TIME_ZONE)
    end = datetime.combine(
        local_now.date() + timedelta(days=1), datetime_time.min, PACIFIC_TIME_ZONE
    )
    return start.timestamp(), end.timestamp()


class PersistentRequestLimiter:
    """Cross-process Gemini request limiter that survives watcher restarts."""

    def __init__(
        self,
        state_path: Path,
        *,
        max_requests_per_day: int,
        min_interval_seconds: float,
    ):
        self.state_path = state_path
        self.lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")
        self.max_requests_per_day = max_requests_per_day
        self.min_interval_seconds = min_interval_seconds

    def _acquire_state_lock(self) -> int:
        deadline = time.monotonic() + 30
        while True:
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(descriptor, str(os.getpid()).encode("ascii"))
                return descriptor
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                    if age > 300:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out acquiring the Gemini rate-limit state lock")
                time.sleep(0.1)

    def _load_timestamps(self, now: float) -> list[float]:
        if not self.state_path.exists():
            return []
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            timestamps = [float(item) for item in value.get("request_timestamps", [])]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.warning("Resetting invalid rate-limit state: %s", self.state_path)
            return []
        valid = sorted(item for item in timestamps if item <= now + 60)
        day_start, _ = pacific_daily_window(now)
        current_day = [item for item in valid if item >= day_start]
        if self.min_interval_seconds > 0:
            previous = [item for item in valid if item < day_start]
            if previous and previous[-1] + self.min_interval_seconds > now:
                current_day.insert(0, previous[-1])
        return current_day

    def seconds_until_daily_reset(self, now: float | None = None) -> float:
        current = time.time() if now is None else now
        _, next_reset = pacific_daily_window(current)
        return max(0.0, next_reset - current)

    def next_daily_reset_iso(self, now: float | None = None) -> str:
        current = time.time() if now is None else now
        _, next_reset = pacific_daily_window(current)
        return datetime.fromtimestamp(next_reset, PACIFIC_TIME_ZONE).isoformat()

    def before_request(self) -> None:
        if self.max_requests_per_day <= 0 and self.min_interval_seconds <= 0:
            return

        while True:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = self._acquire_state_lock()
            try:
                now = time.time()
                timestamps = self._load_timestamps(now)
                day_start, next_daily_reset = pacific_daily_window(now)
                daily_timestamps = [item for item in timestamps if item >= day_start]
                wait_seconds = 0.0
                wait_reason = "minimum request interval"
                if timestamps and self.min_interval_seconds > 0:
                    wait_seconds = max(
                        wait_seconds,
                        timestamps[-1] + self.min_interval_seconds - now,
                    )
                if (
                    self.max_requests_per_day > 0
                    and len(daily_timestamps) >= self.max_requests_per_day
                ):
                    wait_seconds = max(wait_seconds, next_daily_reset - now)
                    wait_reason = (
                        f"daily request allowance reached "
                        f"({len(daily_timestamps)}/{self.max_requests_per_day}); "
                        f"resets {datetime.fromtimestamp(next_daily_reset, PACIFIC_TIME_ZONE).isoformat()}"
                    )

                if wait_seconds <= 0:
                    timestamps.append(now)
                    atomic_write_json(
                        self.state_path,
                        {
                            "window": "calendar-day",
                            "time_zone": "America/Los_Angeles",
                            "next_reset": datetime.fromtimestamp(
                                next_daily_reset, PACIFIC_TIME_ZONE
                            ).isoformat(),
                            "max_requests_per_day": self.max_requests_per_day,
                            "min_interval_seconds": self.min_interval_seconds,
                            "request_timestamps": timestamps,
                        },
                    )
                    return
            finally:
                os.close(descriptor)
                self.lock_path.unlink(missing_ok=True)

            LOGGER.info(
                "Free-tier request governor waiting %.0f seconds before the next Gemini call: %s",
                wait_seconds,
                wait_reason,
            )
            time.sleep(max(0.1, wait_seconds))
