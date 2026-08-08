import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from rag_pdf_ingest.rate_limit import PersistentRequestLimiter, pacific_daily_window


class RateLimitTests(unittest.TestCase):
    def test_request_history_resets_at_midnight_pacific(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "requests.json"
            limiter = PersistentRequestLimiter(
                state_path,
                max_requests_per_day=2,
                min_interval_seconds=0,
            )
            pacific = ZoneInfo("America/Los_Angeles")
            first = datetime(2026, 8, 7, 23, 59, 50, tzinfo=pacific).timestamp()
            after_reset = datetime(2026, 8, 8, 0, 0, 1, tzinfo=pacific).timestamp()
            with patch("rag_pdf_ingest.rate_limit.time.time", return_value=first):
                limiter.before_request()
            with patch("rag_pdf_ingest.rate_limit.time.time", return_value=first + 1):
                limiter.before_request()
            value = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(value["request_timestamps"]), 2)

            with patch("rag_pdf_ingest.rate_limit.time.time", return_value=after_reset):
                limiter.before_request()
            value = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(value["request_timestamps"]), 1)
            self.assertEqual(value["window"], "calendar-day")
            self.assertEqual(value["time_zone"], "America/Los_Angeles")

    def test_daily_window_handles_pacific_daylight_time(self):
        pacific = ZoneInfo("America/Los_Angeles")
        now = datetime(2026, 8, 7, 12, 0, tzinfo=pacific).timestamp()
        start, end = pacific_daily_window(now)
        self.assertEqual(
            datetime.fromtimestamp(start, pacific),
            datetime(2026, 8, 7, 0, 0, tzinfo=pacific),
        )
        self.assertEqual(
            datetime.fromtimestamp(end, pacific),
            datetime(2026, 8, 8, 0, 0, tzinfo=pacific),
        )


if __name__ == "__main__":
    unittest.main()
