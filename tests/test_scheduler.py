from datetime import datetime

import pytest

from jclaw.core.scheduler import next_run_at, parse_schedule


BASE_TIME = datetime.fromisoformat("2026-04-10T10:00:00-04:00")


@pytest.mark.parametrize(
    ("text", "kind", "expected_minutes"),
    [
        ("in 1 minute", "once", 1),
        ("in 30 minutes", "once", 30),
        ("in 2 hours", "once", 120),
        ("in 1 day", "once", 1440),
        ("every 30m", "interval", 30),
        ("every 2h", "interval", 120),
        ("every 3 days", "interval", 4320),
        ("hourly", "interval", 60),
    ],
)
def test_parse_relative_and_interval_schedules(text: str, kind: str, expected_minutes: int) -> None:
    spec = parse_schedule(text)
    run_at = next_run_at(spec, from_dt=BASE_TIME)
    assert spec.kind == kind
    assert int((run_at.timestamp() - BASE_TIME.timestamp()) / 60) == expected_minutes


@pytest.mark.parametrize(
    ("text", "expected_hour", "expected_day"),
    [
        ("daily 09:00", 9, 11),
        ("DAILY 09:00", 9, 11),
        ("  daily 23:59  ", 23, 10),
    ],
)
def test_parse_daily_schedule(text: str, expected_hour: int, expected_day: int) -> None:
    spec = parse_schedule(text)
    run_at = next_run_at(spec, from_dt=BASE_TIME)
    assert spec.kind == "daily"
    assert run_at.hour == expected_hour
    assert run_at.day == expected_day


@pytest.mark.parametrize(
    "text",
    [
        "in thirty minutes",
        "in -1 minutes",
        "in 0 minutes",
        "every 0m",
        "daily 24:00",
        "daily 09:60",
        "sometime soon",
        "tomorrow morning",
    ],
)
def test_parse_schedule_rejects_unsupported_inputs(text: str) -> None:
    with pytest.raises(ValueError, match="unsupported schedule"):
        parse_schedule(text)
