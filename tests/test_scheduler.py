from datetime import datetime

import pytest

from jclaw.core.scheduler import next_run_at, parse_schedule, parse_schedule_input


BASE_TIME = datetime.fromisoformat("2026-04-10T10:00:00-04:00")


@pytest.mark.parametrize(
    ("text", "kind", "expected_minutes"),
    [
        ("once:60", "once", 1),
        ("once:1800", "once", 30),
        ("interval:1800", "interval", 30),
        ("interval:7200", "interval", 120),
    ],
)
def test_parse_canonical_relative_and_interval_schedules(text: str, kind: str, expected_minutes: int) -> None:
    spec = parse_schedule(text)
    run_at = next_run_at(spec, from_dt=BASE_TIME)
    assert spec.kind == kind
    assert int((run_at.timestamp() - BASE_TIME.timestamp()) / 60) == expected_minutes


@pytest.mark.parametrize(
    ("text", "expected_hour", "expected_day"),
    [
        ("daily:09:00", 9, 11),
        ("daily:23:59", 23, 10),
    ],
)
def test_parse_canonical_daily_schedule(text: str, expected_hour: int, expected_day: int) -> None:
    spec = parse_schedule(text)
    run_at = next_run_at(spec, from_dt=BASE_TIME)
    assert spec.kind == "daily"
    assert run_at.hour == expected_hour
    assert run_at.day == expected_day


@pytest.mark.parametrize(
    ("text", "expected_month", "expected_day", "expected_hour", "expected_year"),
    [
        ("date:5-24 09:00", 5, 24, 9, 2026),
        ("date:5-24 15:00", 5, 24, 15, 2026),
        ("date:2026-5-24 15:00", 5, 24, 15, 2026),
    ],
)
def test_parse_canonical_date_schedule(text: str, expected_month: int, expected_day: int, expected_hour: int, expected_year: int) -> None:
    spec = parse_schedule(text)
    run_at = next_run_at(spec, from_dt=BASE_TIME)
    assert spec.kind == "date"
    assert run_at.month == expected_month
    assert run_at.day == expected_day
    assert run_at.hour == expected_hour
    assert run_at.year == expected_year


def test_canonical_date_schedule_rolls_forward_when_date_has_passed() -> None:
    spec = parse_schedule("date:4-1 09:00")
    run_at = next_run_at(spec, from_dt=BASE_TIME)
    assert run_at.year == 2027
    assert run_at.month == 4
    assert run_at.day == 1
    assert run_at.hour == 9


@pytest.mark.parametrize(
    "text",
    [
        "",
        "in 30 minutes",
        "every 30m",
        "daily 09:00",
        "May 24",
        "date:13-40 09:00",
        "once:0",
        "interval:-1",
    ],
)
def test_parse_schedule_rejects_unsupported_or_noncanonical_inputs(text: str) -> None:
    with pytest.raises(ValueError):
        parse_schedule(text)


def test_parse_schedule_input_accepts_structured_date_time() -> None:
    spec = parse_schedule_input(
        when={
            "kind": "date",
            "month": 5,
            "day": 24,
            "hour": 15,
            "minute": 0,
        }
    )
    run_at = next_run_at(spec, from_dt=BASE_TIME)

    assert spec.kind == "date"
    assert run_at.year == 2026
    assert run_at.month == 5
    assert run_at.day == 24
    assert run_at.hour == 15
    assert run_at.minute == 0


def test_parse_schedule_input_ignores_inferred_past_year_without_explicit_year() -> None:
    spec = parse_schedule_input(
        when={
            "kind": "date",
            "year": 2025,
            "month": 4,
            "day": 27,
            "hour": 9,
            "minute": 0,
        }
    )
    run_at = next_run_at(spec, from_dt=datetime.fromisoformat("2026-04-26T12:00:00-04:00"))

    assert spec.explicit_year is False
    assert spec.raw == "date:4-27 09:00"
    assert run_at.year == 2026
    assert run_at.month == 4
    assert run_at.day == 27
    assert run_at.hour == 9


def test_parse_schedule_input_respects_explicit_past_year() -> None:
    spec = parse_schedule_input(
        when={
            "kind": "date",
            "year": 2025,
            "explicit_year": True,
            "month": 4,
            "day": 27,
            "hour": 9,
            "minute": 0,
        }
    )

    with pytest.raises(ValueError, match="cannot compute next run for past schedule"):
        next_run_at(spec, from_dt=datetime.fromisoformat("2026-04-26T12:00:00-04:00"))


def test_parse_schedule_input_accepts_structured_daily() -> None:
    spec = parse_schedule_input(when={"kind": "daily", "hour": 14, "minute": 30})
    run_at = next_run_at(spec, from_dt=BASE_TIME)

    assert spec.kind == "daily"
    assert run_at.hour == 14
    assert run_at.minute == 30
    assert run_at.day == 10


def test_parse_schedule_input_rejects_invalid_structured_payload() -> None:
    with pytest.raises(ValueError, match="structured schedule field 'month' must be between 1 and 12"):
        parse_schedule_input(
            when={
                "kind": "date",
                "month": 13,
                "day": 24,
            }
        )


def test_parse_schedule_input_accepts_canonical_date_schedule() -> None:
    spec = parse_schedule_input(schedule="date:4-27 09:00")
    run_at = next_run_at(spec, from_dt=datetime.fromisoformat("2026-04-26T12:00:00-04:00"))

    assert spec.kind == "date"
    assert run_at.year == 2026
    assert run_at.month == 4
    assert run_at.day == 27
    assert run_at.hour == 9
