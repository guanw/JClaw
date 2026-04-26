from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any


def _local_now() -> datetime:
    return datetime.now().astimezone()


@dataclass(slots=True)
class ScheduleSpec:
    raw: str
    kind: str
    interval_seconds: int | None = None
    hour: int | None = None
    minute: int | None = None
    month: int | None = None
    day: int | None = None
    year: int | None = None
    explicit_year: bool = False


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
DEFAULT_DATE_HOUR = 9
DEFAULT_DATE_MINUTE = 0


def parse_schedule(text: str) -> ScheduleSpec:
    raw = text.strip()
    lowered = raw.lower()

    once_match = re.fullmatch(
        r"in\s+(\d+)\s*(m|min|mins|minute|minutes|h|hour|hours|d|day|days)",
        lowered,
    )
    if once_match:
        amount = int(once_match.group(1))
        if amount <= 0:
            raise ValueError("unsupported schedule; use 'in 30 minutes', 'every 30m', 'hourly', or 'daily 09:00'")
        unit = once_match.group(2)
        multiplier = 60
        if unit in {"h", "hour", "hours"}:
            multiplier = 3600
        elif unit in {"d", "day", "days"}:
            multiplier = 86400
        return ScheduleSpec(
            raw=raw,
            kind="once",
            interval_seconds=amount * multiplier,
        )

    every_match = re.fullmatch(r"every\s+(\d+)\s*([mhd]|min|mins|minute|minutes|hour|hours|day|days)", lowered)
    if every_match:
        amount = int(every_match.group(1))
        if amount <= 0:
            raise ValueError("unsupported schedule; use 'in 30 minutes', 'every 30m', 'hourly', or 'daily 09:00'")
        unit = every_match.group(2)
        multiplier = 60
        if unit in {"h", "hour", "hours"}:
            multiplier = 3600
        elif unit in {"d", "day", "days"}:
            multiplier = 86400
        return ScheduleSpec(raw=raw, kind="interval", interval_seconds=amount * multiplier)

    if lowered == "hourly":
        return ScheduleSpec(raw=raw, kind="interval", interval_seconds=3600)

    daily_match = re.fullmatch(r"daily\s+([01]?\d|2[0-3]):([0-5]\d)", lowered)
    if daily_match:
        return ScheduleSpec(
            raw=raw,
            kind="daily",
            hour=int(daily_match.group(1)),
            minute=int(daily_match.group(2)),
        )

    date_spec = _parse_date_schedule(raw)
    if date_spec is not None:
        return date_spec

    raise ValueError(
        "unsupported schedule; use 'in 30 minutes', 'every 30m', 'hourly', 'daily 09:00', 'May 24', or '5/24'"
    )


def parse_schedule_input(*, schedule: str | None = None, when: dict[str, Any] | None = None) -> ScheduleSpec:
    if when not in (None, {}):
        return _parse_structured_when(when)
    if schedule is None or not str(schedule).strip():
        raise ValueError("creating a schedule requires either schedule or when")
    return parse_schedule(str(schedule))


def next_run_at(spec: ScheduleSpec, *, from_dt: datetime | None = None) -> datetime:
    current = from_dt or _local_now()
    if spec.kind == "once" and spec.interval_seconds is not None:
        return current + timedelta(seconds=spec.interval_seconds)

    if spec.kind == "interval" and spec.interval_seconds is not None:
        return current + timedelta(seconds=spec.interval_seconds)

    if spec.kind == "daily" and spec.hour is not None and spec.minute is not None:
        candidate = current.replace(hour=spec.hour, minute=spec.minute, second=0, microsecond=0)
        if candidate <= current:
            candidate += timedelta(days=1)
        return candidate

    if spec.kind == "date" and spec.month is not None and spec.day is not None:
        year = spec.year if spec.explicit_year and spec.year is not None else current.year
        candidate = current.replace(
            year=year,
            month=spec.month,
            day=spec.day,
            hour=spec.hour if spec.hour is not None else DEFAULT_DATE_HOUR,
            minute=spec.minute if spec.minute is not None else DEFAULT_DATE_MINUTE,
            second=0,
            microsecond=0,
        )
        if candidate <= current:
            if spec.explicit_year:
                raise ValueError(f"cannot compute next run for past schedule: {spec.raw}")
            candidate = candidate.replace(year=candidate.year + 1)
        return candidate

    raise ValueError(f"cannot compute next run for schedule: {spec.raw}")


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_date_schedule(raw: str) -> ScheduleSpec | None:
    text = raw.strip()
    lowered = text.lower()
    if lowered.startswith("on "):
        lowered = lowered[3:].strip()

    md_match = re.fullmatch(
        r"([01]?\d)/([0-3]?\d)(?:/(\d{2,4}))?(?:\s+(?:at\s+)?)?(.+)?",
        lowered,
    )
    if md_match:
        month = int(md_match.group(1))
        day = int(md_match.group(2))
        year = _normalize_year(md_match.group(3))
        hour, minute = _parse_optional_time(md_match.group(4))
        return ScheduleSpec(
            raw=raw,
            kind="date",
            month=month,
            day=day,
            year=year,
            explicit_year=year is not None,
            hour=hour if hour is not None else DEFAULT_DATE_HOUR,
            minute=minute if minute is not None else DEFAULT_DATE_MINUTE,
        )

    named_match = re.fullmatch(
        r"([a-z]+)\s+([0-3]?\d)(?:,?\s+(\d{2,4}))?(?:\s+(?:at\s+)?)?(.+)?",
        lowered,
    )
    if not named_match:
        return None
    month_name = named_match.group(1)
    month = MONTHS.get(month_name)
    if month is None:
        return None
    day = int(named_match.group(2))
    year = _normalize_year(named_match.group(3))
    hour, minute = _parse_optional_time(named_match.group(4))
    return ScheduleSpec(
        raw=raw,
        kind="date",
        month=month,
        day=day,
        year=year,
        explicit_year=year is not None,
        hour=hour if hour is not None else DEFAULT_DATE_HOUR,
        minute=minute if minute is not None else DEFAULT_DATE_MINUTE,
    )


def _normalize_year(raw_year: str | None) -> int | None:
    if raw_year is None or not str(raw_year).strip():
        return None
    year = int(raw_year)
    if year < 100:
        return 2000 + year
    return year


def _parse_optional_time(raw_time: str | None) -> tuple[int | None, int | None]:
    if raw_time is None or not str(raw_time).strip():
        return None, None
    text = str(raw_time).strip().lower()
    time_match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text)
    if time_match:
        return int(time_match.group(1)), int(time_match.group(2))
    ampm_match = re.fullmatch(r"(\d{1,2})(?::([0-5]\d))?\s*(am|pm)", text)
    if ampm_match:
        hour = int(ampm_match.group(1))
        minute = int(ampm_match.group(2) or "00")
        meridiem = ampm_match.group(3)
        if hour == 12:
            hour = 0
        if meridiem == "pm":
            hour += 12
        return hour, minute
    raise ValueError(
        "unsupported schedule; use 'in 30 minutes', 'every 30m', 'hourly', 'daily 09:00', 'May 24', or '5/24'"
    )


def _parse_structured_when(payload: dict[str, Any]) -> ScheduleSpec:
    if not isinstance(payload, dict):
        raise ValueError("structured schedule must be an object")
    kind = str(payload.get("kind", "")).strip().lower()
    if not kind:
        raise ValueError("structured schedule requires kind")

    if kind in {"once", "interval"}:
        seconds = _coerce_positive_int(payload.get("interval_seconds"), field="interval_seconds")
        return ScheduleSpec(
            raw=_structured_raw(kind, payload),
            kind=kind,
            interval_seconds=seconds,
        )

    if kind == "daily":
        hour = _coerce_int_in_range(payload.get("hour"), field="hour", minimum=0, maximum=23)
        minute = _coerce_int_in_range(payload.get("minute", 0), field="minute", minimum=0, maximum=59)
        return ScheduleSpec(
            raw=_structured_raw(kind, payload),
            kind="daily",
            hour=hour,
            minute=minute,
        )

    if kind == "date":
        month = _coerce_int_in_range(payload.get("month"), field="month", minimum=1, maximum=12)
        day = _coerce_int_in_range(payload.get("day"), field="day", minimum=1, maximum=31)
        year = payload.get("year")
        normalized_year = None if year in (None, "") else _normalize_year(str(year))
        hour = _coerce_int_in_range(payload.get("hour", DEFAULT_DATE_HOUR), field="hour", minimum=0, maximum=23)
        minute = _coerce_int_in_range(payload.get("minute", DEFAULT_DATE_MINUTE), field="minute", minimum=0, maximum=59)
        return ScheduleSpec(
            raw=_structured_raw(kind, payload),
            kind="date",
            month=month,
            day=day,
            year=normalized_year,
            explicit_year=normalized_year is not None,
            hour=hour,
            minute=minute,
        )

    raise ValueError("structured schedule kind must be one of once, interval, daily, or date")


def _coerce_positive_int(value: Any, *, field: str) -> int:
    parsed = _coerce_int_in_range(value, field=field, minimum=1)
    return parsed


def _coerce_int_in_range(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"structured schedule field '{field}' must be an integer") from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        if maximum is None:
            raise ValueError(f"structured schedule field '{field}' must be >= {minimum}")
        raise ValueError(f"structured schedule field '{field}' must be between {minimum} and {maximum}")
    return parsed


def _structured_raw(kind: str, payload: dict[str, Any]) -> str:
    if kind in {"once", "interval"}:
        return f"{kind}:{payload.get('interval_seconds')}"
    if kind == "daily":
        return f"daily:{int(payload.get('hour')):02d}:{int(payload.get('minute', 0)):02d}"
    if kind == "date":
        year = payload.get("year")
        prefix = f"{year}-" if year not in (None, "") else ""
        return (
            f"date:{prefix}{int(payload.get('month'))}-{int(payload.get('day'))} "
            f"{int(payload.get('hour', DEFAULT_DATE_HOUR)):02d}:{int(payload.get('minute', DEFAULT_DATE_MINUTE)):02d}"
        )
    return kind
