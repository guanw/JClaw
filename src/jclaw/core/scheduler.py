from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


DEFAULT_DATE_HOUR = 9
DEFAULT_DATE_MINUTE = 0


def parse_schedule(raw: str) -> ScheduleSpec:
    text = str(raw).strip()
    if not text:
        raise ValueError("canonical schedule is required")

    if ":" not in text:
        raise ValueError("schedule must use canonical form such as once:1800, interval:1800, daily:09:00, or date:4-27 09:00")

    kind, _, remainder = text.partition(":")
    kind = kind.strip().lower()
    remainder = remainder.strip()

    if kind in {"once", "interval"}:
        seconds = _coerce_int_in_range(remainder, field="interval_seconds", minimum=1)
        return ScheduleSpec(raw=text, kind=kind, interval_seconds=seconds)

    if kind == "daily":
        hour_text, minute_text = _split_once(remainder, ":", error="daily schedule must use HH:MM")
        hour = _coerce_int_in_range(hour_text, field="hour", minimum=0, maximum=23)
        minute = _coerce_int_in_range(minute_text, field="minute", minimum=0, maximum=59)
        return ScheduleSpec(raw=text, kind="daily", hour=hour, minute=minute)

    if kind == "date":
        date_part, time_part = _split_once(remainder, " ", error="date schedule must use M-D HH:MM or YYYY-M-D HH:MM")
        year, month, day, explicit_year = _parse_canonical_date_part(date_part)
        hour_text, minute_text = _split_once(time_part, ":", error="date schedule time must use HH:MM")
        hour = _coerce_int_in_range(hour_text, field="hour", minimum=0, maximum=23)
        minute = _coerce_int_in_range(minute_text, field="minute", minimum=0, maximum=59)
        return ScheduleSpec(
            raw=text,
            kind="date",
            year=year,
            explicit_year=explicit_year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
        )

    raise ValueError("schedule kind must be one of once, interval, daily, or date")


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
        explicit_year = bool(payload.get("explicit_year")) and normalized_year is not None
        hour = _coerce_int_in_range(payload.get("hour", DEFAULT_DATE_HOUR), field="hour", minimum=0, maximum=23)
        minute = _coerce_int_in_range(payload.get("minute", DEFAULT_DATE_MINUTE), field="minute", minimum=0, maximum=59)
        return ScheduleSpec(
            raw=_structured_raw(kind, payload),
            kind="date",
            month=month,
            day=day,
            year=normalized_year,
            explicit_year=explicit_year,
            hour=hour,
            minute=minute,
        )

    raise ValueError("structured schedule kind must be one of once, interval, daily, or date")


def _parse_canonical_date_part(text: str) -> tuple[int | None, int, int, bool]:
    parts = [part.strip() for part in str(text).split("-")]
    if len(parts) == 2:
        month = _coerce_int_in_range(parts[0], field="month", minimum=1, maximum=12)
        day = _coerce_int_in_range(parts[1], field="day", minimum=1, maximum=31)
        return None, month, day, False
    if len(parts) == 3:
        year = _normalize_year(parts[0])
        month = _coerce_int_in_range(parts[1], field="month", minimum=1, maximum=12)
        day = _coerce_int_in_range(parts[2], field="day", minimum=1, maximum=31)
        return year, month, day, True
    raise ValueError("date schedule must use M-D or YYYY-M-D")


def _normalize_year(raw_year: str | None) -> int | None:
    if raw_year is None or not str(raw_year).strip():
        return None
    year = int(raw_year)
    if year < 100:
        return 2000 + year
    return year


def _coerce_positive_int(value: Any, *, field: str) -> int:
    return _coerce_int_in_range(value, field=field, minimum=1)


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


def _split_once(text: str, separator: str, *, error: str) -> tuple[str, str]:
    if separator not in text:
        raise ValueError(error)
    left, right = text.split(separator, 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(error)
    return left, right


def _structured_raw(kind: str, payload: dict[str, Any]) -> str:
    if kind in {"once", "interval"}:
        return f"{kind}:{payload.get('interval_seconds')}"
    if kind == "daily":
        return f"daily:{int(payload.get('hour')):02d}:{int(payload.get('minute', 0)):02d}"
    if kind == "date":
        year = payload.get("year")
        explicit_year = bool(payload.get("explicit_year")) and year not in (None, "")
        prefix = f"{year}-" if explicit_year else ""
        return (
            f"date:{prefix}{int(payload.get('month'))}-{int(payload.get('day'))} "
            f"{int(payload.get('hour', DEFAULT_DATE_HOUR)):02d}:{int(payload.get('minute', DEFAULT_DATE_MINUTE)):02d}"
        )
    return kind
