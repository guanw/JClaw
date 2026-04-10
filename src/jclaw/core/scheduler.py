from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re


def _local_now() -> datetime:
    return datetime.now().astimezone()


@dataclass(slots=True)
class ScheduleSpec:
    raw: str
    kind: str
    interval_seconds: int | None = None
    hour: int | None = None
    minute: int | None = None


def parse_schedule(text: str) -> ScheduleSpec:
    raw = text.strip()
    lowered = raw.lower()

    every_match = re.fullmatch(r"every\s+(\d+)\s*([mhd]|min|mins|minute|minutes|hour|hours|day|days)", lowered)
    if every_match:
        amount = int(every_match.group(1))
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

    raise ValueError("unsupported schedule; use 'every 30m', 'hourly', or 'daily 09:00'")


def next_run_at(spec: ScheduleSpec, *, from_dt: datetime | None = None) -> datetime:
    current = from_dt or _local_now()
    if spec.kind == "interval" and spec.interval_seconds is not None:
        return current + timedelta(seconds=spec.interval_seconds)

    if spec.kind == "daily" and spec.hour is not None and spec.minute is not None:
        candidate = current.replace(hour=spec.hour, minute=spec.minute, second=0, microsecond=0)
        if candidate <= current:
            candidate += timedelta(days=1)
        return candidate

    raise ValueError(f"cannot compute next run for schedule: {spec.raw}")


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()
