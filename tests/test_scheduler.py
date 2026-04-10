from datetime import datetime

from jclaw.core.scheduler import next_run_at, parse_schedule


def test_parse_interval_schedule() -> None:
    spec = parse_schedule("every 30m")
    run_at = next_run_at(spec, from_dt=datetime.fromisoformat("2026-04-10T10:00:00-04:00"))
    assert spec.kind == "interval"
    assert int((run_at.timestamp() - datetime.fromisoformat("2026-04-10T10:00:00-04:00").timestamp()) / 60) == 30


def test_parse_daily_schedule() -> None:
    spec = parse_schedule("daily 09:00")
    run_at = next_run_at(spec, from_dt=datetime.fromisoformat("2026-04-10T10:00:00-04:00"))
    assert spec.kind == "daily"
    assert run_at.hour == 9
    assert run_at.day == 11

