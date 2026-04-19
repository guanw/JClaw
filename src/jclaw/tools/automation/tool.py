from __future__ import annotations

from datetime import datetime
from typing import Any

from jclaw.core.db import CronJobRecord, Database
from jclaw.core.scheduler import next_run_at, parse_schedule, to_utc_iso
from jclaw.tools.base import ToolContext, ToolResult


class AutomationTool:
    name = "automation"

    def __init__(self, db: Database) -> None:
        self.db = db

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "Create, inspect, update, and remove recurring or one-off schedules for future JClaw tasks.",
            "actions": {
                "list_schedules": {
                    "description": "List the current recurring schedules for this chat.",
                    "use_when": ["the user asks what schedules or reminders are currently configured"],
                },
                "create_schedule": {
                    "description": "Create a recurring or one-off schedule with a schedule string and prompt.",
                    "use_when": ["the user asks to add or schedule a reminder or task, including one-off reminders like 'in 30 minutes'"],
                },
                "update_schedule": {
                    "description": "Update an existing schedule's timing, prompt, or enabled state.",
                    "use_when": ["the user asks to change, edit, pause, resume, or reschedule an existing recurring task"],
                },
                "remove_schedule": {
                    "description": "Remove an existing schedule by job id.",
                    "use_when": ["the user asks to remove, cancel, or delete a recurring task"],
                },
            },
            "implemented": True,
            "read_only": False,
            "prefer_direct_result": True,
            "supports_followup": True,
        }

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        if data.get("job"):
            job = data["job"]
            lines.append(
                f"Job {job['id']}: {job['schedule']} -> {job['prompt']} (next {job['next_run_at']}, enabled={job['enabled']})"
            )
        if data.get("jobs"):
            lines.append("Schedules:")
            for job in data["jobs"]:
                lines.append(
                    f"- {job['id']}. {job['schedule']} -> {job['prompt']} (next {job['next_run_at']}, enabled={job['enabled']})"
                )
        return "\n".join(lines)

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "list_schedules": self._list_schedules,
            "create_schedule": self._create_schedule,
            "update_schedule": self._update_schedule,
            "remove_schedule": self._remove_schedule,
        }
        try:
            handler = handlers[action]
        except KeyError as exc:
            raise ValueError(f"unsupported automation action: {action}") from exc
        return handler(params, ctx)

    def _list_schedules(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        jobs = self.db.list_cron_jobs(ctx.chat_id)
        if not jobs:
            return ToolResult(ok=True, summary="No schedules configured.", data={"jobs": [], "allow_tool_followup": False})
        return ToolResult(
            ok=True,
            summary=f"Listed {len(jobs)} schedules.",
            data={"jobs": [self._serialize_job(job) for job in jobs], "allow_tool_followup": False},
        )

    def _create_schedule(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        schedule = str(params.get("schedule", "")).strip()
        prompt = str(params.get("prompt", "")).strip()
        if not schedule or not prompt:
            return ToolResult(
                ok=False,
                summary="Creating a schedule requires both schedule and prompt.",
                data={"allow_tool_followup": False},
            )
        try:
            spec = parse_schedule(schedule)
        except ValueError as exc:
            return ToolResult(ok=False, summary=str(exc), data={"schedule": schedule, "allow_tool_followup": False})
        next_run = to_utc_iso(next_run_at(spec))
        job_id = self.db.add_cron_job(ctx.chat_id, spec.raw, prompt, next_run)
        job = self.db.get_cron_job(ctx.chat_id, job_id)
        assert job is not None
        return ToolResult(
            ok=True,
            summary=f"Created schedule {job_id}.",
            data={"job": self._serialize_job(job), "allow_tool_followup": False},
        )

    def _update_schedule(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            job_id = int(params.get("job_id"))
        except (TypeError, ValueError):
            return ToolResult(ok=False, summary="Updating a schedule requires a numeric job_id.", data={"allow_tool_followup": False})
        current = self.db.get_cron_job(ctx.chat_id, job_id)
        if current is None:
            return ToolResult(ok=False, summary=f"Schedule {job_id} was not found.", data={"allow_tool_followup": False})

        schedule_param = params.get("schedule")
        prompt_param = params.get("prompt")
        enabled_param = params.get("enabled")

        schedule = None
        next_run = None
        if schedule_param not in (None, ""):
            try:
                spec = parse_schedule(str(schedule_param).strip())
            except ValueError as exc:
                return ToolResult(ok=False, summary=str(exc), data={"schedule": str(schedule_param).strip(), "allow_tool_followup": False})
            schedule = spec.raw
            next_run = to_utc_iso(next_run_at(spec))

        prompt = None if prompt_param in (None, "") else str(prompt_param).strip()

        enabled = None
        if enabled_param is not None:
            enabled = bool(enabled_param)
            if enabled and next_run is None and not current.enabled:
                base_spec = parse_schedule(schedule or current.schedule)
                next_run = to_utc_iso(next_run_at(base_spec, from_dt=datetime.now().astimezone()))

        updated = self.db.update_cron_job(
            ctx.chat_id,
            job_id,
            schedule=schedule,
            prompt=prompt,
            next_run_at=next_run,
            enabled=enabled,
        )
        if not updated:
            return ToolResult(ok=False, summary=f"No schedule changes were applied to {job_id}.", data={"allow_tool_followup": False})
        job = self.db.get_cron_job(ctx.chat_id, job_id)
        assert job is not None
        return ToolResult(
            ok=True,
            summary=f"Updated schedule {job_id}.",
            data={"job": self._serialize_job(job), "allow_tool_followup": False},
        )

    def _remove_schedule(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            job_id = int(params.get("job_id"))
        except (TypeError, ValueError):
            return ToolResult(ok=False, summary="Removing a schedule requires a numeric job_id.", data={"allow_tool_followup": False})
        deleted = self.db.remove_cron_job(ctx.chat_id, job_id)
        if not deleted:
            return ToolResult(ok=False, summary=f"Schedule {job_id} was not found.", data={"allow_tool_followup": False})
        return ToolResult(ok=True, summary=f"Removed schedule {job_id}.", data={"job_id": job_id, "allow_tool_followup": False})

    def _serialize_job(self, job: CronJobRecord) -> dict[str, Any]:
        return {
            "id": job.id,
            "schedule": job.schedule,
            "prompt": job.prompt,
            "next_run_at": job.next_run_at,
            "enabled": job.enabled,
        }
