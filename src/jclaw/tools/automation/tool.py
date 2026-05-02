from __future__ import annotations

from datetime import datetime
from typing import Any

from jclaw.core.db import CronJobRecord, Database
from jclaw.core.scheduler import next_run_at, parse_schedule_input, to_utc_iso
from jclaw.tools.base import ActionSpec, RuntimeState, ToolContext, ToolResult, append_list_section, build_tool_description


class AutomationTool:
    name = "automation"

    def __init__(self, db: Database) -> None:
        self.db = db
        self.cron = db.cron

    def describe(self) -> dict[str, Any]:
        specs = self._action_specs()
        return build_tool_description(
            name=self.name,
            description="Create, inspect, update, and remove recurring or one-off schedules for future JClaw tasks.",
            actions=specs,
        )

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        if data.get("job"):
            job = data["job"]
            lines.append(
                f"Job {job['id']}: {job['schedule']} -> {job['prompt']} (next {job['next_run_at']}, enabled={job['enabled']})"
            )
        append_list_section(
            lines,
            "Schedules:",
            data.get("jobs"),
            lambda job: f"- {job['id']}. {job['schedule']} -> {job['prompt']} (next {job['next_run_at']}, enabled={job['enabled']})",
        )
        return "\n".join(lines)

    def materialize_params(
        self,
        action: str,
        params: dict[str, Any],
        runtime: RuntimeState,
    ) -> dict[str, Any]:
        materialized = dict(params)
        raw_job_id = str(materialized.get("job_id", "")).strip().lower()
        if action in {"update_schedule", "remove_schedule"} and (not raw_job_id or raw_job_id in {"latest", "selected", "automation_job"}):
            job_id = self._job_id_from_runtime(runtime)
            if job_id is not None:
                materialized["job_id"] = job_id
        return materialized

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
        jobs = self.cron.list_jobs(ctx.chat_id)
        if not jobs:
            return ToolResult(ok=True, summary="No schedules configured.", data={"jobs": [], "allow_tool_followup": False})
        return ToolResult(
            ok=True,
            summary=f"Listed {len(jobs)} schedules.",
            data={
                "jobs": [self._serialize_job(job) for job in jobs],
                "allow_tool_followup": True,
                "artifacts": {
                    "automation_job_list:latest": {
                        "jobs": [self._serialize_job(job) for job in jobs],
                    }
                },
            },
        )

    def _create_schedule(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        prompt = str(params.get("prompt", "")).strip()
        when = params.get("when")
        if not isinstance(when, dict) or not prompt:
            return ToolResult(
                ok=False,
                summary="Creating a schedule requires prompt plus a structured when object.",
                data={"allow_tool_followup": False},
            )
        try:
            spec = parse_schedule_input(when=when)
        except ValueError as exc:
            return ToolResult(
                ok=False,
                summary=str(exc),
                data={"when": when, "allow_tool_followup": False},
            )
        next_run = to_utc_iso(next_run_at(spec))
        job_id = self.cron.add_job(ctx.chat_id, spec.raw, prompt, next_run)
        job = self.cron.get_job(ctx.chat_id, job_id)
        assert job is not None
        return ToolResult(
            ok=True,
            summary=f"Created schedule {job_id}.",
            data={
                "job": self._serialize_job(job),
                "allow_tool_followup": False,
                "artifacts": {
                    "automation_job:latest": self._serialize_job(job),
                },
            },
        )

    def _update_schedule(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            job_id = int(params.get("job_id"))
        except (TypeError, ValueError):
            return ToolResult(ok=False, summary="Updating a schedule requires a numeric job_id.", data={"allow_tool_followup": False})
        current = self.cron.get_job(ctx.chat_id, job_id)
        if current is None:
            return ToolResult(ok=False, summary=f"Schedule {job_id} was not found.", data={"allow_tool_followup": False})

        when_param = params.get("when")
        prompt_param = params.get("prompt")
        enabled_param = params.get("enabled")

        schedule = None
        next_run = None
        if when_param not in (None, {}):
            try:
                spec = parse_schedule_input(when=when_param if isinstance(when_param, dict) else None)
            except ValueError as exc:
                return ToolResult(
                    ok=False,
                    summary=str(exc),
                    data={
                        "when": when_param,
                        "allow_tool_followup": False,
                    },
                )
            schedule = spec.raw
            next_run = to_utc_iso(next_run_at(spec))

        prompt = None if prompt_param in (None, "") else str(prompt_param).strip()

        enabled = None
        if enabled_param is not None:
            enabled = bool(enabled_param)
            if enabled and next_run is None and not current.enabled:
                base_spec = parse_schedule_input(schedule=schedule or current.schedule)
                next_run = to_utc_iso(next_run_at(base_spec, from_dt=datetime.now().astimezone()))

        updated = self.cron.update_job(
            ctx.chat_id,
            job_id,
            schedule=schedule,
            prompt=prompt,
            next_run_at=next_run,
            enabled=enabled,
        )
        if not updated:
            return ToolResult(ok=False, summary=f"No schedule changes were applied to {job_id}.", data={"allow_tool_followup": False})
        job = self.cron.get_job(ctx.chat_id, job_id)
        assert job is not None
        return ToolResult(
            ok=True,
            summary=f"Updated schedule {job_id}.",
            data={
                "job": self._serialize_job(job),
                "allow_tool_followup": False,
                "artifacts": {
                    "automation_job:latest": self._serialize_job(job),
                },
            },
        )

    def _remove_schedule(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            job_id = int(params.get("job_id"))
        except (TypeError, ValueError):
            return ToolResult(ok=False, summary="Removing a schedule requires a numeric job_id.", data={"allow_tool_followup": False})
        deleted = self.cron.remove_job(ctx.chat_id, job_id)
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

    def _job_id_from_runtime(self, runtime: RuntimeState) -> int | None:
        job_artifact = runtime.artifacts_by_type.get("automation_job")
        if isinstance(job_artifact, dict) and str(job_artifact.get("id", "")).strip().isdigit():
            return int(str(job_artifact["id"]).strip())
        job_list = runtime.artifacts_by_type.get("automation_job_list")
        if isinstance(job_list, dict):
            jobs = job_list.get("jobs", [])
            if isinstance(jobs, list) and len(jobs) == 1:
                item = jobs[0]
                if isinstance(item, dict) and str(item.get("id", "")).strip().isdigit():
                    return int(str(item["id"]).strip())
        return None

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "list_schedules": ActionSpec(
                tool=self.name,
                action="list_schedules",
                description="List the current recurring schedules for this chat.",
                input_schema={"type": "object", "properties": {}},
                reads=True,
                produces_artifacts=("automation_job_list",),
            ),
            "create_schedule": ActionSpec(
                tool=self.name,
                action="create_schedule",
                description="Create a recurring or one-off schedule with a structured 'when' object and prompt.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "when": self._when_schema(),
                        "prompt": {"type": "string"},
                    },
                    "required": ["when", "prompt"],
                },
                writes=True,
                produces_artifacts=("automation_job",),
            ),
            "update_schedule": ActionSpec(
                tool=self.name,
                action="update_schedule",
                description="Update an existing schedule's timing, prompt, or enabled state using a structured 'when' object.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "integer"},
                        "when": self._when_schema(),
                        "prompt": {"type": "string"},
                        "enabled": {"type": "boolean"},
                    },
                },
                writes=True,
                requires_artifacts=("automation_job", "automation_job_list"),
                produces_artifacts=("automation_job",),
                binding_inputs=("job_id",),
            ),
            "remove_schedule": ActionSpec(
                tool=self.name,
                action="remove_schedule",
                description="Remove an existing schedule by job id.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "integer"},
                    },
                    "required": ["job_id"],
                },
                writes=True,
                requires_artifacts=("automation_job", "automation_job_list"),
                binding_inputs=("job_id",),
            ),
        }

    def _when_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["once", "interval", "daily", "date"],
                },
                "interval_seconds": {"type": "integer"},
                "hour": {"type": "integer"},
                "minute": {"type": "integer"},
                "month": {"type": "integer"},
                "day": {"type": "integer"},
                "year": {"type": "integer"},
                "explicit_year": {"type": "boolean"},
            },
            "required": ["kind"],
        }
