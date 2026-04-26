from pathlib import Path

from jclaw.core.db import Database
from jclaw.tools.base import Observation, RuntimeState
from jclaw.tools.automation.tool import AutomationTool
from jclaw.tools.base import ToolContext


def test_automation_tool_creates_and_lists_schedule(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)

    created = tool.invoke(
        "create_schedule",
        {"schedule": "every 30m", "prompt": "stretch"},
        ToolContext(chat_id="chat-1"),
    )
    assert created.ok is True
    assert created.data["job"]["schedule"] == "every 30m"
    assert created.data["job"]["prompt"] == "stretch"
    assert created.data["artifacts"]["automation_job:latest"]["id"] == created.data["job"]["id"]

    listed = tool.invoke("list_schedules", {}, ToolContext(chat_id="chat-1"))
    assert listed.ok is True
    assert len(listed.data["jobs"]) == 1
    assert listed.data["jobs"][0]["prompt"] == "stretch"
    assert listed.data["allow_tool_followup"] is True
    db.close()


def test_automation_tool_updates_schedule(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)
    created = tool.invoke(
        "create_schedule",
        {"schedule": "every 30m", "prompt": "stretch"},
        ToolContext(chat_id="chat-1"),
    )
    job_id = created.data["job"]["id"]

    updated = tool.invoke(
        "update_schedule",
        {"job_id": job_id, "schedule": "daily 09:00", "prompt": "standup"},
        ToolContext(chat_id="chat-1"),
    )
    assert updated.ok is True
    assert updated.data["job"]["schedule"] == "daily 09:00"
    assert updated.data["job"]["prompt"] == "standup"
    db.close()


def test_automation_tool_creates_one_off_schedule(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)

    created = tool.invoke(
        "create_schedule",
        {"schedule": "in 30 minutes", "prompt": "stretch"},
        ToolContext(chat_id="chat-1"),
    )
    assert created.ok is True
    assert created.data["job"]["schedule"] == "in 30 minutes"
    assert created.data["job"]["prompt"] == "stretch"
    db.close()


def test_automation_tool_creates_date_schedule(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)

    created = tool.invoke(
        "create_schedule",
        {"schedule": "May 24", "prompt": "follow up"},
        ToolContext(chat_id="chat-1"),
    )

    assert created.ok is True
    assert created.data["job"]["schedule"] == "May 24"
    db.close()


def test_automation_tool_creates_date_schedule_from_structured_when(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)

    created = tool.invoke(
        "create_schedule",
        {
            "when": {"kind": "date", "month": 5, "day": 24, "hour": 15, "minute": 0},
            "prompt": "file taxes",
        },
        ToolContext(chat_id="chat-1"),
    )

    assert created.ok is True
    assert created.data["job"]["schedule"] == "date:5-24 15:00"
    db.close()


def test_automation_tool_removes_schedule(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)
    created = tool.invoke(
        "create_schedule",
        {"schedule": "hourly", "prompt": "stretch"},
        ToolContext(chat_id="chat-1"),
    )
    job_id = created.data["job"]["id"]

    removed = tool.invoke("remove_schedule", {"job_id": job_id}, ToolContext(chat_id="chat-1"))
    assert removed.ok is True
    assert removed.data["job_id"] == job_id

    listed = tool.invoke("list_schedules", {}, ToolContext(chat_id="chat-1"))
    assert listed.data["jobs"] == []
    db.close()


def test_automation_tool_materializes_job_id_from_runtime_artifact(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)
    runtime = RuntimeState(request="update reminder")
    runtime.append(
        Observation(
            ok=True,
            summary="Created schedule 3.",
            artifacts={"automation_job:latest": {"id": 3, "schedule": "every 30m", "prompt": "stretch", "next_run_at": "x", "enabled": True}},
            artifact_types=["automation_job"],
        )
    )

    params = tool.materialize_params("update_schedule", {"schedule": "hourly"}, runtime)

    assert params["job_id"] == 3
    db.close()


def test_automation_tool_updates_schedule_from_structured_when(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)
    created = tool.invoke(
        "create_schedule",
        {"schedule": "every 30m", "prompt": "stretch"},
        ToolContext(chat_id="chat-1"),
    )
    job_id = created.data["job"]["id"]

    updated = tool.invoke(
        "update_schedule",
        {
            "job_id": job_id,
            "when": {"kind": "date", "month": 5, "day": 24, "hour": 15, "minute": 0},
        },
        ToolContext(chat_id="chat-1"),
    )

    assert updated.ok is True
    assert updated.data["job"]["schedule"] == "date:5-24 15:00"
    db.close()


def test_automation_tool_rejects_unsupported_schedule_without_creating_job(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)

    created = tool.invoke(
        "create_schedule",
        {"schedule": "sometime soon", "prompt": "stretch"},
        ToolContext(chat_id="chat-1"),
    )

    assert created.ok is False
    assert "unsupported schedule" in created.summary
    listed = tool.invoke("list_schedules", {}, ToolContext(chat_id="chat-1"))
    assert listed.data["jobs"] == []
    db.close()


def test_automation_tool_describe_exposes_structured_action_specs(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)

    description = tool.describe()

    assert description["actions"]["create_schedule"]["input_schema"]["required"] == ["prompt"]
    assert description["actions"]["create_schedule"]["input_schema"]["properties"]["when"]["properties"]["kind"]["enum"] == [
        "once",
        "interval",
        "daily",
        "date",
    ]
    assert description["actions"]["update_schedule"]["binding_inputs"] == ["job_id"]
    db.close()
