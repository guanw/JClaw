from pathlib import Path

from jclaw.core.db import Database
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

    listed = tool.invoke("list_schedules", {}, ToolContext(chat_id="chat-1"))
    assert listed.ok is True
    assert len(listed.data["jobs"]) == 1
    assert listed.data["jobs"][0]["prompt"] == "stretch"
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


def test_automation_tool_rejects_unsupported_schedule_without_creating_job(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = AutomationTool(db)

    created = tool.invoke(
        "create_schedule",
        {"schedule": "in 30 minutes", "prompt": "stretch"},
        ToolContext(chat_id="chat-1"),
    )

    assert created.ok is False
    assert "unsupported schedule" in created.summary
    listed = tool.invoke("list_schedules", {}, ToolContext(chat_id="chat-1"))
    assert listed.data["jobs"] == []
    db.close()
