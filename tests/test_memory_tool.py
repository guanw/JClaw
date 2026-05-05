from __future__ import annotations

from jclaw.core.db import Database
from jclaw.tools.base import ToolContext
from jclaw.tools.memory.tool import MemoryTool


def test_memory_tool_remember_list_search_forget(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = MemoryTool(db, search_limit=10)

    remembered = tool.invoke(
        "remember_fact",
        {"key": "favorite_color", "value": "blue"},
        ToolContext(chat_id="chat-1"),
    )
    assert remembered.ok is True
    assert remembered.data["artifacts"]["memory_fact:latest"] == {"key": "favorite_color", "value": "blue"}

    listed = tool.invoke("list_memories", {}, ToolContext(chat_id="chat-1"))
    assert listed.ok is True
    assert listed.data["items"] == [{"key": "favorite_color", "value": "blue"}]
    assert listed.data["artifacts"]["memory_result_set:latest"]["items"] == [{"key": "favorite_color", "value": "blue"}]

    searched = tool.invoke(
        "search_memories",
        {"query": "color"},
        ToolContext(chat_id="chat-1"),
    )
    assert searched.ok is True
    assert searched.data["items"] == [{"key": "favorite_color", "value": "blue"}]
    assert searched.data["artifacts"]["memory_result_set:latest"]["query"] == "color"

    forgotten = tool.invoke(
        "forget_memory",
        {"key": "favorite_color"},
        ToolContext(chat_id="chat-1"),
    )
    assert forgotten.ok is True

    empty = tool.invoke("list_memories", {}, ToolContext(chat_id="chat-1"))
    assert empty.data["items"] == []
    db.close()


def test_memory_tool_describe_exposes_structured_action_specs(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = MemoryTool(db, search_limit=10)

    description = tool.describe()

    assert description["actions"]["remember_fact"]["input_schema"]["required"] == ["key", "value"]
    assert description["actions"]["search_memories"]["produces_artifacts"] == ["memory_result_set"]
    db.close()


def test_memory_tool_controller_output_exposes_key_value_results(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = MemoryTool(db, search_limit=10)

    remembered = tool.invoke(
        "remember_fact",
        {"key": "favorite_color", "value": "blue"},
        ToolContext(chat_id="chat-1"),
    )
    searched = tool.invoke(
        "search_memories",
        {"query": "color"},
        ToolContext(chat_id="chat-1"),
    )

    remembered_payload = tool.controller_output("remember_fact", remembered)
    searched_payload = tool.controller_output("search_memories", searched)

    assert remembered_payload == {"key": "favorite_color", "value": "blue"}
    assert searched_payload["items"] == [{"key": "favorite_color", "value": "blue"}]
    db.close()
