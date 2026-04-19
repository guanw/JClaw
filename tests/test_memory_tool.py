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

    listed = tool.invoke("list_memories", {}, ToolContext(chat_id="chat-1"))
    assert listed.ok is True
    assert listed.data["items"] == [{"key": "favorite_color", "value": "blue"}]

    searched = tool.invoke(
        "search_memories",
        {"query": "color"},
        ToolContext(chat_id="chat-1"),
    )
    assert searched.ok is True
    assert searched.data["items"] == [{"key": "favorite_color", "value": "blue"}]

    forgotten = tool.invoke(
        "forget_memory",
        {"key": "favorite_color"},
        ToolContext(chat_id="chat-1"),
    )
    assert forgotten.ok is True

    empty = tool.invoke("list_memories", {}, ToolContext(chat_id="chat-1"))
    assert empty.data["items"] == []
    db.close()
