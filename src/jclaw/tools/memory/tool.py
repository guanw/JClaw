from __future__ import annotations

from typing import Any

from jclaw.core.db import Database
from jclaw.tools.base import ActionSpec, ToolContext, ToolResult


class MemoryTool:
    name = "memory"

    def __init__(self, db: Database, *, search_limit: int = 10) -> None:
        self.db = db
        self.search_limit = search_limit

    def describe(self) -> dict[str, Any]:
        specs = self._action_specs()
        return {
            "name": self.name,
            "description": "Store, search, list, and forget long-lived chat memory facts.",
            "actions": {name: spec.to_dict() for name, spec in specs.items()},
            "implemented": True,
            "read_only": False,
            "prefer_direct_result": True,
            "supports_followup": False,
        }

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        if result.data.get("items"):
            lines.append("Memories:")
            for item in result.data["items"]:
                lines.append(f"- {item['key']} = {item['value']}")
        return "\n".join(lines)

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "remember_fact": self._remember_fact,
            "list_memories": self._list_memories,
            "search_memories": self._search_memories,
            "forget_memory": self._forget_memory,
        }
        try:
            handler = handlers[action]
        except KeyError as exc:
            raise ValueError(f"unsupported memory action: {action}") from exc
        return handler(params, ctx)

    def _remember_fact(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        key = str(params.get("key", "")).strip()
        value = str(params.get("value", "")).strip()
        if not key or not value:
            return ToolResult(ok=False, summary="remember_fact requires both key and value.", data={})
        self.db.remember(ctx.chat_id, key, value)
        return ToolResult(
            ok=True,
            summary=f"Remembered '{key}'.",
            data={
                "key": key,
                "value": value,
                "allow_tool_followup": False,
                "artifacts": {
                    "memory_fact:latest": {
                        "key": key,
                        "value": value,
                    }
                },
            },
        )

    def _list_memories(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        items = self.db.list_memories(ctx.chat_id, limit=self.search_limit)
        if not items:
            return ToolResult(ok=True, summary="No memories stored yet.", data={"items": [], "allow_tool_followup": False})
        return ToolResult(
            ok=True,
            summary=f"Listed {len(items)} stored memories.",
            data={
                "items": [{"key": item.key, "value": item.value} for item in items],
                "artifacts": {
                    "memory_result_set:latest": {
                        "items": [{"key": item.key, "value": item.value} for item in items],
                    }
                },
                "allow_tool_followup": False,
            },
        )

    def _search_memories(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(params.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, summary="search_memories requires a query.", data={})
        items = self.db.search_memories(ctx.chat_id, query, self.search_limit)
        if not items:
            return ToolResult(ok=True, summary=f"No stored memories matched '{query}'.", data={"items": [], "allow_tool_followup": False})
        return ToolResult(
            ok=True,
            summary=f"Found {len(items)} memory match(es) for '{query}'.",
            data={
                "items": [{"key": item.key, "value": item.value} for item in items],
                "artifacts": {
                    "memory_result_set:latest": {
                        "query": query,
                        "items": [{"key": item.key, "value": item.value} for item in items],
                    }
                },
                "allow_tool_followup": False,
            },
        )

    def _forget_memory(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        key = str(params.get("key", "")).strip()
        if not key:
            return ToolResult(ok=False, summary="forget_memory requires a key.", data={})
        deleted = self.db.forget(ctx.chat_id, key)
        if not deleted:
            return ToolResult(ok=False, summary=f"I didn't have a memory stored for '{key}'.", data={"key": key, "allow_tool_followup": False})
        return ToolResult(ok=True, summary=f"Forgot '{key}'.", data={"key": key, "allow_tool_followup": False})

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "remember_fact": ActionSpec(
                tool=self.name,
                action="remember_fact",
                description="Store or update a memory fact as key/value.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["key", "value"],
                },
                writes=True,
                produces_artifacts=("memory_fact",),
            ),
            "list_memories": ActionSpec(
                tool=self.name,
                action="list_memories",
                description="List currently stored chat memories.",
                input_schema={"type": "object", "properties": {}},
                reads=True,
                produces_artifacts=("memory_result_set",),
            ),
            "search_memories": ActionSpec(
                tool=self.name,
                action="search_memories",
                description="Search stored chat memories relevant to a query.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
                reads=True,
                produces_artifacts=("memory_result_set",),
            ),
            "forget_memory": ActionSpec(
                tool=self.name,
                action="forget_memory",
                description="Remove a stored memory by key.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                    },
                    "required": ["key"],
                },
                writes=True,
            ),
        }
