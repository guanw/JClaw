from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolLoopFinalizer:
    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionState:
    tool_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    finalizers: dict[str, ToolLoopFinalizer] = field(default_factory=dict)


@dataclass(slots=True)
class ToolContext:
    chat_id: str
    user_id: str = ""
    request_id: str = ""
    cwd: str = ""
    dry_run: bool = False
    execution: ToolExecutionState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    needs_confirmation: bool = False


class Tool(Protocol):
    name: str

    def describe(self) -> dict[str, Any]:
        ...

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...

    def format_result(self, action: str, result: ToolResult) -> str:
        ...
