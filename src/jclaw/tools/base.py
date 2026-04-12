from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolContext:
    chat_id: str
    user_id: str = ""
    request_id: str = ""
    cwd: str = ""
    dry_run: bool = False
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

