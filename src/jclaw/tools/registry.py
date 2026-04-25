from __future__ import annotations

from typing import Any

from jclaw.tools.base import RuntimeState, Tool, ToolContext, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.describe() for tool in self._tools.values()]

    def invoke(self, tool: str, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.get(tool).invoke(action, params, ctx)

    def materialize_params(
        self,
        tool: str,
        action: str,
        params: dict[str, Any],
        runtime: RuntimeState,
    ) -> dict[str, Any]:
        instance = self.get(tool)
        materializer = getattr(instance, "materialize_params", None)
        if callable(materializer):
            return materializer(action, params, runtime)
        return dict(params)
