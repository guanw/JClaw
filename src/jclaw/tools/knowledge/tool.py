from __future__ import annotations

from typing import Any

from jclaw.tools.base import ToolContext, ToolResult


class KnowledgeTool:
    name = "knowledge"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "actions": [
                "analyze_paths",
                "summarize_folder",
                "answer_from_paths",
            ],
            "implemented": False,
            "scaffold_only": True,
        }

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if action not in {"analyze_paths", "summarize_folder", "answer_from_paths"}:
            raise ValueError(f"unsupported knowledge action: {action}")
        return ToolResult(
            ok=True,
            summary="Knowledge tool scaffold is registered but not implemented yet.",
            data={
                "implemented": False,
                "action": action,
                "paths": params.get("paths", []),
            },
        )
