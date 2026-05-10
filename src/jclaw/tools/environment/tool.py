from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jclaw.core.db import Database
from jclaw.core.environment import sync_environment_catalog
from jclaw.tools.base import ActionSpec, ToolContext, ToolResult, build_tool_description


class EnvironmentTool:
    name = "environment"

    def __init__(
        self,
        db: Database,
        *,
        repo_root: str | Path,
        environment_path: str | Path,
    ) -> None:
        self.db = db
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.environment_path = Path(environment_path).expanduser()

    def describe(self) -> dict[str, Any]:
        return build_tool_description(
            name=self.name,
            description="Inspect the local execution environment and known command capabilities available to JClaw.",
            actions=self._action_specs(),
        )

    def format_result(self, action: str, result: ToolResult) -> str:
        if action != "inspect":
            return result.summary
        data = result.data if isinstance(result.data, dict) else {}
        lines = [result.summary]
        repo_root = str(data.get("repo_root", "")).strip()
        if repo_root:
            lines.append(f"Repo root: {repo_root}")
        approved_roots = data.get("approved_roots")
        if isinstance(approved_roots, list) and approved_roots:
            lines.append("Approved roots:")
            for root in approved_roots:
                lines.append(f"- {root}")
        known_commands = data.get("known_commands")
        if isinstance(known_commands, list) and known_commands:
            lines.append("Known commands:")
            for item in known_commands:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                description = str(item.get("description", "")).strip()
                if name and description:
                    lines.append(f"- {name}: {description}")
                elif name:
                    lines.append(f"- {name}")
        return "\n".join(lines)

    def controller_output(self, action: str, result: ToolResult) -> dict[str, Any]:
        if action != "inspect":
            return {}
        data = result.data if isinstance(result.data, dict) else {}
        payload: dict[str, Any] = {}
        repo_root = str(data.get("repo_root", "")).strip()
        if repo_root:
            payload["repo_root"] = repo_root
        approved_roots = data.get("approved_roots")
        if isinstance(approved_roots, list):
            payload["approved_roots"] = approved_roots
        known_commands = data.get("known_commands")
        if isinstance(known_commands, list):
            payload["known_commands"] = known_commands
        return payload

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if action != "inspect":
            raise ValueError(f"unsupported environment action: {action}")
        return self._inspect(params, ctx)

    def _inspect(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        del params
        del ctx
        catalog = sync_environment_catalog(
            self.environment_path,
            repo_root=self.repo_root,
            approved_roots=self._approved_roots(),
            shell=os.environ.get("SHELL", ""),
            cwd=self.repo_root,
            search_path=os.environ.get("PATH", ""),
            command_names=(),
        )
        payload = self._controller_payload(catalog)
        return ToolResult(
            ok=True,
            summary=f"Inspected the local environment and found {len(payload['known_commands'])} known command(s).",
            data={
                **payload,
                "artifacts": {
                    "environment_snapshot:latest": catalog,
                },
            },
        )

    def _approved_roots(self) -> list[Path]:
        roots = {self.repo_root}
        for grant in self.db.list_grants(active_only=True):
            roots.add(Path(grant.root_path).expanduser().resolve())
        return sorted(roots, key=str)

    def _controller_payload(self, catalog: dict[str, Any]) -> dict[str, Any]:
        roots = catalog.get("roots", {})
        commands = catalog.get("commands", {})
        approved_roots = roots.get("approved", [])
        known_commands: list[dict[str, str]] = []
        if isinstance(commands, dict):
            for name in sorted(commands):
                details = commands.get(name)
                if not isinstance(details, dict):
                    continue
                command_name = str(name).strip()
                description = str(details.get("description", "")).strip()
                if not command_name:
                    continue
                known_commands.append(
                    {
                        "name": command_name,
                        "description": description,
                    }
                )
        return {
            "repo_root": str(roots.get("repo_root", "")).strip(),
            "approved_roots": [str(item).strip() for item in approved_roots if str(item).strip()],
            "known_commands": known_commands,
        }

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "inspect": ActionSpec(
                tool=self.name,
                action="inspect",
                description="Inspect the current local command environment, approved roots, and known command capabilities.",
                input_schema={"type": "object", "properties": {}},
                reads=True,
                produces_artifacts=("environment_snapshot",),
            )
        }
