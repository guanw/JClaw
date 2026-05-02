from __future__ import annotations

from pathlib import Path
from typing import Any

from jclaw.core.defaults import WORKSPACE_BLOCKED_GIT_SUBCOMMANDS
from jclaw.tools.base import ToolContext, ToolResult


class WorkspaceGitOpsMixin:
    def _git_status(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Inspect git status")).strip()
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("git",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="git_status",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        git_root = self._detect_git_root(root_path)
        if git_root is None:
            return ToolResult(ok=False, summary=f"{root_path} is not inside a local git repository.", data={})
        git_data = self._collect_git_inspection(git_root=git_root)
        return ToolResult(
            ok=True,
            summary=f"Collected local git status for {git_root}.",
            data={
                "root_path": str(git_root),
                "status": git_data["status"],
                "diff_stat": git_data["diff_stat"],
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_git_status:latest": {
                        "root_path": str(git_root),
                        "status": git_data["status"],
                        "diff_stat": git_data["diff_stat"],
                    },
                },
            },
        )

    def _git_diff(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Inspect git diff")).strip()
        raw_target = params.get("path") or params.get("root_path")
        target_path = self._resolve_target_path(raw_target)
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("git",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="git_diff",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        git_root = self._detect_git_root(target_path)
        if git_root is None:
            return ToolResult(ok=False, summary=f"{root_path} is not inside a local git repository.", data={})

        pathspec: list[str] = []
        if raw_target not in (None, ""):
            try:
                relative = target_path.relative_to(git_root)
            except ValueError:
                return ToolResult(ok=False, summary="Target path is outside the detected git root.", data={})
            if str(relative) != ".":
                pathspec = [str(relative)]
        git_data = self._collect_git_inspection(git_root=git_root, pathspec=pathspec)
        unstaged_text = str(git_data["unstaged"]).strip()
        staged_text = str(git_data["staged"]).strip()
        diff_sections: list[str] = []
        if unstaged_text:
            diff_sections.append(f"### Unstaged\n{unstaged_text}")
        if staged_text:
            diff_sections.append(f"### Staged\n{staged_text}")
        combined_diff = "\n\n".join(diff_sections)
        has_unstaged = bool(unstaged_text)
        has_staged = bool(staged_text)
        if combined_diff:
            summary = f"Collected local git diff for {git_root}."
        else:
            summary = f"No local git diff found for {git_root}."
        artifact = {
            "root_path": str(git_root),
            "target_path": str(target_path),
            "git_root": str(git_root),
            "status": git_data["status"],
            "diff": combined_diff,
            "has_unstaged": has_unstaged,
            "has_staged": has_staged,
        }
        return ToolResult(
            ok=True,
            summary=summary,
            data={
                "root_path": str(git_root),
                "target_path": str(target_path),
                "git_root": str(git_root),
                "status": git_data["status"],
                "diff": combined_diff,
                "has_unstaged": has_unstaged,
                "has_staged": has_staged,
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_diff:latest": artifact,
                },
            },
        )

    def _prepare_git_action(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective") or params.get("action") or "").strip()
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("git",),
            ctx=ctx,
            objective=objective or "Prepare local git action",
            kind="grant",
            continuation_action="prepare_git_action",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        git_root = self._detect_git_root(root_path)
        if git_root is None:
            return ToolResult(ok=False, summary=f"{root_path} is not inside a local git repository.", data={})

        plan = self._plan_git_action(params=params, objective=objective, git_root=git_root)
        if plan is None:
            return ToolResult(ok=False, summary="Unsupported or unsafe git action requested.", data={})
        if plan.get("blocked_reason"):
            return ToolResult(ok=False, summary=str(plan["blocked_reason"]), data={})
        request = self.db.create_approval_request(
            kind="git_mutation",
            chat_id=ctx.chat_id,
            root_path=str(git_root),
            capabilities=("git",),
            objective=objective or str(plan["summary"]),
            payload={
                **plan,
                "continuation": {
                    "tool": self.name,
                    "approve_action": "apply_git_request",
                    "abort_action": "abort_request",
                    "params": {},
                },
            },
        )
        return ToolResult(
            ok=True,
            summary=f"Prepared a git action preview. Approval required: {request.request_id}",
            data={
                "request_id": request.request_id,
                "request_kind": request.kind,
                "root_path": str(git_root),
                "commands": plan["commands"],
                "preview": plan["summary"],
            },
            needs_confirmation=True,
        )

    def _apply_git_request(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        request = self._require_request(str(params.get("request_id", "")).strip(), expected_kind="git_mutation")
        if request is None:
            return ToolResult(ok=False, summary="Pending git request not found.", data={})
        self.db.update_approval_request_status(request.request_id, "approved")
        outputs: list[str] = []
        try:
            for command in request.payload.get("commands", []):
                result = self._run_command([str(part) for part in command], cwd=Path(request.root_path))
                combined = "\n".join(part for part in (result["stdout"], result["stderr"]) if part).strip()
                if combined:
                    outputs.append(combined[: self.shell_output_chars])
        except Exception:  # noqa: BLE001
            self.db.update_approval_request_status(request.request_id, "failed")
            raise
        self.db.update_approval_request_status(request.request_id, "applied")
        return ToolResult(
            ok=True,
            summary=f"Applied approved git request {request.request_id}.",
            data={
                "request_id": request.request_id,
                "root_path": request.root_path,
                "output": "\n".join(outputs)[: self.shell_output_chars],
            },
        )

    def _collect_git_inspection(self, *, git_root: Path, pathspec: list[str] | None = None) -> dict[str, str]:
        scoped_pathspec = pathspec or []
        status = self._run_command(["git", "-C", str(git_root), "status", "--short", "--branch"])
        diff_stat = self._run_command(["git", "-C", str(git_root), "diff", "--stat", "--", *scoped_pathspec])
        unstaged = self._run_command(["git", "-C", str(git_root), "diff", "--", *scoped_pathspec])
        staged = self._run_command(["git", "-C", str(git_root), "diff", "--cached", "--", *scoped_pathspec])
        return {
            "status": status["stdout"],
            "diff_stat": diff_stat["stdout"],
            "unstaged": unstaged["stdout"],
            "staged": staged["stdout"],
        }

    def _plan_git_action(self, *, params: dict[str, Any], objective: str, git_root: Path) -> dict[str, Any] | None:
        requested_action = str(params.get("git_action") or params.get("action") or "").strip().lower()
        lowered_objective = objective.lower()
        if requested_action in WORKSPACE_BLOCKED_GIT_SUBCOMMANDS or any(
            f"git {item}" in lowered_objective for item in WORKSPACE_BLOCKED_GIT_SUBCOMMANDS
        ):
            return {"blocked_reason": "Remote git operations are blocked in v1."}

        if requested_action in {"status", "diff"}:
            return None

        if requested_action == "restore" or "restore" in lowered_objective:
            commands = [["git", "-C", str(git_root), "restore", "--", "."]]
            return {"commands": commands, "summary": "Restore working tree changes in the current repository."}

        message = str(params.get("message", "")).strip()
        if requested_action == "commit" or "commit" in lowered_objective:
            if not message:
                message = "Apply changes via JClaw"
            status = self._run_command(["git", "-C", str(git_root), "status", "--short"])
            if not status["stdout"].strip():
                return {"blocked_reason": "There are no local changes to commit."}
            commands = [
                ["git", "-C", str(git_root), "add", "-A"],
                ["git", "-C", str(git_root), "commit", "-m", message],
            ]
            return {"commands": commands, "summary": f"Stage all current changes and create a local commit: {message}"}

        if requested_action == "add" or "stage" in lowered_objective:
            commands = [["git", "-C", str(git_root), "add", "-A"]]
            return {"commands": commands, "summary": "Stage all current changes in the repository."}
        return None
