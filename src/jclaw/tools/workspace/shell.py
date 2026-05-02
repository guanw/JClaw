from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from jclaw.core.defaults import (
    WORKSPACE_ALLOWED_SHELL_BINARIES,
    WORKSPACE_BLOCKED_GIT_SUBCOMMANDS,
    WORKSPACE_BLOCKED_SHELL_TOKENS,
)
from jclaw.tools.base import ToolContext, ToolResult


class WorkspaceShellMixin:
    def _prepare_shell_action(self, params: dict[str, object], ctx: ToolContext) -> ToolResult:
        command = str(params.get("command") or params.get("objective") or "").strip()
        if not command:
            return ToolResult(ok=False, summary="No shell command was provided.", data={})
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("shell",),
            ctx=ctx,
            objective=command,
            kind="grant",
            continuation_action="prepare_shell_action",
            continuation_params=params,
        )
        if permission is not None:
            return permission

        validation_error = self._validate_shell_command(command)
        if validation_error is not None:
            return ToolResult(ok=False, summary=validation_error, data={"root_path": str(root_path)})

        payload = {
            "command": command,
            "cwd": str(root_path),
            "continuation": {
                "tool": self.name,
                "approve_action": "apply_shell_request",
                "abort_action": "abort_request",
                "params": {},
            },
        }
        request = self.db.create_approval_request(
            kind="shell_mutation",
            chat_id=ctx.chat_id,
            root_path=str(root_path),
            capabilities=("shell",),
            objective=command,
            payload=payload,
        )
        return ToolResult(
            ok=True,
            summary=f"Prepared a shell command preview. Approval required: {request.request_id}",
            data={
                "request_id": request.request_id,
                "request_kind": request.kind,
                "root_path": str(root_path),
                "command": command,
            },
            needs_confirmation=True,
        )

    def _apply_shell_request(self, params: dict[str, object], ctx: ToolContext) -> ToolResult:
        request = self._require_request(str(params.get("request_id", "")).strip(), expected_kind="shell_mutation")
        if request is None:
            return ToolResult(ok=False, summary="Pending shell request not found.", data={})
        self.db.update_approval_request_status(request.request_id, "approved")
        command = str(request.payload.get("command", "")).strip()
        cwd = Path(str(request.payload.get("cwd", request.root_path)))
        result: dict[str, str]
        try:
            result = self._run_command(shlex.split(command), cwd=cwd, timeout=self.shell_timeout_seconds)
        except Exception:  # noqa: BLE001
            self.db.update_approval_request_status(request.request_id, "failed")
            raise
        self.db.update_approval_request_status(request.request_id, "applied")
        return ToolResult(
            ok=True,
            summary=f"Applied approved shell request {request.request_id}.",
            data={
                "request_id": request.request_id,
                "root_path": str(cwd),
                "command": command,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            },
        )

    def _run_command_action(self, params: dict[str, object], ctx: ToolContext) -> ToolResult:
        command = str(params.get("command") or "").strip()
        if not command:
            return ToolResult(ok=False, summary="No shell command was provided.", data={})
        target_path = self._resolve_target_path(params.get("cwd") or params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("shell",),
            ctx=ctx,
            objective=command,
            kind="grant",
            continuation_action="run_command",
            continuation_params=params,
        )
        if permission is not None:
            return permission

        validation_error = self._validate_shell_command(command)
        if validation_error is not None:
            return ToolResult(ok=False, summary=validation_error, data={"root_path": str(root_path), "command": command})

        argv = shlex.split(command)
        command_cwd = target_path if target_path.exists() and target_path.is_dir() else root_path
        result = self._run_command_result(argv, cwd=command_cwd, timeout=self.shell_timeout_seconds)
        ok = result["exit_code"] == 0
        summary = f"Command {'succeeded' if ok else 'failed'}: {command}"
        artifact = {
            "root_path": str(root_path),
            "command": command,
            "cwd": str(command_cwd),
            "exit_code": result["exit_code"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "ok": ok,
        }
        return ToolResult(
            ok=ok,
            summary=summary,
            data={
                "root_path": str(root_path),
                "command": command,
                "cwd": str(command_cwd),
                "exit_code": result["exit_code"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_command_result:latest": artifact,
                },
            },
        )

    def _validate_shell_command(self, command: str) -> str | None:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return f"Invalid shell command: {exc}"
        if not argv:
            return "Shell command is empty."
        if argv[0] not in WORKSPACE_ALLOWED_SHELL_BINARIES:
            return f"Shell command '{argv[0]}' is not allowed in v1."
        lowered_command = command.lower()
        if any(token in lowered_command for token in WORKSPACE_BLOCKED_SHELL_TOKENS):
            return "Shell command appears to require network or host app access, which is blocked in v1."
        if argv[0] == "git" and len(argv) > 1 and argv[1] in WORKSPACE_BLOCKED_GIT_SUBCOMMANDS:
            return "Remote git operations are blocked in v1."
        return None

    def _run_command(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> dict[str, str]:
        result = self._run_command_result(argv, cwd=cwd, timeout=timeout, check=True)
        return {
            "stdout": str(result["stdout"]),
            "stderr": str(result["stderr"]),
        }

    def _run_command_result(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
        check: bool = False,
    ) -> dict[str, str | int]:
        result = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            check=check,
            text=True,
            timeout=timeout,
            env={
                "HOME": str(Path.home()),
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
                "PATH": str(Path("/usr/bin")) + ":" + str(Path("/bin")) + ":" + str(Path("/usr/sbin")) + ":" + str(Path("/sbin")) + ":" + str(Path("/opt/homebrew/bin")),
                "NO_PROXY": "*",
                "http_proxy": "",
                "https_proxy": "",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
            },
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[: self.shell_output_chars],
            "stderr": result.stderr[: self.shell_output_chars],
        }
