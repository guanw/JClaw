from __future__ import annotations

from datetime import datetime, timezone
import difflib
import json
import re
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable

from jclaw.core.db import Database
from jclaw.core.defaults import (
    WORKSPACE_ALLOWED_SHELL_BINARIES,
    WORKSPACE_BLOCKED_GIT_SUBCOMMANDS,
    WORKSPACE_BLOCKED_SHELL_TOKENS,
    WORKSPACE_MAX_FILES_PER_CHANGE,
    WORKSPACE_MAX_INTERNAL_READ_BYTES,
    WORKSPACE_MAX_PATH_ENTRIES,
    WORKSPACE_MAX_PREPARED_DIFF_BYTES,
    WORKSPACE_MAX_STEPS,
    WORKSPACE_SHELL_OUTPUT_CHARS,
    WORKSPACE_SHELL_TIMEOUT_SECONDS,
)
from jclaw.tools.base import ToolContext, ToolResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceTool:
    name = "workspace"
    COMMON_HOME_FOLDERS = {
        "Desktop",
        "Documents",
        "Downloads",
        "Movies",
        "Music",
        "Pictures",
        "Library",
    }

    def __init__(
        self,
        db: Database,
        base_dir: str | Path,
        repo_root: str | Path,
        *,
        draft_change: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.root = Path(base_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.home_dir = Path.home().expanduser().resolve()
        self._draft_change = draft_change
        options = options or {}
        self.max_steps = int(options.get("max_steps", WORKSPACE_MAX_STEPS))
        self.shell_timeout_seconds = int(options.get("shell_timeout_seconds", WORKSPACE_SHELL_TIMEOUT_SECONDS))
        self.shell_output_chars = int(options.get("shell_output_chars", WORKSPACE_SHELL_OUTPUT_CHARS))
        self.max_prepared_diff_bytes = int(
            options.get("max_prepared_diff_bytes", WORKSPACE_MAX_PREPARED_DIFF_BYTES)
        )
        self.max_files_per_change = int(options.get("max_files_per_change", WORKSPACE_MAX_FILES_PER_CHANGE))
        self.max_path_entries = int(options.get("max_path_entries", WORKSPACE_MAX_PATH_ENTRIES))
        self.max_internal_read_bytes = int(
            options.get("max_internal_read_bytes", WORKSPACE_MAX_INTERNAL_READ_BYTES)
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "actions": [
                "inspect_root",
                "prepare_change",
                "apply_change_request",
                "git_status",
                "prepare_git_action",
                "apply_git_request",
                "prepare_shell_action",
                "apply_shell_request",
                "abort_request",
            ],
            "dangerous": True,
            "preview_required": True,
        }

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "inspect_root": self._inspect_root,
            "prepare_change": self._prepare_change,
            "apply_change_request": self._apply_change_request,
            "git_status": self._git_status,
            "prepare_git_action": self._prepare_git_action,
            "apply_git_request": self._apply_git_request,
            "prepare_shell_action": self._prepare_shell_action,
            "apply_shell_request": self._apply_shell_request,
            "abort_request": self._abort_request,
        }
        if action not in handlers:
            raise ValueError(f"unsupported workspace action: {action}")
        self._trace_event("invoke_start", ctx=ctx, action=action, params=params)
        try:
            result = handlers[action](params, ctx)
        except Exception as exc:  # noqa: BLE001
            self._trace_event("invoke_error", ctx=ctx, action=action, params=params, error=str(exc))
            raise
        self._trace_event(
            "invoke_finish",
            ctx=ctx,
            action=action,
            params=params,
            result={"ok": result.ok, "summary": result.summary, "data": result.data},
        )
        return result

    def _inspect_root(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Inspect local workspace")).strip()
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="inspect_root",
            continuation_params=params,
        )
        if permission is not None:
            return permission

        target_exists = target_path.exists()
        kind = "missing"
        entries: list[dict[str, str]] = []
        entry_count = 0
        entries_truncated = False
        if target_exists and target_path.is_dir():
            kind = "directory"
            children = sorted(target_path.iterdir(), key=lambda item: item.name.lower())
            entry_count = len(children)
            entries_truncated = entry_count > self.max_path_entries
            for child in children[: self.max_path_entries]:
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "kind": "directory" if child.is_dir() else "file",
                    }
                )
        elif target_exists and target_path.is_file():
            kind = "file"
        git_root = self._detect_git_root(target_path)
        return ToolResult(
            ok=True,
            summary=f"Inspected {target_path}.",
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "exists": target_exists,
                "kind": kind,
                "entries": entries,
                "entry_count": entry_count,
                "entries_truncated": entries_truncated,
                "git_root": None if git_root is None else str(git_root),
                "approved_path": str(target_path),
            },
        )

    def _prepare_change(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective") or params.get("instruction") or "").strip()
        if not objective:
            return ToolResult(ok=False, summary="No change objective was provided.", data={})

        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read", "write"),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="prepare_change",
            continuation_params=params,
        )
        if permission is not None:
            return permission

        candidate_files = self._select_candidate_files(root_path=root_path, target_path=target_path, objective=objective)
        if not candidate_files:
            return ToolResult(
                ok=False,
                summary="I couldn't identify files to change. Specify a path inside an approved root.",
                data={"root_path": str(root_path)},
            )
        if self._draft_change is None:
            return ToolResult(
                ok=False,
                summary="Workspace drafting is unavailable because no change planner is configured.",
                data={"root_path": str(root_path)},
            )

        file_payload = []
        for candidate in candidate_files[: self.max_files_per_change]:
            file_payload.append(
                {
                    "path": str(candidate.relative_to(root_path)),
                    "content": self._read_text(candidate),
                }
            )
        draft = self._draft_change(
            {
                "objective": objective,
                "root_path": str(root_path),
                "files": file_payload,
            }
        )
        if not draft or not isinstance(draft, dict):
            return ToolResult(ok=False, summary="I couldn't prepare a change plan from the current workspace state.", data={})

        edits = self._normalize_draft_edits(root_path=root_path, draft=draft)
        if not edits:
            return ToolResult(
                ok=True,
                summary="No file changes were prepared.",
                data={"root_path": str(root_path), "implemented": True, "edits": []},
            )

        payload = {
            "summary": str(draft.get("summary", "Prepared workspace edits.")),
            "edits": edits,
        }
        request = self.db.create_approval_request(
            kind="file_mutation",
            chat_id=ctx.chat_id,
            root_path=str(root_path),
            capabilities=("read", "write"),
            objective=objective,
            payload=payload,
        )
        touched_files = [edit["relative_path"] for edit in edits]
        return ToolResult(
            ok=True,
            summary=f"Prepared a change preview for {len(edits)} file(s). Approval required: {request.request_id}",
            data={
                "request_id": request.request_id,
                "request_kind": request.kind,
                "root_path": str(root_path),
                "touched_files": touched_files,
                "diff_preview": self._build_diff_preview(edits),
            },
            needs_confirmation=True,
        )

    def _apply_change_request(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        request = self._require_request(str(params.get("request_id", "")).strip(), expected_kind="file_mutation")
        if request is None:
            return ToolResult(ok=False, summary="Pending file mutation request not found.", data={})
        self.db.update_approval_request_status(request.request_id, "approved")
        try:
            touched_files: list[str] = []
            for edit in request.payload.get("edits", []):
                path = Path(str(edit["path"]))
                before = str(edit.get("before", ""))
                current = self._read_text(path) if path.exists() else ""
                if current != before:
                    raise RuntimeError(f"File changed since preview: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(edit["after"]), encoding="utf-8")
                touched_files.append(str(edit["relative_path"]))
        except Exception:  # noqa: BLE001
            self.db.update_approval_request_status(request.request_id, "failed")
            raise
        self.db.update_approval_request_status(request.request_id, "applied")
        return ToolResult(
            ok=True,
            summary=f"Applied approved file change request {request.request_id}.",
            data={"request_id": request.request_id, "root_path": request.root_path, "touched_files": touched_files},
        )

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
        status = self._run_command(["git", "-C", str(git_root), "status", "--short", "--branch"])
        diff_stat = self._run_command(["git", "-C", str(git_root), "diff", "--stat"])
        return ToolResult(
            ok=True,
            summary=f"Collected local git status for {git_root}.",
            data={
                "root_path": str(git_root),
                "status": status["stdout"],
                "diff_stat": diff_stat["stdout"],
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
            payload=plan,
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

    def _prepare_shell_action(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
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

        payload = {"command": command, "cwd": str(root_path)}
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

    def _apply_shell_request(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
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

    def _abort_request(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        request_id = str(params.get("request_id", "")).strip()
        request = self.db.get_approval_request(request_id)
        if request is None:
            return ToolResult(ok=False, summary="Request not found.", data={})
        if request.status != "pending":
            return ToolResult(ok=False, summary=f"Request {request_id} is already {request.status}.", data={})
        self.db.update_approval_request_status(request_id, "aborted")
        return ToolResult(ok=True, summary=f"Aborted request {request_id}.", data={"request_id": request_id})

    def _ensure_grant(
        self,
        root_path: Path,
        *,
        capabilities: tuple[str, ...],
        ctx: ToolContext,
        objective: str,
        kind: str,
        continuation_action: str,
        continuation_params: dict[str, Any],
    ) -> ToolResult | None:
        missing = [cap for cap in capabilities if self._matching_grant(root_path, cap) is None]
        if not missing:
            return None
        request = self.db.create_approval_request(
            kind=kind,
            chat_id=ctx.chat_id,
            root_path=str(root_path),
            capabilities=missing,
            objective=objective,
            payload={
                "message": f"Grant {', '.join(missing)} access to {root_path}",
                "continuation": {
                    "tool": self.name,
                    "action": continuation_action,
                    "params": continuation_params,
                },
            },
        )
        return ToolResult(
            ok=True,
            summary=(
                f"Approval required to grant {', '.join(missing)} access for {root_path}. "
                f"Use /approve {request.request_id} or /deny {request.request_id}."
            ),
            data={
                "request_id": request.request_id,
                "request_kind": request.kind,
                "root_path": str(root_path),
                "capabilities": list(missing),
            },
            needs_confirmation=True,
        )

    def _matching_grant(self, path: Path, capability: str) -> Any:
        target = self._canonicalize_path(path)
        best_match = None
        best_length = -1
        for grant in self.db.list_grants(active_only=True):
            if capability not in grant.capabilities:
                continue
            root = self._canonicalize_path(grant.root_path)
            if self._path_within(target, root):
                length = len(root.parts)
                if length > best_length:
                    best_match = grant
                    best_length = length
        return best_match

    def _require_request(self, request_id: str, *, expected_kind: str) -> Any:
        request = self.db.get_approval_request(request_id)
        if request is None or request.kind != expected_kind:
            return None
        if request.status in {"denied", "aborted", "applied", "failed"}:
            return None
        return request

    def _normalize_draft_edits(self, *, root_path: Path, draft: dict[str, Any]) -> list[dict[str, str]]:
        edits: list[dict[str, str]] = []
        raw_edits = draft.get("edits", [])
        if not isinstance(raw_edits, list):
            return edits
        for item in raw_edits[: self.max_files_per_change]:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path", "")).strip()
            if not raw_path:
                continue
            resolved = self._resolve_target_path(root_path / raw_path if not Path(raw_path).is_absolute() else raw_path)
            if not self._path_within(resolved, root_path):
                continue
            before = self._read_text(resolved) if resolved.exists() else ""
            after = str(item.get("new_content", ""))
            if before == after:
                continue
            edits.append(
                {
                    "path": str(resolved),
                    "relative_path": str(resolved.relative_to(root_path)),
                    "before": before,
                    "after": after,
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        return edits

    def _build_diff_preview(self, edits: list[dict[str, str]]) -> str:
        chunks: list[str] = []
        for edit in edits:
            before_lines = edit["before"].splitlines()
            after_lines = edit["after"].splitlines()
            diff = difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{edit['relative_path']}",
                tofile=f"b/{edit['relative_path']}",
                lineterm="",
            )
            chunks.append("\n".join(diff))
        return "\n\n".join(chunks)[: self.max_prepared_diff_bytes]

    def _select_candidate_files(self, *, root_path: Path, target_path: Path, objective: str) -> list[Path]:
        if target_path.exists() and target_path.is_file():
            return [target_path]

        objective_terms = {term for term in re.findall(r"[a-z0-9_./-]+", objective.lower()) if len(term) > 2}
        candidate_paths: list[tuple[int, Path]] = []
        scanned = 0
        for path in root_path.rglob("*"):
            if scanned > 400:
                break
            if any(part.startswith(".") and part != ".github" for part in path.parts):
                continue
            if any(part in {"node_modules", ".venv", "__pycache__", ".git"} for part in path.parts):
                continue
            if not path.is_file():
                continue
            scanned += 1
            score = 0
            relative_lower = str(path.relative_to(root_path)).lower()
            for term in objective_terms:
                if term in relative_lower:
                    score += 3
            if path.suffix.lower() in {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".js", ".ts", ".tsx"}:
                score += 1
            if score:
                candidate_paths.append((score, path))
        candidate_paths.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
        if candidate_paths:
            return [item[1] for item in candidate_paths[: self.max_files_per_change]]

        fallback: list[Path] = []
        for path in root_path.rglob("*"):
            if len(fallback) >= self.max_files_per_change:
                break
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".toml", ".txt"}:
                fallback.append(path)
        return fallback

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
        result = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            check=True,
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
            "stdout": result.stdout[: self.shell_output_chars],
            "stderr": result.stderr[: self.shell_output_chars],
        }

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        data = path.read_text(encoding="utf-8", errors="ignore")
        return data[: self.max_internal_read_bytes]

    def _resolve_target_path(self, value: str | Path | None) -> Path:
        if value in (None, ""):
            return self.repo_root
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return self._normalize_user_home_path(path).resolve(strict=False)

    def _canonicalize_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return self._normalize_user_home_path(path).resolve(strict=False)

    def _normalize_user_home_path(self, path: Path) -> Path:
        parts = path.parts
        if len(parts) < 3 or parts[0] != "/" or parts[1] != "Users":
            return path
        requested_user = parts[2]
        current_user = self.home_dir.name
        if requested_user == current_user:
            return path
        remainder = parts[3:]
        candidate = self.home_dir.joinpath(*remainder) if remainder else self.home_dir
        if path.exists():
            return path
        if remainder and remainder[0] not in self.COMMON_HOME_FOLDERS:
            return path
        return candidate

    def _default_root_for_path(self, path: Path) -> Path:
        git_root = self._detect_git_root(path)
        if git_root is not None:
            return git_root
        if path.exists() and path.is_dir():
            return path
        return path.parent if path.parent != Path("") else self.repo_root

    def _detect_git_root(self, path: Path) -> Path | None:
        start = path if path.exists() and path.is_dir() else path.parent
        if not start.exists():
            start = start.parent
        try:
            result = subprocess.run(
                ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            )
        except Exception:  # noqa: BLE001
            return None
        return Path(result.stdout.strip()).resolve() if result.stdout.strip() else None

    def _path_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _trace_event(
        self,
        event: str,
        *,
        ctx: ToolContext,
        action: str,
        params: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "timestamp": _utc_now(),
            "event": event,
            "chat_id": ctx.chat_id,
            "user_id": ctx.user_id,
            "action": action,
            "params": params,
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
