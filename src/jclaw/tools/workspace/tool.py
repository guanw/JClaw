from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from jclaw.core.db import Database
from jclaw.core.defaults import (
    WORKSPACE_MAX_FILES_PER_CHANGE,
    WORKSPACE_MAX_INTERNAL_READ_BYTES,
    WORKSPACE_MAX_PATH_ENTRIES,
    WORKSPACE_MAX_PREPARED_DIFF_BYTES,
    WORKSPACE_MAX_STEPS,
    WORKSPACE_SHELL_OUTPUT_CHARS,
    WORKSPACE_SHELL_TIMEOUT_SECONDS,
)
from jclaw.tools.base import ActionSpec, ToolContext, ToolResult
from jclaw.tools.workspace.formatting import WorkspaceFormattingMixin
from jclaw.tools.workspace.git_ops import WorkspaceGitOpsMixin
from jclaw.tools.workspace.mutations import WorkspaceMutationsMixin
from jclaw.tools.workspace.permissions import WorkspacePermissionsMixin
from jclaw.tools.workspace.python_symbols import WorkspacePythonSymbolsMixin
from jclaw.tools.workspace.reads import WorkspaceReadsMixin
from jclaw.tools.workspace.shell import WorkspaceShellMixin


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceTool(
    WorkspaceFormattingMixin,
    WorkspaceReadsMixin,
    WorkspacePythonSymbolsMixin,
    WorkspaceMutationsMixin,
    WorkspaceGitOpsMixin,
    WorkspacePermissionsMixin,
    WorkspaceShellMixin,
):
    name = "workspace"
    MAX_CHANGE_HISTORY = 10
    CHANGE_HISTORY_TTL_SECONDS = 3600
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
        draft_change: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.root = Path(base_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.home_dir = Path.home().expanduser().resolve()
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
        specs = self._action_specs()
        return {
            "name": self.name,
            "description": "Inspect approved local paths and prepare or apply bounded local mutations such as file edits, local git actions, and local shell commands.",
            "controller_guidance": (
                "For coding tasks, inspect first but switch to mutation as soon as the edit site is known. "
                "Once you know the target file and the function, class, or exact section to change, prefer apply_patch over more reads. "
                "Do not keep repeating overlapping reads, repeated symbol lookups, or repeated searches once you have enough context to edit. "
                "If you have already identified the concrete code changes, make the edit instead of exploring further. "
                "After a code mutation, prefer a verification step such as run_command before answer or complete when a relevant check exists. "
                "If the user asks to revert, undo, or put back the most recent JClaw file edit in this chat, prefer revert_last_change instead of inferring the target from git diff. "
                "If the user asks to reapply a reverted JClaw file edit in this chat, prefer redo_last_change."
            ),
            "actions": {name: spec.to_dict() for name, spec in specs.items()},
            "dangerous": True,
            "preview_required": True,
            "prefer_direct_result": True,
            "path_resolution": {
                "repo_root": str(self.repo_root),
                "home_dir": str(self.home_dir),
                "common_home_aliases": sorted(self.COMMON_HOME_FOLDERS),
            },
            "supports_followup": True,
            "controller_contract": {
                "result_fields": [
                    "root_path",
                    "target_path",
                    "exists",
                    "kind",
                    "entry_count",
                    "entries_truncated",
                    "match_count",
                    "query",
                    "request_id",
                    "request_kind",
                    "diff_preview",
                    "content",
                    "cwd",
                    "exit_code",
                    "stdout",
                    "stderr",
                    "line_count",
                    "start_line",
                    "end_line",
                    "char_count",
                    "bytes_read",
                    "truncated",
                    "git_root",
                    "status",
                    "diff",
                    "has_unstaged",
                    "has_staged",
                ],
                "list_fields": {
                    "entries": 10,
                    "matches": 10,
                    "symbols": 10,
                    "touched_files": 10,
                },
                "result_previews": {
                    "content": 4000,
                    "diff": 4000,
                    "stdout": 4000,
                    "stderr": 4000,
                },
                "artifact_previews": {
                    "workspace_file": {"content": 4000},
                    "workspace_diff": {"diff": 4000},
                    "workspace_patch": {"diff": 4000},
                    "workspace_command_result": {
                        "stdout": 4000,
                        "stderr": 4000,
                    },
                    "workspace_symbol_search": {
                        "query": 220,
                    },
                },
            },
        }

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "inspect_root": self._inspect_root,
            "path_metadata": self._path_metadata,
            "find_files": self._find_files,
            "search_contents": self._search_contents,
            "read_file": self._read_file,
            "read_snippet": self._read_snippet,
            "list_file_symbols": self._list_file_symbols,
            "find_symbol": self._find_symbol,
            "find_references": self._find_references,
            "write_file": self._write_file,
            "apply_patch": self._apply_patch,
            "create_file": self._create_file,
            "revert_last_change": self._revert_last_change,
            "redo_last_change": self._redo_last_change,
            "run_command": self._run_command_action,
            "rename_path": self._rename_path,
            "move_path": self._move_path,
            "copy_path": self._copy_path,
            "delete_path": self._delete_path,
            "apply_change_request": self._apply_change_request,
            "apply_path_request": self._apply_path_request,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
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

    def _abort_request(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        request_id = str(params.get("request_id", "")).strip()
        request = self.db.get_approval_request(request_id)
        if request is None:
            return ToolResult(ok=False, summary="Request not found.", data={})
        if request.status != "pending":
            return ToolResult(ok=False, summary=f"Request {request_id} is already {request.status}.", data={})
        self.db.update_approval_request_status(request_id, "aborted")
        return ToolResult(ok=True, summary=f"Aborted request {request_id}.", data={"request_id": request_id})

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

    def _schema(self, properties: dict[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = list(required)
        return schema

    def _path_properties(self) -> dict[str, Any]:
        return {
            "path": {"type": "string"},
            "root_path": {"type": "string"},
            "objective": {"type": "string"},
        }

    def _root_search_properties(self) -> dict[str, Any]:
        return {
            "path": {"type": "string"},
            "root_path": {"type": "string"},
            "root": {"type": "string"},
            "objective": {"type": "string"},
        }

    def _read_action(
        self,
        *,
        action: str,
        description: str,
        properties: dict[str, Any],
        required: tuple[str, ...] = (),
        produces_artifacts: tuple[str, ...],
    ) -> ActionSpec:
        return ActionSpec(
            tool=self.name,
            action=action,
            description=description,
            input_schema=self._schema(properties, required=required),
            reads=True,
            produces_artifacts=produces_artifacts,
        )

    def _write_action(
        self,
        *,
        action: str,
        description: str,
        properties: dict[str, Any],
        required: tuple[str, ...] = (),
        produces_artifacts: tuple[str, ...],
    ) -> ActionSpec:
        return ActionSpec(
            tool=self.name,
            action=action,
            description=description,
            input_schema=self._schema(properties, required=required),
            writes=True,
            produces_artifacts=produces_artifacts,
        )

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "inspect_root": self._read_action(
                action="inspect_root",
                description="Inspect a local path or list a directory.",
                properties=self._path_properties(),
                produces_artifacts=("workspace_path",),
            ),
            "path_metadata": self._read_action(
                action="path_metadata",
                description="Inspect detailed metadata for a local path.",
                properties=self._path_properties(),
                produces_artifacts=("workspace_path",),
            ),
            "find_files": self._read_action(
                action="find_files",
                description="Find files by name or glob pattern under an approved local path.",
                properties={
                    **self._root_search_properties(),
                    "pattern": {"type": "string"},
                    "query": {"type": "string"},
                },
                produces_artifacts=("workspace_search_results",),
            ),
            "search_contents": self._read_action(
                action="search_contents",
                description="Search literal text inside readable local files under an approved path.",
                properties={
                    **self._root_search_properties(),
                    "query": {"type": "string"},
                    "text": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "case_sensitive": {"type": "boolean"},
                    "file_pattern": {"type": "string"},
                },
                produces_artifacts=("workspace_search_results",),
            ),
            "read_file": self._read_action(
                action="read_file",
                description="Read a local text file for coding-oriented inspection when no specific line range is requested. Do not use this when the user asks for explicit line numbers or a line range; use read_snippet instead.",
                properties={"path": {"type": "string"}, "objective": {"type": "string"}},
                required=("path",),
                produces_artifacts=("workspace_file",),
            ),
            "read_snippet": self._read_action(
                action="read_snippet",
                description="Read a focused inclusive line range from a local text file. Use this when the user asks for explicit line numbers, a line range, or phrases like 'show me lines 10-40'.",
                properties={
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "objective": {"type": "string"},
                },
                required=("path", "start_line", "end_line"),
                produces_artifacts=("workspace_file",),
            ),
            "list_file_symbols": self._read_action(
                action="list_file_symbols",
                description="List top-level and nested Python class/function definitions from a Python source file.",
                properties={"path": {"type": "string"}, "objective": {"type": "string"}},
                required=("path",),
                produces_artifacts=("workspace_symbol_search",),
            ),
            "find_symbol": self._read_action(
                action="find_symbol",
                description="Find Python class or function definitions by exact symbol name within a Python file or directory tree.",
                properties={
                    "name": {"type": "string"},
                    "symbol": {"type": "string"},
                    **self._path_properties(),
                },
                required=("name",),
                produces_artifacts=("workspace_symbol_search",),
            ),
            "find_references": self._read_action(
                action="find_references",
                description="Find exact Python symbol occurrences across Python source files, including whether an occurrence is a definition or a reference.",
                properties={
                    "name": {"type": "string"},
                    "symbol": {"type": "string"},
                    **self._path_properties(),
                },
                required=("name",),
                produces_artifacts=("workspace_symbol_search",),
            ),
            "write_file": self._write_action(
                action="write_file",
                description="Replace the full contents of a local text file. Use this for full-file rewrites, not narrow edits.",
                properties={
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "objective": {"type": "string"},
                },
                required=("path", "content"),
                produces_artifacts=("workspace_patch", "workspace_file"),
            ),
            "apply_patch": self._write_action(
                action="apply_patch",
                description="Apply one or more narrow exact-match text replacements to an existing local text file. Prefer this over write_file for small edits.",
                properties={
                    "path": {"type": "string"},
                    "hunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"},
                            },
                            "required": ["old_text", "new_text"],
                        },
                    },
                    "objective": {"type": "string"},
                },
                required=("path", "hunks"),
                produces_artifacts=("workspace_patch", "workspace_file"),
            ),
            "create_file": self._write_action(
                action="create_file",
                description="Create a new local text file with the provided contents.",
                properties={
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "objective": {"type": "string"},
                },
                required=("path", "content"),
                produces_artifacts=("workspace_patch", "workspace_file"),
            ),
            "revert_last_change": self._write_action(
                action="revert_last_change",
                description="Revert the most recent file mutation that JClaw applied in this chat, if the affected files have not diverged.",
                properties={"objective": {"type": "string"}},
                produces_artifacts=("workspace_patch",),
            ),
            "redo_last_change": self._write_action(
                action="redo_last_change",
                description="Reapply the most recently reverted file mutation that JClaw applied in this chat, if the affected files still match the reverted state.",
                properties={"objective": {"type": "string"}},
                produces_artifacts=("workspace_patch",),
            ),
            "run_command": self._write_action(
                action="run_command",
                description="Run a single allowed local shell command inside the approved workspace and return its exit code and output.",
                properties={
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    **self._path_properties(),
                },
                required=("command",),
                produces_artifacts=("workspace_command_result",),
            ),
            "git_status": self._read_action(
                action="git_status",
                description="Read local git status and diff summary.",
                properties=self._path_properties(),
                produces_artifacts=("workspace_git_status",),
            ),
            "git_diff": self._read_action(
                action="git_diff",
                description="Read local git diff details for the current repository or a scoped path.",
                properties=self._path_properties(),
                produces_artifacts=("workspace_diff",),
            ),
        }
