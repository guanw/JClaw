from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from jclaw.core.db import Database
from jclaw.core.defaults import (
    WORKSPACE_MAX_FILES_PER_CHANGE,
    WORKSPACE_MAX_PATH_ENTRIES,
    WORKSPACE_MAX_PREPARED_DIFF_BYTES,
    WORKSPACE_MAX_STEPS,
    WORKSPACE_SHELL_TIMEOUT_SECONDS,
)
from jclaw.core.environment import environment_catalog_path
from jclaw.tools.base import ActionSpec, RuntimeState, ToolContext, ToolResult, build_tool_description
from jclaw.tools.workspace.formatting import WorkspaceFormattingMixin
from jclaw.tools.workspace.git_ops import WorkspaceGitOpsMixin
from jclaw.tools.workspace.mutations import WorkspaceMutationsMixin
from jclaw.tools.workspace.permissions import WorkspacePermissionsMixin
from jclaw.tools.workspace.python_symbols import WorkspaceSymbolSearchMixin
from jclaw.tools.workspace.reads import WorkspaceReadsMixin
from jclaw.tools.workspace.shell import WorkspaceShellMixin


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class WorkspaceTool(
    WorkspaceFormattingMixin,
    WorkspaceReadsMixin,
    WorkspaceSymbolSearchMixin,
    WorkspaceMutationsMixin,
    WorkspaceGitOpsMixin,
    WorkspacePermissionsMixin,
    WorkspaceShellMixin,
):
    name = "workspace"
    MAX_CHANGE_HISTORY = 10
    CHANGE_HISTORY_TTL_SECONDS = 3600
    COMMON_HOME_FOLDERS: ClassVar[set[str]] = {
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
        self.environment_path = Path(
            options.get("environment_path") or environment_catalog_path(self.root)
        ).expanduser()
        self.max_steps = int(options.get("max_steps", WORKSPACE_MAX_STEPS))
        self.shell_timeout_seconds = int(options.get("shell_timeout_seconds", WORKSPACE_SHELL_TIMEOUT_SECONDS))
        self.shell_output_chars = self._normalize_limit(options.get("shell_output_chars"))
        self.max_prepared_diff_bytes = int(
            options.get("max_prepared_diff_bytes", WORKSPACE_MAX_PREPARED_DIFF_BYTES)
        )
        self.max_files_per_change = int(options.get("max_files_per_change", WORKSPACE_MAX_FILES_PER_CHANGE))
        self.max_path_entries = int(options.get("max_path_entries", WORKSPACE_MAX_PATH_ENTRIES))
        self.max_internal_read_bytes = self._normalize_limit(options.get("max_internal_read_bytes"))

    def describe(self) -> dict[str, Any]:
        specs = self._action_specs()
        return build_tool_description(
            name=self.name,
            description="Inspect approved local paths and prepare or apply bounded local mutations such as file edits, local git actions, and local shell commands.",
            actions=specs,
            controller_guidance=(
                "For coding tasks, inspect first but switch to mutation as soon as the edit site is known. "
                "Once you know the target file and the function, class, or exact section to change, prefer apply_patch over more reads. "
                "Do not keep repeating overlapping reads, repeated symbol lookups, or repeated searches once you have enough context to edit. "
                "If you have already identified the concrete code changes, make the edit instead of exploring further. "
                "After a code mutation, prefer a verification step such as run_command before answer or complete when a relevant check exists. "
                "Before complete on a coding task, prefer to inspect the latest diff and state what was verified or not verified. "
                "If the latest verification step failed and a plausible repair step exists, prefer another tool call instead of stopping. "
                "If the user asks to revert, undo, or put back the most recent JClaw file edit in this chat, prefer revert_last_change instead of inferring the target from git diff. "
                "If the user asks to reapply a reverted JClaw file edit in this chat, prefer redo_last_change."
            ),
            dangerous=True,
            preview_required=True,
            path_resolution={
                "repo_root": str(self.repo_root),
                "home_dir": str(self.home_dir),
                "common_home_aliases": sorted(self.COMMON_HOME_FOLDERS),
            },
        )

    def artifact_preview_limits(self) -> dict[str, dict[str, int]]:
        return {
            "workspace_file": {"content": 1_000_000},
            "workspace_diff": {"diff": 1_000_000},
            "workspace_patch": {"diff": 1_000_000},
            "workspace_command_result": {
                "stdout": 1_000_000,
                "stderr": 1_000_000,
            },
            "workspace_symbol_search": {"query": 1_000_000},
        }

    def controller_output(self, action: str, result: ToolResult) -> dict[str, Any]:
        data = result.data if isinstance(result.data, dict) else {}
        if action in {"inspect_root", "path_metadata"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "exists",
                "kind",
                "entry_count",
                "entries_truncated",
                "git_root",
                "metadata",
            )
            if isinstance(data.get("entries"), list):
                payload["entries"] = data["entries"]
            return payload
        if action in {"find_files", "search_contents", "grep"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "query",
                "match_count",
            )
            if isinstance(data.get("matches"), list):
                payload["matches"] = data["matches"]
            return payload
        if action in {"read_file"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "exists",
                "kind",
                "line_count",
                "start_line",
                "end_line",
                "char_count",
                "bytes_read",
                "truncated",
                "git_root",
            )
            if "content" in data:
                payload["content"] = self._controller_text(data.get("content"))
            return payload
        if action in {"list_file_symbols", "find_symbol", "find_references"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "query",
                "match_count",
            )
            if isinstance(data.get("symbols"), list):
                payload["symbols"] = data["symbols"]
            if isinstance(data.get("matches"), list):
                payload["matches"] = data["matches"]
            return payload
        if action in {
            "write_file",
            "create_file",
            "apply_patch",
            "apply_change_request",
            "revert_last_change",
            "redo_last_change",
            "rename_path",
            "move_path",
            "copy_path",
            "delete_path",
            "apply_path_request",
        }:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "request_id",
                "request_kind",
                "line_count",
                "char_count",
                "bytes_read",
                "truncated",
            )
            if isinstance(data.get("touched_files"), list):
                payload["touched_files"] = data["touched_files"]
            if "diff_preview" in data:
                payload["diff_preview"] = self._controller_text(data.get("diff_preview"))
            if "content" in data:
                payload["content"] = self._controller_text(data.get("content"))
            return payload
        if action in {"run_command", "apply_shell_request"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "command",
                "cwd",
                "exit_code",
                "request_id",
            )
            if "stdout" in data:
                payload["stdout"] = self._controller_text(data.get("stdout"))
            if "stderr" in data:
                payload["stderr"] = self._controller_text(data.get("stderr"))
            return payload
        if action in {"git_status"}:
            payload = self._pick_controller_fields(data, "root_path", "status")
            if "diff_stat" in data:
                payload["diff_stat"] = self._controller_text(data.get("diff_stat"))
            return payload
        if action in {"git_log"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "git_root",
                "commit_count",
                "since",
                "until",
                "author",
                "rev",
            )
            if isinstance(data.get("commits"), list):
                payload["commits"] = data["commits"]
            return payload
        if action in {"git_diff"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "git_root",
                "status",
                "has_unstaged",
                "has_staged",
            )
            if "diff" in data:
                payload["diff"] = self._controller_text(data.get("diff"))
            return payload
        if action in {"prepare_git_action", "prepare_shell_action", "abort_request"}:
            payload = self._pick_controller_fields(
                data,
                "root_path",
                "target_path",
                "request_id",
                "request_kind",
                "command",
                "preview",
            )
            if isinstance(data.get("touched_files"), list):
                payload["touched_files"] = data["touched_files"][:10]
            return payload
        return {}

    def reply_evidence(
        self,
        action: str,
        runtime: RuntimeState,
        steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if action not in {"run_command", "apply_shell_request"}:
            return []
        # Verification replies need recent code/diff evidence so the agent can
        # describe what changed without inventing edit details from the test result alone.
        evidence_actions = {
            "read_file",
            "find_symbol",
            "find_references",
            "list_file_symbols",
            "apply_patch",
            "write_file",
            "create_file",
            "git_diff",
        }
        evidence: list[dict[str, Any]] = []
        paired = list(zip(steps, runtime.observations, strict=False))
        for step, observation in reversed(paired[:-1]):
            if step.get("tool") != self.name:
                continue
            if step.get("action") not in evidence_actions:
                continue
            payload: dict[str, Any] = {
                "action": step.get("action"),
                "summary": observation.summary,
            }
            if observation.data_preview:
                payload["data_preview"] = observation.data_preview
            evidence.append(payload)
            if len(evidence) >= 3:
                break
        evidence.reverse()
        return evidence

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "inspect_root": self._inspect_root,
            "path_metadata": self._path_metadata,
            "find_files": self._find_files,
            "search_contents": lambda params, call_ctx: self._search_contents(
                {**params, "_continuation_action": "search_contents"}, call_ctx
            ),
            "grep": lambda params, call_ctx: self._search_contents({**params, "_continuation_action": "grep"}, call_ctx),
            "read_file": self._read_file,
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
            "git_log": self._git_log,
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
        except Exception as exc:
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

    def _pick_controller_fields(self, data: dict[str, Any], *keys: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in keys:
            if key in data:
                payload[key] = data[key]
        return payload

    def _normalize_limit(self, value: Any) -> int | None:
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        return limit if limit > 0 else None

    def _controller_text(self, value: Any) -> str:
        return str(value).strip()

    def _truncate_chars(self, text: str, *, limit: int | None = None) -> tuple[str, bool]:
        actual_limit = self.shell_output_chars if limit is None else limit
        if actual_limit is None or len(text) <= actual_limit:
            return text, False
        return text[:actual_limit], True

    def _truncate_bytes(self, text: str) -> tuple[str, int, bool]:
        raw = text.encode("utf-8")
        if self.max_internal_read_bytes is None or len(raw) <= self.max_internal_read_bytes:
            return text, len(raw), False
        content = raw[: self.max_internal_read_bytes].decode("utf-8", errors="ignore")
        return content, self.max_internal_read_bytes, True

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
            "grep": self._read_action(
                action="grep",
                description="Search text inside readable local files under an approved path. Supports literal text, regex, case-sensitive search, and file glob filters. Prefer this for code and text search.",
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
            "search_contents": self._read_action(
                action="search_contents",
                description="Alias of grep for controller and backward compatibility.",
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
                description="Read a local text file for coding-oriented inspection. Optional start_line and end_line let you request a focused inclusive line range. If start_line is omitted, reading starts at the beginning. If end_line is omitted, reading continues to the end of file.",
                properties={
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "objective": {"type": "string"},
                },
                required=("path",),
                produces_artifacts=("workspace_file",),
            ),
            "list_file_symbols": self._read_action(
                action="list_file_symbols",
                description="List structural symbols from one source file, such as classes, functions, interfaces, enums, and similar definitions when they can be recognized heuristically. Use this to understand file structure before choosing a narrower snippet read.",
                properties={"path": {"type": "string"}, "objective": {"type": "string"}},
                required=("path",),
                produces_artifacts=("workspace_symbol_search",),
            ),
            "find_symbol": self._read_action(
                action="find_symbol",
                description="Find exact symbol definitions by name within a source file or directory tree using language-agnostic heuristics. Prefer this when the request names a function or class, and use it for interfaces, enums, or similar symbols too.",
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
                description="Find exact symbol occurrences across source files, including whether each occurrence is likely a definition or a reference. Prefer this before edits that may affect callers or usages.",
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
                description="Replace the full contents of a local text file. Use this for full-file rewrites or generated file contents, not for narrow edits to existing code.",
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
                description="Apply one or more narrow exact-match text replacements to an existing local text file. Prefer this over write_file for small, localized edits once the target code section is known.",
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
                description="Run a single allowed local shell command inside the approved workspace and return its exit code and output. Prefer this after code edits when a meaningful local verification command is known.",
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
            "git_log": self._read_action(
                action="git_log",
                description="Read local git commit history for the current repository or a scoped path, including recent, date-bounded, author-scoped, or path-scoped commit history.",
                properties={
                    **self._path_properties(),
                    "since": {"type": "string"},
                    "until": {"type": "string"},
                    "author": {"type": "string"},
                    "rev": {"type": "string"},
                    "max_count": {"type": "integer"},
                },
                produces_artifacts=("workspace_git_log",),
            ),
            "git_diff": self._read_action(
                action="git_diff",
                description="Read local git diff details for the current repository or a scoped path.",
                properties=self._path_properties(),
                produces_artifacts=("workspace_diff",),
            ),
        }
