from __future__ import annotations

import ast
from datetime import datetime, timezone
import difflib
import fnmatch
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

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
from jclaw.tools.base import ActionSpec, ToolContext, ToolResult


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
                "After a code mutation, prefer a verification step such as run_command before answer or complete when a relevant check exists."
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

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        if data.get("root_path"):
            lines.append(f"Root: {data['root_path']}")
        if data.get("target_path"):
            lines.append(f"Target: {data['target_path']}")
        if "exists" in data:
            lines.append(f"Exists: {data['exists']}")
        if data.get("kind"):
            lines.append(f"Kind: {data['kind']}")
        if data.get("entry_count") and data.get("kind") == "directory":
            lines.append(f"Total entries: {data['entry_count']}")
        if data.get("request_id"):
            lines.append(f"Request: {data['request_id']}")
        if data.get("request_kind"):
            lines.append(f"Request kind: {data['request_kind']}")
        if data.get("capabilities"):
            lines.append(f"Capabilities: {', '.join(str(item) for item in data['capabilities'])}")
        if data.get("entries"):
            lines.append("Entries:")
            for entry in data["entries"][:10]:
                lines.append(f"- {entry['kind']}: {entry['name']}")
            if data.get("entries_truncated"):
                shown = len(data["entries"])
                total = data.get("entry_count", shown)
                lines.append(f"Shown {shown} of {total} entries.")
        elif data.get("kind") == "directory":
            lines.append("Entries: none")
        if data.get("touched_files"):
            lines.append("Touched files:")
            for file_path in data["touched_files"][:10]:
                lines.append(f"- {file_path}")
        if data.get("source_path"):
            lines.append(f"Source: {data['source_path']}")
        if data.get("destination_path"):
            lines.append(f"Destination: {data['destination_path']}")
        if data.get("metadata"):
            lines.append("Metadata:")
            for key in ("size_bytes", "modified_at", "created_at", "suffix", "mode"):
                if key in data["metadata"]:
                    lines.append(f"- {key}: {data['metadata'][key]}")
        if "start_line" in data and "end_line" in data:
            lines.append(f"Lines: {data['start_line']}-{data['end_line']}")
        if "line_count" in data:
            lines.append(f"Line count: {data['line_count']}")
        if "char_count" in data:
            lines.append(f"Characters: {data['char_count']}")
        if "bytes_read" in data:
            lines.append(f"Bytes read: {data['bytes_read']}")
        if "truncated" in data:
            lines.append(f"Truncated: {data['truncated']}")
        if data.get("content"):
            lines.append(f"Content:\n{str(data['content'])[:4000]}")
        if data.get("symbols"):
            lines.append("Symbols:")
            for item in data["symbols"][:10]:
                lines.append(
                    f"- {item['kind']} {item['name']} ({item['path']}:{item['line_number']}-{item['end_line']})"
                )
            if data.get("match_count", 0) > len(data["symbols"]):
                lines.append(f"Shown {len(data['symbols'])} of {data['match_count']} symbols.")
        if data.get("matches"):
            lines.append("Matches:")
            for item in data["matches"][:10]:
                if "kind" in item and "name" in item and "line_number" in item:
                    lines.append(
                        f"- {item['kind']} {item['name']} ({item['path']}:{item['line_number']}-{item['end_line']})"
                    )
                elif "line_number" in item:
                    lines.append(f"- {item['path']}:{item['line_number']}: {item['line']}")
                else:
                    lines.append(f"- {item['path']}")
            if data.get("match_count", 0) > len(data["matches"]):
                lines.append(f"Shown {len(data['matches'])} of {data['match_count']} matches.")
        if data.get("diff_preview"):
            lines.append(f"Diff preview:\n{str(data['diff_preview'])[:1500]}")
        if "diff" in data:
            diff_text = str(data["diff"])
            lines.append(f"Diff:\n{diff_text[:4000]}")
        if data.get("command"):
            lines.append(f"Command: {data['command']}")
        if data.get("preview"):
            lines.append(f"Preview: {data['preview']}")
        if data.get("status"):
            lines.append(f"Git status:\n{str(data['status'])[:1200]}")
        if data.get("diff_stat"):
            lines.append(f"Git diff:\n{str(data['diff_stat'])[:1200]}")
        if data.get("stdout"):
            lines.append(f"Stdout:\n{str(data['stdout'])[:1200]}")
        if data.get("stderr"):
            lines.append(f"Stderr:\n{str(data['stderr'])[:1200]}")
        if data.get("output"):
            lines.append(f"Output:\n{str(data['output'])[:1200]}")
        return "\n".join(lines)

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
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_path:latest": {
                        "root_path": str(root_path),
                        "target_path": str(target_path),
                        "kind": kind,
                        "exists": target_exists,
                        "git_root": None if git_root is None else str(git_root),
                    },
                },
            },
        )

    def _path_metadata(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Inspect path metadata")).strip()
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="path_metadata",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        if not target_path.exists():
            return ToolResult(ok=False, summary=f"{target_path} does not exist.", data={"target_path": str(target_path)})
        stats = target_path.stat()
        kind = "directory" if target_path.is_dir() else "file"
        metadata = {
            "size_bytes": stats.st_size,
            "modified_at": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat(),
            "created_at": datetime.fromtimestamp(stats.st_ctime, tz=timezone.utc).isoformat(),
            "suffix": target_path.suffix,
            "mode": oct(stats.st_mode & 0o777),
        }
        if kind == "directory":
            metadata["entry_count"] = sum(1 for _ in target_path.iterdir())
        return ToolResult(
            ok=True,
            summary=f"Collected metadata for {target_path}.",
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "exists": True,
                "kind": kind,
                "metadata": metadata,
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_path:latest": {
                        "root_path": str(root_path),
                        "target_path": str(target_path),
                        "kind": kind,
                        "metadata": metadata,
                    },
                },
            },
        )

    def _find_files(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Find local files")).strip()
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path") or params.get("root"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="find_files",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        pattern = str(params.get("pattern", "")).strip()
        patterns = self._expand_glob_patterns(pattern)
        query = str(params.get("query", "")).strip().lower()
        if not patterns and not query:
            return ToolResult(ok=False, summary="Finding files requires a pattern or query.", data={"root_path": str(root_path)})
        search_root = target_path if target_path.exists() and target_path.is_dir() else root_path
        matches: list[dict[str, str]] = []
        match_count = 0
        for path in self._iter_searchable_paths(search_root):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root_path))
            name = path.name
            matched = False
            if patterns and any(
                fnmatch.fnmatch(name, candidate) or fnmatch.fnmatch(relative, candidate) for candidate in patterns
            ):
                matched = True
            if query and (query in name.lower() or query in relative.lower()):
                matched = True
            if not matched:
                continue
            match_count += 1
            if len(matches) < self.max_path_entries:
                matches.append({"path": relative, "name": name})
        return ToolResult(
            ok=True,
            summary=f"Found {match_count} matching file(s) under {search_root}.",
            data={
                "root_path": str(root_path),
                "target_path": str(search_root),
                "matches": matches,
                "match_count": match_count,
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_search_results:latest": {
                        "root_path": str(root_path),
                        "target_path": str(search_root),
                        "match_count": match_count,
                        "matches": matches[:10],
                    },
                },
            },
        )

    def _search_contents(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Search file contents")).strip()
        query = str(params.get("query") or params.get("text") or "").strip()
        if not query:
            return ToolResult(ok=False, summary="Searching contents requires a query string.", data={})
        case_sensitive = bool(params.get("case_sensitive", False))
        file_pattern = str(params.get("file_pattern", "")).strip()
        file_patterns = self._expand_glob_patterns(file_pattern)
        use_regex = bool(params.get("regex")) or self._looks_like_regex(query)
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path") or params.get("root"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective or query,
            kind="grant",
            continuation_action="search_contents",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        search_root = target_path if target_path.exists() and target_path.is_dir() else target_path
        candidate_files = [search_root] if search_root.exists() and search_root.is_file() else self._iter_searchable_paths(search_root)
        matches: list[dict[str, Any]] = []
        match_count = 0
        needle = query if case_sensitive else query.lower()
        pattern: re.Pattern[str] | None = None
        if use_regex:
            try:
                pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
            except re.error as exc:
                return ToolResult(ok=False, summary=f"Invalid regex query: {exc}", data={"query": query})
        for path in candidate_files:
            if not path.is_file():
                continue
            relative_path = str(path.relative_to(root_path))
            if file_patterns and not any(
                fnmatch.fnmatch(path.name, candidate) or fnmatch.fnmatch(relative_path, candidate)
                for candidate in file_patterns
            ):
                continue
            content = self._read_text(path)
            if not content:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern is not None:
                    if pattern.search(line) is None:
                        continue
                else:
                    haystack = line if case_sensitive else line.lower()
                    if needle not in haystack:
                        continue
                match_count += 1
                if len(matches) < self.max_path_entries:
                    matches.append(
                        {
                            "path": relative_path,
                            "line_number": line_number,
                            "line": line[:200],
                        }
                    )
        return ToolResult(
            ok=True,
            summary=f"Found {match_count} content match(es) for '{query}'.",
            data={
                "root_path": str(root_path),
                "target_path": str(search_root),
                "matches": matches,
                "match_count": match_count,
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_search_results:latest": {
                        "root_path": str(root_path),
                        "target_path": str(search_root),
                        "match_count": match_count,
                        "matches": matches[:10],
                        "query": query,
                    },
                },
            },
        )

    def _read_file(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Read local file")).strip()
        if params.get("path") in (None, ""):
            return ToolResult(ok=False, summary="Reading a file requires a path.", data={})
        target_path = self._resolve_target_path(params.get("path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="read_file",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        file_state = self._read_text_file_state(target_path)
        if file_state["error"]:
            return ToolResult(
                ok=False,
                summary=str(file_state["error"]),
                data={"root_path": str(root_path), "target_path": str(target_path)},
            )
        git_root = self._detect_git_root(target_path)
        line_count = int(file_state["line_count"])
        content = str(file_state["content"])
        artifact = {
            "root_path": str(root_path),
            "target_path": str(target_path),
            "kind": "file",
            "start_line": 1,
            "end_line": line_count,
            "line_count": line_count,
            "content": content,
            "truncated": bool(file_state["truncated"]),
            "git_root": None if git_root is None else str(git_root),
        }
        return ToolResult(
            ok=True,
            summary=f"Read {target_path}.",
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "exists": True,
                "kind": "file",
                "content": content,
                "line_count": line_count,
                "char_count": int(file_state["char_count"]),
                "bytes_read": int(file_state["bytes_read"]),
                "truncated": bool(file_state["truncated"]),
                "git_root": None if git_root is None else str(git_root),
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_file:latest": artifact,
                },
            },
        )

    def _read_snippet(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Read local file snippet")).strip()
        if params.get("path") in (None, ""):
            return ToolResult(ok=False, summary="Reading a snippet requires a path.", data={})
        target_path = self._resolve_target_path(params.get("path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="read_snippet",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        try:
            start_line = int(params.get("start_line", 0))
            end_line = int(params.get("end_line", 0))
        except (TypeError, ValueError):
            return ToolResult(ok=False, summary="Reading a snippet requires integer start_line and end_line.", data={})
        if start_line < 1 or end_line < 1 or start_line > end_line:
            return ToolResult(ok=False, summary="Invalid line range for read_snippet.", data={})
        file_state = self._read_text_file_state(target_path, include_full_text=True)
        if file_state["error"]:
            return ToolResult(
                ok=False,
                summary=str(file_state["error"]),
                data={"root_path": str(root_path), "target_path": str(target_path)},
            )
        line_count = int(file_state["line_count"])
        if start_line > line_count:
            return ToolResult(
                ok=False,
                summary=f"Requested snippet starts after end of file ({line_count} lines).",
                data={"root_path": str(root_path), "target_path": str(target_path), "line_count": line_count},
            )
        actual_end_line = min(end_line, line_count)
        all_lines = str(file_state["full_text"]).splitlines(keepends=True)
        raw_content = "".join(all_lines[start_line - 1 : actual_end_line])
        raw_content_bytes = raw_content.encode("utf-8")
        snippet_bytes_read = min(len(raw_content_bytes), self.max_internal_read_bytes)
        content = raw_content_bytes[: self.max_internal_read_bytes].decode("utf-8", errors="ignore")
        snippet_truncated = len(raw_content_bytes) > self.max_internal_read_bytes
        git_root = self._detect_git_root(target_path)
        artifact = {
            "root_path": str(root_path),
            "target_path": str(target_path),
            "kind": "file",
            "start_line": start_line,
            "end_line": actual_end_line,
            "line_count": line_count,
            "content": content,
            "truncated": snippet_truncated,
            "git_root": None if git_root is None else str(git_root),
        }
        return ToolResult(
            ok=True,
            summary=f"Read lines {start_line}-{actual_end_line} from {target_path}.",
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "exists": True,
                "kind": "file",
                "content": content,
                "start_line": start_line,
                "end_line": actual_end_line,
                "line_count": line_count,
                "char_count": len(content),
                "bytes_read": snippet_bytes_read,
                "truncated": snippet_truncated,
                "git_root": None if git_root is None else str(git_root),
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_file:latest": artifact,
                },
            },
        )

    def _list_file_symbols(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "List Python file symbols")).strip()
        if params.get("path") in (None, ""):
            return ToolResult(ok=False, summary="Listing file symbols requires a path.", data={})
        target_path = self._resolve_target_path(params.get("path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="list_file_symbols",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        if not target_path.exists():
            return ToolResult(ok=False, summary=f"{target_path} does not exist.", data={"target_path": str(target_path)})
        if not target_path.is_file():
            return ToolResult(ok=False, summary=f"{target_path} is not a file.", data={"target_path": str(target_path)})
        if target_path.suffix != ".py":
            return ToolResult(ok=False, summary=f"{target_path} is not a Python source file.", data={"target_path": str(target_path)})
        symbols = self._python_symbols_for_file(target_path, root_path=root_path)
        artifact = {
            "mode": "list_file_symbols",
            "root_path": str(root_path),
            "target_path": str(target_path),
            "symbols": symbols[:10],
            "symbol_count": len(symbols),
        }
        return ToolResult(
            ok=True,
            summary=f"Found {len(symbols)} symbol(s) in {target_path}.",
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "symbols": symbols,
                "match_count": len(symbols),
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_symbol_search:latest": artifact,
                },
            },
        )

    def _find_symbol(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Find Python symbol definition")).strip()
        symbol_name = str(params.get("name") or params.get("symbol") or "").strip()
        if not symbol_name:
            return ToolResult(ok=False, summary="Finding a symbol requires a name.", data={})
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective or symbol_name,
            kind="grant",
            continuation_action="find_symbol",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        matches: list[dict[str, Any]] = []
        for path in self._iter_python_search_paths(target_path, root_path=root_path):
            for symbol in self._python_symbols_for_file(path, root_path=root_path):
                if symbol["name"] != symbol_name:
                    continue
                matches.append(symbol)
        matches.sort(key=lambda item: (item["path"], item["line_number"]))
        artifact = {
            "mode": "find_symbol",
            "root_path": str(root_path),
            "target_path": str(target_path),
            "query": symbol_name,
            "matches": matches[:10],
            "match_count": len(matches),
        }
        return ToolResult(
            ok=True,
            summary=f"Found {len(matches)} symbol definition match(es) for '{symbol_name}'.",
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "query": symbol_name,
                "matches": matches,
                "match_count": len(matches),
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_symbol_search:latest": artifact,
                },
            },
        )

    def _find_references(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Find Python symbol references")).strip()
        symbol_name = str(params.get("name") or params.get("symbol") or "").strip()
        if not symbol_name:
            return ToolResult(ok=False, summary="Finding references requires a name.", data={})
        target_path = self._resolve_target_path(params.get("path") or params.get("root_path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read",),
            ctx=ctx,
            objective=objective or symbol_name,
            kind="grant",
            continuation_action="find_references",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol_name)}(?![A-Za-z0-9_])")
        definition_lines: dict[str, set[int]] = {}
        matches: list[dict[str, Any]] = []
        for path in self._iter_python_search_paths(target_path, root_path=root_path):
            symbols = self._python_symbols_for_file(path, root_path=root_path)
            definition_lines[str(path)] = {
                int(item["line_number"])
                for item in symbols
                if item["name"] == symbol_name
            }
            content = self._read_text(path)
            if not content:
                continue
            relative_path = str(path.relative_to(root_path))
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line) is None:
                    continue
                match_type = "definition" if line_number in definition_lines[str(path)] else "reference"
                matches.append(
                    {
                        "path": relative_path,
                        "line_number": line_number,
                        "line": line[:200],
                        "match_type": match_type,
                    }
                )
        matches.sort(key=lambda item: (0 if item["match_type"] == "definition" else 1, item["path"], item["line_number"]))
        artifact = {
            "mode": "find_references",
            "root_path": str(root_path),
            "target_path": str(target_path),
            "query": symbol_name,
            "matches": matches[:10],
            "match_count": len(matches),
        }
        return ToolResult(
            ok=True,
            summary=f"Found {len(matches)} reference match(es) for '{symbol_name}'.",
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "query": symbol_name,
                "matches": matches,
                "match_count": len(matches),
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_symbol_search:latest": artifact,
                },
            },
        )

    def _write_file(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Write local file")).strip()
        if params.get("path") in (None, ""):
            return ToolResult(ok=False, summary="Writing a file requires a path.", data={})
        if "content" not in params:
            return ToolResult(ok=False, summary="Writing a file requires content.", data={})
        target_path = self._resolve_target_path(params.get("path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read", "write"),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="write_file",
            continuation_params=params,
        )
        if permission is not None:
            return permission

        new_content = str(params.get("content", ""))
        if target_path.exists():
            file_state = self._read_text_file_state(target_path, include_full_text=True)
            if file_state["error"]:
                return ToolResult(
                    ok=False,
                    summary=str(file_state["error"]),
                    data={"root_path": str(root_path), "target_path": str(target_path)},
                )
            before = str(file_state["full_text"])
            if before == new_content:
                return ToolResult(
                    ok=True,
                    summary=f"No changes needed for {target_path}.",
                    data={
                        "root_path": str(root_path),
                        "target_path": str(target_path),
                        "touched_files": [],
                        "allow_tool_followup": True,
                    },
                )
            if before:
                return self._prepare_file_edit_request(
                    root_path=root_path,
                    target_path=target_path,
                    before=before,
                    after=new_content,
                    ctx=ctx,
                    objective=objective or f"Overwrite {target_path.name}",
                )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(new_content, encoding="utf-8")
        return self._direct_file_edit_result(
            root_path=root_path,
            target_path=target_path,
            before="",
            after=new_content,
            summary=f"Wrote {target_path}.",
            operation="write_file",
        )

    def _create_file(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Create local file")).strip()
        if params.get("path") in (None, ""):
            return ToolResult(ok=False, summary="Creating a file requires a path.", data={})
        if "content" not in params:
            return ToolResult(ok=False, summary="Creating a file requires content.", data={})
        target_path = self._resolve_target_path(params.get("path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("write",),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="create_file",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        if target_path.exists():
            return ToolResult(
                ok=False,
                summary=f"{target_path} already exists. Use write_file or apply_patch instead.",
                data={"root_path": str(root_path), "target_path": str(target_path)},
            )
        content = str(params.get("content", ""))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return self._direct_file_edit_result(
            root_path=root_path,
            target_path=target_path,
            before="",
            after=content,
            summary=f"Created {target_path}.",
            operation="create_file",
        )

    def _apply_patch(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Apply local patch")).strip()
        if params.get("path") in (None, ""):
            return ToolResult(ok=False, summary="Applying a patch requires a path.", data={})
        raw_hunks = params.get("hunks")
        if not isinstance(raw_hunks, list) or not raw_hunks:
            return ToolResult(ok=False, summary="Applying a patch requires a non-empty hunks list.", data={})
        target_path = self._resolve_target_path(params.get("path"))
        root_path = self._default_root_for_path(target_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("read", "write"),
            ctx=ctx,
            objective=objective,
            kind="grant",
            continuation_action="apply_patch",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        file_state = self._read_text_file_state(target_path, include_full_text=True)
        if file_state["error"]:
            return ToolResult(
                ok=False,
                summary=str(file_state["error"]),
                data={"root_path": str(root_path), "target_path": str(target_path)},
            )
        before = str(file_state["full_text"])
        after = before
        for index, raw_hunk in enumerate(raw_hunks, start=1):
            if not isinstance(raw_hunk, dict):
                return ToolResult(ok=False, summary=f"Patch hunk {index} must be an object.", data={})
            old_text = raw_hunk.get("old_text")
            new_text = raw_hunk.get("new_text", "")
            if not isinstance(old_text, str):
                return ToolResult(ok=False, summary=f"Patch hunk {index} requires string old_text.", data={})
            if not isinstance(new_text, str):
                return ToolResult(ok=False, summary=f"Patch hunk {index} requires string new_text.", data={})
            match_count = after.count(old_text)
            if match_count == 0:
                return ToolResult(
                    ok=False,
                    summary=f"Patch hunk {index} did not match the current file contents.",
                    data={"root_path": str(root_path), "target_path": str(target_path)},
                )
            if match_count > 1:
                return ToolResult(
                    ok=False,
                    summary=f"Patch hunk {index} matched multiple locations. Narrow the patch context.",
                    data={"root_path": str(root_path), "target_path": str(target_path)},
                )
            after = after.replace(old_text, new_text, 1)
        if after == before:
            return ToolResult(
                ok=True,
                summary=f"No patch changes were needed for {target_path}.",
                data={
                    "root_path": str(root_path),
                    "target_path": str(target_path),
                    "touched_files": [],
                    "allow_tool_followup": True,
                },
            )
        target_path.write_text(after, encoding="utf-8")
        return self._direct_file_edit_result(
            root_path=root_path,
            target_path=target_path,
            before=before,
            after=after,
            summary=f"Applied patch to {target_path}.",
            operation="apply_patch",
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
            data={
                "request_id": request.request_id,
                "root_path": request.root_path,
                "touched_files": touched_files,
                "allow_tool_followup": True,
            },
        )

    def _rename_path(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        source = self._resolve_target_path(params.get("path") or params.get("source_path"))
        new_name = str(params.get("new_name", "")).strip()
        if not new_name or Path(new_name).name != new_name:
            return ToolResult(ok=False, summary="Renaming a path requires a simple new_name.", data={})
        destination = source.parent / new_name
        return self._prepare_path_mutation(
            operation="rename",
            source_path=source,
            destination_path=destination,
            ctx=ctx,
            objective=str(params.get("objective", f"Rename {source.name} to {new_name}")).strip(),
            continuation_params=params,
        )

    def _move_path(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        source = self._resolve_target_path(params.get("path") or params.get("source_path"))
        destination = self._resolve_target_path(params.get("destination_path") or params.get("to_path"))
        return self._prepare_path_mutation(
            operation="move",
            source_path=source,
            destination_path=destination,
            ctx=ctx,
            objective=str(params.get("objective", f"Move {source.name}")).strip(),
            continuation_params=params,
        )

    def _copy_path(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        source = self._resolve_target_path(params.get("path") or params.get("source_path"))
        destination = self._resolve_target_path(params.get("destination_path") or params.get("to_path"))
        return self._prepare_path_mutation(
            operation="copy",
            source_path=source,
            destination_path=destination,
            ctx=ctx,
            objective=str(params.get("objective", f"Copy {source.name}")).strip(),
            continuation_params=params,
        )

    def _delete_path(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        source = self._resolve_target_path(params.get("path") or params.get("source_path"))
        return self._prepare_path_mutation(
            operation="delete",
            source_path=source,
            destination_path=None,
            ctx=ctx,
            objective=str(params.get("objective", f"Delete {source.name}")).strip(),
            continuation_params=params,
        )

    def _apply_path_request(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        request = self._require_request(str(params.get("request_id", "")).strip(), expected_kind="path_mutation")
        if request is None:
            return ToolResult(ok=False, summary="Pending path request not found.", data={})
        self.db.update_approval_request_status(request.request_id, "approved")
        operation = str(request.payload.get("operation", "")).strip()
        source = Path(str(request.payload.get("source_path", "")))
        destination_raw = request.payload.get("destination_path")
        destination = None if destination_raw in (None, "") else Path(str(destination_raw))
        try:
            if operation == "rename":
                assert destination is not None
                source.rename(destination)
            elif operation == "move":
                assert destination is not None
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            elif operation == "copy":
                assert destination is not None
                if source.is_dir():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, destination)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            elif operation == "delete":
                if source.is_dir():
                    shutil.rmtree(source)
                else:
                    source.unlink()
            else:
                raise RuntimeError(f"Unsupported path operation: {operation}")
        except Exception:  # noqa: BLE001
            self.db.update_approval_request_status(request.request_id, "failed")
            raise
        self.db.update_approval_request_status(request.request_id, "applied")
        touched_files = [str(item) for item in request.payload.get("touched_files", [])]
        return ToolResult(
            ok=True,
            summary=f"Applied approved path request {request.request_id}.",
            data={
                "request_id": request.request_id,
                "root_path": request.root_path,
                "source_path": str(source),
                "destination_path": None if destination is None else str(destination),
                "touched_files": touched_files,
            },
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

    def _run_command_action(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
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

    def _prepare_path_mutation(
        self,
        *,
        operation: str,
        source_path: Path,
        destination_path: Path | None,
        ctx: ToolContext,
        objective: str,
        continuation_params: dict[str, Any],
    ) -> ToolResult:
        mutation_root = self._common_mutation_root(source_path, destination_path)
        permission = self._ensure_grant(
            mutation_root,
            capabilities=("write",),
            ctx=ctx,
            objective=objective or f"{operation} path",
            kind="grant",
            continuation_action=f"{operation}_path" if operation in {"rename", "move", "copy", "delete"} else "inspect_root",
            continuation_params=continuation_params,
        )
        if permission is not None:
            return permission
        validation_error = self._validate_path_mutation(operation, source_path, destination_path, mutation_root)
        if validation_error is not None:
            return ToolResult(ok=False, summary=validation_error, data={"root_path": str(mutation_root)})
        touched_files = [str(source_path.relative_to(mutation_root))]
        if destination_path is not None:
            touched_files.append(str(destination_path.relative_to(mutation_root)))
        payload = {
            "operation": operation,
            "source_path": str(source_path),
            "destination_path": None if destination_path is None else str(destination_path),
            "touched_files": touched_files,
        }
        request = self.db.create_approval_request(
            kind="path_mutation",
            chat_id=ctx.chat_id,
            root_path=str(mutation_root),
            capabilities=("write",),
            objective=objective or f"{operation} path",
            payload={
                **payload,
                "continuation": {
                    "tool": self.name,
                    "approve_action": "apply_path_request",
                    "abort_action": "abort_request",
                    "params": {},
                },
            },
        )
        destination_summary = "" if destination_path is None else f" -> {destination_path.relative_to(mutation_root)}"
        return ToolResult(
            ok=True,
            summary=f"Prepared a path {operation} preview. Approval required: {request.request_id}",
            data={
                "request_id": request.request_id,
                "request_kind": request.kind,
                "root_path": str(mutation_root),
                "source_path": str(source_path),
                "destination_path": None if destination_path is None else str(destination_path),
                "touched_files": touched_files,
                "preview": f"{operation}: {source_path.relative_to(mutation_root)}{destination_summary}",
            },
            needs_confirmation=True,
        )

    def _common_mutation_root(self, source_path: Path, destination_path: Path | None) -> Path:
        candidates = [source_path]
        if destination_path is not None:
            candidates.append(destination_path)
        common = Path(os.path.commonpath([str(path) for path in candidates]))
        if common.exists() and common.is_file():
            common = common.parent
        if self._detect_git_root(common) is not None:
            return self._detect_git_root(common) or common
        return common

    def _validate_path_mutation(
        self,
        operation: str,
        source_path: Path,
        destination_path: Path | None,
        root_path: Path,
    ) -> str | None:
        if not source_path.exists():
            return f"{source_path} does not exist."
        if not self._path_within(source_path, root_path):
            return "Source path is outside the approved root."
        if operation == "delete" and source_path == root_path:
            return "Refusing to delete the approved root directly."
        if destination_path is None:
            return None
        if not self._path_within(destination_path, root_path):
            return "Destination path is outside the approved root."
        if destination_path.exists():
            return f"{destination_path} already exists."
        if operation == "rename" and destination_path.parent != source_path.parent:
            return "rename_path only supports renaming within the same parent directory."
        return None

    def _iter_python_search_paths(self, target_path: Path, *, root_path: Path) -> list[Path]:
        if target_path.exists() and target_path.is_file():
            return [target_path] if target_path.suffix == ".py" else []
        search_root = target_path if target_path.exists() and target_path.is_dir() else root_path
        return [path for path in self._iter_searchable_paths(search_root) if path.is_file() and path.suffix == ".py"]

    def _python_symbols_for_file(self, path: Path, *, root_path: Path) -> list[dict[str, Any]]:
        file_state = self._read_text_file_state(path, include_full_text=True)
        if file_state["error"]:
            return []
        text = str(file_state["full_text"])
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            return []
        lines = text.splitlines()
        relative_path = str(path.relative_to(root_path))
        symbols: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            kind = ""
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif isinstance(node, ast.FunctionDef):
                kind = "function"
            elif isinstance(node, ast.AsyncFunctionDef):
                kind = "async_function"
            if not kind:
                continue
            line_number = int(getattr(node, "lineno", 0) or 0)
            end_line = int(getattr(node, "end_lineno", line_number) or line_number)
            source_line = lines[line_number - 1][:200] if 1 <= line_number <= len(lines) else ""
            symbols.append(
                {
                    "path": relative_path,
                    "name": node.name,
                    "kind": kind,
                    "line_number": line_number,
                    "end_line": end_line,
                    "line": source_line,
                }
            )
        symbols.sort(key=lambda item: (item["line_number"], item["name"]))
        return symbols

    def _iter_searchable_paths(self, search_root: Path) -> list[Path]:
        if not search_root.exists():
            return []
        blocked_parts = {"node_modules", ".venv", "__pycache__", ".git"}
        paths: list[Path] = []
        for path in sorted(search_root.rglob("*"), key=lambda item: str(item).lower()):
            if any(part in blocked_parts for part in path.parts):
                continue
            if any(part.startswith(".") and part not in {".github"} for part in path.parts):
                continue
            paths.append(path)
        return paths

    def _looks_like_regex(self, query: str) -> bool:
        return any(token in query for token in ("\\", "|", "(", ")", "[", "]", "{", "}", "+", "*", "?"))

    def _expand_glob_patterns(self, pattern: str) -> list[str]:
        cleaned = pattern.strip()
        if not cleaned:
            return []
        if "{" not in cleaned and "}" not in cleaned and "," in cleaned:
            parts = [part.strip() for part in cleaned.split(",") if part.strip()]
            return parts or [cleaned]
        if "{" not in cleaned or "}" not in cleaned:
            return [cleaned]
        prefix, remainder = cleaned.split("{", 1)
        choices_raw, suffix = remainder.split("}", 1)
        choices = [choice.strip() for choice in choices_raw.split(",") if choice.strip()]
        if not choices:
            return [cleaned]
        return [f"{prefix}{choice}{suffix}" for choice in choices]

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

    def _prepare_file_edit_request(
        self,
        *,
        root_path: Path,
        target_path: Path,
        before: str,
        after: str,
        ctx: ToolContext,
        objective: str,
    ) -> ToolResult:
        relative_path = str(target_path.relative_to(root_path))
        edit = {
            "path": str(target_path),
            "relative_path": relative_path,
            "before": before,
            "after": after,
            "reason": objective,
        }
        payload = {
            "summary": f"Prepared overwrite for {relative_path}.",
            "edits": [edit],
            "continuation": {
                "tool": self.name,
                "approve_action": "apply_change_request",
                "abort_action": "abort_request",
                "params": {},
            },
        }
        request = self.db.create_approval_request(
            kind="file_mutation",
            chat_id=ctx.chat_id,
            root_path=str(root_path),
            capabilities=("read", "write"),
            objective=objective,
            payload=payload,
        )
        return ToolResult(
            ok=True,
            summary=f"Prepared an overwrite preview for {target_path}. Approval required: {request.request_id}",
            data={
                "request_id": request.request_id,
                "request_kind": request.kind,
                "root_path": str(root_path),
                "target_path": str(target_path),
                "touched_files": [relative_path],
                "diff_preview": self._build_diff_preview([edit]),
            },
            needs_confirmation=True,
        )

    def _direct_file_edit_result(
        self,
        *,
        root_path: Path,
        target_path: Path,
        before: str,
        after: str,
        summary: str,
        operation: str,
    ) -> ToolResult:
        relative_path = str(target_path.relative_to(root_path))
        diff_preview = self._build_diff_preview(
            [
                {
                    "path": str(target_path),
                    "relative_path": relative_path,
                    "before": before,
                    "after": after,
                    "reason": operation,
                }
            ]
        )
        after_bytes = after.encode("utf-8")
        content = after_bytes[: self.max_internal_read_bytes].decode("utf-8", errors="ignore")
        truncated = len(after_bytes) > self.max_internal_read_bytes
        patch_artifact = {
            "root_path": str(root_path),
            "target_path": str(target_path),
            "operation": operation,
            "touched_files": [relative_path],
            "diff": diff_preview,
        }
        file_artifact = {
            "root_path": str(root_path),
            "target_path": str(target_path),
            "kind": "file",
            "start_line": 1,
            "end_line": len(after.splitlines()),
            "line_count": len(after.splitlines()),
            "content": content,
            "truncated": truncated,
            "git_root": None if self._detect_git_root(target_path) is None else str(self._detect_git_root(target_path)),
        }
        return ToolResult(
            ok=True,
            summary=summary,
            data={
                "root_path": str(root_path),
                "target_path": str(target_path),
                "touched_files": [relative_path],
                "diff_preview": diff_preview,
                "content": content,
                "line_count": len(after.splitlines()),
                "char_count": len(content),
                "bytes_read": min(len(after_bytes), self.max_internal_read_bytes),
                "truncated": truncated,
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_patch:latest": patch_artifact,
                    "workspace_file:latest": file_artifact,
                },
            },
        )

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

    def _read_text_file_state(self, path: Path, *, include_full_text: bool = False) -> dict[str, Any]:
        if not path.exists():
            return {"error": f"{path} does not exist."}
        if not path.is_file():
            return {"error": f"{path} is not a file."}
        sample = path.read_bytes()[:8192]
        if b"\x00" in sample:
            return {"error": f"{path} appears to be a binary file and cannot be read as text."}
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="ignore")
        max_bytes = min(len(raw), self.max_internal_read_bytes)
        content = raw[: self.max_internal_read_bytes].decode("utf-8", errors="ignore")
        state = {
            "error": "",
            "content": content,
            "line_count": len(text.splitlines()),
            "char_count": len(content),
            "bytes_read": max_bytes,
            "truncated": len(raw) > self.max_internal_read_bytes,
        }
        if include_full_text:
            state["full_text"] = text
        return state

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

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "inspect_root": ActionSpec(
                tool=self.name,
                action="inspect_root",
                description="Inspect a local path or list a directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("workspace_path",),
            ),
            "path_metadata": ActionSpec(
                tool=self.name,
                action="path_metadata",
                description="Inspect detailed metadata for a local path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("workspace_path",),
            ),
            "find_files": ActionSpec(
                tool=self.name,
                action="find_files",
                description="Find files by name or glob pattern under an approved local path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "root": {"type": "string"},
                        "pattern": {"type": "string"},
                        "query": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("workspace_search_results",),
            ),
            "search_contents": ActionSpec(
                tool=self.name,
                action="search_contents",
                description="Search literal text inside readable local files under an approved path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "root": {"type": "string"},
                        "query": {"type": "string"},
                        "text": {"type": "string"},
                        "regex": {"type": "boolean"},
                        "case_sensitive": {"type": "boolean"},
                        "file_pattern": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("workspace_search_results",),
            ),
            "read_file": ActionSpec(
                tool=self.name,
                action="read_file",
                description="Read a local text file for coding-oriented inspection when no specific line range is requested. Do not use this when the user asks for explicit line numbers or a line range; use read_snippet instead.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    "required": ["path"],
                },
                reads=True,
                produces_artifacts=("workspace_file",),
            ),
            "read_snippet": ActionSpec(
                tool=self.name,
                action="read_snippet",
                description="Read a focused inclusive line range from a local text file. Use this when the user asks for explicit line numbers, a line range, or phrases like 'show me lines 10-40'.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "objective": {"type": "string"},
                    },
                    "required": ["path", "start_line", "end_line"],
                },
                reads=True,
                produces_artifacts=("workspace_file",),
            ),
            "list_file_symbols": ActionSpec(
                tool=self.name,
                action="list_file_symbols",
                description="List top-level and nested Python class/function definitions from a Python source file.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    "required": ["path"],
                },
                reads=True,
                produces_artifacts=("workspace_symbol_search",),
            ),
            "find_symbol": ActionSpec(
                tool=self.name,
                action="find_symbol",
                description="Find Python class or function definitions by exact symbol name within a Python file or directory tree.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "symbol": {"type": "string"},
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    "required": ["name"],
                },
                reads=True,
                produces_artifacts=("workspace_symbol_search",),
            ),
            "find_references": ActionSpec(
                tool=self.name,
                action="find_references",
                description="Find exact Python symbol occurrences across Python source files, including whether an occurrence is a definition or a reference.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "symbol": {"type": "string"},
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    "required": ["name"],
                },
                reads=True,
                produces_artifacts=("workspace_symbol_search",),
            ),
            "write_file": ActionSpec(
                tool=self.name,
                action="write_file",
                description="Replace the full contents of a local text file. Use this for full-file rewrites, not narrow edits.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                writes=True,
                produces_artifacts=("workspace_patch", "workspace_file"),
            ),
            "apply_patch": ActionSpec(
                tool=self.name,
                action="apply_patch",
                description="Apply one or more narrow exact-match text replacements to an existing local text file. Prefer this over write_file for small edits.",
                input_schema={
                    "type": "object",
                    "properties": {
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
                    "required": ["path", "hunks"],
                },
                writes=True,
                produces_artifacts=("workspace_patch", "workspace_file"),
            ),
            "create_file": ActionSpec(
                tool=self.name,
                action="create_file",
                description="Create a new local text file with the provided contents.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                writes=True,
                produces_artifacts=("workspace_patch", "workspace_file"),
            ),
            "run_command": ActionSpec(
                tool=self.name,
                action="run_command",
                description="Run a single allowed local shell command inside the approved workspace and return its exit code and output.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    "required": ["command"],
                },
                writes=True,
                produces_artifacts=("workspace_command_result",),
            ),
            "git_status": ActionSpec(
                tool=self.name,
                action="git_status",
                description="Read local git status and diff summary.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("workspace_git_status",),
            ),
            "git_diff": ActionSpec(
                tool=self.name,
                action="git_diff",
                description="Read local git diff details for the current repository or a scoped path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "root_path": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("workspace_diff",),
            ),
        }
