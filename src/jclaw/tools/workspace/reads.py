from __future__ import annotations

from datetime import datetime, timezone
import fnmatch
from pathlib import Path
import re
from typing import Any

from jclaw.tools.base import ToolContext, ToolResult


class WorkspaceReadsMixin:
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

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        data = path.read_text(encoding="utf-8", errors="ignore")
        return data[: self.max_internal_read_bytes]
