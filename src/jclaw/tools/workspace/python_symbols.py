from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from jclaw.tools.base import ToolContext, ToolResult


class WorkspaceSymbolSearchMixin:
    SYMBOL_PATTERNS: ClassVar[tuple[tuple[str, re.Pattern[str]], ...]] = (
        ("class", re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("enum", re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("struct", re.compile(r"^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("trait", re.compile(r"^\s*trait\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("type", re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("function", re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("function", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("function", re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
        ("function", re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_!?=]*)\b")),
        (
            "function",
            re.compile(
                r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>"
            ),
        ),
        (
            "function",
            re.compile(
                r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?function\b"
            ),
        ),
        ("function", re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{")),
    )
    def _list_file_symbols(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "List source file symbols")).strip()
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
        symbols = self._symbols_for_file(target_path, root_path=root_path)
        artifact = {
            "mode": "list_file_symbols",
            "root_path": str(root_path),
            "target_path": str(target_path),
            "symbols": symbols,
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
                "artifacts": {
                    "workspace_symbol_search:latest": artifact,
                },
            },
        )

    def _find_symbol(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Find symbol definition")).strip()
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
        for path in self._iter_symbol_search_paths(target_path, root_path=root_path):
            for symbol in self._symbols_for_file(path, root_path=root_path):
                if symbol["name"] == symbol_name:
                    matches.append(symbol)
        matches.sort(key=lambda item: (item["path"], item["line_number"]))
        artifact = {
            "mode": "find_symbol",
            "root_path": str(root_path),
            "target_path": str(target_path),
            "query": symbol_name,
            "matches": matches,
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
                "artifacts": {
                    "workspace_symbol_search:latest": artifact,
                },
            },
        )

    def _find_references(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Find symbol references")).strip()
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
        for path in self._iter_symbol_search_paths(target_path, root_path=root_path):
            symbols = self._symbols_for_file(path, root_path=root_path)
            definition_lines[str(path)] = {int(item["line_number"]) for item in symbols if item["name"] == symbol_name}
            content = self._read_text(path)
            if not content:
                continue
            relative_path = str(path.relative_to(root_path))
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line) is None:
                    continue
                matches.append(
                    {
                        "path": relative_path,
                        "line_number": line_number,
                        "line": line[:200],
                        "match_type": "definition" if line_number in definition_lines[str(path)] else "reference",
                    }
                )
        matches.sort(key=lambda item: (0 if item["match_type"] == "definition" else 1, item["path"], item["line_number"]))
        artifact = {
            "mode": "find_references",
            "root_path": str(root_path),
            "target_path": str(target_path),
            "query": symbol_name,
            "matches": matches,
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
                "artifacts": {
                    "workspace_symbol_search:latest": artifact,
                },
            },
        )

    def _iter_symbol_search_paths(self, target_path: Path, *, root_path: Path) -> list[Path]:
        if target_path.exists() and target_path.is_file():
            return [target_path]
        search_root = target_path if target_path.exists() and target_path.is_dir() else root_path
        return [
            path
            for path in self._iter_searchable_paths(search_root)
            if path.is_file() and self._is_symbol_searchable_file(path)
        ]

    def _symbols_for_file(self, path: Path, *, root_path: Path) -> list[dict[str, Any]]:
        if not self._is_symbol_searchable_file(path):
            return []
        content = self._read_text(path)
        if not content:
            return []
        relative_path = str(path.relative_to(root_path))
        symbols: list[dict[str, Any]] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            for kind, pattern in self.SYMBOL_PATTERNS:
                match = pattern.search(line)
                if match is None:
                    continue
                symbols.append(
                    {
                        "path": relative_path,
                        "name": match.group(1),
                        "kind": kind,
                        "line_number": line_number,
                        "end_line": line_number,
                        "line": line[:200],
                    }
                )
                break
        symbols.sort(key=lambda item: (item["line_number"], item["name"]))
        return symbols

    def _is_symbol_searchable_file(self, path: Path) -> bool:
        try:
            sample = path.read_bytes()[:8192]
        except OSError:
            return False
        return b"\x00" not in sample
