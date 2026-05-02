from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any

from jclaw.tools.base import ToolContext, ToolResult


class WorkspacePythonSymbolsMixin:
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
            definition_lines[str(path)] = {int(item["line_number"]) for item in symbols if item["name"] == symbol_name}
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
