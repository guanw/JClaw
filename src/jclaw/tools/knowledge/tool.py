from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from jclaw.core.db import Database
from jclaw.core.defaults import (
    KNOWLEDGE_MAX_ANSWER_CITATIONS,
    KNOWLEDGE_MAX_CHUNKS_PER_FILE,
    KNOWLEDGE_MAX_FILE_READ_BYTES,
    KNOWLEDGE_MAX_FOLDER_SCAN_FILES,
    KNOWLEDGE_MAX_TOTAL_CHUNKS,
    KNOWLEDGE_TEXT_PREVIEW_CHARS,
)
from jclaw.tools.base import ActionSpec, ToolContext, ToolResult, append_field, append_list_section, build_tool_description
from jclaw.tools.knowledge.models import DocumentChunk, ExtractedDocument
from jclaw.tools.knowledge.registry import KnowledgeReaderRegistry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeTool:
    name = "knowledge"
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
        summarize_documents: Any | None = None,
        answer_question: Any | None = None,
        analyze_image: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.root = Path(base_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.home_dir = Path.home().expanduser().resolve()
        options = options or {}
        self.max_file_read_bytes = int(options.get("max_file_read_bytes", KNOWLEDGE_MAX_FILE_READ_BYTES))
        self.max_folder_scan_files = int(options.get("max_folder_scan_files", KNOWLEDGE_MAX_FOLDER_SCAN_FILES))
        self.max_chunks_per_file = int(options.get("max_chunks_per_file", KNOWLEDGE_MAX_CHUNKS_PER_FILE))
        self.max_total_chunks = int(options.get("max_total_chunks", KNOWLEDGE_MAX_TOTAL_CHUNKS))
        self.text_preview_chars = int(options.get("text_preview_chars", KNOWLEDGE_TEXT_PREVIEW_CHARS))
        self.max_answer_citations = int(options.get("max_answer_citations", KNOWLEDGE_MAX_ANSWER_CITATIONS))
        self.registry = KnowledgeReaderRegistry()

    def describe(self) -> dict[str, Any]:
        specs = self._action_specs()
        return build_tool_description(
            name=self.name,
            description="Read approved local files and answer questions from grounded extracted content with citations.",
            actions=specs,
            grounded=True,
            supported_suffixes=[
                ".txt",
                ".md",
                ".py",
                ".js",
                ".ts",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".csv",
                ".pdf",
            ],
            controller_contract={
                "result_fields": [
                    "grounded",
                    "partial",
                    "summary_text",
                    "scanned_files",
                    "scan_truncated",
                ],
                "list_fields": {
                    "supported_files": 10,
                    "unsupported_files": 10,
                    "citations": 4,
                },
                "result_previews": {
                    "summary_text": self.text_preview_chars,
                },
            },
        )

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        append_field(lines, "Grounded", data.get("grounded"), include_when=lambda value: value is not None)
        append_field(lines, "Partial", data.get("partial"), include_when=lambda value: value is not None)
        if data.get("partial"):
            append_list_section(
                lines,
                "Unsupported files:",
                data.get("unsupported_files"),
                lambda item: f"- {item['path']}: {item['reason']}",
                limit=10,
            )
        if action == "analyze_paths":
            append_list_section(
                lines,
                "Supported files:",
                data.get("supported_files"),
                lambda item: f"- {item['path']}",
                limit=10,
            )
        if action == "analyze_paths" or data.get("partial") or data.get("scan_truncated"):
            append_field(lines, "Scanned files", data.get("scanned_files"))
            append_field(lines, "Scan truncated", data.get("scan_truncated"), include_when=lambda value: value is not None)
        append_list_section(
            lines,
            "Citations:",
            data.get("citations"),
            lambda item: f"- {item['path']} [{item['chunk_id']}]",
            limit=6,
        )
        return "\n".join(lines)

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "analyze_paths": self._analyze_paths,
        }
        if action not in handlers:
            raise ValueError(f"unsupported knowledge action: {action}")
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

    def _analyze_paths(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = str(params.get("question", "")).strip()
        requested_paths = self._coerce_paths(params)
        if not requested_paths:
            return ToolResult(ok=False, summary="No paths were provided for analysis.", data={})
        permission = self._ensure_read_grants(
            requested_paths,
            ctx=ctx,
            objective=question or "Analyze local files",
            continuation_action="analyze_paths",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        docs, unsupported, scan_meta = self._collect_documents(requested_paths)
        payload = self._build_analysis_payload(docs, unsupported, scan_meta)
        return ToolResult(
            ok=True,
            summary=f"Analyzed {len(payload['supported_files'])} readable file(s).",
            data={
                **payload,
                "allow_tool_followup": True,
                "artifacts": {
                    "knowledge_context:latest": self._knowledge_context_artifact(payload),
                },
            },
        )

    def _coerce_paths(self, params: dict[str, Any]) -> list[Path]:
        raw_paths = params.get("paths")
        items: list[str | Path] = []
        if isinstance(raw_paths, list):
            items.extend(raw_paths)
        elif raw_paths not in (None, ""):
            items.append(raw_paths)
        elif params.get("path"):
            items.append(params["path"])
        return [self._resolve_target_path(item) for item in items if item not in (None, "")]

    def _ensure_read_grants(
        self,
        requested_paths: list[Path],
        *,
        ctx: ToolContext,
        objective: str,
        continuation_action: str,
        continuation_params: dict[str, Any],
    ) -> ToolResult | None:
        for requested_path in requested_paths:
            root_path = self._default_root_for_path(requested_path)
            if self._matching_grant(root_path, "read") is not None:
                continue
            request = self.db.create_approval_request(
                kind="grant",
                chat_id=ctx.chat_id,
                root_path=str(root_path),
                capabilities=("read",),
                objective=objective,
                payload={
                    "message": f"Grant read access to {root_path}",
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
                    f"Approval required to grant read access for {root_path}. "
                    f"Use /approve {request.request_id} or /deny {request.request_id}."
                ),
                data={
                    "request_id": request.request_id,
                    "request_kind": request.kind,
                    "root_path": str(root_path),
                    "capabilities": ["read"],
                },
                needs_confirmation=True,
            )
        return None

    def _collect_documents(
        self,
        requested_paths: list[Path],
    ) -> tuple[list[ExtractedDocument], list[dict[str, str]], dict[str, Any]]:
        documents: list[ExtractedDocument] = []
        unsupported: list[dict[str, str]] = []
        scanned_files = 0
        truncated = False
        for requested_path in requested_paths:
            for file_path in self._expand_files(requested_path):
                if scanned_files >= self.max_folder_scan_files:
                    truncated = True
                    break
                scanned_files += 1
                reader = self.registry.get_reader(file_path)
                if reader is None:
                    unsupported.append({"path": str(file_path), "reason": "Unsupported file type."})
                    continue
                document = reader.extract(file_path, max_bytes=self.max_file_read_bytes)
                if not document.text.strip():
                    unsupported.append({"path": str(file_path), "reason": "No readable text extracted."})
                    continue
                documents.append(document)
            if scanned_files >= self.max_folder_scan_files:
                break
        return (
            documents,
            unsupported,
            {
                "requested_paths": [str(path) for path in requested_paths],
                "scanned_files": scanned_files,
                "scan_truncated": truncated,
            },
        )

    def _build_analysis_payload(
        self,
        documents: list[ExtractedDocument],
        unsupported: list[dict[str, str]],
        scan_meta: dict[str, Any],
    ) -> dict[str, Any]:
        chunks = self._chunk_documents(documents)
        supported_files = []
        for index, document in enumerate(documents, start=1):
            supported_files.append(
                {
                    "order": index,
                    "path": document.path,
                    "title": document.title,
                    "file_type": document.file_type,
                    "preview": document.text[: self.text_preview_chars],
                    "warnings": document.warnings,
                }
            )
        return {
            "requested_paths": scan_meta["requested_paths"],
            "supported_files": supported_files,
            "unsupported_files": unsupported,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "path": chunk.path,
                    "text": chunk.text,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                }
                for chunk in chunks
            ],
            "citations": [],
            "grounded": bool(documents),
            "partial": bool(unsupported) or bool(scan_meta["scan_truncated"]),
            "scanned_files": scan_meta["scanned_files"],
            "scan_truncated": scan_meta["scan_truncated"],
        }

    def _chunk_documents(self, documents: list[ExtractedDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        max_chars = max(600, self.text_preview_chars * 3)
        for document in documents:
            start = 0
            for chunk_index in range(self.max_chunks_per_file):
                if start >= len(document.text):
                    break
                end = min(len(document.text), start + max_chars)
                chunk_text = document.text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{Path(document.path).name}:{chunk_index + 1}",
                            path=document.path,
                            text=chunk_text,
                            start_offset=start,
                            end_offset=end,
                        )
                    )
                if len(chunks) >= self.max_total_chunks:
                    return chunks
                start = end
        return chunks

    def _expand_files(self, requested_path: Path) -> list[Path]:
        if requested_path.exists() and requested_path.is_file():
            return [requested_path]
        if not requested_path.exists() or not requested_path.is_dir():
            return []
        files: list[Path] = []
        for path in sorted(requested_path.rglob("*"), key=lambda item: str(item).lower()):
            if len(files) >= self.max_folder_scan_files:
                break
            if any(part.startswith(".") and part not in {".github"} for part in path.parts):
                continue
            if any(part in {"node_modules", ".venv", "__pycache__", ".git"} for part in path.parts):
                continue
            if path.is_file():
                files.append(path)
        return files

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
        if path.exists() and path.is_dir():
            return path
        return path.parent if path.parent != Path("") else self.repo_root

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
            "analyze_paths": ActionSpec(
                tool=self.name,
                action="analyze_paths",
                description="Extract and inventory readable content from approved paths.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paths": {"type": "array", "items": {"type": "string"}},
                        "path": {"type": "string"},
                        "question": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("knowledge_context",),
            ),
        }

    def _knowledge_context_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "requested_paths": payload.get("requested_paths", []),
            "supported_files": payload.get("supported_files", [])[:5],
            "unsupported_files": payload.get("unsupported_files", [])[:5],
            "grounded": payload.get("grounded"),
            "partial": payload.get("partial"),
            "citations": payload.get("citations", [])[:5],
        }
