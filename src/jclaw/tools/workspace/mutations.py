from __future__ import annotations

import difflib
from pathlib import Path
import shutil
from typing import Any

from jclaw.tools.base import ToolContext, ToolResult


class WorkspaceMutationsMixin:
    def _write_file(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        objective = str(params.get("objective", "Write local file")).strip()
        if params.get("path") in (None, ""):
            return ToolResult(ok=False, summary="Writing a file requires a path.", data={})
        if "content" not in params:
            return ToolResult(ok=False, summary="Writing a file requires content.", data={})
        target_path = self._resolve_target_path(params.get("path"))
        target_existed = target_path.exists()
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
            before_exists=target_existed,
            before="",
            after=new_content,
            summary=f"Wrote {target_path}.",
            operation="write_file",
            ctx=ctx,
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
            before_exists=False,
            before="",
            after=content,
            summary=f"Created {target_path}.",
            operation="create_file",
            ctx=ctx,
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
            before_exists=True,
            before=before,
            after=after,
            summary=f"Applied patch to {target_path}.",
            operation="apply_patch",
            ctx=ctx,
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
        result = ToolResult(
            ok=True,
            summary=f"Applied approved file change request {request.request_id}.",
            data={
                "request_id": request.request_id,
                "root_path": request.root_path,
                "touched_files": touched_files,
                "allow_tool_followup": True,
            },
        )
        self._record_workspace_change(
            chat_id=ctx.chat_id,
            root_path=Path(request.root_path),
            operation="apply_change_request",
            edits=[
                {
                    "path": str(edit["path"]),
                    "relative_path": str(edit["relative_path"]),
                    "before_exists": True,
                    "before": str(edit.get("before", "")),
                    "after_exists": True,
                    "after": str(edit.get("after", "")),
                }
                for edit in request.payload.get("edits", [])
            ],
        )
        return result

    def _revert_last_change(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        record = self.db.get_latest_workspace_change(
            ctx.chat_id,
            state="applied",
            max_age_seconds=self.CHANGE_HISTORY_TTL_SECONDS,
        )
        if record is None:
            return ToolResult(ok=False, summary="No recent workspace change is available to revert.", data={})
        root_path = Path(record.root_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("write",),
            ctx=ctx,
            objective=str(params.get("objective", "Revert last workspace change")).strip() or "Revert last workspace change",
            kind="grant",
            continuation_action="revert_last_change",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        try:
            applied_edits = self._apply_workspace_change_record(record, direction="undo")
        except RuntimeError as exc:
            return ToolResult(ok=False, summary=str(exc), data={"root_path": str(root_path)})
        self.db.update_workspace_change_state(record.id, state="undone")
        return self._workspace_change_result(
            root_path=root_path,
            operation="revert_last_change",
            summary=f"Reverted workspace change {record.id}.",
            edits=applied_edits,
        )

    def _redo_last_change(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        record = self.db.get_latest_workspace_change(
            ctx.chat_id,
            state="undone",
            max_age_seconds=self.CHANGE_HISTORY_TTL_SECONDS,
        )
        if record is None:
            return ToolResult(ok=False, summary="No recent workspace change is available to redo.", data={})
        root_path = Path(record.root_path)
        permission = self._ensure_grant(
            root_path,
            capabilities=("write",),
            ctx=ctx,
            objective=str(params.get("objective", "Redo last workspace change")).strip() or "Redo last workspace change",
            kind="grant",
            continuation_action="redo_last_change",
            continuation_params=params,
        )
        if permission is not None:
            return permission
        try:
            applied_edits = self._apply_workspace_change_record(record, direction="redo")
        except RuntimeError as exc:
            return ToolResult(ok=False, summary=str(exc), data={"root_path": str(root_path)})
        self.db.update_workspace_change_state(record.id, state="applied")
        return self._workspace_change_result(
            root_path=root_path,
            operation="redo_last_change",
            summary=f"Reapplied workspace change {record.id}.",
            edits=applied_edits,
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
        before_exists: bool,
        before: str,
        after: str,
        summary: str,
        operation: str,
        ctx: ToolContext,
    ) -> ToolResult:
        relative_path = str(target_path.relative_to(root_path))
        edit = {
            "path": str(target_path),
            "relative_path": relative_path,
            "before": before,
            "after": after,
            "reason": operation,
        }
        self._record_workspace_change(
            chat_id=ctx.chat_id,
            root_path=root_path,
            operation=operation,
            edits=[{**edit, "before_exists": before_exists, "after_exists": True}],
        )
        diff_preview = self._build_diff_preview([edit])
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
        detected_git_root = self._detect_git_root(target_path)
        file_artifact = {
            "root_path": str(root_path),
            "target_path": str(target_path),
            "kind": "file",
            "start_line": 1,
            "end_line": len(after.splitlines()),
            "line_count": len(after.splitlines()),
            "content": content,
            "truncated": truncated,
            "git_root": None if detected_git_root is None else str(detected_git_root),
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

    def _record_workspace_change(
        self,
        *,
        chat_id: str,
        root_path: Path,
        operation: str,
        edits: list[dict[str, str]],
    ) -> None:
        file_states = [
            {
                "path": str(item["path"]),
                "relative_path": str(item["relative_path"]),
                "before_exists": bool(item.get("before_exists", True)),
                "before": str(item["before"]),
                "after_exists": bool(item.get("after_exists", True)),
                "after": str(item["after"]),
            }
            for item in edits
        ]
        self.db.record_workspace_change(
            chat_id=chat_id,
            root_path=str(root_path),
            operation=operation,
            touched_files=[str(item["relative_path"]) for item in edits],
            file_states=file_states,
        )
        self.db.prune_workspace_changes(
            chat_id,
            keep_latest=self.MAX_CHANGE_HISTORY,
            max_age_seconds=self.CHANGE_HISTORY_TTL_SECONDS,
        )

    def _apply_workspace_change_record(self, record: Any, *, direction: str) -> list[dict[str, str]]:
        applied_edits: list[dict[str, str]] = []
        for item in record.file_states:
            path = Path(str(item["path"]))
            before_exists = bool(item.get("before_exists"))
            before = str(item.get("before", ""))
            after_exists = bool(item.get("after_exists", True))
            after = str(item.get("after", ""))
            expected_exists = after_exists if direction == "undo" else before_exists
            expected_text = after if direction == "undo" else before
            target_exists = before_exists if direction == "undo" else after_exists
            target_text = before if direction == "undo" else after
            current_exists = path.exists()
            current_text = path.read_text(encoding="utf-8") if current_exists else ""
            if current_exists != expected_exists or current_text != expected_text:
                raise RuntimeError(f"Cannot safely {direction} change for {path}; file contents have diverged.")
            if target_exists:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(target_text, encoding="utf-8")
            elif path.exists():
                path.unlink()
            applied_edits.append(
                {
                    "path": str(path),
                    "relative_path": str(item["relative_path"]),
                    "before": current_text,
                    "after": target_text if target_exists else "",
                    "reason": direction,
                }
            )
        return applied_edits

    def _workspace_change_result(
        self,
        *,
        root_path: Path,
        operation: str,
        summary: str,
        edits: list[dict[str, str]],
    ) -> ToolResult:
        diff_preview = self._build_diff_preview(edits)
        touched_files = [str(item["relative_path"]) for item in edits]
        patch_artifact = {
            "root_path": str(root_path),
            "target_path": str(root_path),
            "operation": operation,
            "touched_files": touched_files,
            "diff": diff_preview,
        }
        return ToolResult(
            ok=True,
            summary=summary,
            data={
                "root_path": str(root_path),
                "target_path": str(root_path),
                "touched_files": touched_files,
                "diff_preview": diff_preview,
                "allow_tool_followup": True,
                "artifacts": {
                    "workspace_patch:latest": patch_artifact,
                },
            },
        )
