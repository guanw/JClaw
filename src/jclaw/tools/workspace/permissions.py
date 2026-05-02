from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jclaw.tools.base import ToolContext, ToolResult


class WorkspacePermissionsMixin:
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

    def _path_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
