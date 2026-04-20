from __future__ import annotations

from typing import Any

from jclaw.core.db import Database
from jclaw.tools.base import ToolContext, ToolResult


class PermissionsTool:
    name = "permissions"

    def __init__(self, db: Database) -> None:
        self.db = db

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "Inspect and manage local access grants and pending approval requests.",
            "prefer_direct_result": True,
            "actions": {
                "list_grants": {
                    "description": "List active local access grants.",
                    "use_when": ["the user asks what access is currently granted"],
                },
                "grant_access": {
                    "description": "Create or update a local access grant for a root path and capabilities.",
                    "use_when": ["the user explicitly asks to grant or allow access to a path"],
                },
                "revoke_grant": {
                    "description": "Revoke an active grant by id.",
                    "use_when": ["the user explicitly asks to revoke granted access"],
                },
                "list_pending_requests": {
                    "description": "List pending approval requests.",
                    "use_when": ["the user asks what approvals are waiting"],
                },
            },
            "supports_followup": True,
        }

    def format_result(self, action: str, result: ToolResult) -> str:
        data = result.data
        if action == "list_grants":
            grants = data.get("grants", [])
            if not grants:
                return "No active grants."
            lines = ["Active grants:"]
            for grant in grants:
                lines.append(f"{grant['id']}. {grant['root_path']} [{', '.join(grant['capabilities'])}]")
            return "\n".join(lines)
        if action == "list_pending_requests":
            requests = data.get("requests", [])
            if not requests:
                return "No pending approval requests."
            lines = ["Pending approval requests:"]
            for request in requests:
                lines.append(
                    f"{request['request_id']}. {request['kind']} {request['root_path']} "
                    f"[{', '.join(request['capabilities'])}]"
                )
            return "\n".join(lines)
        if action == "grant_access" and "grant" in data:
            grant = data["grant"]
            return f"Granted {', '.join(grant['capabilities'])} access for {grant['root_path']}. Grant id: {grant['id']}"
        if action == "revoke_grant" and data.get("revoked"):
            return f"Grant {data['grant_id']} revoked."
        return result.summary

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "list_grants": self._list_grants,
            "grant_access": self._grant_access,
            "revoke_grant": self._revoke_grant,
            "list_pending_requests": self._list_pending_requests,
        }
        try:
            handler = handlers[action]
        except KeyError as exc:
            raise ValueError(f"unsupported permissions action: {action}") from exc
        return handler(params, ctx)

    def _list_grants(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        grants = [self._serialize_grant(item) for item in self.db.list_grants(active_only=True)]
        if not grants:
            return ToolResult(ok=True, summary="No active grants.", data={"grants": [], "allow_tool_followup": False})
        return ToolResult(
            ok=True,
            summary=f"Found {len(grants)} active grant(s).",
            data={"grants": grants, "allow_tool_followup": False},
        )

    def _grant_access(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root_path = str(params.get("root_path") or params.get("path") or "").strip()
        raw_capabilities = params.get("capabilities", [])
        if isinstance(raw_capabilities, str):
            capabilities = [item.strip() for item in raw_capabilities.split(",") if item.strip()]
        elif isinstance(raw_capabilities, list):
            capabilities = [str(item).strip() for item in raw_capabilities if str(item).strip()]
        else:
            capabilities = []
        if not root_path or not capabilities:
            return ToolResult(ok=False, summary="Granting access requires a root_path and capabilities.", data={})
        grant = self.db.upsert_grant(root_path, capabilities, ctx.chat_id)
        return ToolResult(
            ok=True,
            summary=f"Granted {', '.join(grant.capabilities)} access for {grant.root_path}.",
            data={"grant": self._serialize_grant(grant), "allow_tool_followup": False},
        )

    def _revoke_grant(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        token = params.get("grant_id")
        grant_id = int(str(token).strip()) if str(token).strip().isdigit() else 0
        if grant_id <= 0:
            return ToolResult(ok=False, summary="Revoking access requires a numeric grant_id.", data={})
        revoked = self.db.revoke_grant(grant_id)
        if not revoked:
            return ToolResult(ok=False, summary=f"Grant {grant_id} was not found.", data={"grant_id": grant_id})
        return ToolResult(
            ok=True,
            summary=f"Grant {grant_id} revoked.",
            data={"grant_id": grant_id, "revoked": True, "allow_tool_followup": False},
        )

    def _list_pending_requests(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        requests = [self._serialize_request(item) for item in self.db.list_approval_requests(status="pending")]
        if not requests:
            return ToolResult(
                ok=True,
                summary="No pending approval requests.",
                data={"requests": [], "allow_tool_followup": False},
            )
        return ToolResult(
            ok=True,
            summary=f"Found {len(requests)} pending approval request(s).",
            data={"requests": requests, "allow_tool_followup": False},
        )

    def _serialize_grant(self, grant: Any) -> dict[str, Any]:
        return {
            "id": grant.id,
            "root_path": grant.root_path,
            "capabilities": list(grant.capabilities),
            "granted_by_chat_id": grant.granted_by_chat_id,
            "created_at": grant.created_at,
        }

    def _serialize_request(self, request: Any) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "kind": request.kind,
            "chat_id": request.chat_id,
            "root_path": request.root_path,
            "capabilities": list(request.capabilities),
            "objective": request.objective,
            "status": request.status,
            "created_at": request.created_at,
        }
