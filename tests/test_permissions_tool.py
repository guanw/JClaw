from __future__ import annotations

from jclaw.core.db import Database
from jclaw.tools.base import ToolContext
from jclaw.tools.permissions.tool import PermissionsTool


def test_permissions_tool_lists_grants_grants_access_revokes_and_lists_pending_requests(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = PermissionsTool(db)

    listed_empty = tool.invoke("list_grants", {}, ToolContext(chat_id="chat-1"))
    assert listed_empty.ok is True
    assert listed_empty.data["grants"] == []

    granted = tool.invoke(
        "grant_access",
        {"root_path": "/Users/Jude", "capabilities": ["read", "write"]},
        ToolContext(chat_id="chat-1"),
    )
    assert granted.ok is True
    assert granted.data["grant"]["root_path"] == "/Users/Jude"
    assert granted.data["grant"]["capabilities"] == ["read", "write"]

    listed = tool.invoke("list_grants", {}, ToolContext(chat_id="chat-1"))
    assert listed.ok is True
    assert listed.data["grants"] == [granted.data["grant"]]

    created_request = db.create_approval_request(
        kind="grant",
        chat_id="chat-1",
        root_path="/Users/Jude/Documents",
        capabilities=("read",),
        objective="Inspect documents",
        payload={"continuation": {"tool": "workspace", "action": "inspect_root"}},
    )
    pending = tool.invoke("list_pending_requests", {}, ToolContext(chat_id="chat-1"))
    assert pending.ok is True
    assert pending.data["requests"] == [
        {
            "request_id": created_request.request_id,
            "kind": "grant",
            "chat_id": "chat-1",
            "root_path": "/Users/Jude/Documents",
            "capabilities": ["read"],
            "objective": "Inspect documents",
            "status": "pending",
            "created_at": created_request.created_at,
        }
    ]

    revoked = tool.invoke(
        "revoke_grant",
        {"grant_id": granted.data["grant"]["id"]},
        ToolContext(chat_id="chat-1"),
    )
    assert revoked.ok is True
    assert revoked.data["revoked"] is True

    listed_after_revoke = tool.invoke("list_grants", {}, ToolContext(chat_id="chat-1"))
    assert listed_after_revoke.ok is True
    assert listed_after_revoke.data["grants"] == []
    db.close()
