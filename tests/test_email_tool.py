from __future__ import annotations

from pathlib import Path

from jclaw.core.db import Database
from jclaw.core.config import _detect_email_oauth_client
from jclaw.tools.base import Observation, RuntimeState, ToolContext
from jclaw.tools.email.auth import ConnectedEmailAccount
from jclaw.tools.email.tool import EmailTool


class FakeGmailClient:
    def __init__(self) -> None:
        self.messages = [
            {
                "id": "msg-1",
                "thread_id": "thread-1",
                "subject": "Taxes",
                "from": "alice@example.com",
                "to": "me@example.com",
                "cc": "",
                "date": "Mon, 20 Apr 2026 10:00:00 -0400",
                "snippet": "Can you review this?",
                "labels": ["UNREAD", "INBOX"],
                "unread": True,
                "text_body": "Can you review this?",
                "html_body": "",
                "message_id_header": "<msg-1@example.com>",
                "references": "",
                "in_reply_to": "",
            }
        ]
        self.last_draft: dict[str, str] | None = None

    def list_unread(self, alias: str, *, max_results: int = 10) -> list[dict]:
        return self.messages[:max_results]

    def search_messages(self, alias: str, *, query: str, max_results: int = 10) -> list[dict]:
        return [item for item in self.messages if query.lower() in item["subject"].lower()][:max_results]

    def get_message(self, alias: str, *, message_id: str) -> dict:
        return self.messages[0]

    def get_thread(self, alias: str, *, thread_id: str) -> dict:
        return {"thread_id": thread_id, "messages": list(self.messages)}

    def draft_reply(self, alias: str, *, message: dict, body_text: str) -> dict:
        self.last_draft = {"alias": alias, "message_id": message["id"], "body": body_text}
        return {
            "draft_id": "draft-1",
            "message_id": "msg-draft-1",
            "thread_id": message["thread_id"],
            "subject": f"Re: {message['subject']}",
            "to": message["from"],
            "body_preview": body_text,
        }


def test_email_tool_connect_list_search_get_and_draft(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    fake_client = FakeGmailClient()
    tool = EmailTool(
        db,
        oauth_client_path=Path("/tmp/client.json"),
        token_dir=tmp_path / "tokens",
        connect_account=lambda alias, scopes: ConnectedEmailAccount(
            alias=alias,
            provider="gmail",
            email_address="me@example.com",
            scopes=scopes,
            metadata={"history_id": "123"},
        ),
        get_client=lambda alias: fake_client,
    )

    connected = tool.invoke("connect_account", {"alias": "gmail"}, ToolContext(chat_id="chat-1"))
    assert connected.ok is True
    assert connected.data["account"]["email_address"] == "me@example.com"

    listed = tool.invoke("list_accounts", {}, ToolContext(chat_id="chat-1"))
    assert listed.ok is True
    assert listed.data["accounts"][0]["alias"] == "gmail"

    unread = tool.invoke("list_unread", {"alias": "gmail"}, ToolContext(chat_id="chat-1"))
    assert unread.ok is True
    assert unread.data["messages"][0]["subject"] == "Taxes"

    searched = tool.invoke("search_messages", {"alias": "gmail", "query": "tax"}, ToolContext(chat_id="chat-1"))
    assert searched.ok is True
    assert searched.data["messages"][0]["id"] == "msg-1"
    assert searched.data["artifacts"]["email_result_set:latest"]["alias"] == "gmail"
    assert searched.data["allow_tool_followup"] is True

    selected = tool.invoke("select_message", {"alias": "gmail", "selection": "latest"}, ToolContext(chat_id="chat-1"))
    assert selected.ok is True
    assert selected.data["message"]["id"] == "msg-1"
    assert selected.data["artifacts"]["message_ref:selected"]["message_id"] == "msg-1"
    assert selected.data["allow_tool_followup"] is True

    message = tool.invoke("get_message", {"alias": "gmail", "message_id": "msg-1"}, ToolContext(chat_id="chat-1"))
    assert message.ok is True
    assert message.data["message"]["from"] == "alice@example.com"
    assert message.data["artifacts"]["message_ref:latest"]["thread_id"] == "thread-1"
    assert message.data["allow_tool_followup"] is True

    thread = tool.invoke("get_thread", {"alias": "gmail", "thread_id": "thread-1"}, ToolContext(chat_id="chat-1"))
    assert thread.ok is True
    assert thread.data["thread"]["thread_id"] == "thread-1"
    assert thread.data["artifacts"]["thread_ref:latest"]["thread_id"] == "thread-1"
    assert thread.data["allow_tool_followup"] is True

    draft = tool.invoke(
        "draft_reply",
        {"alias": "gmail", "message_id": "msg-1", "body": "Looks good to me."},
        ToolContext(chat_id="chat-1"),
    )
    assert draft.ok is True
    assert draft.data["draft"]["draft_id"] == "draft-1"
    assert draft.data["artifacts"]["email_draft:latest"]["draft_id"] == "draft-1"
    assert fake_client.last_draft == {"alias": "gmail", "message_id": "msg-1", "body": "Looks good to me."}

    searched = tool.invoke("search_messages", {"alias": "gmail", "query": "tax"}, ToolContext(chat_id="chat-1"))
    assert searched.ok is True
    draft_from_index = tool.invoke(
        "draft_reply",
        {"alias": "gmail", "message_id": "1", "body": "Following up tomorrow."},
        ToolContext(chat_id="chat-1"),
    )
    assert draft_from_index.ok is True
    assert fake_client.last_draft == {"alias": "gmail", "message_id": "msg-1", "body": "Following up tomorrow."}
    db.close()


def test_email_tool_formats_message_lists_as_readable_blocks(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    fake_client = FakeGmailClient()
    fake_client.messages = [
        {
            "id": "msg-1",
            "thread_id": "thread-1",
            "subject": "Re: Insurance quote",
            "from": "Jenell Lessley <jenell@pinnaclemutual.com>",
            "to": "me@example.com",
            "cc": "",
            "date": "Mon, 20 Apr 2026 16:52:00 -0400",
            "snippet": "You would need 250/500/100 limits&#39; On Mon, Apr 20, 2026 at 4:52 PM Jenell wrote:",
            "labels": ["UNREAD", "INBOX"],
            "unread": True,
            "text_body": "",
            "html_body": "",
            "message_id_header": "<msg-1@example.com>",
            "references": "",
            "in_reply_to": "",
        }
    ]
    tool = EmailTool(
        db,
        oauth_client_path=Path("/tmp/client.json"),
        token_dir=tmp_path / "tokens",
        connect_account=lambda alias, scopes: ConnectedEmailAccount(
            alias=alias,
            provider="gmail",
            email_address="me@example.com",
            scopes=scopes,
            metadata={"history_id": "123"},
        ),
        get_client=lambda alias: fake_client,
    )
    tool.invoke("connect_account", {"alias": "gmail"}, ToolContext(chat_id="chat-1"))

    searched = tool.invoke("search_messages", {"alias": "gmail", "query": "insurance"}, ToolContext(chat_id="chat-1"))
    rendered = tool.format_result("search_messages", searched)

    assert "1. Jenell Lessley <jenell@pinnaclemutual.com>" in rendered
    assert "Subject: Re: Insurance quote" in rendered
    assert "Unread: Yes" in rendered
    assert "Preview: You would need 250/500/100 limits'" in rendered
    assert "wrote:" not in rendered
    assert "&#39;" not in rendered
    db.close()


def test_detect_email_oauth_client_requires_exactly_one_match(tmp_path) -> None:
    with (tmp_path / "client_secret_one.json").open("w", encoding="utf-8") as handle:
        handle.write("{}")
    assert _detect_email_oauth_client(tmp_path) == tmp_path / "client_secret_one.json"

    with (tmp_path / "client_secret_two.json").open("w", encoding="utf-8") as handle:
        handle.write("{}")
    try:
        _detect_email_oauth_client(tmp_path)
    except RuntimeError as exc:
        assert "exactly one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError when multiple OAuth client files exist")


def test_email_materialize_params_normalizes_alias_and_placeholder_ids(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    db.upsert_email_account(
        alias="gmail",
        provider="gmail",
        email_address="guanw0826@gmail.com",
        scopes=("https://www.googleapis.com/auth/gmail.readonly",),
        status="connected",
        metadata={},
    )
    tool = EmailTool(
        db,
        oauth_client_path=Path("/tmp/client.json"),
        token_dir=tmp_path / "tokens",
        get_client=lambda alias: FakeGmailClient(),
    )
    runtime = RuntimeState(request="draft a reply")
    runtime.append(
        Observation(
            ok=True,
            summary="Selected a message.",
            artifacts={
                "message_ref:selected": {
                    "alias": "gmail",
                    "message_id": "msg-1",
                    "thread_id": "thread-1",
                }
            },
            artifact_types=["message_ref"],
        )
    )

    draft_params = tool.materialize_params(
        "draft_reply",
        {
            "alias": "guanw0826@gmail.com",
            "message_id": "selected",
            "body": "hello",
        },
        runtime,
    )
    assert draft_params == {
        "alias": "gmail",
        "message_id": "msg-1",
        "body": "hello",
        "thread_id": "thread-1",
    }

    thread_params = tool.materialize_params(
        "get_thread",
        {
            "alias": "guanw0826@gmail.com",
            "thread_id": "selected",
        },
        runtime,
    )
    assert thread_params == {
        "alias": "gmail",
        "thread_id": "thread-1",
    }
    db.close()
