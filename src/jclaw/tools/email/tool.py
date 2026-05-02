from __future__ import annotations

import html
import json
from email.utils import getaddresses
from pathlib import Path
import re
from typing import Any, Callable

from jclaw.core.db import Database, EmailAccountRecord
from jclaw.tools.base import ActionSpec, RuntimeState, ToolContext, ToolResult, build_tool_description
from jclaw.tools.email.auth import ConnectedEmailAccount, GmailOAuthManager
from jclaw.tools.email.gmail_client import GmailClient


GMAIL_READ_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"


class EmailTool:
    name = "email"
    MAX_CACHED_MESSAGES = 20

    def __init__(
        self,
        db: Database,
        *,
        oauth_client_path: Path | None,
        token_dir: Path,
        default_account_alias: str = "gmail",
        connect_account: Callable[[str, tuple[str, ...]], ConnectedEmailAccount] | None = None,
        get_client: Callable[[str], GmailClient] | None = None,
    ) -> None:
        self.db = db
        self.default_account_alias = default_account_alias
        self.auth = GmailOAuthManager(oauth_client_path=oauth_client_path, token_dir=token_dir)
        self._connect_account = connect_account or self.auth.connect_account
        self._get_client = get_client or (lambda alias: GmailClient(self.auth))

    def describe(self) -> dict[str, Any]:
        specs = self._action_specs()
        return build_tool_description(
            name=self.name,
            description="Connect a Gmail account, search messages, read threads, and create draft replies.",
            actions=specs,
        )

    def format_result(self, action: str, result: ToolResult) -> str:
        data = result.data
        if action == "list_accounts":
            items = data.get("accounts", [])
            if not items:
                return "No email accounts connected."
            lines = ["Connected email accounts:"]
            for item in items:
                lines.append(f"- {item['alias']}: {item['email_address']} [{', '.join(item['scopes'])}]")
            return "\n".join(lines)
        if action in {"list_unread", "search_messages"}:
            items = data.get("messages", [])
            if not items:
                return result.summary
            lines = [result.summary]
            for index, item in enumerate(items[:10], start=1):
                subject = self._display_text(item.get("subject")) or "(no subject)"
                sender = self._display_text(item.get("from")) or "(unknown sender)"
                date = self._display_text(item.get("date")) or "(unknown date)"
                unread = "Yes" if bool(item.get("unread")) else "No"
                preview = self._message_preview(item)
                lines.append("")
                lines.append(f"{index}. {sender}")
                lines.append(f"Subject: {subject}")
                lines.append(f"Date: {date}")
                lines.append(f"Unread: {unread}")
                lines.append(f"Preview: {preview}")
            return "\n".join(lines)
        if action == "select_message" and data.get("message"):
            message = data["message"]
            return (
                f"Selected: {self._display_text(message['subject'])}\n"
                f"From: {self._display_text(message['from'])}\n"
                f"Date: {self._display_text(message['date'])}\n"
                f"Preview: {self._message_preview(message)}"
            )
        if action == "get_message" and data.get("message"):
            message = data["message"]
            return (
                f"Subject: {self._display_text(message['subject'])}\n"
                f"From: {self._display_text(message['from'])}\n"
                f"Date: {self._display_text(message['date'])}\n"
                f"Snippet: {self._display_text(message['snippet'])}\n"
                f"Body:\n{self._display_text(message['text_body'])[:2000]}"
            )
        if action == "get_thread" and data.get("thread"):
            thread = data["thread"]
            lines = [f"Thread {thread['thread_id']} ({len(thread['messages'])} message(s)):"]
            for message in thread["messages"][:10]:
                lines.append(
                    f"- {self._display_text(message['subject'])} | "
                    f"{self._display_text(message['from'])} | "
                    f"{self._message_preview(message)}"
                )
            return "\n".join(lines)
        if action == "draft_reply" and data.get("draft"):
            draft = data["draft"]
            return (
                f"Created Gmail draft {draft['draft_id']}.\n"
                f"To: {self._display_text(draft['to'])}\n"
                f"Subject: {self._display_text(draft['subject'])}\n"
                f"Preview:\n{self._display_text(draft['body_preview'])}"
            )
        return result.summary

    def materialize_params(
        self,
        action: str,
        params: dict[str, Any],
        runtime: RuntimeState,
    ) -> dict[str, Any]:
        materialized = dict(params)
        raw_alias = str(materialized.get("alias", "")).strip()
        alias = self._coerce_runtime_alias(raw_alias)
        if alias:
            materialized["alias"] = alias
        elif raw_alias:
            materialized.pop("alias", None)
        for key in ("message_id", "thread_id", "result_set_id"):
            raw_value = str(materialized.get(key, "")).strip()
            if self._is_placeholder_identifier(raw_value):
                materialized.pop(key, None)
        if action == "select_message" and not str(materialized.get("alias", "")).strip():
            alias = self._alias_from_runtime(runtime)
            if alias:
                materialized["alias"] = alias
        if action == "search_messages" and not str(materialized.get("alias", "")).strip():
            alias = self._alias_from_runtime(runtime)
            if alias:
                materialized["alias"] = alias
        if action == "get_message" and not str(materialized.get("message_id", "")).strip():
            message_id = self._message_id_from_runtime(runtime)
            if message_id:
                materialized["message_id"] = message_id
        if action == "get_thread":
            if not str(materialized.get("thread_id", "")).strip():
                thread_id = self._thread_id_from_runtime(runtime)
                if thread_id:
                    materialized["thread_id"] = thread_id
            if not str(materialized.get("alias", "")).strip():
                alias = self._alias_from_runtime(runtime)
                if alias:
                    materialized["alias"] = alias
        if action == "draft_reply":
            if not str(materialized.get("message_id", "")).strip():
                message_id = self._message_id_from_runtime(runtime)
                if message_id:
                    materialized["message_id"] = message_id
            if not str(materialized.get("thread_id", "")).strip():
                thread_id = self._thread_id_from_runtime(runtime)
                if thread_id:
                    materialized["thread_id"] = thread_id
            if not str(materialized.get("alias", "")).strip():
                alias = self._alias_from_runtime(runtime)
                if alias:
                    materialized["alias"] = alias
        return materialized

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "connect_account": self._connect,
            "list_accounts": self._list_accounts,
            "list_unread": self._list_unread,
            "search_messages": self._search_messages,
            "select_message": self._select_message,
            "get_message": self._get_message,
            "get_thread": self._get_thread,
            "draft_reply": self._draft_reply,
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError(f"unsupported email action: {action}")
        return handler(params, ctx)

    def _connect(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        alias = str(params.get("alias") or self.default_account_alias).strip()
        scopes = self._requested_scopes(params, include_compose=True)
        account = self._connect_account(alias, scopes)
        record = self.db.upsert_email_account(
            alias=account.alias,
            provider=account.provider,
            email_address=account.email_address,
            scopes=account.scopes,
            status="connected",
            metadata=account.metadata,
        )
        return ToolResult(
            ok=True,
            summary=f"Connected Gmail account {record.email_address} as '{record.alias}'.",
            data={"account": self._serialize_account(record), "allow_tool_followup": False},
        )

    def _list_accounts(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        accounts = [self._serialize_account(item) for item in self.db.list_email_accounts()]
        if not accounts:
            return ToolResult(ok=True, summary="No email accounts connected.", data={"accounts": [], "allow_tool_followup": False})
        return ToolResult(
            ok=True,
            summary=f"Listed {len(accounts)} connected email account(s).",
            data={"accounts": accounts, "allow_tool_followup": False},
        )

    def _list_unread(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        alias = self._resolve_alias(params)
        max_results = int(params.get("max_results", 10))
        client = self._get_client(alias)
        messages = self._order_messages_for_display(client.list_unread(alias, max_results=max_results))
        self._cache_recent_messages(ctx.chat_id, alias, messages)
        return ToolResult(
            ok=True,
            summary=f"Found {len(messages)} unread email(s).",
            data={"alias": alias, "messages": messages, "allow_tool_followup": False},
        )

    def _search_messages(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        alias = self._resolve_alias(params)
        query = str(params.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, summary="search_messages requires a query.", data={})
        max_results = int(params.get("max_results", 10))
        client = self._get_client(alias)
        messages = self._order_messages_for_display(client.search_messages(alias, query=query, max_results=max_results))
        self._cache_recent_messages(ctx.chat_id, alias, messages)
        return ToolResult(
            ok=True,
            summary=f"Found {len(messages)} matching email(s) for '{query}'.",
            data={
                "alias": alias,
                "messages": messages,
                "allow_tool_followup": True,
                "artifacts": {
                    "email_result_set:latest": {
                        "alias": alias,
                        "query": query,
                        "messages": messages,
                    }
                },
            },
        )

    def _select_message(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        alias = self._resolve_alias(params)
        messages = self._load_cached_messages(ctx.chat_id, alias)
        if not messages:
            return ToolResult(ok=False, summary="No recent email search results are available to select from.", data={})
        selection = str(params.get("selection", "latest")).strip().lower()
        if selection in {"latest", "most_recent", "most recent", ""}:
            message = messages[0]
        elif selection.isdigit():
            index = int(selection)
            if index <= 0 or index > len(messages):
                return ToolResult(ok=False, summary=f"Selection {index} is out of range for the cached email results.", data={})
            message = messages[index - 1]
        else:
            return ToolResult(ok=False, summary=f"Unsupported email selection '{selection}'.", data={})
        return ToolResult(
            ok=True,
            summary=f"Selected email '{message['subject']}'.",
            data={
                "alias": alias,
                "message": message,
                "allow_tool_followup": True,
                "artifacts": {
                    "message_ref:selected": self._message_ref_artifact(alias, message),
                },
            },
        )

    def _get_message(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        alias = self._resolve_alias(params)
        message_id = str(params.get("message_id", "")).strip()
        if not message_id:
            return ToolResult(ok=False, summary="get_message requires a message_id.", data={})
        client = self._get_client(alias)
        message = client.get_message(alias, message_id=message_id)
        return ToolResult(
            ok=True,
            summary=f"Loaded email '{message['subject']}'.",
            data={
                "alias": alias,
                "message": message,
                "allow_tool_followup": True,
                "artifacts": {
                    "message_ref:latest": self._message_ref_artifact(alias, message),
                },
            },
        )

    def _get_thread(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        alias = self._resolve_alias(params)
        thread_id = str(params.get("thread_id", "")).strip()
        if not thread_id:
            return ToolResult(ok=False, summary="get_thread requires a thread_id.", data={})
        client = self._get_client(alias)
        thread = client.get_thread(alias, thread_id=thread_id)
        return ToolResult(
            ok=True,
            summary=f"Loaded thread {thread['thread_id']}.",
            data={
                "alias": alias,
                "thread": thread,
                "allow_tool_followup": True,
                "artifacts": {
                    "thread_ref:latest": {
                        "alias": alias,
                        "thread_id": str(thread["thread_id"]).strip(),
                        "message_count": len(thread.get("messages", [])),
                    }
                },
            },
        )

    def _draft_reply(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        alias = self._resolve_alias(params)
        body_text = str(params.get("body") or params.get("reply_body") or "").strip()
        if not body_text:
            return ToolResult(ok=False, summary="draft_reply requires a body.", data={})
        client = self._get_client(alias)
        if params.get("message_id"):
            message_ref = str(params["message_id"]).strip()
            cached_message = self._resolve_cached_message_reference(ctx.chat_id, alias, message_ref)
            resolved_message_id = str(cached_message["id"]) if cached_message is not None else message_ref
            message = client.get_message(alias, message_id=resolved_message_id)
        elif params.get("thread_id"):
            thread_ref = str(params["thread_id"]).strip()
            cached_message = self._resolve_cached_message_reference(ctx.chat_id, alias, thread_ref)
            if cached_message is not None:
                thread = client.get_thread(alias, thread_id=str(cached_message["thread_id"]))
            else:
                thread = client.get_thread(alias, thread_id=thread_ref)
            if not thread["messages"]:
                return ToolResult(ok=False, summary="The target thread is empty.", data={})
            message = thread["messages"][-1]
        else:
            return ToolResult(ok=False, summary="draft_reply requires a message_id or thread_id.", data={})
        message = self._prepare_reply_message(alias, message)
        draft = client.draft_reply(alias, message=message, body_text=body_text)
        return ToolResult(
            ok=True,
            summary=f"Created a Gmail draft reply for '{message['subject']}'.",
            data={
                "alias": alias,
                "draft": draft,
                "allow_tool_followup": False,
                "artifacts": {
                    "email_draft:latest": {
                        "alias": alias,
                        "draft_id": str(draft.get("draft_id", "")).strip(),
                        "message_id": str(draft.get("message_id", "")).strip(),
                        "thread_id": str(draft.get("thread_id", "")).strip(),
                        "to": str(draft.get("to", "")).strip(),
                        "subject": str(draft.get("subject", "")).strip(),
                    }
                },
            },
        )

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "connect_account": ActionSpec(
                tool=self.name,
                action="connect_account",
                description="Connect a Gmail account through local OAuth and store its token locally.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                    },
                },
                writes=True,
            ),
            "list_accounts": ActionSpec(
                tool=self.name,
                action="list_accounts",
                description="List connected email accounts.",
                input_schema={"type": "object", "properties": {}},
                reads=True,
            ),
            "list_unread": ActionSpec(
                tool=self.name,
                action="list_unread",
                description="List recent unread emails for a connected account.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                },
                reads=True,
            ),
            "search_messages": ActionSpec(
                tool=self.name,
                action="search_messages",
                description="Search Gmail messages using Gmail query syntax or a topic-oriented query.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                reads=True,
                produces_artifacts=("email_result_set",),
                binding_inputs=("alias",),
            ),
            "select_message": ActionSpec(
                tool=self.name,
                action="select_message",
                description="Select one message from the latest email search results.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "selection": {"type": "string"},
                    },
                },
                reads=True,
                requires_artifacts=("email_result_set",),
                produces_artifacts=("message_ref",),
                binding_inputs=("alias",),
            ),
            "get_message": ActionSpec(
                tool=self.name,
                action="get_message",
                description="Get a single Gmail message in normalized form.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "message_id": {"type": "string"},
                    },
                },
                reads=True,
                requires_artifacts=("message_ref",),
                produces_artifacts=("message_ref",),
                binding_inputs=("alias", "message_id"),
            ),
            "get_thread": ActionSpec(
                tool=self.name,
                action="get_thread",
                description="Get an email thread in normalized form.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "thread_id": {"type": "string"},
                    },
                },
                reads=True,
                requires_artifacts=("message_ref", "thread_ref"),
                produces_artifacts=("thread_ref",),
                binding_inputs=("alias", "thread_id"),
            ),
            "draft_reply": ActionSpec(
                tool=self.name,
                action="draft_reply",
                description="Create a Gmail draft reply to a message or thread without sending it.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alias": {"type": "string"},
                        "message_id": {"type": "string"},
                        "thread_id": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["body"],
                },
                writes=True,
                requires_artifacts=("message_ref", "thread_ref"),
                produces_artifacts=("email_draft",),
                binding_inputs=("alias", "message_id", "thread_id"),
            ),
        }

    def _resolve_alias(self, params: dict[str, Any]) -> str:
        alias = self._coerce_runtime_alias(str(params.get("alias") or "").strip())
        if not alias:
            alias = self._coerce_runtime_alias(str(self.default_account_alias).strip())
        record = self.db.get_email_account(alias)
        if record is None:
            raise RuntimeError(f"Email account '{alias}' is not connected.")
        return alias

    def _normalize_runtime_alias(self, raw_alias: str) -> str:
        alias = str(raw_alias).strip()
        if not alias:
            return ""
        if self.db.get_email_account(alias) is not None:
            return alias
        for account in self.db.list_email_accounts():
            if str(account.email_address).strip().lower() == alias.lower():
                return str(account.alias).strip()
        return alias

    def _coerce_runtime_alias(self, raw_alias: str) -> str:
        alias = self._normalize_runtime_alias(raw_alias)
        if not alias:
            return ""
        if self.db.get_email_account(alias) is not None:
            return alias
        accounts = self.db.list_email_accounts()
        if not accounts:
            return alias
        if len(accounts) == 1:
            return str(accounts[0].alias).strip()
        default_alias = self._normalize_runtime_alias(str(self.default_account_alias).strip())
        if default_alias and self.db.get_email_account(default_alias) is not None:
            return default_alias
        return ""

    def _requested_scopes(self, params: dict[str, Any], *, include_compose: bool) -> tuple[str, ...]:
        scopes = {GMAIL_READ_SCOPE}
        if include_compose:
            scopes.add(GMAIL_COMPOSE_SCOPE)
        extra_scopes = params.get("scopes", [])
        if isinstance(extra_scopes, list):
            scopes.update(str(item).strip() for item in extra_scopes if str(item).strip())
        return tuple(sorted(scopes))

    def _serialize_account(self, account: EmailAccountRecord) -> dict[str, Any]:
        return {
            "alias": account.alias,
            "provider": account.provider,
            "email_address": account.email_address,
            "scopes": list(account.scopes),
            "status": account.status,
            "created_at": account.created_at,
            "updated_at": account.updated_at,
        }

    def _order_messages_for_display(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(reversed(messages))

    def _message_preview(self, message: dict[str, Any]) -> str:
        for field in ("snippet", "text_body", "html_body"):
            raw = self._display_text(message.get(field, ""))
            if not raw:
                continue
            preview = self._trim_quoted_history(raw)
            if preview:
                return preview[:220]
        return "(no preview)"

    def _display_text(self, value: Any) -> str:
        text = html.unescape(str(value or ""))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _trim_quoted_history(self, text: str) -> str:
        if " reacted via Gmail " in text:
            text = text.split(" reacted via Gmail ", 1)[0].strip()
        quoted_markers = (
            " On Mon, ",
            " On Tue, ",
            " On Wed, ",
            " On Thu, ",
            " On Fri, ",
            " On Sat, ",
            " On Sun, ",
            " wrote:",
            "-----Original Message-----",
        )
        trimmed = text
        for marker in quoted_markers:
            index = trimmed.find(marker)
            if index > 0:
                trimmed = trimmed[:index].strip()
        return trimmed

    def _cache_recent_messages(self, chat_id: str, alias: str, messages: list[dict[str, Any]]) -> None:
        payload = {
            "alias": alias,
            "messages": [
                dict(item)
                for item in messages[: self.MAX_CACHED_MESSAGES]
                if str(item.get("id", "")).strip() and str(item.get("thread_id", "")).strip()
            ],
        }
        self.db.set_kv(f"email.recent_results.{chat_id}", json.dumps(payload, ensure_ascii=True))

    def _load_cached_messages(self, chat_id: str, alias: str) -> list[dict[str, Any]]:
        raw = self.db.get_kv(f"email.recent_results.{chat_id}", "")
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if str(payload.get("alias", "")).strip() != alias:
            return []
        messages = payload.get("messages", [])
        return list(messages) if isinstance(messages, list) else []

    def _resolve_cached_message_reference(self, chat_id: str, alias: str, raw_reference: str) -> dict[str, str] | None:
        if not raw_reference.isdigit():
            return None
        messages = self._load_cached_messages(chat_id, alias)
        if not messages:
            return None
        index = int(raw_reference)
        if index <= 0 or index > len(messages):
            return None
        item = messages[index - 1]
        if not isinstance(item, dict):
            return None
        return {
            "id": str(item.get("id", "")).strip(),
            "thread_id": str(item.get("thread_id", "")).strip(),
        }

    def _is_placeholder_identifier(self, raw_value: str) -> bool:
        value = str(raw_value).strip().lower()
        if not value:
            return False
        return value in {
            "selected",
            "latest",
            "first",
            "message_ref",
            "thread_ref",
            "email_result_set",
            "selected_message",
        }

    def _alias_from_runtime(self, runtime: RuntimeState) -> str:
        for key in ("message_ref", "thread_ref", "email_result_set"):
            artifact = runtime.artifacts_by_type.get(key)
            if isinstance(artifact, dict):
                alias = self._normalize_runtime_alias(str(artifact.get("alias", "")).strip())
                if alias:
                    return alias
        return ""

    def _message_id_from_runtime(self, runtime: RuntimeState) -> str:
        artifact = runtime.artifacts_by_type.get("message_ref")
        if isinstance(artifact, dict):
            message_id = str(artifact.get("message_id", "")).strip()
            if message_id:
                return message_id
        return ""

    def _thread_id_from_runtime(self, runtime: RuntimeState) -> str:
        thread_artifact = runtime.artifacts_by_type.get("thread_ref")
        if isinstance(thread_artifact, dict):
            thread_id = str(thread_artifact.get("thread_id", "")).strip()
            if thread_id:
                return thread_id
        message_artifact = runtime.artifacts_by_type.get("message_ref")
        if isinstance(message_artifact, dict):
            thread_id = str(message_artifact.get("thread_id", "")).strip()
            if thread_id:
                return thread_id
        return ""

    def _message_ref_artifact(self, alias: str, message: dict[str, Any]) -> dict[str, str]:
        return {
            "alias": alias,
            "message_id": str(message.get("id", "")).strip(),
            "thread_id": str(message.get("thread_id", "")).strip(),
            "from": str(message.get("from", "")).strip(),
            "to": str(message.get("to", "")).strip(),
            "subject": str(message.get("subject", "")).strip(),
            "date": str(message.get("date", "")).strip(),
        }

    def _prepare_reply_message(self, alias: str, message: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(message)
        reply_to = self._reply_target_for_message(alias, prepared)
        if reply_to:
            prepared["reply_to_address"] = reply_to
        return prepared

    def _reply_target_for_message(self, alias: str, message: dict[str, Any]) -> str:
        own_address = self._account_email_address(alias).lower()
        from_header = str(message.get("from", "")).strip()
        to_header = str(message.get("to", "")).strip()
        from_addresses = self._extract_addresses(from_header)
        to_addresses = self._extract_addresses(to_header)
        if own_address and any(address.lower() == own_address for address in from_addresses):
            for address in to_addresses:
                if address and address.lower() != own_address:
                    return address
            if to_header:
                return to_header
        if from_header:
            return from_header
        if to_header:
            return to_header
        return ""

    def _account_email_address(self, alias: str) -> str:
        record = self.db.get_email_account(alias)
        if record is None:
            return ""
        return str(record.email_address).strip()

    def _extract_addresses(self, raw_header: str) -> list[str]:
        return [
            str(address).strip()
            for _, address in getaddresses([str(raw_header or "")])
            if str(address).strip()
        ]
