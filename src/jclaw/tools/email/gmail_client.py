from __future__ import annotations

from email.message import EmailMessage
from typing import Any
import base64

from jclaw.tools.email.auth import GmailOAuthManager


def _header_map(payload: dict[str, Any]) -> dict[str, str]:
    headers = payload.get("headers", []) or []
    return {str(item.get("name", "")).lower(): str(item.get("value", "")) for item in headers}


def _decode_body(data: str) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    return decoded.decode("utf-8", errors="replace")


def _extract_bodies(payload: dict[str, Any]) -> tuple[str, str]:
    mime_type = str(payload.get("mimeType", ""))
    body = payload.get("body", {}) or {}
    parts = payload.get("parts", []) or []
    plain = ""
    html = ""
    if mime_type == "text/plain":
        plain = _decode_body(str(body.get("data", "")))
    elif mime_type == "text/html":
        html = _decode_body(str(body.get("data", "")))
    for part in parts:
        child_plain, child_html = _extract_bodies(part)
        if child_plain and not plain:
            plain = child_plain
        if child_html and not html:
            html = child_html
    return plain, html


def normalize_gmail_message(message: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("payload", {}) or {}
    headers = _header_map(payload)
    plain, html = _extract_bodies(payload)
    labels = [str(item) for item in message.get("labelIds", [])]
    return {
        "id": str(message.get("id", "")),
        "thread_id": str(message.get("threadId", "")),
        "subject": headers.get("subject", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "date": headers.get("date", ""),
        "snippet": str(message.get("snippet", "")),
        "labels": labels,
        "unread": "UNREAD" in labels,
        "text_body": plain,
        "html_body": html,
        "message_id_header": headers.get("message-id", ""),
        "references": headers.get("references", ""),
        "in_reply_to": headers.get("in-reply-to", ""),
    }


class GmailClient:
    def __init__(self, auth: GmailOAuthManager) -> None:
        self.auth = auth

    def list_unread(self, alias: str, *, max_results: int = 10) -> list[dict[str, Any]]:
        return self.search_messages(alias, query="is:unread", max_results=max_results)

    def search_messages(self, alias: str, *, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        service = self._service(alias, ("https://www.googleapis.com/auth/gmail.readonly",))
        response = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        messages = response.get("messages", []) or []
        items: list[dict[str, Any]] = []
        for item in messages:
            message = (
                service.users()
                .messages()
                .get(userId="me", id=str(item.get("id", "")), format="full")
                .execute()
            )
            items.append(normalize_gmail_message(message))
        return items

    def get_message(self, alias: str, *, message_id: str) -> dict[str, Any]:
        service = self._service(alias, ("https://www.googleapis.com/auth/gmail.readonly",))
        message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        return normalize_gmail_message(message)

    def get_thread(self, alias: str, *, thread_id: str) -> dict[str, Any]:
        service = self._service(alias, ("https://www.googleapis.com/auth/gmail.readonly",))
        thread = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
        messages = [normalize_gmail_message(item) for item in thread.get("messages", []) or []]
        return {"thread_id": str(thread.get("id", thread_id)), "messages": messages}

    def draft_reply(self, alias: str, *, message: dict[str, Any], body_text: str) -> dict[str, Any]:
        service = self._service(alias, ("https://www.googleapis.com/auth/gmail.compose",))
        email = EmailMessage()
        email["To"] = message.get("from", "")
        subject = str(message.get("subject", "")).strip()
        email["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        if message.get("message_id_header"):
            email["In-Reply-To"] = str(message["message_id_header"])
        references = str(message.get("references", "")).strip()
        if references and message.get("message_id_header"):
            email["References"] = f"{references} {message['message_id_header']}".strip()
        elif message.get("message_id_header"):
            email["References"] = str(message["message_id_header"])
        email.set_content(body_text)
        raw = base64.urlsafe_b64encode(email.as_bytes()).decode("ascii")
        payload = {
            "message": {
                "threadId": message["thread_id"],
                "raw": raw,
            }
        }
        draft = service.users().drafts().create(userId="me", body=payload).execute()
        return {
            "draft_id": str(draft.get("id", "")),
            "message_id": str((draft.get("message") or {}).get("id", "")),
            "thread_id": message["thread_id"],
            "subject": email["Subject"],
            "to": email["To"],
            "body_preview": body_text[:500],
        }

    def _service(self, alias: str, scopes: tuple[str, ...]) -> Any:
        creds = self.auth.load_credentials(alias, scopes)
        return self.auth._build_service(creds)  # noqa: SLF001
