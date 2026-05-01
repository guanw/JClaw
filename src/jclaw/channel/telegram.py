from __future__ import annotations

import html
import logging
import re

import httpx

from jclaw.channel.base import IncomingMessage
from jclaw.core.config import TelegramConfig


LOGGER = logging.getLogger(__name__)

class TelegramBotChannel:
    _FENCED_CODE_PATTERN = re.compile(r"```(?:[\w.+-]+)?\n?(.*?)```", re.DOTALL)

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=config.poll_timeout_seconds + 10)

    def close(self) -> None:
        self._client.close()

    def validate_token(self) -> dict[str, object]:
        result = self._request("getMe", {})
        return dict(result)

    def poll_updates(self, offset: int) -> list[IncomingMessage]:
        payload = {
            "offset": offset,
            "timeout": self.config.poll_timeout_seconds,
            "allowed_updates": ["message"],
        }
        result = self._request("getUpdates", payload)
        items: list[IncomingMessage] = []
        for update in result:
            message = update.get("message", {})
            if not isinstance(message, dict):
                continue
            text = message.get("text")
            chat = message.get("chat", {})
            sender = message.get("from", {})
            if not text or not chat:
                continue
            if sender.get("is_bot"):
                continue
            items.append(
                IncomingMessage(
                    update_id=int(update["update_id"]),
                    chat_id=str(chat["id"]),
                    message_id=str(message["message_id"]),
                    sender_id=str(sender.get("id", "")),
                    sender_name=str(sender.get("first_name") or sender.get("username") or "telegram-user"),
                    text=str(text),
                )
            )
        return items

    def send_message(self, chat_id: str, text: str, *, reply_to_message_id: str | None = None) -> str | None:
        chunks, parse_mode = self._message_chunks(text)
        first_message_id: str | None = None
        for index, chunk in enumerate(chunks):
            payload: dict[str, object] = self._message_payload(chat_id, chunk, parse_mode=parse_mode)
            if index == 0 and reply_to_message_id:
                payload["reply_to_message_id"] = int(reply_to_message_id)
            result = self._request("sendMessage", payload)
            if index == 0 and isinstance(result, dict) and "message_id" in result:
                first_message_id = str(result["message_id"])
        return first_message_id

    def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
        chunks, parse_mode = self._message_chunks(text)
        payload: dict[str, object] = self._message_payload(chat_id, chunks[0], parse_mode=parse_mode)
        payload["message_id"] = int(message_id)
        self._request("editMessageText", payload)
        for chunk in chunks[1:]:
            self._request("sendMessage", self._message_payload(chat_id, chunk, parse_mode=parse_mode))

    def _request(self, method: str, payload: dict[str, object]) -> object:
        if not self.config.bot_token:
            raise RuntimeError("telegram.bot_token is missing")
        response = self._client.post(f"{self.config.base_url}{self.config.bot_token}/{method}", json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"telegram API error for {method}: {data}")
        return data.get("result")

    def _chunk_text(self, text: str, size: int = 3800) -> list[str]:
        if len(text) <= size:
            return [text]
        return [text[index : index + size] for index in range(0, len(text), size)]

    def _message_payload(self, chat_id: str, text: str, *, parse_mode: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        return payload

    def _message_chunks(self, text: str, size: int = 3800) -> tuple[list[str], str | None]:
        rendered, parse_mode = self._render_message_text(text)
        if parse_mode != "HTML":
            return self._chunk_text(rendered, size), parse_mode
        return self._chunk_html_text(rendered, size), parse_mode

    def _render_message_text(self, text: str) -> tuple[str, str | None]:
        if "```" in text:
            return self._render_fenced_code_html(text), "HTML"
        if "<pre>" in text or "<code>" in text:
            return text, "HTML"
        return text, None

    def _render_fenced_code_html(self, text: str) -> str:
        parts: list[str] = []
        index = 0
        for match in self._FENCED_CODE_PATTERN.finditer(text):
            parts.append(html.escape(text[index : match.start()]))
            code = match.group(1).strip("\n")
            parts.append(f"<pre>{html.escape(code)}</pre>")
            index = match.end()
        parts.append(html.escape(text[index:]))
        return "".join(parts)

    def _chunk_html_text(self, text: str, size: int) -> list[str]:
        tokens = [token for token in re.split(r"(<pre>.*?</pre>)", text, flags=re.DOTALL) if token]
        chunks: list[str] = []
        current = ""
        for token in tokens:
            for piece in self._split_html_token(token, size):
                if current and len(current) + len(piece) > size:
                    chunks.append(current)
                    current = ""
                if len(piece) > size:
                    chunks.extend(self._chunk_text(piece, size))
                    continue
                current += piece
        if current:
            chunks.append(current)
        return chunks or [text[:size]]

    def _split_html_token(self, token: str, size: int) -> list[str]:
        if not token.startswith("<pre>") or len(token) <= size:
            return [token]
        content = token[len("<pre>") : -len("</pre>")]
        available = max(1, size - len("<pre></pre>"))
        return [f"<pre>{content[index : index + available]}</pre>" for index in range(0, len(content), available)]
