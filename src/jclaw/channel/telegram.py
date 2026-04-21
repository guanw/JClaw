from __future__ import annotations

import logging

import httpx

from jclaw.channel.base import IncomingMessage
from jclaw.core.config import TelegramConfig


LOGGER = logging.getLogger(__name__)

class TelegramBotChannel:
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
        chunks = self._chunk_text(text)
        first_message_id: str | None = None
        for index, chunk in enumerate(chunks):
            payload: dict[str, object] = {"chat_id": chat_id, "text": chunk}
            if index == 0 and reply_to_message_id:
                payload["reply_to_message_id"] = int(reply_to_message_id)
            result = self._request("sendMessage", payload)
            if index == 0 and isinstance(result, dict) and "message_id" in result:
                first_message_id = str(result["message_id"])
        return first_message_id

    def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
        chunks = self._chunk_text(text)
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "message_id": int(message_id),
            "text": chunks[0],
        }
        self._request("editMessageText", payload)
        for chunk in chunks[1:]:
            self._request("sendMessage", {"chat_id": chat_id, "text": chunk})

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
