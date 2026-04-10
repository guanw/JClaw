from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IncomingMessage:
    update_id: int
    chat_id: str
    message_id: str
    sender_id: str
    sender_name: str
    text: str

