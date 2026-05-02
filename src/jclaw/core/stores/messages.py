from __future__ import annotations

import sqlite3

from jclaw.core.records import MessageRecord
from jclaw.core.time import utc_now


class MessageStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                external_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_external
                ON messages(channel, external_id)
                WHERE external_id IS NOT NULL;
            """
        )

    def store(
        self,
        chat_id: str,
        role: str,
        content: str,
        *,
        channel: str = "telegram",
        external_id: str | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO messages(channel, chat_id, role, content, external_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (channel, chat_id, role, content, external_id, utc_now()),
        )
        self._connection.commit()

    def recent(self, chat_id: str, limit: int) -> list[MessageRecord]:
        rows = self._connection.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        ).fetchall()
        items = [MessageRecord(row["role"], row["content"], row["created_at"]) for row in rows]
        items.reverse()
        return items
