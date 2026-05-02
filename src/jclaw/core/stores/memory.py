from __future__ import annotations

import re
import sqlite3

from jclaw.core.records import MemoryRecord
from jclaw.core.time import utc_now


class MemoryStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(scope, key)
            );
            """
        )

    def remember(self, scope: str, key: str, value: str) -> None:
        now = utc_now()
        self._connection.execute(
            """
            INSERT INTO memories(scope, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scope, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (scope, key, value, now, now),
        )
        self._connection.commit()

    def forget(self, scope: str, key: str) -> int:
        cursor = self._connection.execute(
            "DELETE FROM memories WHERE scope = ? AND key = ?",
            (scope, key),
        )
        self._connection.commit()
        return cursor.rowcount

    def list(self, scope: str, limit: int = 20) -> list[MemoryRecord]:
        rows = self._connection.execute(
            """
            SELECT key, value, updated_at
            FROM memories
            WHERE scope = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (scope, limit),
        ).fetchall()
        return [MemoryRecord(row["key"], row["value"], row["updated_at"]) for row in rows]

    def search(self, scope: str, query: str, limit: int) -> list[MemoryRecord]:
        rows = self._connection.execute(
            """
            SELECT key, value, updated_at
            FROM memories
            WHERE scope IN (?, 'global')
            """,
            (scope,),
        ).fetchall()
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[int, str, str, str]] = []
        for row in rows:
            haystack_terms = set(re.findall(r"[a-z0-9]+", f"{row['key']} {row['value']}".lower()))
            score = len(query_terms & haystack_terms)
            if score == 0 and query_terms:
                continue
            scored.append((score, str(row["updated_at"]), str(row["key"]), str(row["value"])))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [MemoryRecord(key=item[2], value=item[3], updated_at=item[1]) for item in scored[:limit]]
