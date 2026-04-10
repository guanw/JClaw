from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MessageRecord:
    role: str
    content: str
    created_at: str


@dataclass(slots=True)
class MemoryRecord:
    key: str
    value: str
    updated_at: str


@dataclass(slots=True)
class CronJobRecord:
    id: int
    chat_id: str
    schedule: str
    prompt: str
    next_run_at: str
    enabled: bool


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self.initialize()

    def initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

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

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(scope, key)
            );

            CREATE TABLE IF NOT EXISTS cron_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                schedule TEXT NOT NULL,
                prompt TEXT NOT NULL,
                next_run_at TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def get_kv(self, key: str, default: str = "") -> str:
        row = self._connection.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_kv(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO kv(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self._connection.commit()

    def get_telegram_offset(self) -> int:
        raw = self.get_kv("telegram.offset", "0")
        return int(raw or "0")

    def set_telegram_offset(self, offset: int) -> None:
        self.set_kv("telegram.offset", str(offset))

    def store_message(
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

    def recent_messages(self, chat_id: str, limit: int) -> list[MessageRecord]:
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

    def list_memories(self, scope: str, limit: int = 20) -> list[MemoryRecord]:
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

    def search_memories(self, scope: str, query: str, limit: int) -> list[MemoryRecord]:
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

    def add_cron_job(self, chat_id: str, schedule: str, prompt: str, next_run_at: str) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO cron_jobs(chat_id, schedule, prompt, next_run_at, enabled, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (chat_id, schedule, prompt, next_run_at, utc_now()),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def list_cron_jobs(self, chat_id: str) -> list[CronJobRecord]:
        rows = self._connection.execute(
            """
            SELECT id, chat_id, schedule, prompt, next_run_at, enabled
            FROM cron_jobs
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        ).fetchall()
        return [
            CronJobRecord(
                id=int(row["id"]),
                chat_id=str(row["chat_id"]),
                schedule=str(row["schedule"]),
                prompt=str(row["prompt"]),
                next_run_at=str(row["next_run_at"]),
                enabled=bool(row["enabled"]),
            )
            for row in rows
        ]

    def remove_cron_job(self, chat_id: str, job_id: int) -> int:
        cursor = self._connection.execute(
            "DELETE FROM cron_jobs WHERE chat_id = ? AND id = ?",
            (chat_id, job_id),
        )
        self._connection.commit()
        return cursor.rowcount

    def due_cron_jobs(self, now: str) -> list[CronJobRecord]:
        rows = self._connection.execute(
            """
            SELECT id, chat_id, schedule, prompt, next_run_at, enabled
            FROM cron_jobs
            WHERE enabled = 1 AND next_run_at <= ?
            ORDER BY next_run_at ASC
            """,
            (now,),
        ).fetchall()
        return [
            CronJobRecord(
                id=int(row["id"]),
                chat_id=str(row["chat_id"]),
                schedule=str(row["schedule"]),
                prompt=str(row["prompt"]),
                next_run_at=str(row["next_run_at"]),
                enabled=bool(row["enabled"]),
            )
            for row in rows
        ]

    def update_cron_next_run(self, job_id: int, next_run_at: str) -> None:
        self._connection.execute(
            "UPDATE cron_jobs SET next_run_at = ? WHERE id = ?",
            (next_run_at, job_id),
        )
        self._connection.commit()

