from __future__ import annotations

import sqlite3

from jclaw.core.records import CronJobRecord
from jclaw.core.time import utc_now


class CronStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
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

    def add_job(self, chat_id: str, schedule: str, prompt: str, next_run_at: str) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO cron_jobs(chat_id, schedule, prompt, next_run_at, enabled, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (chat_id, schedule, prompt, next_run_at, utc_now()),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def list_jobs(self, chat_id: str) -> list[CronJobRecord]:
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

    def get_job(self, chat_id: str, job_id: int) -> CronJobRecord | None:
        row = self._connection.execute(
            """
            SELECT id, chat_id, schedule, prompt, next_run_at, enabled
            FROM cron_jobs
            WHERE chat_id = ? AND id = ?
            """,
            (chat_id, job_id),
        ).fetchone()
        if row is None:
            return None
        return CronJobRecord(
            id=int(row["id"]),
            chat_id=str(row["chat_id"]),
            schedule=str(row["schedule"]),
            prompt=str(row["prompt"]),
            next_run_at=str(row["next_run_at"]),
            enabled=bool(row["enabled"]),
        )

    def remove_job(self, chat_id: str, job_id: int) -> int:
        cursor = self._connection.execute(
            "DELETE FROM cron_jobs WHERE chat_id = ? AND id = ?",
            (chat_id, job_id),
        )
        self._connection.commit()
        return cursor.rowcount

    def update_job(
        self,
        chat_id: str,
        job_id: int,
        *,
        schedule: str | None = None,
        prompt: str | None = None,
        next_run_at: str | None = None,
        enabled: bool | None = None,
    ) -> int:
        updates: list[str] = []
        params: list[object] = []
        if schedule is not None:
            updates.append("schedule = ?")
            params.append(schedule)
        if prompt is not None:
            updates.append("prompt = ?")
            params.append(prompt)
        if next_run_at is not None:
            updates.append("next_run_at = ?")
            params.append(next_run_at)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if enabled else 0)
        if not updates:
            return 0
        params.extend([chat_id, job_id])
        cursor = self._connection.execute(
            f"UPDATE cron_jobs SET {', '.join(updates)} WHERE chat_id = ? AND id = ?",  # noqa: S608
            tuple(params),
        )
        self._connection.commit()
        return cursor.rowcount

    def due_jobs(self, now: str) -> list[CronJobRecord]:
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

    def update_next_run(self, job_id: int, next_run_at: str) -> None:
        self._connection.execute(
            "UPDATE cron_jobs SET next_run_at = ? WHERE id = ?",
            (next_run_at, job_id),
        )
        self._connection.commit()
