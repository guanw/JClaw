from __future__ import annotations

import json
import sqlite3
import uuid

from jclaw.core.records import ExecutionTraceEventRecord, ExecutionTraceSessionRecord
from jclaw.core.time import utc_now


class TraceStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS execution_trace_sessions (
                trace_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                user_text TEXT NOT NULL,
                status TEXT NOT NULL,
                final_reply TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_execution_trace_sessions_chat
                ON execution_trace_sessions(chat_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS execution_trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(trace_id) REFERENCES execution_trace_sessions(trace_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_execution_trace_events_trace
                ON execution_trace_events(trace_id, event_index ASC);
            """
        )

    def create_session(self, chat_id: str, user_text: str) -> ExecutionTraceSessionRecord:
        now = utc_now()
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        self._connection.execute(
            """
            INSERT INTO execution_trace_sessions(trace_id, chat_id, user_text, status, final_reply, created_at, updated_at)
            VALUES (?, ?, ?, 'running', '', ?, ?)
            """,
            (trace_id, chat_id, user_text, now, now),
        )
        self._connection.commit()
        return ExecutionTraceSessionRecord(
            trace_id=trace_id,
            chat_id=chat_id,
            user_text=user_text,
            status="running",
            created_at=now,
            updated_at=now,
            final_reply="",
        )

    def append_event(
        self,
        trace_id: str,
        *,
        event_type: str,
        summary: str,
        payload: dict[str, object] | None = None,
    ) -> ExecutionTraceEventRecord:
        payload_data = payload or {}
        row = self._connection.execute(
            "SELECT COALESCE(MAX(event_index), 0) + 1 FROM execution_trace_events WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        event_index = int(row[0]) if row is not None else 1
        now = utc_now()
        cursor = self._connection.execute(
            """
            INSERT INTO execution_trace_events(trace_id, event_index, event_type, summary, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trace_id, event_index, event_type, summary, json.dumps(payload_data, ensure_ascii=True), now),
        )
        self._connection.execute(
            "UPDATE execution_trace_sessions SET updated_at = ? WHERE trace_id = ?",
            (now, trace_id),
        )
        self._connection.commit()
        return ExecutionTraceEventRecord(
            id=int(cursor.lastrowid),
            trace_id=trace_id,
            event_index=event_index,
            event_type=event_type,
            summary=summary,
            payload=payload_data,
            created_at=now,
        )

    def finish_session(self, trace_id: str, *, status: str, final_reply: str = "") -> None:
        self._connection.execute(
            """
            UPDATE execution_trace_sessions
            SET status = ?, final_reply = ?, updated_at = ?
            WHERE trace_id = ?
            """,
            (status, final_reply, utc_now(), trace_id),
        )
        self._connection.commit()

    def get_session(self, trace_id: str) -> ExecutionTraceSessionRecord | None:
        row = self._connection.execute(
            """
            SELECT trace_id, chat_id, user_text, status, created_at, updated_at, final_reply
            FROM execution_trace_sessions
            WHERE trace_id = ?
            """,
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        return ExecutionTraceSessionRecord(
            trace_id=str(row["trace_id"]),
            chat_id=str(row["chat_id"]),
            user_text=str(row["user_text"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            final_reply=str(row["final_reply"]),
        )

    def get_latest_session(self, chat_id: str, *, status: str | None = None) -> ExecutionTraceSessionRecord | None:
        query = """
            SELECT trace_id, chat_id, user_text, status, created_at, updated_at, final_reply
            FROM execution_trace_sessions
            WHERE chat_id = ?
        """
        params: list[object] = [chat_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT 1"
        row = self._connection.execute(query, tuple(params)).fetchone()
        if row is None:
            return None
        return ExecutionTraceSessionRecord(
            trace_id=str(row["trace_id"]),
            chat_id=str(row["chat_id"]),
            user_text=str(row["user_text"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            final_reply=str(row["final_reply"]),
        )

    def list_events(self, trace_id: str, *, limit: int | None = None) -> list[ExecutionTraceEventRecord]:
        query = """
            SELECT id, trace_id, event_index, event_type, summary, payload_json, created_at
            FROM execution_trace_events
            WHERE trace_id = ?
            ORDER BY event_index ASC
        """
        params: tuple[object, ...]
        if limit is not None:
            query += " LIMIT ?"
            params = (trace_id, limit)
        else:
            params = (trace_id,)
        rows = self._connection.execute(query, params).fetchall()
        return [
            ExecutionTraceEventRecord(
                id=int(row["id"]),
                trace_id=str(row["trace_id"]),
                event_index=int(row["event_index"]),
                event_type=str(row["event_type"]),
                summary=str(row["summary"]),
                payload=json.loads(str(row["payload_json"])),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]
