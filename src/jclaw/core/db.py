from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import uuid


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


@dataclass(slots=True)
class GrantRecord:
    id: int
    root_path: str
    capabilities: tuple[str, ...]
    granted_by_chat_id: str
    created_at: str
    revoked_at: str | None


@dataclass(slots=True)
class ApprovalRequestRecord:
    request_id: str
    kind: str
    chat_id: str
    root_path: str
    capabilities: tuple[str, ...]
    objective: str
    payload: dict[str, object]
    status: str
    created_at: str
    resolved_at: str | None


@dataclass(slots=True)
class EmailAccountRecord:
    alias: str
    provider: str
    email_address: str
    scopes: tuple[str, ...]
    status: str
    created_at: str
    updated_at: str
    metadata: dict[str, object]


@dataclass(slots=True)
class WorkspaceChangeRecord:
    id: int
    chat_id: str
    root_path: str
    operation: str
    touched_files: tuple[str, ...]
    file_states: list[dict[str, object]]
    state: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ExecutionTraceSessionRecord:
    trace_id: str
    chat_id: str
    user_text: str
    status: str
    created_at: str
    updated_at: str
    final_reply: str


@dataclass(slots=True)
class ExecutionTraceEventRecord:
    id: int
    trace_id: str
    event_index: int
    event_type: str
    summary: str
    payload: dict[str, object]
    created_at: str


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

            CREATE TABLE IF NOT EXISTS grants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_path TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                granted_by_chat_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_grants_active_root
                ON grants(root_path)
                WHERE revoked_at IS NULL;

            CREATE TABLE IF NOT EXISTS approval_requests (
                request_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                root_path TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                objective TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS email_accounts (
                alias TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                email_address TEXT NOT NULL,
                scopes_json TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspace_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                root_path TEXT NOT NULL,
                operation TEXT NOT NULL,
                touched_files_json TEXT NOT NULL,
                file_states_json TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

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

    def get_trace_mode(self, chat_id: str) -> str:
        raw = self.get_kv(f"trace.mode.{chat_id}", "off").strip().lower()
        return raw if raw in {"off", "summary"} else "off"

    def set_trace_mode(self, chat_id: str, mode: str) -> None:
        normalized = mode.strip().lower()
        self.set_kv(f"trace.mode.{chat_id}", normalized if normalized in {"off", "summary"} else "off")

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

    def create_execution_trace_session(self, chat_id: str, user_text: str) -> ExecutionTraceSessionRecord:
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

    def append_execution_trace_event(
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

    def finish_execution_trace_session(self, trace_id: str, *, status: str, final_reply: str = "") -> None:
        self._connection.execute(
            """
            UPDATE execution_trace_sessions
            SET status = ?, final_reply = ?, updated_at = ?
            WHERE trace_id = ?
            """,
            (status, final_reply, utc_now(), trace_id),
        )
        self._connection.commit()

    def get_execution_trace_session(self, trace_id: str) -> ExecutionTraceSessionRecord | None:
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

    def get_latest_execution_trace_session(
        self,
        chat_id: str,
        *,
        status: str | None = None,
    ) -> ExecutionTraceSessionRecord | None:
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

    def list_execution_trace_events(self, trace_id: str, *, limit: int | None = None) -> list[ExecutionTraceEventRecord]:
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

    def record_workspace_change(
        self,
        *,
        chat_id: str,
        root_path: str,
        operation: str,
        touched_files: Iterable[str],
        file_states: list[dict[str, object]],
    ) -> WorkspaceChangeRecord:
        now = utc_now()
        touched_files_tuple = tuple(str(item) for item in touched_files)
        self._connection.execute(
            "DELETE FROM workspace_changes WHERE chat_id = ? AND state = 'undone'",
            (chat_id,),
        )
        cursor = self._connection.execute(
            """
            INSERT INTO workspace_changes(
                chat_id, root_path, operation, touched_files_json, file_states_json, state, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'applied', ?, ?)
            """,
            (
                chat_id,
                root_path,
                operation,
                json.dumps(touched_files_tuple, ensure_ascii=True),
                json.dumps(file_states, ensure_ascii=True),
                now,
                now,
            ),
        )
        self._connection.commit()
        return WorkspaceChangeRecord(
            id=int(cursor.lastrowid),
            chat_id=chat_id,
            root_path=root_path,
            operation=operation,
            touched_files=touched_files_tuple,
            file_states=file_states,
            state="applied",
            created_at=now,
            updated_at=now,
        )

    def get_latest_workspace_change(
        self,
        chat_id: str,
        *,
        state: str,
        max_age_seconds: int | None = None,
    ) -> WorkspaceChangeRecord | None:
        row = self._connection.execute(
            """
            SELECT id, chat_id, root_path, operation, touched_files_json, file_states_json, state, created_at, updated_at
            FROM workspace_changes
            WHERE chat_id = ? AND state = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id, state),
        ).fetchone()
        if row is None:
            return None
        record = WorkspaceChangeRecord(
            id=int(row["id"]),
            chat_id=str(row["chat_id"]),
            root_path=str(row["root_path"]),
            operation=str(row["operation"]),
            touched_files=tuple(json.loads(str(row["touched_files_json"]))),
            file_states=list(json.loads(str(row["file_states_json"]))),
            state=str(row["state"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        if max_age_seconds is not None:
            created = datetime.fromisoformat(record.created_at)
            age_seconds = (datetime.now(timezone.utc) - created).total_seconds()
            if age_seconds > max_age_seconds:
                return None
        return record

    def update_workspace_change_state(self, change_id: int, *, state: str) -> None:
        self._connection.execute(
            "UPDATE workspace_changes SET state = ?, updated_at = ? WHERE id = ?",
            (state, utc_now(), change_id),
        )
        self._connection.commit()

    def prune_workspace_changes(self, chat_id: str, *, keep_latest: int, max_age_seconds: int) -> None:
        cutoff = (datetime.now(timezone.utc).timestamp() - max_age_seconds)
        rows = self._connection.execute(
            """
            SELECT id, created_at
            FROM workspace_changes
            WHERE chat_id = ?
            ORDER BY id DESC
            """,
            (chat_id,),
        ).fetchall()
        keep_ids: set[int] = set()
        for index, row in enumerate(rows):
            record_id = int(row["id"])
            created_at = datetime.fromisoformat(str(row["created_at"])).timestamp()
            if index < keep_latest and created_at >= cutoff:
                keep_ids.add(record_id)
        if keep_ids:
            placeholders = ", ".join("?" for _ in keep_ids)
            params: tuple[object, ...] = (chat_id, *keep_ids)
            self._connection.execute(
                f"DELETE FROM workspace_changes WHERE chat_id = ? AND id NOT IN ({placeholders})",  # noqa: S608
                params,
            )
        else:
            self._connection.execute("DELETE FROM workspace_changes WHERE chat_id = ?", (chat_id,))
        self._connection.commit()

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

    def get_cron_job(self, chat_id: str, job_id: int) -> CronJobRecord | None:
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

    def remove_cron_job(self, chat_id: str, job_id: int) -> int:
        cursor = self._connection.execute(
            "DELETE FROM cron_jobs WHERE chat_id = ? AND id = ?",
            (chat_id, job_id),
        )
        self._connection.commit()
        return cursor.rowcount

    def update_cron_job(
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

    def upsert_grant(self, root_path: str, capabilities: Iterable[str], granted_by_chat_id: str) -> GrantRecord:
        normalized_caps = tuple(sorted({cap.strip() for cap in capabilities if cap.strip()}))
        now = utc_now()
        row = self._connection.execute(
            """
            SELECT id, root_path, capabilities_json, granted_by_chat_id, created_at, revoked_at
            FROM grants
            WHERE root_path = ? AND revoked_at IS NULL
            """,
            (root_path,),
        ).fetchone()
        if row is None:
            cursor = self._connection.execute(
                """
                INSERT INTO grants(root_path, capabilities_json, granted_by_chat_id, created_at, revoked_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (root_path, json.dumps(normalized_caps), granted_by_chat_id, now),
            )
            self._connection.commit()
            return GrantRecord(
                id=int(cursor.lastrowid),
                root_path=root_path,
                capabilities=normalized_caps,
                granted_by_chat_id=granted_by_chat_id,
                created_at=now,
                revoked_at=None,
            )

        merged_caps = tuple(sorted(set(json.loads(str(row["capabilities_json"]))) | set(normalized_caps)))
        self._connection.execute(
            """
            UPDATE grants
            SET capabilities_json = ?, granted_by_chat_id = ?
            WHERE id = ?
            """,
            (json.dumps(merged_caps), granted_by_chat_id, int(row["id"])),
        )
        self._connection.commit()
        return GrantRecord(
            id=int(row["id"]),
            root_path=str(row["root_path"]),
            capabilities=merged_caps,
            granted_by_chat_id=granted_by_chat_id,
            created_at=str(row["created_at"]),
            revoked_at=None if row["revoked_at"] is None else str(row["revoked_at"]),
        )

    def list_grants(self, *, active_only: bool = True) -> list[GrantRecord]:
        query = """
            SELECT id, root_path, capabilities_json, granted_by_chat_id, created_at, revoked_at
            FROM grants
        """
        params: tuple[object, ...] = ()
        if active_only:
            query += " WHERE revoked_at IS NULL"
        query += " ORDER BY id ASC"
        rows = self._connection.execute(query, params).fetchall()
        return [
            GrantRecord(
                id=int(row["id"]),
                root_path=str(row["root_path"]),
                capabilities=tuple(json.loads(str(row["capabilities_json"]))),
                granted_by_chat_id=str(row["granted_by_chat_id"]),
                created_at=str(row["created_at"]),
                revoked_at=None if row["revoked_at"] is None else str(row["revoked_at"]),
            )
            for row in rows
        ]

    def revoke_grant(self, grant_id: int) -> int:
        cursor = self._connection.execute(
            "UPDATE grants SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (utc_now(), grant_id),
        )
        self._connection.commit()
        return cursor.rowcount

    def create_approval_request(
        self,
        *,
        kind: str,
        chat_id: str,
        root_path: str,
        capabilities: Iterable[str],
        objective: str,
        payload: dict[str, object],
    ) -> ApprovalRequestRecord:
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        capabilities_tuple = tuple(sorted({cap.strip() for cap in capabilities if cap.strip()}))
        payload_json = json.dumps(payload, ensure_ascii=True)
        now = utc_now()
        self._connection.execute(
            """
            INSERT INTO approval_requests(
                request_id, kind, chat_id, root_path, capabilities_json, objective, payload_json, status, created_at, resolved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL)
            """,
            (request_id, kind, chat_id, root_path, json.dumps(capabilities_tuple), objective, payload_json, now),
        )
        self._connection.commit()
        return ApprovalRequestRecord(
            request_id=request_id,
            kind=kind,
            chat_id=chat_id,
            root_path=root_path,
            capabilities=capabilities_tuple,
            objective=objective,
            payload=payload,
            status="pending",
            created_at=now,
            resolved_at=None,
        )

    def get_approval_request(self, request_id: str) -> ApprovalRequestRecord | None:
        row = self._connection.execute(
            """
            SELECT request_id, kind, chat_id, root_path, capabilities_json, objective, payload_json, status, created_at, resolved_at
            FROM approval_requests
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return ApprovalRequestRecord(
            request_id=str(row["request_id"]),
            kind=str(row["kind"]),
            chat_id=str(row["chat_id"]),
            root_path=str(row["root_path"]),
            capabilities=tuple(json.loads(str(row["capabilities_json"]))),
            objective=str(row["objective"]),
            payload=json.loads(str(row["payload_json"])),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            resolved_at=None if row["resolved_at"] is None else str(row["resolved_at"]),
        )

    def list_approval_requests(self, *, status: str | None = None) -> list[ApprovalRequestRecord]:
        query = """
            SELECT request_id, kind, chat_id, root_path, capabilities_json, objective, payload_json, status, created_at, resolved_at
            FROM approval_requests
        """
        params: tuple[object, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at ASC"
        rows = self._connection.execute(query, params).fetchall()
        return [
            ApprovalRequestRecord(
                request_id=str(row["request_id"]),
                kind=str(row["kind"]),
                chat_id=str(row["chat_id"]),
                root_path=str(row["root_path"]),
                capabilities=tuple(json.loads(str(row["capabilities_json"]))),
                objective=str(row["objective"]),
                payload=json.loads(str(row["payload_json"])),
                status=str(row["status"]),
                created_at=str(row["created_at"]),
                resolved_at=None if row["resolved_at"] is None else str(row["resolved_at"]),
            )
            for row in rows
        ]

    def resolve_approval_request(self, request_id: str, status: str) -> int:
        cursor = self._connection.execute(
            """
            UPDATE approval_requests
            SET status = ?, resolved_at = ?
            WHERE request_id = ? AND status = 'pending'
            """,
            (status, utc_now(), request_id),
        )
        self._connection.commit()
        return cursor.rowcount

    def update_approval_request_status(self, request_id: str, status: str) -> int:
        cursor = self._connection.execute(
            """
            UPDATE approval_requests
            SET status = ?, resolved_at = CASE
                WHEN ? IN ('denied', 'aborted', 'applied', 'failed') THEN ?
                ELSE resolved_at
            END
            WHERE request_id = ?
            """,
            (status, status, utc_now(), request_id),
        )
        self._connection.commit()
        return cursor.rowcount

    def upsert_email_account(
        self,
        *,
        alias: str,
        provider: str,
        email_address: str,
        scopes: Iterable[str],
        status: str,
        metadata: dict[str, object],
    ) -> EmailAccountRecord:
        normalized_scopes = tuple(sorted({scope.strip() for scope in scopes if scope.strip()}))
        now = utc_now()
        self._connection.execute(
            """
            INSERT INTO email_accounts(alias, provider, email_address, scopes_json, status, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alias) DO UPDATE SET
                provider = excluded.provider,
                email_address = excluded.email_address,
                scopes_json = excluded.scopes_json,
                status = excluded.status,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                alias,
                provider,
                email_address,
                json.dumps(normalized_scopes),
                status,
                json.dumps(metadata, ensure_ascii=True),
                now,
                now,
            ),
        )
        self._connection.commit()
        record = self.get_email_account(alias)
        assert record is not None
        return record

    def list_email_accounts(self) -> list[EmailAccountRecord]:
        rows = self._connection.execute(
            """
            SELECT alias, provider, email_address, scopes_json, status, metadata_json, created_at, updated_at
            FROM email_accounts
            ORDER BY alias ASC
            """
        ).fetchall()
        return [
            EmailAccountRecord(
                alias=str(row["alias"]),
                provider=str(row["provider"]),
                email_address=str(row["email_address"]),
                scopes=tuple(json.loads(str(row["scopes_json"]))),
                status=str(row["status"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                metadata=json.loads(str(row["metadata_json"])),
            )
            for row in rows
        ]

    def get_email_account(self, alias: str) -> EmailAccountRecord | None:
        row = self._connection.execute(
            """
            SELECT alias, provider, email_address, scopes_json, status, metadata_json, created_at, updated_at
            FROM email_accounts
            WHERE alias = ?
            """,
            (alias,),
        ).fetchone()
        if row is None:
            return None
        return EmailAccountRecord(
            alias=str(row["alias"]),
            provider=str(row["provider"]),
            email_address=str(row["email_address"]),
            scopes=tuple(json.loads(str(row["scopes_json"]))),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            metadata=json.loads(str(row["metadata_json"])),
        )
