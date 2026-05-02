from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3

from jclaw.core.records import (
    ApprovalRequestRecord,
    CronJobRecord,
    EmailAccountRecord,
    ExecutionTraceEventRecord,
    ExecutionTraceSessionRecord,
    GrantRecord,
    MemoryRecord,
    MessageRecord,
    WorkspaceChangeRecord,
)
from jclaw.core.stores.permissions import PermissionStore
from jclaw.core.stores.traces import TraceStore
from jclaw.core.stores.workspace_changes import WorkspaceChangeStore
from jclaw.core.time import utc_now


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self.traces = TraceStore(self._connection)
        self.workspace_changes = WorkspaceChangeStore(self._connection)
        self.permissions = PermissionStore(self._connection)
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

            """
        )
        self.traces.initialize()
        self.workspace_changes.initialize()
        self.permissions.initialize()
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
        return self.traces.create_session(chat_id, user_text)

    def append_execution_trace_event(
        self,
        trace_id: str,
        *,
        event_type: str,
        summary: str,
        payload: dict[str, object] | None = None,
    ) -> ExecutionTraceEventRecord:
        return self.traces.append_event(
            trace_id,
            event_type=event_type,
            summary=summary,
            payload=payload,
        )

    def finish_execution_trace_session(self, trace_id: str, *, status: str, final_reply: str = "") -> None:
        self.traces.finish_session(trace_id, status=status, final_reply=final_reply)

    def get_execution_trace_session(self, trace_id: str) -> ExecutionTraceSessionRecord | None:
        return self.traces.get_session(trace_id)

    def get_latest_execution_trace_session(
        self,
        chat_id: str,
        *,
        status: str | None = None,
    ) -> ExecutionTraceSessionRecord | None:
        return self.traces.get_latest_session(chat_id, status=status)

    def list_execution_trace_events(self, trace_id: str, *, limit: int | None = None) -> list[ExecutionTraceEventRecord]:
        return self.traces.list_events(trace_id, limit=limit)

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
        return self.workspace_changes.record_change(
            chat_id=chat_id,
            root_path=root_path,
            operation=operation,
            touched_files=touched_files,
            file_states=file_states,
        )

    def get_latest_workspace_change(
        self,
        chat_id: str,
        *,
        state: str,
        max_age_seconds: int | None = None,
    ) -> WorkspaceChangeRecord | None:
        return self.workspace_changes.get_latest_change(chat_id, state=state, max_age_seconds=max_age_seconds)

    def update_workspace_change_state(self, change_id: int, *, state: str) -> None:
        self.workspace_changes.update_change_state(change_id, state=state)

    def prune_workspace_changes(self, chat_id: str, *, keep_latest: int, max_age_seconds: int) -> None:
        self.workspace_changes.prune_changes(chat_id, keep_latest=keep_latest, max_age_seconds=max_age_seconds)

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
        return self.permissions.upsert_grant(root_path, capabilities, granted_by_chat_id)

    def list_grants(self, *, active_only: bool = True) -> list[GrantRecord]:
        return self.permissions.list_grants(active_only=active_only)

    def revoke_grant(self, grant_id: int) -> int:
        return self.permissions.revoke_grant(grant_id)

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
        return self.permissions.create_approval_request(
            kind=kind,
            chat_id=chat_id,
            root_path=root_path,
            capabilities=capabilities,
            objective=objective,
            payload=payload,
        )

    def get_approval_request(self, request_id: str) -> ApprovalRequestRecord | None:
        return self.permissions.get_approval_request(request_id)

    def list_approval_requests(self, *, status: str | None = None) -> list[ApprovalRequestRecord]:
        return self.permissions.list_approval_requests(status=status)

    def resolve_approval_request(self, request_id: str, status: str) -> int:
        return self.permissions.resolve_approval_request(request_id, status)

    def update_approval_request_status(self, request_id: str, status: str) -> int:
        return self.permissions.update_approval_request_status(request_id, status)

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
