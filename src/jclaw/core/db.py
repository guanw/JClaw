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
from jclaw.core.stores.cron import CronStore
from jclaw.core.stores.email_accounts import EmailAccountStore
from jclaw.core.stores.memory import MemoryStore
from jclaw.core.stores.messages import MessageStore
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
        self.messages = MessageStore(self._connection)
        self.memories = MemoryStore(self._connection)
        self.cron = CronStore(self._connection)
        self.email_accounts = EmailAccountStore(self._connection)
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
            """
        )
        self.messages.initialize()
        self.memories.initialize()
        self.cron.initialize()
        self.email_accounts.initialize()
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
        self.messages.store(chat_id, role, content, channel=channel, external_id=external_id)

    def recent_messages(self, chat_id: str, limit: int) -> list[MessageRecord]:
        return self.messages.recent(chat_id, limit)

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
        self.memories.remember(scope, key, value)

    def forget(self, scope: str, key: str) -> int:
        return self.memories.forget(scope, key)

    def list_memories(self, scope: str, limit: int = 20) -> list[MemoryRecord]:
        return self.memories.list(scope, limit=limit)

    def search_memories(self, scope: str, query: str, limit: int) -> list[MemoryRecord]:
        return self.memories.search(scope, query, limit)

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
        return self.cron.add_job(chat_id, schedule, prompt, next_run_at)

    def list_cron_jobs(self, chat_id: str) -> list[CronJobRecord]:
        return self.cron.list_jobs(chat_id)

    def get_cron_job(self, chat_id: str, job_id: int) -> CronJobRecord | None:
        return self.cron.get_job(chat_id, job_id)

    def remove_cron_job(self, chat_id: str, job_id: int) -> int:
        return self.cron.remove_job(chat_id, job_id)

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
        return self.cron.update_job(
            chat_id,
            job_id,
            schedule=schedule,
            prompt=prompt,
            next_run_at=next_run_at,
            enabled=enabled,
        )

    def due_cron_jobs(self, now: str) -> list[CronJobRecord]:
        return self.cron.due_jobs(now)

    def update_cron_next_run(self, job_id: int, next_run_at: str) -> None:
        self.cron.update_next_run(job_id, next_run_at)

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
        return self.email_accounts.upsert_account(
            alias=alias,
            provider=provider,
            email_address=email_address,
            scopes=scopes,
            status=status,
            metadata=metadata,
        )

    def list_email_accounts(self) -> list[EmailAccountRecord]:
        return self.email_accounts.list_accounts()

    def get_email_account(self, alias: str) -> EmailAccountRecord | None:
        return self.email_accounts.get_account(alias)
