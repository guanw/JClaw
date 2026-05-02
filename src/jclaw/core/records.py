from __future__ import annotations

from dataclasses import dataclass


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
