from __future__ import annotations

from collections.abc import Iterable
import json
import sqlite3
import uuid

from jclaw.core.records import ApprovalRequestRecord, GrantRecord
from jclaw.core.time import utc_now


class PermissionStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
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
            """
        )

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
