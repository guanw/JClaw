from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
import sqlite3

from jclaw.core.records import WorkspaceChangeRecord
from jclaw.core.time import utc_now


class WorkspaceChangeStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
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
            """
        )

    def record_change(
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

    def get_latest_change(
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

    def update_change_state(self, change_id: int, *, state: str) -> None:
        self._connection.execute(
            "UPDATE workspace_changes SET state = ?, updated_at = ? WHERE id = ?",
            (state, utc_now(), change_id),
        )
        self._connection.commit()

    def prune_changes(self, chat_id: str, *, keep_latest: int, max_age_seconds: int) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
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
