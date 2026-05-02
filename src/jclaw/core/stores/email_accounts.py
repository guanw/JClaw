from __future__ import annotations

from collections.abc import Iterable
import json
import sqlite3

from jclaw.core.records import EmailAccountRecord
from jclaw.core.time import utc_now


class EmailAccountStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def initialize(self) -> None:
        self._connection.executescript(
            """
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

    def upsert_account(
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
        record = self.get_account(alias)
        assert record is not None
        return record

    def list_accounts(self) -> list[EmailAccountRecord]:
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

    def get_account(self, alias: str) -> EmailAccountRecord | None:
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
