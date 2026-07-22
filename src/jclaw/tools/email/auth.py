from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jclaw.tools.google.auth import GoogleOAuthManager


@dataclass(slots=True)
class ConnectedEmailAccount:
    alias: str
    provider: str
    email_address: str
    scopes: tuple[str, ...]
    metadata: dict[str, object]


class GmailOAuthManager:
    def __init__(self, *, oauth_client_path: Path | None, token_dir: Path) -> None:
        self.google = GoogleOAuthManager(
            oauth_client_path=oauth_client_path,
            token_dir=token_dir,
            support_name="Gmail",
        )

    def connect_account(self, alias: str, scopes: tuple[str, ...]) -> ConnectedEmailAccount:
        creds = self.google.run_local_auth_flow(scopes)
        self.google.write_token(alias, creds.to_json())
        profile = self._build_service(creds).users().getProfile(userId="me").execute()
        return ConnectedEmailAccount(
            alias=alias,
            provider="gmail",
            email_address=str(profile.get("emailAddress", "")).strip(),
            scopes=scopes,
            metadata={"history_id": str(profile.get("historyId", ""))},
        )

    def load_credentials(self, alias: str, scopes: tuple[str, ...]) -> Any:
        return self.google.load_credentials(alias, scopes)

    def _run_local_auth_flow(self, scopes: tuple[str, ...]) -> Any:
        return self.google.run_local_auth_flow(scopes)

    def _build_service(self, creds: Any) -> Any:
        return self.google.build_service("gmail", "v1", creds)

    def _token_path(self, alias: str) -> Path:
        return self.google.token_path(alias)

    def _write_token(self, alias: str, raw_json: str) -> None:
        self.google.write_token(alias, raw_json)

    def _import_google_build(self) -> Any:
        return self.google._import_google_build()

    def _import_google_credentials(self) -> Any:
        return self.google._import_google_credentials()

    def _import_google_auth_requests(self) -> Any:
        return self.google._import_google_auth_requests()

    def _import_installed_app_flow(self) -> Any:
        return self.google._import_installed_app_flow()
