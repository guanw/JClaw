from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class ConnectedEmailAccount:
    alias: str
    provider: str
    email_address: str
    scopes: tuple[str, ...]
    metadata: dict[str, object]


class GmailOAuthManager:
    def __init__(self, *, oauth_client_path: Path | None, token_dir: Path) -> None:
        self.oauth_client_path = oauth_client_path
        self.token_dir = token_dir

    def connect_account(self, alias: str, scopes: tuple[str, ...]) -> ConnectedEmailAccount:
        if self.oauth_client_path is None or not self.oauth_client_path.exists():
            raise RuntimeError("Gmail OAuth client JSON is not configured.")
        creds = self._run_local_auth_flow(scopes)
        self._write_token(alias, creds.to_json())
        profile = self._build_service(creds).users().getProfile(userId="me").execute()
        return ConnectedEmailAccount(
            alias=alias,
            provider="gmail",
            email_address=str(profile.get("emailAddress", "")).strip(),
            scopes=scopes,
            metadata={"history_id": str(profile.get("historyId", ""))},
        )

    def load_credentials(self, alias: str, scopes: tuple[str, ...]) -> Any:
        token_path = self._token_path(alias)
        if not token_path.exists():
            raise RuntimeError(f"Gmail account '{alias}' is not connected yet.")
        google_auth_requests = self._import_google_auth_requests()
        credentials_cls = self._import_google_credentials()
        creds = credentials_cls.from_authorized_user_file(str(token_path), scopes=list(scopes))
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(google_auth_requests.Request())
                self._write_token(alias, creds.to_json())
            else:
                raise RuntimeError(f"Gmail account '{alias}' needs to reconnect.")
        return creds

    def _run_local_auth_flow(self, scopes: tuple[str, ...]) -> Any:
        flow_cls = self._import_installed_app_flow()
        flow = flow_cls.from_client_secrets_file(str(self.oauth_client_path), scopes=list(scopes))
        return flow.run_local_server(port=0, open_browser=True)

    def _build_service(self, creds: Any) -> Any:
        build = self._import_google_build()
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    def _token_path(self, alias: str) -> Path:
        return self.token_dir / f"{alias}.json"

    def _write_token(self, alias: str, raw_json: str) -> None:
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self._token_path(alias).write_text(raw_json, encoding="utf-8")

    def _import_google_build(self) -> Any:
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError("Gmail support requires google-api-python-client to be installed.") from exc
        return build

    def _import_google_credentials(self) -> Any:
        try:
            from google.oauth2.credentials import Credentials
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError("Gmail support requires google-auth to be installed.") from exc
        return Credentials

    def _import_google_auth_requests(self) -> Any:
        try:
            from google.auth.transport import requests
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError("Gmail support requires google-auth to be installed.") from exc
        return requests

    def _import_installed_app_flow(self) -> Any:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError("Gmail support requires google-auth-oauthlib to be installed.") from exc
        return InstalledAppFlow
