from __future__ import annotations

from pathlib import Path
from typing import Any


class GoogleOAuthManager:
    def __init__(
        self,
        *,
        oauth_client_path: Path | None,
        token_dir: Path,
        support_name: str = "Google API",
    ) -> None:
        self.oauth_client_path = oauth_client_path
        self.token_dir = token_dir
        self.support_name = support_name

    def run_local_auth_flow(self, scopes: tuple[str, ...]) -> Any:
        if self.oauth_client_path is None or not self.oauth_client_path.exists():
            raise RuntimeError(f"{self.support_name} OAuth client JSON is not configured.")
        flow_cls = self._import_installed_app_flow()
        flow = flow_cls.from_client_secrets_file(str(self.oauth_client_path), scopes=list(scopes))
        return flow.run_local_server(port=0, open_browser=True)

    def load_credentials(self, token_name: str, scopes: tuple[str, ...]) -> Any:
        token_path = self.token_path(token_name)
        if not token_path.exists():
            raise RuntimeError(f"{self.support_name} credentials '{token_name}' are not connected yet.")
        google_auth_requests = self._import_google_auth_requests()
        credentials_cls = self._import_google_credentials()
        creds = credentials_cls.from_authorized_user_file(str(token_path), scopes=list(scopes))
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(google_auth_requests.Request())
                self.write_token(token_name, creds.to_json())
            else:
                raise RuntimeError(f"{self.support_name} credentials '{token_name}' need to reconnect.")
        return creds

    def build_service(self, service_name: str, version: str, creds: Any) -> Any:
        build = self._import_google_build()
        return build(service_name, version, credentials=creds, cache_discovery=False)

    def token_path(self, token_name: str) -> Path:
        return self.token_dir / f"{token_name}.json"

    def write_token(self, token_name: str, raw_json: str) -> None:
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.token_path(token_name).write_text(raw_json, encoding="utf-8")

    def _import_google_build(self) -> Any:
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError(f"{self.support_name} support requires google-api-python-client to be installed.") from exc
        return build

    def _import_google_credentials(self) -> Any:
        try:
            from google.oauth2.credentials import Credentials
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError(f"{self.support_name} support requires google-auth to be installed.") from exc
        return Credentials

    def _import_google_auth_requests(self) -> Any:
        try:
            from google.auth.transport import requests
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError(f"{self.support_name} support requires google-auth to be installed.") from exc
        return requests

    def _import_installed_app_flow(self) -> Any:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise RuntimeError(f"{self.support_name} support requires google-auth-oauthlib to be installed.") from exc
        return InstalledAppFlow
