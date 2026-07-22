from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from jclaw.tools.google.auth import GoogleOAuthManager


class FakeFlow:
    client_path = ""
    scopes: ClassVar[list[str]] = []

    @classmethod
    def from_client_secrets_file(cls, client_path: str, *, scopes: list[str]) -> FakeFlow:
        cls.client_path = client_path
        cls.scopes = scopes
        return cls()

    def run_local_server(self, *, port: int, open_browser: bool) -> str:
        assert port == 0
        assert open_browser is True
        return "fake-creds"


class FakeCredentials:
    loaded_path = ""
    loaded_scopes: ClassVar[list[str]] = []
    instance: FakeCredentials

    def __init__(self, *, valid: bool, expired: bool = False, refresh_token: str = "") -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refreshed = False

    @classmethod
    def from_authorized_user_file(cls, token_path: str, *, scopes: list[str]) -> FakeCredentials:
        cls.loaded_path = token_path
        cls.loaded_scopes = scopes
        return cls.instance

    def refresh(self, request: object) -> None:
        _ = request
        self.refreshed = True
        self.valid = True

    def to_json(self) -> str:
        return '{"refreshed": true}'


class FakeRequests:
    class Request:
        pass


class FakeGoogleOAuthManager(GoogleOAuthManager):
    def __init__(self, *, oauth_client_path: Path | None, token_dir: Path) -> None:
        super().__init__(oauth_client_path=oauth_client_path, token_dir=token_dir, support_name="Google Docs")
        self.build_calls: list[tuple[str, str, object, bool]] = []

    def _import_installed_app_flow(self) -> Any:
        return FakeFlow

    def _import_google_credentials(self) -> Any:
        return FakeCredentials

    def _import_google_auth_requests(self) -> Any:
        return FakeRequests

    def _import_google_build(self) -> Any:
        def _build(service_name: str, version: str, *, credentials: object, cache_discovery: bool) -> dict[str, object]:
            self.build_calls.append((service_name, version, credentials, cache_discovery))
            return {"service": service_name, "version": version, "credentials": credentials}

        return _build


def test_google_oauth_manager_runs_local_auth_flow_with_configured_client(tmp_path) -> None:
    client_path = tmp_path / "client.json"
    client_path.write_text("{}", encoding="utf-8")
    manager = FakeGoogleOAuthManager(oauth_client_path=client_path, token_dir=tmp_path / "tokens")

    creds = manager.run_local_auth_flow(("scope-one", "scope-two"))

    assert creds == "fake-creds"
    assert FakeFlow.client_path == str(client_path)
    assert FakeFlow.scopes == ["scope-one", "scope-two"]


def test_google_oauth_manager_rejects_missing_client_json(tmp_path) -> None:
    manager = FakeGoogleOAuthManager(oauth_client_path=tmp_path / "missing.json", token_dir=tmp_path / "tokens")

    try:
        manager.run_local_auth_flow(("scope-one",))
    except RuntimeError as exc:
        assert "Google Docs OAuth client JSON is not configured" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing OAuth client JSON to raise RuntimeError")


def test_google_oauth_manager_loads_and_refreshes_credentials(tmp_path) -> None:
    token_dir = tmp_path / "tokens"
    manager = FakeGoogleOAuthManager(oauth_client_path=None, token_dir=token_dir)
    manager.write_token("default", "{}")
    FakeCredentials.instance = FakeCredentials(valid=False, expired=True, refresh_token="refresh-token")

    creds = manager.load_credentials("default", ("scope-one",))

    assert creds is FakeCredentials.instance
    assert creds.refreshed is True
    assert FakeCredentials.loaded_path == str(token_dir / "default.json")
    assert FakeCredentials.loaded_scopes == ["scope-one"]
    assert (token_dir / "default.json").read_text(encoding="utf-8") == '{"refreshed": true}'


def test_google_oauth_manager_builds_google_service(tmp_path) -> None:
    manager = FakeGoogleOAuthManager(oauth_client_path=None, token_dir=tmp_path / "tokens")
    creds = object()

    service = manager.build_service("docs", "v1", creds)

    assert service == {"service": "docs", "version": "v1", "credentials": creds}
    assert manager.build_calls == [("docs", "v1", creds, False)]
