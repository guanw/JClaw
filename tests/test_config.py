import tomllib

from jclaw.core.config import load_config, render_default_config
from jclaw.core.defaults import GOOGLE_DOCS_DEFAULT_SCOPES
from jclaw.core.environment import environment_catalog_path


def test_load_config_allows_email_disabled_without_oauth_client(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[email]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.email.enabled is False
    assert config.email.oauth_client_path is None


def test_load_config_uses_explicit_email_oauth_client_path(tmp_path) -> None:
    oauth_client = tmp_path / "client.json"
    oauth_client.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[email]
enabled = true
oauth_client_path = "{oauth_client}"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.email.enabled is True
    assert config.email.oauth_client_path == oauth_client


def test_load_config_allows_google_docs_disabled_without_oauth_client(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[google_docs]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.google_docs.enabled is False
    assert config.google_docs.oauth_client_path is None
    assert config.google_docs.scopes == GOOGLE_DOCS_DEFAULT_SCOPES


def test_load_config_reads_google_docs_settings(tmp_path) -> None:
    oauth_client = tmp_path / "google-client.json"
    oauth_client.write_text("{}", encoding="utf-8")
    token_dir = tmp_path / "google-tokens"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[google_docs]
enabled = true
oauth_client_path = "{oauth_client}"
token_dir = "{token_dir}"
scopes = ["scope-one", "scope-two"]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.google_docs.enabled is True
    assert config.google_docs.oauth_client_path == oauth_client
    assert config.google_docs.token_dir == token_dir
    assert config.google_docs.scopes == ("scope-one", "scope-two")


def test_render_default_config_includes_google_docs_section() -> None:
    data = tomllib.loads(render_default_config())

    assert data["google_docs"]["enabled"] is False
    assert data["google_docs"]["oauth_client_path"] == ""
    assert data["google_docs"]["scopes"] == list(GOOGLE_DOCS_DEFAULT_SCOPES)


def test_load_config_supports_extends_overlay(tmp_path) -> None:
    base_config = tmp_path / "base.toml"
    base_config.write_text(
        """
[provider]
api_key = "base-key"
base_url = "https://api.deepseek.com"
model = "deepseek-chat"

[telegram]
bot_token = "prod-token"
poll_timeout_seconds = 20

[memory]
max_memory_items = 4
""".strip(),
        encoding="utf-8",
    )
    dev_config = tmp_path / "dev.toml"
    dev_config.write_text(
        f"""
extends = "{base_config}"

[telegram]
bot_token = "dev-token"

[memory]
max_memory_items = 100
""".strip(),
        encoding="utf-8",
    )

    config = load_config(dev_config)

    assert config.provider.api_key == "base-key"
    assert config.telegram.bot_token == "dev-token"
    assert config.memory.max_memory_items == 100


def test_load_config_rejects_extends_cycles(tmp_path) -> None:
    a_config = tmp_path / "a.toml"
    b_config = tmp_path / "b.toml"
    a_config.write_text(f'extends = "{b_config}"\n', encoding="utf-8")
    b_config.write_text(f'extends = "{a_config}"\n', encoding="utf-8")

    try:
        load_config(a_config)
    except RuntimeError as exc:
        assert "cycle" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected extends cycle to raise RuntimeError")


def test_load_config_reads_notion_settings(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[notion]
enabled = true
api_token = "secret-token"
default_parent_id = "page-123"
writable_parent_ids = ["page-123", "db-456"]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.notion.enabled is True
    assert config.notion.api_token == "secret-token"
    assert config.notion.default_parent_id == "page-123"
    assert config.notion.writable_parent_ids == ("page-123", "db-456")


def test_load_config_uses_default_environment_catalog_path(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    config = load_config(config_path)

    assert config.daemon.environment_path == environment_catalog_path(config.daemon.state_dir)
