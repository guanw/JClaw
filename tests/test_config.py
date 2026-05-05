from pathlib import Path

from jclaw.core.config import load_config


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
