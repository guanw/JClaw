from pathlib import Path

from jclaw.ai.agent import AssistantAgent
from jclaw.core.config import Config, DaemonConfig, MemoryConfig, ProviderConfig, TelegramConfig
from jclaw.core.db import Database


class DummyLLM:
    def chat(self, messages):  # noqa: ANN001
        return "stubbed"


def test_command_flow(tmp_path) -> None:
    config = Config(
        provider=ProviderConfig(),
        telegram=TelegramConfig(),
        daemon=DaemonConfig(
            state_dir=tmp_path,
            db_path=tmp_path / "jclaw.db",
            stdout_log=tmp_path / "stdout.log",
            stderr_log=tmp_path / "stderr.log",
        ),
        memory=MemoryConfig(),
        config_path=tmp_path / "config.toml",
        repo_root=Path("/Users/guanw/Documents/JClaw"),
    )
    db = Database(config.daemon.db_path)
    agent = AssistantAgent(config, db, DummyLLM())

    assert "Remembered" in agent.handle_text("chat-1", "/remember owner = guan")
    assert "owner = guan" in agent.handle_text("chat-1", "/memory")
    assert "Cron job" in agent.handle_text("chat-1", "/cron add every 30m | stretch")
    assert "1." in agent.handle_text("chat-1", "/cron list")
    db.close()

