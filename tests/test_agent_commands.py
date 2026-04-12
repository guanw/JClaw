from pathlib import Path

from jclaw.ai.agent import AssistantAgent
from jclaw.core.config import Config, DaemonConfig, MemoryConfig, ProviderConfig, TelegramConfig
from jclaw.core.db import Database


class DummyLLM:
    def chat(self, messages):  # noqa: ANN001
        return "stubbed"


class SequenceLLM:
    def __init__(self, responses) -> None:  # noqa: ANN001
        self._responses = iter(responses)

    def chat(self, messages):  # noqa: ANN001
        return next(self._responses)


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


def test_llm_selected_tool_routes_to_browser(tmp_path) -> None:
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
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"use_tool": true, "tool": "browser", "action": "run_objective", "params": {"objective": "open example.com", "start_url": "https://example.com"}, "reason": "The user wants browser help."}',
                '{"status":"complete","chosen_element_id":null,"reason":"The current page is already the intended destination."}',
                "I opened the requested page and captured it.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "please open example.com for me")
    assert "opened" in reply.lower()
    db.close()


def test_choose_browser_link_uses_inspected_elements(tmp_path) -> None:
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
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(['{"chosen_element_id": "e2", "reason": "This is the relevant article."}']),
    )

    href = agent._choose_browser_link_via_llm(  # noqa: SLF001
        "latest deepseek news",
        {
            "url": "https://html.duckduckgo.com/html/?q=latest+deepseek+news",
            "title": "search results",
            "page_kind": "search_results",
            "text": "search results for deepseek news",
            "elements": [
                {"id": "e1", "role": "link", "text": "Settings", "href": "https://duckduckgo.com/settings", "area": "nav", "clickable": True},
                {"id": "e2", "role": "link", "text": "DeepSeek launches new model", "href": "https://example.com/deepseek-news", "area": "main", "clickable": True},
            ],
        },
    )
    assert href == "https://example.com/deepseek-news"
    db.close()


def test_choose_browser_next_action_uses_controller_output(tmp_path) -> None:
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
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(['{"status":"follow","chosen_element_id":"e2","reason":"Reuters market page is the strongest next source."}']),
    )

    decision = agent._choose_browser_next_action_via_llm(  # noqa: SLF001
        "latest trend on us stock market",
        {
            "url": "https://html.duckduckgo.com/html/?q=us+stock+market+trend",
            "title": "search results",
            "page_kind": "search_results",
            "text": "search results for us stock market trend",
            "elements": [
                {"id": "e1", "role": "link", "text": "Settings", "href": "https://duckduckgo.com/settings", "area": "nav", "clickable": True},
                {"id": "e2", "role": "link", "text": "US Markets News - Reuters", "href": "https://www.reuters.com/markets/us/", "area": "main", "clickable": True},
            ],
        },
        [],
    )
    assert decision == {
        "status": "follow",
        "url": "https://www.reuters.com/markets/us/",
        "reason": "Reuters market page is the strongest next source.",
    }
    db.close()


def test_llm_can_decline_tool_and_fall_back_to_chat(tmp_path) -> None:
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
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"use_tool": false, "tool": "", "action": "", "params": {}, "reason": "No tool needed."}',
                "Normal chat reply.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "say hello")
    assert reply == "Normal chat reply."
    db.close()
