from pathlib import Path

from jclaw.ai.agent import AssistantAgent
from jclaw.core.config import Config, DaemonConfig, MemoryConfig, ProviderConfig, TelegramConfig
from jclaw.core.defaults import (
    BROWSER_MAX_OBJECTIVE_STEPS,
    BROWSER_MAX_RESEARCH_SOURCES,
    BROWSER_VIEWPORT_HEIGHT,
    BROWSER_VIEWPORT_WIDTH,
    KNOWLEDGE_MAX_ANSWER_CITATIONS,
    KNOWLEDGE_MAX_CHUNKS_PER_FILE,
    KNOWLEDGE_MAX_FILE_READ_BYTES,
    KNOWLEDGE_MAX_FOLDER_SCAN_FILES,
    KNOWLEDGE_MAX_TOTAL_CHUNKS,
    KNOWLEDGE_TEXT_PREVIEW_CHARS,
    WORKSPACE_MAX_FILES_PER_CHANGE,
    WORKSPACE_MAX_INTERNAL_READ_BYTES,
    WORKSPACE_MAX_PATH_ENTRIES,
    WORKSPACE_MAX_PREPARED_DIFF_BYTES,
    WORKSPACE_MAX_STEPS,
    WORKSPACE_SHELL_OUTPUT_CHARS,
    WORKSPACE_SHELL_TIMEOUT_SECONDS,
)
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


def test_browser_config_defaults_are_centralized(tmp_path) -> None:
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
    assert config.browser.viewport_width == BROWSER_VIEWPORT_WIDTH
    assert config.browser.viewport_height == BROWSER_VIEWPORT_HEIGHT
    assert config.browser.max_objective_steps == BROWSER_MAX_OBJECTIVE_STEPS
    assert config.browser.max_research_sources == BROWSER_MAX_RESEARCH_SOURCES


def test_workspace_and_knowledge_defaults_are_centralized(tmp_path) -> None:
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
    assert config.workspace.max_steps == WORKSPACE_MAX_STEPS
    assert config.workspace.shell_timeout_seconds == WORKSPACE_SHELL_TIMEOUT_SECONDS
    assert config.workspace.shell_output_chars == WORKSPACE_SHELL_OUTPUT_CHARS
    assert config.workspace.max_prepared_diff_bytes == WORKSPACE_MAX_PREPARED_DIFF_BYTES
    assert config.workspace.max_files_per_change == WORKSPACE_MAX_FILES_PER_CHANGE
    assert config.workspace.max_path_entries == WORKSPACE_MAX_PATH_ENTRIES
    assert config.workspace.max_internal_read_bytes == WORKSPACE_MAX_INTERNAL_READ_BYTES
    assert config.knowledge.max_file_read_bytes == KNOWLEDGE_MAX_FILE_READ_BYTES
    assert config.knowledge.max_folder_scan_files == KNOWLEDGE_MAX_FOLDER_SCAN_FILES
    assert config.knowledge.max_chunks_per_file == KNOWLEDGE_MAX_CHUNKS_PER_FILE
    assert config.knowledge.max_total_chunks == KNOWLEDGE_MAX_TOTAL_CHUNKS
    assert config.knowledge.text_preview_chars == KNOWLEDGE_TEXT_PREVIEW_CHARS
    assert config.knowledge.max_answer_citations == KNOWLEDGE_MAX_ANSWER_CITATIONS


def test_workspace_preview_pauses_until_approval(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target = repo_root / "app.py"
    target.write_text("print('hello')\n", encoding="utf-8")
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
        repo_root=repo_root,
    )
    db = Database(config.daemon.db_path)
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"use_tool": true, "tool": "workspace", "action": "prepare_change", "params": {"objective": "Update app.py", "path": "app.py"}, "reason": "Local code change requested."}',
                '{"summary":"Update app.py","edits":[{"path":"app.py","reason":"Update greeting","new_content":"print(\\"goodbye\\")\\n"}]}',
            ]
        ),
    )

    grant_request_reply = agent.handle_text("chat-1", "update app.py")
    assert "Approval required" in grant_request_reply
    request_id = grant_request_reply.split("Use /approve ", 1)[1].split(" ", 1)[0]
    assert target.read_text(encoding="utf-8") == "print('hello')\n"

    approval_reply = agent.handle_text("chat-1", f"/approve {request_id}")
    assert "Granted" in approval_reply
    assert "Prepared a change preview" in approval_reply
    preview_request_id = approval_reply.split("Request: ", 1)[1].splitlines()[0]
    assert target.read_text(encoding="utf-8") == "print('hello')\n"

    apply_reply = agent.handle_text("chat-1", f"/approve {preview_request_id}")
    assert "Applied approved file change request" in apply_reply
    assert target.read_text(encoding="utf-8") == "print(\"goodbye\")\n"
    db.close()


def test_workspace_inspect_reply_is_raw_tool_output(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "notes.txt").write_text("hello\n", encoding="utf-8")
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
        repo_root=repo_root,
    )
    db = Database(config.daemon.db_path)
    db.upsert_grant(str(repo_root.resolve()), ("read",), "chat-1")
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"use_tool": true, "tool": "workspace", "action": "inspect_root", "params": {"path": ".", "objective": "Inspect repo root"}, "reason": "Local workspace inspection requested."}',
                "Hallucinated workspace summary that should never be used.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "what is in the repo root?")
    assert "Inspected" in reply
    assert "Entries:" in reply
    assert "notes.txt" in reply
    assert "Total entries: 1" in reply
    assert "Hallucinated workspace summary" not in reply
    db.close()


def test_workspace_inspect_reply_mentions_truncation(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for index in range(55):
        (repo_root / f"file-{index:02d}.txt").write_text("hello\n", encoding="utf-8")
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
        repo_root=repo_root,
    )
    db = Database(config.daemon.db_path)
    db.upsert_grant(str(repo_root.resolve()), ("read",), "chat-1")
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"use_tool": true, "tool": "workspace", "action": "inspect_root", "params": {"path": ".", "objective": "Inspect repo root"}, "reason": "Local workspace inspection requested."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "list the repo root")
    assert "Total entries: 55" in reply
    assert "Shown 50 of 55 entries." in reply
    db.close()


def test_workspace_grant_approval_resumes_inspect_root(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "notes.txt").write_text("hello\n", encoding="utf-8")
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
        repo_root=repo_root,
    )
    db = Database(config.daemon.db_path)
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"use_tool": true, "tool": "workspace", "action": "inspect_root", "params": {"path": ".", "objective": "Inspect repo root"}, "reason": "Local workspace inspection requested."}',
            ]
        ),
    )

    grant_request_reply = agent.handle_text("chat-1", "check the repo root")
    request_id = grant_request_reply.split("Use /approve ", 1)[1].split(" ", 1)[0]

    approval_reply = agent.handle_text("chat-1", f"/approve {request_id}")
    assert "Granted read access" in approval_reply
    assert "Inspected" in approval_reply
    assert "notes.txt" in approval_reply
    db.close()


def test_llm_selected_tool_routes_to_knowledge_and_returns_raw_result(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target = repo_root / "notes.txt"
    target.write_text("Project owner is guan.\n", encoding="utf-8")
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
        repo_root=repo_root,
    )
    db = Database(config.daemon.db_path)
    db.upsert_grant(str(repo_root.resolve()), ("read",), "chat-1")
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"use_tool": true, "tool": "knowledge", "action": "answer_from_paths", "params": {"paths": ["notes.txt"], "question": "Who is the project owner?"}, "reason": "The user is asking about local file contents."}',
                '{"answer":"The project owner is guan.","cited_chunk_ids":["notes.txt:1"],"grounded":true,"partial":false}',
                "Hallucinated post-processing that should not be used.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "who is the project owner in notes.txt?")
    assert "Answer:" in reply
    assert "The project owner is guan." in reply
    assert "Citations:" in reply
    assert "notes.txt [notes.txt:1]" in reply
    assert "Hallucinated post-processing" not in reply
    db.close()
