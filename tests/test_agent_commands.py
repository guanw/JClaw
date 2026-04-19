from pathlib import Path

from jclaw.ai.agent import AssistantAgent
from jclaw.core.config import Config, DaemonConfig, MemoryConfig, ProviderConfig, TelegramConfig
from jclaw.core.defaults import (
    AUTOMATION_ENABLED,
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
from jclaw.tools.base import ToolContext, ToolLoopFinalizer, ToolResult


class DummyLLM:
    def chat(self, messages):  # noqa: ANN001
        return "stubbed"


class SequenceLLM:
    def __init__(self, responses) -> None:  # noqa: ANN001
        self._responses = iter(responses)

    def chat(self, messages):  # noqa: ANN001
        return next(self._responses)


class FakeTool:
    name = "fake"

    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict]] = []

    def describe(self) -> dict:
        return {
            "name": "fake",
            "description": "Fake tool used for loop tests.",
            "implemented": True,
            "actions": {
                "step_one": {
                    "description": "Produce an intermediate result.",
                    "use_when": ["A first fake step is needed."],
                },
                "step_two": {
                    "description": "Produce the final result.",
                    "use_when": ["A second fake step is needed."],
                },
            },
        }

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append((action, dict(params)))
        if action == "step_one":
            return ToolResult(ok=True, summary="Completed fake step one.", data={"phase": "intermediate"})
        if action == "step_two":
            return ToolResult(ok=True, summary="Completed fake step two.", data={"answer": "final fake answer", "grounded": True})
        raise AssertionError(f"unexpected action: {action}")

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        if result.data.get("answer"):
            lines.append(f"Answer:\n{result.data['answer']}")
        return "\n".join(lines)


class ExplodingTool:
    name = "exploding"

    def describe(self) -> dict:
        return {
            "name": "exploding",
            "description": "Tool that raises to verify agent failure handling.",
            "implemented": True,
            "prefer_direct_result": True,
            "actions": {
                "boom": {
                    "description": "Raise an exception.",
                    "use_when": ["Testing tool failure handling."],
                },
            },
        }

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        raise ValueError("boom")

    def format_result(self, action: str, result: ToolResult) -> str:
        return result.summary


class FakeBrowserTool:
    name = "browser"

    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict]] = []

    def describe(self) -> dict:
        return {
            "name": "browser",
            "description": "Fake browser tool used for loop session tests.",
            "actions": {
                "search_web": {"description": "Search the web."},
                "read_page": {"description": "Read the current page."},
                "extract": {"description": "Extract fields from the current page."},
                "close_session": {"description": "Close the current page session."},
            },
            "supports_followup": True,
        }

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append((action, dict(params)))
        if action == "search_web":
            assert ctx.execution is not None
            ctx.execution.tool_state["browser"] = {"session_id": "sess_browser_loop"}
            ctx.execution.finalizers["browser"] = ToolLoopFinalizer(
                action="close_session",
                params={"session_id": "sess_browser_loop"},
            )
            return ToolResult(
                ok=True,
                summary="Searched the web.",
                data={"session_id": "sess_browser_loop", "url": "https://example.com/result"},
            )
        if action == "read_page":
            assert params == {}
            assert ctx.execution is not None
            assert ctx.execution.tool_state["browser"]["session_id"] == "sess_browser_loop"
            return ToolResult(
                ok=True,
                summary="Read page.",
                data={"session_id": "sess_browser_loop", "url": "https://example.com/result", "title": "result"},
            )
        if action == "extract":
            assert params == {"fields": {"name": "contractor name"}}
            assert ctx.execution is not None
            assert ctx.execution.tool_state["browser"]["session_id"] == "sess_browser_loop"
            return ToolResult(
                ok=True,
                summary="Extracted fields.",
                data={"session_id": "sess_browser_loop", "fields": {"name": "Breathe Easy Remodeling"}},
            )
        if action == "close_session":
            assert params.get("session_id") == "sess_browser_loop"
            return ToolResult(ok=True, summary="Closed session.", data={"session_id": "sess_browser_loop"})
        raise AssertionError(f"unexpected action: {action}")

    def format_result(self, action: str, result: ToolResult) -> str:
        return result.summary


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


def test_llm_selected_tool_routes_to_memory_remember(tmp_path) -> None:
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
                '{"status":"continue","tool":"memory","action":"remember_fact","params":{"key":"favorite_color","value":"blue"},"reason":"The user asked to remember a preference."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "remember that my favorite color is blue")
    assert "Remembered 'favorite_color'." in reply
    assert db.list_memories("chat-1")[0].value == "blue"
    db.close()


def test_llm_selected_tool_routes_to_memory_search(tmp_path) -> None:
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
    db.remember("chat-1", "favorite_color", "blue")
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"status":"continue","tool":"memory","action":"search_memories","params":{"query":"favorite color"},"reason":"The user is asking what is remembered."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "what do you remember about my favorite color")
    assert "favorite_color = blue" in reply
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
                '{"status":"continue","tool":"browser","action":"run_objective","params":{"objective":"open example.com","start_url":"https://example.com"},"reason":"The user wants browser help."}',
                '{"status":"complete","tool":"","action":"","params":{},"reason":"The current page is already the intended destination."}',
                "I opened the requested page and captured it.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "please open example.com for me")
    assert "example.com" in reply.lower()
    assert "example domain" in reply.lower()
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

    browser = agent.tools.get("browser")
    href = browser._choose_follow_up_url_via_llm(  # noqa: SLF001
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

    browser = agent.tools.get("browser")
    decision = browser._decide_next_action_via_llm(  # noqa: SLF001
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
        "evidence_refs": [],
        "missing_information": "",
        "chosen_element_id": "e2",
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
                '{"status":"complete","tool":"","action":"","params":{},"reason":"No tool needed."}',
                "Normal chat reply.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "say hello")
    assert reply == "Normal chat reply."
    db.close()


def test_tool_loop_returns_failure_reply_when_tool_raises(tmp_path) -> None:
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
                '{"status":"continue","tool":"exploding","action":"boom","params":{},"reason":"Exercise tool failure handling."}',
            ]
        ),
    )
    agent.tools.register(ExplodingTool())

    reply = agent.handle_text("chat-1", "trigger failure")
    assert reply == "Tool exploding.boom failed: boom"
    db.close()


def test_browser_tool_loop_reuses_one_session_across_steps_and_closes_at_end(tmp_path) -> None:
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
                '{"status":"continue","tool":"browser","action":"search_web","params":{"query":"millburn contractors"},"reason":"Search first."}',
                '{"status":"continue","tool":"browser","action":"read_page","params":{},"reason":"Read the result page."}',
                '{"status":"continue","tool":"browser","action":"extract","params":{"fields":{"name":"contractor name"}},"reason":"Extract the contractor name."}',
                '{"status":"complete","tool":"","action":"","params":{},"reason":"Enough information gathered."}',
                "Found one contractor.",
            ]
        ),
    )
    fake_browser = FakeBrowserTool()
    agent.tools._tools["browser"] = fake_browser  # noqa: SLF001

    reply = agent.handle_text("chat-1", "find a millburn remodeling contractor")

    assert reply == "Found one contractor."
    assert fake_browser.invocations == [
        ("search_web", {"query": "millburn contractors"}),
        ("read_page", {}),
        ("extract", {"fields": {"name": "contractor name"}}),
        ("close_session", {"session_id": "sess_browser_loop"}),
    ]
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
    assert config.automation.enabled == AUTOMATION_ENABLED
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
                '{"status":"continue","tool":"workspace","action":"prepare_change","params":{"objective":"Update app.py","path":"app.py"},"reason":"Local code change requested."}',
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


def test_workspace_path_mutation_approval_applies_request(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target = repo_root / "notes.txt"
    target.write_text("hello\n", encoding="utf-8")
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
    db.upsert_grant(str(repo_root.resolve()), ("write",), "chat-1")
    agent = AssistantAgent(config, db, DummyLLM())

    preview = agent.tools.get("workspace").invoke(
        "rename_path",
        {"path": str(target), "new_name": "renamed.txt"},
        ToolContext(chat_id="chat-1"),
    )
    assert preview.needs_confirmation is True

    apply_reply = agent.handle_text("chat-1", f"/approve {preview.data['request_id']}")
    assert "Applied approved path request" in apply_reply
    assert (repo_root / "renamed.txt").exists()
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
                '{"status":"continue","tool":"workspace","action":"inspect_root","params":{"path":".","objective":"Inspect repo root"},"reason":"Local workspace inspection requested."}',
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


def test_workspace_find_files_reply_preserves_all_matches(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for index in range(6):
        (repo_root / f"file-{index}.py").write_text("print('x')\n", encoding="utf-8")
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
                '{"status":"continue","tool":"workspace","action":"find_files","params":{"path":".","pattern":"*.py"},"reason":"Find python files."}',
                "This fallback LLM text should not be used.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "find all python files")
    for index in range(6):
        assert f"file-{index}.py" in reply
    assert "This fallback LLM text should not be used." not in reply
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
                '{"status":"continue","tool":"workspace","action":"inspect_root","params":{"path":".","objective":"Inspect repo root"},"reason":"Local workspace inspection requested."}',
                '{"status":"complete","tool":"","action":"","params":{},"reason":"A plain directory listing satisfies the request."}',
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
                '{"status":"continue","tool":"workspace","action":"inspect_root","params":{"path":".","objective":"Inspect repo root"},"reason":"Local workspace inspection requested."}',
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
                '{"status":"continue","tool":"knowledge","action":"answer_from_paths","params":{"paths":["notes.txt"],"question":"Who is the project owner?"},"reason":"The user is asking about local file contents."}',
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
    assert "Supported files:" not in reply
    assert "Hallucinated post-processing" not in reply
    db.close()


def test_knowledge_summary_recovers_from_truncated_json(tmp_path) -> None:
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
                '```json\n{"summary":"The documents concern a real estate transaction.","cited_chunk_ids":["file.pdf:1","file.pdf:2',
            ]
        ),
    )

    result = agent._summarize_knowledge_documents_via_llm(  # noqa: SLF001
        {
            "chunks": [
                {"chunk_id": "file.pdf:1", "path": "/tmp/file.pdf", "text": "sample", "start_offset": 0, "end_offset": 6},
            ]
        }
    )
    assert result is not None
    assert result["summary"] == "The documents concern a real estate transaction."
    db.close()


def test_tool_loop_can_chain_workspace_then_knowledge(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    folder = repo_root / "Desktop" / "819"
    folder.mkdir(parents=True)
    (folder / "a-contract.txt").write_text("Purchase price is $1,250,000.\n", encoding="utf-8")
    (folder / "b-notes.txt").write_text("Secondary note.\n", encoding="utf-8")

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
                '{"status":"continue","tool":"workspace","action":"inspect_root","params":{"path":"Desktop/819","objective":"Inspect Desktop/819"},"reason":"Need to inspect the folder first."}',
                '{"status":"continue","tool":"knowledge","action":"answer_from_paths","params":{"paths":["Desktop/819"],"question":"Summarize the first file found in this folder."},"reason":"Now answer from the folder contents."}',
                '{"answer":"The first file says the purchase price is $1,250,000.","cited_chunk_ids":["a-contract.txt:1"],"grounded":true,"partial":false}',
                '{"status":"complete","tool":"","action":"","params":{},"reason":"The grounded answer satisfies the request."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "summarize the first file you found in Desktop/819 folder")
    assert "Answer:" in reply
    assert "purchase price is $1,250,000" in reply
    assert "a-contract.txt [a-contract.txt:1]" in reply
    db.close()


def test_tool_loop_continuation_is_generic_not_tool_specific(tmp_path) -> None:
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
                '{"status":"continue","tool":"fake","action":"step_one","params":{},"reason":"Need the first fake step."}',
                '{"status":"continue","tool":"fake","action":"step_two","params":{},"reason":"A second fake step completes the request."}',
                '{"status":"complete","tool":"","action":"","params":{},"reason":"The objective is complete."}',
            ]
        ),
    )
    fake_tool = FakeTool()
    agent.tools.register(fake_tool)

    reply = agent.handle_text("chat-1", "finish the fake task")
    assert "final fake answer" in reply
    assert fake_tool.invocations == [("step_one", {}), ("step_two", {})]
    db.close()


def test_llm_selected_tool_routes_to_automation(tmp_path) -> None:
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
                '{"status":"continue","tool":"automation","action":"create_schedule","params":{"schedule":"every 30m","prompt":"stretch"},"reason":"The user wants a recurring reminder."}',
                '{"status":"complete","tool":"","action":"","params":{},"reason":"The schedule was created successfully."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "remind me every 30 minutes to stretch")
    assert "Created schedule" in reply
    assert "every 30m" in reply
    assert "stretch" in reply
    db.close()


def test_automation_terminal_result_skips_continuation_planner(tmp_path) -> None:
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
                '{"status":"continue","tool":"automation","action":"create_schedule","params":{"schedule":"in 30 minutes","prompt":"stretch"},"reason":"The user wants a one-off reminder."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "remind me in 30 minutes to stretch")
    assert "Created schedule" in reply
    assert "in 30 minutes" in reply
    db.close()
