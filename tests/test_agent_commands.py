import json
from datetime import datetime
from pathlib import Path

from jclaw.ai.agent import AssistantAgent
from jclaw.core.config import Config, DaemonConfig, MemoryConfig, ProviderConfig, TelegramConfig
from jclaw.core.defaults import (
    AGENT_CONTINUE_TOOL_STEPS,
    AGENT_MAX_TOOL_STEPS,
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
    WORKSPACE_AGENT_MAX_TOOL_STEPS,
    WORKSPACE_CONTINUE_TOOL_STEPS,
    WORKSPACE_MAX_INTERNAL_READ_BYTES,
    WORKSPACE_MAX_PATH_ENTRIES,
    WORKSPACE_MAX_PREPARED_DIFF_BYTES,
    WORKSPACE_MAX_STEPS,
    WORKSPACE_SHELL_OUTPUT_CHARS,
    WORKSPACE_SHELL_TIMEOUT_SECONDS,
)
from jclaw.core.db import Database
from jclaw.tools.base import Observation, RuntimeState, ToolContext, ToolExecutionState, ToolLoopFinalizer, ToolLoopState, ToolResult
from jclaw.tools.email.tool import EmailTool


class DummyLLM:
    def chat(self, messages):  # noqa: ANN001
        return "stubbed"


class SequenceLLM:
    def __init__(self, responses) -> None:  # noqa: ANN001
        self._responses = iter(responses)

    def chat(self, messages):  # noqa: ANN001
        return next(self._responses)


class RecordingSequenceLLM:
    def __init__(self, responses) -> None:  # noqa: ANN001
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        return next(self._responses)


class FreshToolReplyLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        flattened = "\n".join(message["content"] for message in messages)
        if "already delegates to _read_text_file_state" in flattened:
            return "stale-history-used"
        if "Tool result:\nFresh file state says caching is not enabled." in flattened:
            return "fresh-tool-result-used"
        return "unexpected"


def _build_agent_for_test(tmp_path, llm) -> tuple[AssistantAgent, Database]:  # noqa: ANN001
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
    return AssistantAgent(config, db, llm), db


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


class FakeBindingTool:
    name = "binding"

    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict]] = []

    def describe(self) -> dict:
        return {
            "name": "binding",
            "description": "Fake tool used to verify runtime-owned parameter binding.",
            "actions": {
                "select": {
                    "description": "Produce a selected message reference.",
                },
                "reply": {
                    "description": "Reply using the selected message reference.",
                },
            },
        }

    def materialize_params(self, action: str, params: dict, runtime) -> dict:  # noqa: ANN001
        materialized = dict(params)
        if action == "reply" and not materialized.get("message_id"):
            message_ref = runtime.artifacts_by_type.get("message_ref", {})
            if isinstance(message_ref, dict) and message_ref.get("message_id"):
                materialized["message_id"] = message_ref["message_id"]
        return materialized

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append((action, dict(params)))
        if action == "select":
            return ToolResult(
                ok=True,
                summary="Selected the target message.",
                data={
                    "artifacts": {
                        "message_ref:selected": {
                            "message_id": "msg-1",
                            "thread_id": "thread-1",
                            "alias": "gmail",
                        }
                    }
                },
            )
        if action == "reply":
            return ToolResult(
                ok=True,
                summary="Drafted the reply.",
                data={"answer": f"reply:{params['message_id']}:{params['body']}"},
            )
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


class FakeLongWorkspaceTool:
    name = "workspace"

    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict]] = []

    def describe(self) -> dict:
        return {
            "name": "workspace",
            "description": "Fake workspace tool used for step-budget tests.",
            "actions": {
                "step_one": {"description": "Workspace step one."},
                "step_two": {"description": "Workspace step two."},
                "step_three": {"description": "Workspace step three."},
                "step_four": {"description": "Workspace step four."},
                "step_five": {"description": "Workspace step five."},
            },
            "supports_followup": True,
        }

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append((action, dict(params)))
        return ToolResult(ok=True, summary=f"Completed {action}.", data={"phase": action})

    def format_result(self, action: str, result: ToolResult) -> str:
        return result.summary


class FakeLongTool:
    name = "longtool"

    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict]] = []

    def describe(self) -> dict:
        return {
            "name": "longtool",
            "description": "Fake general tool used for continuation tests.",
            "actions": {
                "step_one": {"description": "General step one."},
                "step_two": {"description": "General step two."},
                "step_three": {"description": "General step three."},
                "step_four": {"description": "General step four."},
                "step_five": {"description": "General step five."},
            },
            "supports_followup": True,
        }

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append((action, dict(params)))
        return ToolResult(ok=True, summary=f"Completed {action}.", data={"phase": action})

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
            return ToolResult(
                ok=True,
                summary="Searched the web.",
                data={"session_id": "sess_browser_loop", "url": "https://example.com/result"},
                loop_state=ToolLoopState(
                    state={"session_id": "sess_browser_loop"},
                    finalizer=ToolLoopFinalizer(
                        action="close_session",
                        params={"session_id": "sess_browser_loop"},
                    ),
                ),
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


class FakeScratchpadTool:
    name = "scratchpad"

    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict]] = []

    def describe(self) -> dict:
        return {
            "name": "scratchpad",
            "description": "Fake loop-state tool used for generic contract tests.",
            "actions": {
                "start": {"description": "Start a scratchpad loop."},
                "continue_work": {"description": "Continue the scratchpad loop."},
                "release": {"description": "Release the scratchpad resource."},
            },
            "supports_followup": True,
        }

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append((action, dict(params)))
        if action == "start":
            assert ctx.execution is not None
            return ToolResult(
                ok=True,
                summary="Scratchpad started.",
                data={"resource_id": "scratch-1"},
                loop_state=ToolLoopState(
                    state={"resource_id": "scratch-1"},
                    finalizer=ToolLoopFinalizer(
                        action="release",
                        params={"resource_id": "scratch-1"},
                    ),
                ),
            )
        if action == "continue_work":
            assert ctx.execution is not None
            assert ctx.execution.tool_state["scratchpad"]["resource_id"] == "scratch-1"
            return ToolResult(
                ok=True,
                summary="Scratchpad continued.",
                data={"resource_id": "scratch-1"},
            )
        if action == "release":
            assert params == {"resource_id": "scratch-1"}
            return ToolResult(
                ok=True,
                summary="Scratchpad released.",
                data={"resource_id": "scratch-1"},
                loop_state=ToolLoopState(clear=True),
            )
        raise AssertionError(f"unexpected action: {action}")

    def format_result(self, action: str, result: ToolResult) -> str:
        return result.summary


class FakeBrowserObjectiveTool:
    name = "browser"

    def describe(self) -> dict:
        return {
            "name": "browser",
            "description": "Fake browser tool used for route tests.",
            "prefer_direct_result": True,
            "actions": {
                "run_objective": {"description": "Run a browser objective."},
            },
        }

    def invoke(self, action: str, params: dict, ctx: ToolContext) -> ToolResult:
        assert action == "run_objective"
        assert params == {"objective": "open example.com", "start_url": "https://example.com"}
        return ToolResult(
            ok=True,
            summary="Opened example.com.",
            data={
                "url": "https://example.com",
                "title": "Example Domain",
                "text": "Example Domain",
                "allow_tool_followup": False,
            },
        )

    def format_result(self, action: str, result: ToolResult) -> str:
        return f"{result.summary}\nURL: {result.data['url']}\nTitle: {result.data['title']}"


class FakeAbigailGmailClient:
    def __init__(self) -> None:
        self.messages = [
            {
                "id": "msg-1",
                "thread_id": "thread-1",
                "subject": "15 minute chat next week",
                "from": "Abigail Clifford <aclifford@candidatelabs.com>",
                "to": "guanw0826@gmail.com",
                "cc": "",
                "date": "Fri, 24 Apr 2026 13:49:06 +0000",
                "snippet": "Initial invite for our chat.",
                "labels": ["INBOX"],
                "unread": False,
                "text_body": "Initial invite for our chat.",
                "html_body": "",
                "message_id_header": "<msg-1@example.com>",
                "references": "",
                "in_reply_to": "",
            },
            {
                "id": "msg-2",
                "thread_id": "thread-2",
                "subject": "Re: 15 minute chat next week",
                "from": "Abigail Clifford <aclifford@candidatelabs.com>",
                "to": "guanw0826@gmail.com",
                "cc": "",
                "date": "Fri, 24 Apr 2026 18:44:52 +0000",
                "snippet": "New calendar invite for our chat next week.",
                "labels": ["INBOX"],
                "unread": False,
                "text_body": "New calendar invite for our chat next week.",
                "html_body": "",
                "message_id_header": "<msg-2@example.com>",
                "references": "",
                "in_reply_to": "",
            },
        ]
        self.last_draft: dict[str, str] | None = None

    def list_unread(self, alias: str, *, max_results: int = 10) -> list[dict]:
        return self.messages[:max_results]

    def search_messages(self, alias: str, *, query: str, max_results: int = 10) -> list[dict]:
        return self.messages[:max_results]

    def get_message(self, alias: str, *, message_id: str) -> dict:
        for item in self.messages:
            if item["id"] == message_id:
                return item
        raise AssertionError(f"unexpected message_id: {message_id}")

    def get_thread(self, alias: str, *, thread_id: str) -> dict:
        return {"thread_id": thread_id, "messages": [item for item in self.messages if item["thread_id"] == thread_id]}

    def draft_reply(self, alias: str, *, message: dict, body_text: str) -> dict:
        self.last_draft = {
            "alias": alias,
            "message_id": message["id"],
            "body": body_text,
            "to": str(message.get("reply_to_address") or message["from"]),
        }
        return {
            "draft_id": "draft-1",
            "message_id": "draft-msg-1",
            "thread_id": message["thread_id"],
            "subject": f"Re: {message['subject']}",
            "to": str(message.get("reply_to_address") or message["from"]),
            "body_preview": body_text,
        }


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
                '{"type":"tool_call","tool":"memory","action":"remember_fact","params":{"key":"favorite_color","value":"blue"},"reason":"The user asked to remember a preference."}',
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
                '{"type":"tool_call","tool":"memory","action":"search_memories","params":{"query":"favorite color"},"reason":"The user is asking what is remembered."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "what do you remember about my favorite color")
    assert "favorite_color = blue" in reply
    db.close()


def test_llm_selected_tool_routes_to_permissions_list_grants(tmp_path) -> None:
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
    db.upsert_grant("/Users/Jude", ("read",), "chat-1")
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"permissions","action":"list_grants","params":{},"reason":"The user is asking what access has been granted."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "what are all the granted access right now")
    assert "Active grants:" in reply
    assert "/Users/Jude [read]" in reply
    db.close()


def test_llm_selected_tool_routes_to_email_list_accounts(tmp_path) -> None:
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
    db.upsert_email_account(
        alias="gmail",
        provider="gmail",
        email_address="me@example.com",
        scopes=("https://www.googleapis.com/auth/gmail.readonly",),
        status="connected",
        metadata={},
    )
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"email","action":"list_accounts","params":{},"reason":"The user is asking which mail accounts are connected."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "what email accounts are connected")
    assert "Connected email accounts:" in reply
    assert "gmail: me@example.com" in reply
    db.close()


def test_email_controller_can_search_select_and_answer(tmp_path) -> None:
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
    db.upsert_email_account(
        alias="gmail",
        provider="gmail",
        email_address="guanw0826@gmail.com",
        scopes=(
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
        ),
        status="connected",
        metadata={},
    )
    fake_client = FakeAbigailGmailClient()
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"email","action":"search_messages","params":{"query":"from:abigail OR to:abigail","max_results":5},"reason":"Search for the recent email thread first."}',
                '{"type":"tool_call","tool":"email","action":"select_message","params":{"selection":"latest"},"reason":"Select the latest message from the search results."}',
                '{"type":"answer","tool":"","action":"","params":{},"answer":"The last email involving Abigail was on Fri, 24 Apr 2026 18:44:52 +0000.","reason":"The selected message already contains the date."}',
            ]
        ),
    )
    agent.tools._tools["email"] = EmailTool(  # noqa: SLF001
        db,
        oauth_client_path=Path("/tmp/client.json"),
        token_dir=tmp_path / "tokens",
        default_account_alias="gmail",
        get_client=lambda alias: fake_client,
    )

    reply = agent.handle_text("chat-1", "what's the last time i emailed abigail?")

    assert reply == "The last email involving Abigail was on Fri, 24 Apr 2026 18:44:52 +0000."
    db.close()


def test_email_controller_repairs_prose_answer_instead_of_falling_back_to_raw_search_dump(tmp_path) -> None:
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
    db.upsert_email_account(
        alias="gmail",
        provider="gmail",
        email_address="guanw0826@gmail.com",
        scopes=(
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
        ),
        status="connected",
        metadata={},
    )
    fake_client = FakeAbigailGmailClient()
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"email","action":"search_messages","params":{"query":"from:aclifford@candidatelabs.com OR to:aclifford@candidatelabs.com","max_results":10},"reason":"Search for the most recent email exchange with Abigail."}',
                "The last email involving Abigail was on Fri, 24 Apr 2026 18:44:52 +0000.",
                '{"type":"answer","tool":"","action":"","params":{},"answer":"The last email involving Abigail was on Fri, 24 Apr 2026 18:44:52 +0000.","reason":"The search results already contain the answer."}',
            ]
        ),
    )
    agent.tools._tools["email"] = EmailTool(  # noqa: SLF001
        db,
        oauth_client_path=Path("/tmp/client.json"),
        token_dir=tmp_path / "tokens",
        default_account_alias="gmail",
        get_client=lambda alias: fake_client,
    )

    reply = agent.handle_text("chat-1", "what's the last time i emailed abigail?")

    assert reply == "The last email involving Abigail was on Fri, 24 Apr 2026 18:44:52 +0000."
    assert "Found 10 matching email(s)" not in reply
    db.close()


def test_email_controller_can_search_select_and_draft_reply(tmp_path) -> None:
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
    db.upsert_email_account(
        alias="gmail",
        provider="gmail",
        email_address="guanw0826@gmail.com",
        scopes=(
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
        ),
        status="connected",
        metadata={},
    )
    fake_client = FakeAbigailGmailClient()
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"email","action":"search_messages","params":{"query":"from:abigail OR to:abigail","max_results":5},"reason":"Find the relevant thread first."}',
                '{"type":"tool_call","tool":"email","action":"select_message","params":{"selection":"latest"},"reason":"Pick the latest Abigail thread."}',
                '{"type":"tool_call","tool":"email","action":"draft_reply","params":{"body":"Hi Abigail, looking forward to our chat next week."},"reason":"Draft the requested reply."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"The draft is ready."}',
            ]
        ),
    )
    agent.tools._tools["email"] = EmailTool(  # noqa: SLF001
        db,
        oauth_client_path=Path("/tmp/client.json"),
        token_dir=tmp_path / "tokens",
        default_account_alias="gmail",
        get_client=lambda alias: fake_client,
    )

    reply = agent.handle_text("chat-1", "Draft reply with Abigail I scheduled sometime next week to chat")

    assert "Created Gmail draft draft-1." in reply
    assert fake_client.last_draft == {
        "alias": "gmail",
        "message_id": "msg-2",
        "body": "Hi Abigail, looking forward to our chat next week.",
        "to": "Abigail Clifford <aclifford@candidatelabs.com>",
    }
    db.close()


def test_email_controller_ignores_person_name_in_alias_and_uses_connected_mailbox(tmp_path) -> None:
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
    db.upsert_email_account(
        alias="gmail",
        provider="gmail",
        email_address="guanw0826@gmail.com",
        scopes=(
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
        ),
        status="connected",
        metadata={},
    )
    fake_client = FakeAbigailGmailClient()
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"email","action":"search_messages","params":{"alias":"abigail","query":"from:abigail OR to:abigail","max_results":10},"reason":"Find messages involving Abigail first."}',
                '{"type":"tool_call","tool":"email","action":"select_message","params":{"selection":"latest"},"reason":"Pick the latest Abigail thread."}',
                '{"type":"tool_call","tool":"email","action":"draft_reply","params":{"body":"Hi Abigail, looking forward to our chat next week."},"reason":"Draft the requested reply."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"The draft is ready."}',
            ]
        ),
    )
    agent.tools._tools["email"] = EmailTool(  # noqa: SLF001
        db,
        oauth_client_path=Path("/tmp/client.json"),
        token_dir=tmp_path / "tokens",
        default_account_alias="gmail",
        get_client=lambda alias: fake_client,
    )

    reply = agent.handle_text("chat-1", "Draft reply with Abigail I scheduled sometime next week to chat")

    assert "Created Gmail draft draft-1." in reply
    assert fake_client.last_draft == {
        "alias": "gmail",
        "message_id": "msg-2",
        "body": "Hi Abigail, looking forward to our chat next week.",
        "to": "Abigail Clifford <aclifford@candidatelabs.com>",
    }
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
                '{"type":"tool_call","tool":"browser","action":"run_objective","params":{"objective":"open example.com","start_url":"https://example.com"},"reason":"The user wants browser help."}',
            ]
        ),
    )
    agent.tools._tools["browser"] = FakeBrowserObjectiveTool()  # noqa: SLF001

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
                '{"type":"complete","tool":"","action":"","params":{},"reason":"No tool needed."}',
                "Normal chat reply.",
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "say hello")
    assert reply == "Normal chat reply."
    db.close()


def test_controller_can_answer_without_tool_use(tmp_path) -> None:
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
                '{"type":"answer","tool":"","action":"","params":{},"answer":"Hello.","reason":"No tool is needed."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "say hello")

    assert reply == "Hello."
    db.close()


def test_controller_can_answer_from_prior_observation(tmp_path) -> None:
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
                '{"type":"tool_call","tool":"fake","action":"step_one","params":{},"reason":"Gather the intermediate observation first."}',
                '{"type":"answer","tool":"","action":"","params":{},"answer":"The current phase is intermediate.","reason":"The prior observation already answers the request."}',
            ]
        ),
    )
    agent.tools.register(FakeTool())

    reply = agent.handle_text("chat-1", "what phase are we in")

    assert reply == "The current phase is intermediate."
    db.close()


def test_controller_can_block_without_tool_use(tmp_path) -> None:
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
                '{"type":"blocked","tool":"","action":"","params":{},"reason":"I need clarification about which Abigail thread you mean."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "draft a reply to Abigail")

    assert reply == "I need clarification about which Abigail thread you mean."
    db.close()


def test_controller_prompt_includes_structured_runtime_observations(tmp_path) -> None:
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
    llm = RecordingSequenceLLM(
        [
            '{"type":"tool_call","tool":"fake","action":"step_one","params":{},"reason":"Gather the intermediate observation first."}',
            '{"type":"blocked","tool":"","action":"","params":{},"reason":"Inspection complete."}',
        ]
    )
    agent = AssistantAgent(config, db, llm)
    agent.tools.register(FakeTool())

    reply = agent.handle_text("chat-1", "inspect the current phase")

    assert reply == "Inspection complete."
    continuation_payload = json.loads(llm.calls[-1][-1]["content"])
    controller_state = continuation_payload["controller_state"]
    assert controller_state["step_count"] == 1
    assert controller_state["latest_observation"]["summary"] == "Completed fake step one."
    assert controller_state["observations"][0]["tool"] == "fake"
    assert controller_state["observations"][0]["action"] == "step_one"
    assert controller_state["observations"][0]["observation"]["data_preview"]["phase"] == "intermediate"
    assert controller_state["artifact_types"] == []
    db.close()


def test_controller_state_limits_observations_to_latest_five(tmp_path) -> None:
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
    runtime = RuntimeState(request="run a long tool flow")
    steps: list[dict[str, object]] = []
    for index in range(6):
        result = ToolResult(ok=True, summary=f"step {index + 1}", data={"phase": index + 1})
        runtime.append(Observation.from_tool_result(result))
        steps.append(
            {
                "tool": "fake",
                "action": f"step_{index + 1}",
                "reason": f"reason {index + 1}",
                "result": result,
            }
        )

    controller_state = agent._controller_state_for_prompt(steps, runtime)  # noqa: SLF001

    assert controller_state["step_count"] == 6
    assert len(controller_state["observations"]) == 5
    assert [item["step"] for item in controller_state["observations"]] == [2, 3, 4, 5, 6]
    assert controller_state["latest_observation"]["summary"] == "step 6"


def test_controller_state_includes_authoritative_local_datetime(tmp_path) -> None:
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
    fixed_now = datetime.fromisoformat("2026-04-26T12:34:56-04:00")
    agent._controller_now = lambda: fixed_now  # type: ignore[method-assign]  # noqa: SLF001

    controller_state = agent._controller_state_for_prompt([], RuntimeState(request="schedule reminder"))  # noqa: SLF001

    assert controller_state["current_local_time"] == "2026-04-26T12:34:56-04:00"
    assert controller_state["current_local_date"] == "2026-04-26"
    assert controller_state["current_local_timezone"] == "UTC-04:00"
    db.close()


def test_controller_state_preserves_workspace_file_and_diff_artifact_previews(tmp_path) -> None:
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
    runtime = RuntimeState(request="inspect source code")
    long_content = "x" * 5000
    long_diff = "y" * 5000
    result = ToolResult(
        ok=True,
        summary="Read file and diff.",
        data={
            "content": long_content,
            "diff": long_diff,
            "artifacts": {
                "workspace_file:latest": {
                    "root_path": "/repo",
                    "target_path": "/repo/app.py",
                    "kind": "file",
                    "start_line": 1,
                    "end_line": 100,
                    "line_count": 100,
                    "content": long_content,
                    "truncated": False,
                    "git_root": "/repo",
                },
                "workspace_diff:latest": {
                    "root_path": "/repo",
                    "target_path": "/repo/app.py",
                    "git_root": "/repo",
                    "status": "M app.py",
                    "diff": long_diff,
                    "has_unstaged": True,
                    "has_staged": False,
                },
            },
        },
    )
    runtime.append(
        Observation.from_tool_result(
            result,
            controller_contract=agent.tools.get("workspace").describe()["controller_contract"],
        )
    )
    controller_state = agent._controller_state_for_prompt(  # noqa: SLF001
        [{"tool": "workspace", "action": "read_file", "reason": "Inspect file", "result": result}],
        runtime,
    )

    file_preview = controller_state["artifacts_by_type"]["workspace_file"]["content"]
    diff_preview = controller_state["artifacts_by_type"]["workspace_diff"]["diff"]
    assert len(file_preview) > 220
    assert len(file_preview) == 4003
    assert len(diff_preview) > 220
    assert len(diff_preview) == 4003
    latest_preview = controller_state["latest_observation"]["data_preview"]["content"]
    assert len(latest_preview) > 220
    assert len(latest_preview) == 4003
    db.close()


def test_controller_state_preserves_workspace_patch_artifact_preview(tmp_path) -> None:
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
    runtime = RuntimeState(request="patch source code")
    long_patch = "z" * 5000
    result = ToolResult(
        ok=True,
        summary="Applied patch.",
        data={
            "touched_files": ["app.py"],
            "diff_preview": long_patch,
            "artifacts": {
                "workspace_patch:latest": {
                    "root_path": "/repo",
                    "target_path": "/repo/app.py",
                    "operation": "apply_patch",
                    "touched_files": ["app.py"],
                    "diff": long_patch,
                }
            },
        },
    )
    runtime.append(Observation.from_tool_result(result))

    controller_state = agent._controller_state_for_prompt(  # noqa: SLF001
        [{"tool": "workspace", "action": "apply_patch", "reason": "Patch file", "result": result}],
        runtime,
    )

    patch_preview = controller_state["artifacts_by_type"]["workspace_patch"]["diff"]
    assert len(patch_preview) > 220
    assert len(patch_preview) == 4003
    db.close()


def test_controller_state_preserves_workspace_command_result_preview(tmp_path) -> None:
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
    runtime = RuntimeState(request="run tests")
    long_stdout = "o" * 5000
    long_stderr = "e" * 5000
    result = ToolResult(
        ok=False,
        summary="Command failed: python -m pytest",
        data={
            "command": "python -m pytest",
            "cwd": "/repo",
            "exit_code": 1,
            "stdout": long_stdout,
            "stderr": long_stderr,
            "artifacts": {
                "workspace_command_result:latest": {
                    "root_path": "/repo",
                    "command": "python -m pytest",
                    "cwd": "/repo",
                    "exit_code": 1,
                    "stdout": long_stdout,
                    "stderr": long_stderr,
                    "ok": False,
                }
            },
        },
    )
    runtime.append(Observation.from_tool_result(result))

    controller_state = agent._controller_state_for_prompt(  # noqa: SLF001
        [{"tool": "workspace", "action": "run_command", "reason": "Run tests", "result": result}],
        runtime,
    )

    command_preview = controller_state["artifacts_by_type"]["workspace_command_result"]
    assert len(command_preview["stdout"]) > 220
    assert len(command_preview["stdout"]) == 4003
    assert len(command_preview["stderr"]) > 220
    assert len(command_preview["stderr"]) == 4003
    db.close()


def test_tool_result_for_controller_includes_workspace_read_fields(tmp_path) -> None:
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
    result = ToolResult(
        ok=True,
        summary="Read source file.",
        data={
            "root_path": "/repo",
            "target_path": "/repo/app.py",
            "content": "print('hello')\n",
            "line_count": 1,
            "start_line": 1,
            "end_line": 1,
            "char_count": 15,
            "bytes_read": 15,
            "truncated": False,
            "git_root": "/repo",
            "status": "M app.py",
            "diff": "### Unstaged\n...",
            "has_unstaged": True,
            "has_staged": False,
        },
    )

    controller_view = agent._tool_result_for_controller("workspace", result)  # noqa: SLF001

    assert controller_view["content"] == "print('hello')\n"
    assert controller_view["line_count"] == 1
    assert controller_view["start_line"] == 1
    assert controller_view["end_line"] == 1
    assert controller_view["bytes_read"] == 15
    assert controller_view["git_root"] == "/repo"
    assert controller_view["diff"] == "### Unstaged\n..."
    assert controller_view["has_unstaged"] is True
    assert controller_view["has_staged"] is False
    db.close()


def test_tool_catalog_and_controller_prompt_bias_workspace_line_requests_to_read_snippet(tmp_path) -> None:
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

    class PromptCapturingLLM(SequenceLLM):
        def __init__(self) -> None:
            super().__init__(
                [
                    '{"type":"tool_call","tool":"workspace","action":"read_snippet","params":{"path":"src/jclaw/tools/workspace/tool.py","start_line":1290,"end_line":1415},"answer":"","reason":"The user requested a specific line range."}'
                ]
            )
            self.last_system_prompt = ""

        def chat(self, messages: list[dict[str, str]]) -> str:  # type: ignore[override]
            self.last_system_prompt = messages[0]["content"]
            return super().chat(messages)

    llm = PromptCapturingLLM()
    agent = AssistantAgent(config, db, llm)

    catalog = json.loads(agent._tool_catalog_for_prompt(agent.tools.list_tools()))  # noqa: SLF001
    workspace_tool = next(item for item in catalog if item["name"] == "workspace")
    read_file = workspace_tool["actions"]["read_file"]["description"]
    read_snippet = workspace_tool["actions"]["read_snippet"]["description"]
    controller_guidance = workspace_tool["controller_guidance"]

    assert "controller_contract" not in workspace_tool
    assert "Do not use this when the user asks for explicit line numbers" in read_file
    assert "Use this when the user asks for explicit line numbers" in read_snippet
    assert "switch to mutation as soon as the edit site is known" in controller_guidance
    assert "prefer apply_patch over more reads" in controller_guidance
    assert "After a code mutation, prefer a verification step such as run_command" in controller_guidance
    assert "prefer revert_last_change instead of inferring the target from git diff" in controller_guidance
    assert "prefer redo_last_change" in controller_guidance

    decision = agent._decide_next_tool_step(  # noqa: SLF001
        "chat-1",
        "Show me lines 1290 to 1415 of src/jclaw/tools/workspace/tool.py",
        user_name="tester",
        steps=[],
        runtime=RuntimeState(request="Show me lines 1290 to 1415 of src/jclaw/tools/workspace/tool.py"),
    )

    assert "prefer the focused range read" in llm.last_system_prompt
    assert decision is not None
    assert decision.tool == "workspace"
    assert decision.action == "read_snippet"
    assert decision.params["start_line"] == 1290
    assert decision.params["end_line"] == 1415
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
                '{"type":"tool_call","tool":"exploding","action":"boom","params":{},"reason":"Exercise tool failure handling."}',
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
                '{"type":"tool_call","tool":"browser","action":"search_web","params":{"query":"millburn contractors"},"reason":"Search first."}',
                '{"type":"tool_call","tool":"browser","action":"read_page","params":{},"reason":"Read the result page."}',
                '{"type":"tool_call","tool":"browser","action":"extract","params":{"fields":{"name":"contractor name"}},"reason":"Extract the contractor name."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"Enough information gathered."}',
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


def test_generic_tool_loop_state_contract_supports_non_browser_finalizers(tmp_path) -> None:
    agent, db = _build_agent_for_test(
        tmp_path,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"scratchpad","action":"start","params":{},"reason":"Allocate state."}',
                '{"type":"tool_call","tool":"scratchpad","action":"continue_work","params":{},"reason":"Use saved loop state."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"Enough work completed."}',
                "Scratchpad work completed.",
            ]
        ),
    )
    scratchpad = FakeScratchpadTool()
    agent.tools._tools["scratchpad"] = scratchpad  # noqa: SLF001

    reply = agent.handle_text("chat-1", "do scratchpad work")

    assert reply == "Scratchpad work completed."
    assert scratchpad.invocations == [
        ("start", {}),
        ("continue_work", {}),
        ("release", {"resource_id": "scratch-1"}),
    ]
    db.close()


def test_workspace_tool_loop_uses_higher_step_budget_than_general_tasks(tmp_path) -> None:
    agent, db = _build_agent_for_test(
        tmp_path,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"workspace","action":"step_one","params":{},"reason":"Read first."}',
                '{"type":"tool_call","tool":"workspace","action":"step_two","params":{},"reason":"Read second."}',
                '{"type":"tool_call","tool":"workspace","action":"step_three","params":{},"reason":"Read third."}',
                '{"type":"tool_call","tool":"workspace","action":"step_four","params":{},"reason":"Read fourth."}',
                '{"type":"tool_call","tool":"workspace","action":"step_five","params":{},"reason":"Read fifth."}',
                '{"type":"answer","tool":"","action":"","params":{},"answer":"Done after five workspace steps.","reason":"Enough inspection completed."}',
            ]
        ),
    )
    workspace = FakeLongWorkspaceTool()
    agent.tools._tools["workspace"] = workspace  # noqa: SLF001

    reply = agent.handle_text("chat-1", "inspect and then answer")

    assert reply == "Done after five workspace steps."
    assert [action for action, _ in workspace.invocations] == [
        "step_one",
        "step_two",
        "step_three",
        "step_four",
        "step_five",
    ]
    db.close()


def test_general_tool_loop_exhaustion_requires_explicit_continue(tmp_path) -> None:
    agent, db = _build_agent_for_test(
        tmp_path,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"longtool","action":"step_one","params":{},"reason":"General step one."}',
                '{"type":"tool_call","tool":"longtool","action":"step_two","params":{},"reason":"General step two."}',
                '{"type":"tool_call","tool":"longtool","action":"step_three","params":{},"reason":"General step three."}',
                '{"type":"tool_call","tool":"longtool","action":"step_four","params":{},"reason":"General step four."}',
                '{"type":"tool_call","tool":"longtool","action":"step_five","params":{},"reason":"General step five."}',
            ]
        ),
    )
    tool = FakeLongTool()
    agent.tools._tools["longtool"] = tool  # noqa: SLF001

    reply = agent.handle_text("chat-1", "do a long general task")

    assert f"({AGENT_MAX_TOOL_STEPS} tool steps)" in reply
    assert f"{AGENT_CONTINUE_TOOL_STEPS} more tool steps" in reply
    assert [action for action, _ in tool.invocations] == [
        "step_one",
        "step_two",
        "step_three",
        "step_four",
    ]
    db.close()


def test_continue_without_pending_tool_loop_returns_clear_message(tmp_path) -> None:
    agent, db = _build_agent_for_test(tmp_path, DummyLLM())

    reply = agent.handle_text("chat-1", "continue")

    assert reply == "There is no paused tool run waiting for continuation."
    db.close()


def test_workspace_tool_loop_continue_adds_more_budget_after_exhaustion(tmp_path) -> None:
    agent, db = _build_agent_for_test(
        tmp_path,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"workspace","action":"step_one","params":{},"reason":"Read first."}',
                '{"type":"tool_call","tool":"workspace","action":"step_two","params":{},"reason":"Read second."}',
                '{"type":"tool_call","tool":"workspace","action":"step_three","params":{},"reason":"Read third."}',
                '{"type":"tool_call","tool":"workspace","action":"step_four","params":{},"reason":"Read fourth."}',
                '{"type":"tool_call","tool":"workspace","action":"step_five","params":{},"reason":"Read fifth."}',
                '{"type":"answer","tool":"","action":"","params":{},"answer":"Done after explicit continuation.","reason":"Enough inspection completed."}',
            ]
        ),
    )
    workspace = FakeLongWorkspaceTool()
    agent.tools._tools["workspace"] = workspace  # noqa: SLF001
    agent.config.workspace.agent_max_tool_steps = 4

    first_reply = agent.handle_text("chat-1", "inspect and then answer")
    second_reply = agent.handle_text("chat-1", "continue")

    assert "(4 tool steps)" in first_reply
    assert f"{WORKSPACE_CONTINUE_TOOL_STEPS} more tool steps" in first_reply
    assert second_reply == "Done after explicit continuation."
    assert [action for action, _ in workspace.invocations] == [
        "step_one",
        "step_two",
        "step_three",
        "step_four",
        "step_five",
    ]
    db.close()


def test_compose_tool_reply_uses_latest_tool_result_not_stale_history(tmp_path) -> None:
    agent, db = _build_agent_for_test(tmp_path, FreshToolReplyLLM())
    agent.tools._tools["fake"] = FakeTool()  # noqa: SLF001
    db.store_message("chat-1", "assistant", "_read_file already delegates to _read_text_file_state with caching enabled.")

    reply = agent._compose_tool_reply(  # noqa: SLF001
        "chat-1",
        "let's enable caching or memoization for _read_file in src/jclaw/tools/workspace/tool.py",
        user_name="tester",
        decision={"tool": "fake", "action": "step_one", "reason": "Inspect file", "params": {}},
        result=ToolResult(
            ok=True,
            summary="Fresh file state says caching is not enabled.",
            data={"phase": "intermediate"},
        ),
    )

    assert reply == "fresh-tool-result-used"
    db.close()


def test_apply_tool_loop_state_sets_state_and_finalizer(tmp_path) -> None:
    agent, db = _build_agent_for_test(tmp_path, DummyLLM())
    execution = ToolExecutionState()
    result = ToolResult(
        ok=True,
        summary="Allocated resource.",
        loop_state=ToolLoopState(
            state={"resource_id": "scratch-1"},
            finalizer=ToolLoopFinalizer(action="release", params={"resource_id": "scratch-1"}),
        ),
    )

    agent._apply_tool_loop_state(execution, "scratchpad", result)  # noqa: SLF001

    assert execution.tool_state["scratchpad"] == {"resource_id": "scratch-1"}
    assert execution.finalizers["scratchpad"].action == "release"
    assert execution.finalizers["scratchpad"].params == {"resource_id": "scratch-1"}
    db.close()


def test_apply_tool_loop_state_clears_requested_state_and_finalizer(tmp_path) -> None:
    agent, db = _build_agent_for_test(tmp_path, DummyLLM())
    execution = ToolExecutionState(
        tool_state={"scratchpad": {"resource_id": "scratch-1"}},
        finalizers={"scratchpad": ToolLoopFinalizer(action="release", params={"resource_id": "scratch-1"})},
    )

    agent._apply_tool_loop_state(  # noqa: SLF001
        execution,
        "scratchpad",
        ToolResult(ok=True, summary="Released resource.", loop_state=ToolLoopState(clear=True)),
    )

    assert "scratchpad" not in execution.tool_state
    assert "scratchpad" not in execution.finalizers
    db.close()


def test_apply_tool_loop_state_can_clear_only_finalizer_without_touching_state(tmp_path) -> None:
    agent, db = _build_agent_for_test(tmp_path, DummyLLM())
    execution = ToolExecutionState(
        tool_state={"browser": {"session_id": "sess-1"}},
        finalizers={"browser": ToolLoopFinalizer(action="close_session", params={"session_id": "sess-1"})},
    )

    agent._apply_tool_loop_state(  # noqa: SLF001
        execution,
        "browser",
        ToolResult(ok=True, summary="Session should stay open.", loop_state=ToolLoopState(clear_finalizer=True)),
    )

    assert execution.tool_state["browser"] == {"session_id": "sess-1"}
    assert "browser" not in execution.finalizers
    db.close()


def test_apply_tool_loop_state_ignores_missing_loop_state(tmp_path) -> None:
    agent, db = _build_agent_for_test(tmp_path, DummyLLM())
    execution = ToolExecutionState(
        tool_state={"browser": {"session_id": "sess-1"}},
        finalizers={"browser": ToolLoopFinalizer(action="close_session", params={"session_id": "sess-1"})},
    )

    agent._apply_tool_loop_state(  # noqa: SLF001
        execution,
        "browser",
        ToolResult(ok=True, summary="No loop-state change."),
    )

    assert execution.tool_state["browser"] == {"session_id": "sess-1"}
    assert execution.finalizers["browser"].action == "close_session"
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
    assert config.workspace.agent_max_tool_steps == WORKSPACE_AGENT_MAX_TOOL_STEPS
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


def test_workspace_write_file_preview_pauses_until_approval(tmp_path) -> None:
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
                '{"type":"tool_call","tool":"workspace","action":"write_file","params":{"path":"app.py","content":"print(\\"goodbye\\")\\n","objective":"Replace app.py with the updated greeting."},"reason":"Local code change requested."}',
            ]
        ),
    )

    grant_request_reply = agent.handle_text("chat-1", "update app.py")
    assert "Approval required" in grant_request_reply
    request_id = grant_request_reply.split("Use /approve ", 1)[1].split(" ", 1)[0]
    assert target.read_text(encoding="utf-8") == "print('hello')\n"

    approval_reply = agent.handle_text("chat-1", f"/approve {request_id}")
    assert "Granted write, read access" in approval_reply or "Granted read, write access" in approval_reply
    assert "Prepared an overwrite preview" in approval_reply
    preview_request_id = approval_reply.split("Request: ", 1)[1].splitlines()[0]

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


def test_workspace_abort_request_uses_request_continuation(tmp_path) -> None:
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

    abort_reply = agent.handle_text("chat-1", f"/abort {preview.data['request_id']}")
    assert f"Aborted request {preview.data['request_id']}." in abort_reply
    assert target.exists()
    assert not (repo_root / "renamed.txt").exists()
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
                '{"type":"tool_call","tool":"workspace","action":"inspect_root","params":{"path":".","objective":"Inspect repo root"},"reason":"Local workspace inspection requested."}',
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
                '{"type":"tool_call","tool":"workspace","action":"find_files","params":{"path":".","pattern":"*.py"},"reason":"Find python files."}',
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
                '{"type":"tool_call","tool":"workspace","action":"inspect_root","params":{"path":".","objective":"Inspect repo root"},"reason":"Local workspace inspection requested."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"A plain directory listing satisfies the request."}',
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
                '{"type":"tool_call","tool":"workspace","action":"inspect_root","params":{"path":".","objective":"Inspect repo root"},"reason":"Local workspace inspection requested."}',
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
                '{"type":"tool_call","tool":"fake","action":"step_one","params":{},"reason":"Need the first fake step."}',
                '{"type":"tool_call","tool":"fake","action":"step_two","params":{},"reason":"A second fake step completes the request."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"The objective is complete."}',
            ]
        ),
    )
    fake_tool = FakeTool()
    agent.tools.register(fake_tool)

    reply = agent.handle_text("chat-1", "finish the fake task")
    assert "final fake answer" in reply
    assert fake_tool.invocations == [("step_one", {}), ("step_two", {})]
    db.close()


def test_tool_loop_materializes_params_from_runtime_artifacts(tmp_path) -> None:
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
                '{"type":"tool_call","tool":"binding","action":"select","params":{},"reason":"Select the message first."}',
                '{"type":"tool_call","tool":"binding","action":"reply","params":{"body":"hello"},"reason":"Reply using the selected message."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"The reply draft is complete."}',
            ]
        ),
    )
    binding_tool = FakeBindingTool()
    agent.tools.register(binding_tool)

    reply = agent.handle_text("chat-1", "reply to the selected message")

    assert "reply:msg-1:hello" in reply
    assert binding_tool.invocations == [
        ("select", {}),
        ("reply", {"body": "hello", "message_id": "msg-1"}),
    ]
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
                '{"type":"tool_call","tool":"automation","action":"create_schedule","params":{"when":{"kind":"interval","interval_seconds":1800},"prompt":"stretch"},"reason":"The user wants a recurring reminder."}',
                '{"type":"complete","tool":"","action":"","params":{},"reason":"The schedule was created successfully."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "remind me every 30 minutes to stretch")
    assert "Created schedule" in reply
    assert "interval:1800" in reply
    assert "stretch" in reply
    db.close()


def test_automation_terminal_result_skips_continuation_controller_turn(tmp_path) -> None:
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
                '{"type":"tool_call","tool":"automation","action":"create_schedule","params":{"when":{"kind":"once","interval_seconds":1800},"prompt":"stretch"},"reason":"The user wants a one-off reminder."}',
            ]
        ),
    )

    reply = agent.handle_text("chat-1", "remind me in 30 minutes to stretch")
    assert "Created schedule" in reply
    assert "once:1800" in reply
    db.close()
def test_automation_prompt_catalog_prefers_structured_when_only(tmp_path) -> None:
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

    catalog = json.loads(agent._tool_catalog_for_prompt(agent.tools.list_tools()))  # noqa: SLF001
    automation = next(item for item in catalog if item["name"] == "automation")
    create_schema = automation["actions"]["create_schedule"]["input_schema"]["properties"]
    update_schema = automation["actions"]["update_schedule"]["input_schema"]["properties"]

    assert "when" in create_schema
    assert "schedule" not in create_schema
    assert "when" in update_schema
    assert "schedule" not in update_schema
    db.close()
