from __future__ import annotations

from pathlib import Path
from typing import Any

from jclaw.ai.agent import AssistantAgent
from jclaw.core.config import (
    AutomationConfig,
    BrowserConfig,
    Config,
    DaemonConfig,
    EmailConfig,
    GoogleDocsConfig,
    KnowledgeConfig,
    MemoryConfig,
    NotionConfig,
    ProviderConfig,
    TelegramConfig,
    WorkspaceConfig,
)
from jclaw.core.db import Database
from jclaw.tools.base import Observation, RuntimeState, ToolContext
from jclaw.tools.google_docs.tool import GoogleDocsTool


class DummyLLM:
    def chat(self, messages: list[dict[str, str]]) -> str:
        del messages
        return "stubbed"


class SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def chat(self, messages: list[dict[str, str]]) -> str:
        del messages
        return next(self._responses)


class FakeGoogleDocsClient:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.copies: list[tuple[str, str]] = []

    def get_document(self, document: str) -> dict[str, Any]:
        self.documents.append(document)
        return {
            "document_id": "doc-123",
            "title": "Lease Template",
            "revision_id": "rev-1",
            "text_preview": "Tenant: [tenant_names]\nRent: [total_monthly_rent]",
            "text_truncated": False,
            "placeholders": ["[tenant_names]", "[total_monthly_rent]"],
            "placeholder_count": 2,
            "body_element_count": 12,
        }

    def copy_document(self, document: str, *, name: str = "") -> dict[str, str]:
        self.copies.append((document, name))
        return {
            "document_id": "copy-123",
            "title": name or "Lease Template Copy",
            "url": "https://docs.google.com/document/d/copy-123/edit",
        }


def test_google_docs_inspect_document_returns_snapshot_artifact(tmp_path) -> None:
    client = FakeGoogleDocsClient()
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=client,  # type: ignore[arg-type]
    )

    result = tool.invoke(
        "inspect_document",
        {"document": "https://docs.google.com/document/d/doc-123/edit"},
        ToolContext(chat_id="chat"),
    )

    assert result.ok is True
    assert result.summary == "Inspected Google Doc 'Lease Template' and found 2 placeholder(s)."
    assert client.documents == ["https://docs.google.com/document/d/doc-123/edit"]
    assert result.data["document_id"] == "doc-123"
    assert result.data["placeholders"] == ["[tenant_names]", "[total_monthly_rent]"]
    assert result.data["artifacts"]["google_doc:latest"]["title"] == "Lease Template"
    assert result.data["artifacts"]["google_doc:doc-123"]["document_id"] == "doc-123"


def test_google_docs_controller_output_is_compact(tmp_path) -> None:
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=FakeGoogleDocsClient(),  # type: ignore[arg-type]
    )
    result = tool.invoke("inspect_document", {"document": "doc-123"}, ToolContext(chat_id="chat"))

    payload = tool.controller_output("inspect_document", result)

    assert payload == {
        "document_id": "doc-123",
        "title": "Lease Template",
        "placeholder_count": 2,
        "placeholders": ["[tenant_names]", "[total_monthly_rent]"],
        "body_element_count": 12,
        "text_truncated": False,
        "text_preview": "Tenant: [tenant_names]\nRent: [total_monthly_rent]",
    }
    assert "artifacts" not in payload


def test_google_docs_copy_document_returns_copy_artifact(tmp_path) -> None:
    client = FakeGoogleDocsClient()
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=client,  # type: ignore[arg-type]
    )

    result = tool.invoke(
        "copy_document",
        {"document": "doc-123", "name": "Lease for Alice"},
        ToolContext(chat_id="chat"),
    )

    assert result.ok is True
    assert result.summary == "Copied Google Doc to 'Lease for Alice'."
    assert client.copies == [("doc-123", "Lease for Alice")]
    assert result.data["document_id"] == "copy-123"
    assert result.data["url"] == "https://docs.google.com/document/d/copy-123/edit"
    assert result.data["source_document"] == "doc-123"
    assert result.data["artifacts"]["google_doc_copy:latest"]["title"] == "Lease for Alice"
    assert result.data["artifacts"]["google_doc_copy:copy-123"]["document_id"] == "copy-123"


def test_google_docs_copy_controller_output_is_compact(tmp_path) -> None:
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=FakeGoogleDocsClient(),  # type: ignore[arg-type]
    )
    result = tool.invoke("copy_document", {"document": "doc-123"}, ToolContext(chat_id="chat"))

    payload = tool.controller_output("copy_document", result)

    assert payload == {
        "document_id": "copy-123",
        "title": "Lease Template Copy",
        "url": "https://docs.google.com/document/d/copy-123/edit",
        "source_document": "doc-123",
    }
    assert "artifacts" not in payload


def test_google_docs_tool_catalog_exposes_initial_actions(tmp_path) -> None:
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=FakeGoogleDocsClient(),  # type: ignore[arg-type]
    )

    description = tool.describe()

    assert sorted(description["actions"]) == ["copy_document", "inspect_document"]
    assert description["actions"]["inspect_document"]["reads"] is True
    assert description["actions"]["inspect_document"]["produces_artifacts"] == ["google_doc"]
    assert description["actions"]["copy_document"]["writes"] is True
    assert description["actions"]["copy_document"]["requires_artifacts"] == ["google_doc"]
    assert description["actions"]["copy_document"]["produces_artifacts"] == ["google_doc_copy"]


def test_google_docs_inspect_document_returns_direct_result(tmp_path) -> None:
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=FakeGoogleDocsClient(),  # type: ignore[arg-type]
    )
    result = tool.invoke("inspect_document", {"document": "doc-123"}, ToolContext(chat_id="chat"))

    assert tool.should_return_direct("inspect_document", result) is True


def test_google_docs_copy_document_returns_direct_result(tmp_path) -> None:
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=FakeGoogleDocsClient(),  # type: ignore[arg-type]
    )
    result = tool.invoke("copy_document", {"document": "doc-123"}, ToolContext(chat_id="chat"))

    assert tool.should_return_direct("copy_document", result) is True


def test_google_docs_copy_materializes_latest_inspected_document(tmp_path) -> None:
    tool = GoogleDocsTool(
        GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
        client=FakeGoogleDocsClient(),  # type: ignore[arg-type]
    )
    runtime = RuntimeState(request="copy the inspected document")
    inspected = tool.invoke("inspect_document", {"document": "doc-123"}, ToolContext(chat_id="chat"))
    runtime.append(
        Observation.from_tool_result(
            inspected,
            controller_output=tool.controller_output("inspect_document", inspected),
        )
    )

    materialized = tool.materialize_params("copy_document", {"name": "Lease Copy"}, runtime)

    assert materialized == {"name": "Lease Copy", "document": "doc-123"}


def test_agent_registers_google_docs_tool_when_enabled(tmp_path) -> None:
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
        automation=AutomationConfig(enabled=False),
        email=EmailConfig(enabled=False),
        google_docs=GoogleDocsConfig(enabled=True, token_dir=tmp_path / "google-docs-tokens"),
        browser=BrowserConfig(enabled=False),
        workspace=WorkspaceConfig(enabled=False),
        knowledge=KnowledgeConfig(enabled=False),
        notion=NotionConfig(enabled=False),
        config_path=tmp_path / "config.toml",
        repo_root=Path("/Users/guanw/Documents/JClaw"),
    )
    db = Database(config.daemon.db_path)

    agent = AssistantAgent(config, db, DummyLLM())

    tool_names = [item["name"] for item in agent.tools.list_tools()]
    assert "google_docs" in tool_names
    db.close()


def test_agent_returns_google_docs_inspect_result_without_followup_controller_json(tmp_path) -> None:
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
        automation=AutomationConfig(enabled=False),
        email=EmailConfig(enabled=False),
        google_docs=GoogleDocsConfig(enabled=False, token_dir=tmp_path / "google-docs-tokens"),
        browser=BrowserConfig(enabled=False),
        workspace=WorkspaceConfig(enabled=False),
        knowledge=KnowledgeConfig(enabled=False),
        notion=NotionConfig(enabled=False),
        config_path=tmp_path / "config.toml",
        repo_root=Path("/Users/guanw/Documents/JClaw"),
    )
    db = Database(config.daemon.db_path)
    agent = AssistantAgent(
        config,
        db,
        SequenceLLM(
            [
                '{"type":"tool_call","tool":"google_docs","action":"inspect_document","params":{"document":"doc-123"},"reason":"Inspect requested document."}',
            ]
        ),
    )
    agent.tools.register(
        GoogleDocsTool(
            GoogleDocsConfig(enabled=True, token_dir=tmp_path / "tokens"),
            client=FakeGoogleDocsClient(),  # type: ignore[arg-type]
        )
    )

    reply = agent.handle_text("chat", "inspect this google doc: doc-123")

    assert "Inspected Google Doc 'Lease Template' and found 2 placeholder(s)." in reply
    assert "Title: Lease Template" in reply
    assert "[tenant_names]" in reply
    db.close()
