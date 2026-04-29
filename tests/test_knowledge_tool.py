from __future__ import annotations

from pathlib import Path

import pytest

from jclaw.core.db import Database
from jclaw.tools.base import ToolContext
from jclaw.tools.knowledge.readers.pdf_reader import PdfReaderTool
from jclaw.tools.knowledge.tool import KnowledgeTool


def _grant_read(db: Database, root: Path, chat_id: str = "chat-1") -> None:
    db.upsert_grant(str(root.resolve()), ("read",), chat_id)


def test_analyze_paths_requires_grant_then_extracts_text_file(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "notes.txt"
    target.write_text("hello world\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    tool = KnowledgeTool(db, tmp_path / "state", root)

    gated = tool.invoke("analyze_paths", {"paths": [str(target)]}, ToolContext(chat_id="chat-1"))
    assert gated.needs_confirmation is True
    assert gated.data["request_kind"] == "grant"

    _grant_read(db, root)
    result = tool.invoke("analyze_paths", {"paths": [str(target)]}, ToolContext(chat_id="chat-1"))
    assert result.ok is True
    assert result.data["grounded"] is True
    assert result.data["supported_files"][0]["path"] == str(target.resolve())
    assert result.data["chunks"][0]["path"] == str(target.resolve())
    assert result.data["allow_tool_followup"] is True
    assert result.data["artifacts"]["knowledge_context:latest"]["supported_files"][0]["path"] == str(target.resolve())
    db.close()


def test_pdf_reader_extracts_text(monkeypatch, tmp_path) -> None:
    target = tmp_path / "scan.pdf"
    target.write_bytes(b"%PDF-1.4")

    class FakePage:
        def extract_text(self):  # noqa: ANN001
            return "Quarterly report revenue grew 20%."

    class FakeReader:
        def __init__(self, path):  # noqa: ANN001
            self.pages = [FakePage()]

    monkeypatch.setattr("jclaw.tools.knowledge.readers.pdf_reader.PdfReader", FakeReader)
    reader = PdfReaderTool()
    document = reader.extract(target, max_bytes=20000)
    assert document.file_type == "pdf"
    assert "revenue grew 20%" in document.text


def test_analyze_paths_supports_pdf_files(monkeypatch, tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "scan.pdf"
    target.write_bytes(b"%PDF-1.4")

    class FakePage:
        def extract_text(self):  # noqa: ANN001
            return "Policy number 12345."

    class FakeReader:
        def __init__(self, path):  # noqa: ANN001
            self.pages = [FakePage()]

    monkeypatch.setattr("jclaw.tools.knowledge.readers.pdf_reader.PdfReader", FakeReader)
    db = Database(tmp_path / "jclaw.db")
    _grant_read(db, root)
    tool = KnowledgeTool(db, tmp_path / "state", root)

    result = tool.invoke("analyze_paths", {"paths": [str(target)]}, ToolContext(chat_id="chat-1"))
    assert result.ok is True
    assert result.data["supported_files"][0]["path"] == str(target.resolve())
    assert result.data["supported_files"][0]["file_type"] == "pdf"
    assert result.data["grounded"] is True
    db.close()


@pytest.mark.parametrize("suffix", [".png", ".webp", ".gif", ".tiff", ".tif", ".icns"])
def test_analyze_paths_marks_images_unsupported_without_llm_callback(tmp_path, suffix: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / f"scan{suffix}"
    target.write_bytes(b"not-a-real-image-but-good-enough-for-the-stub")
    db = Database(tmp_path / "jclaw.db")
    _grant_read(db, root)
    tool = KnowledgeTool(db, tmp_path / "state", root)

    result = tool.invoke("analyze_paths", {"paths": [str(target)]}, ToolContext(chat_id="chat-1"))
    assert result.ok is True
    assert result.data["supported_files"] == []
    assert result.data["grounded"] is False
    assert result.data["partial"] is True
    assert result.data["unsupported_files"][0]["path"] == str(target.resolve())
    assert result.data["unsupported_files"][0]["reason"] == "Unsupported file type."
    db.close()


def test_knowledge_tool_describe_exposes_structured_action_specs(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = KnowledgeTool(db, tmp_path / "state", tmp_path / "repo")

    description = tool.describe()

    assert description["actions"]["analyze_paths"]["produces_artifacts"] == ["knowledge_context"]
    assert sorted(description["actions"].keys()) == ["analyze_paths"]
    db.close()
