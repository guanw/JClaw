from __future__ import annotations

from pathlib import Path

from jclaw.core.db import Database
from jclaw.tools.base import ToolContext
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
    db.close()


def test_answer_from_paths_returns_grounded_answer_with_citations(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "notes.txt"
    target.write_text("Project owner is guan.\nDeadline is Friday.\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_read(db, root)

    def answer_question(payload):  # noqa: ANN001
        return {
            "answer": "The project owner is guan.",
            "cited_chunk_ids": [payload["chunks"][0]["chunk_id"]],
            "grounded": True,
            "partial": False,
        }

    tool = KnowledgeTool(
        db,
        tmp_path / "state",
        root,
        answer_question=answer_question,
    )
    result = tool.invoke(
        "answer_from_paths",
        {"paths": [str(target)], "question": "Who is the project owner?"},
        ToolContext(chat_id="chat-1"),
    )
    assert result.ok is True
    assert result.data["answer"] == "The project owner is guan."
    assert result.data["grounded"] is True
    assert result.data["partial"] is False
    assert result.data["citations"][0]["path"] == str(target.resolve())
    db.close()


def test_analyze_paths_reports_unsupported_files(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "scan.pdf"
    target.write_bytes(b"%PDF-1.4 test")
    db = Database(tmp_path / "jclaw.db")
    _grant_read(db, root)
    tool = KnowledgeTool(db, tmp_path / "state", root)

    result = tool.invoke("analyze_paths", {"paths": [str(target)]}, ToolContext(chat_id="chat-1"))
    assert result.ok is True
    assert result.data["supported_files"] == []
    assert result.data["unsupported_files"][0]["path"] == str(target.resolve())
    assert result.data["grounded"] is False
    db.close()
