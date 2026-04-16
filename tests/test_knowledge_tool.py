from __future__ import annotations

from pathlib import Path

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


def test_analyze_paths_supports_images_with_callback(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "scan.png"
    target.write_bytes(b"not-a-real-png-but-good-enough-for-the-stub")
    db = Database(tmp_path / "jclaw.db")
    _grant_read(db, root)

    def analyze_image(path):  # noqa: ANN001
        return {
            "text": "Screenshot of a dashboard showing total revenue 42.",
            "warnings": [],
        }

    tool = KnowledgeTool(
        db,
        tmp_path / "state",
        root,
        analyze_image=analyze_image,
    )

    result = tool.invoke("analyze_paths", {"paths": [str(target)]}, ToolContext(chat_id="chat-1"))
    assert result.ok is True
    assert result.data["supported_files"][0]["path"] == str(target.resolve())
    assert result.data["supported_files"][0]["file_type"] == "image"
    assert result.data["grounded"] is True
    assert "total revenue 42" in result.data["chunks"][0]["text"]
    db.close()


def test_answer_from_folder_preserves_deterministic_file_order(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "b-notes.txt").write_text("Second file.\n", encoding="utf-8")
    (root / "a-contract.txt").write_text("First file.\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_read(db, root)

    def answer_question(payload):  # noqa: ANN001
        supported = payload["supported_files"]
        assert supported[0]["path"].endswith("a-contract.txt")
        assert supported[0]["order"] == 1
        return {
            "answer": "The first file is a-contract.txt.",
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
        {"paths": [str(root)], "question": "Summarize the first file you found."},
        ToolContext(chat_id="chat-1"),
    )
    assert result.ok is True
    assert result.data["answer"] == "The first file is a-contract.txt."
    db.close()
