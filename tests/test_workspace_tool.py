from __future__ import annotations

from pathlib import Path
import subprocess

from jclaw.core.db import Database
from jclaw.tools.base import ToolContext
from jclaw.tools.workspace.tool import WorkspaceTool


def _grant_all(db: Database, root: Path, chat_id: str = "chat-1") -> None:
    db.upsert_grant(str(root.resolve()), ("read", "write", "git", "shell"), chat_id)


def test_inspect_root_requires_grant_then_lists_entries(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "notes.txt").write_text("hello\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    gated = tool.invoke("inspect_root", {"path": str(root)}, ToolContext(chat_id="chat-1"))
    assert gated.needs_confirmation is True
    assert gated.data["request_kind"] == "grant"

    _grant_all(db, root)
    inspected = tool.invoke("inspect_root", {"path": str(root)}, ToolContext(chat_id="chat-1"))
    assert inspected.ok is True
    assert inspected.data["kind"] == "directory"
    assert inspected.data["entry_count"] == 1
    assert inspected.data["entries_truncated"] is False
    assert inspected.data["entries"][0]["name"] == "notes.txt"
    db.close()


def test_inspect_root_remaps_foreign_home_path_to_local_home(tmp_path) -> None:
    fake_home = tmp_path / "home"
    documents = fake_home / "Documents"
    documents.mkdir(parents=True)
    (documents / "notes.txt").write_text("hello\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, fake_home)
    tool = WorkspaceTool(db, tmp_path / "state", tmp_path / "repo", draft_change=lambda payload: None)
    tool.home_dir = fake_home.resolve()

    inspected = tool.invoke(
        "inspect_root",
        {"path": "/Users/Jude/Documents"},
        ToolContext(chat_id="chat-1"),
    )
    assert inspected.ok is True
    assert inspected.data["target_path"] == str(documents.resolve())
    assert inspected.data["entry_count"] == 1
    assert inspected.data["entries"][0]["name"] == "notes.txt"
    db.close()


def test_inspect_root_reports_when_entries_are_truncated(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(60):
        (root / f"file-{index:02d}.txt").write_text("x\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(
        db,
        tmp_path / "state",
        root,
        draft_change=lambda payload: None,
        options={"max_path_entries": 10},
    )

    inspected = tool.invoke("inspect_root", {"path": str(root)}, ToolContext(chat_id="chat-1"))
    assert inspected.ok is True
    assert inspected.data["entry_count"] == 60
    assert inspected.data["entries_truncated"] is True
    assert len(inspected.data["entries"]) == 10
    db.close()


def test_path_metadata_find_files_and_search_contents(tmp_path) -> None:
    root = tmp_path / "repo"
    docs = root / "docs"
    docs.mkdir(parents=True)
    target = docs / "notes.txt"
    target.write_text("alpha\nbeta keyword\n", encoding="utf-8")
    other = docs / "todo.md"
    other.write_text("keyword appears here too\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    metadata = tool.invoke("path_metadata", {"path": str(target)}, ToolContext(chat_id="chat-1"))
    assert metadata.ok is True
    assert metadata.data["kind"] == "file"
    assert metadata.data["metadata"]["size_bytes"] > 0

    found = tool.invoke(
        "find_files",
        {"path": str(root), "pattern": "*.txt"},
        ToolContext(chat_id="chat-1"),
    )
    assert found.ok is True
    assert found.data["match_count"] == 1
    assert found.data["matches"][0]["path"] == "docs/notes.txt"

    searched = tool.invoke(
        "search_contents",
        {"path": str(root), "query": "keyword"},
        ToolContext(chat_id="chat-1"),
    )
    assert searched.ok is True
    assert searched.data["match_count"] == 2
    assert {item["path"] for item in searched.data["matches"]} == {"docs/notes.txt", "docs/todo.md"}
    db.close()


def test_search_contents_supports_regex_queries_and_file_pattern(tmp_path) -> None:
    root = tmp_path / "repo"
    src = root / "src"
    src.mkdir(parents=True)
    py_file = src / "tool.py"
    py_file.write_text("value = re.search('x', text)\n", encoding="utf-8")
    txt_file = src / "notes.txt"
    txt_file.write_text("re.search should not count when file_pattern is python-only\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    searched = tool.invoke(
        "search_contents",
        {
            "root": str(root),
            "query": r"re\.(compile|match|search|findall)",
            "file_pattern": "*.py",
        },
        ToolContext(chat_id="chat-1"),
    )

    assert searched.ok is True
    assert searched.data["match_count"] == 1
    assert searched.data["matches"][0]["path"] == "src/tool.py"
    db.close()


def test_prepare_change_previews_before_apply(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(
        db,
        tmp_path / "state",
        root,
        draft_change=lambda payload: {
            "summary": "Update greeting.",
            "edits": [
                {
                    "path": "app.py",
                    "reason": "Change greeting text.",
                    "new_content": "print('goodbye')\n",
                }
            ],
        },
    )

    preview = tool.invoke(
        "prepare_change",
        {"objective": "Update the greeting in app.py", "path": str(target)},
        ToolContext(chat_id="chat-1"),
    )
    assert preview.needs_confirmation is True
    assert "app.py" in preview.data["touched_files"]
    assert "goodbye" not in target.read_text(encoding="utf-8")

    applied = tool.invoke(
        "apply_change_request",
        {"request_id": preview.data["request_id"]},
        ToolContext(chat_id="chat-1"),
    )
    assert applied.ok is True
    assert target.read_text(encoding="utf-8") == "print('goodbye')\n"
    db.close()


def test_rename_move_copy_and_delete_paths_require_preview_then_apply(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "notes.txt"
    source.write_text("hello\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    rename_preview = tool.invoke(
        "rename_path",
        {"path": str(source), "new_name": "renamed.txt"},
        ToolContext(chat_id="chat-1"),
    )
    assert rename_preview.needs_confirmation is True
    rename_apply = tool.invoke(
        "apply_path_request",
        {"request_id": rename_preview.data["request_id"]},
        ToolContext(chat_id="chat-1"),
    )
    renamed = root / "renamed.txt"
    assert rename_apply.ok is True
    assert renamed.exists()

    move_preview = tool.invoke(
        "move_path",
        {"source_path": str(renamed), "destination_path": str(root / "nested" / "moved.txt")},
        ToolContext(chat_id="chat-1"),
    )
    assert move_preview.needs_confirmation is True
    tool.invoke(
        "apply_path_request",
        {"request_id": move_preview.data["request_id"]},
        ToolContext(chat_id="chat-1"),
    )
    moved = root / "nested" / "moved.txt"
    assert moved.exists()

    copy_preview = tool.invoke(
        "copy_path",
        {"source_path": str(moved), "destination_path": str(root / "copy.txt")},
        ToolContext(chat_id="chat-1"),
    )
    assert copy_preview.needs_confirmation is True
    tool.invoke(
        "apply_path_request",
        {"request_id": copy_preview.data["request_id"]},
        ToolContext(chat_id="chat-1"),
    )
    copied = root / "copy.txt"
    assert copied.exists()

    delete_preview = tool.invoke(
        "delete_path",
        {"path": str(copied)},
        ToolContext(chat_id="chat-1"),
    )
    assert delete_preview.needs_confirmation is True
    delete_apply = tool.invoke(
        "apply_path_request",
        {"request_id": delete_preview.data["request_id"]},
        ToolContext(chat_id="chat-1"),
    )
    assert delete_apply.ok is True
    assert not copied.exists()
    db.close()


def test_prepare_git_action_blocks_remote_operations(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    result = tool.invoke(
        "prepare_git_action",
        {"action": "push", "path": str(root)},
        ToolContext(chat_id="chat-1"),
    )
    assert result.ok is False
    assert "blocked" in result.summary.lower()
    db.close()


def test_prepare_and_apply_local_git_commit(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "JClaw Test"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "jclaw@example.com"], cwd=root, check=True, capture_output=True)
    target = root / "app.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=root, check=True, capture_output=True)
    target.write_text("print('updated')\n", encoding="utf-8")

    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)
    preview = tool.invoke(
        "prepare_git_action",
        {"action": "commit", "message": "Apply update", "path": str(root)},
        ToolContext(chat_id="chat-1"),
    )
    assert preview.needs_confirmation is True

    applied = tool.invoke(
        "apply_git_request",
        {"request_id": preview.data["request_id"]},
        ToolContext(chat_id="chat-1"),
    )
    assert applied.ok is True
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert log.stdout.strip() == "Apply update"
    db.close()


def test_prepare_shell_action_enforces_policy_and_apply_runs_locally(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    blocked = tool.invoke(
        "prepare_shell_action",
        {"command": "curl https://example.com", "path": str(root)},
        ToolContext(chat_id="chat-1"),
    )
    assert blocked.ok is False
    assert "not allowed" in blocked.summary.lower() or "blocked" in blocked.summary.lower()

    preview = tool.invoke(
        "prepare_shell_action",
        {"command": "pwd", "path": str(root)},
        ToolContext(chat_id="chat-1"),
    )
    assert preview.needs_confirmation is True

    applied = tool.invoke(
        "apply_shell_request",
        {"request_id": preview.data["request_id"]},
        ToolContext(chat_id="chat-1"),
    )
    assert applied.ok is True
    assert str(root.resolve()) in applied.data["stdout"]
    db.close()
