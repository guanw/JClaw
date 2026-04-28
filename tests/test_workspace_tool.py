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
    assert inspected.data["allow_tool_followup"] is True
    assert inspected.data["artifacts"]["workspace_path:latest"]["target_path"] == str(root.resolve())
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
    assert searched.data["allow_tool_followup"] is True
    assert searched.data["artifacts"]["workspace_search_results:latest"]["query"] == "keyword"
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


def test_search_contents_supports_brace_expansion_file_patterns(tmp_path) -> None:
    root = tmp_path / "repo"
    src = root / "src"
    src.mkdir(parents=True)
    (src / "tool.py").write_text("keyword here\n", encoding="utf-8")
    (src / "view.ts").write_text("keyword there\n", encoding="utf-8")
    (src / "notes.txt").write_text("keyword elsewhere\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    searched = tool.invoke(
        "search_contents",
        {
            "root": str(root),
            "query": "keyword",
            "file_pattern": "*.{py,ts}",
        },
        ToolContext(chat_id="chat-1"),
    )

    assert searched.ok is True
    assert searched.data["match_count"] == 2
    assert {item["path"] for item in searched.data["matches"]} == {"src/tool.py", "src/view.ts"}
    db.close()


def test_find_files_supports_brace_expansion_patterns(tmp_path) -> None:
    root = tmp_path / "repo"
    images = root / "images"
    images.mkdir(parents=True)
    (images / "one.png").write_text("png\n", encoding="utf-8")
    (images / "two.jpg").write_text("jpg\n", encoding="utf-8")
    (images / "three.txt").write_text("txt\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    found = tool.invoke(
        "find_files",
        {"root": str(root), "pattern": "*.{jpg,png,gif}"},
        ToolContext(chat_id="chat-1"),
    )

    assert found.ok is True
    assert found.data["match_count"] == 2
    assert {item["path"] for item in found.data["matches"]} == {"images/one.png", "images/two.jpg"}
    db.close()


def test_find_files_supports_comma_separated_patterns(tmp_path) -> None:
    root = tmp_path / "desktop"
    root.mkdir(parents=True)
    (root / "one.png").write_text("png\n", encoding="utf-8")
    (root / "two.jpg").write_text("jpg\n", encoding="utf-8")
    (root / "three.pdf").write_text("pdf\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    found = tool.invoke(
        "find_files",
        {"root": str(root), "pattern": "*.png,*.jpg,*.jpeg,*.gif,*.webp,*.tiff,*.tif,*.icns"},
        ToolContext(chat_id="chat-1"),
    )

    assert found.ok is True
    assert found.data["match_count"] == 2
    assert {item["path"] for item in found.data["matches"]} == {"one.png", "two.jpg"}
    db.close()


def test_read_file_requires_grant_then_returns_workspace_file_artifact(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_text("print('hello')\nprint('world')\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    gated = tool.invoke("read_file", {"path": str(target)}, ToolContext(chat_id="chat-1"))
    assert gated.needs_confirmation is True
    assert gated.data["request_kind"] == "grant"

    _grant_all(db, root)
    result = tool.invoke("read_file", {"path": str(target)}, ToolContext(chat_id="chat-1"))
    assert result.ok is True
    assert result.data["content"] == "print('hello')\nprint('world')\n"
    assert result.data["line_count"] == 2
    assert result.data["truncated"] is False
    assert result.data["allow_tool_followup"] is True
    assert result.data["artifacts"]["workspace_file:latest"]["start_line"] == 1
    assert result.data["artifacts"]["workspace_file:latest"]["end_line"] == 2
    db.close()


def test_read_file_rejects_missing_and_non_file_paths(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    missing_arg = tool.invoke("read_file", {}, ToolContext(chat_id="chat-1"))
    assert missing_arg.ok is False
    assert "requires a path" in missing_arg.summary.lower()

    missing_file = tool.invoke("read_file", {"path": str(root / "missing.py")}, ToolContext(chat_id="chat-1"))
    assert missing_file.ok is False
    assert "does not exist" in missing_file.summary.lower()

    directory = tool.invoke("read_file", {"path": str(root)}, ToolContext(chat_id="chat-1"))
    assert directory.ok is False
    assert "is not a file" in directory.summary.lower()
    db.close()


def test_read_file_rejects_binary_like_files(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "image.bin"
    target.write_bytes(b"\x89PNG\x00binary")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    result = tool.invoke("read_file", {"path": str(target)}, ToolContext(chat_id="chat-1"))
    assert result.ok is False
    assert "binary file" in result.summary.lower()
    db.close()


def test_read_file_marks_truncation_when_content_exceeds_internal_limit(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "long.txt"
    target.write_text("abcdef\n" * 20, encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(
        db,
        tmp_path / "state",
        root,
        draft_change=lambda payload: None,
        options={"max_internal_read_bytes": 20},
    )

    result = tool.invoke("read_file", {"path": str(target)}, ToolContext(chat_id="chat-1"))
    assert result.ok is True
    assert result.data["truncated"] is True
    assert result.data["bytes_read"] == 20
    assert len(result.data["content"]) <= 20
    db.close()


def test_read_snippet_returns_inclusive_line_range_and_clamps_to_eof(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_text("a\nb\nc\nd\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    snippet = tool.invoke(
        "read_snippet",
        {"path": str(target), "start_line": 2, "end_line": 3},
        ToolContext(chat_id="chat-1"),
    )
    assert snippet.ok is True
    assert snippet.data["content"] == "b\nc\n"
    assert snippet.data["start_line"] == 2
    assert snippet.data["end_line"] == 3

    clamped = tool.invoke(
        "read_snippet",
        {"path": str(target), "start_line": 3, "end_line": 10},
        ToolContext(chat_id="chat-1"),
    )
    assert clamped.ok is True
    assert clamped.data["content"] == "c\nd\n"
    assert clamped.data["end_line"] == 4
    assert clamped.data["truncated"] is False
    db.close()


def test_read_snippet_rejects_invalid_ranges(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_text("a\nb\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    invalid = tool.invoke(
        "read_snippet",
        {"path": str(target), "start_line": 3, "end_line": 2},
        ToolContext(chat_id="chat-1"),
    )
    assert invalid.ok is False

    out_of_range = tool.invoke(
        "read_snippet",
        {"path": str(target), "start_line": 5, "end_line": 6},
        ToolContext(chat_id="chat-1"),
    )
    assert out_of_range.ok is False
    assert "starts after end of file" in out_of_range.summary.lower()
    db.close()


def test_read_snippet_can_read_late_lines_from_large_file_without_whole_file_truncation(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "big.py"
    lines = [f"line_{index:04d} = {index}\n" for index in range(1, 2201)]
    target.write_text("".join(lines), encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(
        db,
        tmp_path / "state",
        root,
        draft_change=lambda payload: None,
        options={"max_internal_read_bytes": 200},
    )

    snippet = tool.invoke(
        "read_snippet",
        {"path": str(target), "start_line": 1290, "end_line": 1295},
        ToolContext(chat_id="chat-1"),
    )

    assert snippet.ok is True
    assert "line_1290 = 1290" in snippet.data["content"]
    assert "line_1295 = 1295" in snippet.data["content"]
    assert snippet.data["truncated"] is False
    assert snippet.data["start_line"] == 1290
    assert snippet.data["end_line"] == 1295
    db.close()


def test_read_snippet_truncates_only_the_requested_range_when_snippet_is_too_large(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "big.py"
    lines = [f"{index:04d} abcdefghijklmnopqrstuvwxyz\n" for index in range(1, 200)]
    target.write_text("".join(lines), encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(
        db,
        tmp_path / "state",
        root,
        draft_change=lambda payload: None,
        options={"max_internal_read_bytes": 80},
    )

    snippet = tool.invoke(
        "read_snippet",
        {"path": str(target), "start_line": 1, "end_line": 10},
        ToolContext(chat_id="chat-1"),
    )

    assert snippet.ok is True
    assert snippet.data["truncated"] is True
    assert snippet.data["bytes_read"] == 80
    assert len(snippet.data["content"].encode("utf-8")) <= 80
    db.close()


def test_format_result_includes_snippet_content_and_diff_text(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    snippet = tool.invoke(
        "read_snippet",
        {"path": str(target), "start_line": 2, "end_line": 3},
        ToolContext(chat_id="chat-1"),
    )
    snippet_text = tool.format_result("read_snippet", snippet)
    assert "Content:" in snippet_text
    assert "b\nc\n" in snippet_text

    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "JClaw Test"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "jclaw@example.com"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "app.py"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=root, check=True, capture_output=True, text=True)
    target.write_text("a\nb\nchanged\n", encoding="utf-8")

    diff = tool.invoke("git_diff", {"path": str(target)}, ToolContext(chat_id="chat-1"))
    diff_text = tool.format_result("git_diff", diff)
    assert "Diff:" in diff_text
    assert "changed" in diff_text
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


def test_git_status_emits_followup_artifact(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    (root / "notes.txt").write_text("hello\n", encoding="utf-8")
    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    status = tool.invoke("git_status", {"path": str(root)}, ToolContext(chat_id="chat-1"))

    assert status.ok is True
    assert status.data["allow_tool_followup"] is True
    assert status.data["artifacts"]["workspace_git_status:latest"]["root_path"] == str(root.resolve())
    db.close()


def test_git_diff_returns_unstaged_and_staged_diffs_and_scopes_to_path(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "JClaw Test"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "jclaw@example.com"], cwd=root, check=True, capture_output=True, text=True)
    target = root / "app.py"
    other = root / "other.py"
    target.write_text("print('one')\n", encoding="utf-8")
    other.write_text("print('other')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py", "other.py"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=root, check=True, capture_output=True, text=True)

    target.write_text("print('two')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=root, check=True, capture_output=True, text=True)
    other.write_text("print('other changed')\n", encoding="utf-8")

    db = Database(tmp_path / "jclaw.db")
    _grant_all(db, root)
    tool = WorkspaceTool(db, tmp_path / "state", root, draft_change=lambda payload: None)

    diff = tool.invoke("git_diff", {"path": str(target)}, ToolContext(chat_id="chat-1"))
    assert diff.ok is True
    assert diff.data["has_staged"] is True
    assert diff.data["has_unstaged"] is False
    assert "### Staged" in diff.data["diff"]
    assert "app.py" in diff.data["diff"]
    assert "other.py" not in diff.data["diff"]

    full_diff = tool.invoke("git_diff", {"path": str(root)}, ToolContext(chat_id="chat-1"))
    assert full_diff.ok is True
    assert full_diff.data["has_unstaged"] is True
    assert "### Unstaged" in full_diff.data["diff"]
    assert "other.py" in full_diff.data["diff"]
    db.close()


def test_workspace_tool_describe_exposes_structured_action_specs(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    tool = WorkspaceTool(db, tmp_path / "state", tmp_path / "repo", draft_change=lambda payload: None)

    description = tool.describe()

    assert description["actions"]["inspect_root"]["produces_artifacts"] == ["workspace_path"]
    assert description["actions"]["search_contents"]["produces_artifacts"] == ["workspace_search_results"]
    assert description["actions"]["git_status"]["produces_artifacts"] == ["workspace_git_status"]
    assert description["actions"]["read_file"]["produces_artifacts"] == ["workspace_file"]
    assert description["actions"]["read_snippet"]["produces_artifacts"] == ["workspace_file"]
    assert description["actions"]["git_diff"]["produces_artifacts"] == ["workspace_diff"]
    assert description["actions"]["prepare_change"]["requires_confirmation"] is True
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
