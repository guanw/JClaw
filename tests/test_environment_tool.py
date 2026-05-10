from __future__ import annotations

from jclaw.core.db import Database
from jclaw.core.environment import sync_environment_catalog
from jclaw.tools.base import ToolContext
from jclaw.tools.environment.tool import EnvironmentTool


def test_environment_inspect_bootstraps_catalog_and_attaches_artifact(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/sh\necho git version 2.50.0\n", encoding="utf-8")
    git.chmod(0o755)
    rg = bin_dir / "rg"
    rg.write_text("#!/bin/sh\necho ripgrep help\n", encoding="utf-8")
    rg.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("SHELL", "/bin/zsh")

    db = Database(tmp_path / "jclaw.db")
    tool = EnvironmentTool(
        db,
        repo_root=repo_root,
        environment_path=tmp_path / "state" / "environment.json",
    )

    result = tool.invoke("inspect", {}, ToolContext(chat_id="chat-1"))

    assert result.ok is True
    assert result.data["repo_root"] == str(repo_root.resolve())
    assert result.data["approved_roots"] == [str(repo_root.resolve())]
    assert result.data["known_commands"][0]["name"] == "git"
    assert result.data["artifacts"]["environment_snapshot:latest"]["commands"]["git"]["path"] == str(git)
    assert (tmp_path / "state" / "environment.json").exists() is True
    db.close()


def test_environment_controller_output_exposes_structured_known_commands(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    custom = bin_dir / "customcmd"
    custom.write_text("#!/bin/sh\necho custom command help\n", encoding="utf-8")
    custom.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("SHELL", "/bin/zsh")

    db = Database(tmp_path / "jclaw.db")
    tool = EnvironmentTool(
        db,
        repo_root=repo_root,
        environment_path=tmp_path / "state" / "environment.json",
    )
    sync_environment_catalog(
        tmp_path / "state" / "environment.json",
        repo_root=repo_root,
        approved_roots=[repo_root],
        shell="/bin/zsh",
        cwd=repo_root,
        search_path=str(bin_dir),
        command_names=("customcmd",),
    )
    result = tool.invoke("inspect", {}, ToolContext(chat_id="chat-1"))
    payload = tool.controller_output("inspect", result)

    assert payload["repo_root"] == str(repo_root.resolve())
    assert payload["approved_roots"] == [str(repo_root.resolve())]
    assert payload["known_commands"] == [
        {
            "name": "customcmd",
            "description": "custom command help",
        }
    ]
    db.close()


def test_environment_inspect_includes_granted_roots(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    extra_root = tmp_path / "Desktop"
    extra_root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/sh\necho git help\n", encoding="utf-8")
    git.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("SHELL", "/bin/zsh")

    db = Database(tmp_path / "jclaw.db")
    db.upsert_grant(str(extra_root.resolve()), ("read",), "chat-1")
    tool = EnvironmentTool(
        db,
        repo_root=repo_root,
        environment_path=tmp_path / "state" / "environment.json",
    )

    result = tool.invoke("inspect", {}, ToolContext(chat_id="chat-1"))

    assert result.ok is True
    assert result.data["approved_roots"] == sorted([str(extra_root.resolve()), str(repo_root.resolve())])
    db.close()
