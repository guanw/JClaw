from pathlib import Path

from jclaw.core.environment import (
    ENVIRONMENT_CATALOG_VERSION,
    build_environment_catalog,
    environment_catalog_path,
    load_environment_catalog,
    save_environment_catalog,
    sync_environment_catalog,
)


def test_environment_catalog_path_uses_state_dir() -> None:
    state_dir = Path("/tmp/jclaw-state")
    assert environment_catalog_path(state_dir) == state_dir / "environment.json"


def test_build_environment_catalog_normalizes_roots_and_defaults(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    other_root = tmp_path / "Desktop"
    other_root.mkdir()

    payload = build_environment_catalog(
        repo_root=repo_root,
        approved_roots=[repo_root, other_root, repo_root],
        shell="/bin/zsh",
        cwd=repo_root,
        commands={
            "git": {
                "path": "/opt/homebrew/bin/git",
                "description": "Version control CLI.",
            }
        },
    )

    assert payload["version"] == ENVIRONMENT_CATALOG_VERSION
    assert payload["host"]["platform"] in {"macOS", "Linux"}
    assert payload["host"]["shell"] == "/bin/zsh"
    assert payload["host"]["cwd"] == str(repo_root.resolve())
    assert payload["roots"]["repo_root"] == str(repo_root.resolve())
    assert payload["roots"]["approved"] == sorted([str(other_root.resolve()), str(repo_root.resolve())])
    assert payload["interpreters"] == {}
    assert payload["apps"] == {}
    assert payload["commands"]["git"]["path"] == "/opt/homebrew/bin/git"


def test_save_and_load_environment_catalog_round_trip(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target = environment_catalog_path(tmp_path / "state")
    payload = build_environment_catalog(
        repo_root=repo_root,
        approved_roots=[repo_root],
        shell="/bin/zsh",
        cwd=repo_root,
        interpreters={
            "python": {
                "preferred": str(repo_root / ".venv" / "bin" / "python"),
                "version": "3.13.12",
                "description": "Preferred Python interpreter.",
            }
        },
        commands={
            "pytest": {
                "path": str(repo_root / ".venv" / "bin" / "pytest"),
                "description": "Python test runner.",
            }
        },
        apps={
            "Safari": {
                "bundle_id": "com.apple.Safari",
                "description": "Apple web browser.",
            }
        },
    )

    saved = save_environment_catalog(target, payload)
    loaded = load_environment_catalog(target)

    assert target.exists() is True
    assert saved == loaded


def test_load_environment_catalog_returns_none_when_missing(tmp_path) -> None:
    assert load_environment_catalog(tmp_path / "missing.json") is None


def test_save_environment_catalog_normalizes_roots_and_missing_fields(tmp_path) -> None:
    target = environment_catalog_path(tmp_path / "state")
    payload = {
        "version": 1,
        "host": {
            "platform": "macOS",
            "shell": "/bin/zsh",
            "cwd": "/tmp/work",
        },
        "roots": {
            "repo_root": "/tmp/work",
            "approved": ["/tmp/work", "/tmp/Desktop", "/tmp/work"],
        },
        "interpreters": {
            "python": {
                "preferred": "/tmp/work/.venv/bin/python",
                "description": "Preferred Python interpreter.",
            }
        },
        "commands": {
            "git": {
                "path": "/opt/homebrew/bin/git",
                "description": "Version control CLI.",
            }
        },
        "apps": {
            "Notion": {
                "bundle_id": "notion.id",
                "description": "Desktop note-taking app.",
            }
        },
    }

    saved = save_environment_catalog(target, payload)

    assert saved["roots"]["approved"] == ["/tmp/Desktop", "/tmp/work"]
    assert saved["interpreters"]["python"]["preferred"] == "/tmp/work/.venv/bin/python"
    assert saved["interpreters"]["python"]["description"] == "Preferred Python interpreter."


def test_sync_environment_catalog_bootstraps_missing_file_and_discovers_commands(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    bin_dir = tmp_path / "bin"
    repo_root.mkdir()
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/sh\necho git version 2.50.0\n", encoding="utf-8")
    git.chmod(0o755)
    custom = bin_dir / "customcmd"
    custom.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")
    custom.chmod(0o755)

    target = environment_catalog_path(tmp_path / "state")
    saved = sync_environment_catalog(
        target,
        repo_root=repo_root,
        approved_roots=[repo_root],
        shell="/bin/zsh",
        cwd=repo_root,
        search_path=str(bin_dir),
        command_names=("customcmd",),
    )

    assert target.exists() is True
    assert saved["commands"]["git"]["path"] == str(git)
    assert saved["commands"]["customcmd"]["path"] == str(custom)
    assert saved["commands"]["customcmd"]["description"] == "custom"
    assert saved["interpreters"] == {}
    assert saved["apps"] == {}


def test_sync_environment_catalog_removes_missing_command_and_updates_roots(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    extra_root = tmp_path / "Desktop"
    extra_root.mkdir()
    target = environment_catalog_path(tmp_path / "state")

    save_environment_catalog(
        target,
        build_environment_catalog(
            repo_root=repo_root,
            approved_roots=[repo_root],
            shell="/bin/zsh",
            cwd=repo_root,
            commands={
                "ruff": {
                    "path": "/missing/ruff",
                    "description": "Python linter.",
                }
            },
        ),
    )

    saved = sync_environment_catalog(
        target,
        repo_root=repo_root,
        approved_roots=[repo_root, extra_root],
        shell="/bin/zsh",
        cwd=repo_root,
        search_path=str(tmp_path / "empty-bin"),
        command_names=("ruff",),
    )

    assert "ruff" not in saved["commands"]
    assert saved["roots"]["approved"] == sorted([str(extra_root.resolve()), str(repo_root.resolve())])
    assert saved["apps"] == {}
