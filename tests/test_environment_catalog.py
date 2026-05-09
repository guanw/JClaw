from pathlib import Path

from jclaw.core.environment import (
    ENVIRONMENT_CATALOG_VERSION,
    build_environment_catalog,
    environment_catalog_path,
    load_environment_catalog,
    save_environment_catalog,
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
