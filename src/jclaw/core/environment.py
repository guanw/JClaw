from __future__ import annotations

import json
import os
import platform
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

ENVIRONMENT_CATALOG_VERSION = 1


def environment_catalog_path(state_dir: str | Path) -> Path:
    return Path(state_dir).expanduser() / "environment.json"


def build_environment_catalog(
    *,
    repo_root: str | Path,
    approved_roots: Iterable[str | Path],
    shell: str | None = None,
    cwd: str | Path | None = None,
    interpreters: dict[str, dict[str, str]] | None = None,
    commands: dict[str, dict[str, str]] | None = None,
    apps: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    repo_root_path = Path(repo_root).expanduser().resolve()
    cwd_path = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    normalized_roots = sorted(
        {
            str(Path(root).expanduser().resolve())
            for root in approved_roots
        }
        | {str(repo_root_path)}
    )
    return {
        "version": ENVIRONMENT_CATALOG_VERSION,
        "host": {
            "platform": _platform_name(),
            "shell": str(shell or os.environ.get("SHELL", "")).strip(),
            "cwd": str(cwd_path),
        },
        "roots": {
            "repo_root": str(repo_root_path),
            "approved": normalized_roots,
        },
        "interpreters": dict(interpreters or {}),
        "commands": dict(commands or {}),
        "apps": dict(apps or {}),
    }


def load_environment_catalog(path: str | Path) -> dict[str, Any] | None:
    target = Path(path).expanduser()
    if not target.exists():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    return normalize_environment_catalog(data)


def save_environment_catalog(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser()
    normalized = normalize_environment_catalog(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(target)
    return normalized


def normalize_environment_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("environment catalog payload must be a dict")

    version = payload.get("version", ENVIRONMENT_CATALOG_VERSION)
    if not isinstance(version, int):
        raise ValueError("environment catalog version must be an integer")

    host = payload.get("host", {})
    if not isinstance(host, dict):
        raise ValueError("environment catalog host must be a dict")

    roots = payload.get("roots", {})
    if not isinstance(roots, dict):
        raise ValueError("environment catalog roots must be a dict")

    interpreters = payload.get("interpreters", {})
    commands = payload.get("commands", {})
    apps = payload.get("apps", {})
    for key, value in (
        ("interpreters", interpreters),
        ("commands", commands),
        ("apps", apps),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"environment catalog {key} must be a dict")

    repo_root = str(roots.get("repo_root", "")).strip()
    approved_raw = roots.get("approved", [])
    if approved_raw in (None, ""):
        approved_raw = []
    if not isinstance(approved_raw, list):
        raise ValueError("environment catalog roots.approved must be a list")
    approved = sorted({str(item).strip() for item in approved_raw if str(item).strip()})
    if repo_root and repo_root not in approved:
        approved.append(repo_root)
        approved.sort()

    normalized = {
        "version": version,
        "host": {
            "platform": str(host.get("platform", _platform_name())).strip(),
            "shell": str(host.get("shell", "")).strip(),
            "cwd": str(host.get("cwd", "")).strip(),
        },
        "roots": {
            "repo_root": repo_root,
            "approved": approved,
        },
        "interpreters": _normalize_capability_mapping(interpreters, required_keys=("preferred", "description")),
        "commands": _normalize_capability_mapping(commands, required_keys=("path", "description")),
        "apps": _normalize_capability_mapping(apps, required_keys=("bundle_id", "description")),
    }
    return normalized


def _normalize_capability_mapping(
    payload: dict[str, Any],
    *,
    required_keys: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for name, details in payload.items():
        key = str(name).strip()
        if not key or not isinstance(details, dict):
            continue
        item = {field: str(details.get(field, "")).strip() for field in required_keys}
        for extra_key, extra_value in details.items():
            if extra_key in item:
                continue
            item[str(extra_key)] = str(extra_value).strip()
        normalized[key] = item
    return normalized


def _platform_name() -> str:
    return "macOS" if platform.system() == "Darwin" else platform.system()
