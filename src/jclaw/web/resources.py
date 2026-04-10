from __future__ import annotations

from pathlib import Path

from jclaw.core.config import repo_root


def load_web_asset(name: str) -> str:
    path = repo_root() / "web" / name
    return path.read_text(encoding="utf-8")
