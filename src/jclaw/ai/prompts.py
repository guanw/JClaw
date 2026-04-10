from __future__ import annotations

from pathlib import Path

from jclaw.core.config import repo_root


def load_system_prompt(prompt_files: tuple[str, ...]) -> str:
    prompts_dir = repo_root() / "prompts"
    sections: list[str] = []
    for name in prompt_files:
        path = Path(name)
        if not path.is_absolute():
            path = prompts_dir / name
        if path.exists():
            sections.append(path.read_text(encoding="utf-8").strip())
    sections.append("Respond briefly unless the user explicitly asks for detail.")
    return "\n\n".join(section for section in sections if section)

