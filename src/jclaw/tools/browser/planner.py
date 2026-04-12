from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jclaw.tools.browser.models import BrowserAction


@dataclass(slots=True)
class PlannedStep:
    action: str
    params: dict[str, Any]
    reason: str


class BrowserPlanner:
    def next_step(self, objective: str, observation: dict[str, Any], history: list[dict[str, Any]]) -> PlannedStep:
        if not history:
            return PlannedStep(
                action="open_url",
                params={"url": observation.get("url") or "about:blank"},
                reason=f"Start objective: {objective}",
            )
        return PlannedStep(
            action="read_page",
            params={"session_id": observation.get("session_id", "")},
            reason="Refresh page understanding before planning another step.",
        )

