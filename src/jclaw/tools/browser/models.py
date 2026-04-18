from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class Target:
    selector: str | None = None
    role: str | None = None
    name: str | None = None
    text: str | None = None
    xpath: str | None = None


@dataclass(slots=True)
class BrowserAction:
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class BrowserObservation:
    session_id: str = ""
    tab_id: str = ""
    url: str = ""
    title: str = ""
    text: str = ""
    mode: str = "playwright"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InspectedElement:
    id: str
    role: str
    text: str = ""
    href: str = ""
    area: str = "body"
    clickable: bool = False
    visible: bool = True
    selector_hint: str = ""
    score_hint: float = 0.0


class BrowserReasoner(Protocol):
    def choose_link(self, objective: str, page_data: dict[str, Any]) -> str | None:
        ...

    def decide_next_action(
        self,
        objective: str,
        page_data: dict[str, Any],
        sources: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        ...
