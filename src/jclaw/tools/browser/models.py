from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

