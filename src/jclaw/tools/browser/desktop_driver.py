from __future__ import annotations

from typing import Any
from dataclasses import asdict

from jclaw.tools.browser.models import Target


class DesktopBrowserDriver:
    mode = "desktop"

    def open_url(self, session_id: str, url: str, *, new_tab: bool) -> dict[str, Any]:
        return {"session_id": session_id, "url": url, "new_tab": new_tab, "mode": self.mode}

    def click(self, session_id: str, target: Target) -> dict[str, Any]:
        return {"session_id": session_id, "target": asdict(target), "mode": self.mode}

    def type(self, session_id: str, target: Target, text: str, *, submit: bool) -> dict[str, Any]:
        return {"session_id": session_id, "target": asdict(target), "text": text, "submit": submit, "mode": self.mode}

    def scroll(self, session_id: str, *, direction: str, amount: int) -> dict[str, Any]:
        return {"session_id": session_id, "direction": direction, "amount": amount, "mode": self.mode}

    def wait_for(self, session_id: str, target: Target | None, timeout_ms: int) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "target": None if target is None else asdict(target),
            "timeout_ms": timeout_ms,
            "mode": self.mode,
        }

    def read_page(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "url": "",
            "title": "",
            "text": "",
            "page_kind": "unknown",
            "elements": [],
            "links": [],
            "forms": [],
            "mode": self.mode,
        }

    def screenshot(self, session_id: str, *, full_page: bool) -> dict[str, Any]:
        return {"session_id": session_id, "full_page": full_page, "mode": self.mode}
