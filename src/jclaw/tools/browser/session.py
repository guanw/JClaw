from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BrowserSession:
    session_id: str
    created_at: str
    mode: str
    profile_dir: str
    current_tab_id: str | None = None
    current_url: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "mode": self.mode,
            "profile_dir": self.profile_dir,
            "current_tab_id": self.current_tab_id,
            "current_url": self.current_url,
        }


class BrowserSessionStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, BrowserSession] = {}

    def create_session(self, *, visible: bool = True, persistent: bool = True, mode: str = "playwright") -> BrowserSession:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        profile_dir = self.base_dir / session_id
        if persistent:
            profile_dir.mkdir(parents=True, exist_ok=True)
        session = BrowserSession(
            session_id=session_id,
            created_at=_utc_now(),
            mode=mode if visible else f"{mode}-headless",
            profile_dir=str(profile_dir),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> BrowserSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown browser session: {session_id}") from exc

    def close_session(self, session_id: str) -> None:
        self.get_session(session_id)
        del self._sessions[session_id]

    def list_sessions(self) -> list[BrowserSession]:
        return list(self._sessions.values())
