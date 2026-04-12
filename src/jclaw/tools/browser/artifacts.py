from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BrowserArtifact:
    artifact_id: str
    kind: str
    path: str
    created_at: str
    session_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "created_at": self.created_at,
            "session_id": self.session_id,
        }


class BrowserArtifactStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_artifact(self, *, session_id: str, kind: str, suffix: str) -> BrowserArtifact:
        artifact_id = f"art_{uuid.uuid4().hex[:12]}"
        path = self.base_dir / f"{artifact_id}.{suffix.lstrip('.')}"
        return BrowserArtifact(
            artifact_id=artifact_id,
            kind=kind,
            path=str(path),
            created_at=_utc_now(),
            session_id=session_id,
        )
