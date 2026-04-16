from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jclaw.tools.knowledge.models import ExtractedDocument


class KnowledgeReader(Protocol):
    name: str

    def supports(self, path: Path, *, mime_type: str, suffix: str) -> bool:
        ...

    def extract(self, path: Path, *, max_bytes: int) -> ExtractedDocument:
        ...

