from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    path: str
    text: str
    start_offset: int
    end_offset: int


@dataclass(slots=True)
class ExtractedDocument:
    path: str
    file_type: str
    title: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

