from __future__ import annotations

import mimetypes
from pathlib import Path

from jclaw.tools.knowledge.base import KnowledgeReader
from jclaw.tools.knowledge.readers.pdf_reader import PdfReaderTool
from jclaw.tools.knowledge.readers.text_reader import TextReader


class KnowledgeReaderRegistry:
    def __init__(self) -> None:
        self._readers: list[KnowledgeReader] = [
            TextReader(),
            PdfReaderTool(),
        ]

    def get_reader(self, path: Path) -> KnowledgeReader | None:
        mime_type, _ = mimetypes.guess_type(path.name)
        mime = mime_type or ""
        suffix = path.suffix.lower()
        for reader in self._readers:
            if reader.supports(path, mime_type=mime, suffix=suffix):
                return reader
        return None
