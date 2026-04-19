from __future__ import annotations

from pathlib import Path
from typing import Callable

from jclaw.tools.knowledge.models import ExtractedDocument


class ImageReader:
    name = "image"
    SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff", ".tif", ".icns"}

    def __init__(
        self,
        analyze_image: Callable[[Path], dict[str, object] | None] | None,
    ) -> None:
        self._analyze_image = analyze_image

    def supports(self, path: Path, *, mime_type: str, suffix: str) -> bool:
        return suffix in self.SUPPORTED_SUFFIXES or mime_type.startswith("image/")

    def extract(self, path: Path, *, max_bytes: int) -> ExtractedDocument:
        if self._analyze_image is None:
            return ExtractedDocument(
                path=str(path),
                file_type="image",
                title=path.name,
                text="",
                metadata={
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                },
                warnings=["Image analysis callback is not configured."],
            )
        result = self._analyze_image(path)
        text = ""
        warnings: list[str] = []
        if isinstance(result, dict):
            text = str(result.get("text", "")).strip()
            raw_warnings = result.get("warnings", [])
            if isinstance(raw_warnings, list):
                warnings = [str(item) for item in raw_warnings if str(item).strip()]
        return ExtractedDocument(
            path=str(path),
            file_type="image",
            title=path.name,
            text=text[:max_bytes],
            metadata={
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            },
            warnings=warnings if text else warnings or ["No readable image description extracted."],
        )
