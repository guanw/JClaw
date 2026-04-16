from __future__ import annotations

from pathlib import Path

from jclaw.tools.knowledge.models import ExtractedDocument


TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".env",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".log",
    ".md",
    ".mjs",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".svg",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


class TextReader:
    name = "text"

    def supports(self, path: Path, *, mime_type: str, suffix: str) -> bool:
        if suffix in TEXT_SUFFIXES:
            return True
        return mime_type.startswith("text/")

    def extract(self, path: Path, *, max_bytes: int) -> ExtractedDocument:
        raw = path.read_bytes()[:max_bytes]
        text = raw.decode("utf-8", errors="ignore")
        return ExtractedDocument(
            path=str(path),
            file_type="text",
            title=path.name,
            text=text,
            metadata={
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            },
        )

