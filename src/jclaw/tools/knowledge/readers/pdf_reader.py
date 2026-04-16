from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from jclaw.tools.knowledge.models import ExtractedDocument


class PdfReaderTool:
    name = "pdf"

    def supports(self, path: Path, *, mime_type: str, suffix: str) -> bool:
        return suffix == ".pdf" or mime_type == "application/pdf"

    def extract(self, path: Path, *, max_bytes: int) -> ExtractedDocument:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted:
                parts.append(extracted)
            joined = "\n\n".join(parts)
            if len(joined.encode("utf-8")) >= max_bytes:
                break
        text = "\n\n".join(parts)
        encoded = text.encode("utf-8")[:max_bytes]
        safe_text = encoded.decode("utf-8", errors="ignore")
        return ExtractedDocument(
            path=str(path),
            file_type="pdf",
            title=path.name,
            text=safe_text,
            metadata={
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "page_count": len(reader.pages),
            },
            warnings=[] if safe_text.strip() else ["No readable PDF text extracted."],
        )

