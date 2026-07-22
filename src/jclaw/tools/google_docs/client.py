from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from jclaw.tools.google.auth import GoogleOAuthManager

DOCUMENT_URL_RE = re.compile(r"/document/d/([A-Za-z0-9_-]+)")
PLACEHOLDER_RE = re.compile(r"(?:\{\{[^{}\n]{1,120}\}\}|\[[^\[\]\n]{1,120}\])")


def extract_document_id(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Google Docs document id or URL is required.")
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        match = DOCUMENT_URL_RE.search(parsed.path)
        if match:
            return match.group(1)
        query_id = parse_qs(parsed.query).get("id", [""])[0].strip()
        if query_id:
            return query_id
        raise ValueError("Could not extract a Google Docs document id from the URL.")
    if "/" in text or "?" in text:
        raise ValueError("Could not extract a Google Docs document id from the input.")
    return text


def _extract_text_from_structural_elements(elements: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for element in elements:
        paragraph = element.get("paragraph")
        if isinstance(paragraph, dict):
            for item in paragraph.get("elements", []) or []:
                text_run = item.get("textRun", {}) if isinstance(item, dict) else {}
                content = text_run.get("content", "") if isinstance(text_run, dict) else ""
                if content:
                    chunks.append(str(content))
        table = element.get("table")
        if isinstance(table, dict):
            for row in table.get("tableRows", []) or []:
                for cell in row.get("tableCells", []) or []:
                    chunks.append(_extract_text_from_structural_elements(cell.get("content", []) or []))
        table_of_contents = element.get("tableOfContents")
        if isinstance(table_of_contents, dict):
            chunks.append(_extract_text_from_structural_elements(table_of_contents.get("content", []) or []))
    return "".join(chunks)


def normalize_document(document: dict[str, Any], *, text_preview_chars: int = 4000) -> dict[str, Any]:
    body = document.get("body", {}) if isinstance(document.get("body"), dict) else {}
    content = body.get("content", []) if isinstance(body.get("content"), list) else []
    text = _extract_text_from_structural_elements(content)
    placeholders = sorted(set(PLACEHOLDER_RE.findall(text)))
    preview = text.strip()
    return {
        "document_id": str(document.get("documentId", "")),
        "title": str(document.get("title", "")),
        "revision_id": str(document.get("revisionId", "")),
        "text_preview": preview[:text_preview_chars],
        "text_truncated": len(preview) > text_preview_chars,
        "placeholders": placeholders,
        "placeholder_count": len(placeholders),
        "body_element_count": len(content),
    }


class GoogleDocsClient:
    def __init__(self, auth: GoogleOAuthManager, *, token_name: str = "default", scopes: tuple[str, ...]) -> None:
        self.auth = auth
        self.token_name = token_name
        self.scopes = scopes

    def get_document(self, document: str) -> dict[str, Any]:
        document_id = extract_document_id(document)
        raw = self._docs_service().documents().get(documentId=document_id).execute()
        return normalize_document(raw)

    def copy_document(self, document: str, *, name: str = "") -> dict[str, str]:
        document_id = extract_document_id(document)
        body = {"name": name} if name.strip() else {}
        copied = self._drive_service().files().copy(fileId=document_id, body=body).execute()
        new_id = str(copied.get("id", ""))
        return {
            "document_id": new_id,
            "title": str(copied.get("name", "")),
            "url": f"https://docs.google.com/document/d/{new_id}/edit" if new_id else "",
        }

    def replace_text(self, document: str, replacements: dict[str, str]) -> dict[str, Any]:
        document_id = extract_document_id(document)
        requests = [
            {
                "replaceAllText": {
                    "containsText": {
                        "text": str(source),
                        "matchCase": True,
                    },
                    "replaceText": str(target),
                }
            }
            for source, target in replacements.items()
            if str(source)
        ]
        response = self._docs_service().documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests},
        ).execute()
        replies = response.get("replies", []) or []
        replacement_counts = [
            int((reply.get("replaceAllText") or {}).get("occurrencesChanged", 0))
            for reply in replies
            if isinstance(reply, dict)
        ]
        return {
            "document_id": document_id,
            "replacement_count": sum(replacement_counts),
            "request_count": len(requests),
            "revision_id": str((response.get("writeControl") or {}).get("requiredRevisionId", "")),
        }

    def _docs_service(self) -> Any:
        return self.auth.build_service("docs", "v1", self._credentials())

    def _drive_service(self) -> Any:
        return self.auth.build_service("drive", "v3", self._credentials())

    def _credentials(self) -> Any:
        return self.auth.load_credentials(self.token_name, self.scopes)
