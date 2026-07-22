from __future__ import annotations

from typing import Any

from jclaw.core.config import GoogleDocsConfig
from jclaw.tools.base import ActionSpec, RuntimeState, ToolContext, ToolResult, build_tool_description
from jclaw.tools.google.auth import GoogleOAuthManager
from jclaw.tools.google_docs.client import GoogleDocsClient


class GoogleDocsTool:
    name = "google_docs"

    def __init__(
        self,
        config: GoogleDocsConfig,
        *,
        auth: GoogleOAuthManager | None = None,
        client: GoogleDocsClient | None = None,
    ) -> None:
        self.config = config
        self.auth = auth or GoogleOAuthManager(
            oauth_client_path=config.oauth_client_path,
            token_dir=config.token_dir,
            support_name="Google Docs",
        )
        self.client = client or GoogleDocsClient(self.auth, token_name="default", scopes=config.scopes)

    def describe(self) -> dict[str, Any]:
        return build_tool_description(
            name=self.name,
            description="Inspect and copy Google Docs documents through the Google Docs and Drive APIs.",
            actions=self._action_specs(),
        )

    def controller_output(self, action: str, result: ToolResult) -> dict[str, Any]:
        if action == "copy_document":
            return self._copy_controller_output(result)
        if action != "inspect_document":
            return {}
        data = result.data if isinstance(result.data, dict) else {}
        payload: dict[str, Any] = {}
        for key in (
            "document_id",
            "title",
            "placeholder_count",
            "placeholders",
            "body_element_count",
            "text_truncated",
        ):
            if key in data:
                payload[key] = data[key]
        preview = str(data.get("text_preview", "")).strip()
        if preview:
            payload["text_preview"] = preview[:2000]
        return payload

    def format_result(self, action: str, result: ToolResult) -> str:
        if action == "copy_document":
            return self._format_copy_result(result)
        if action != "inspect_document":
            return result.summary
        data = result.data if isinstance(result.data, dict) else {}
        lines = [result.summary]
        title = str(data.get("title", "")).strip()
        document_id = str(data.get("document_id", "")).strip()
        if title:
            lines.append(f"Title: {title}")
        if document_id:
            lines.append(f"Document ID: {document_id}")
        lines.append(f"Placeholders: {int(data.get('placeholder_count') or 0)}")
        placeholders = data.get("placeholders")
        if isinstance(placeholders, list) and placeholders:
            lines.append("First placeholders:")
            for item in placeholders[:30]:
                lines.append(f"- {item}")
        preview = str(data.get("text_preview", "")).strip()
        if preview:
            lines.append("Preview:")
            lines.append(preview[:2000])
        if data.get("text_truncated"):
            lines.append("Preview truncated.")
        return "\n".join(lines)

    def should_return_direct(self, action: str, result: ToolResult) -> bool:
        return action in {"inspect_document", "copy_document"} and result.ok

    def artifact_preview_limits(self) -> dict[str, dict[str, int]]:
        return {
            "google_doc": {
                "text_preview": 20_000,
            }
        }

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "inspect_document": self._inspect_document,
            "copy_document": self._copy_document,
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError(f"unsupported google_docs action: {action}")
        return handler(params, ctx)

    def materialize_params(
        self,
        action: str,
        params: dict[str, Any],
        runtime: RuntimeState,
    ) -> dict[str, Any]:
        materialized = dict(params)
        if action == "copy_document" and not self._document_param(materialized):
            latest_doc = runtime.artifacts_by_type.get("google_doc")
            if isinstance(latest_doc, dict):
                document_id = str(latest_doc.get("document_id", "")).strip()
                if document_id:
                    materialized["document"] = document_id
        return materialized

    def _inspect_document(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        del ctx
        document = self._document_param(params)
        if not document:
            return ToolResult(ok=False, summary="inspect_document requires a Google Docs URL or document id.", data={})
        snapshot = self.client.get_document(document)
        document_id = str(snapshot.get("document_id", "")).strip()
        title = str(snapshot.get("title", "")).strip() or "Untitled document"
        data = {
            **snapshot,
            "artifacts": {
                "google_doc:latest": snapshot,
            },
        }
        if document_id:
            data["artifacts"][f"google_doc:{document_id}"] = snapshot
        return ToolResult(
            ok=True,
            summary=f"Inspected Google Doc '{title}' and found {int(snapshot.get('placeholder_count') or 0)} placeholder(s).",
            data=data,
        )

    def _copy_document(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        del ctx
        document = self._document_param(params)
        if not document:
            return ToolResult(ok=False, summary="copy_document requires a Google Docs URL or document id.", data={})
        name = str(params.get("name") or params.get("title") or "").strip()
        copied = self.client.copy_document(document, name=name)
        document_id = str(copied.get("document_id", "")).strip()
        title = str(copied.get("title", "")).strip() or "Copied Google Doc"
        data = {
            **copied,
            "source_document": document,
            "artifacts": {
                "google_doc_copy:latest": copied,
            },
        }
        if document_id:
            data["artifacts"][f"google_doc_copy:{document_id}"] = copied
        return ToolResult(
            ok=True,
            summary=f"Copied Google Doc to '{title}'.",
            data=data,
        )

    def _copy_controller_output(self, result: ToolResult) -> dict[str, Any]:
        data = result.data if isinstance(result.data, dict) else {}
        payload: dict[str, Any] = {}
        for key in ("document_id", "title", "url", "source_document"):
            if key in data:
                payload[key] = data[key]
        return payload

    def _format_copy_result(self, result: ToolResult) -> str:
        data = result.data if isinstance(result.data, dict) else {}
        lines = [result.summary]
        title = str(data.get("title", "")).strip()
        document_id = str(data.get("document_id", "")).strip()
        url = str(data.get("url", "")).strip()
        if title:
            lines.append(f"Title: {title}")
        if document_id:
            lines.append(f"Document ID: {document_id}")
        if url:
            lines.append(f"URL: {url}")
        return "\n".join(lines)

    def _document_param(self, params: dict[str, Any]) -> str:
        return str(params.get("document") or params.get("document_id") or params.get("url") or "").strip()

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "inspect_document": ActionSpec(
                tool=self.name,
                action="inspect_document",
                description="Inspect a Google Docs document by URL or document id and return title, placeholders, and a text preview.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "document": {
                            "type": "string",
                            "description": "Google Docs URL or raw document id.",
                        }
                    },
                    "required": ["document"],
                },
                reads=True,
                produces_artifacts=("google_doc",),
            ),
            "copy_document": ActionSpec(
                tool=self.name,
                action="copy_document",
                description="Copy a Google Docs document by URL or document id and return the new editable document URL.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "document": {
                            "type": "string",
                            "description": "Google Docs URL or raw document id. Defaults to the latest inspected Google Doc when available.",
                        },
                        "name": {
                            "type": "string",
                            "description": "Optional title for the copied document.",
                        },
                    },
                },
                writes=True,
                requires_artifacts=("google_doc",),
                produces_artifacts=("google_doc_copy",),
                binding_inputs=("document",),
            ),
        }
