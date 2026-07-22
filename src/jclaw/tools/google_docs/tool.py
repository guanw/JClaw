from __future__ import annotations

from typing import Any

from jclaw.core.config import GoogleDocsConfig
from jclaw.core.db import Database
from jclaw.tools.base import ActionSpec, RuntimeState, ToolContext, ToolResult, build_tool_description
from jclaw.tools.google.auth import GoogleOAuthManager
from jclaw.tools.google_docs.client import GoogleDocsClient


class GoogleDocsTool:
    name = "google_docs"

    def __init__(
        self,
        config: GoogleDocsConfig,
        *,
        db: Database | None = None,
        auth: GoogleOAuthManager | None = None,
        client: GoogleDocsClient | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.auth = auth or GoogleOAuthManager(
            oauth_client_path=config.oauth_client_path,
            token_dir=config.token_dir,
            support_name="Google Docs",
        )
        self.client = client or GoogleDocsClient(self.auth, token_name="default", scopes=config.scopes)

    def describe(self) -> dict[str, Any]:
        return build_tool_description(
            name=self.name,
            description=(
                "Inspect, copy, and safely update Google Docs documents through the Google Docs and Drive APIs. "
                "Use update_document for natural-language requests to fill or change known placeholders in a Google Doc."
            ),
            actions=self._action_specs(),
            controller_guidance=(
                "For user requests like 'update/fill/change this Google Doc', infer exact replacements from the user's "
                "natural-language values and the latest inspected placeholder list. If placeholder keys are not known, "
                "call inspect_document first. Map plain field names to matching placeholders, for example tenant name -> "
                "[tenant_names], lease start -> [lease_start], property address -> [property_address]. Do not invent "
                "replacement values; only infer keys and values that are explicit in the user request or observations."
            ),
        )

    def controller_output(self, action: str, result: ToolResult) -> dict[str, Any]:
        if action == "copy_document":
            return self._copy_controller_output(result)
        if action in {"update_document", "apply_update_document"}:
            return self._update_controller_output(result)
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
        if action in {"update_document", "apply_update_document"}:
            return self._format_update_result(result)
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

    def should_return_direct(self, action: str, result: ToolResult, runtime: RuntimeState | None = None) -> bool:
        if not result.ok:
            return False
        if action == "inspect_document":
            request = (runtime.request if runtime is not None else "").lower()
            update_terms = ("update", "fill", "edit", "change", "replace")
            return not any(term in request for term in update_terms)
        return action == "copy_document"

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
            "update_document": self._update_document,
            "apply_update_document": self._apply_update_document,
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
        if action == "update_document" and not self._document_param(materialized):
            latest_copy = runtime.artifacts_by_type.get("google_doc_copy")
            if isinstance(latest_copy, dict):
                document_id = str(latest_copy.get("document_id", "")).strip()
                if document_id:
                    materialized["document"] = document_id
            if not self._document_param(materialized):
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

    def _update_document(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        document = self._document_param(params)
        if not document:
            return ToolResult(ok=False, summary="update_document requires a Google Docs URL or document id.", data={})
        replacements = self._replacements_param(params)
        if not replacements:
            return ToolResult(ok=False, summary="update_document requires a non-empty replacements object.", data={})
        copy_name = str(params.get("copy_name") or params.get("name") or "").strip()
        if self.db is None:
            return ToolResult(ok=False, summary="update_document requires database-backed approval support.", data={})
        payload = {
            "summary": f"Prepared Google Doc update with {len(replacements)} replacement(s).",
            "document": document,
            "copy_name": copy_name,
            "replacements": replacements,
            "continuation": {
                "tool": self.name,
                "approve_action": "apply_update_document",
                "params": {},
            },
        }
        request = self.db.create_approval_request(
            kind="google_doc_update",
            chat_id=ctx.chat_id,
            root_path=document,
            capabilities=("write",),
            objective=f"Copy Google Doc and apply {len(replacements)} replacement(s).",
            payload=payload,
        )
        return ToolResult(
            ok=True,
            summary=f"Prepared Google Doc update preview. Approval required: {request.request_id}",
            data={
                "request_id": request.request_id,
                "request_kind": request.kind,
                "document": document,
                "copy_name": copy_name,
                "replacement_count": len(replacements),
                "replacements": replacements,
                "copy_before_update": True,
            },
            needs_confirmation=True,
        )

    def _apply_update_document(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        del ctx
        if self.db is None:
            return ToolResult(ok=False, summary="apply_update_document requires database-backed approval support.", data={})
        request_id = str(params.get("request_id", "")).strip()
        if not request_id:
            return ToolResult(ok=False, summary="apply_update_document requires a request_id.", data={})
        request = self.db.get_approval_request(request_id)
        if request is None or request.kind != "google_doc_update":
            return ToolResult(ok=False, summary="Pending Google Doc update request not found.", data={})
        self.db.update_approval_request_status(request.request_id, "approved")
        try:
            document = str(request.payload.get("document", "")).strip()
            copy_name = str(request.payload.get("copy_name", "")).strip()
            replacements = self._replacements_param(request.payload)
            copied = self.client.copy_document(document, name=copy_name)
            copied_document_id = str(copied.get("document_id", "")).strip()
            if not copied_document_id:
                raise RuntimeError("Google Docs copy did not return a document id.")
            update = self.client.replace_text(copied_document_id, replacements)
        except Exception:
            self.db.update_approval_request_status(request.request_id, "failed")
            raise
        self.db.update_approval_request_status(request.request_id, "applied")
        replacement_count = int(update.get("replacement_count") or 0)
        request_count = int(update.get("request_count") or len(replacements))
        data = {
            "request_id": request.request_id,
            "source_document": document,
            "document_id": copied_document_id,
            "title": str(copied.get("title", "")).strip(),
            "url": str(copied.get("url", "")).strip(),
            "replacement_count": replacement_count,
            "request_count": request_count,
            "replacements": replacements,
            "artifacts": {
                "google_doc_copy:latest": copied,
                "google_doc_update:latest": {
                    "request_id": request.request_id,
                    "source_document": document,
                    "document_id": copied_document_id,
                    "url": str(copied.get("url", "")).strip(),
                    "replacement_count": replacement_count,
                    "request_count": request_count,
                    "replacements": replacements,
                },
            },
        }
        if copied_document_id:
            data["artifacts"][f"google_doc_copy:{copied_document_id}"] = copied
            data["artifacts"][f"google_doc_update:{copied_document_id}"] = data["artifacts"]["google_doc_update:latest"]
        return ToolResult(
            ok=True,
            summary=f"Copied and updated Google Doc with {replacement_count} replacement occurrence(s).",
            data=data,
        )

    def _copy_controller_output(self, result: ToolResult) -> dict[str, Any]:
        data = result.data if isinstance(result.data, dict) else {}
        payload: dict[str, Any] = {}
        for key in ("document_id", "title", "url", "source_document"):
            if key in data:
                payload[key] = data[key]
        return payload

    def _update_controller_output(self, result: ToolResult) -> dict[str, Any]:
        data = result.data if isinstance(result.data, dict) else {}
        payload: dict[str, Any] = {}
        for key in (
            "request_id",
            "document",
            "source_document",
            "document_id",
            "title",
            "url",
            "replacement_count",
            "request_count",
            "copy_before_update",
        ):
            if key in data:
                payload[key] = data[key]
        replacements = data.get("replacements")
        if isinstance(replacements, dict):
            payload["replacement_keys"] = list(replacements)[:20]
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

    def _format_update_result(self, result: ToolResult) -> str:
        data = result.data if isinstance(result.data, dict) else {}
        lines = [result.summary]
        request_id = str(data.get("request_id", "")).strip()
        if result.needs_confirmation and request_id:
            lines.append(f"Use /approve {request_id} to copy the document and apply these replacements.")
            lines.append(f"Use /deny {request_id} to cancel.")
        title = str(data.get("title", "")).strip()
        document_id = str(data.get("document_id", "")).strip()
        url = str(data.get("url", "")).strip()
        if title:
            lines.append(f"Title: {title}")
        if document_id:
            lines.append(f"Document ID: {document_id}")
        if url:
            lines.append(f"URL: {url}")
        replacements = data.get("replacements")
        if isinstance(replacements, dict) and replacements:
            lines.append("Replacements:")
            for source, target in list(replacements.items())[:20]:
                lines.append(f"- {source} -> {target}")
        if "replacement_count" in data and result.needs_confirmation:
            lines.append(f"Replacement entries prepared: {int(data.get('replacement_count') or 0)}")
        elif "replacement_count" in data:
            lines.append(f"Replacement occurrences changed: {int(data.get('replacement_count') or 0)}")
        return "\n".join(lines)

    def _document_param(self, params: dict[str, Any]) -> str:
        return str(params.get("document") or params.get("document_id") or params.get("url") or "").strip()

    def _replacements_param(self, params: dict[str, Any]) -> dict[str, str]:
        raw = params.get("replacements") or params.get("replacement_map") or {}
        if not isinstance(raw, dict):
            return {}
        return {str(source): str(target) for source, target in raw.items() if str(source)}

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
            "update_document": ActionSpec(
                tool=self.name,
                action="update_document",
                description=(
                    "Prepare a safe Google Docs update from natural language by copying the source document first, "
                    "then replacing exact placeholders or text in the copied document after user approval. "
                    "Use this when the user asks to update, fill, edit, or change a Google Doc and provides replacement values. "
                    "Infer the replacements map from natural language by matching user-provided field values to known "
                    "placeholder keys from inspect_document, such as tenant name -> [tenant_names]. Inspect the document first "
                    "when the relevant placeholders are not already available."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "document": {
                            "type": "string",
                            "description": "Google Docs URL or raw document id. Defaults to the latest inspected or copied Google Doc when available.",
                        },
                        "copy_name": {
                            "type": "string",
                            "description": "Optional title for the copied document that will receive the updates.",
                        },
                        "replacements": {
                            "type": "object",
                            "description": "Map of exact placeholder/text to replacement text, such as {'[tenant_names]': 'Alice Zhang'}.",
                        },
                    },
                    "required": ["replacements"],
                },
                writes=True,
                requires_artifacts=("google_doc", "google_doc_copy"),
                produces_artifacts=("google_doc_update", "google_doc_copy"),
                binding_inputs=("document",),
                requires_confirmation=True,
            ),
            "apply_update_document": ActionSpec(
                tool=self.name,
                action="apply_update_document",
                description="Apply an approved Google Docs update request by copying the source document and replacing text in the copy.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string"},
                    },
                    "required": ["request_id"],
                },
                writes=True,
                produces_artifacts=("google_doc_update", "google_doc_copy"),
            ),
        }
