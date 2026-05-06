from __future__ import annotations

from typing import Any, Callable

from jclaw.core.config import NotionConfig
from jclaw.tools.base import ActionSpec, ToolContext, ToolResult, append_list_section, build_tool_description
from jclaw.tools.notion.client import (
    NotionClient,
    NotionConfigError,
    NotionDisabledError,
    NotionError,
    NotionNotFoundError,
    NotionRateLimitedError,
    NotionUnauthorizedError,
)


def _plain_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item.get("plain_text", "")) for item in value if isinstance(item, dict)).strip()
    return str(value or "").strip()


def _normalize_parent(parent: Any) -> dict[str, str]:
    if not isinstance(parent, dict):
        return {}
    kind = str(parent.get("type", "")).strip()
    parent_id = str(parent.get(kind, "")).strip() if kind else ""
    if not parent_id:
        parent_id = str(parent.get("page_id") or parent.get("database_id") or parent.get("workspace", "")).strip()
    return {"type": kind or "unknown", "id": parent_id}


def _extract_page_title(page: dict[str, Any]) -> str:
    properties = page.get("properties", {})
    if not isinstance(properties, dict):
        return "Untitled"
    for value in properties.values():
        if not isinstance(value, dict):
            continue
        if value.get("type") == "title":
            title = _plain_text(value.get("title"))
            if title:
                return title
    return "Untitled"


def normalize_notion_page(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_id": str(page.get("id", "")),
        "title": _extract_page_title(page),
        "url": str(page.get("url", "")),
        "last_edited_time": str(page.get("last_edited_time", "")),
        "parent": _normalize_parent(page.get("parent")),
    }


def _normalize_property_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    kind = str(value.get("type", "")).strip()
    if kind == "title":
        return _plain_text(value.get("title"))
    if kind == "rich_text":
        return _plain_text(value.get("rich_text"))
    if kind in {"number", "checkbox", "url", "email", "phone_number"}:
        return value.get(kind)
    if kind in {"select", "status"}:
        item = value.get(kind)
        return str(item.get("name", "")).strip() if isinstance(item, dict) else ""
    if kind == "multi_select":
        items = value.get("multi_select")
        if isinstance(items, list):
            return [str(item.get("name", "")).strip() for item in items if isinstance(item, dict)]
        return []
    if kind == "date":
        item = value.get("date")
        if not isinstance(item, dict):
            return {}
        return {
            "start": str(item.get("start", "")).strip(),
            "end": str(item.get("end", "")).strip(),
        }
    if kind == "people":
        items = value.get("people")
        if isinstance(items, list):
            return [
                str(item.get("name") or item.get("id") or "").strip()
                for item in items
                if isinstance(item, dict)
            ]
        return []
    if kind == "relation":
        items = value.get("relation")
        if isinstance(items, list):
            return [str(item.get("id", "")).strip() for item in items if isinstance(item, dict)]
        return []
    if kind == "files":
        items = value.get("files")
        if isinstance(items, list):
            return [
                str(item.get("name") or item.get("type") or "").strip()
                for item in items
                if isinstance(item, dict)
            ]
        return []
    if kind == "formula":
        formula = value.get("formula")
        if not isinstance(formula, dict):
            return {}
        formula_kind = str(formula.get("type", "")).strip()
        return formula.get(formula_kind)
    return value.get(kind) if kind else ""


def _normalize_page_properties(page: dict[str, Any]) -> dict[str, Any]:
    raw = page.get("properties", {})
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        clean_key = str(key).strip()
        if not clean_key:
            continue
        normalized[clean_key] = _normalize_property_value(value)
    return normalized


def _block_text(block_data: dict[str, Any], field: str) -> str:
    value = block_data.get(field)
    if not isinstance(value, dict):
        return ""
    rich_text = value.get("rich_text")
    return _plain_text(rich_text)


def _normalize_block(block: dict[str, Any]) -> dict[str, Any]:
    block_type = str(block.get("type", "")).strip()
    normalized: dict[str, Any] = {
        "block_id": str(block.get("id", "")).strip(),
        "type": block_type or "unknown",
        "has_children": bool(block.get("has_children")),
    }
    if block_type == "paragraph":
        normalized["text"] = _block_text(block, "paragraph")
    elif block_type in {"heading_1", "heading_2", "heading_3"}:
        normalized["text"] = _block_text(block, block_type)
    elif block_type == "bulleted_list_item":
        normalized["text"] = _block_text(block, "bulleted_list_item")
    elif block_type == "numbered_list_item":
        normalized["text"] = _block_text(block, "numbered_list_item")
    elif block_type == "to_do":
        todo = block.get("to_do")
        if isinstance(todo, dict):
            normalized["text"] = _plain_text(todo.get("rich_text"))
            normalized["checked"] = bool(todo.get("checked"))
        else:
            normalized["text"] = ""
            normalized["checked"] = False
    else:
        normalized["unsupported"] = True
    return normalized


def _content_preview(blocks: list[dict[str, Any]], *, limit: int = 6) -> str:
    parts: list[str] = []
    for block in blocks[:limit]:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        parts.append(text)
    return "\n".join(parts)


class NotionTool:
    name = "notion"

    def __init__(
        self,
        config: NotionConfig,
        *,
        build_client: Callable[[NotionConfig], NotionClient] | None = None,
    ) -> None:
        self.config = config
        self._build_client = build_client or NotionClient.from_config

    def describe(self) -> dict[str, Any]:
        return build_tool_description(
            name=self.name,
            description="Search and read Notion pages through the Notion API.",
            actions=self._action_specs(),
        )

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "search_pages": self._search_pages,
            "get_page": self._get_page,
            "get_page_content": self._get_page_content,
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError(f"unsupported notion action: {action}")
        return handler(params, ctx)

    def controller_output(self, action: str, result: ToolResult) -> dict[str, Any]:
        data = result.data if isinstance(result.data, dict) else {}
        if action == "search_pages":
            payload: dict[str, Any] = {}
            if "query" in data:
                payload["query"] = data.get("query")
            if "result_count" in data:
                payload["result_count"] = data.get("result_count")
            results = data.get("results")
            if isinstance(results, list):
                payload["results"] = results[:10]
            return payload
        if action == "get_page":
            payload: dict[str, Any] = {}
            for field in ("page_id", "title", "url", "last_edited_time", "parent"):
                if field in data:
                    payload[field] = data.get(field)
            properties = data.get("properties")
            if isinstance(properties, dict):
                payload["properties"] = dict(list(properties.items())[:12])
            return payload
        if action == "get_page_content":
            payload = {}
            for field in ("page_id", "title", "block_count", "truncated", "content_preview"):
                if field in data:
                    payload[field] = data.get(field)
            blocks = data.get("blocks")
            if isinstance(blocks, list):
                payload["blocks"] = blocks[:12]
            return payload
        return {}

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        if action == "search_pages":
            append_list_section(
                lines,
                "Pages:",
                result.data.get("results"),
                lambda item: (
                    f"- {item['title']} ({item['page_id']})"
                    + (f" | {item['last_edited_time']}" if item.get("last_edited_time") else "")
                ),
                limit=10,
            )
        elif action == "get_page":
            properties = result.data.get("properties")
            if isinstance(properties, dict) and properties:
                lines.append("Properties:")
                for key, value in list(properties.items())[:12]:
                    lines.append(f"- {key}: {value}")
        elif action == "get_page_content":
            preview = str(result.data.get("content_preview", "")).strip()
            if preview:
                lines.append("Content preview:")
                lines.append(preview)
        return "\n".join(lines)

    def _search_pages(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(params.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, summary="search_pages requires a query.", data={})
        parent_id = str(params.get("parent_id", "")).strip()
        limit = int(params.get("limit", 10))
        try:
            client = self._build_client(self.config)
            payload = client.search_pages(query, limit=limit)
        except (NotionDisabledError, NotionConfigError, NotionUnauthorizedError, NotionNotFoundError, NotionRateLimitedError, NotionError) as exc:
            return ToolResult(ok=False, summary=str(exc), data={"query": query})

        raw_results = payload.get("results", [])
        normalized = [normalize_notion_page(item) for item in raw_results if isinstance(item, dict)]
        if parent_id:
            normalized = [item for item in normalized if item.get("parent", {}).get("id") == parent_id]
        result_count = len(normalized)
        artifact = {
            "query": query,
            "result_count": result_count,
            "results": normalized[:10],
        }
        if parent_id:
            artifact["parent_id"] = parent_id
        return ToolResult(
            ok=True,
            summary=f"Found {result_count} Notion page(s) for '{query}'.",
            data={
                "query": query,
                "parent_id": parent_id,
                "result_count": result_count,
                "results": normalized,
                "allow_tool_followup": False,
                "artifacts": {
                    "notion_search_results:latest": artifact,
                },
            },
        )

    def _get_page(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        page_id = str(params.get("page_id", "")).strip()
        if not page_id:
            return ToolResult(ok=False, summary="get_page requires a page_id.", data={})
        try:
            client = self._build_client(self.config)
            payload = client.get_page(page_id)
        except (NotionDisabledError, NotionConfigError, NotionUnauthorizedError, NotionNotFoundError, NotionRateLimitedError, NotionError) as exc:
            return ToolResult(ok=False, summary=str(exc), data={"page_id": page_id})

        normalized = normalize_notion_page(payload)
        properties = _normalize_page_properties(payload)
        artifact = {
            **normalized,
            "properties": properties,
        }
        return ToolResult(
            ok=True,
            summary=f"Loaded Notion page '{normalized['title']}' ({normalized['page_id']}).",
            data={
                **normalized,
                "properties": properties,
                "allow_tool_followup": False,
                "artifacts": {
                    "notion_page:latest": artifact,
                },
            },
        )

    def _get_page_content(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        page_id = str(params.get("page_id", "")).strip()
        if not page_id:
            return ToolResult(ok=False, summary="get_page_content requires a page_id.", data={})
        max_blocks = max(1, min(int(params.get("max_blocks", 20)), 100))
        try:
            client = self._build_client(self.config)
            page = client.get_page(page_id)
            payload = client.get_page_content(page_id, max_blocks=max_blocks)
        except (NotionDisabledError, NotionConfigError, NotionUnauthorizedError, NotionNotFoundError, NotionRateLimitedError, NotionError) as exc:
            return ToolResult(ok=False, summary=str(exc), data={"page_id": page_id})

        normalized_page = normalize_notion_page(page)
        raw_results = payload.get("results", [])
        blocks = [_normalize_block(item) for item in raw_results if isinstance(item, dict)]
        preview = _content_preview(blocks)
        truncated = bool(payload.get("has_more"))
        artifact = {
            **normalized_page,
            "block_count": len(blocks),
            "truncated": truncated,
            "blocks": blocks,
            "content_preview": preview,
        }
        return ToolResult(
            ok=True,
            summary=f"Loaded content for Notion page '{normalized_page['title']}' ({len(blocks)} block(s)).",
            data={
                **normalized_page,
                "block_count": len(blocks),
                "truncated": truncated,
                "blocks": blocks,
                "content_preview": preview,
                "allow_tool_followup": False,
                "artifacts": {
                    "notion_page:latest": artifact,
                },
            },
        )

    def _action_specs(self) -> dict[str, ActionSpec]:
        return {
            "search_pages": ActionSpec(
                tool=self.name,
                action="search_pages",
                description="Search Notion pages by query and return compact page matches.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "parent_id": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                reads=True,
                produces_artifacts=("notion_search_results",),
            ),
            "get_page": ActionSpec(
                tool=self.name,
                action="get_page",
                description="Read a Notion page's normalized metadata and properties by page id.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "page_id": {"type": "string"},
                    },
                    "required": ["page_id"],
                },
                reads=True,
                produces_artifacts=("notion_page",),
            ),
            "get_page_content": ActionSpec(
                tool=self.name,
                action="get_page_content",
                description="Read a Notion page's content blocks in a compact, normalized format by page id.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "page_id": {"type": "string"},
                        "max_blocks": {"type": "integer"},
                    },
                    "required": ["page_id"],
                },
                reads=True,
                produces_artifacts=("notion_page",),
            ),
        }
