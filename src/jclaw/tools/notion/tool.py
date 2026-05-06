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


def _rich_text(text: Any) -> list[dict[str, Any]]:
    value = str(text or "").strip()
    if not value:
        return []
    return [{"type": "text", "text": {"content": value}}]


def _normalize_create_property_value(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Property '{name}' must use an explicit Notion property payload shape, for example "
            "{'number': 3} or {'select': {'name': 'Draft'}}."
        )
    allowed_keys = {
        "rich_text",
        "number",
        "checkbox",
        "select",
        "status",
        "multi_select",
        "date",
        "url",
        "email",
        "phone_number",
    }
    present = [key for key in allowed_keys if key in value]
    if len(present) != 1:
        raise ValueError(
            f"Property '{name}' must contain exactly one supported Notion property type: "
            + ", ".join(sorted(allowed_keys))
        )
    kind = present[0]
    raw = value.get(kind)
    if kind == "rich_text":
        if isinstance(raw, list):
            return {"rich_text": raw}
        return {"rich_text": _rich_text(raw)}
    if kind == "number":
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            raise ValueError(f"Property '{name}' number value must be numeric.")
        return {"number": raw}
    if kind == "checkbox":
        return {"checkbox": bool(raw)}
    if kind in {"url", "email", "phone_number"}:
        return {kind: str(raw or "").strip()}
    if kind in {"select", "status"}:
        if isinstance(raw, str):
            return {kind: {"name": raw.strip()}}
        if isinstance(raw, dict):
            option_name = str(raw.get("name", "")).strip()
            if not option_name:
                raise ValueError(f"Property '{name}' {kind} value must include a non-empty name.")
            return {kind: {"name": option_name}}
        raise ValueError(f"Property '{name}' {kind} value must be a string or {{'name': ...}}.")
    if kind == "multi_select":
        if not isinstance(raw, list):
            raise ValueError(f"Property '{name}' multi_select value must be a list.")
        items: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, str):
                option_name = item.strip()
            elif isinstance(item, dict):
                option_name = str(item.get("name", "")).strip()
            else:
                option_name = ""
            if option_name:
                items.append({"name": option_name})
        return {"multi_select": items}
    if kind == "date":
        if not isinstance(raw, dict):
            raise ValueError(f"Property '{name}' date value must be an object with start/end.")
        start = str(raw.get("start", "")).strip()
        end = str(raw.get("end", "")).strip() or None
        if not start:
            raise ValueError(f"Property '{name}' date value must include a non-empty start.")
        return {"date": {"start": start, "end": end}}
    raise ValueError(f"Property '{name}' uses an unsupported Notion property type.")


def _build_create_properties(
    title: str,
    extra_properties: Any,
    *,
    parent_type: str,
    title_property: str,
) -> dict[str, Any]:
    title_key = "title" if parent_type == "page_id" else title_property
    properties: dict[str, Any] = {
        title_key: {
            "title": _rich_text(title),
        }
    }
    if not isinstance(extra_properties, dict):
        return properties
    for key, value in extra_properties.items():
        clean_key = str(key).strip()
        if not clean_key or clean_key == title_key:
            continue
        properties[clean_key] = _normalize_create_property_value(clean_key, value)
    return properties


def _build_content_blocks(content: Any) -> list[dict[str, Any]]:
    if content in (None, ""):
        return []
    lines: list[str]
    if isinstance(content, str):
        lines = content.splitlines()
    elif isinstance(content, list):
        lines = [str(item) for item in content]
    else:
        lines = [str(content)]

    blocks: list[dict[str, Any]] = []
    for raw_line in lines:
        line = str(raw_line).rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": _rich_text(stripped[2:])},
                }
            )
            continue
        if stripped.startswith("[ ] "):
            blocks.append(
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {"rich_text": _rich_text(stripped[4:]), "checked": False},
                }
            )
            continue
        if stripped.startswith("[x] ") or stripped.startswith("[X] "):
            blocks.append(
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {"rich_text": _rich_text(stripped[4:]), "checked": True},
                }
            )
            continue
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _rich_text(stripped)},
            }
        )
    return blocks


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
            description="Search, read, and create simple Notion pages through the Notion API.",
            actions=self._action_specs(),
        )

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "search_pages": self._search_pages,
            "get_page": self._get_page,
            "get_page_content": self._get_page_content,
            "create_page": self._create_page,
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
        if action == "create_page":
            payload = {}
            for field in ("page_id", "title", "url", "last_edited_time", "parent", "block_count"):
                if field in data:
                    payload[field] = data.get(field)
            properties = data.get("properties")
            if isinstance(properties, dict):
                payload["properties"] = dict(list(properties.items())[:12])
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
        elif action == "create_page":
            properties = result.data.get("properties")
            if isinstance(properties, dict) and properties:
                lines.append("Properties:")
                for key, value in list(properties.items())[:12]:
                    lines.append(f"- {key}: {value}")
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

    def _create_page(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        parent_id = str(params.get("parent_id") or self.config.default_parent_id).strip()
        if not parent_id:
            return ToolResult(ok=False, summary="create_page requires a parent_id or configured default_parent_id.", data={})
        title = str(params.get("title", "")).strip()
        if not title:
            return ToolResult(ok=False, summary="create_page requires a title.", data={"parent_id": parent_id})

        writable = tuple(str(item).strip() for item in self.config.writable_parent_ids if str(item).strip())
        if writable and parent_id not in writable:
            return ToolResult(
                ok=False,
                summary=f"Notion writes are not allowed under parent '{parent_id}'.",
                data={"parent_id": parent_id, "title": title},
            )

        parent_type = str(params.get("parent_type", "page_id")).strip() or "page_id"
        if parent_type not in {"page_id", "database_id"}:
            return ToolResult(
                ok=False,
                summary="create_page parent_type must be 'page_id' or 'database_id'.",
                data={"parent_id": parent_id, "title": title, "parent_type": parent_type},
            )
        title_property = str(params.get("title_property", "Name")).strip() or "Name"
        try:
            properties = _build_create_properties(
                title,
                params.get("properties"),
                parent_type=parent_type,
                title_property=title_property,
            )
        except ValueError as exc:
            return ToolResult(
                ok=False,
                summary=str(exc),
                data={"parent_id": parent_id, "title": title, "parent_type": parent_type},
            )
        children = _build_content_blocks(params.get("content"))
        try:
            client = self._build_client(self.config)
            payload = client.create_page(
                parent={parent_type: parent_id},
                properties=properties,
                children=children,
            )
        except (NotionDisabledError, NotionConfigError, NotionUnauthorizedError, NotionNotFoundError, NotionRateLimitedError, NotionError) as exc:
            return ToolResult(ok=False, summary=str(exc), data={"parent_id": parent_id, "title": title})

        normalized = normalize_notion_page(payload)
        normalized_properties = _normalize_page_properties(payload)
        artifact = {
            **normalized,
            "properties": normalized_properties,
            "block_count": len(children),
        }
        return ToolResult(
            ok=True,
            summary=f"Created Notion page '{normalized['title']}' ({normalized['page_id']}).",
            data={
                **normalized,
                "properties": normalized_properties,
                "block_count": len(children),
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
            "create_page": ActionSpec(
                tool=self.name,
                action="create_page",
                description="Create a simple Notion page under a parent page or database with constrained content and properties.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "parent_id": {"type": "string"},
                        "parent_type": {"type": "string"},
                        "title": {"type": "string"},
                        "title_property": {"type": "string"},
                        "content": {},
                        "properties": {"type": "object"},
                    },
                    "required": ["title"],
                },
                writes=True,
                produces_artifacts=("notion_page",),
            ),
        }
