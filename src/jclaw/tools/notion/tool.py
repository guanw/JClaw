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
            )
        }
