from __future__ import annotations

from typing import Any

import httpx

from jclaw.core.config import NotionConfig
from jclaw.core.defaults import NOTION_API_BASE_URL, NOTION_API_VERSION


class NotionError(RuntimeError):
    pass


class NotionDisabledError(NotionError):
    pass


class NotionConfigError(NotionError):
    pass


class NotionUnauthorizedError(NotionError):
    pass


class NotionNotFoundError(NotionError):
    pass


class NotionRateLimitedError(NotionError):
    pass


def notion_headers(api_token: str, *, notion_version: str = NOTION_API_VERSION) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Notion-Version": notion_version,
    }


class NotionClient:
    def __init__(
        self,
        api_token: str,
        *,
        base_url: str = NOTION_API_BASE_URL,
        notion_version: str = NOTION_API_VERSION,
        http_client: httpx.Client | None = None,
    ) -> None:
        token = str(api_token).strip()
        if not token:
            raise NotionConfigError("Notion API token is missing.")
        self.api_token = token
        self.base_url = str(base_url).rstrip("/")
        self.notion_version = str(notion_version).strip() or NOTION_API_VERSION
        self._http = http_client or httpx.Client(timeout=30.0)

    @classmethod
    def from_config(
        cls,
        config: NotionConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> "NotionClient":
        if not config.enabled:
            raise NotionDisabledError("Notion integration is disabled.")
        if not str(config.api_token).strip():
            raise NotionConfigError("Notion API token is missing.")
        return cls(
            config.api_token,
            base_url=config.base_url,
            notion_version=config.notion_version,
            http_client=http_client,
        )

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", path, params=params)

    def post(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", path, json=payload)

    def patch(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("PATCH", path, json=payload)

    def search_pages(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": str(query).strip(),
            "filter": {"property": "object", "value": "page"},
            "page_size": max(1, min(int(limit), 100)),
        }
        return self.post("/search", payload=payload)

    def create_page(
        self,
        *,
        parent: dict[str, Any],
        properties: dict[str, Any],
        children: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "parent": dict(parent),
            "properties": dict(properties),
        }
        if isinstance(children, list) and children:
            payload["children"] = list(children)
        return self.post("/pages", payload=payload)

    def update_page_metadata(
        self,
        page_id: str,
        *,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        return self.patch(f"/pages/{str(page_id).strip()}", payload={"properties": dict(properties)})

    def update_page_markdown(
        self,
        page_id: str,
        *,
        markdown: str,
        allow_deleting_content: bool = False,
    ) -> dict[str, Any]:
        return self.patch(
            f"/pages/{str(page_id).strip()}/markdown",
            payload={
                "type": "replace_content",
                "replace_content": {
                    "new_str": str(markdown),
                    "allow_deleting_content": bool(allow_deleting_content),
                },
            },
        )

    def get_page_metadata(self, page_id: str) -> dict[str, Any]:
        return self.get(f"/pages/{str(page_id).strip()}")

    def update_page(
        self,
        page_id: str,
        *,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        return self.update_page_metadata(page_id, properties=properties)

    def get_block(self, block_id: str) -> dict[str, Any]:
        return self.get(f"/blocks/{str(block_id).strip()}")

    def update_block(
        self,
        block_id: str,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self.patch(f"/blocks/{str(block_id).strip()}", payload=payload)

    def update_table_row(
        self,
        row_id: str,
        *,
        cells: list[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        return self.update_block(
            row_id,
            payload={
                "table_row": {
                    "cells": list(cells),
                }
            },
        )

    def get_block_children(
        self,
        block_id: str,
        *,
        page_size: int = 100,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page_size": max(1, min(int(page_size), 100)),
        }
        cursor = str(start_cursor or "").strip()
        if cursor:
            params["start_cursor"] = cursor
        return self.get(f"/blocks/{str(block_id).strip()}/children", params=params)

    def get_page(
        self,
        page_id: str,
        *,
        max_blocks: int = 50,
    ) -> dict[str, Any]:
        remaining = max(1, int(max_blocks))
        all_results: list[dict[str, Any]] = []
        next_cursor = ""
        has_more = False
        while remaining > 0:
            payload = self.get_block_children(
                page_id,
                page_size=min(remaining, 100),
                start_cursor=next_cursor or None,
            )
            raw_results = payload.get("results", [])
            page_results = [item for item in raw_results if isinstance(item, dict)]
            all_results.extend(page_results)
            has_more = bool(payload.get("has_more"))
            next_cursor = str(payload.get("next_cursor", "") or "")
            remaining = max_blocks - len(all_results)
            if remaining <= 0 or not has_more or not next_cursor:
                break
        self._populate_supported_children(all_results)
        return {
            "results": all_results[:max_blocks],
            "has_more": has_more and len(all_results) >= max_blocks,
            "next_cursor": next_cursor,
        }

    def _populate_supported_children(self, blocks: list[dict[str, Any]]) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if not bool(block.get("has_children")):
                continue
            if str(block.get("type", "")).strip() != "table":
                continue
            children_payload = self.get_block_children(str(block.get("id", "")).strip(), page_size=100)
            children = [item for item in children_payload.get("results", []) if isinstance(item, dict)]
            block["children"] = children

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._http.request(
            method.upper(),
            self._build_url(path),
            params=params,
            json=json,
            headers=notion_headers(self.api_token, notion_version=self.notion_version),
        )
        return self._handle_response(response)

    def _build_url(self, path: str) -> str:
        cleaned = str(path).strip()
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned
        return f"{self.base_url}/{cleaned.lstrip('/')}"

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code in {401, 403}:
            raise NotionUnauthorizedError("Notion integration is unauthorized.")
        if response.status_code == 404:
            raise NotionNotFoundError("Requested Notion resource was not found.")
        if response.status_code == 429:
            raise NotionRateLimitedError("Notion API rate limit exceeded.")
        if response.status_code >= 400:
            message = ""
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if isinstance(payload, dict):
                message = str(payload.get("message", "")).strip()
            suffix = f": {message}" if message else ""
            raise NotionError(f"Notion API request failed with {response.status_code}{suffix}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise NotionError("Notion API returned a non-object response.")
        return payload
