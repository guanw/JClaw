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
