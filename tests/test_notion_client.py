import httpx
import pytest

from jclaw.core.config import NotionConfig
from jclaw.tools.notion.client import (
    NotionClient,
    NotionConfigError,
    NotionDisabledError,
    NotionNotFoundError,
    NotionRateLimitedError,
    NotionUnauthorizedError,
    notion_headers,
)


def test_notion_client_from_config_rejects_disabled_integration() -> None:
    config = NotionConfig(enabled=False, api_token="secret-token")

    with pytest.raises(NotionDisabledError):
        NotionClient.from_config(config)


def test_notion_client_from_config_requires_token() -> None:
    config = NotionConfig(enabled=True, api_token="")

    with pytest.raises(NotionConfigError):
        NotionClient.from_config(config)


def test_notion_headers_include_auth_and_version() -> None:
    headers = notion_headers("secret-token", notion_version="2022-06-28")

    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Notion-Version"] == "2022-06-28"
    assert headers["Content-Type"] == "application/json"


def test_notion_client_maps_common_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/401"):
            return httpx.Response(401, json={"message": "bad auth"})
        if request.url.path.endswith("/404"):
            return httpx.Response(404, json={"message": "missing"})
        return httpx.Response(429, json={"message": "slow down"})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    with pytest.raises(NotionUnauthorizedError):
        client.get("/401")
    with pytest.raises(NotionNotFoundError):
        client.get("/404")
    with pytest.raises(NotionRateLimitedError):
        client.get("/429")


def test_notion_client_uses_expected_headers_on_success() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        seen["version"] = request.headers.get("Notion-Version", "")
        return httpx.Response(200, json={"object": "list", "results": []})

    transport = httpx.MockTransport(handler)
    client = NotionClient(
        "secret-token",
        notion_version="2025-01-01",
        http_client=httpx.Client(transport=transport),
    )

    payload = client.get("/search")

    assert payload["object"] == "list"
    assert seen["auth"] == "Bearer secret-token"
    assert seen["version"] == "2025-01-01"


def test_notion_client_search_pages_posts_expected_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"results": []})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    client.search_pages("roadmap", limit=7)

    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/search"
    assert '"query":"roadmap"' in str(seen["body"]).replace(" ", "")
    assert '"page_size":7' in str(seen["body"]).replace(" ", "")


def test_notion_client_get_page_metadata_uses_expected_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"id": "page-1"})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    payload = client.get_page_metadata("page-1")

    assert payload["id"] == "page-1"
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/pages/page-1"


def test_notion_client_get_page_content_uses_block_children_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query"] = str(request.url.query)
        return httpx.Response(200, json={"results": [], "has_more": False, "next_cursor": None})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    payload = client.get_page("page-1", max_blocks=12)

    assert payload["results"] == []
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/blocks/page-1/children"
    assert "page_size=12" in str(seen["query"])


def test_notion_client_get_page_populates_table_children() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/v1/blocks/page-1/children":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "table-1",
                            "type": "table",
                            "has_children": True,
                            "table": {
                                "table_width": 2,
                                "has_column_header": True,
                                "has_row_header": False,
                            },
                        }
                    ],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        if request.url.path == "/v1/blocks/table-1/children":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "row-1",
                            "type": "table_row",
                            "table_row": {"cells": [[{"plain_text": "Company"}], [{"plain_text": "Stage"}]]},
                        }
                    ],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    payload = client.get_page("page-1", max_blocks=12)

    assert seen_paths == ["/v1/blocks/page-1/children", "/v1/blocks/table-1/children"]
    assert payload["results"][0]["children"][0]["id"] == "row-1"


def test_notion_client_create_page_posts_expected_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"id": "page-created"})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    payload = client.create_page(
        parent={"page_id": "parent-1"},
        properties={"title": {"title": [{"type": "text", "text": {"content": "Sprint notes"}}]}},
        children=[{"object": "block", "type": "paragraph", "paragraph": {"rich_text": []}}],
    )

    assert payload["id"] == "page-created"
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/pages"
    body = str(seen["body"]).replace(" ", "")
    assert '"page_id":"parent-1"' in body
    assert '"content":"Sprintnotes"' in body
    assert '"children":' in body


def test_notion_client_update_page_metadata_patches_expected_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"id": "page-1"})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    payload = client.update_page_metadata(
        "page-1",
        properties={
            "Name": {"title": [{"type": "text", "text": {"content": "Renamed page"}}]},
            "Points": {"number": 5},
        },
    )

    assert payload["id"] == "page-1"
    assert seen["method"] == "PATCH"
    assert seen["path"] == "/v1/pages/page-1"
    body = str(seen["body"]).replace(" ", "")
    assert '"content":"Renamedpage"' in body
    assert '"number":5' in body


def test_notion_client_update_page_markdown_patches_expected_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"object": "page_markdown", "id": "page-1", "markdown": "haha"})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    payload = client.update_page_markdown("page-1", markdown="haha")

    assert payload["id"] == "page-1"
    assert seen["method"] == "PATCH"
    assert seen["path"] == "/v1/pages/page-1/markdown"
    body = str(seen["body"]).replace(" ", "")
    assert '"type":"replace_content"' in body
    assert '"new_str":"haha"' in body


def test_notion_client_update_table_row_patches_expected_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"id": "row-1", "type": "table_row", "table_row": {"cells": []}})

    transport = httpx.MockTransport(handler)
    client = NotionClient("secret-token", http_client=httpx.Client(transport=transport))

    payload = client.update_table_row(
        "row-1",
        cells=[
            [{"type": "text", "text": {"content": "Chariot"}}],
            [{"type": "text", "text": {"content": "system design rescheduled"}}],
        ],
    )

    assert payload["id"] == "row-1"
    assert seen["method"] == "PATCH"
    assert seen["path"] == "/v1/blocks/row-1"
    body = str(seen["body"]).replace(" ", "")
    assert '"table_row":{"cells":' in body
    assert '"content":"systemdesignrescheduled"' in body
