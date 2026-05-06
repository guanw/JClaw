from jclaw.core.config import NotionConfig
from jclaw.tools.base import ToolContext
from jclaw.tools.notion.client import NotionDisabledError
from jclaw.tools.notion.tool import NotionTool


class FakeNotionClient:
    def __init__(self, results=None, page=None, content=None) -> None:  # noqa: ANN001
        self.results = results or []
        self.page = page or {}
        self.content = content or {"results": [], "has_more": False}
        self.calls: list[tuple[str, int]] = []
        self.page_calls: list[str] = []
        self.content_calls: list[tuple[str, int]] = []

    def search_pages(self, query: str, *, limit: int = 10) -> dict:
        self.calls.append((query, limit))
        return {"results": self.results}

    def get_page(self, page_id: str) -> dict:
        self.page_calls.append(page_id)
        return dict(self.page)

    def get_page_content(self, page_id: str, *, max_blocks: int = 50) -> dict:
        self.content_calls.append((page_id, max_blocks))
        return dict(self.content)


def test_notion_search_pages_returns_compact_results() -> None:
    results = [
        {
            "id": "page-1",
            "url": "https://notion.so/page-1",
            "last_edited_time": "2026-05-05T00:00:00.000Z",
            "parent": {"type": "page_id", "page_id": "parent-1"},
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Project roadmap"}],
                }
            },
        }
    ]
    fake = FakeNotionClient(results)
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token"),
        build_client=lambda config: fake,
    )

    result = tool.invoke("search_pages", {"query": "roadmap"}, ToolContext(chat_id="chat-1"))

    assert result.ok is True
    assert result.data["result_count"] == 1
    assert result.data["results"][0]["title"] == "Project roadmap"
    assert result.data["artifacts"]["notion_search_results:latest"]["query"] == "roadmap"
    controller_output = tool.controller_output("search_pages", result)
    assert controller_output["query"] == "roadmap"
    assert controller_output["result_count"] == 1
    assert controller_output["results"][0]["page_id"] == "page-1"
    assert "Project roadmap" in tool.format_result("search_pages", result)


def test_notion_search_pages_supports_parent_filter() -> None:
    results = [
        {
            "id": "page-1",
            "url": "https://notion.so/page-1",
            "last_edited_time": "2026-05-05T00:00:00.000Z",
            "parent": {"type": "page_id", "page_id": "keep-me"},
            "properties": {"Name": {"type": "title", "title": [{"plain_text": "Keep"}]}},
        },
        {
            "id": "page-2",
            "url": "https://notion.so/page-2",
            "last_edited_time": "2026-05-05T00:00:00.000Z",
            "parent": {"type": "page_id", "page_id": "skip-me"},
            "properties": {"Name": {"type": "title", "title": [{"plain_text": "Skip"}]}},
        },
    ]
    fake = FakeNotionClient(results)
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token"),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "search_pages",
        {"query": "roadmap", "parent_id": "keep-me"},
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is True
    assert result.data["result_count"] == 1
    assert result.data["results"][0]["page_id"] == "page-1"


def test_notion_search_pages_reports_disabled_or_unconfigured_state() -> None:
    tool = NotionTool(
        NotionConfig(enabled=False, api_token=""),
        build_client=lambda config: (_ for _ in ()).throw(NotionDisabledError("Notion integration is disabled.")),
    )

    result = tool.invoke("search_pages", {"query": "roadmap"}, ToolContext(chat_id="chat-1"))

    assert result.ok is False
    assert "disabled" in result.summary.lower()


def test_notion_get_page_returns_normalized_metadata() -> None:
    page = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-05-05T00:00:00.000Z",
        "parent": {"type": "page_id", "page_id": "parent-1"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Project roadmap"}]},
            "Status": {"type": "status", "status": {"name": "In progress"}},
            "Estimate": {"type": "number", "number": 3},
        },
    }
    fake = FakeNotionClient(page=page)
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token"),
        build_client=lambda config: fake,
    )

    result = tool.invoke("get_page", {"page_id": "page-1"}, ToolContext(chat_id="chat-1"))

    assert result.ok is True
    assert result.data["title"] == "Project roadmap"
    assert result.data["properties"]["Status"] == "In progress"
    assert result.data["properties"]["Estimate"] == 3
    assert result.data["artifacts"]["notion_page:latest"]["page_id"] == "page-1"
    controller_output = tool.controller_output("get_page", result)
    assert controller_output["page_id"] == "page-1"
    assert controller_output["properties"]["Status"] == "In progress"


def test_notion_get_page_content_returns_normalized_blocks() -> None:
    page = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-05-05T00:00:00.000Z",
        "parent": {"type": "page_id", "page_id": "parent-1"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Project roadmap"}]},
        },
    }
    content = {
        "results": [
            {
                "id": "block-1",
                "type": "heading_2",
                "has_children": False,
                "heading_2": {"rich_text": [{"plain_text": "Plan"}]},
            },
            {
                "id": "block-2",
                "type": "paragraph",
                "has_children": False,
                "paragraph": {"rich_text": [{"plain_text": "Ship the first draft."}]},
            },
            {
                "id": "block-3",
                "type": "to_do",
                "has_children": False,
                "to_do": {"rich_text": [{"plain_text": "Write tests"}], "checked": True},
            },
            {
                "id": "block-4",
                "type": "embed",
                "has_children": False,
                "embed": {"url": "https://example.com"},
            },
        ],
        "has_more": True,
    }
    fake = FakeNotionClient(page=page, content=content)
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token"),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "get_page_content",
        {"page_id": "page-1", "max_blocks": 4},
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is True
    assert result.data["block_count"] == 4
    assert result.data["truncated"] is True
    assert result.data["blocks"][0]["type"] == "heading_2"
    assert result.data["blocks"][1]["text"] == "Ship the first draft."
    assert result.data["blocks"][2]["checked"] is True
    assert result.data["blocks"][3]["unsupported"] is True
    assert "Plan" in result.data["content_preview"]
    controller_output = tool.controller_output("get_page_content", result)
    assert controller_output["page_id"] == "page-1"
    assert controller_output["block_count"] == 4
    assert controller_output["blocks"][0]["type"] == "heading_2"


def test_notion_tool_describe_exposes_read_actions() -> None:
    tool = NotionTool(NotionConfig(enabled=True, api_token="secret-token"))

    description = tool.describe()

    assert description["name"] == "notion"
    assert description["actions"]["search_pages"]["produces_artifacts"] == ["notion_search_results"]
    assert description["actions"]["get_page"]["produces_artifacts"] == ["notion_page"]
    assert description["actions"]["get_page_content"]["produces_artifacts"] == ["notion_page"]
