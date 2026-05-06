from jclaw.core.config import NotionConfig
from jclaw.tools.base import ToolContext
from jclaw.tools.notion.client import NotionDisabledError
from jclaw.tools.notion.tool import NotionTool


class FakeNotionClient:
    def __init__(self, results) -> None:  # noqa: ANN001
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search_pages(self, query: str, *, limit: int = 10) -> dict:
        self.calls.append((query, limit))
        return {"results": self.results}


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


def test_notion_tool_describe_exposes_search_action() -> None:
    tool = NotionTool(NotionConfig(enabled=True, api_token="secret-token"))

    description = tool.describe()

    assert description["name"] == "notion"
    assert description["actions"]["search_pages"]["produces_artifacts"] == ["notion_search_results"]
