from jclaw.core.config import NotionConfig
from jclaw.tools.base import ToolContext
from jclaw.tools.notion.client import NotionDisabledError
from jclaw.tools.notion.tool import NotionTool


class FakeNotionClient:
    def __init__(self, results=None, page=None, content=None, created_page=None, markdown_update=None) -> None:  # noqa: ANN001
        self.results = results or []
        self.page = page or {}
        self.content = content or {"results": [], "has_more": False}
        self.created_page = created_page or {}
        self.markdown_update = markdown_update or {"object": "page_markdown", "id": "", "markdown": ""}
        self.calls: list[tuple[str, int]] = []
        self.page_calls: list[str] = []
        self.content_calls: list[tuple[str, int]] = []
        self.create_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.markdown_calls: list[dict] = []

    def search_pages(self, query: str, *, limit: int = 10) -> dict:
        self.calls.append((query, limit))
        return {"results": self.results}

    def get_page_metadata(self, page_id: str) -> dict:
        self.page_calls.append(page_id)
        return dict(self.page)

    def get_page(self, page_id: str, *, max_blocks: int = 50) -> dict:
        self.content_calls.append((page_id, max_blocks))
        return dict(self.content)

    def create_page(self, *, parent: dict, properties: dict, children: list[dict] | None = None) -> dict:
        self.create_calls.append(
            {
                "parent": dict(parent),
                "properties": dict(properties),
                "children": list(children or []),
            }
        )
        return dict(self.created_page)

    def update_page(self, page_id: str, *, properties: dict) -> dict:
        self.update_calls.append({"page_id": page_id, "properties": dict(properties)})
        return dict(self.created_page or self.page)

    def update_page_metadata(self, page_id: str, *, properties: dict) -> dict:
        return self.update_page(page_id, properties=properties)

    def update_page_markdown(self, page_id: str, *, markdown: str, allow_deleting_content: bool = False) -> dict:
        self.markdown_calls.append(
            {
                "page_id": page_id,
                "markdown": markdown,
                "allow_deleting_content": allow_deleting_content,
            }
        )
        return dict(self.markdown_update)


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


def test_notion_get_page_metadata_returns_normalized_metadata() -> None:
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

    result = tool.invoke("get_page_metadata", {"page_id": "page-1"}, ToolContext(chat_id="chat-1"))

    assert result.ok is True
    assert result.data["title"] == "Project roadmap"
    assert result.data["properties"]["Status"] == "In progress"
    assert result.data["properties"]["Estimate"] == 3
    assert result.data["artifacts"]["notion_page:latest"]["page_id"] == "page-1"
    controller_output = tool.controller_output("get_page_metadata", result)
    assert controller_output["page_id"] == "page-1"
    assert controller_output["properties"]["Status"] == "In progress"


def test_notion_get_page_returns_normalized_blocks() -> None:
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
        "get_page",
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
    controller_output = tool.controller_output("get_page", result)
    assert controller_output["page_id"] == "page-1"
    assert controller_output["block_count"] == 4
    assert controller_output["blocks"][0]["type"] == "heading_2"


def test_notion_create_page_shapes_parent_properties_and_children() -> None:
    created_page = {
        "id": "page-created",
        "url": "https://notion.so/page-created",
        "last_edited_time": "2026-05-06T00:00:00.000Z",
        "parent": {"type": "page_id", "page_id": "parent-1"},
        "properties": {
            "title": {"type": "title", "title": [{"plain_text": "Sprint notes"}]},
            "Status": {"type": "select", "select": {"name": "Draft"}},
            "Points": {"type": "number", "number": 2},
        },
    }
    fake = FakeNotionClient(created_page=created_page)
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token"),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "create_page",
        {
            "parent_id": "parent-1",
            "title": "Sprint notes",
            "content": ["Intro line", "- Ship search", "[x] Add tests"],
            "properties": {"Status": {"select": {"name": "Draft"}}, "Points": {"number": 2}},
        },
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is True
    assert result.data["page_id"] == "page-created"
    assert result.data["properties"]["Status"] == "Draft"
    assert result.data["block_count"] == 3
    assert fake.create_calls[0]["parent"] == {"page_id": "parent-1"}
    assert fake.create_calls[0]["properties"]["title"]["title"][0]["text"]["content"] == "Sprint notes"
    assert fake.create_calls[0]["properties"]["Status"]["select"]["name"] == "Draft"
    assert fake.create_calls[0]["children"][1]["type"] == "bulleted_list_item"
    assert fake.create_calls[0]["children"][2]["type"] == "to_do"
    controller_output = tool.controller_output("create_page", result)
    assert controller_output["page_id"] == "page-created"
    assert controller_output["block_count"] == 3


def test_notion_create_page_rejects_implicit_property_shapes() -> None:
    fake = FakeNotionClient(created_page={})
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token"),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "create_page",
        {
            "parent_id": "parent-1",
            "title": "Sprint notes",
            "properties": {"Points": 2},
        },
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is False
    assert "explicit notion property payload shape" in result.summary.lower()
    assert fake.create_calls == []


def test_notion_create_page_enforces_writable_parent_ids() -> None:
    fake = FakeNotionClient(created_page={})
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token", writable_parent_ids=("allowed-parent",)),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "create_page",
        {"parent_id": "blocked-parent", "title": "Sprint notes"},
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is False
    assert "not allowed" in result.summary.lower()
    assert fake.create_calls == []


def test_notion_update_page_updates_title_and_simple_properties() -> None:
    current_page = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-05-06T00:00:00.000Z",
        "parent": {"type": "page_id", "page_id": "parent-1"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Old title"}]},
            "Status": {"type": "status", "status": {"name": "Draft"}},
        },
    }
    updated_page = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-05-06T01:00:00.000Z",
        "parent": {"type": "page_id", "page_id": "parent-1"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "New title"}]},
            "Status": {"type": "status", "status": {"name": "Published"}},
            "Points": {"type": "number", "number": 5},
        },
    }
    fake = FakeNotionClient(page=current_page, created_page=updated_page)
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token", writable_parent_ids=("parent-1",)),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "update_page",
        {
            "page_id": "page-1",
            "properties": {
                "title": {"title": "New title"},
                "Status": {"status": {"name": "Published"}},
                "Points": {"number": 5},
            },
        },
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is True
    assert result.data["page_id"] == "page-1"
    assert result.data["title"] == "New title"
    assert result.data["updated_properties"]["Status"] == "Published"
    assert result.data["updated_properties"]["Points"] == 5
    assert fake.page_calls == ["page-1"]
    assert fake.update_calls[0]["page_id"] == "page-1"
    assert "Name" in fake.update_calls[0]["properties"]
    assert fake.update_calls[0]["properties"]["Points"]["number"] == 5
    controller_output = tool.controller_output("update_page", result)
    assert controller_output["updated_properties"]["Status"] == "Published"


def test_notion_update_page_can_replace_content() -> None:
    current_page = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-05-06T00:00:00.000Z",
        "parent": {"type": "page_id", "page_id": "parent-1"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Jude test"}]},
        },
    }
    updated_page = dict(current_page)
    fake = FakeNotionClient(
        page=current_page,
        created_page=updated_page,
        markdown_update={"object": "page_markdown", "id": "page-1", "markdown": "haha"},
    )

    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token"),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "update_page",
        {"page_id": "page-1", "content": "haha"},
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is True
    assert result.data["content_updated"] is True
    assert result.data["content_preview"] == "haha"
    assert fake.markdown_calls[0]["page_id"] == "page-1"
    assert fake.markdown_calls[0]["markdown"] == "haha"


def test_notion_update_page_enforces_writable_parent_ids() -> None:
    current_page = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-05-06T00:00:00.000Z",
        "parent": {"type": "page_id", "page_id": "blocked-parent"},
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Old title"}]},
        },
    }
    fake = FakeNotionClient(page=current_page)
    tool = NotionTool(
        NotionConfig(enabled=True, api_token="secret-token", writable_parent_ids=("allowed-parent",)),
        build_client=lambda config: fake,
    )

    result = tool.invoke(
        "update_page",
        {"page_id": "page-1", "properties": {"title": {"title": "New title"}}},
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is False
    assert "not allowed" in result.summary.lower()
    assert fake.update_calls == []


def test_notion_tool_describe_exposes_read_and_write_actions() -> None:
    tool = NotionTool(NotionConfig(enabled=True, api_token="secret-token"))

    description = tool.describe()

    assert description["name"] == "notion"
    assert description["actions"]["search_pages"]["produces_artifacts"] == ["notion_search_results"]
    assert description["actions"]["get_page_metadata"]["produces_artifacts"] == ["notion_page"]
    assert description["actions"]["get_page"]["produces_artifacts"] == ["notion_page"]
    assert description["actions"]["create_page"]["produces_artifacts"] == ["notion_page"]
    assert description["actions"]["update_page"]["produces_artifacts"] == ["notion_page"]
