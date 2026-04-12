from jclaw.tools.base import ToolContext
from jclaw.tools.browser.tool import BrowserTool


def _stub_browser(tool: BrowserTool) -> BrowserTool:
    tool.playwright = tool.desktop
    return tool


def test_browser_tool_lists_actions(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    description = tool.describe()
    assert description["name"] == "browser"
    assert "run_objective" in description["actions"]


def test_browser_tool_creates_session_for_objective(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    result = tool.invoke(
        "run_objective",
        {"objective": "Open example.com", "start_url": "https://example.com"},
        ToolContext(chat_id="chat-1"),
    )
    assert result.ok is True
    assert result.data["session_id"].startswith("sess_")
    assert result.data["steps"][0]["action"] == "open_url"


def test_browser_tool_reuses_session_per_chat(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    first = tool.invoke("open_url", {"url": "https://example.com"}, ToolContext(chat_id="chat-1"))
    second = tool.invoke("read_page", {}, ToolContext(chat_id="chat-1"))
    assert first.data["session_id"] == second.data["session_id"]


def test_pick_follow_up_url_prefers_real_search_result(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    url = tool._pick_follow_up_url(  # noqa: SLF001
        "latest deepseek news",
        {
            "url": "https://duckduckgo.com/?q=latest+deepseek+news",
            "links": [
                {"text": "Search options", "href": "https://duckduckgo.com/settings"},
                {"text": "DeepSeek launches new model", "href": "https://example.com/deepseek-news"},
            ],
        },
    )
    assert url == "https://example.com/deepseek-news"
