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
            "elements": [
                {"id": "e1", "role": "link", "text": "Search options", "href": "https://duckduckgo.com/settings", "area": "nav", "clickable": True},
                {"id": "e2", "role": "link", "text": "DeepSeek launches new model", "href": "https://example.com/deepseek-news", "area": "main", "clickable": True},
            ],
        },
    )
    assert url == "https://example.com/deepseek-news"


def test_pick_follow_up_url_rejects_app_store_and_engine_links(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    url = tool._pick_follow_up_url(  # noqa: SLF001
        "latest deepseek news",
        {
            "url": "https://duckduckgo.com/?q=latest+deepseek+news",
            "elements": [
                {"id": "e1", "role": "link", "text": "DuckDuckGo Browser with Duck AI", "href": "https://apps.apple.com/us/app/duckduckgo-duck-ai-vpn/id663592361", "area": "main", "clickable": True},
                {"id": "e2", "role": "link", "text": "DuckDuckGo settings", "href": "https://duckduckgo.com/settings", "area": "nav", "clickable": True},
                {"id": "e3", "role": "link", "text": "DeepSeek launches model update", "href": "https://news.example.com/deepseek-update", "area": "main", "clickable": True},
            ],
        },
    )
    assert url == "https://news.example.com/deepseek-update"


def test_choose_follow_up_url_prefers_llm_choice(tmp_path) -> None:
    tool = _stub_browser(
        BrowserTool(
            tmp_path,
            choose_link=lambda objective, page_data: "https://chosen.example.com/story",
        )
    )
    url = tool._choose_follow_up_url(  # noqa: SLF001
        "latest deepseek news",
        {
            "url": "https://html.duckduckgo.com/html/?q=latest+deepseek+news",
            "links": [{"text": "fallback", "href": "https://fallback.example.com"}],
        },
    )
    assert url == "https://chosen.example.com/story"


def test_extract_candidate_elements_falls_back_to_links(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    candidates = tool._extract_candidate_elements(  # noqa: SLF001
        {
            "links": [
                {"text": "DeepSeek news", "href": "https://example.com/news"},
                {"text": "Settings", "href": "https://duckduckgo.com/settings"},
            ]
        }
    )
    assert candidates[0]["href"] == "https://example.com/news"


def test_normalize_duckduckgo_redirect_url(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    href = tool._normalize_url(  # noqa: SLF001
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.reuters.com%2Fmarkets%2Fus%2F&rut=abc"
    )
    assert href == "https://www.reuters.com/markets/us/"


def test_run_objective_auto_closes_ephemeral_session(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    result = tool.invoke(
        "run_objective",
        {"objective": "Open example.com", "start_url": "https://example.com"},
        ToolContext(chat_id="chat-ephemeral"),
    )
    assert result.ok is True
    assert tool.sessions.list_sessions() == []


def test_run_objective_keeps_explicit_session(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    session = tool.sessions.create_session()
    tool.invoke(
        "run_objective",
        {"objective": "Open example.com", "start_url": "https://example.com", "session_id": session.session_id},
        ToolContext(chat_id="chat-explicit"),
    )
    assert len(tool.sessions.list_sessions()) == 1


def test_run_objective_keeps_session_when_requested(tmp_path) -> None:
    tool = _stub_browser(BrowserTool(tmp_path))
    tool.invoke(
        "run_objective",
        {"objective": "Open example.com", "start_url": "https://example.com", "keep_session": True},
        ToolContext(chat_id="chat-keep"),
    )
    assert len(tool.sessions.list_sessions()) == 1
