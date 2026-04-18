from jclaw.tools.base import ToolContext
from jclaw.tools.browser.models import BrowserReasoner
from jclaw.tools.browser.tool import BrowserTool


def _stub_browser(tool: BrowserTool) -> BrowserTool:
    tool.playwright = tool.desktop
    return tool


class StubBrowserReasoner:
    def __init__(self, *, chosen_link=None, next_action=None) -> None:  # noqa: ANN001
        self._chosen_link = chosen_link
        self._next_action = next_action

    def choose_link(self, objective: str, page_data: dict) -> str | None:  # noqa: ANN001
        if callable(self._chosen_link):
            return self._chosen_link(objective, page_data)
        return self._chosen_link

    def decide_next_action(self, objective: str, page_data: dict, sources: list[dict]) -> dict | None:  # noqa: ANN001
        if callable(self._next_action):
            return self._next_action(objective, page_data, sources)
        return self._next_action


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
            reasoner=StubBrowserReasoner(chosen_link="https://chosen.example.com/story"),
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


def test_run_objective_feedback_loop_follows_until_complete(tmp_path) -> None:
    tool = _stub_browser(
        BrowserTool(
            tmp_path,
            reasoner=StubBrowserReasoner(next_action=lambda objective, page_data, sources: (
                {"status": "follow", "url": "https://example.com/source-1", "reason": "Need one concrete source."}
                if not sources
                else {"status": "complete", "url": None, "reason": "Enough evidence gathered."}
            )),
        )
    )

    visited_urls: list[str] = []
    page_reads = {
        "https://html.duckduckgo.com/html/?q=test": {
            "session_id": "sess_fake",
            "tab_id": "tab_1",
            "url": "https://html.duckduckgo.com/html/?q=test",
            "title": "search",
            "page_kind": "search_results",
            "text": "search results",
            "elements": [{"id": "e1", "role": "link", "text": "Source 1", "href": "https://example.com/source-1", "area": "main", "clickable": True}],
            "links": [{"text": "Source 1", "href": "https://example.com/source-1"}],
            "forms": [],
            "mode": "desktop",
        },
        "https://example.com/source-1": {
            "session_id": "sess_fake",
            "tab_id": "tab_1",
            "url": "https://example.com/source-1",
            "title": "source 1",
            "page_kind": "article",
            "text": "source content",
            "elements": [],
            "links": [],
            "forms": [],
            "mode": "desktop",
        },
    }

    def fake_open_url(params, ctx):  # noqa: ANN001
        url = params["url"]
        visited_urls.append(url)
        return tool._open_url_original(params, ctx)

    def fake_read_page(params, ctx):  # noqa: ANN001
        return type(
            "FakeResult",
            (),
            {"ok": True, "summary": "read", "data": page_reads[visited_urls[-1]]},
        )()

    tool._open_url_original = tool._open_url  # type: ignore[attr-defined]  # noqa: SLF001
    tool._open_url = fake_open_url  # type: ignore[method-assign]  # noqa: SLF001
    tool._read_page = fake_read_page  # type: ignore[method-assign]  # noqa: SLF001

    result = tool.invoke(
        "run_objective",
        {"objective": "test", "start_url": "https://html.duckduckgo.com/html/?q=test"},
        ToolContext(chat_id="chat-loop"),
    )

    assert result.data["research_complete"] is True
    assert result.data["termination_reason"] == "controller_complete"
    assert result.data["sources"] == [
        {
            "url": "https://example.com/source-1",
            "title": "source 1",
            "text": "source content",
        }
    ]
    assert visited_urls == [
        "https://html.duckduckgo.com/html/?q=test",
        "https://example.com/source-1",
    ]


def test_run_objective_stops_when_controller_sees_no_meaningful_progress(tmp_path) -> None:
    tool = _stub_browser(
        BrowserTool(
            tmp_path,
            reasoner=StubBrowserReasoner(next_action=lambda objective, page_data, sources: {
                "status": "stop",
                "url": None,
                "reason": "No meaningful result pages were found.",
            }),
        )
    )

    page_reads = {
        "https://html.duckduckgo.com/html/?q=test": {
            "session_id": "sess_fake",
            "tab_id": "tab_1",
            "url": "https://html.duckduckgo.com/html/?q=test",
            "title": "search",
            "page_kind": "search_results",
            "text": "search results",
            "elements": [],
            "links": [],
            "forms": [],
            "mode": "desktop",
        }
    }
    visited_urls: list[str] = []

    def fake_open_url(params, ctx):  # noqa: ANN001
        url = params["url"]
        visited_urls.append(url)
        return tool._open_url_original(params, ctx)

    def fake_read_page(params, ctx):  # noqa: ANN001
        return type(
            "FakeResult",
            (),
            {"ok": True, "summary": "read", "data": page_reads[visited_urls[-1]]},
        )()

    tool._open_url_original = tool._open_url  # type: ignore[attr-defined]  # noqa: SLF001
    tool._open_url = fake_open_url  # type: ignore[method-assign]  # noqa: SLF001
    tool._read_page = fake_read_page  # type: ignore[method-assign]  # noqa: SLF001

    result = tool.invoke(
        "run_objective",
        {"objective": "test", "start_url": "https://html.duckduckgo.com/html/?q=test"},
        ToolContext(chat_id="chat-stop"),
    )

    assert result.data["research_complete"] is False
    assert result.data["termination_reason"] == "controller_stop"
    assert result.data["sources"] == []
    assert visited_urls == ["https://html.duckduckgo.com/html/?q=test"]
