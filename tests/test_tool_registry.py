from pathlib import Path

from jclaw.tools.base import ToolContext
from jclaw.tools.browser.tool import BrowserTool
from jclaw.tools.registry import ToolRegistry


def test_registry_invokes_browser_tool(tmp_path) -> None:
    registry = ToolRegistry()
    tool = BrowserTool(tmp_path)
    tool.playwright = tool.desktop
    registry.register(tool)

    result = registry.invoke(
        "browser",
        "run_objective",
        {"objective": "Open example.com"},
        ToolContext(chat_id="chat-1"),
    )

    assert result.ok is True
    assert result.data["session_id"].startswith("sess_")
