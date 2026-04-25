from __future__ import annotations

import logging

from jclaw.core.logging import JClawConsoleFormatter


def _render(logger_name: str, message: str) -> str:
    formatter = JClawConsoleFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    record = logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    return formatter.format(record)


def test_console_formatter_colors_agent_raw_response_logs() -> None:
    rendered = _render("jclaw.ai.agent", "tool continuation raw response: {\"type\":\"answer\"}")

    assert "\033[" in rendered
    assert "tool continuation raw response:" in rendered


def test_console_formatter_leaves_non_agent_logs_plain() -> None:
    rendered = _render("jclaw.daemon.service", "starting JClaw daemon")

    assert "\033[" not in rendered
    assert "starting JClaw daemon" in rendered
