from __future__ import annotations

from unittest.mock import Mock

import httpx

from jclaw.channel.telegram import TelegramBotChannel, TelegramPollingConflictError
from jclaw.core.config import TelegramConfig


def test_message_payload_uses_html_parse_mode_for_preformatted_blocks() -> None:
    channel = TelegramBotChannel(TelegramConfig())
    try:
        payload = channel._message_payload(  # noqa: SLF001
            "chat-1",
            "Content:\n<pre>print(&#x27;hi&#x27;)\n</pre>",
            parse_mode="HTML",
        )
        assert payload["parse_mode"] == "HTML"
    finally:
        channel.close()


def test_message_payload_leaves_plain_text_unformatted() -> None:
    channel = TelegramBotChannel(TelegramConfig())
    try:
        payload = channel._message_payload("chat-1", "plain text")  # noqa: SLF001
        assert "parse_mode" not in payload
    finally:
        channel.close()


def test_message_chunks_convert_fenced_code_blocks_to_html() -> None:
    channel = TelegramBotChannel(TelegramConfig())
    try:
        chunks, parse_mode = channel._message_chunks("Before\n```bash\ngit status\n```\nAfter")  # noqa: SLF001
        assert parse_mode == "HTML"
        assert chunks == ["Before\n<pre>git status</pre>\nAfter"]
    finally:
        channel.close()


def test_request_raises_clear_error_for_getupdates_conflict() -> None:
    channel = TelegramBotChannel(TelegramConfig(bot_token="test-token"))
    request = httpx.Request("POST", "https://api.telegram.org/bottest/getUpdates")
    response = httpx.Response(409, request=request)
    channel._client = Mock(post=Mock(return_value=response))  # noqa: SLF001
    try:
        try:
            channel._request("getUpdates", {})  # noqa: SLF001
        except TelegramPollingConflictError as exc:
            assert "another JClaw instance" in str(exc)
        else:
            raise AssertionError("expected TelegramPollingConflictError")
    finally:
        channel.close()
