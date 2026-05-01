from __future__ import annotations

from jclaw.channel.telegram import TelegramBotChannel
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
