from __future__ import annotations

from types import SimpleNamespace

from jclaw.channel.base import IncomingMessage
from jclaw.core.db import Database
from jclaw.daemon.service import JClawDaemon


class FakeDB:
    def __init__(self) -> None:
        self.offset = 0

    def get_telegram_offset(self) -> int:
        return self.offset

    def set_telegram_offset(self, offset: int) -> None:
        self.offset = offset

    def due_cron_jobs(self, now: str) -> list[object]:
        return []

    def close(self) -> None:
        return None


class FakeChannel:
    def __init__(self, messages: list[IncomingMessage]) -> None:
        self.messages = list(messages)

    def poll_updates(self, offset: int) -> list[IncomingMessage]:
        returned = list(self.messages)
        self.messages.clear()
        return returned

    def close(self) -> None:
        return None


def test_daemon_advances_offset_when_message_handling_fails() -> None:
    daemon = object.__new__(JClawDaemon)
    daemon.config = SimpleNamespace(daemon=SimpleNamespace(idle_sleep_seconds=0.0))
    daemon.db = FakeDB()
    daemon.channel = FakeChannel(
        [
            IncomingMessage(
                update_id=7,
                chat_id="chat-1",
                message_id="1",
                sender_id="user-1",
                sender_name="Jude",
                text="unsupported schedule request",
            )
        ]
    )
    daemon.llm = SimpleNamespace(close=lambda: None)
    daemon._running = True

    def stop_after_first_message(chat_id: str, message_id: str, sender_name: str, text: str) -> None:
        daemon._running = False
        raise ValueError("boom")

    daemon._handle_message = stop_after_first_message
    daemon._run_due_cron_jobs = lambda: None

    daemon.run_forever()

    assert daemon.db.offset == 8


def test_run_due_cron_jobs_disables_one_off_job(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    job_id = db.add_cron_job(
        "chat-1",
        "in 30 minutes",
        "stretch",
        "2000-01-01T00:00:00+00:00",
    )
    daemon = object.__new__(JClawDaemon)
    daemon.db = db

    sent_messages: list[tuple[str, str]] = []
    daemon.channel = SimpleNamespace(
        send_message=lambda chat_id, text, reply_to_message_id=None: sent_messages.append((chat_id, text)),
        close=lambda: None,
    )
    daemon.agent = SimpleNamespace(handle_cron=lambda chat_id, prompt: f"Reminder: {prompt}")

    daemon._run_due_cron_jobs()

    job = db.get_cron_job("chat-1", job_id)
    assert sent_messages == [("chat-1", "Reminder: stretch")]
    assert job is not None
    assert job.enabled is False
    db.close()
