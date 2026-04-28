from __future__ import annotations

from types import SimpleNamespace
import time

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
        self.sent_messages: list[tuple[str, str, str | None]] = []
        self.edited_messages: list[tuple[str, str, str]] = []

    def poll_updates(self, offset: int) -> list[IncomingMessage]:
        returned = list(self.messages)
        self.messages.clear()
        return returned

    def send_message(self, chat_id: str, text: str, reply_to_message_id: str | None = None) -> str:
        self.sent_messages.append((chat_id, text, reply_to_message_id))
        return "placeholder-1"

    def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
        self.edited_messages.append((chat_id, message_id, text))

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
        "once:1800",
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


def test_run_due_cron_jobs_disables_explicit_date_job_after_execution(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    job_id = db.add_cron_job(
        "chat-1",
        "date:2026-4-27 09:00",
        "practice interview",
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
    assert sent_messages == [("chat-1", "Reminder: practice interview")]
    assert job is not None
    assert job.enabled is False
    db.close()


def test_handle_message_edits_placeholder_with_final_reply() -> None:
    daemon = object.__new__(JClawDaemon)
    daemon.config = SimpleNamespace(telegram=SimpleNamespace(allowed_chat_ids=()))
    daemon.channel = FakeChannel([])
    daemon.agent = SimpleNamespace(handle_text=lambda chat_id, text, user_name=None: "Final answer")
    daemon.THINKING_STATUS_INTERVAL_SECONDS = 10.0

    daemon._handle_message("chat-1", "42", "Jude", "hello")

    assert len(daemon.channel.sent_messages) == 1
    sent_chat_id, sent_text, sent_reply_to = daemon.channel.sent_messages[0]
    assert sent_chat_id == "chat-1"
    assert sent_reply_to == "42"
    assert sent_text in JClawDaemon.THINKING_STATUSES
    assert daemon.channel.edited_messages == [("chat-1", "placeholder-1", "Final answer")]


def test_handle_message_updates_placeholder_while_waiting() -> None:
    daemon = object.__new__(JClawDaemon)
    daemon.config = SimpleNamespace(telegram=SimpleNamespace(allowed_chat_ids=()))
    daemon.channel = FakeChannel([])
    daemon.THINKING_STATUS_INTERVAL_SECONDS = 0.01

    def slow_reply(chat_id: str, text: str, user_name: str | None = None) -> str:
        time.sleep(0.035)
        return "Final answer"

    daemon.agent = SimpleNamespace(handle_text=slow_reply)

    daemon._handle_message("chat-1", "42", "Jude", "hello")

    assert len(daemon.channel.sent_messages) == 1
    assert len(daemon.channel.edited_messages) >= 2
    assert daemon.channel.edited_messages[-1] == ("chat-1", "placeholder-1", "Final answer")
    interim_updates = [entry[2] for entry in daemon.channel.edited_messages[:-1]]
    assert interim_updates
    assert all(text in JClawDaemon.THINKING_STATUSES for text in interim_updates)
