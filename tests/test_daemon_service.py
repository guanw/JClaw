from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from jclaw.ai.tool_loop import RunInterruptedError
from jclaw.channel.base import IncomingMessage
from jclaw.channel.telegram import TelegramPollingConflictError
from jclaw.core.db import Database
from jclaw.daemon.service import JClawDaemon


class FakeDB:
    def __init__(self) -> None:
        self.offset = 0
        self.trace_mode = "off"
        self.pruned_before: list[str] = []

    def get_telegram_offset(self) -> int:
        return self.offset

    def set_telegram_offset(self, offset: int) -> None:
        self.offset = offset

    def get_trace_mode(self, chat_id: str) -> str:
        _ = chat_id
        return self.trace_mode

    def due_cron_jobs(self, now: str) -> list[object]:
        return []

    def prune_disabled_cron_jobs(self, before: str) -> int:
        self.pruned_before.append(before)
        return 0

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
        return f"placeholder-{len(self.sent_messages)}"

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


def test_daemon_stops_after_telegram_polling_conflict() -> None:
    daemon = object.__new__(JClawDaemon)
    daemon.config = SimpleNamespace(daemon=SimpleNamespace(idle_sleep_seconds=0.0))
    daemon.db = FakeDB()
    daemon.llm = SimpleNamespace(close=lambda: None)
    daemon.channel = SimpleNamespace(
        poll_updates=lambda offset: (_ for _ in ()).throw(TelegramPollingConflictError("conflict")),
        close=lambda: None,
    )
    daemon._run_due_cron_jobs = lambda: None
    daemon._running = True

    daemon.run_forever()

    assert daemon._running is False


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


def test_prune_completed_cron_jobs_daily_removes_disabled_jobs_once_per_day(tmp_path) -> None:
    db = Database(tmp_path / "jclaw.db")
    active_job_id = db.add_cron_job(
        "chat-1",
        "interval:1800",
        "stretch",
        "2000-01-01T00:00:00+00:00",
    )
    disabled_job_id = db.add_cron_job(
        "chat-1",
        "once:1800",
        "one-off",
        "2000-01-01T00:00:00+00:00",
    )
    db.update_cron_job("chat-1", disabled_job_id, enabled=False)

    daemon = object.__new__(JClawDaemon)
    daemon.db = db
    daemon._last_cron_cleanup_date = None

    daemon._prune_completed_cron_jobs_daily()
    daemon._prune_completed_cron_jobs_daily()

    assert db.get_cron_job("chat-1", disabled_job_id) is None
    assert db.get_cron_job("chat-1", active_job_id) is not None
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


def test_handle_message_sends_and_updates_trace_message_when_enabled() -> None:
    daemon = object.__new__(JClawDaemon)
    daemon.config = SimpleNamespace(telegram=SimpleNamespace(allowed_chat_ids=()))
    daemon.channel = FakeChannel([])
    daemon.db = FakeDB()
    daemon.db.trace_mode = "summary"
    daemon.THINKING_STATUS_INTERVAL_SECONDS = 10.0
    daemon.agent = SimpleNamespace(
        handle_text=lambda chat_id, text, user_name=None: "Final answer",
        render_running_trace=lambda chat_id: "",
        render_latest_trace=lambda chat_id: "```text\nTrace [completed]\n1. Received a new user turn.\n```",
    )

    daemon._handle_message("chat-1", "42", "Jude", "hello")

    assert len(daemon.channel.sent_messages) == 2
    assert daemon.channel.sent_messages[1] == ("chat-1", "```text\nTrace\n(waiting for first event)\n```", "42")
    assert daemon.channel.edited_messages[-1] == (
        "chat-1",
        "placeholder-2",
        "```text\nTrace [completed]\n1. Received a new user turn.\n```",
    )


def test_dispatch_message_interrupts_active_chat_and_runs_latest_pending_message() -> None:
    daemon = object.__new__(JClawDaemon)
    daemon.config = SimpleNamespace(telegram=SimpleNamespace(allowed_chat_ids=()))
    daemon.channel = FakeChannel([])
    daemon.db = FakeDB()
    daemon.THINKING_STATUS_INTERVAL_SECONDS = 0.01
    daemon._chat_workers = {}
    daemon._chat_workers_lock = threading.Lock()

    class InterruptibleAgent:
        def __init__(self) -> None:
            self.interrupted = threading.Event()
            self.started = threading.Event()
            self.calls: list[str] = []

        def request_interrupt(self, chat_id: str) -> bool:
            _ = chat_id
            self.interrupted.set()
            return True

        def handle_text(self, chat_id: str, text: str, user_name=None) -> str:
            _ = chat_id, user_name
            self.calls.append(text)
            if text == "first":
                self.started.set()
                for _ in range(50):
                    if self.interrupted.wait(0.01):
                        raise RunInterruptedError("superseded")
                return "first-reply"
            return "second-reply"

    daemon.agent = InterruptibleAgent()

    daemon._dispatch_message("chat-1", "1", "Jude", "first")
    assert daemon.agent.started.wait(timeout=1.0)
    daemon._dispatch_message("chat-1", "2", "Jude", "second")

    deadline = time.time() + 2.0
    while time.time() < deadline:
        with daemon._chat_workers_lock:
            if "chat-1" not in daemon._chat_workers:
                break
        time.sleep(0.01)

    assert daemon.agent.calls == ["first", "second"]
    assert ("chat-1", "placeholder-1", "Interrupted by a newer message.") in daemon.channel.edited_messages
    assert ("chat-1", "placeholder-2", "second-reply") in daemon.channel.edited_messages
