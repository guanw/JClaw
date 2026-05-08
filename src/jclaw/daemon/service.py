from __future__ import annotations

from datetime import datetime, timezone
import logging
from dataclasses import dataclass
import threading
import time

from jclaw.ai.agent import AssistantAgent
from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.tool_loop import RunInterruptedError
from jclaw.channel.telegram import TelegramBotChannel, TelegramPollingConflictError
from jclaw.core.config import Config
from jclaw.core.db import Database
from jclaw.core.scheduler import next_run_at, parse_schedule, to_utc_iso


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QueuedChatMessage:
    chat_id: str
    message_id: str
    sender_name: str
    text: str


@dataclass(slots=True)
class ChatWorkerState:
    thread: threading.Thread
    pending_message: QueuedChatMessage | None = None


class JClawDaemon:
    THINKING_STATUSES = (
        "JClaw is thinking...",
        "JClaw is rummaging through the toolbox...",
        "JClaw is chasing down the answer...",
        "JClaw is lining up the next move...",
        "JClaw is pulling threads together...",
    )
    THINKING_STATUS_INTERVAL_SECONDS = 1.2

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.daemon.db_path)
        self.cron = self.db.cron
        self.llm = OpenAICompatibleClient(config.provider)
        self.channel = TelegramBotChannel(config.telegram)
        self.agent = AssistantAgent(config, self.db, self.llm)
        self._running = True
        self._chat_workers: dict[str, ChatWorkerState] = {}
        self._chat_workers_lock = threading.Lock()

    def close(self) -> None:
        self.channel.close()
        self.llm.close()
        self.db.close()

    def run_forever(self) -> None:
        LOGGER.info("starting JClaw daemon")
        try:
            while self._running:
                try:
                    self._run_due_cron_jobs()
                    offset = self.db.get_telegram_offset()
                    messages = self.channel.poll_updates(offset)
                    next_offset = offset
                    for item in messages:
                        next_offset = max(next_offset, item.update_id + 1)
                        self._dispatch_message(item.chat_id, item.message_id, item.sender_name, item.text)
                    if next_offset != offset:
                        self.db.set_telegram_offset(next_offset)
                    if not messages:
                        time.sleep(self.config.daemon.idle_sleep_seconds)
                except KeyboardInterrupt:
                    raise
                except TelegramPollingConflictError:
                    LOGGER.error(
                        "stopping JClaw daemon because Telegram rejected getUpdates with 409 Conflict; "
                        "another instance is already polling this bot"
                    )
                    self._running = False
                except Exception:  # noqa: BLE001
                    LOGGER.exception("daemon loop failed")
                    time.sleep(max(self.config.daemon.idle_sleep_seconds, 3.0))
        finally:
            self.close()

    def _handle_message(self, chat_id: str, message_id: str, sender_name: str, text: str) -> None:
        if self.config.telegram.allowed_chat_ids and chat_id not in self.config.telegram.allowed_chat_ids:
            LOGGER.info("ignoring message from unauthorized chat %s", chat_id)
            return
        LOGGER.info("processing message from chat %s: %s", chat_id, text)
        initial_status = self._thinking_status(message_id, text, step=0)
        placeholder_id = self.channel.send_message(
            chat_id,
            initial_status,
            reply_to_message_id=message_id,
        )
        trace_placeholder_id: str | None = None
        trace_mode = self.db.get_trace_mode(chat_id) if hasattr(self, "db") else "off"
        if trace_mode != "off" and not text.strip().startswith("/"):
            trace_placeholder_id = self.channel.send_message(
                chat_id,
                "```text\nTrace\n(waiting for first event)\n```",
                reply_to_message_id=message_id,
            )
        stop_indicator = threading.Event()
        indicator_thread: threading.Thread | None = None
        if placeholder_id:
            indicator_thread = threading.Thread(
                target=self._run_thinking_indicator,
                args=(stop_indicator, chat_id, placeholder_id, trace_placeholder_id, message_id, text),
                daemon=True,
            )
            indicator_thread.start()
        try:
            reply = self.agent.handle_text(chat_id, text, user_name=sender_name)
        except RunInterruptedError:
            if placeholder_id:
                self.channel.edit_message(chat_id, placeholder_id, "Interrupted by a newer message.")
            elif trace_placeholder_id:
                self.channel.send_message(chat_id, "Interrupted by a newer message.", reply_to_message_id=message_id)
            if trace_placeholder_id:
                trace_text = self._render_trace_text(chat_id, latest=True)
                if trace_text:
                    self.channel.edit_message(chat_id, trace_placeholder_id, trace_text)
            return
        finally:
            stop_indicator.set()
            if indicator_thread is not None:
                indicator_thread.join(timeout=0.5)
        if placeholder_id:
            self.channel.edit_message(chat_id, placeholder_id, reply)
        else:
            self.channel.send_message(chat_id, reply, reply_to_message_id=message_id)
        if trace_placeholder_id:
            trace_text = self._render_trace_text(chat_id, latest=True)
            if trace_text:
                self.channel.edit_message(chat_id, trace_placeholder_id, trace_text)

    def _dispatch_message(self, chat_id: str, message_id: str, sender_name: str, text: str) -> None:
        self._ensure_chat_worker_state()
        queued = QueuedChatMessage(chat_id=chat_id, message_id=message_id, sender_name=sender_name, text=text)
        with self._chat_workers_lock:
            worker = self._chat_workers.get(chat_id)
            if worker is not None and worker.thread.is_alive():
                worker.pending_message = queued
                self.agent.request_interrupt(chat_id)
                return
            thread = threading.Thread(
                target=self._run_chat_worker,
                args=(queued,),
                daemon=True,
            )
            self._chat_workers[chat_id] = ChatWorkerState(thread=thread)
            thread.start()

    def _run_chat_worker(self, initial_message: QueuedChatMessage) -> None:
        self._ensure_chat_worker_state()
        current: QueuedChatMessage | None = initial_message
        while current is not None:
            try:
                self._handle_message(
                    current.chat_id,
                    current.message_id,
                    current.sender_name,
                    current.text,
                )
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001
                LOGGER.exception("message handling failed for chat %s", current.chat_id)
            with self._chat_workers_lock:
                state = self._chat_workers.get(current.chat_id)
                if state is None:
                    current = None
                    continue
                current = state.pending_message
                state.pending_message = None
                if current is None:
                    self._chat_workers.pop(initial_message.chat_id, None)

    def _ensure_chat_worker_state(self) -> None:
        if not hasattr(self, "_chat_workers"):
            self._chat_workers = {}
        if not hasattr(self, "_chat_workers_lock"):
            self._chat_workers_lock = threading.Lock()

    def _thinking_status(self, message_id: str, text: str, *, step: int) -> str:
        seed = f"{message_id}:{text}"
        index = (sum(ord(char) for char in seed) + step) % len(self.THINKING_STATUSES)
        return self.THINKING_STATUSES[index]

    def _run_thinking_indicator(
        self,
        stop_indicator: threading.Event,
        chat_id: str,
        placeholder_id: str,
        trace_placeholder_id: str | None,
        message_id: str,
        text: str,
    ) -> None:
        step = 1
        last_trace_text = ""
        while not stop_indicator.wait(self.THINKING_STATUS_INTERVAL_SECONDS):
            try:
                self.channel.edit_message(chat_id, placeholder_id, self._thinking_status(message_id, text, step=step))
                if trace_placeholder_id:
                    trace_text = self._render_trace_text(chat_id, latest=False)
                    if trace_text and trace_text != last_trace_text:
                        self.channel.edit_message(chat_id, trace_placeholder_id, trace_text)
                        last_trace_text = trace_text
            except Exception:  # noqa: BLE001
                LOGGER.exception("failed to update thinking indicator for chat %s", chat_id)
                return
            step += 1

    def _render_trace_text(self, chat_id: str, *, latest: bool) -> str:
        renderer_name = "render_latest_trace" if latest else "render_running_trace"
        renderer = getattr(self.agent, renderer_name, None)
        if callable(renderer):
            rendered = renderer(chat_id)
            return str(rendered) if rendered else ""
        return ""

    def _run_due_cron_jobs(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cron = getattr(self, "cron", self.db.cron)
        for job in cron.due_jobs(now):
            LOGGER.info("running cron job %s for chat %s", job.id, job.chat_id)
            reply = self.agent.handle_cron(job.chat_id, job.prompt)
            self.channel.send_message(job.chat_id, reply)
            spec = parse_schedule(job.schedule)
            if spec.kind == "once" or (spec.kind == "date" and spec.explicit_year):
                cron.update_job(job.chat_id, job.id, enabled=False)
            else:
                cron.update_next_run(job.id, to_utc_iso(next_run_at(spec)))
