from __future__ import annotations

from datetime import datetime, timezone
import logging
import time

from jclaw.ai.agent import AssistantAgent
from jclaw.ai.client import OpenAICompatibleClient
from jclaw.channel.telegram import TelegramBotChannel
from jclaw.core.config import Config
from jclaw.core.db import Database
from jclaw.core.scheduler import next_run_at, parse_schedule, to_utc_iso


LOGGER = logging.getLogger(__name__)


class JClawDaemon:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.daemon.db_path)
        self.llm = OpenAICompatibleClient(config.provider)
        self.channel = TelegramBotChannel(config.telegram)
        self.agent = AssistantAgent(config, self.db, self.llm)
        self._running = True

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
                        try:
                            self._handle_message(item.chat_id, item.message_id, item.sender_name, item.text)
                        except KeyboardInterrupt:
                            raise
                        except Exception:  # noqa: BLE001
                            LOGGER.exception("message handling failed for update %s", item.update_id)
                    if next_offset != offset:
                        self.db.set_telegram_offset(next_offset)
                    if not messages:
                        time.sleep(self.config.daemon.idle_sleep_seconds)
                except KeyboardInterrupt:
                    raise
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
        reply = self.agent.handle_text(chat_id, text, user_name=sender_name)
        self.channel.send_message(chat_id, reply, reply_to_message_id=message_id)

    def _run_due_cron_jobs(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for job in self.db.due_cron_jobs(now):
            LOGGER.info("running cron job %s for chat %s", job.id, job.chat_id)
            reply = self.agent.handle_cron(job.chat_id, job.prompt)
            self.channel.send_message(job.chat_id, reply)
            spec = parse_schedule(job.schedule)
            self.db.update_cron_next_run(job.id, to_utc_iso(next_run_at(spec)))
