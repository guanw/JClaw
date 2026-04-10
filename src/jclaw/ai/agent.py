from __future__ import annotations

from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.prompts import load_system_prompt
from jclaw.core.config import Config
from jclaw.core.db import Database, MemoryRecord
from jclaw.core.scheduler import next_run_at, parse_schedule, to_utc_iso


class AssistantAgent:
    def __init__(self, config: Config, db: Database, llm: OpenAICompatibleClient) -> None:
        self.config = config
        self.db = db
        self.llm = llm
        self.system_prompt = load_system_prompt(config.provider.system_prompt_files)

    def handle_text(self, chat_id: str, text: str, *, user_name: str = "") -> str:
        command_reply = self._handle_command(chat_id, text)
        if command_reply is not None:
            self.db.store_message(chat_id, "user", text)
            self.db.store_message(chat_id, "assistant", command_reply)
            return command_reply

        self.db.store_message(chat_id, "user", text)
        messages = self._build_messages(chat_id, user_text=text, user_name=user_name)
        reply = self.llm.chat(messages)
        self.db.store_message(chat_id, "assistant", reply)
        return reply

    def handle_cron(self, chat_id: str, prompt: str) -> str:
        messages = self._build_messages(
            chat_id,
            user_text=f"Scheduled task: {prompt}",
            user_name="scheduler",
            persist_user_message=False,
        )
        reply = self.llm.chat(messages)
        self.db.store_message(chat_id, "assistant", reply)
        return reply

    def _build_messages(
        self,
        chat_id: str,
        *,
        user_text: str,
        user_name: str,
        persist_user_message: bool = True,
    ) -> list[dict[str, str]]:
        memories = self.db.search_memories(chat_id, user_text, self.config.memory.max_memory_items)
        history = self.db.recent_messages(chat_id, self.config.memory.max_context_messages * 2)
        system = self._render_system_prompt(memories)
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for item in history:
            messages.append({"role": item.role, "content": item.content})
        if not persist_user_message:
            prefix = f"From {user_name}: " if user_name else ""
            messages.append({"role": "user", "content": f"{prefix}{user_text}"})
        return messages

    def _render_system_prompt(self, memories: list[MemoryRecord]) -> str:
        if not memories:
            return self.system_prompt
        memory_lines = "\n".join(f"- {item.key}: {item.value}" for item in memories)
        return f"{self.system_prompt}\n\nRelevant memory:\n{memory_lines}"

    def _handle_command(self, chat_id: str, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return "Send a message or use /help for command syntax."

        command, _, remainder = stripped.partition(" ")
        if command.startswith("/"):
            command = command.split("@", 1)[0]

        if command in {"/help", "help"}:
            return self._help_text()
        if command in {"/remember", "remember"}:
            return self._remember(chat_id, remainder)
        if command in {"/forget", "forget"}:
            return self._forget(chat_id, remainder)
        if command in {"/memory", "memory"}:
            return self._memory(chat_id)
        if command in {"/cron", "cron"}:
            return self._cron(chat_id, remainder)
        return None

    def _remember(self, chat_id: str, remainder: str) -> str:
        key, sep, value = remainder.partition("=")
        if not sep:
            return "Usage: /remember key = value"
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return "Usage: /remember key = value"
        self.db.remember(chat_id, key, value)
        return f"Remembered '{key}'."

    def _forget(self, chat_id: str, remainder: str) -> str:
        key = remainder.strip()
        if not key:
            return "Usage: /forget key"
        deleted = self.db.forget(chat_id, key)
        if deleted:
            return f"Forgot '{key}'."
        return f"I didn't have a memory stored for '{key}'."

    def _memory(self, chat_id: str) -> str:
        items = self.db.list_memories(chat_id)
        if not items:
            return "No memories stored yet."
        lines = [f"{item.key} = {item.value}" for item in items]
        return "Stored memories:\n" + "\n".join(lines)

    def _cron(self, chat_id: str, remainder: str) -> str:
        action, _, payload = remainder.strip().partition(" ")
        action = action or "list"

        if action == "list":
            jobs = self.db.list_cron_jobs(chat_id)
            if not jobs:
                return "No cron jobs configured."
            lines = [
                f"{job.id}. {job.schedule} -> {job.prompt} (next {job.next_run_at})"
                for job in jobs
            ]
            return "Cron jobs:\n" + "\n".join(lines)

        if action == "remove":
            if not payload.strip().isdigit():
                return "Usage: /cron remove 1"
            deleted = self.db.remove_cron_job(chat_id, int(payload.strip()))
            if deleted:
                return "Cron job removed."
            return "Cron job not found."

        if action == "add":
            schedule_text, sep, prompt = payload.partition("|")
        else:
            schedule_text, sep, prompt = remainder.partition("|")

        if not sep:
            return "Usage: /cron add every 30m | your prompt"

        spec = parse_schedule(schedule_text.strip())
        scheduled_at = to_utc_iso(next_run_at(spec))
        job_id = self.db.add_cron_job(chat_id, spec.raw, prompt.strip(), scheduled_at)
        return f"Cron job {job_id} added for {spec.raw}. Next run: {scheduled_at}"

    def _help_text(self) -> str:
        return (
            "Commands:\n"
            "/remember key = value\n"
            "/memory\n"
            "/forget key\n"
            "/cron add every 30m | remind me to stretch\n"
            "/cron add daily 09:00 | ask for my standup update\n"
            "/cron list\n"
            "/cron remove 1"
        )
