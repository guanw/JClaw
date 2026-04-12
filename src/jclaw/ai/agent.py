from __future__ import annotations

import json
import logging
import re
from typing import Any

from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.prompts import load_system_prompt
from jclaw.core.config import Config
from jclaw.core.db import Database, MemoryRecord
from jclaw.core.scheduler import next_run_at, parse_schedule, to_utc_iso
from jclaw.tools.base import ToolContext, ToolResult
from jclaw.tools.browser.tool import BrowserTool
from jclaw.tools.registry import ToolRegistry


LOGGER = logging.getLogger(__name__)


class AssistantAgent:
    def __init__(self, config: Config, db: Database, llm: OpenAICompatibleClient) -> None:
        self.config = config
        self.db = db
        self.llm = llm
        self.system_prompt = load_system_prompt(config.provider.system_prompt_files)
        self.tools = ToolRegistry()
        if config.browser.enabled:
            self.tools.register(
                BrowserTool(
                    config.daemon.state_dir / "tools" / "browser",
                    options={
                        "channel": config.browser.channel,
                        "headless": config.browser.headless,
                        "slow_mo_ms": config.browser.slow_mo_ms,
                        "viewport_width": config.browser.viewport_width,
                        "viewport_height": config.browser.viewport_height,
                    },
                )
            )

    def handle_text(self, chat_id: str, text: str, *, user_name: str = "") -> str:
        command_reply = self._handle_command(chat_id, text)
        if command_reply is not None:
            self.db.store_message(chat_id, "user", text)
            self.db.store_message(chat_id, "assistant", command_reply)
            return command_reply

        self.db.store_message(chat_id, "user", text)

        tool_reply = self._handle_tool_request(chat_id, text, user_name=user_name)
        if tool_reply is not None:
            self.db.store_message(chat_id, "assistant", tool_reply)
            return tool_reply

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

    def _handle_tool_request(self, chat_id: str, text: str, *, user_name: str) -> str | None:
        decision = self._decide_tool_use(chat_id, text, user_name=user_name)
        if decision is None:
            return None
        result = self.tools.invoke(
            str(decision["tool"]),
            str(decision["action"]),
            dict(decision["params"]),
            ToolContext(chat_id=chat_id, user_id=user_name),
        )
        return self._compose_tool_reply(chat_id, text, user_name=user_name, decision=decision, result=result)

    def _decide_tool_use(self, chat_id: str, text: str, *, user_name: str) -> dict[str, Any] | None:
        available_tools = self.tools.list_tools()
        if not available_tools:
            return None

        recent_history = [
            {"role": item.role, "content": item.content}
            for item in self.db.recent_messages(chat_id, 4)
        ]
        router_prompt = (
            "You are the JClaw tool router. Decide whether the user's latest message should use one tool.\n"
            "Use tools when they are clearly better than a direct model-only answer.\n"
            "Prefer tools for browsing websites, checking live information, reading pages, or interacting with web content.\n"
            "Return strict JSON only. No markdown.\n"
            "Schema:\n"
            '{"use_tool": boolean, "tool": string, "action": string, "params": object, "reason": string}\n'
            "If no tool is needed, return {\"use_tool\": false, \"tool\": \"\", \"action\": \"\", \"params\": {}, \"reason\": \"...\"}.\n"
            "If using the browser tool:\n"
            '- use "open_url" when the user clearly wants a page opened\n'
            '- use "read_page" when they want the current page read\n'
            '- use "search_web" when the user wants current web results for a query\n'
            '- use "run_objective" for multi-step browsing after opening/searching\n'
            "Important: search_web and run_objective both execute real browser actions. Do not invent unsupported actions.\n"
            "- Include only parameters needed for the action.\n"
            f"Available tools: {json.dumps(available_tools, ensure_ascii=True)}"
        )
        messages = [{"role": "system", "content": router_prompt}]
        messages.extend(recent_history)
        messages.append(
            {
                "role": "user",
                "content": f"User name: {user_name or 'unknown'}\nLatest message: {text}",
            }
        )
        raw = self.llm.chat(messages)
        LOGGER.info("tool router raw response: %s", raw)
        decision = self._parse_json_object(raw)
        if not decision or not isinstance(decision, dict):
            LOGGER.info("tool router returned unparsable response; using normal chat fallback")
            return None
        if not decision.get("use_tool"):
            LOGGER.info("tool router declined tool use: %s", decision.get("reason", ""))
            return None
        tool = str(decision.get("tool", "")).strip()
        action = str(decision.get("action", "")).strip()
        params = decision.get("params", {})
        if not tool or not action or not isinstance(params, dict):
            LOGGER.info("tool router returned incomplete tool decision: %s", decision)
            return None
        LOGGER.info("tool router selected tool=%s action=%s reason=%s", tool, action, decision.get("reason", ""))
        return {"tool": tool, "action": action, "params": params, "reason": str(decision.get("reason", ""))}

    def _compose_tool_reply(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        decision: dict[str, Any],
        result: ToolResult,
    ) -> str:
        tool_result_text = self._format_tool_result(result)
        if result.data.get("implemented") is False:
            LOGGER.info("tool result is scaffold-only; returning raw tool result to avoid overclaiming")
            return tool_result_text
        messages = self._build_messages(chat_id, user_text=text, user_name=user_name)
        messages.append(
            {
                "role": "system",
                "content": (
                    "A tool has already been executed. Use the tool result to answer the user naturally.\n"
                    "Do not invent tool results. Be concise. Mention limits if the tool result is only partial.\n"
                    "Never claim that you searched a site, verified facts, clicked anything, or completed browsing steps unless the tool result explicitly shows it."
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Tool used: {decision['tool']}\n"
                    f"Action: {decision['action']}\n"
                    f"Reason: {decision.get('reason', '')}\n"
                    f"Tool result:\n{tool_result_text}"
                ),
            }
        )
        try:
            return self.llm.chat(messages)
        except Exception:  # noqa: BLE001
            return tool_result_text

    def _format_tool_result(self, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        if "url" in data and data["url"]:
            lines.append(f"URL: {data['url']}")
        if "title" in data and data["title"]:
            lines.append(f"Title: {data['title']}")
        if "text" in data and data["text"]:
            lines.append(f"Text: {str(data['text'])[:800]}")
        if "steps" in data and data["steps"]:
            lines.append("Planned steps:")
            for step in data["steps"][:5]:
                lines.append(f"- {step['action']}: {step.get('reason', '')}".strip())
        if "sessions" in data:
            lines.append(f"Sessions: {len(data['sessions'])}")
        return "\n".join(lines)

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
