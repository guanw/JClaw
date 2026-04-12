from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.prompts import load_system_prompt
from jclaw.core.config import Config
from jclaw.core.db import Database, MemoryRecord
from jclaw.core.scheduler import next_run_at, parse_schedule, to_utc_iso
from jclaw.tools.base import ToolContext, ToolResult
from jclaw.tools.browser.tool import BrowserTool
from jclaw.tools.knowledge.tool import KnowledgeTool
from jclaw.tools.registry import ToolRegistry
from jclaw.tools.workspace.tool import WorkspaceTool


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
                        "max_objective_steps": config.browser.max_objective_steps,
                        "max_research_sources": config.browser.max_research_sources,
                    },
                    choose_link=self._choose_browser_link_via_llm,
                    choose_next_action=self._choose_browser_next_action_via_llm,
                )
            )
        if config.workspace.enabled:
            self.tools.register(
                WorkspaceTool(
                    self.db,
                    config.daemon.state_dir / "tools" / "workspace",
                    config.repo_root,
                    draft_change=self._draft_workspace_change_via_llm,
                    options={
                        "max_steps": config.workspace.max_steps,
                        "shell_timeout_seconds": config.workspace.shell_timeout_seconds,
                        "shell_output_chars": config.workspace.shell_output_chars,
                        "max_prepared_diff_bytes": config.workspace.max_prepared_diff_bytes,
                        "max_files_per_change": config.workspace.max_files_per_change,
                        "max_path_entries": config.workspace.max_path_entries,
                        "max_internal_read_bytes": config.workspace.max_internal_read_bytes,
                    },
                )
            )
        if config.knowledge.enabled:
            self.tools.register(KnowledgeTool())

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
        if command in {"/approve", "approve"}:
            return self._approve(chat_id, remainder)
        if command in {"/deny", "deny"}:
            return self._deny(chat_id, remainder)
        if command in {"/grants", "grants"}:
            return self._grants()
        if command in {"/revoke", "revoke"}:
            return self._revoke(remainder)
        if command in {"/abort", "abort"}:
            return self._abort(chat_id, remainder)
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
            "/cron remove 1\n"
            "/approve req_123\n"
            "/deny req_123\n"
            "/grants\n"
            "/revoke 1\n"
            "/abort req_123"
        )

    def _approve(self, chat_id: str, remainder: str) -> str:
        request_id = remainder.strip()
        if not request_id:
            return "Usage: /approve req_123"
        request = self.db.get_approval_request(request_id)
        if request is None or request.chat_id != chat_id:
            return "Approval request not found."
        if request.status != "pending":
            return f"Approval request {request_id} is already {request.status}."
        if request.kind == "grant":
            grant = self.db.upsert_grant(request.root_path, request.capabilities, chat_id)
            self.db.update_approval_request_status(request_id, "approved")
            grant_message = (
                f"Granted {', '.join(grant.capabilities)} access for {grant.root_path}. "
                f"Grant id: {grant.id}"
            )
            continuation = request.payload.get("continuation", {})
            if not isinstance(continuation, dict):
                return grant_message
            tool = str(continuation.get("tool", "")).strip()
            action = str(continuation.get("action", "")).strip()
            params = continuation.get("params", {})
            if tool != "workspace" or not action or not isinstance(params, dict):
                return grant_message
            result = self.tools.invoke(
                tool,
                action,
                params,
                ToolContext(chat_id=chat_id, user_id="approval"),
            )
            return f"{grant_message}\n\n{self._format_tool_result(result)}"
        action_map = {
            "file_mutation": "apply_change_request",
            "git_mutation": "apply_git_request",
            "shell_mutation": "apply_shell_request",
        }
        action = action_map.get(request.kind)
        if action is None:
            return f"Approval request kind '{request.kind}' is not supported."
        result = self.tools.invoke(
            "workspace",
            action,
            {"request_id": request_id},
            ToolContext(chat_id=chat_id, user_id="approval"),
        )
        return self._format_tool_result(result)

    def _deny(self, chat_id: str, remainder: str) -> str:
        request_id = remainder.strip()
        if not request_id:
            return "Usage: /deny req_123"
        request = self.db.get_approval_request(request_id)
        if request is None or request.chat_id != chat_id:
            return "Approval request not found."
        if request.status != "pending":
            return f"Approval request {request_id} is already {request.status}."
        self.db.update_approval_request_status(request_id, "denied")
        return f"Denied request {request_id}."

    def _grants(self) -> str:
        grants = self.db.list_grants(active_only=True)
        if not grants:
            return "No active grants."
        lines = [f"{grant.id}. {grant.root_path} [{', '.join(grant.capabilities)}]" for grant in grants]
        return "Active grants:\n" + "\n".join(lines)

    def _revoke(self, remainder: str) -> str:
        token = remainder.strip()
        if not token.isdigit():
            return "Usage: /revoke 1"
        revoked = self.db.revoke_grant(int(token))
        if revoked:
            return "Grant revoked."
        return "Grant not found."

    def _abort(self, chat_id: str, remainder: str) -> str:
        request_id = remainder.strip()
        if not request_id:
            return "Usage: /abort req_123"
        request = self.db.get_approval_request(request_id)
        if request is None or request.chat_id != chat_id:
            return "Request not found."
        result = self.tools.invoke(
            "workspace",
            "abort_request",
            {"request_id": request_id},
            ToolContext(chat_id=chat_id, user_id="abort"),
        )
        return self._format_tool_result(result)

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
            "Prefer tools for browsing websites, checking live information, interacting with web content, inspecting local workspaces, preparing file/code changes, checking git state, or previewing local shell commands.\n"
            "Return strict JSON only. No markdown.\n"
            "Schema:\n"
            '{"use_tool": boolean, "tool": string, "action": string, "params": object, "reason": string}\n'
            "If no tool is needed, return {\"use_tool\": false, \"tool\": \"\", \"action\": \"\", \"params\": {}, \"reason\": \"...\"}.\n"
            "Do not choose tools whose description says implemented=false or scaffold_only=true.\n"
            "If using the browser tool:\n"
            '- use "open_url" when the user clearly wants a page opened\n'
            '- use "read_page" when they want the current page read\n'
            '- use "search_web" when the user wants current web results for a query\n'
            '- use "run_objective" for multi-step browsing after opening/searching\n'
            "If using the workspace tool:\n"
            f'- use "inspect_root" for listing or checking local paths, defaulting to the repo root {self.config.repo_root}\n'
            f'- when the user refers to common Mac folders like Documents, Desktop, Downloads, Pictures, or Library, prefer paths under the local home directory {Path.home()}\n'
            '- use "prepare_change" for local file or code modifications\n'
            '- use "git_status" when the user wants local git state\n'
            '- use "prepare_git_action" for local git mutations such as stage, restore, or commit\n'
            '- use "prepare_shell_action" for local non-interactive commands in an approved root\n'
            "Important: workspace mutation actions are preview-only until the user approves them with a command.\n"
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
        if decision["tool"] == "workspace":
            LOGGER.info("workspace tool result returned directly to avoid hallucinated summaries")
            return tool_result_text
        if result.data.get("implemented") is False or result.needs_confirmation:
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
        if "root_path" in data and data["root_path"]:
            lines.append(f"Root: {data['root_path']}")
        if "target_path" in data and data["target_path"]:
            lines.append(f"Target: {data['target_path']}")
        if "exists" in data:
            lines.append(f"Exists: {data['exists']}")
        if "kind" in data and data["kind"]:
            lines.append(f"Kind: {data['kind']}")
        if "entry_count" in data and data["entry_count"] and data.get("kind") == "directory":
            lines.append(f"Total entries: {data['entry_count']}")
        if "text" in data and data["text"]:
            lines.append(f"Text: {str(data['text'])[:800]}")
        if "request_id" in data and data["request_id"]:
            lines.append(f"Request: {data['request_id']}")
        if "request_kind" in data and data["request_kind"]:
            lines.append(f"Request kind: {data['request_kind']}")
        if "capabilities" in data and data["capabilities"]:
            lines.append(f"Capabilities: {', '.join(str(item) for item in data['capabilities'])}")
        if "entries" in data and data["entries"]:
            lines.append("Entries:")
            for entry in data["entries"][:10]:
                lines.append(f"- {entry['kind']}: {entry['name']}")
            if data.get("entries_truncated"):
                shown = len(data["entries"])
                total = data.get("entry_count", shown)
                lines.append(f"Shown {shown} of {total} entries.")
        elif data.get("kind") == "directory":
            lines.append("Entries: none")
        if "touched_files" in data and data["touched_files"]:
            lines.append("Touched files:")
            for file_path in data["touched_files"][:10]:
                lines.append(f"- {file_path}")
        if "diff_preview" in data and data["diff_preview"]:
            lines.append(f"Diff preview:\n{str(data['diff_preview'])[:1500]}")
        if "command" in data and data["command"]:
            lines.append(f"Command: {data['command']}")
        if "preview" in data and data["preview"]:
            lines.append(f"Preview: {data['preview']}")
        if "status" in data and data["status"]:
            lines.append(f"Git status:\n{str(data['status'])[:1200]}")
        if "diff_stat" in data and data["diff_stat"]:
            lines.append(f"Git diff:\n{str(data['diff_stat'])[:1200]}")
        if "stdout" in data and data["stdout"]:
            lines.append(f"Stdout:\n{str(data['stdout'])[:1200]}")
        if "stderr" in data and data["stderr"]:
            lines.append(f"Stderr:\n{str(data['stderr'])[:1200]}")
        if "output" in data and data["output"]:
            lines.append(f"Output:\n{str(data['output'])[:1200]}")
        if "sources" in data and data["sources"]:
            lines.append("Sources:")
            for source in data["sources"][:4]:
                title = str(source.get("title", "")).strip() or "Untitled"
                url = str(source.get("url", "")).strip()
                lines.append(f"- {title}: {url}")
        if "termination_reason" in data and data["termination_reason"]:
            lines.append(f"Termination: {data['termination_reason']}")
        if "steps" in data and data["steps"]:
            lines.append("Executed steps:")
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

    def _choose_browser_link_via_llm(self, objective: str, page_data: dict[str, Any]) -> str | None:
        elements = page_data.get("elements", [])
        if not isinstance(elements, list) or not elements:
            return None

        compact_elements: list[dict[str, Any]] = []
        for item in elements[:20]:
            if not isinstance(item, dict):
                continue
            href = str(item.get("href", "")).strip()
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            if role != "link":
                continue
            if not href.startswith("http"):
                continue
            compact_elements.append(
                {
                    "id": str(item.get("id", "")),
                    "role": role,
                    "text": text[:200],
                    "href": href,
                    "area": str(item.get("area", "")),
                    "clickable": bool(item.get("clickable", False)),
                    "score_hint": item.get("score_hint", 0),
                }
            )
        if not compact_elements:
            return None

        chooser_prompt = (
            "You are helping JClaw choose the next browser link to follow.\n"
            "You are given a user objective and a compact inspected-elements snapshot from the current page.\n"
            "Choose the single best link element that most likely advances the user's objective.\n"
            "Avoid search-engine homepages, settings, privacy/help pages, app-store links, download/install pages, and obvious ads.\n"
            "Prefer result/article/documentation/news pages that directly match the user's request.\n"
            "Return strict JSON only with schema:\n"
            '{"chosen_element_id": string | null, "reason": string}\n'
            "Use null if none of the elements look useful."
        )
        chooser_payload = {
            "objective": objective,
            "page_url": page_data.get("url", ""),
            "page_title": page_data.get("title", ""),
            "page_kind": page_data.get("page_kind", ""),
            "page_text_preview": str(page_data.get("text", ""))[:800],
            "elements": compact_elements,
        }
        raw = self.llm.chat(
            [
                {"role": "system", "content": chooser_prompt},
                {"role": "user", "content": json.dumps(chooser_payload, ensure_ascii=True)},
            ]
        )
        LOGGER.info("browser link chooser raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if not parsed:
            return None
        chosen_element_id = parsed.get("chosen_element_id")
        if chosen_element_id in (None, "", "null"):
            return None
        chosen_id = str(chosen_element_id).strip()
        for item in compact_elements:
            if item.get("id") == chosen_id:
                href = str(item.get("href", "")).strip()
                return href if href.startswith("http") else None
        return None

    def _choose_browser_next_action_via_llm(
        self,
        objective: str,
        page_data: dict[str, Any],
        sources: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        elements = page_data.get("elements", [])
        if not isinstance(elements, list):
            return None

        compact_elements: list[dict[str, Any]] = []
        for item in elements[:20]:
            if not isinstance(item, dict):
                continue
            href = str(item.get("href", "")).strip()
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            compact_elements.append(
                {
                    "id": str(item.get("id", "")),
                    "role": role,
                    "text": text[:180],
                    "href": href,
                    "area": str(item.get("area", "")),
                    "clickable": bool(item.get("clickable", False)),
                    "score_hint": item.get("score_hint", 0),
                }
            )

        controller_prompt = (
            "You are JClaw's browser controller.\n"
            "Decide whether the current browser mission is complete, should continue by following one visible link, or should stop because no meaningful progress is likely.\n"
            "Use only the provided page observation and gathered sources.\n"
            "Prefer COMPLETE only when there is enough concrete information to answer the objective.\n"
            "Prefer STOP when the page is low-signal, repetitive, blocked, or unlikely to add value.\n"
            "Prefer FOLLOW when a visible link is likely to materially improve the answer.\n"
            "Avoid describing future actions as completed work.\n"
            "Return strict JSON only with schema:\n"
            '{"status":"follow|complete|stop","chosen_element_id":string|null,"reason":string}\n'
            "If status is follow, chosen_element_id must identify one visible link element from the snapshot.\n"
            "If no meaningful link should be followed, return complete or stop."
        )
        payload = {
            "objective": objective,
            "current_page": {
                "url": page_data.get("url", ""),
                "title": page_data.get("title", ""),
                "page_kind": page_data.get("page_kind", ""),
                "text_preview": str(page_data.get("text", ""))[:1200],
            },
            "sources": sources[-3:],
            "elements": compact_elements,
        }
        raw = self.llm.chat(
            [
                {"role": "system", "content": controller_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
        )
        LOGGER.info("browser controller raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if not parsed:
            return None
        status = str(parsed.get("status", "")).strip().lower()
        if status not in {"follow", "complete", "stop"}:
            return None
        chosen_element_id = parsed.get("chosen_element_id")
        if status != "follow":
            return {"status": status, "url": None, "reason": str(parsed.get("reason", ""))}
        chosen_id = None if chosen_element_id in (None, "", "null") else str(chosen_element_id).strip()
        if not chosen_id:
            return None
        for item in compact_elements:
            if item.get("id") == chosen_id and str(item.get("href", "")).startswith("http"):
                return {"status": "follow", "url": str(item["href"]), "reason": str(parsed.get("reason", ""))}
        return None

    def _draft_workspace_change_via_llm(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        files = payload.get("files", [])
        if not isinstance(files, list) or not files:
            return None
        prompt = (
            "You are JClaw's workspace change planner.\n"
            "You are given an objective and a bounded set of candidate files from a local workspace.\n"
            "Draft file edits using only the provided files. Do not invent extra files.\n"
            "Return strict JSON only with schema:\n"
            '{"summary": string, "edits": [{"path": string, "reason": string, "new_content": string}]}\n'
            "Use the provided relative file paths exactly.\n"
            "If no safe or useful change can be prepared, return an empty edits list."
        )
        raw = self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
        )
        LOGGER.info("workspace change planner raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if not parsed:
            return None
        edits = parsed.get("edits", [])
        if not isinstance(edits, list):
            return None
        return {
            "summary": str(parsed.get("summary", "Prepared workspace change.")),
            "edits": edits,
        }
