from __future__ import annotations

import json
import logging
import base64
import mimetypes
from pathlib import Path
import re
from typing import Any

from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.prompts import load_system_prompt
from jclaw.core.config import Config
from jclaw.core.db import Database, MemoryRecord
from jclaw.core.defaults import AGENT_MAX_TOOL_STEPS
from jclaw.core.scheduler import next_run_at, parse_schedule, to_utc_iso
from jclaw.tools.automation.tool import AutomationTool
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
        if config.automation.enabled:
            self.tools.register(AutomationTool(db))
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
                    llm_chat=self.llm.chat,
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
            self.tools.register(
                KnowledgeTool(
                    self.db,
                    config.daemon.state_dir / "tools" / "knowledge",
                    config.repo_root,
                    summarize_documents=self._summarize_knowledge_documents_via_llm,
                    answer_question=self._answer_from_knowledge_documents_via_llm,
                    analyze_image=self._analyze_knowledge_image_via_llm,
                    options={
                        "max_file_read_bytes": config.knowledge.max_file_read_bytes,
                        "max_folder_scan_files": config.knowledge.max_folder_scan_files,
                        "max_chunks_per_file": config.knowledge.max_chunks_per_file,
                        "max_total_chunks": config.knowledge.max_total_chunks,
                        "text_preview_chars": config.knowledge.text_preview_chars,
                        "max_answer_citations": config.knowledge.max_answer_citations,
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
            "/cron add in 30 minutes | remind me to stretch\n"
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
            if not tool or not action or not isinstance(params, dict):
                return grant_message
            result = self.tools.invoke(
                tool,
                action,
                params,
                ToolContext(chat_id=chat_id, user_id="approval"),
            )
            formatted = self.tools.get(tool).format_result(action, result)
            return f"{grant_message}\n\n{formatted}"
        action_map = {
            "file_mutation": "apply_change_request",
            "path_mutation": "apply_path_request",
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
        return self.tools.get("workspace").format_result(action, result)

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
        return self._run_tool_loop(chat_id, text, user_name=user_name)

    def _run_tool_loop(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
    ) -> str | None:
        decision = self._decide_next_tool_step(chat_id, text, user_name=user_name, steps=[])
        if decision is None:
            return None
        if decision.get("status") in {"complete", "stop"}:
            LOGGER.info("initial tool planner declined tool use: %s", decision.get("reason", ""))
            return None
        steps: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for _ in range(AGENT_MAX_TOOL_STEPS):
            signature = json.dumps(
                {
                    "tool": decision["tool"],
                    "action": decision["action"],
                    "params": decision["params"],
                },
                sort_keys=True,
                ensure_ascii=True,
            )
            if signature in seen_signatures:
                LOGGER.info("tool loop detected repeated step; stopping")
                return "Stopped because the tool plan repeated without making progress."
            seen_signatures.add(signature)

            try:
                result = self.tools.invoke(
                    str(decision["tool"]),
                    str(decision["action"]),
                    dict(decision["params"]),
                    ToolContext(chat_id=chat_id, user_id=user_name),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "tool step failed tool=%s action=%s",
                    decision.get("tool"),
                    decision.get("action"),
                )
                return f"Tool {decision['tool']}.{decision['action']} failed: {exc}"
            steps.append(
                {
                    "tool": decision["tool"],
                    "action": decision["action"],
                    "params": dict(decision["params"]),
                    "reason": decision.get("reason", ""),
                    "result": result,
                }
            )
            if result.needs_confirmation or result.data.get("implemented") is False:
                return self._compose_tool_reply(
                    chat_id,
                    text,
                    user_name=user_name,
                    decision=decision,
                    result=result,
                )
            if result.data.get("allow_tool_followup") is False:
                return self._compose_tool_reply(
                    chat_id,
                    text,
                    user_name=user_name,
                    decision=decision,
                    result=result,
                )

            next_decision = self._decide_next_tool_step(
                chat_id,
                text,
                user_name=user_name,
                steps=steps,
            )
            if next_decision is None:
                LOGGER.info("tool continuation returned no usable decision; stopping with latest result")
                return self._compose_tool_reply(
                    chat_id,
                    text,
                    user_name=user_name,
                    decision=decision,
                    result=result,
                )
            if next_decision.get("status") in {"complete", "stop"}:
                return self._compose_tool_reply(
                    chat_id,
                    text,
                    user_name=user_name,
                    decision=decision,
                    result=result,
                )
            decision = {
                "tool": next_decision["tool"],
                "action": next_decision["action"],
                "params": next_decision["params"],
                "reason": next_decision.get("reason", ""),
            }
        last = steps[-1]
        return self._compose_tool_reply(
            chat_id,
            text,
            user_name=user_name,
            decision={"tool": last["tool"], "action": last["action"], "reason": last["reason"], "params": last["params"]},
            result=last["result"],
        )

    def _decide_next_tool_step(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        available_tools = self.tools.list_tools()
        if not available_tools:
            return None

        step_summaries = []
        for index, step in enumerate(steps, start=1):
            step_summaries.append(
                {
                    "step": index,
                    "tool": step["tool"],
                    "action": step["action"],
                    "reason": step["reason"],
                    "result": self._tool_result_for_planner(step["result"]),
                }
            )
        recent_history = [
            {"role": item.role, "content": item.content}
            for item in self.db.recent_messages(chat_id, 4)
        ]
        prompt = (
            "You are JClaw's tool step planner.\n"
            "Given the original user request and completed tool steps, decide whether the task should stop, is complete, or should continue with exactly one tool step.\n"
            "You are the only tool-planning gate for both the initial step and later continuation.\n"
            "If there are no completed steps yet, decide whether any tool should be used at all.\n"
            "Evaluate the request against the actual observed tool outputs.\n"
            "Use continue only when another tool call materially advances the user's request.\n"
            "Use complete when the user's objective has already been satisfied by the latest tool output or when no tool is needed because the task can be answered directly without tools.\n"
            "Use stop when progress is blocked, evidence is insufficient to continue safely, or another tool call is unlikely to help.\n"
            "Do not require any hardcoded tool-specific rules from the caller; rely on the tool catalog and prior step results.\n"
            "Keep params minimal and choose only one next tool step.\n"
            "Return strict JSON only.\n"
            "Schema:\n"
            '{"status":"continue|complete|stop","tool":string,"action":string,"params":object,"reason":string}\n'
            "If status is complete or stop, tool/action may be empty and params should be {}.\n"
            f"Tool catalog: {self._tool_catalog_for_prompt(available_tools)}"
        )
        raw = self.llm.chat(
            [
                {"role": "system", "content": prompt},
                *recent_history,
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_name": user_name or "unknown",
                            "request": text,
                            "steps": step_summaries,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
        )
        if step_summaries:
            LOGGER.info("tool continuation raw response: %s", raw)
        else:
            LOGGER.info("tool initial planner raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if not parsed:
            return None
        status = str(parsed.get("status", "")).strip().lower()
        if status not in {"continue", "complete", "stop"}:
            return None
        if status != "continue":
            return {"status": status, "reason": str(parsed.get("reason", "")).strip()}
        tool = str(parsed.get("tool", "")).strip()
        action = str(parsed.get("action", "")).strip()
        params = parsed.get("params", {})
        if not tool or not action or not isinstance(params, dict):
            return None
        if step_summaries:
            LOGGER.info("tool continuation selected tool=%s action=%s reason=%s", tool, action, parsed.get("reason", ""))
        else:
            LOGGER.info("tool initial planner selected tool=%s action=%s reason=%s", tool, action, parsed.get("reason", ""))
        return {
            "status": "continue",
            "tool": tool,
            "action": action,
            "params": params,
            "reason": str(parsed.get("reason", "")).strip(),
        }

    def _tool_result_for_planner(self, result: ToolResult) -> dict[str, Any]:
        data = result.data
        planner_data: dict[str, Any] = {
            "summary": result.summary,
            "needs_confirmation": result.needs_confirmation,
            "implemented": data.get("implemented", True),
        }
        for key in (
            "root_path",
            "target_path",
            "exists",
            "kind",
            "entry_count",
            "entries_truncated",
            "supported_files",
            "unsupported_files",
            "grounded",
            "partial",
            "answer",
            "summary_text",
            "request_id",
            "request_kind",
        ):
            if key in data:
                planner_data[key] = data[key]
        if "entries" in data:
            planner_data["entries"] = data["entries"][:10]
        if "citations" in data:
            planner_data["citations"] = data["citations"][:4]
        return planner_data

    def _tool_catalog_for_prompt(self, available_tools: list[dict[str, Any]]) -> str:
        catalog: list[dict[str, Any]] = []
        for tool in available_tools:
            entry: dict[str, Any] = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
            if "actions" in tool:
                entry["actions"] = tool["actions"]
            for key in (
                "implemented",
                "scaffold_only",
                "dangerous",
                "preview_required",
                "read_only",
                "grounded",
                "supports_followup",
                "path_resolution",
                "supported_suffixes",
            ):
                if key in tool:
                    entry[key] = tool[key]
            catalog.append(entry)
        return json.dumps(catalog, ensure_ascii=True)

    def _compose_tool_reply(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        decision: dict[str, Any],
        result: ToolResult,
    ) -> str:
        tool = self.tools.get(str(decision["tool"]))
        tool_result_text = tool.format_result(str(decision["action"]), result)
        if tool.describe().get("prefer_direct_result"):
            LOGGER.info("%s tool result returned directly based on tool metadata", decision["tool"])
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

    def _summarize_knowledge_documents_via_llm(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            return None
        compact_payload = {
            **payload,
            "chunks": chunks[:8],
        }
        prompt = (
            "You are JClaw's knowledge summarizer.\n"
            "Summarize only from the provided local file chunks.\n"
            "Do not invent facts or files. If the evidence is weak, say so.\n"
            "Return minified JSON only. Do not use markdown fences.\n"
            "Keep the summary under 120 words.\n"
            "Cite at most 3 chunk ids.\n"
            "Schema:\n"
            '{"summary": string, "cited_chunk_ids": [string]}\n'
            "Only cite chunk ids that appear in the payload."
        )
        raw = self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(compact_payload, ensure_ascii=True)},
            ]
        )
        LOGGER.info("knowledge summarizer raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        return self._coerce_knowledge_summary_response(raw, parsed)

    def _answer_from_knowledge_documents_via_llm(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        chunks = payload.get("chunks", [])
        question = str(payload.get("question", "")).strip()
        if not isinstance(chunks, list) or not chunks or not question:
            return None
        compact_payload = {
            **payload,
            "chunks": chunks[:8],
        }
        prompt = (
            "You are JClaw's grounded knowledge answerer.\n"
            "Answer only from the provided local file chunks.\n"
            "If the evidence is insufficient, say that directly.\n"
            "Do not invent facts, file contents, or citations.\n"
            "Return minified JSON only. Do not use markdown fences.\n"
            "Keep the answer under 120 words.\n"
            "Cite at most 3 chunk ids.\n"
            "Schema:\n"
            '{"answer": string, "cited_chunk_ids": [string], "grounded": boolean, "partial": boolean}\n'
            "Only cite chunk ids that appear in the payload."
        )
        raw = self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(compact_payload, ensure_ascii=True)},
            ]
        )
        LOGGER.info("knowledge answerer raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        return self._coerce_knowledge_answer_response(raw, parsed)

    def _coerce_knowledge_summary_response(
        self,
        raw: str,
        parsed: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if parsed:
            cited_chunk_ids = parsed.get("cited_chunk_ids", [])
            if not isinstance(cited_chunk_ids, list):
                cited_chunk_ids = []
            return {
                "summary": str(parsed.get("summary", "")).strip(),
                "cited_chunk_ids": [str(item).strip() for item in cited_chunk_ids if str(item).strip()][:3],
            }
        summary = self._extract_json_string_field(raw, "summary")
        cited_chunk_ids = self._extract_json_string_list_field(raw, "cited_chunk_ids")
        if not summary and not cited_chunk_ids:
            return None
        return {
            "summary": summary,
            "cited_chunk_ids": cited_chunk_ids[:3],
        }

    def _coerce_knowledge_answer_response(
        self,
        raw: str,
        parsed: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if parsed:
            cited_chunk_ids = parsed.get("cited_chunk_ids", [])
            if not isinstance(cited_chunk_ids, list):
                cited_chunk_ids = []
            return {
                "answer": str(parsed.get("answer", "")).strip(),
                "cited_chunk_ids": [str(item).strip() for item in cited_chunk_ids if str(item).strip()][:3],
                "grounded": bool(parsed.get("grounded", False)),
                "partial": bool(parsed.get("partial", True)),
            }
        answer = self._extract_json_string_field(raw, "answer")
        cited_chunk_ids = self._extract_json_string_list_field(raw, "cited_chunk_ids")
        if not answer and not cited_chunk_ids:
            return None
        return {
            "answer": answer,
            "cited_chunk_ids": cited_chunk_ids[:3],
            "grounded": bool(answer),
            "partial": True,
        }

    def _extract_json_string_field(self, raw: str, field_name: str) -> str:
        pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:[^"\\]|\\.)*)'
        match = re.search(pattern, raw, flags=re.DOTALL)
        if not match:
            return ""
        value = match.group(1)
        value = value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        return value.strip()

    def _extract_json_string_list_field(self, raw: str, field_name: str) -> list[str]:
        marker = f'"{field_name}"'
        start = raw.find(marker)
        if start == -1:
            return []
        bracket_start = raw.find("[", start)
        if bracket_start == -1:
            return []
        chunk = raw[bracket_start:]
        return re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', chunk)

    def _analyze_knowledge_image_via_llm(self, path: Path) -> dict[str, object] | None:
        mime_type, _ = mimetypes.guess_type(path.name)
        mime = mime_type or "image/png"
        image_bytes = path.read_bytes()
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        prompt = (
            "You are JClaw's image understanding helper.\n"
            "Describe the image and extract any visible text that matters for later file-grounded Q&A.\n"
            "Be concise and factual. Do not infer beyond the visible content.\n"
            "Return strict JSON only with schema:\n"
            '{"text": string, "warnings": [string]}\n'
        )
        raw = self.llm.chat_content(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this local image for visible content and text."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ]
        )
        LOGGER.info("knowledge image analyzer raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if not parsed:
            return None
        warnings = parsed.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        return {
            "text": str(parsed.get("text", "")).strip(),
            "warnings": [str(item).strip() for item in warnings if str(item).strip()],
        }
