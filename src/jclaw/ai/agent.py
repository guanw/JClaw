from __future__ import annotations

import json
import logging
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.prompts import load_system_prompt
from jclaw.core.config import Config
from jclaw.core.db import Database, MemoryRecord
from jclaw.core.defaults import AGENT_MAX_TOOL_STEPS
from jclaw.tools.automation.tool import AutomationTool
from jclaw.tools.base import Decision, DecisionType, Observation, RuntimeState, ToolContext, ToolExecutionState, ToolResult
from jclaw.tools.browser.tool import BrowserTool
from jclaw.tools.email.tool import EmailTool
from jclaw.tools.knowledge.tool import KnowledgeTool
from jclaw.tools.memory.tool import MemoryTool
from jclaw.tools.permissions.tool import PermissionsTool
from jclaw.tools.registry import ToolRegistry
from jclaw.tools.workspace.tool import WorkspaceTool


LOGGER = logging.getLogger(__name__)
MAX_CONTROLLER_OBSERVATIONS = 5


class AssistantAgent:
    def __init__(self, config: Config, db: Database, llm: OpenAICompatibleClient) -> None:
        self.config = config
        self.db = db
        self.llm = llm
        self.system_prompt = load_system_prompt(config.provider.system_prompt_files)
        self.tools = ToolRegistry()
        self.tools.register(MemoryTool(db, search_limit=config.memory.max_memory_items))
        self.tools.register(PermissionsTool(db))
        if config.automation.enabled:
            self.tools.register(AutomationTool(db))
        if config.email.enabled:
            self.tools.register(
                EmailTool(
                    db,
                    oauth_client_path=config.email.oauth_client_path,
                    token_dir=config.email.token_dir,
                    default_account_alias=config.email.default_account_alias,
                )
            )
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
        if command == "/remember":
            return self._remember(chat_id, remainder)
        if command == "/forget":
            return self._forget(chat_id, remainder)
        if command == "/memory":
            return self._memory(chat_id)
        if command == "/approve":
            return self._approve(chat_id, remainder)
        if command == "/deny":
            return self._deny(chat_id, remainder)
        if command == "/grants":
            return self._grants()
        if command == "/revoke":
            return self._revoke(remainder)
        if command == "/abort":
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

    def _help_text(self) -> str:
        return (
            "Commands:\n"
            "/remember key = value\n"
            "/memory\n"
            "/forget key\n"
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
        runtime = RuntimeState(request=text)
        decision = self._decide_next_tool_step(chat_id, text, user_name=user_name, steps=[], runtime=runtime)
        if decision is None:
            return None
        if decision.type is DecisionType.ANSWER:
            return decision.answer
        if decision.type is DecisionType.BLOCKED:
            return decision.reason or "Stopped because progress is blocked."
        if decision.type is DecisionType.COMPLETE:
            LOGGER.info("initial controller completed without tool use: %s", decision.reason)
            return None
        steps: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        execution = ToolExecutionState()
        try:
            for _ in range(AGENT_MAX_TOOL_STEPS):
                signature = json.dumps(
                    {
                        "tool": decision.tool,
                        "action": decision.action,
                        "params": decision.params,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                if signature in seen_signatures:
                    LOGGER.info("tool loop detected repeated step; stopping")
                    return "Stopped because the tool loop repeated without making progress."
                seen_signatures.add(signature)

                try:
                    materialized_params = self.tools.materialize_params(
                        decision.tool,
                        decision.action,
                        dict(decision.params),
                        runtime,
                    )
                    result = self.tools.invoke(
                        decision.tool,
                        decision.action,
                        materialized_params,
                        ToolContext(
                            chat_id=chat_id,
                            user_id=user_name,
                            execution=execution,
                            metadata={"loop_managed": True},
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "tool step failed tool=%s action=%s",
                        decision.tool,
                        decision.action,
                    )
                    return f"Tool {decision.tool}.{decision.action} failed: {exc}"
                steps.append(
                    {
                        "tool": decision.tool,
                        "action": decision.action,
                        "params": dict(materialized_params),
                        "reason": decision.reason,
                        "result": result,
                    }
                )
                runtime.last_decision = decision
                runtime.append(Observation.from_tool_result(result))
                if result.needs_confirmation:
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                if result.data.get("allow_tool_followup") is False:
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )

                next_decision = self._decide_next_tool_step(
                    chat_id,
                    text,
                    user_name=user_name,
                    steps=steps,
                    runtime=runtime,
                )
                if next_decision is None:
                    LOGGER.info("tool continuation returned no usable decision; stopping with latest result")
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                if next_decision.type is DecisionType.ANSWER:
                    return next_decision.answer
                if next_decision.type is DecisionType.BLOCKED:
                    return next_decision.reason or self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                if next_decision.type is DecisionType.COMPLETE:
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                decision = next_decision
            last = steps[-1]
            return self._compose_tool_reply(
                chat_id,
                text,
                user_name=user_name,
                decision={"tool": last["tool"], "action": last["action"], "reason": last["reason"], "params": last["params"]},
                result=last["result"],
            )
        finally:
            for tool_name, finalizer in execution.finalizers.items():
                try:
                    self.tools.invoke(
                        tool_name,
                        finalizer.action,
                        dict(finalizer.params),
                        ToolContext(chat_id=chat_id, user_id=user_name),
                    )
                except Exception:  # noqa: BLE001
                    LOGGER.exception("failed to run tool loop cleanup for %s", tool_name)

    def _decide_next_tool_step(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        steps: list[dict[str, Any]],
        runtime: RuntimeState,
    ) -> Decision | None:
        available_tools = self.tools.list_tools()
        if not available_tools:
            return None

        controller_state = self._controller_state_for_prompt(steps, runtime)
        recent_history = [
            {"role": item.role, "content": item.content}
            for item in self.db.recent_messages(chat_id, 4)
        ]
        prompt = (
            "You are JClaw's live tool controller.\n"
            "Given the user request and prior tool observations, choose exactly one next decision.\n"
            "Use tool_call when one tool step materially advances the request.\n"
            "Use answer when the request can now be answered directly from the observations.\n"
            "Use blocked when progress is unsafe or impossible without clarification, permission, or missing prerequisites.\n"
            "Use complete when the operational task is finished and the latest tool result should be returned to the user.\n"
            "The runtime state includes normalized observations and the current artifact frontier. Treat the latest observation as authoritative.\n"
            "The runtime state also includes the authoritative current local date, time, and timezone. Use that instead of guessing today's date or year.\n"
            "When a tool offers both a whole-resource read and a focused range read, prefer the focused range read whenever the user asks for explicit line numbers, a line range, or another clearly bounded subsection.\n"
            "Keep params minimal and choose only one next decision.\n"
            "Return strict JSON only.\n"
            "Schema:\n"
            '{"type":"tool_call|answer|blocked|complete","tool":string,"action":string,"params":object,"answer":string,"reason":string}\n'
            "For answer, provide answer and leave tool/action empty.\n"
            "For blocked or complete, leave tool/action empty and params {}.\n"
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
                            "controller_state": controller_state,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
        )
        if controller_state["observations"]:
            LOGGER.info("tool continuation raw response: %s", raw)
        else:
            LOGGER.info("tool initial controller raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if not parsed:
            parsed = self._repair_controller_response(
                raw,
                text=text,
                controller_state=controller_state,
            )
            if not parsed:
                return None
        try:
            decision = Decision.from_dict(parsed)
        except ValueError:
            return None
        if controller_state["observations"]:
            LOGGER.info(
                "tool continuation selected type=%s tool=%s action=%s reason=%s",
                decision.type.value,
                decision.tool,
                decision.action,
                decision.reason,
            )
        else:
            LOGGER.info(
                "tool initial controller selected type=%s tool=%s action=%s reason=%s",
                decision.type.value,
                decision.tool,
                decision.action,
                decision.reason,
            )
        return decision

    def _controller_state_for_prompt(
        self,
        steps: list[dict[str, Any]],
        runtime: RuntimeState,
    ) -> dict[str, Any]:
        now = self._controller_now()
        observations: list[dict[str, Any]] = []
        start_index = max(0, len(steps) - MAX_CONTROLLER_OBSERVATIONS)
        for index, step in enumerate(steps[start_index:], start=start_index + 1):
            observation = (
                runtime.observations[index - 1].to_dict()
                if index - 1 < len(runtime.observations)
                else self._tool_result_for_controller(step["result"])
            )
            observations.append(
                {
                    "step": index,
                    "tool": step["tool"],
                    "action": step["action"],
                    "reason": step["reason"],
                    "observation": observation,
                }
            )
        return {
            "step_count": runtime.step_count,
            "pending_confirmation": runtime.pending_confirmation,
            "current_local_time": now.isoformat(),
            "current_local_date": now.date().isoformat(),
            "current_local_timezone": str(now.tzinfo or ""),
            "artifact_types": sorted(runtime.artifacts_by_type.keys()),
            "artifacts_by_type": {
                str(key): self._preview_runtime_value(value)
                for key, value in runtime.artifacts_by_type.items()
            },
            "latest_observation": runtime.last_observation.to_dict() if runtime.last_observation else {},
            "observations": observations,
        }

    def _controller_now(self) -> datetime:
        return datetime.now().astimezone()

    def _preview_runtime_value(self, value: Any, *, depth: int = 0) -> Any:
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            text = value.strip()
            return f"{text[:220]}..." if len(text) > 220 else text
        if depth >= 2:
            return f"<{type(value).__name__}>"
        if isinstance(value, list):
            return [self._preview_runtime_value(item, depth=depth + 1) for item in value[:3]]
        if isinstance(value, dict):
            preview: dict[str, Any] = {}
            is_workspace_file = {"target_path", "start_line", "end_line", "line_count", "content"}.issubset(value.keys())
            is_workspace_diff = {"git_root", "status", "diff", "has_unstaged", "has_staged"}.issubset(value.keys())
            for index, (key, item) in enumerate(value.items()):
                if index >= 8:
                    preview["__truncated__"] = True
                    break
                if isinstance(item, str) and key == "content" and is_workspace_file:
                    text = item.strip()
                    preview[str(key)] = f"{text[:4000]}..." if len(text) > 4000 else text
                    continue
                if isinstance(item, str) and key == "diff" and is_workspace_diff:
                    text = item.strip()
                    preview[str(key)] = f"{text[:4000]}..." if len(text) > 4000 else text
                    continue
                preview[str(key)] = self._preview_runtime_value(item, depth=depth + 1)
            return preview
        return str(value)

    def _tool_result_for_controller(self, result: ToolResult) -> dict[str, Any]:
        data = result.data
        controller_data: dict[str, Any] = {
            "summary": result.summary,
            "needs_confirmation": result.needs_confirmation,
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
            "content",
            "line_count",
            "start_line",
            "end_line",
            "char_count",
            "bytes_read",
            "truncated",
            "git_root",
            "status",
            "diff",
            "has_unstaged",
            "has_staged",
        ):
            if key in data:
                controller_data[key] = data[key]
        if "entries" in data:
            controller_data["entries"] = data["entries"][:10]
        if "citations" in data:
            controller_data["citations"] = data["citations"][:4]
        return controller_data

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
        if self._should_return_direct_tool_result(tool.describe(), result):
            LOGGER.info("%s tool result returned directly based on tool metadata", decision["tool"])
            return tool_result_text
        if result.needs_confirmation:
            LOGGER.info("tool result requires confirmation; returning raw tool result")
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

    def _should_return_direct_tool_result(self, tool_description: dict[str, Any], result: ToolResult) -> bool:
        if not tool_description.get("prefer_direct_result"):
            return False
        if result.needs_confirmation:
            return True
        if tool_description.get("supports_followup") and result.data.get("allow_tool_followup") is not False:
            return False
        return True

    def _repair_controller_response(
        self,
        raw: str,
        *,
        text: str,
        controller_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not str(raw).strip():
            return None
        try:
            repaired = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the prior controller response as strict JSON only.\n"
                            "Allowed schema:\n"
                            '{"type":"tool_call|answer|blocked|complete","tool":string,"action":string,"params":object,"answer":string,"reason":string}\n'
                            "If the response already answers the user from available evidence, use type=answer.\n"
                            "If it requests clarification or says progress is blocked, use type=blocked.\n"
                            "If it says the operation is finished and the latest tool result should be returned, use type=complete.\n"
                            "Otherwise use type=tool_call.\n"
                            "Return strict JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "request": text,
                                "controller_state": controller_state,
                                "raw_controller_response": raw,
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
            )
        except Exception:  # noqa: BLE001
            return None
        LOGGER.info("tool controller repair raw response: %s", repaired)
        return self._parse_json_object(repaired)

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
            "You are JClaw's workspace change drafter.\n"
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
        LOGGER.info("workspace change drafter raw response: %s", raw)
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
