from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
import uuid

from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.commands import AgentCommandsMixin
from jclaw.ai.controller import AgentControllerMixin
from jclaw.ai.prompts import load_system_prompt
from jclaw.ai.replying import AgentReplyingMixin
from jclaw.ai.tool_loop import AgentToolLoopMixin, PendingToolLoopContinuation, RunInterruptedError
from jclaw.ai.tracing import AgentTracingMixin
from jclaw.core.config import Config
from jclaw.core.db import Database, MemoryRecord
from jclaw.tools.automation.tool import AutomationTool
from jclaw.tools.base import ToolExecutionState
from jclaw.tools.browser.tool import BrowserTool
from jclaw.tools.email.tool import EmailTool
from jclaw.tools.knowledge.tool import KnowledgeTool
from jclaw.tools.memory.tool import MemoryTool
from jclaw.tools.notion.tool import NotionTool
from jclaw.tools.permissions.tool import PermissionsTool
from jclaw.tools.registry import ToolRegistry
from jclaw.tools.workspace.tool import WorkspaceTool


LOGGER = logging.getLogger(__name__)


class AssistantAgent(
    AgentTracingMixin,
    AgentCommandsMixin,
    AgentToolLoopMixin,
    AgentControllerMixin,
    AgentReplyingMixin,
):
    def __init__(self, config: Config, db: Database, llm: OpenAICompatibleClient) -> None:
        self.config = config
        self.db = db
        self.messages = db.messages
        self.memories = db.memories
        self.llm = llm
        self.system_prompt = load_system_prompt(config.provider.system_prompt_files)
        self._pending_tool_loop_continuations: dict[str, PendingToolLoopContinuation] = {}
        self._active_trace_ids: dict[str, str] = {}
        self._active_trace_statuses: dict[str, str] = {}
        self._run_state_lock = threading.Lock()
        self._active_run_ids: dict[str, str] = {}
        self._interrupt_requested_run_ids: dict[str, str] = {}
        self._pending_interrupted_contexts: dict[str, dict[str, object]] = {}
        self._active_interrupted_contexts: dict[str, dict[str, object]] = {}
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
        if config.notion.enabled:
            self.tools.register(NotionTool(config.notion))

    def handle_text(self, chat_id: str, text: str, *, user_name: str = "") -> str:
        command_reply = self._handle_command(chat_id, text)
        if command_reply is not None:
            self.messages.store(chat_id, "user", text)
            self.messages.store(chat_id, "assistant", command_reply)
            return command_reply

        run_id = self._begin_run(chat_id)
        if text.strip().lower() != "continue":
            self._supersede_pending_tool_loop_continuation(chat_id)
        trace_id = self._start_execution_trace(chat_id, text)
        reply = ""
        try:
            continuation_reply = self._handle_tool_loop_continuation(chat_id, text, user_name=user_name)
            if continuation_reply is not None:
                self.messages.store(chat_id, "user", text)
                self.messages.store(chat_id, "assistant", continuation_reply)
                reply = continuation_reply
                return continuation_reply

            self.messages.store(chat_id, "user", text)

            tool_reply = self._handle_tool_request(chat_id, text, user_name=user_name)
            if tool_reply is not None:
                self.messages.store(chat_id, "assistant", tool_reply)
                reply = tool_reply
                return tool_reply

            self._append_execution_trace_event(
                chat_id,
                "tool_path_skipped",
                "Tool selection did not produce a usable next step; falling back to a direct reply.",
                {"mode": "tool_request_returned_none"},
            )
            messages = self._build_messages(chat_id, user_text=text, user_name=user_name)
            reply = self.llm.chat(messages)
            if self._is_interrupt_requested(chat_id):
                self._mark_run_interrupted(
                    chat_id,
                    request=text,
                    summary="Interrupted because a newer user message superseded this run.",
                    trace_payload={"mode": "direct_llm"},
                )
                raise RunInterruptedError("Interrupted because a newer user message superseded this run.")
            self._append_execution_trace_event(
                chat_id,
                "turn_answered",
                "Composed a direct reply without tool use.",
                {"mode": "direct_llm"},
            )
            self._set_execution_trace_status(chat_id, "answered")
            self.messages.store(chat_id, "assistant", reply)
            return reply
        except RunInterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._append_execution_trace_event(
                chat_id,
                "turn_failed",
                f"Turn failed: {exc}",
                {"error": str(exc)},
            )
            self._set_execution_trace_status(chat_id, "failed")
            raise
        finally:
            self._end_run(chat_id, run_id)
            if trace_id:
                self._finish_execution_trace(chat_id, final_reply=reply)

    def handle_cron(self, chat_id: str, prompt: str) -> str:
        messages = self._build_messages(
            chat_id,
            user_text=f"Scheduled task: {prompt}",
            user_name="scheduler",
            persist_user_message=False,
        )
        reply = self.llm.chat(messages)
        self.messages.store(chat_id, "assistant", reply)
        return reply

    def _build_messages(
        self,
        chat_id: str,
        *,
        user_text: str,
        user_name: str,
        persist_user_message: bool = True,
    ) -> list[dict[str, str]]:
        memories = self.memories.search(chat_id, user_text, self.config.memory.max_memory_items)
        history = self.messages.recent(chat_id, self.config.memory.max_context_messages * 2)
        system = self._render_system_prompt(memories)
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        interrupted_context = self._current_interrupted_context(chat_id)
        if interrupted_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Interrupted in-flight context from the immediately previous superseded user turn:\n"
                        f"{json.dumps(interrupted_context, ensure_ascii=True)}\n"
                        "Use this only if it is relevant to the new request. If the new request appears to refine or revise the interrupted one, "
                        "prefer reusing the interrupted context rather than treating it as an unrelated task."
                    ),
                }
            )
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

    def request_interrupt(self, chat_id: str) -> bool:
        with self._run_state_lock:
            run_id = self._active_run_ids.get(chat_id, "")
            if not run_id:
                return False
            self._interrupt_requested_run_ids[chat_id] = run_id
            return True

    def _begin_run(self, chat_id: str) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        with self._run_state_lock:
            self._active_run_ids[chat_id] = run_id
            self._interrupt_requested_run_ids.pop(chat_id, None)
            pending_context = self._pending_interrupted_contexts.pop(chat_id, None)
            if pending_context is not None:
                self._active_interrupted_contexts[chat_id] = pending_context
            else:
                self._active_interrupted_contexts.pop(chat_id, None)
        return run_id

    def _end_run(self, chat_id: str, run_id: str) -> None:
        with self._run_state_lock:
            if self._active_run_ids.get(chat_id) == run_id:
                self._active_run_ids.pop(chat_id, None)
                self._interrupt_requested_run_ids.pop(chat_id, None)
            self._active_interrupted_contexts.pop(chat_id, None)

    def _is_interrupt_requested(self, chat_id: str) -> bool:
        with self._run_state_lock:
            active_run_id = self._active_run_ids.get(chat_id, "")
            requested_run_id = self._interrupt_requested_run_ids.get(chat_id, "")
            return bool(active_run_id) and active_run_id == requested_run_id

    def _record_interrupted_run_context(self, chat_id: str, context: dict[str, object]) -> None:
        with self._run_state_lock:
            self._pending_interrupted_contexts[chat_id] = dict(context)

    def _current_interrupted_context(self, chat_id: str) -> dict[str, object] | None:
        with self._run_state_lock:
            active_context = self._active_interrupted_contexts.get(chat_id)
            if active_context is not None:
                return dict(active_context)
            pending_context = self._pending_interrupted_contexts.get(chat_id)
            return dict(pending_context) if pending_context is not None else None

    def _build_interrupted_run_context(
        self,
        *,
        request: str,
        summary: str,
        step_count: int = 0,
        latest_tool: str = "",
        latest_action: str = "",
        latest_observation: dict[str, object] | None = None,
        artifact_types: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "request": request,
            "step_count": int(step_count),
            "summary": summary,
            "latest_tool": latest_tool,
            "latest_action": latest_action,
            "latest_observation": dict(latest_observation or {}),
            "artifact_types": list(artifact_types or []),
            "latest_observation_summary": str((latest_observation or {}).get("summary", "")),
        }

    def _mark_run_interrupted(
        self,
        chat_id: str,
        *,
        request: str,
        summary: str,
        step_count: int = 0,
        latest_tool: str = "",
        latest_action: str = "",
        latest_observation: dict[str, object] | None = None,
        artifact_types: list[str] | None = None,
        trace_payload: dict[str, object] | None = None,
    ) -> None:
        self._append_execution_trace_event(
            chat_id,
            "turn_interrupted",
            summary,
            dict(trace_payload or {}),
        )
        self._set_execution_trace_status(chat_id, "interrupted")
        self._record_interrupted_run_context(
            chat_id,
            self._build_interrupted_run_context(
                request=request,
                step_count=step_count,
                summary=summary,
                latest_tool=latest_tool,
                latest_action=latest_action,
                latest_observation=latest_observation,
                artifact_types=artifact_types,
            ),
        )
