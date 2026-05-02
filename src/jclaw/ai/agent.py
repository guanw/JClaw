from __future__ import annotations

import logging
from pathlib import Path

from jclaw.ai.client import OpenAICompatibleClient
from jclaw.ai.commands import AgentCommandsMixin
from jclaw.ai.controller import AgentControllerMixin
from jclaw.ai.prompts import load_system_prompt
from jclaw.ai.replying import AgentReplyingMixin
from jclaw.ai.tool_loop import AgentToolLoopMixin, PendingToolLoopContinuation
from jclaw.ai.tracing import AgentTracingMixin
from jclaw.core.config import Config
from jclaw.core.db import Database, MemoryRecord
from jclaw.tools.automation.tool import AutomationTool
from jclaw.tools.base import ToolExecutionState
from jclaw.tools.browser.tool import BrowserTool
from jclaw.tools.email.tool import EmailTool
from jclaw.tools.knowledge.tool import KnowledgeTool
from jclaw.tools.memory.tool import MemoryTool
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
        self.llm = llm
        self.system_prompt = load_system_prompt(config.provider.system_prompt_files)
        self._pending_tool_loop_continuations: dict[str, PendingToolLoopContinuation] = {}
        self._active_trace_ids: dict[str, str] = {}
        self._active_trace_statuses: dict[str, str] = {}
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

    def handle_text(self, chat_id: str, text: str, *, user_name: str = "") -> str:
        command_reply = self._handle_command(chat_id, text)
        if command_reply is not None:
            self.db.store_message(chat_id, "user", text)
            self.db.store_message(chat_id, "assistant", command_reply)
            return command_reply

        trace_id = self._start_execution_trace(chat_id, text)
        reply = ""
        try:
            continuation_reply = self._handle_tool_loop_continuation(chat_id, text, user_name=user_name)
            if continuation_reply is not None:
                self.db.store_message(chat_id, "user", text)
                self.db.store_message(chat_id, "assistant", continuation_reply)
                reply = continuation_reply
                return continuation_reply

            self.db.store_message(chat_id, "user", text)

            tool_reply = self._handle_tool_request(chat_id, text, user_name=user_name)
            if tool_reply is not None:
                self.db.store_message(chat_id, "assistant", tool_reply)
                reply = tool_reply
                return tool_reply

            messages = self._build_messages(chat_id, user_text=text, user_name=user_name)
            reply = self.llm.chat(messages)
            self._append_execution_trace_event(
                chat_id,
                "answer_composed",
                "Composed a direct reply without tool use.",
                {"mode": "direct_llm"},
            )
            self._set_execution_trace_status(chat_id, "answered")
            self.db.store_message(chat_id, "assistant", reply)
            return reply
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
