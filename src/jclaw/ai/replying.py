from __future__ import annotations

from typing import Any

from jclaw.tools.base import ToolResult


class AgentReplyingMixin:
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
            self._append_execution_trace_event(
                chat_id,
                "answer_composed",
                f"Returned the direct result from {decision['tool']}.{decision['action']}.",
                {"mode": "direct_tool_result"},
            )
            return tool_result_text
        if result.needs_confirmation:
            self._append_execution_trace_event(
                chat_id,
                "answer_composed",
                f"Returned the raw confirmation result from {decision['tool']}.{decision['action']}.",
                {"mode": "confirmation_result"},
            )
            return tool_result_text
        messages = self._build_tool_reply_messages(chat_id, text, user_name=user_name)
        messages.append(
            {
                "role": "system",
                "content": (
                    "A tool has already been executed. Use the tool result to answer the user naturally.\n"
                    "Do not invent tool results. Be concise. Mention limits if the tool result is only partial.\n"
                    "Treat the latest tool result as authoritative even if earlier conversation turns describe a different file state.\n"
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
            reply = self.llm.chat(messages)
            self._append_execution_trace_event(
                chat_id,
                "answer_composed",
                f"Composed a natural-language reply from {decision['tool']}.{decision['action']}.",
                {"mode": "tool_reply_synthesis"},
            )
            return reply
        except Exception:  # noqa: BLE001
            self._append_execution_trace_event(
                chat_id,
                "answer_composed",
                f"Fell back to the raw tool result from {decision['tool']}.{decision['action']}.",
                {"mode": "tool_reply_fallback"},
            )
            return tool_result_text

    def _build_tool_reply_messages(self, chat_id: str, text: str, *, user_name: str) -> list[dict[str, str]]:
        memories = self.db.search_memories(chat_id, text, self.config.memory.max_memory_items)
        system = self._render_system_prompt(memories)
        prefix = f"From {user_name}: " if user_name else ""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{prefix}{text}"},
        ]

    def _should_return_direct_tool_result(self, tool_description: dict[str, Any], result: ToolResult) -> bool:
        if not tool_description.get("prefer_direct_result"):
            return False
        if result.needs_confirmation:
            return True
        if tool_description.get("supports_followup") and result.data.get("allow_tool_followup") is not False:
            return False
        return True
