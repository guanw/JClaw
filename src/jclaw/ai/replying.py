from __future__ import annotations

import json
from typing import Any

from jclaw.tools.base import RuntimeState, ToolResult


class AgentReplyingMixin:
    def _compose_tool_reply(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        decision: dict[str, Any],
        result: ToolResult,
        runtime: RuntimeState | None = None,
        steps: list[dict[str, Any]] | None = None,
    ) -> str:
        tool = self.tools.get(str(decision["tool"]))
        tool_result_text = tool.format_result(str(decision["action"]), result)
        if self._should_return_direct_tool_result(result):
            self._append_execution_trace_event(
                chat_id,
                "turn_answered",
                f"Returned the direct result from {decision['tool']}.{decision['action']}.",
                {"mode": "direct_tool_result"},
            )
            return tool_result_text
        if result.needs_confirmation:
            self._append_execution_trace_event(
                chat_id,
                "turn_answered",
                f"Returned the raw confirmation result from {decision['tool']}.{decision['action']}.",
                {"mode": "confirmation_result"},
            )
            return tool_result_text
        messages = self._build_tool_reply_messages(
            chat_id,
            text,
            user_name=user_name,
            decision=decision,
            runtime=runtime,
            steps=steps,
        )
        grounding_rules = [
            "A tool has already been executed. Use the tool result to answer the user naturally.",
            "Do not invent tool results. Be concise. Mention limits if the tool result is only partial.",
            "Treat the latest tool result as authoritative even if earlier conversation turns describe a different file state.",
            "Never claim that you searched a site, verified facts, clicked anything, or completed browsing steps unless the tool result explicitly shows it.",
        ]
        if self._should_use_strict_tool_reply_grounding(decision):
            grounding_rules.extend(
                [
                    "For this reply, answer only from the latest tool result and the user request.",
                    "Do not mention tests, commands, diffs, edits, or verification unless the latest tool result explicitly contains them.",
                    "If the latest tool result is a code read or snippet, answer only about that code.",
                ]
            )
        messages.append(
            {
                "role": "system",
                "content": "\n".join(grounding_rules),
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
                "turn_answered",
                f"Composed a natural-language reply from {decision['tool']}.{decision['action']}.",
                {"mode": "tool_reply_synthesis"},
            )
            return reply
        except Exception:  # noqa: BLE001
            self._append_execution_trace_event(
                chat_id,
                "turn_answered",
                f"Fell back to the raw tool result from {decision['tool']}.{decision['action']}.",
                {"mode": "tool_reply_fallback"},
            )
            return tool_result_text

    def _build_tool_reply_messages(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        decision: dict[str, Any],
        runtime: RuntimeState | None = None,
        steps: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        memories = []
        if not self._should_use_strict_tool_reply_grounding(decision):
            memories = self.memories.search(chat_id, text, self.config.memory.max_memory_items)
        system = self._render_system_prompt(memories)
        prefix = f"From {user_name}: " if user_name else ""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{prefix}{text}"},
        ]
        evidence = self._recent_tool_reply_evidence(decision, runtime=runtime, steps=steps)
        if evidence:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Recent relevant tool observations are provided as additional grounding. "
                        "Use them only if they are explicitly present below. "
                        "Do not claim a code edit, diff, or verification detail unless that evidence appears in the provided observations or latest tool result."
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Recent relevant observations:\n{json.dumps(evidence, ensure_ascii=True)}",
                }
            )
        return messages

    def _recent_tool_reply_evidence(
        self,
        decision: dict[str, Any],
        *,
        runtime: RuntimeState | None = None,
        steps: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        tool_name = str(decision.get("tool", "")).strip()
        action = str(decision.get("action", "")).strip()
        if not tool_name or runtime is None or steps is None:
            return []
        tool = self.tools.get(tool_name)
        builder = getattr(tool, "reply_evidence", None)
        if not callable(builder):
            return []
        try:
            evidence = builder(action, runtime, steps)
        except Exception:  # noqa: BLE001
            return []
        return list(evidence) if isinstance(evidence, list) else []

    def _should_use_strict_tool_reply_grounding(self, decision: dict[str, Any]) -> bool:
        tool_name = str(decision.get("tool", "")).strip()
        action = str(decision.get("action", "")).strip()
        return tool_name == "workspace" and action in {
            "read_file",
            "read_snippet",
            "find_symbol",
            "find_references",
            "list_file_symbols",
            "git_log",
            "git_diff",
        }

    def _should_return_direct_tool_result(self, result: ToolResult) -> bool:
        if result.needs_confirmation:
            return True
        return result.data.get("allow_tool_followup") is False
