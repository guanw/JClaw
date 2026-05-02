from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from jclaw.core.defaults import AGENT_CONTINUE_TOOL_STEPS, AGENT_MAX_TOOL_STEPS, WORKSPACE_CONTINUE_TOOL_STEPS
from jclaw.tools.base import Decision, DecisionType, Observation, RuntimeState, ToolContext, ToolExecutionState, ToolResult


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingToolLoopContinuation:
    request: str
    user_name: str
    decision: Decision
    steps: list[dict[str, Any]]
    runtime: RuntimeState
    execution: ToolExecutionState
    seen_signatures: set[str]
    tool_name: str
    extra_steps: int
    total_steps: int


class AgentToolLoopMixin:
    def _handle_tool_request(self, chat_id: str, text: str, *, user_name: str) -> str | None:
        self._pending_tool_loop_continuations.pop(chat_id, None)
        return self._run_tool_loop(chat_id, text, user_name=user_name)

    def _handle_tool_loop_continuation(self, chat_id: str, text: str, *, user_name: str) -> str | None:
        if text.strip().lower() != "continue":
            return None
        pending = self._pending_tool_loop_continuations.pop(chat_id, None)
        if pending is None:
            self._append_execution_trace_event(
                chat_id,
                "turn_blocked",
                "No paused tool run was available to continue.",
                {"reason": "no_pending_continuation"},
            )
            self._set_execution_trace_status(chat_id, "blocked")
            return "There is no paused tool run waiting for continuation."
        self._append_execution_trace_event(
            chat_id,
            "continuation_resumed",
            f"Resumed the paused {pending.tool_name} tool loop.",
            {
                "tool": pending.tool_name,
                "step_count": len(pending.steps),
                "previous_budget": pending.total_steps,
                "extra_steps": pending.extra_steps,
            },
        )
        return self._execute_tool_loop(
            chat_id,
            pending.request,
            user_name=user_name or pending.user_name,
            decision=pending.decision,
            runtime=pending.runtime,
            steps=pending.steps,
            execution=pending.execution,
            seen_signatures=pending.seen_signatures,
            step_budget=pending.total_steps + pending.extra_steps,
        )

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
            self._append_execution_trace_event(
                chat_id,
                "controller_decision",
                self._trace_decision_summary(decision),
                self._trace_decision_payload(decision),
            )
            self._append_execution_trace_event(
                chat_id,
                "answer_composed",
                "Answered directly without tool use.",
                {"mode": "controller_answer"},
            )
            self._set_execution_trace_status(chat_id, "answered")
            return decision.answer
        if decision.type is DecisionType.BLOCKED:
            self._append_execution_trace_event(
                chat_id,
                "controller_decision",
                self._trace_decision_summary(decision),
                self._trace_decision_payload(decision),
            )
            self._append_execution_trace_event(
                chat_id,
                "turn_blocked",
                decision.reason or "Stopped because progress is blocked.",
                {"mode": "controller_blocked"},
            )
            self._set_execution_trace_status(chat_id, "blocked")
            return decision.reason or "Stopped because progress is blocked."
        if decision.type is DecisionType.COMPLETE:
            LOGGER.info("initial controller completed without tool use: %s", decision.reason)
            self._append_execution_trace_event(
                chat_id,
                "controller_decision",
                self._trace_decision_summary(decision),
                self._trace_decision_payload(decision),
            )
            self._append_execution_trace_event(
                chat_id,
                "turn_completed",
                decision.reason or "Completed without tool use.",
                {"mode": "controller_complete"},
            )
            self._set_execution_trace_status(chat_id, "completed")
            return None
        return self._execute_tool_loop(
            chat_id,
            text,
            user_name=user_name,
            decision=decision,
            runtime=runtime,
            steps=[],
            execution=ToolExecutionState(),
            seen_signatures=set(),
            step_budget=self._tool_loop_step_budget(decision.tool),
        )

    def _execute_tool_loop(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        decision: Decision,
        runtime: RuntimeState,
        steps: list[dict[str, Any]],
        execution: ToolExecutionState,
        seen_signatures: set[str],
        step_budget: int,
    ) -> str | None:
        paused = False
        try:
            while len(steps) < step_budget:
                step_budget = max(step_budget, self._tool_loop_step_budget(decision.tool))
                self._append_execution_trace_event(
                    chat_id,
                    "controller_decision",
                    self._trace_decision_summary(decision),
                    self._trace_decision_payload(decision),
                )
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
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_blocked",
                        "Stopped because the tool loop repeated without making progress.",
                        {"reason": "repeated_step"},
                    )
                    self._set_execution_trace_status(chat_id, "blocked")
                    return "Stopped because the tool loop repeated without making progress."
                seen_signatures.add(signature)

                try:
                    materialized_params = self.tools.materialize_params(
                        decision.tool,
                        decision.action,
                        dict(decision.params),
                        runtime,
                    )
                    self._append_execution_trace_event(
                        chat_id,
                        "tool_started",
                        f"Starting {decision.tool}.{decision.action}.",
                        {
                            "tool": decision.tool,
                            "action": decision.action,
                            "params": materialized_params,
                        },
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
                    self._apply_tool_loop_state(execution, decision.tool, result)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "tool step failed tool=%s action=%s",
                        decision.tool,
                        decision.action,
                    )
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_failed",
                        f"Tool {decision.tool}.{decision.action} failed: {exc}",
                        {
                            "tool": decision.tool,
                            "action": decision.action,
                            "error": str(exc),
                        },
                    )
                    self._set_execution_trace_status(chat_id, "failed")
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
                observation = Observation.from_tool_result(
                    result,
                    controller_contract=self._tool_controller_contract(decision.tool),
                )
                runtime.append(observation)
                self._append_execution_trace_event(
                    chat_id,
                    "tool_finished",
                    f"{decision.tool}.{decision.action}: {result.summary}",
                    {
                        "tool": decision.tool,
                        "action": decision.action,
                        "ok": result.ok,
                        "needs_confirmation": result.needs_confirmation,
                    },
                )
                self._append_execution_trace_event(
                    chat_id,
                    "observation_recorded",
                    f"Observed: {observation.summary}",
                    {
                        "artifact_types": observation.artifact_types,
                        "data_preview": observation.data_preview,
                        "needs_confirmation": observation.needs_confirmation,
                    },
                )
                if result.needs_confirmation:
                    self._set_execution_trace_status(chat_id, "completed")
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                if result.data.get("allow_tool_followup") is False:
                    self._set_execution_trace_status(chat_id, "completed")
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
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_completed",
                        "Controller produced no further usable decision; returning the latest tool result.",
                        {"mode": "latest_tool_result"},
                    )
                    self._set_execution_trace_status(chat_id, "completed")
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                if next_decision.type is DecisionType.ANSWER:
                    self._pending_tool_loop_continuations.pop(chat_id, None)
                    self._append_execution_trace_event(
                        chat_id,
                        "controller_decision",
                        self._trace_decision_summary(next_decision),
                        self._trace_decision_payload(next_decision),
                    )
                    self._append_execution_trace_event(
                        chat_id,
                        "answer_composed",
                        "Controller answered directly from the accumulated observations.",
                        {"mode": "controller_answer"},
                    )
                    self._set_execution_trace_status(chat_id, "answered")
                    return next_decision.answer
                if next_decision.type is DecisionType.BLOCKED:
                    self._pending_tool_loop_continuations.pop(chat_id, None)
                    self._append_execution_trace_event(
                        chat_id,
                        "controller_decision",
                        self._trace_decision_summary(next_decision),
                        self._trace_decision_payload(next_decision),
                    )
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_blocked",
                        next_decision.reason or "Stopped because progress is blocked.",
                        {"mode": "controller_blocked"},
                    )
                    self._set_execution_trace_status(chat_id, "blocked")
                    return next_decision.reason or self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                if next_decision.type is DecisionType.COMPLETE:
                    self._pending_tool_loop_continuations.pop(chat_id, None)
                    self._append_execution_trace_event(
                        chat_id,
                        "controller_decision",
                        self._trace_decision_summary(next_decision),
                        self._trace_decision_payload(next_decision),
                    )
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_completed",
                        next_decision.reason or "Controller marked the tool run as complete.",
                        {"mode": "controller_complete"},
                    )
                    self._set_execution_trace_status(chat_id, "completed")
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                    )
                decision = next_decision
            paused = True
            self._pending_tool_loop_continuations[chat_id] = PendingToolLoopContinuation(
                request=text,
                user_name=user_name,
                decision=decision,
                steps=steps,
                runtime=runtime,
                execution=execution,
                seen_signatures=seen_signatures,
                tool_name=decision.tool,
                extra_steps=self._tool_loop_continue_step_increment(decision.tool),
                total_steps=step_budget,
            )
            self._append_execution_trace_event(
                chat_id,
                "continuation_offered",
                self._tool_loop_exhausted_reply(decision.tool, step_budget),
                {
                    "tool": decision.tool,
                    "step_budget": step_budget,
                    "extra_steps": self._tool_loop_continue_step_increment(decision.tool),
                },
            )
            self._set_execution_trace_status(chat_id, "paused")
            return self._tool_loop_exhausted_reply(decision.tool, step_budget)
        finally:
            if not paused:
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

    def _apply_tool_loop_state(self, execution: ToolExecutionState, tool_name: str, result: ToolResult) -> None:
        loop_state = result.loop_state
        if loop_state is None:
            return
        if loop_state.clear:
            execution.tool_state.pop(tool_name, None)
            execution.finalizers.pop(tool_name, None)
            return
        if loop_state.state is not None:
            execution.tool_state[tool_name] = dict(loop_state.state)
        if loop_state.clear_finalizer:
            execution.finalizers.pop(tool_name, None)
        if loop_state.finalizer is not None:
            execution.finalizers[tool_name] = loop_state.finalizer

    def _tool_loop_step_budget(self, tool_name: str) -> int:
        if tool_name == "workspace":
            return max(AGENT_MAX_TOOL_STEPS, int(self.config.workspace.agent_max_tool_steps))
        return AGENT_MAX_TOOL_STEPS

    def _tool_loop_continue_step_increment(self, tool_name: str) -> int:
        if tool_name == "workspace":
            return WORKSPACE_CONTINUE_TOOL_STEPS
        return AGENT_CONTINUE_TOOL_STEPS

    def _tool_loop_exhausted_reply(self, tool_name: str, step_budget: int) -> str:
        extra_steps = self._tool_loop_continue_step_increment(tool_name)
        task_kind = "workspace/coding" if tool_name == "workspace" else "tool"
        return (
            f"Stopped after exhausting the current {task_kind} step budget ({step_budget} tool steps). "
            f"Reply `continue` if you want me to continue with {extra_steps} more tool steps."
        )
