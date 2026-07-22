from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from jclaw.core.defaults import AGENT_CONTINUE_TOOL_STEPS, AGENT_MAX_TOOL_STEPS, WORKSPACE_CONTINUE_TOOL_STEPS
from jclaw.tools.base import (
    Decision,
    DecisionType,
    Observation,
    RuntimeState,
    ToolContext,
    ToolExecutionState,
    ToolResult,
)

LOGGER = logging.getLogger(__name__)


class RunInterruptedError(RuntimeError):
    """Raised when a newer user turn supersedes the current run."""


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
            retrospective_followup = self._next_decision_after_retrospective(
                chat_id,
                text,
                user_name=user_name,
                runtime=runtime,
                steps=[],
                decision_type=DecisionType.ANSWER,
                candidate_answer=decision.answer,
                candidate_reason=decision.reason,
            )
            if isinstance(retrospective_followup, Decision):
                decision = retrospective_followup
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
            if isinstance(retrospective_followup, str):
                self._append_execution_trace_event(
                    chat_id,
                    "turn_blocked",
                    retrospective_followup,
                    {"mode": "retrospective_blocked"},
                )
                self._set_execution_trace_status(chat_id, "blocked")
                return retrospective_followup
            self._append_execution_trace_event(
                chat_id,
                "turn_answered",
                "Answered directly without tool use.",
                {"mode": "controller_answer", "reason": decision.reason},
            )
            self._set_execution_trace_status(chat_id, "answered")
            return decision.answer
        if decision.type is DecisionType.BLOCKED:
            self._append_execution_trace_event(
                chat_id,
                "turn_blocked",
                decision.reason or "Stopped because progress is blocked.",
                {"mode": "controller_blocked", "reason": decision.reason},
            )
            self._set_execution_trace_status(chat_id, "blocked")
            return decision.reason or "Stopped because progress is blocked."
        if decision.type is DecisionType.COMPLETE:
            retrospective_followup = self._next_decision_after_retrospective(
                chat_id,
                text,
                user_name=user_name,
                runtime=runtime,
                steps=[],
                decision_type=DecisionType.COMPLETE,
                candidate_reason=decision.reason,
            )
            if isinstance(retrospective_followup, Decision):
                decision = retrospective_followup
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
            if isinstance(retrospective_followup, str):
                self._append_execution_trace_event(
                    chat_id,
                    "turn_blocked",
                    retrospective_followup,
                    {"mode": "retrospective_blocked"},
                )
                self._set_execution_trace_status(chat_id, "blocked")
                return retrospective_followup
            LOGGER.info("initial controller completed without tool use: %s", decision.reason)
            self._append_execution_trace_event(
                chat_id,
                "turn_completed",
                decision.reason or "Completed without tool use.",
                {"mode": "controller_complete", "reason": decision.reason},
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
                self._interrupt_if_requested(
                    chat_id,
                    text,
                    runtime=runtime,
                    steps=steps,
                    decision=decision,
                )
                step_budget = max(step_budget, self._tool_loop_step_budget(decision.tool))
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
                            "reason": decision.reason,
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
                except Exception as exc:
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
                controller_output = self._tool_controller_output(decision.tool, decision.action, result)
                observation = Observation.from_tool_result(
                    result,
                    controller_output=controller_output,
                )
                runtime.append(observation)
                self._append_execution_trace_event(
                    chat_id,
                    "tool_observed",
                    f"Observed: {observation.summary}",
                    {
                        "tool": decision.tool,
                        "action": decision.action,
                        "ok": result.ok,
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
                        runtime=runtime,
                        steps=steps,
                    )
                tool = self.tools.get(decision.tool)
                if self._should_return_direct_tool_result(tool, decision.action, result):
                    self._set_execution_trace_status(chat_id, "completed")
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                        runtime=runtime,
                        steps=steps,
                    )

                self._interrupt_if_requested(
                    chat_id,
                    text,
                    runtime=runtime,
                    steps=steps,
                    decision=decision,
                )
                next_decision = self._decide_next_tool_step(
                    chat_id,
                    text,
                    user_name=user_name,
                    steps=steps,
                    runtime=runtime,
                )
                if next_decision is None:
                    retrospective_followup = self._next_decision_after_retrospective(
                        chat_id,
                        text,
                        user_name=user_name,
                        runtime=runtime,
                        steps=steps,
                        decision_type=DecisionType.COMPLETE,
                        candidate_reason="Controller produced no further usable decision; returning the latest tool result.",
                        force=True,
                    )
                    if isinstance(retrospective_followup, Decision):
                        decision = retrospective_followup
                        continue
                    if isinstance(retrospective_followup, str):
                        self._set_execution_trace_status(chat_id, "blocked")
                        return retrospective_followup
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
                        runtime=runtime,
                        steps=steps,
                    )
                if next_decision.type is DecisionType.ANSWER:
                    retrospective_followup = self._next_decision_after_retrospective(
                        chat_id,
                        text,
                        user_name=user_name,
                        runtime=runtime,
                        steps=steps,
                        decision_type=DecisionType.ANSWER,
                        candidate_answer=next_decision.answer,
                        candidate_reason=next_decision.reason,
                    )
                    if isinstance(retrospective_followup, Decision):
                        decision = retrospective_followup
                        continue
                    if isinstance(retrospective_followup, str):
                        self._pending_tool_loop_continuations.pop(chat_id, None)
                        self._append_execution_trace_event(
                            chat_id,
                            "turn_blocked",
                            retrospective_followup,
                            {"mode": "retrospective_blocked"},
                        )
                        self._set_execution_trace_status(chat_id, "blocked")
                        return retrospective_followup
                    self._pending_tool_loop_continuations.pop(chat_id, None)
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_answered",
                        "Controller answered directly from the accumulated observations.",
                        {"mode": "controller_answer", "reason": next_decision.reason},
                    )
                    self._set_execution_trace_status(chat_id, "answered")
                    return next_decision.answer
                if next_decision.type is DecisionType.BLOCKED:
                    self._pending_tool_loop_continuations.pop(chat_id, None)
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_blocked",
                        next_decision.reason or "Stopped because progress is blocked.",
                        {"mode": "controller_blocked", "reason": next_decision.reason},
                    )
                    self._set_execution_trace_status(chat_id, "blocked")
                    return next_decision.reason or self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                        runtime=runtime,
                        steps=steps,
                    )
                if next_decision.type is DecisionType.COMPLETE:
                    retrospective_followup = self._next_decision_after_retrospective(
                        chat_id,
                        text,
                        user_name=user_name,
                        runtime=runtime,
                        steps=steps,
                        decision_type=DecisionType.COMPLETE,
                        candidate_reason=next_decision.reason,
                    )
                    if isinstance(retrospective_followup, Decision):
                        decision = retrospective_followup
                        continue
                    if isinstance(retrospective_followup, str):
                        self._pending_tool_loop_continuations.pop(chat_id, None)
                        self._append_execution_trace_event(
                            chat_id,
                            "turn_blocked",
                            retrospective_followup,
                            {"mode": "retrospective_blocked"},
                        )
                        self._set_execution_trace_status(chat_id, "blocked")
                        return retrospective_followup
                    self._pending_tool_loop_continuations.pop(chat_id, None)
                    self._append_execution_trace_event(
                        chat_id,
                        "turn_completed",
                        next_decision.reason or "Controller marked the tool run as complete.",
                        {"mode": "controller_complete", "reason": next_decision.reason},
                    )
                    self._set_execution_trace_status(chat_id, "completed")
                    return self._compose_tool_reply(
                        chat_id,
                        text,
                        user_name=user_name,
                        decision=decision.to_dict(),
                        result=result,
                        runtime=runtime,
                        steps=steps,
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
                    except Exception:
                        LOGGER.exception("failed to run tool loop cleanup for %s", tool_name)

    def _interrupt_if_requested(
        self,
        chat_id: str,
        text: str,
        *,
        runtime: RuntimeState,
        steps: list[dict[str, Any]],
        decision: Decision,
    ) -> None:
        if not self._is_interrupt_requested(chat_id):
            return
        latest_tool = steps[-1]["tool"] if steps else decision.tool
        latest_action = steps[-1]["action"] if steps else decision.action
        self._mark_run_interrupted(
            chat_id,
            request=text,
            step_count=runtime.step_count,
            summary="Interrupted because a newer user message superseded this run.",
            latest_tool=latest_tool,
            latest_action=latest_action,
            latest_observation=runtime.last_observation.to_dict() if runtime.last_observation else {},
            artifact_types=sorted(runtime.artifacts_by_type.keys()),
            trace_payload={
                "latest_tool": latest_tool,
                "latest_action": latest_action,
                "step_count": len(steps),
            },
        )
        raise RunInterruptedError("Interrupted because a newer user message superseded this run.")

    def _supersede_pending_tool_loop_continuation(self, chat_id: str) -> None:
        pending = self._pending_tool_loop_continuations.pop(chat_id, None)
        if pending is None:
            return
        self._record_interrupted_run_context(
            chat_id,
            self._build_interrupted_run_context(
                request=pending.request,
                step_count=pending.runtime.step_count,
                summary="Interrupted a paused tool run because a newer user message arrived.",
                latest_tool=pending.decision.tool,
                latest_action=pending.decision.action,
                latest_observation=pending.runtime.last_observation.to_dict() if pending.runtime.last_observation else {},
                artifact_types=sorted(pending.runtime.artifacts_by_type.keys()),
            ),
        )
        self._mark_latest_paused_trace_interrupted(chat_id)

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

    def _tool_controller_output(self, tool_name: str, action: str, result: ToolResult) -> dict[str, Any] | None:
        tool = self.tools.get(tool_name)
        controller_output = getattr(tool, "controller_output", None)
        if not callable(controller_output):
            return None
        try:
            payload = controller_output(action, result)
        except Exception:
            LOGGER.exception("tool controller_output failed tool=%s action=%s", tool_name, action)
            return None
        return dict(payload) if isinstance(payload, dict) else None

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
