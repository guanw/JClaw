from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jclaw.tools.base import DecisionType

LOGGER = logging.getLogger(__name__)


class RetrospectiveNextAction(StrEnum):
    ANSWER = "answer"
    TOOL_CALL = "tool_call"


@dataclass(slots=True)
class RetrospectiveCritique:
    ready_to_complete: bool
    issues: list[str] = field(default_factory=list)
    missing_verification: bool = False
    recommended_next_action: RetrospectiveNextAction = RetrospectiveNextAction.ANSWER
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_to_complete": self.ready_to_complete,
            "issues": list(self.issues),
            "missing_verification": self.missing_verification,
            "recommended_next_action": self.recommended_next_action.value,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrospectiveCritique:
        ready_to_complete = payload.get("ready_to_complete")
        if not isinstance(ready_to_complete, bool):
            raise ValueError("retrospective critique requires boolean ready_to_complete")

        missing_verification = payload.get("missing_verification", False)
        if not isinstance(missing_verification, bool):
            raise ValueError("retrospective critique requires boolean missing_verification")

        issues_raw = payload.get("issues", [])
        if not isinstance(issues_raw, list):
            raise ValueError("retrospective critique issues must be a list")
        issues = [str(item).strip() for item in issues_raw if str(item).strip()]

        action_raw = str(payload.get("recommended_next_action", "")).strip().lower()
        if not action_raw:
            raise ValueError("retrospective critique requires recommended_next_action")
        try:
            recommended_next_action = RetrospectiveNextAction(action_raw)
        except ValueError as exc:
            raise ValueError("retrospective critique recommended_next_action must be answer or tool_call") from exc

        rationale = str(payload.get("rationale", "")).strip()

        return cls(
            ready_to_complete=ready_to_complete,
            issues=issues,
            missing_verification=missing_verification,
            recommended_next_action=recommended_next_action,
            rationale=rationale,
        )


class AgentRetrospectiveMixin:
    def _run_retrospective_critique(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        runtime: Any,
        steps: list[dict[str, Any]],
        decision_type: Any,
        candidate_answer: str = "",
        candidate_reason: str = "",
    ) -> RetrospectiveCritique | None:
        retrospective_state = self._retrospective_state_for_prompt(
            chat_id=chat_id,
            runtime=runtime,
            steps=steps,
            decision_type=decision_type,
            candidate_answer=candidate_answer,
            candidate_reason=candidate_reason,
        )
        recent_history = [
            {"role": item.role, "content": item.content}
            for item in self.messages.recent(chat_id, 4)
        ]
        prompt = (
            "You are JClaw's retrospective reviewer.\n"
            "Review whether the agent is actually ready to stop on this task.\n"
            "Use the provided runtime evidence only. Be skeptical, concrete, and concise.\n"
            "Look for request mismatch, missing verification, failed checks, unresolved risk, or overbroad work.\n"
            "If the task appears complete and adequately supported by evidence, set ready_to_complete=true.\n"
            "If another tool step is needed before stopping, set ready_to_complete=false and recommended_next_action=tool_call.\n"
            "Return strict JSON only.\n"
            "Schema:\n"
            '{"ready_to_complete":boolean,"issues":[string],"missing_verification":boolean,"recommended_next_action":"answer|tool_call","rationale":string}'
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
                            "retrospective_state": retrospective_state,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
        )
        LOGGER.info("retrospective raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if parsed is None:
            parsed = self._repair_retrospective_response(
                raw,
                text=text,
                retrospective_state=retrospective_state,
            )
            if parsed is None:
                return None
        try:
            return RetrospectiveCritique.from_dict(parsed)
        except ValueError:
            repaired = self._repair_retrospective_response(
                json.dumps(parsed, ensure_ascii=True),
                text=text,
                retrospective_state=retrospective_state,
            )
            if repaired is None:
                return None
            try:
                return RetrospectiveCritique.from_dict(repaired)
            except ValueError:
                return None

    def _maybe_apply_retrospective_critique(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        runtime: Any,
        steps: list[dict[str, Any]],
        decision_type: Any,
        candidate_answer: str = "",
        candidate_reason: str = "",
        force: bool = False,
    ) -> RetrospectiveCritique | None:
        if not force and not self._should_run_retrospective_critique(  # type: ignore[attr-defined]
            decision_type=decision_type,
            runtime=runtime,
            steps=steps,
        ):
            return None
        self._append_execution_trace_event(  # type: ignore[attr-defined]
            chat_id,
            "retrospective_started",
            "Running retrospective critique before stopping.",
            {
                "decision_type": str(getattr(decision_type, "value", decision_type)),
                "step_count": runtime.step_count,
                "forced": force,
            },
        )
        critique = self._run_retrospective_critique(
            chat_id,
            text,
            user_name=user_name,
            runtime=runtime,
            steps=steps,
            decision_type=decision_type,
            candidate_answer=candidate_answer,
            candidate_reason=candidate_reason,
        )
        if critique is None:
            self._append_execution_trace_event(  # type: ignore[attr-defined]
                chat_id,
                "retrospective_unusable",
                "Retrospective critique did not return a usable structured result.",
                {"decision_type": str(getattr(decision_type, "value", decision_type))},
            )
            return None
        critique_payload = critique.to_dict()
        runtime.append_retrospective_critique(critique_payload)
        if critique.ready_to_complete:
            self._append_execution_trace_event(  # type: ignore[attr-defined]
                chat_id,
                "retrospective_allowed",
                "Retrospective critique allowed completion.",
                critique_payload,
            )
        else:
            self._append_execution_trace_event(  # type: ignore[attr-defined]
                chat_id,
                "retrospective_blocked",
                "Retrospective critique blocked completion and requested another tool step.",
                critique_payload,
            )
        return critique

    def _next_decision_after_retrospective(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        runtime: Any,
        steps: list[dict[str, Any]],
        decision_type: Any,
        candidate_answer: str = "",
        candidate_reason: str = "",
        force: bool = False,
    ) -> Any:
        critique = self._maybe_apply_retrospective_critique(
            chat_id,
            text,
            user_name=user_name,
            runtime=runtime,
            steps=steps,
            decision_type=decision_type,
            candidate_answer=candidate_answer,
            candidate_reason=candidate_reason,
            force=force,
        )
        if critique is None or critique.ready_to_complete:
            return None
        next_decision = self._decide_next_tool_step(  # type: ignore[attr-defined]
            chat_id,
            text,
            user_name=user_name,
            steps=steps,
            runtime=runtime,
        )
        if next_decision is not None and next_decision.type is DecisionType.TOOL_CALL:
            self._append_execution_trace_event(  # type: ignore[attr-defined]
                chat_id,
                "retrospective_continued",
                "Retrospective critique requested another tool step; returning to the normal controller loop.",
                {
                    "tool": next_decision.tool,
                    "action": next_decision.action,
                    "reason": next_decision.reason,
                },
            )
            return next_decision
        message = (
            "Another tool step is required before this task can be completed, but the controller did not produce "
            "a usable follow-up tool call. No new approval request was created."
        )
        self._append_execution_trace_event(  # type: ignore[attr-defined]
            chat_id,
            "retrospective_unresolved",
            "Retrospective critique requested another tool step, but no usable follow-up tool decision was produced.",
            {
                "message": message,
                "rationale": critique.rationale,
                "issues": list(critique.issues),
                "recommended_next_action": critique.recommended_next_action.value,
            },
        )
        return message

    def _retrospective_state_for_prompt(
        self,
        *,
        chat_id: str,
        runtime: Any,
        steps: list[dict[str, Any]],
        decision_type: Any,
        candidate_answer: str,
        candidate_reason: str,
    ) -> dict[str, Any]:
        controller_state = self._controller_state_for_prompt(steps, runtime, chat_id=chat_id)
        candidate_completion: dict[str, Any] = {
            "decision_type": str(getattr(decision_type, "value", decision_type)),
            "candidate_answer": str(candidate_answer).strip(),
            "candidate_reason": str(candidate_reason).strip(),
        }
        latest_step = steps[-1] if steps else {}
        if latest_step:
            candidate_completion["latest_step"] = {
                "tool": str(latest_step.get("tool", "")).strip(),
                "action": str(latest_step.get("action", "")).strip(),
                "reason": str(latest_step.get("reason", "")).strip(),
            }
        return {
            "controller_state": controller_state,
            "candidate_completion": candidate_completion,
        }

    def _repair_retrospective_response(
        self,
        raw: str,
        *,
        text: str,
        retrospective_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not str(raw).strip():
            return None
        try:
            repaired = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the prior retrospective response as strict JSON only.\n"
                            "Allowed schema:\n"
                            '{"ready_to_complete":boolean,"issues":[string],"missing_verification":boolean,"recommended_next_action":"answer|tool_call","rationale":string}\n'
                            "Return strict JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "request": text,
                                "retrospective_state": retrospective_state,
                                "raw_retrospective_response": raw,
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
            )
        except Exception:  # noqa: BLE001
            return None
        LOGGER.info("retrospective repair raw response: %s", repaired)
        return self._parse_json_object(repaired)
