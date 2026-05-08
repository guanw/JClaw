from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


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
    def from_dict(cls, payload: dict[str, Any]) -> "RetrospectiveCritique":
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
