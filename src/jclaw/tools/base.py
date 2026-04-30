from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


@dataclass(slots=True)
class ToolLoopFinalizer:
    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolLoopState:
    state: dict[str, Any] | None = None
    finalizer: ToolLoopFinalizer | None = None
    clear: bool = False
    clear_finalizer: bool = False


@dataclass(slots=True)
class ToolExecutionState:
    tool_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    finalizers: dict[str, ToolLoopFinalizer] = field(default_factory=dict)


@dataclass(slots=True)
class ToolContext:
    chat_id: str
    user_id: str = ""
    request_id: str = ""
    cwd: str = ""
    dry_run: bool = False
    execution: ToolExecutionState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    needs_confirmation: bool = False
    loop_state: ToolLoopState | None = None


@dataclass(slots=True)
class ActionSpec:
    tool: str
    action: str
    description: str
    input_schema: dict[str, Any]
    reads: bool = False
    writes: bool = False
    requires_artifacts: tuple[str, ...] = ()
    produces_artifacts: tuple[str, ...] = ()
    binding_inputs: tuple[str, ...] = ()
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "action": self.action,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "reads": self.reads,
            "writes": self.writes,
        }
        if self.requires_artifacts:
            payload["requires_artifacts"] = list(self.requires_artifacts)
        if self.produces_artifacts:
            payload["produces_artifacts"] = list(self.produces_artifacts)
        if self.binding_inputs:
            payload["binding_inputs"] = list(self.binding_inputs)
        if self.requires_confirmation:
            payload["requires_confirmation"] = True
        return payload


class DecisionType(StrEnum):
    TOOL_CALL = "tool_call"
    ANSWER = "answer"
    BLOCKED = "blocked"
    COMPLETE = "complete"


@dataclass(slots=True)
class Observation:
    ok: bool
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    artifact_types: list[str] = field(default_factory=list)
    data_preview: dict[str, Any] = field(default_factory=dict)
    missing_requirements: list[str] = field(default_factory=list)
    suggested_next_actions: list[str] = field(default_factory=list)
    needs_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "artifacts": dict(self.artifacts),
            "artifact_types": list(self.artifact_types),
            "data_preview": dict(self.data_preview),
            "missing_requirements": list(self.missing_requirements),
            "suggested_next_actions": list(self.suggested_next_actions),
            "needs_confirmation": self.needs_confirmation,
        }

    @classmethod
    def from_tool_result(cls, result: ToolResult, *, controller_contract: dict[str, Any] | None = None) -> "Observation":
        data = result.data if isinstance(result.data, dict) else {}
        artifacts = data.get("artifacts", {})
        normalized_artifacts = dict(artifacts) if isinstance(artifacts, dict) else {}
        artifact_types = sorted(
            {
                cls._artifact_type_from_id(str(artifact_id))
                for artifact_id in normalized_artifacts
                if cls._artifact_type_from_id(str(artifact_id))
            }
        )
        return cls(
            ok=result.ok,
            summary=result.summary,
            artifacts=normalized_artifacts,
            artifact_types=artifact_types,
            data_preview=cls._build_data_preview(data, controller_contract=controller_contract),
            missing_requirements=cls._normalize_string_list(data.get("missing_requirements", [])),
            suggested_next_actions=cls._normalize_string_list(data.get("suggested_next_actions", [])),
            needs_confirmation=result.needs_confirmation,
        )

    @staticmethod
    def _artifact_type_from_id(artifact_id: str) -> str:
        raw = str(artifact_id).strip()
        if not raw or ":" not in raw:
            return ""
        return raw.partition(":")[0].strip()

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @classmethod
    def _build_data_preview(cls, data: dict[str, Any], *, controller_contract: dict[str, Any] | None = None) -> dict[str, Any]:
        contract = controller_contract if isinstance(controller_contract, dict) else {}
        result_fields = [str(item) for item in contract.get("result_fields", []) if str(item).strip()]
        list_fields = contract.get("list_fields", {})
        result_previews = contract.get("result_previews", {})
        excluded_keys = {"artifacts", "missing_requirements", "suggested_next_actions"}
        preview: dict[str, Any] = {}
        if result_fields or (isinstance(list_fields, dict) and bool(list_fields)):
            for key in result_fields:
                if key in excluded_keys or key not in data:
                    continue
                preview[key] = cls._preview_value(data[key], field_name=key, preview_limits=result_previews)
            if isinstance(list_fields, dict):
                for key, limit in list_fields.items():
                    if key in excluded_keys or key not in data or not isinstance(data[key], list):
                        continue
                    try:
                        count = int(limit)
                    except (TypeError, ValueError):
                        continue
                    preview[str(key)] = [
                        cls._preview_value(item, preview_limits=result_previews, depth=1)
                        for item in data[key][:count]
                    ]
            return preview
        for index, (key, value) in enumerate(data.items()):
            if key in excluded_keys:
                continue
            if index >= 8:
                preview["__truncated__"] = True
                break
            preview[str(key)] = cls._preview_value(value)
        return preview

    @classmethod
    def _preview_value(
        cls,
        value: Any,
        *,
        depth: int = 0,
        field_name: str = "",
        preview_limits: dict[str, Any] | None = None,
    ) -> Any:
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            text = value.strip()
            limit_value = (preview_limits or {}).get(field_name)
            limit = int(limit_value) if isinstance(limit_value, int | float) and int(limit_value) > 0 else 220
            return f"{text[:limit]}..." if len(text) > limit else text
        if depth >= 2:
            return f"<{type(value).__name__}>"
        if isinstance(value, list):
            return [
                cls._preview_value(item, depth=depth + 1, preview_limits=preview_limits)
                for item in value[:3]
            ]
        if isinstance(value, dict):
            preview: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 8:
                    preview["__truncated__"] = True
                    break
                preview[str(key)] = cls._preview_value(
                    item,
                    depth=depth + 1,
                    field_name=str(key),
                    preview_limits=preview_limits,
                )
            return preview
        return str(value)


@dataclass(slots=True)
class Decision:
    type: DecisionType
    tool: str = ""
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "tool": self.tool,
            "action": self.action,
            "params": dict(self.params),
            "answer": self.answer,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Decision":
        decision_type = DecisionType(str(payload.get("type", "")).strip().lower())
        tool = str(payload.get("tool", "")).strip()
        action = str(payload.get("action", "")).strip()
        params = payload.get("params", {})
        answer = str(payload.get("answer", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        if decision_type == DecisionType.TOOL_CALL and (not tool or not action or not isinstance(params, dict)):
            raise ValueError("tool_call decisions require tool, action, and params")
        if decision_type == DecisionType.ANSWER and not answer:
            raise ValueError("answer decisions require answer text")
        if decision_type in {DecisionType.BLOCKED, DecisionType.COMPLETE}:
            tool = ""
            action = ""
            params = {}
        if decision_type == DecisionType.ANSWER:
            tool = ""
            action = ""
            params = {}
        return cls(
            type=decision_type,
            tool=tool,
            action=action,
            params=dict(params) if isinstance(params, dict) else {},
            answer=answer,
            reason=reason,
        )


@dataclass(slots=True)
class RuntimeState:
    request: str
    step_count: int = 0
    observations: list[Observation] = field(default_factory=list)
    artifacts_by_type: dict[str, Any] = field(default_factory=dict)
    artifacts_by_id: dict[str, Any] = field(default_factory=dict)
    last_decision: Decision | None = None
    last_observation: Observation | None = None
    pending_confirmation: bool = False

    def append(self, observation: Observation, *, artifact_ids: dict[str, Any] | None = None) -> None:
        self.step_count += 1
        self.observations.append(observation)
        self.last_observation = observation
        self.pending_confirmation = observation.needs_confirmation
        for artifact_id, value in (artifact_ids or observation.artifacts).items():
            artifact_type = Observation._artifact_type_from_id(str(artifact_id))
            if artifact_type:
                self.artifacts_by_type[artifact_type] = value
            self.artifacts_by_id[str(artifact_id)] = value


class Tool(Protocol):
    name: str

    def describe(self) -> dict[str, Any]:
        ...

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...

    def format_result(self, action: str, result: ToolResult) -> str:
        ...

    def materialize_params(
        self,
        action: str,
        params: dict[str, Any],
        runtime: RuntimeState,
    ) -> dict[str, Any]:
        ...
