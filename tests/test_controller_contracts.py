from __future__ import annotations

import pytest

from jclaw.tools.base import (
    ActionSpec,
    Decision,
    DecisionType,
    Observation,
    RuntimeState,
    ToolResult,
)


def test_action_spec_to_dict_serializes_contract_fields() -> None:
    spec = ActionSpec(
        tool="email",
        action="draft_reply",
        description="Create a draft reply.",
        input_schema={"type": "object", "properties": {"body": {"type": "string"}}},
        reads=True,
        writes=True,
        requires_artifacts=("message_ref",),
        produces_artifacts=("email_draft",),
        binding_inputs=("message_ref", "thread_ref"),
        requires_confirmation=True,
    )

    assert spec.to_dict() == {
        "tool": "email",
        "action": "draft_reply",
        "description": "Create a draft reply.",
        "input_schema": {"type": "object", "properties": {"body": {"type": "string"}}},
        "reads": True,
        "writes": True,
        "requires_artifacts": ["message_ref"],
        "produces_artifacts": ["email_draft"],
        "binding_inputs": ["message_ref", "thread_ref"],
        "requires_confirmation": True,
    }


def test_observation_from_tool_result_normalizes_artifacts_and_preview() -> None:
    result = ToolResult(
        ok=True,
        summary="Selected email 'Follow Up'.",
        data={
            "artifacts": {
                "message_ref:s2": {
                    "alias": "gmail",
                    "message_id": "msg-1",
                    "thread_id": "thread-1",
                }
            },
            "message": {
                "id": "msg-1",
                "subject": "Follow Up",
                "text_body": "hello" * 100,
            },
            "missing_requirements": [],
            "suggested_next_actions": ["draft_reply", "get_thread"],
        },
    )

    observation = Observation.from_tool_result(result)

    assert observation.ok is True
    assert observation.summary == "Selected email 'Follow Up'."
    assert observation.artifact_types == ["message_ref"]
    assert observation.artifacts["message_ref:s2"]["message_id"] == "msg-1"
    assert observation.data_preview["message"]["id"] == "msg-1"
    assert observation.suggested_next_actions == ["draft_reply", "get_thread"]


def test_observation_from_tool_result_truncates_large_preview_values() -> None:
    result = ToolResult(
        ok=False,
        summary="query failed",
        data={
            "error": "x" * 300,
            "artifacts": {},
        },
    )

    observation = Observation.from_tool_result(result)

    assert observation.ok is False
    assert observation.data_preview["error"].endswith("...")
    assert len(observation.data_preview["error"]) == 223


def test_observation_from_tool_result_uses_controller_contract_preview_rules() -> None:
    result = ToolResult(
        ok=True,
        summary="Read source file.",
        data={
            "content": "x" * 5000,
            "line_count": 200,
            "artifacts": {},
        },
    )

    observation = Observation.from_tool_result(
        result,
        controller_contract={
            "result_fields": ["content", "line_count"],
            "result_previews": {"content": 4000},
        },
    )

    assert observation.data_preview["line_count"] == 200
    assert len(observation.data_preview["content"]) == 4003


def test_decision_from_dict_validates_tool_call_shape() -> None:
    decision = Decision.from_dict(
        {
            "type": "tool_call",
            "tool": "email",
            "action": "search_messages",
            "params": {"query": "from:abigail"},
            "reason": "Need to search first.",
        }
    )

    assert decision.type is DecisionType.TOOL_CALL
    assert decision.tool == "email"
    assert decision.action == "search_messages"
    assert decision.params == {"query": "from:abigail"}


def test_decision_from_dict_validates_answer_shape() -> None:
    decision = Decision.from_dict(
        {
            "type": "answer",
            "answer": "The last email was on April 24.",
            "reason": "Latest matching message confirms the date.",
        }
    )

    assert decision.type is DecisionType.ANSWER
    assert decision.answer == "The last email was on April 24."
    assert decision.tool == ""
    assert decision.action == ""
    assert decision.params == {}


def test_decision_from_dict_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="tool_call decisions require tool, action, and params"):
        Decision.from_dict({"type": "tool_call", "tool": "email", "action": "", "params": {}})

    with pytest.raises(ValueError, match="answer decisions require answer text"):
        Decision.from_dict({"type": "answer", "answer": ""})


def test_runtime_state_append_updates_indexes() -> None:
    state = RuntimeState(request="draft a reply to Abigail")
    observation = Observation(
        ok=True,
        summary="Selected email 'Follow Up'.",
        artifacts={"message_ref:s2": {"message_id": "msg-1", "thread_id": "thread-1"}},
        artifact_types=["message_ref"],
        data_preview={"message": {"id": "msg-1"}},
    )

    state.append(observation)

    assert state.step_count == 1
    assert state.last_observation is observation
    assert state.artifacts_by_type["message_ref"]["message_id"] == "msg-1"
    assert state.artifacts_by_id["message_ref:s2"]["thread_id"] == "thread-1"
    assert state.pending_confirmation is False
