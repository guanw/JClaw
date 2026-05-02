from __future__ import annotations

import json
import logging
import re
from typing import Any

from jclaw.tools.base import Decision, DecisionType, RuntimeState, ToolResult


LOGGER = logging.getLogger(__name__)
MAX_CONTROLLER_OBSERVATIONS = 5


class AgentControllerMixin:
    def _decide_next_tool_step(
        self,
        chat_id: str,
        text: str,
        *,
        user_name: str,
        steps: list[dict[str, Any]],
        runtime: RuntimeState,
    ) -> Decision | None:
        available_tools = self.tools.list_tools()
        if not available_tools:
            return None

        controller_state = self._controller_state_for_prompt(steps, runtime)
        recent_history = [
            {"role": item.role, "content": item.content}
            for item in self.messages.recent(chat_id, 4)
        ]
        prompt = (
            "You are JClaw's live tool controller.\n"
            "Given the user request and prior tool observations, choose exactly one next decision.\n"
            "Use tool_call when one tool step materially advances the request.\n"
            "Use answer when the request can now be answered directly from the observations.\n"
            "Use blocked when progress is unsafe or impossible without clarification, permission, or missing prerequisites.\n"
            "Use complete when the operational task is finished and the latest tool result should be returned to the user.\n"
            "The runtime state includes normalized observations and the current artifact frontier. Treat the latest observation as authoritative.\n"
            "The runtime state also includes the authoritative current local date, time, and timezone. Use that instead of guessing today's date or year.\n"
            "When a tool offers both a whole-resource read and a focused range read, prefer the focused range read whenever the user asks for explicit line numbers, a line range, or another clearly bounded subsection.\n"
            "Keep params minimal and choose only one next decision.\n"
            "Return strict JSON only.\n"
            "Schema:\n"
            '{"type":"tool_call|answer|blocked|complete","tool":string,"action":string,"params":object,"answer":string,"reason":string}\n'
            "For answer, provide answer and leave tool/action empty.\n"
            "For blocked or complete, leave tool/action empty and params {}.\n"
            f"Tool catalog: {self._tool_catalog_for_prompt(available_tools)}"
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
                            "controller_state": controller_state,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
        )
        if controller_state["observations"]:
            LOGGER.info("tool continuation raw response: %s", raw)
        else:
            LOGGER.info("tool initial controller raw response: %s", raw)
        parsed = self._parse_json_object(raw)
        if not parsed:
            parsed = self._repair_controller_response(
                raw,
                text=text,
                controller_state=controller_state,
            )
            if not parsed:
                return None
        try:
            decision = Decision.from_dict(parsed)
        except ValueError:
            return None
        if controller_state["observations"]:
            LOGGER.info(
                "tool continuation selected type=%s tool=%s action=%s reason=%s",
                decision.type.value,
                decision.tool,
                decision.action,
                decision.reason,
            )
        else:
            LOGGER.info(
                "tool initial controller selected type=%s tool=%s action=%s reason=%s",
                decision.type.value,
                decision.tool,
                decision.action,
                decision.reason,
            )
        return decision

    def _controller_state_for_prompt(
        self,
        steps: list[dict[str, Any]],
        runtime: RuntimeState,
    ) -> dict[str, Any]:
        now = self._controller_now()
        artifact_preview_contracts = self._artifact_preview_contracts()
        observations: list[dict[str, Any]] = []
        start_index = max(0, len(steps) - MAX_CONTROLLER_OBSERVATIONS)
        for index, step in enumerate(steps[start_index:], start=start_index + 1):
            observation = (
                runtime.observations[index - 1].to_dict()
                if index - 1 < len(runtime.observations)
                else self._tool_result_for_controller(step["tool"], step["result"])
            )
            observations.append(
                {
                    "step": index,
                    "tool": step["tool"],
                    "action": step["action"],
                    "reason": step["reason"],
                    "observation": observation,
                }
            )
        return {
            "step_count": runtime.step_count,
            "pending_confirmation": runtime.pending_confirmation,
            "current_local_time": now.isoformat(),
            "current_local_date": now.date().isoformat(),
            "current_local_timezone": str(now.tzinfo or ""),
            "artifact_types": sorted(runtime.artifacts_by_type.keys()),
            "artifacts_by_type": {
                str(key): self._preview_runtime_value(
                    value,
                    artifact_type=str(key),
                    artifact_preview_contracts=artifact_preview_contracts,
                )
                for key, value in runtime.artifacts_by_type.items()
            },
            "latest_observation": runtime.last_observation.to_dict() if runtime.last_observation else {},
            "observations": observations,
        }

    def _preview_runtime_value(
        self,
        value: Any,
        *,
        depth: int = 0,
        artifact_type: str = "",
        artifact_preview_contracts: dict[str, dict[str, int]] | None = None,
    ) -> Any:
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            text = value.strip()
            return f"{text[:220]}..." if len(text) > 220 else text
        if depth >= 2:
            return f"<{type(value).__name__}>"
        if isinstance(value, list):
            return [
                self._preview_runtime_value(
                    item,
                    depth=depth + 1,
                    artifact_preview_contracts=artifact_preview_contracts,
                )
                for item in value[:3]
            ]
        if isinstance(value, dict):
            preview: dict[str, Any] = {}
            preview_limits = (artifact_preview_contracts or {}).get(artifact_type, {})
            for index, (key, item) in enumerate(value.items()):
                if index >= 8:
                    preview["__truncated__"] = True
                    break
                limit = preview_limits.get(str(key))
                if isinstance(item, str) and isinstance(limit, int) and limit > 0:
                    text = item.strip()
                    preview[str(key)] = f"{text[:limit]}..." if len(text) > limit else text
                    continue
                preview[str(key)] = self._preview_runtime_value(
                    item,
                    depth=depth + 1,
                    artifact_preview_contracts=artifact_preview_contracts,
                )
            return preview
        return str(value)

    def _tool_result_for_controller(self, tool_name: str, result: ToolResult) -> dict[str, Any]:
        data = result.data
        controller_data: dict[str, Any] = {
            "summary": result.summary,
            "needs_confirmation": result.needs_confirmation,
        }
        contract = self._tool_controller_contract(tool_name)
        for key in contract.get("result_fields", []):
            if key in data:
                controller_data[key] = data[key]
        for key, limit in contract.get("list_fields", {}).items():
            if key in data and isinstance(data[key], list):
                controller_data[str(key)] = data[key][: int(limit)]
        return controller_data

    def _tool_catalog_for_prompt(self, available_tools: list[dict[str, Any]]) -> str:
        catalog: list[dict[str, Any]] = []
        for tool in available_tools:
            entry: dict[str, Any] = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
            if "actions" in tool:
                entry["actions"] = tool["actions"]
            for key, value in tool.items():
                if key in {"name", "description", "actions", "controller_contract"}:
                    continue
                entry[key] = value
            catalog.append(entry)
        return json.dumps(catalog, ensure_ascii=True)

    def _tool_controller_contract(self, tool_name: str) -> dict[str, Any]:
        description = self.tools.get(tool_name).describe()
        contract = description.get("controller_contract", {})
        return dict(contract) if isinstance(contract, dict) else {}

    def _artifact_preview_contracts(self) -> dict[str, dict[str, int]]:
        contracts: dict[str, dict[str, int]] = {}
        for tool in self.tools.list_tools():
            controller_contract = tool.get("controller_contract", {})
            if not isinstance(controller_contract, dict):
                continue
            artifact_previews = controller_contract.get("artifact_previews", {})
            if not isinstance(artifact_previews, dict):
                continue
            for artifact_type, field_limits in artifact_previews.items():
                if not isinstance(field_limits, dict):
                    continue
                contracts[str(artifact_type)] = {
                    str(field): int(limit)
                    for field, limit in field_limits.items()
                    if isinstance(limit, int | float) and int(limit) > 0
                }
        return contracts

    def _repair_controller_response(
        self,
        raw: str,
        *,
        text: str,
        controller_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not str(raw).strip():
            return None
        try:
            repaired = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the prior controller response as strict JSON only.\n"
                            "Allowed schema:\n"
                            '{"type":"tool_call|answer|blocked|complete","tool":string,"action":string,"params":object,"answer":string,"reason":string}\n'
                            "If the response already answers the user from available evidence, use type=answer.\n"
                            "If it requests clarification or says progress is blocked, use type=blocked.\n"
                            "If it says the operation is finished and the latest tool result should be returned, use type=complete.\n"
                            "Otherwise use type=tool_call.\n"
                            "Return strict JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "request": text,
                                "controller_state": controller_state,
                                "raw_controller_response": raw,
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
            )
        except Exception:  # noqa: BLE001
            return None
        LOGGER.info("tool controller repair raw response: %s", repaired)
        return self._parse_json_object(repaired)

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
