from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

from jclaw.tools.base import ToolContext, ToolResult


class BrowserObjectiveLoopMixin:
    def _run_objective(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        objective = str(params.get("objective", "")).strip()
        max_steps = max(1, min(int(params.get("max_steps", self.max_objective_steps)), 8))
        max_sources = max(1, min(int(params.get("max_sources", self.max_research_sources)), 5))
        target_url = str(params.get("start_url", "")).strip()
        if not target_url:
            url_match = re.search(r"https?://\S+", objective)
            if url_match:
                target_url = url_match.group(0)
            elif objective:
                target_url = f"https://html.duckduckgo.com/html/?q={quote_plus(objective)}"
            else:
                target_url = "about:blank"

        executed_steps: list[dict[str, Any]] = []

        open_result = self._open_url({"session_id": session.session_id, "url": target_url}, ctx)
        executed_steps.append(
            {
                "action": "open_url",
                "params": {"url": target_url},
                "reason": "Open the target page or search results.",
                "url": open_result.data.get("url", ""),
                "title": open_result.data.get("title", ""),
            }
        )

        read_result = self._read_page({"session_id": session.session_id}, ctx)
        executed_steps.append(
            {
                "action": "read_page",
                "params": {"session_id": session.session_id},
                "reason": "Capture the current page state.",
                "url": read_result.data.get("url", ""),
                "title": read_result.data.get("title", ""),
            }
        )

        current_read = read_result
        sources: list[dict[str, str]] = []
        visited_urls = {self._normalize_url(str(current_read.data.get("url", "")))}
        recorded_source_urls: set[str] = set()
        observations: list[dict[str, Any]] = []
        novelty_history: list[dict[str, Any]] = []
        repeated_decision_count = 0
        repeated_observation_count = 0
        previous_decision_signature = ""
        previous_observation_signature = ""
        final_decision: dict[str, Any] | None = None

        def record_source(page_data: dict[str, Any]) -> None:
            source_url = self._normalize_url(str(page_data.get("url", "")).strip())
            if not source_url or source_url in recorded_source_urls:
                return
            recorded_source_urls.add(source_url)
            sources.append(
                {
                    "url": source_url,
                    "title": str(page_data.get("title", ""))[:200],
                    "text": str(page_data.get("text", ""))[:700],
                }
            )

        observation = self._build_observation(current_read.data, index=1)
        observations.append(observation)
        novelty = self._compute_novelty(observation, observations[:-1])
        novelty_history.append(novelty)
        previous_observation_signature = self._observation_signature(observation)

        if self._observation_adds_source(observation):
            record_source(current_read.data)

        termination_reason = "step_budget_exhausted"
        for _ in range(max(0, max_steps - 1)):
            decision = self._decide_next_action(objective, current_read.data, sources, observations)
            final_decision = decision
            chosen_url = self._normalize_url(str(decision.get("url", ""))) if decision else ""
            valid_evidence_refs = self._validate_evidence_refs(
                decision.get("evidence_refs", []) if decision else [],
                observations,
            )
            self._trace_event(
                "browser_loop_step",
                ctx=ctx,
                action="run_objective",
                params={
                    "objective": objective,
                    "step_index": len(observations),
                    "current_url": current_read.data.get("url", ""),
                    "action_taken": executed_steps[-1]["action"] if executed_steps else None,
                    "action_params": executed_steps[-1].get("params", {}) if executed_steps else {},
                    "observation": self._compact_observation_for_trace(observation),
                    "novelty": novelty_history[-1],
                    "decision": decision,
                    "valid_evidence_refs": valid_evidence_refs,
                },
            )
            self._trace_event(
                "follow_up_choice",
                ctx=ctx,
                action="run_objective",
                params={
                    "objective": objective,
                    "current_url": current_read.data.get("url", ""),
                    "candidate_elements": current_read.data.get("elements", [])[:12],
                    "decision": decision,
                    "chosen_url": chosen_url or None,
                },
            )
            if not decision:
                termination_reason = "no_decision"
                break
            decision_signature = self._decision_signature(decision)
            if decision_signature == previous_decision_signature:
                repeated_decision_count += 1
            else:
                repeated_decision_count = 0
            previous_decision_signature = decision_signature
            if repeated_decision_count >= self.MAX_REPEATED_DECISIONS:
                termination_reason = "repeated_decision_limit"
                break
            status = str(decision.get("status", "stop"))
            if status == "complete":
                if not valid_evidence_refs:
                    termination_reason = "missing_evidence_refs"
                    break
                termination_reason = "controller_complete"
                break
            if status == "stop":
                termination_reason = "controller_stop"
                break
            if not chosen_url or chosen_url in visited_urls:
                termination_reason = "no_meaningful_next_url"
                break
            visited_urls.add(chosen_url)
            follow_open = self._open_url({"session_id": session.session_id, "url": chosen_url}, ctx)
            executed_steps.append(
                {
                    "action": "open_url",
                    "params": {"url": chosen_url},
                    "reason": str(decision.get("reason", "Follow a likely relevant result.")),
                    "url": follow_open.data.get("url", ""),
                    "title": follow_open.data.get("title", ""),
                }
            )
            current_read = self._read_page({"session_id": session.session_id}, ctx)
            executed_steps.append(
                {
                    "action": "read_page",
                    "params": {"session_id": session.session_id},
                    "reason": "Capture the followed page state.",
                    "url": current_read.data.get("url", ""),
                    "title": current_read.data.get("title", ""),
                }
            )
            observation = self._build_observation(current_read.data, index=len(observations) + 1)
            novelty = self._compute_novelty(observation, observations)
            observations.append(observation)
            novelty_history.append(novelty)
            observation_signature = self._observation_signature(observation)
            if observation_signature == previous_observation_signature:
                repeated_observation_count += 1
            else:
                repeated_observation_count = 0
            previous_observation_signature = observation_signature
            if self._observation_adds_source(observation):
                record_source(current_read.data)
            if len(sources) >= max_sources:
                termination_reason = "source_budget_reached"
                break
            if repeated_observation_count >= self.MAX_REPEATED_OBSERVATIONS:
                termination_reason = "repeated_observation_limit"
                break

        result = ToolResult(
            ok=True,
            summary=(
                f"Executed browser objective and captured {len(sources)} source page"
                f"{'' if len(sources) == 1 else 's'}."
            ),
            data={
                "session_id": session.session_id,
                "objective": objective,
                "steps": executed_steps,
                "sources": sources,
                "implemented": True,
                "research_complete": bool(sources)
                and termination_reason in {"controller_complete", "source_budget_reached"},
                "termination_reason": termination_reason,
                "evidence_refs": self._validate_evidence_refs(
                    final_decision.get("evidence_refs", []) if final_decision else [],
                    observations,
                ),
                "missing_information": str(final_decision.get("missing_information", "")).strip()
                if final_decision
                else "",
                "observation_count": len(observations),
                "novelty_history": novelty_history,
                "observations": [self._compact_observation_for_trace(item) for item in observations],
                **current_read.data,
                "mode": self._driver(params).mode,
                **self._browser_result_payload(
                    current_read.data,
                    include_candidates=True,
                    include_extract={
                        "objective": objective,
                        "termination_reason": termination_reason,
                        "evidence_refs": self._validate_evidence_refs(
                            final_decision.get("evidence_refs", []) if final_decision else [],
                            observations,
                        ),
                        "missing_information": str(final_decision.get("missing_information", "")).strip()
                        if final_decision
                        else "",
                        "sources": sources[:3],
                    },
                ),
            },
        )
        self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="run_objective", params=params)
        return result
