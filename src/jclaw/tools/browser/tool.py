from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from typing import Any, Callable

from jclaw.core.defaults import (
    BROWSER_CHANNEL,
    BROWSER_MAX_OBJECTIVE_STEPS,
    BROWSER_MAX_RESEARCH_SOURCES,
    BROWSER_SLOW_MO_MS,
    BROWSER_VIEWPORT_HEIGHT,
    BROWSER_VIEWPORT_WIDTH,
)
from jclaw.tools.base import ToolContext, ToolResult
from jclaw.tools.browser.artifacts import BrowserArtifactStore
from jclaw.tools.browser.desktop_driver import DesktopBrowserDriver
from jclaw.tools.browser.models import BrowserReasoner, Target
from jclaw.tools.browser.playwright_driver import PlaywrightBrowserDriver
from jclaw.tools.browser.session import BrowserSessionStore


def _parse_json_object(text: str) -> dict[str, Any] | None:
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


class BrowserTool:
    name = "browser"
    MAX_REPEATED_DECISIONS = 2
    MAX_REPEATED_OBSERVATIONS = 2

    def __init__(
        self,
        base_dir: str | Path,
        options: dict[str, Any] | None = None,
        llm_chat: Callable[[list[dict[str, str]]], str] | None = None,
        reasoner: BrowserReasoner | None = None,
    ) -> None:
        root = Path(base_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.sessions = BrowserSessionStore(root / "sessions")
        self.artifacts = BrowserArtifactStore(root / "artifacts")
        self.playwright = PlaywrightBrowserDriver(
            self.sessions,
            channel=str(options.get("channel", BROWSER_CHANNEL)) if options else BROWSER_CHANNEL,
            headless=bool(options.get("headless", False)) if options else False,
            slow_mo_ms=int(options.get("slow_mo_ms", BROWSER_SLOW_MO_MS)) if options else BROWSER_SLOW_MO_MS,
            viewport_width=int(options.get("viewport_width", BROWSER_VIEWPORT_WIDTH)) if options else BROWSER_VIEWPORT_WIDTH,
            viewport_height=int(options.get("viewport_height", BROWSER_VIEWPORT_HEIGHT)) if options else BROWSER_VIEWPORT_HEIGHT,
        )
        self.desktop = DesktopBrowserDriver()
        self._chat_sessions: dict[str, str] = {}
        self._reasoner = reasoner or (LLMBrowserReasoner(llm_chat) if llm_chat is not None else None)
        self.max_objective_steps = (
            int(options.get("max_objective_steps", BROWSER_MAX_OBJECTIVE_STEPS))
            if options
            else BROWSER_MAX_OBJECTIVE_STEPS
        )
        self.max_research_sources = (
            int(options.get("max_research_sources", BROWSER_MAX_RESEARCH_SOURCES))
            if options
            else BROWSER_MAX_RESEARCH_SOURCES
        )

    def close(self) -> None:
        self.playwright.close()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "Browse the web, inspect current pages, interact with websites, and run bounded multi-step browser objectives.",
            "actions": {
                "open_url": {
                    "description": "Open a specific URL in a browser session.",
                    "use_when": ["the user explicitly wants a page opened or navigated to"],
                },
                "search_web": {
                    "description": "Run a web search query through the browser tool.",
                    "use_when": ["the user wants current web results or news for a query"],
                },
                "read_page": {
                    "description": "Read the current page state from the active session.",
                    "use_when": ["the user wants to inspect or summarize the page that is already open"],
                },
                "click": {"description": "Click a target on the current page."},
                "type": {"description": "Type into a page input or form control."},
                "scroll": {"description": "Scroll the current page."},
                "wait_for": {"description": "Wait for an element or page condition."},
                "screenshot": {"description": "Capture a screenshot of the current page."},
                "extract": {"description": "Extract structured fields from the current page."},
                "run_objective": {
                    "description": "Execute a bounded multi-step browser objective.",
                    "use_when": ["the task needs several browser actions before answering"],
                },
                "close_session": {"description": "Close an existing browser session."},
                "list_sessions": {"description": "List active browser sessions."},
            },
            "read_only_by_default": False,
            "supports_followup": True,
        }

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        if data.get("url"):
            lines.append(f"URL: {data['url']}")
        if data.get("title"):
            lines.append(f"Title: {data['title']}")
        if data.get("text"):
            lines.append(f"Text: {str(data['text'])[:800]}")
        if data.get("sources"):
            lines.append("Sources:")
            for source in data["sources"][:4]:
                title = str(source.get("title", "")).strip() or "Untitled"
                url = str(source.get("url", "")).strip()
                lines.append(f"- {title}: {url}")
        if data.get("termination_reason"):
            lines.append(f"Termination: {data['termination_reason']}")
        if data.get("missing_information"):
            lines.append(f"Missing information: {data['missing_information']}")
        if data.get("evidence_refs"):
            lines.append(f"Evidence refs: {', '.join(data['evidence_refs'][:8])}")
        if data.get("observation_count") is not None:
            lines.append(f"Observations: {data['observation_count']}")
        if data.get("steps"):
            lines.append("Executed steps:")
            for step in data["steps"][:5]:
                lines.append(f"- {step['action']}: {step.get('reason', '')}".strip())
        if "sessions" in data:
            lines.append(f"Sessions: {len(data['sessions'])}")
        return "\n".join(lines)

    def invoke(self, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handlers = {
            "open_url": self._open_url,
            "search_web": self._search_web,
            "read_page": self._read_page,
            "click": self._click,
            "type": self._type,
            "scroll": self._scroll,
            "wait_for": self._wait_for,
            "screenshot": self._screenshot,
            "extract": self._extract,
            "run_objective": self._run_objective,
            "close_session": self._close_session,
            "list_sessions": self._list_sessions,
        }
        try:
            handler = handlers[action]
        except KeyError as exc:
            raise ValueError(f"unsupported browser action: {action}") from exc
        self._trace_event("invoke_start", ctx=ctx, action=action, params=params)
        try:
            result = handler(params, ctx)
        except Exception as exc:  # noqa: BLE001
            self._trace_event("invoke_error", ctx=ctx, action=action, params=params, error=str(exc))
            raise
        self._trace_event(
            "invoke_finish",
            ctx=ctx,
            action=action,
            params=params,
            result={"ok": result.ok, "summary": result.summary, "data": result.data},
        )
        return result

    def _driver(self, params: dict[str, Any]):
        if params.get("allow_desktop_fallback") and params.get("mode") == "desktop":
            return self.desktop
        return self.playwright

    def _ensure_session(self, params: dict[str, Any], ctx: ToolContext):
        session_id = params.get("session_id")
        if session_id:
            session = self.sessions.get_session(str(session_id))
            self._chat_sessions[ctx.chat_id] = session.session_id
            return session
        mapped_session_id = self._chat_sessions.get(ctx.chat_id)
        if mapped_session_id:
            try:
                return self.sessions.get_session(mapped_session_id)
            except KeyError:
                self._chat_sessions.pop(ctx.chat_id, None)
        session = self.sessions.create_session(
            visible=bool(params.get("visible", True)),
            persistent=bool(params.get("persistent", True)),
        )
        self._chat_sessions[ctx.chat_id] = session.session_id
        return session

    def _should_auto_close_session(self, params: dict[str, Any]) -> bool:
        if bool(params.get("keep_session", False)):
            return False
        if params.get("session_id"):
            return False
        return True

    def _cleanup_session_if_needed(self, session_id: str, *, ctx: ToolContext, action: str, params: dict[str, Any]) -> None:
        if not self._should_auto_close_session(params):
            return
        if ctx.metadata.get("loop_managed"):
            return
        try:
            self._close_session({"session_id": session_id}, ctx)
        except Exception as exc:  # noqa: BLE001
            self._trace_event(
                "session_cleanup_error",
                ctx=ctx,
                action=action,
                params={"session_id": session_id, **params},
                error=str(exc),
            )

    def _target(self, params: dict[str, Any]) -> Target:
        raw = params.get("target", {})
        if isinstance(raw, Target):
            return raw
        return Target(
            selector=raw.get("selector"),
            role=raw.get("role"),
            name=raw.get("name"),
            text=raw.get("text"),
            xpath=raw.get("xpath"),
        )

    def _open_url(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        url = str(params.get("url", "about:blank"))
        data = driver.open_url(session.session_id, url, new_tab=bool(params.get("new_tab", False)))
        session.current_url = url
        return ToolResult(ok=True, summary=f"Opened {url}", data=self._with_loop_state(session.session_id, params, data))

    def _search_web(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(params.get("query", "")).strip()
        if not query:
            return ToolResult(
                ok=False,
                summary="Search query is empty.",
                data={"query": query, "results": [], "implemented": True},
            )
        objective_params = {
            "objective": query,
            "start_url": f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            "max_steps": params.get("max_steps", 3),
            "visible": params.get("visible", True),
            "allow_desktop_fallback": params.get("allow_desktop_fallback", True),
            "keep_session": params.get("keep_session", False),
        }
        if params.get("session_id"):
            objective_params["session_id"] = params["session_id"]
        return self._run_objective(objective_params, ctx)

    def _read_page(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.read_page(session.session_id)
        return ToolResult(ok=True, summary="Read current page state.", data=self._with_loop_state(session.session_id, params, data))

    def _click(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.click(session.session_id, self._target(params))
        return ToolResult(ok=True, summary="Clicked target.", data=self._with_loop_state(session.session_id, params, data))

    def _type(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.type(
            session.session_id,
            self._target(params),
            str(params.get("text", "")),
            submit=bool(params.get("submit", False)),
        )
        return ToolResult(ok=True, summary="Typed into target.", data=self._with_loop_state(session.session_id, params, data))

    def _scroll(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.scroll(
            session.session_id,
            direction=str(params.get("direction", "down")),
            amount=int(params.get("amount", 600)),
        )
        return ToolResult(ok=True, summary="Scrolled page.", data=self._with_loop_state(session.session_id, params, data))

    def _wait_for(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        raw_target = params.get("target")
        target = self._target(params) if raw_target else None
        data = driver.wait_for(session.session_id, target, timeout_ms=int(params.get("timeout_ms", 5000)))
        return ToolResult(ok=True, summary="Waited for page state.", data=self._with_loop_state(session.session_id, params, data))

    def _screenshot(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        artifact = self.artifacts.create_artifact(session_id=session.session_id, kind="screenshot", suffix="png")
        if driver is self.playwright:
            shot = driver.screenshot(
                session.session_id,
                full_page=bool(params.get("full_page", False)),
                path=artifact.path,
            )
        else:
            shot = driver.screenshot(session.session_id, full_page=bool(params.get("full_page", False)))
        data = {"artifact_id": artifact.artifact_id, "path": artifact.path, **shot}
        return ToolResult(ok=True, summary="Created screenshot artifact.", data=self._with_loop_state(session.session_id, params, data))

    def _extract(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        page_result = self._read_page({"session_id": session.session_id}, ctx)
        page_data = page_result.data
        fields = params.get("schema") or params.get("fields") or {}
        if not isinstance(fields, dict) or not fields:
            result = ToolResult(
                ok=False,
                summary="No extraction fields were provided.",
                data=self._with_loop_state(session.session_id, params, {"fields": {}, **page_data}),
            )
            self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="extract", params=params)
            return result
        extracted = self._extract_fields_via_reasoner(page_data, fields)
        if not extracted:
            result = ToolResult(
                ok=False,
                summary="Browser extraction is unavailable for the current page.",
                data=self._with_loop_state(session.session_id, params, {"fields": {}, **page_data}),
            )
            self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="extract", params=params)
            return result
        result = ToolResult(
            ok=True,
            summary=f"Extracted {len(extracted.get('fields', {}))} field(s) from the current page.",
            data=self._with_loop_state(session.session_id, params, {
                "fields": extracted.get("fields", {}),
                "evidence_refs": extracted.get("evidence_refs", []),
                "missing_information": extracted.get("missing_information", ""),
                **page_data,
            }),
        )
        self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="extract", params=params)
        return result

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
            data=self._with_loop_state(session.session_id, params, {
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
            }),
        )
        self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="run_objective", params=params)
        return result

    def _close_session(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session_id = str(params["session_id"])
        close_session = getattr(self.playwright, "close_session", None)
        if callable(close_session):
            close_session(session_id)
        self.sessions.close_session(session_id)
        for chat_id, mapped_session_id in list(self._chat_sessions.items()):
            if mapped_session_id == session_id:
                self._chat_sessions.pop(chat_id, None)
        return ToolResult(ok=True, summary=f"Closed session {session_id}.", data={"session_id": session_id})

    def _list_sessions(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        sessions = self.sessions.list_sessions()
        return ToolResult(
            ok=True,
            summary=f"Listed {len(sessions)} browser sessions.",
            data={"sessions": [session.as_dict() for session in sessions]},
        )

    def _trace_event(
        self,
        event: str,
        *,
        ctx: ToolContext,
        action: str,
        params: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "chat_id": ctx.chat_id,
            "user_id": ctx.user_id,
            "action": action,
            "params": params,
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _choose_follow_up_url(self, objective: str, page_data: dict[str, Any]) -> str | None:
        ranked_urls = self._rank_follow_up_urls(objective, page_data)
        return ranked_urls[0] if ranked_urls else None

    def _rank_follow_up_urls(self, objective: str, page_data: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        llm_choice = self._choose_follow_up_url_via_llm(objective, page_data)
        if llm_choice:
            urls.append(self._normalize_url(llm_choice))
        fallback = self._pick_follow_up_urls(objective, page_data)
        for item in fallback:
            if item not in urls:
                urls.append(item)
        return urls

    def _choose_follow_up_url_via_llm(self, objective: str, page_data: dict[str, Any]) -> str | None:
        if self._reasoner is None:
            return None
        try:
            return self._reasoner.choose_link(objective, page_data)
        except Exception:  # noqa: BLE001
            return None

    def _pick_follow_up_url(self, objective: str, page_data: dict[str, Any]) -> str | None:
        ranked = self._pick_follow_up_urls(objective, page_data)
        return ranked[0] if ranked else None

    def _pick_follow_up_urls(self, objective: str, page_data: dict[str, Any]) -> list[str]:
        current_url = str(page_data.get("url", ""))
        candidates = self._extract_candidate_elements(page_data)
        if not candidates:
            return []

        objective_terms = {term for term in re.findall(r"[a-z0-9]+", objective.lower()) if len(term) > 2}
        scored_links: list[tuple[int, str, str]] = []
        for item in candidates:
            href = self._normalize_url(str(item.get("href", "")).strip())
            text = str(item.get("text", "")).lower()
            if not href.startswith("http"):
                continue
            if self._is_junk_link(href, text, current_url):
                continue
            haystack = f"{text} {href}".lower()
            score = sum(1 for term in objective_terms if term in haystack)
            if str(item.get("area", "")) == "main":
                score += 2
            if bool(item.get("clickable", False)):
                score += 1
            if self._looks_like_article_or_result(href, text):
                score += 2
            if "news" in haystack or "blog" in haystack or "article" in haystack:
                score += 1
            if score:
                scored_links.append((score, href, text))
        scored_links.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored_links]

    def _decide_next_action(
        self,
        objective: str,
        page_data: dict[str, Any],
        sources: list[dict[str, str]],
        observations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        observations = observations or []
        llm_decision = self._decide_next_action_via_llm(objective, page_data, sources, observations)
        if llm_decision:
            return llm_decision
        next_url = self._choose_follow_up_url(objective, page_data)
        if next_url:
            return {
                "status": "follow",
                "url": next_url,
                "reason": "Fallback selected the most relevant candidate URL.",
                "evidence_refs": [],
                "missing_information": "Need additional evidence from a followed page.",
            }
        if sources:
            latest_observation = observations[-1] if observations else {}
            fallback_refs = self._observation_ref_ids(latest_observation)[:2]
            return {
                "status": "complete",
                "url": None,
                "reason": "Fallback found enough source material to stop.",
                "evidence_refs": fallback_refs,
                "missing_information": "",
            }
        return {
            "status": "stop",
            "url": None,
            "reason": "Fallback found no meaningful next action.",
            "evidence_refs": [],
            "missing_information": "No grounded source material was gathered.",
        }

    def _decide_next_action_via_llm(
        self,
        objective: str,
        page_data: dict[str, Any],
        sources: list[dict[str, str]],
        observations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if self._reasoner is None:
            return None
        try:
            return self._reasoner.decide_next_action(objective, page_data, sources, observations or [])
        except Exception:  # noqa: BLE001
            return None

    def _extract_fields_via_reasoner(
        self,
        page_data: dict[str, Any],
        fields: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._reasoner is None:
            return None
        try:
            return self._reasoner.extract_fields(page_data, fields)
        except Exception:  # noqa: BLE001
            return None

    def _build_observation(self, page_data: dict[str, Any], *, index: int) -> dict[str, Any]:
        elements = page_data.get("elements", [])
        interactive_elements: list[dict[str, Any]] = []
        if isinstance(elements, list):
            for item in elements[:20]:
                if not isinstance(item, dict):
                    continue
                interactive_elements.append(
                    {
                        "id": str(item.get("id", "")),
                        "role": str(item.get("role", "")),
                        "text": str(item.get("text", ""))[:180],
                        "href": str(item.get("href", "")),
                    }
                )
        content_blocks = page_data.get("content_blocks", [])
        normalized_blocks: list[dict[str, Any]] = []
        if isinstance(content_blocks, list):
            for item in content_blocks[:20]:
                if not isinstance(item, dict):
                    continue
                normalized_blocks.append(
                    {
                        "id": str(item.get("id", "")),
                        "text": str(item.get("text", ""))[:240],
                        "tag": str(item.get("tag", "")),
                    }
                )
        return {
            "id": f"obs_{index}",
            "url": self._normalize_url(str(page_data.get("url", ""))),
            "title": str(page_data.get("title", ""))[:200],
            "page_kind": str(page_data.get("page_kind", "")),
            "text_preview": str(page_data.get("text", ""))[:1200],
            "text_fingerprint": str(page_data.get("text_fingerprint", "")),
            "content_blocks": normalized_blocks,
            "interactive_elements": interactive_elements,
        }

    def _compute_novelty(self, observation: dict[str, Any], prior_observations: list[dict[str, Any]]) -> dict[str, Any]:
        if not prior_observations:
            return {
                "is_new_url": True,
                "is_new_fingerprint": True,
                "new_ref_count": len(self._observation_ref_ids(observation)),
                "score": 1.0,
            }
        prior_urls = {str(item.get("url", "")) for item in prior_observations}
        prior_fingerprints = {str(item.get("text_fingerprint", "")) for item in prior_observations if item.get("text_fingerprint")}
        prior_refs = {
            ref
            for item in prior_observations
            for ref in self._observation_ref_ids(item)
        }
        current_refs = self._observation_ref_ids(observation)
        new_refs = [ref for ref in current_refs if ref not in prior_refs]
        is_new_url = observation.get("url", "") not in prior_urls
        fingerprint = str(observation.get("text_fingerprint", ""))
        is_new_fingerprint = bool(fingerprint) and fingerprint not in prior_fingerprints
        score = 0.0
        if is_new_url:
            score += 0.45 if is_new_fingerprint or new_refs else 0.1
        if is_new_fingerprint:
            score += 0.35
        if new_refs:
            score += min(0.2, len(new_refs) * 0.05)
        return {
            "is_new_url": is_new_url,
            "is_new_fingerprint": is_new_fingerprint,
            "new_ref_count": len(new_refs),
            "score": round(min(score, 1.0), 2),
        }

    def _compact_observation_for_trace(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": observation.get("id", ""),
            "url": observation.get("url", ""),
            "title": observation.get("title", ""),
            "text_fingerprint": observation.get("text_fingerprint", ""),
            "content_block_count": len(observation.get("content_blocks", [])),
            "interactive_count": len(observation.get("interactive_elements", [])),
        }

    def _observation_ref_ids(self, observation: dict[str, Any]) -> list[str]:
        refs: list[str] = []
        for item in observation.get("content_blocks", []):
            if isinstance(item, dict) and item.get("id"):
                refs.append(str(item["id"]))
        for item in observation.get("interactive_elements", []):
            if isinstance(item, dict) and item.get("id"):
                refs.append(str(item["id"]))
        return refs

    def _validate_evidence_refs(self, refs: list[Any], observations: list[dict[str, Any]]) -> list[str]:
        valid_refs = {
            ref
            for observation in observations
            for ref in self._observation_ref_ids(observation)
        }
        normalized: list[str] = []
        for item in refs[:8]:
            ref = str(item).strip()
            if ref and ref in valid_refs and ref not in normalized:
                normalized.append(ref)
        return normalized

    def _decision_signature(self, decision: dict[str, Any]) -> str:
        return "|".join(
            [
                str(decision.get("status", "")),
                self._normalize_url(str(decision.get("url", ""))),
                str(decision.get("chosen_element_id", "")),
            ]
        )

    def _observation_signature(self, observation: dict[str, Any]) -> str:
        return "|".join(
            [
                str(observation.get("url", "")),
                str(observation.get("text_fingerprint", "")),
            ]
        )

    def _observation_adds_source(self, observation: dict[str, Any]) -> bool:
        if observation.get("page_kind") == "search_results":
            return False
        return bool(observation.get("text_preview", "").strip() or observation.get("content_blocks"))

    def _with_loop_state(self, session_id: str, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(data)
        enriched["followup_params"] = {"session_id": session_id}
        if not params.get("session_id") and not bool(params.get("keep_session", False)):
            enriched["loop_cleanup"] = {"action": "close_session", "params": {"session_id": session_id}}
        return enriched

    def _extract_candidate_elements(self, page_data: dict[str, Any]) -> list[dict[str, Any]]:
        elements = page_data.get("elements", [])
        if isinstance(elements, list) and elements:
            candidates = []
            for item in elements:
                if not isinstance(item, dict):
                    continue
                if item.get("role") != "link":
                    continue
                href = self._normalize_url(str(item.get("href", "")).strip())
                if not href:
                    continue
                normalized = dict(item)
                normalized["href"] = href
                candidates.append(normalized)
            if candidates:
                return candidates
        links = page_data.get("links", [])
        if isinstance(links, list):
            normalized_links = []
            for item in links:
                if not isinstance(item, dict):
                    continue
                href = self._normalize_url(str(item.get("href", "")).strip())
                if not href:
                    continue
                normalized = dict(item)
                normalized["href"] = href
                normalized_links.append(normalized)
            return normalized_links
        return []

    def _normalize_url(self, href: str) -> str:
        value = href.strip()
        if not value:
            return ""
        if value.startswith("//"):
            value = f"https:{value}"
        parsed = urlparse(value)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return unquote(target).strip()
        return value

    def _is_junk_link(self, href: str, text: str, current_url: str) -> bool:
        lowered_href = href.lower()
        lowered_text = text.lower()

        blocked_domains = (
            "duckduckgo.com",
            "google.com",
            "bing.com",
            "search.yahoo.com",
            "apps.apple.com",
            "itunes.apple.com",
            "play.google.com",
        )
        if any(domain in lowered_href for domain in blocked_domains):
            return True

        blocked_tokens = (
            "privacy",
            "terms",
            "settings",
            "login",
            "sign in",
            "sign-in",
            "signup",
            "sign up",
            "advertis",
            "sponsored",
            "ad choice",
            "support",
            "help",
            "install",
            "download app",
            "duckduckgo browser",
            "duck ai",
            "vpn",
        )
        haystack = f"{lowered_text} {lowered_href}"
        if any(token in haystack for token in blocked_tokens):
            return True

        if current_url and lowered_href.rstrip("/") == current_url.lower().rstrip("/"):
            return True

        return False

    def _looks_like_article_or_result(self, href: str, text: str) -> bool:
        lowered_href = href.lower()
        lowered_text = text.lower()
        positive_tokens = (
            "/news",
            "/article",
            "/blog",
            "/posts",
            "/story",
            "news",
            "article",
            "announces",
            "launches",
            "update",
            "report",
        )
        haystack = f"{lowered_text} {lowered_href}"
        return any(token in haystack for token in positive_tokens)


class LLMBrowserReasoner:
    def __init__(self, llm_chat: Callable[[list[dict[str, str]]], str]) -> None:
        self._llm_chat = llm_chat

    def choose_link(self, objective: str, page_data: dict[str, Any]) -> str | None:
        elements = page_data.get("elements", [])
        if not isinstance(elements, list) or not elements:
            return None
        compact_elements: list[dict[str, Any]] = []
        for item in elements[:20]:
            if not isinstance(item, dict):
                continue
            href = str(item.get("href", "")).strip()
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            if role != "link":
                continue
            if not href.startswith("http"):
                continue
            compact_elements.append(
                {
                    "id": str(item.get("id", "")),
                    "role": role,
                    "text": text[:200],
                    "href": href,
                    "area": str(item.get("area", "")),
                    "clickable": bool(item.get("clickable", False)),
                    "score_hint": item.get("score_hint", 0),
                }
            )
        if not compact_elements:
            return None
        chooser_prompt = (
            "You are helping JClaw choose the next browser link to follow.\n"
            "You are given a user objective and a compact inspected-elements snapshot from the current page.\n"
            "Choose the single best link element that most likely advances the user's objective.\n"
            "Avoid search-engine homepages, settings, privacy/help pages, app-store links, download/install pages, and obvious ads.\n"
            "Prefer result/article/documentation/news pages that directly match the user's request.\n"
            "Return strict JSON only with schema:\n"
            '{"chosen_element_id": string | null, "reason": string}\n'
            "Use null if none of the elements look useful."
        )
        chooser_payload = {
            "objective": objective,
            "page_url": page_data.get("url", ""),
            "page_title": page_data.get("title", ""),
            "page_kind": page_data.get("page_kind", ""),
            "page_text_preview": str(page_data.get("text", ""))[:800],
            "elements": compact_elements,
        }
        raw = self._llm_chat(
            [
                {"role": "system", "content": chooser_prompt},
                {"role": "user", "content": json.dumps(chooser_payload, ensure_ascii=True)},
            ]
        )
        parsed = _parse_json_object(raw)
        if not parsed:
            return None
        chosen_element_id = parsed.get("chosen_element_id")
        if chosen_element_id in (None, "", "null"):
            return None
        chosen_id = str(chosen_element_id).strip()
        for item in compact_elements:
            if item.get("id") == chosen_id:
                href = str(item.get("href", "")).strip()
                return href if href.startswith("http") else None
        return None

    def extract_fields(
        self,
        page_data: dict[str, Any],
        fields: dict[str, Any],
    ) -> dict[str, Any] | None:
        extractor_prompt = (
            "You are JClaw's browser extraction helper.\n"
            "Extract only the requested fields from the current page observation.\n"
            "Use only the provided page text, content blocks, and interactive elements.\n"
            "Do not invent missing values. Use empty string or empty list when a field is not supported by the evidence.\n"
            "Return strict JSON only with schema:\n"
            '{"fields":object,"evidence_refs":[string],"missing_information":string}\n'
            "The fields object must contain exactly the same top-level keys as the requested fields object.\n"
            "evidence_refs must reference ids from content_blocks or interactive_elements.\n"
        )
        payload = {
            "page": {
                "url": page_data.get("url", ""),
                "title": page_data.get("title", ""),
                "text_preview": str(page_data.get("text", ""))[:1800],
                "content_blocks": page_data.get("content_blocks", [])[:15],
                "interactive_elements": page_data.get("elements", [])[:15],
            },
            "requested_fields": fields,
        }
        raw = self._llm_chat(
            [
                {"role": "system", "content": extractor_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
        )
        parsed = _parse_json_object(raw)
        if not parsed:
            return None
        extracted_fields = parsed.get("fields")
        if not isinstance(extracted_fields, dict):
            return None
        normalized_fields = {key: extracted_fields.get(key, "" if not isinstance(spec, list) else []) for key, spec in fields.items()}
        return {
            "fields": normalized_fields,
            "evidence_refs": parsed.get("evidence_refs", []),
            "missing_information": str(parsed.get("missing_information", "")),
        }

    def decide_next_action(
        self,
        objective: str,
        page_data: dict[str, Any],
        sources: list[dict[str, str]],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        elements = page_data.get("elements", [])
        if not isinstance(elements, list):
            return None
        compact_elements: list[dict[str, Any]] = []
        for item in elements[:20]:
            if not isinstance(item, dict):
                continue
            href = str(item.get("href", "")).strip()
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            compact_elements.append(
                {
                    "id": str(item.get("id", "")),
                    "role": role,
                    "text": text[:180],
                    "href": href,
                    "area": str(item.get("area", "")),
                    "clickable": bool(item.get("clickable", False)),
                    "score_hint": item.get("score_hint", 0),
                }
            )
        controller_prompt = (
            "You are JClaw's browser controller.\n"
            "Decide whether the current browser mission is complete, should continue by following one visible link, or should stop because no meaningful progress is likely.\n"
            "Use only the provided observations and gathered sources.\n"
            "Base the decision on whether the current observations contain enough evidence for the objective.\n"
            "Do not complete unless you can cite concrete observation references.\n"
            "Prefer follow when a visible link is likely to materially improve the evidence.\n"
            "Prefer stop when further browsing is unlikely to add useful information.\n"
            "Return strict JSON only with schema:\n"
            '{"status":"follow|complete|stop","chosen_element_id":string|null,"reason":string,"evidence_refs":[string],"missing_information":string}\n'
            "evidence_refs must reference ids from content_blocks or interactive_elements in the current or prior observations.\n"
            "If status is follow, chosen_element_id must identify one visible link element from the snapshot.\n"
            "If status is complete or stop, chosen_element_id must be null."
        )
        payload = {
            "objective": objective,
            "current_page": {
                "url": page_data.get("url", ""),
                "title": page_data.get("title", ""),
                "text_preview": str(page_data.get("text", ""))[:1200],
                "content_blocks": page_data.get("content_blocks", [])[:12],
            },
            "sources": sources[-3:],
            "prior_observations": observations[-3:],
            "elements": compact_elements,
        }
        raw = self._llm_chat(
            [
                {"role": "system", "content": controller_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
        )
        parsed = _parse_json_object(raw)
        if not parsed:
            return None
        status = str(parsed.get("status", "")).strip().lower()
        if status not in {"follow", "complete", "stop"}:
            return None
        chosen_element_id = parsed.get("chosen_element_id")
        if status != "follow":
            return {
                "status": status,
                "url": None,
                "reason": str(parsed.get("reason", "")),
                "evidence_refs": parsed.get("evidence_refs", []),
                "missing_information": str(parsed.get("missing_information", "")),
            }
        chosen_id = None if chosen_element_id in (None, "", "null") else str(chosen_element_id).strip()
        if not chosen_id:
            return None
        for item in compact_elements:
            if item.get("id") == chosen_id and str(item.get("href", "")).startswith("http"):
                return {
                    "status": "follow",
                    "url": str(item["href"]),
                    "reason": str(parsed.get("reason", "")),
                    "evidence_refs": parsed.get("evidence_refs", []),
                    "missing_information": str(parsed.get("missing_information", "")),
                    "chosen_element_id": chosen_id,
                }
        return None
