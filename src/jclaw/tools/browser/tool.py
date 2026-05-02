from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

from jclaw.core.defaults import (
    BROWSER_CHANNEL,
    BROWSER_MAX_OBJECTIVE_STEPS,
    BROWSER_MAX_RESEARCH_SOURCES,
    BROWSER_SLOW_MO_MS,
    BROWSER_VIEWPORT_HEIGHT,
    BROWSER_VIEWPORT_WIDTH,
)
from jclaw.tools.base import (
    ActionSpec,
    ToolContext,
    ToolLoopFinalizer,
    ToolLoopState,
    ToolResult,
    append_field,
    append_list_section,
    build_tool_description,
)
from jclaw.tools.browser.artifacts import BrowserArtifactStore
from jclaw.tools.browser.desktop_driver import DesktopBrowserDriver
from jclaw.tools.browser.models import BrowserReasoner, Target
from jclaw.tools.browser.navigation import BrowserNavigationMixin
from jclaw.tools.browser.objective_loop import BrowserObjectiveLoopMixin
from jclaw.tools.browser.observations import BrowserObservationsMixin
from jclaw.tools.browser.playwright_driver import PlaywrightBrowserDriver
from jclaw.tools.browser.reasoning import BrowserReasoningMixin, LLMBrowserReasoner
from jclaw.tools.browser.session import BrowserSessionStore

class BrowserTool(
    BrowserNavigationMixin,
    BrowserObservationsMixin,
    BrowserReasoningMixin,
    BrowserObjectiveLoopMixin,
):
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
        specs = self._action_specs()
        return build_tool_description(
            name=self.name,
            description="Browse the web, inspect current pages, interact with websites, and run bounded multi-step browser objectives.",
            actions=specs,
        )

    def format_result(self, action: str, result: ToolResult) -> str:
        lines = [result.summary]
        data = result.data
        append_field(lines, "URL", data.get("url"))
        append_field(lines, "Title", data.get("title"))
        append_field(lines, "Text", data.get("text"), formatter=lambda value: str(value)[:800])
        append_list_section(
            lines,
            "Sources:",
            data.get("sources"),
            lambda source: f"- {(str(source.get('title', '')).strip() or 'Untitled')}: {str(source.get('url', '')).strip()}",
            limit=4,
        )
        append_field(lines, "Termination", data.get("termination_reason"))
        append_field(lines, "Missing information", data.get("missing_information"))
        append_field(
            lines,
            "Evidence refs",
            data.get("evidence_refs"),
            formatter=lambda refs: ", ".join(str(item) for item in refs[:8]),
        )
        append_field(lines, "Observations", data.get("observation_count"), include_when=lambda value: value is not None)
        append_list_section(
            lines,
            "Executed steps:",
            data.get("steps"),
            lambda step: f"- {step['action']}: {step.get('reason', '')}".strip(),
            limit=5,
        )
        append_field(
            lines,
            "Sessions",
            data.get("sessions"),
            include_when=lambda value: isinstance(value, list),
            formatter=lambda sessions: str(len(sessions)),
        )
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
        return self._with_loop_state(result, action=action, params=params, ctx=ctx)

    def _with_loop_state(self, result: ToolResult, *, action: str, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.execution is None or result.loop_state is not None:
            return result
        session_id = str(result.data.get("session_id", "")).strip()
        if not session_id:
            return result
        if action == "close_session":
            result.loop_state = ToolLoopState(clear=True)
            return result
        auto_close = self._should_auto_close_session(params)
        result.loop_state = ToolLoopState(
            state={"session_id": session_id},
            finalizer=(
                ToolLoopFinalizer(
                    action="close_session",
                    params={"session_id": session_id},
                )
                if auto_close
                else None
            ),
            clear_finalizer=not auto_close,
        )
        return result

    def _driver(self, params: dict[str, Any]):
        if params.get("allow_desktop_fallback") and params.get("mode") == "desktop":
            return self.desktop
        return self.playwright

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

    def _action_specs(self) -> dict[str, ActionSpec]:
        target_schema = {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "role": {"type": "string"},
                "name": {"type": "string"},
                "text": {"type": "string"},
                "xpath": {"type": "string"},
            },
        }
        return {
            "open_url": ActionSpec(
                tool=self.name,
                action="open_url",
                description="Open a specific URL in a browser session.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "new_tab": {"type": "boolean"},
                        "visible": {"type": "boolean"},
                        "persistent": {"type": "boolean"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["url"],
                },
                writes=True,
                produces_artifacts=("browser_page", "browser_candidates"),
            ),
            "search_web": ActionSpec(
                tool=self.name,
                action="search_web",
                description="Run a web search query through the browser tool.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_steps": {"type": "integer"},
                        "visible": {"type": "boolean"},
                        "allow_desktop_fallback": {"type": "boolean"},
                        "keep_session": {"type": "boolean"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["query"],
                },
                writes=True,
                produces_artifacts=("browser_page", "browser_candidates", "browser_extract"),
            ),
            "read_page": ActionSpec(
                tool=self.name,
                action="read_page",
                description="Read the current page state from the active session.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("browser_page", "browser_candidates"),
            ),
            "click": ActionSpec(
                tool=self.name,
                action="click",
                description="Click a target on the current page.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": target_schema,
                        "session_id": {"type": "string"},
                    },
                    "required": ["target"],
                },
                writes=True,
                produces_artifacts=("browser_page", "browser_candidates"),
            ),
            "type": ActionSpec(
                tool=self.name,
                action="type",
                description="Type into a page input or form control.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": target_schema,
                        "text": {"type": "string"},
                        "submit": {"type": "boolean"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["target", "text"],
                },
                writes=True,
                produces_artifacts=("browser_page", "browser_candidates"),
            ),
            "scroll": ActionSpec(
                tool=self.name,
                action="scroll",
                description="Scroll the current page.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string"},
                        "amount": {"type": "integer"},
                        "session_id": {"type": "string"},
                    },
                },
                writes=True,
                produces_artifacts=("browser_page", "browser_candidates"),
            ),
            "wait_for": ActionSpec(
                tool=self.name,
                action="wait_for",
                description="Wait for an element or page condition.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": target_schema,
                        "timeout_ms": {"type": "integer"},
                        "session_id": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("browser_page", "browser_candidates"),
            ),
            "screenshot": ActionSpec(
                tool=self.name,
                action="screenshot",
                description="Capture a screenshot of the current page.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "full_page": {"type": "boolean"},
                        "session_id": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("browser_page", "browser_screenshot"),
            ),
            "extract": ActionSpec(
                tool=self.name,
                action="extract",
                description="Extract structured fields from the current page.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "schema": {"type": "object"},
                        "fields": {"type": "object"},
                        "session_id": {"type": "string"},
                    },
                },
                reads=True,
                produces_artifacts=("browser_page", "browser_candidates", "browser_extract"),
            ),
            "run_objective": ActionSpec(
                tool=self.name,
                action="run_objective",
                description="Execute a bounded multi-step browser objective.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "start_url": {"type": "string"},
                        "max_steps": {"type": "integer"},
                        "max_sources": {"type": "integer"},
                        "visible": {"type": "boolean"},
                        "allow_desktop_fallback": {"type": "boolean"},
                        "keep_session": {"type": "boolean"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["objective"],
                },
                writes=True,
                produces_artifacts=("browser_page", "browser_candidates", "browser_extract"),
            ),
            "close_session": ActionSpec(
                tool=self.name,
                action="close_session",
                description="Close an existing browser session.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                    },
                    "required": ["session_id"],
                },
                writes=True,
            ),
            "list_sessions": ActionSpec(
                tool=self.name,
                action="list_sessions",
                description="List active browser sessions.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                reads=True,
                produces_artifacts=("browser_sessions",),
            ),
        }
