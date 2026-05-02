from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from jclaw.tools.base import ToolContext, ToolLoopState, ToolResult
from jclaw.tools.browser.models import Target


class BrowserNavigationMixin:
    def _ensure_session(self, params: dict[str, Any], ctx: ToolContext):
        session_id = params.get("session_id")
        if session_id:
            session = self.sessions.get_session(str(session_id))
            self._chat_sessions[ctx.chat_id] = session.session_id
            return session
        if ctx.execution is not None:
            loop_state = ctx.execution.tool_state.get(self.name, {})
            loop_session_id = str(loop_state.get("session_id", "")).strip()
            if loop_session_id:
                try:
                    session = self.sessions.get_session(loop_session_id)
                    self._chat_sessions[ctx.chat_id] = session.session_id
                    return session
                except KeyError:
                    pass
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
        if ctx.execution is not None:
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
        data = {
            **data,
            "session_id": session.session_id,
            "url": str(data.get("url", "")).strip() or url,
        }
        return ToolResult(
            ok=True,
            summary=f"Opened {url}",
            data={
                **data,
                **self._browser_result_payload(data, include_candidates=True),
            },
        )

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
        data = {
            **data,
            "session_id": session.session_id,
            "url": str(data.get("url", "")).strip() or session.current_url or "about:blank",
        }
        session.current_url = str(data.get("url", "")).strip() or session.current_url
        return ToolResult(
            ok=True,
            summary="Read current page state.",
            data={
                **data,
                **self._browser_result_payload(data, include_candidates=True),
            },
        )

    def _click(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.click(session.session_id, self._target(params))
        data = {
            **data,
            "session_id": session.session_id,
            "url": str(data.get("url", "")).strip() or session.current_url or "about:blank",
        }
        session.current_url = str(data.get("url", "")).strip() or session.current_url
        return ToolResult(
            ok=True,
            summary="Clicked target.",
            data={
                **data,
                **self._browser_result_payload(data, include_candidates=True),
            },
        )

    def _type(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.type(
            session.session_id,
            self._target(params),
            str(params.get("text", "")),
            submit=bool(params.get("submit", False)),
        )
        data = {
            **data,
            "session_id": session.session_id,
            "url": str(data.get("url", "")).strip() or session.current_url or "about:blank",
        }
        session.current_url = str(data.get("url", "")).strip() or session.current_url
        return ToolResult(
            ok=True,
            summary="Typed into target.",
            data={
                **data,
                **self._browser_result_payload(data, include_candidates=True),
            },
        )

    def _scroll(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.scroll(
            session.session_id,
            direction=str(params.get("direction", "down")),
            amount=int(params.get("amount", 600)),
        )
        data = {
            **data,
            "session_id": session.session_id,
            "url": str(data.get("url", "")).strip() or session.current_url or "about:blank",
        }
        session.current_url = str(data.get("url", "")).strip() or session.current_url
        return ToolResult(
            ok=True,
            summary="Scrolled page.",
            data={
                **data,
                **self._browser_result_payload(data, include_candidates=True),
            },
        )

    def _wait_for(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        raw_target = params.get("target")
        target = self._target(params) if raw_target else None
        data = driver.wait_for(session.session_id, target, timeout_ms=int(params.get("timeout_ms", 5000)))
        data = {
            **data,
            "session_id": session.session_id,
            "url": str(data.get("url", "")).strip() or session.current_url or "about:blank",
        }
        session.current_url = str(data.get("url", "")).strip() or session.current_url
        return ToolResult(
            ok=True,
            summary="Waited for page state.",
            data={
                **data,
                **self._browser_result_payload(data, include_candidates=True),
            },
        )

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
        return ToolResult(
            ok=True,
            summary="Created screenshot artifact.",
            data={
                **data,
                **self._browser_result_payload(
                    data,
                    include_candidates=False,
                    include_screenshot={
                        "artifact_id": artifact.artifact_id,
                        "path": artifact.path,
                    },
                ),
            },
        )

    def _extract(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        page_result = self._read_page({"session_id": session.session_id}, ctx)
        page_data = page_result.data
        fields = params.get("schema") or params.get("fields") or {}
        if not isinstance(fields, dict) or not fields:
            result = ToolResult(
                ok=False,
                summary="No extraction fields were provided.",
                data={"fields": {}, **page_data},
            )
            self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="extract", params=params)
            return result
        extracted = self._extract_fields_via_reasoner(page_data, fields)
        if not extracted:
            result = ToolResult(
                ok=False,
                summary="Browser extraction is unavailable for the current page.",
                data={"fields": {}, **page_data},
            )
            self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="extract", params=params)
            return result
        result = ToolResult(
            ok=True,
            summary=f"Extracted {len(extracted.get('fields', {}))} field(s) from the current page.",
            data={
                "fields": extracted.get("fields", {}),
                "evidence_refs": extracted.get("evidence_refs", []),
                "missing_information": extracted.get("missing_information", ""),
                **page_data,
                **self._browser_result_payload(
                    page_data,
                    include_candidates=True,
                    include_extract={
                        "fields": extracted.get("fields", {}),
                        "evidence_refs": extracted.get("evidence_refs", []),
                        "missing_information": extracted.get("missing_information", ""),
                    },
                ),
            },
        )
        self._cleanup_session_if_needed(session.session_id, ctx=ctx, action="extract", params=params)
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
        return ToolResult(
            ok=True,
            summary=f"Closed session {session_id}.",
            data={"session_id": session_id},
            loop_state=ToolLoopState(clear=True),
        )

    def _list_sessions(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        sessions = self.sessions.list_sessions()
        return ToolResult(
            ok=True,
            summary=f"Listed {len(sessions)} browser sessions.",
            data={
                "sessions": [session.as_dict() for session in sessions],
                "allow_tool_followup": True,
                "artifacts": {
                    "browser_sessions:latest": {
                        "count": len(sessions),
                        "sessions": [session.as_dict() for session in sessions[:5]],
                    }
                },
            },
        )
