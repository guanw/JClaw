from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import quote_plus
from typing import Any, Callable

from jclaw.tools.base import ToolContext, ToolResult
from jclaw.tools.browser.artifacts import BrowserArtifactStore
from jclaw.tools.browser.desktop_driver import DesktopBrowserDriver
from jclaw.tools.browser.models import Target
from jclaw.tools.browser.permissions import check_permissions
from jclaw.tools.browser.planner import BrowserPlanner
from jclaw.tools.browser.playwright_driver import PlaywrightBrowserDriver
from jclaw.tools.browser.session import BrowserSessionStore


class BrowserTool:
    name = "browser"

    def __init__(
        self,
        base_dir: str | Path,
        options: dict[str, Any] | None = None,
        choose_link: Callable[[str, dict[str, Any]], str | None] | None = None,
    ) -> None:
        root = Path(base_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.sessions = BrowserSessionStore(root / "sessions")
        self.artifacts = BrowserArtifactStore(root / "artifacts")
        self.playwright = PlaywrightBrowserDriver(
            self.sessions,
            channel=str(options.get("channel", "chromium")) if options else "chromium",
            headless=bool(options.get("headless", False)) if options else False,
            slow_mo_ms=int(options.get("slow_mo_ms", 0)) if options else 0,
            viewport_width=int(options.get("viewport_width", 1440)) if options else 1440,
            viewport_height=int(options.get("viewport_height", 960)) if options else 960,
        )
        self.desktop = DesktopBrowserDriver()
        self.planner = BrowserPlanner()
        self._chat_sessions: dict[str, str] = {}
        self._choose_link = choose_link

    def close(self) -> None:
        self.playwright.close()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "actions": [
                "open_url",
                "search_web",
                "read_page",
                "click",
                "type",
                "scroll",
                "wait_for",
                "screenshot",
                "extract",
                "run_objective",
                "close_session",
                "list_sessions",
                "permissions",
            ],
        }

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
            "permissions": self._permissions,
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
        return ToolResult(ok=True, summary=f"Opened {url}", data=data)

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
        return ToolResult(ok=True, summary="Read current page state.", data=data)

    def _click(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.click(session.session_id, self._target(params))
        return ToolResult(ok=True, summary="Clicked target.", data=data)

    def _type(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.type(
            session.session_id,
            self._target(params),
            str(params.get("text", "")),
            submit=bool(params.get("submit", False)),
        )
        return ToolResult(ok=True, summary="Typed into target.", data=data)

    def _scroll(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        data = driver.scroll(
            session.session_id,
            direction=str(params.get("direction", "down")),
            amount=int(params.get("amount", 600)),
        )
        return ToolResult(ok=True, summary="Scrolled page.", data=data)

    def _wait_for(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        driver = self._driver(params)
        raw_target = params.get("target")
        target = self._target(params) if raw_target else None
        data = driver.wait_for(session.session_id, target, timeout_ms=int(params.get("timeout_ms", 5000)))
        return ToolResult(ok=True, summary="Waited for page state.", data=data)

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
        return ToolResult(ok=True, summary="Created screenshot artifact.", data=data)

    def _extract(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        schema = params.get("schema", {})
        return ToolResult(
            ok=True,
            summary="Extraction scaffold ready.",
            data={"fields": {key: "" for key in schema}, "implemented": False},
        )

    def _run_objective(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._ensure_session(params, ctx)
        objective = str(params.get("objective", "")).strip()
        max_steps = max(1, min(int(params.get("max_steps", 3)), 6))
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
        for _ in range(max_steps - 1):
            next_url = self._choose_follow_up_url(objective, current_read.data)
            self._trace_event(
                "follow_up_choice",
                ctx=ctx,
                action="run_objective",
                params={
                    "objective": objective,
                    "current_url": current_read.data.get("url", ""),
                    "candidate_elements": current_read.data.get("elements", [])[:12],
                    "chosen_url": next_url,
                },
            )
            if not next_url or next_url == current_read.data.get("url"):
                break
            follow_open = self._open_url({"session_id": session.session_id, "url": next_url}, ctx)
            executed_steps.append(
                {
                    "action": "open_url",
                    "params": {"url": next_url},
                    "reason": "Follow a likely relevant result.",
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
            break

        observation = {"session_id": session.session_id, "url": current_read.data.get("url", target_url)}
        step = self.planner.next_step(objective, observation, history=executed_steps)
        result = ToolResult(
            ok=True,
            summary="Executed browser objective and captured the latest page.",
            data={
                "session_id": session.session_id,
                "objective": objective,
                "steps": executed_steps + [{"action": step.action, "params": step.params, "reason": step.reason}],
                "implemented": True,
                **current_read.data,
                "mode": self._driver(params).mode,
            },
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

    def _permissions(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        status = check_permissions()
        return ToolResult(ok=True, summary="Read browser permission status.", data=asdict(status))

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
        llm_choice = self._choose_follow_up_url_via_llm(objective, page_data)
        if llm_choice:
            return llm_choice
        return self._pick_follow_up_url(objective, page_data)

    def _choose_follow_up_url_via_llm(self, objective: str, page_data: dict[str, Any]) -> str | None:
        if self._choose_link is None:
            return None
        try:
            return self._choose_link(objective, page_data)
        except Exception:  # noqa: BLE001
            return None

    def _pick_follow_up_url(self, objective: str, page_data: dict[str, Any]) -> str | None:
        current_url = str(page_data.get("url", ""))
        candidates = self._extract_candidate_elements(page_data)
        if not candidates:
            return None

        objective_terms = {term for term in re.findall(r"[a-z0-9]+", objective.lower()) if len(term) > 2}
        scored_links: list[tuple[int, str, str]] = []
        for item in candidates:
            href = str(item.get("href", "")).strip()
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
        if scored_links:
            return scored_links[0][1]
        return None

    def _extract_candidate_elements(self, page_data: dict[str, Any]) -> list[dict[str, Any]]:
        elements = page_data.get("elements", [])
        if isinstance(elements, list) and elements:
            candidates = []
            for item in elements:
                if not isinstance(item, dict):
                    continue
                if item.get("role") != "link":
                    continue
                href = str(item.get("href", "")).strip()
                if not href:
                    continue
                candidates.append(item)
            if candidates:
                return candidates
        links = page_data.get("links", [])
        if isinstance(links, list):
            return [item for item in links if isinstance(item, dict)]
        return []

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
