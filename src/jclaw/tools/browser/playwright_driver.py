from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from playwright.sync_api import BrowserContext, Error, Page, Playwright, TimeoutError, sync_playwright

from jclaw.tools.browser.models import Target
from jclaw.tools.browser.session import BrowserSessionStore


def _clean_text(value: str, *, limit: int = 4000) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit]


class PlaywrightBrowserDriver:
    mode = "playwright"

    def __init__(
        self,
        session_store: BrowserSessionStore,
        *,
        channel: str = "chromium",
        headless: bool = False,
        slow_mo_ms: int = 0,
        viewport_width: int = 1440,
        viewport_height: int = 960,
    ) -> None:
        self.session_store = session_store
        self.channel = channel
        self.headless = headless
        self.slow_mo_ms = slow_mo_ms
        self.viewport = {"width": viewport_width, "height": viewport_height}
        self._playwright: Playwright | None = None
        self._contexts: dict[str, BrowserContext] = {}
        self._pages: dict[str, Page] = {}
        self._tab_ids: dict[str, str] = {}
        self._tab_seq = 0

    def close(self) -> None:
        for session_id in list(self._contexts):
            self.close_session(session_id)
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def close_session(self, session_id: str) -> None:
        context = self._contexts.pop(session_id, None)
        page = self._pages.pop(session_id, None)
        if page is not None:
            self._tab_ids.pop(str(id(page)), None)
        if context is not None:
            context.close()

    def open_url(self, session_id: str, url: str, *, new_tab: bool) -> dict[str, Any]:
        page = self._get_or_create_page(session_id, new_tab=new_tab)
        page.goto(url, wait_until="domcontentloaded")
        session = self.session_store.get_session(session_id)
        session.current_url = page.url
        session.current_tab_id = self._tab_id(page)
        return {
            "session_id": session_id,
            "tab_id": session.current_tab_id,
            "url": page.url,
            "title": page.title(),
            "mode": self.mode,
        }

    def click(self, session_id: str, target: Target) -> dict[str, Any]:
        page = self._get_or_create_page(session_id)
        locator = self._resolve_target(page, target)
        locator.first.click()
        session = self.session_store.get_session(session_id)
        session.current_url = page.url
        return {"session_id": session_id, "tab_id": self._tab_id(page), "url": page.url, "mode": self.mode}

    def type(self, session_id: str, target: Target, text: str, *, submit: bool) -> dict[str, Any]:
        page = self._get_or_create_page(session_id)
        locator = self._resolve_target(page, target).first
        locator.click()
        locator.fill(text)
        if submit:
            locator.press("Enter")
        return {
            "session_id": session_id,
            "tab_id": self._tab_id(page),
            "text": text,
            "submit": submit,
            "mode": self.mode,
        }

    def scroll(self, session_id: str, *, direction: str, amount: int) -> dict[str, Any]:
        page = self._get_or_create_page(session_id)
        delta = amount if direction == "down" else -amount
        page.mouse.wheel(0, delta)
        return {"session_id": session_id, "direction": direction, "amount": amount, "mode": self.mode}

    def wait_for(self, session_id: str, target: Target | None, timeout_ms: int) -> dict[str, Any]:
        page = self._get_or_create_page(session_id)
        if target is None:
            page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            matched = "load_state"
        else:
            self._resolve_target(page, target).first.wait_for(timeout=timeout_ms)
            matched = "target"
        return {"session_id": session_id, "matched": matched, "timeout_ms": timeout_ms, "mode": self.mode}

    def read_page(self, session_id: str) -> dict[str, Any]:
        page = self._get_or_create_page(session_id)
        title = page.title()
        visible_text = _clean_text(
            page.locator("body").inner_text(timeout=5000) if page.locator("body").count() else ""
        )
        links = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('a'))
              .slice(0, 20)
              .map((a) => ({text: (a.innerText || a.textContent || '').trim(), href: a.href || ''}))
              .filter((item) => item.text || item.href)
            """
        )
        forms = page.evaluate(
            """
            () => Array.from(document.forms).slice(0, 10).map((form, index) => ({
              id: form.id || `form-${index}`,
              fields: Array.from(form.querySelectorAll('input, textarea, select'))
                .map((el) => el.name || el.id || el.type || el.tagName.toLowerCase())
            }))
            """
        )
        session = self.session_store.get_session(session_id)
        session.current_url = page.url
        session.current_tab_id = self._tab_id(page)
        return {
            "session_id": session_id,
            "tab_id": session.current_tab_id,
            "url": page.url,
            "title": title,
            "text": visible_text,
            "links": links,
            "forms": forms,
            "mode": self.mode,
        }

    def screenshot(self, session_id: str, *, full_page: bool, path: str) -> dict[str, Any]:
        page = self._get_or_create_page(session_id)
        page.screenshot(path=path, full_page=full_page)
        return {"session_id": session_id, "tab_id": self._tab_id(page), "full_page": full_page, "mode": self.mode}

    def _start(self) -> Playwright:
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        return self._playwright

    def _context(self, session_id: str) -> BrowserContext:
        context = self._contexts.get(session_id)
        if context is not None:
            return context

        session = self.session_store.get_session(session_id)
        browser_type = self._start().chromium
        context = browser_type.launch_persistent_context(
            user_data_dir=Path(session.profile_dir),
            headless=self.headless,
            channel=self.channel,
            slow_mo=self.slow_mo_ms,
            viewport=self.viewport,
        )
        self._contexts[session_id] = context
        if context.pages:
            page = context.pages[0]
            self._pages[session_id] = page
            session.current_tab_id = self._tab_id(page)
            session.current_url = page.url
        return context

    def _get_or_create_page(self, session_id: str, *, new_tab: bool = False) -> Page:
        context = self._context(session_id)
        page = self._pages.get(session_id)
        if page is None or page.is_closed() or new_tab:
            page = context.new_page()
            self._pages[session_id] = page
        self.session_store.get_session(session_id).current_tab_id = self._tab_id(page)
        return page

    def _resolve_target(self, page: Page, target: Target):
        if target.selector:
            return page.locator(target.selector)
        if target.role and target.name:
            return page.get_by_role(target.role, name=target.name)
        if target.text:
            return page.get_by_text(target.text, exact=False)
        if target.xpath:
            return page.locator(f"xpath={target.xpath}")
        raise ValueError("browser target requires selector, role+name, text, or xpath")

    def _tab_id(self, page: Page) -> str:
        key = str(id(page))
        if key not in self._tab_ids:
            self._tab_seq += 1
            self._tab_ids[key] = f"tab_{self._tab_seq}"
        return self._tab_ids[key]
