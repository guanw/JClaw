from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from playwright.sync_api import BrowserContext, Error, Page, Playwright, TimeoutError, sync_playwright

from jclaw.core.defaults import BROWSER_SLOW_MO_MS, BROWSER_VIEWPORT_HEIGHT, BROWSER_VIEWPORT_WIDTH
from jclaw.tools.browser.models import Target
from jclaw.tools.browser.session import BrowserSessionStore


def _clean_text(value: str, *, limit: int = 4000) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit]


def _infer_page_kind(url: str, title: str, text: str) -> str:
    haystack = f"{url} {title} {text[:400]}".lower()
    if any(token in haystack for token in ("search results", "/search", "?q=", "duckduckgo", "google", "bing")):
        return "search_results"
    if any(token in haystack for token in ("sign in", "log in", "login", "password")):
        return "auth"
    if any(token in haystack for token in ("checkout", "cart", "buy now")):
        return "commerce"
    if any(token in haystack for token in ("article", "blog", "news", "report", "press release")):
        return "article"
    return "page"


class PlaywrightBrowserDriver:
    mode = "playwright"

    def __init__(
        self,
        session_store: BrowserSessionStore,
        *,
        channel: str = "chromium",
        headless: bool = False,
        slow_mo_ms: int = BROWSER_SLOW_MO_MS,
        viewport_width: int = BROWSER_VIEWPORT_WIDTH,
        viewport_height: int = BROWSER_VIEWPORT_HEIGHT,
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
        page_kind = _infer_page_kind(page.url, title, visible_text)
        elements = self._extract_elements(page, page_kind=page_kind)
        links = [
            {"id": item["id"], "text": item["text"], "href": item["href"], "area": item["area"]}
            for item in elements
            if item.get("role") == "link" and item.get("href")
        ]
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
            "page_kind": page_kind,
            "text": visible_text,
            "elements": elements,
            "links": links,
            "forms": forms,
            "mode": self.mode,
        }

    def _extract_elements(self, page: Page, *, page_kind: str) -> list[dict[str, Any]]:
        elements = page.evaluate(
            """
            ({ pageKind }) => {
              const normalize = (value, limit = 220) => {
                const text = (value || '').replace(/\\s+/g, ' ').trim();
                return text.slice(0, limit);
              };
              const classifyArea = (node) => {
                if (!node || !node.closest) return 'body';
                if (node.closest('main, article, [role="main"]')) return 'main';
                if (node.closest('nav, [role="navigation"]')) return 'nav';
                if (node.closest('header, [role="banner"]')) return 'header';
                if (node.closest('footer, [role="contentinfo"]')) return 'footer';
                if (node.closest('aside, [role="complementary"]')) return 'aside';
                if (node.closest('form')) return 'form';
                if (node.closest('dialog, [role="dialog"], [aria-modal="true"]')) return 'modal';
                return 'body';
              };
              const inferRole = (node) => {
                const explicit = (node.getAttribute('role') || '').trim().toLowerCase();
                if (explicit) return explicit;
                const tag = node.tagName.toLowerCase();
                if (tag === 'a') return 'link';
                if (tag === 'button') return 'button';
                if (tag === 'input') return node.type === 'search' ? 'searchbox' : 'textbox';
                if (tag === 'textarea') return 'textbox';
                if (tag === 'select') return 'combobox';
                if (/^h[1-6]$/.test(tag)) return 'heading';
                if (tag === 'summary') return 'button';
                return tag;
              };
              const inferText = (node) => {
                const tag = node.tagName.toLowerCase();
                return normalize(
                  node.getAttribute('aria-label') ||
                  node.getAttribute('title') ||
                  (tag === 'input' ? (node.value || node.placeholder || node.name) : '') ||
                  node.innerText ||
                  node.textContent ||
                  node.getAttribute('alt') ||
                  ''
                );
              };
              const selectorHint = (node) => {
                const tag = node.tagName.toLowerCase();
                if (node.id) return `${tag}#${node.id}`;
                const name = node.getAttribute('name');
                if (name) return `${tag}[name="${name}"]`;
                const href = node.getAttribute('href');
                if (href) return `${tag}[href]`;
                return tag;
              };
              const visible = (node) => {
                const style = window.getComputedStyle(node);
                if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
                if (node.getAttribute('aria-hidden') === 'true') return false;
                const rect = node.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              };
              const candidates = Array.from(
                document.querySelectorAll('a[href], button, input, textarea, select, summary, [role], h1, h2, h3')
              );
              const seen = new Set();
              const items = [];
              for (const node of candidates) {
                if (!(node instanceof HTMLElement)) continue;
                if (!visible(node)) continue;
                const role = inferRole(node);
                const href = normalize(node.getAttribute('href') || node.href || '', 500);
                const text = inferText(node);
                const clickable = ['link', 'button', 'menuitem', 'option', 'tab'].includes(role)
                  || Boolean(href)
                  || node.tagName.toLowerCase() === 'summary';
                if (!text && !href && !clickable) continue;
                const area = classifyArea(node);
                const key = `${role}|${text}|${href}|${area}`;
                if (seen.has(key)) continue;
                seen.add(key);
                const rect = node.getBoundingClientRect();
                let scoreHint = 0;
                if (area === 'main') scoreHint += 0.35;
                else if (area === 'nav' || area === 'footer') scoreHint -= 0.25;
                if (role === 'link') scoreHint += 0.25;
                if (pageKind === 'search_results' && area === 'main') scoreHint += 0.15;
                if (text.length > 18) scoreHint += 0.1;
                items.push({
                  id: `e${items.length + 1}`,
                  role,
                  text,
                  href,
                  area,
                  clickable,
                  visible: true,
                  selector_hint: selectorHint(node),
                  score_hint: Number(scoreHint.toFixed(2)),
                  y: Math.round(rect.top),
                });
              }
              items.sort((a, b) => {
                if (b.score_hint !== a.score_hint) return b.score_hint - a.score_hint;
                return a.y - b.y;
              });
              return items.slice(0, 40).map(({ y, ...item }) => item);
            }
            """,
            {"pageKind": page_kind},
        )
        return elements if isinstance(elements, list) else []

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
