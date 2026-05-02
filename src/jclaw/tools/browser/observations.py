from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


class BrowserObservationsMixin:
    def _browser_result_payload(
        self,
        page_data: dict[str, Any],
        *,
        include_candidates: bool,
        include_extract: dict[str, Any] | None = None,
        include_screenshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            "browser_page:latest": self._browser_page_artifact(page_data),
        }
        if include_candidates:
            artifacts["browser_candidates:latest"] = self._browser_candidates_artifact(page_data)
        if include_extract is not None:
            artifacts["browser_extract:latest"] = self._browser_extract_artifact(include_extract)
        if include_screenshot is not None:
            artifacts["browser_screenshot:latest"] = dict(include_screenshot)
        return {
            "allow_tool_followup": True,
            "artifacts": artifacts,
        }

    def _browser_page_artifact(self, page_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_id": str(page_data.get("session_id", "")),
            "tab_id": str(page_data.get("tab_id", "")),
            "url": str(page_data.get("url", "")),
            "title": str(page_data.get("title", ""))[:200],
            "page_kind": str(page_data.get("page_kind", "")),
            "mode": str(page_data.get("mode", "")),
        }

    def _browser_candidates_artifact(self, page_data: dict[str, Any]) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for item in self._extract_candidate_elements(page_data)[:8]:
            href = str(item.get("href", "")).strip()
            candidates.append(
                {
                    "id": str(item.get("id", "")),
                    "role": str(item.get("role", "")),
                    "text": str(item.get("text", ""))[:140],
                    "href": self._normalize_url(href) if href else "",
                    "area": str(item.get("area", "")),
                    "clickable": bool(item.get("clickable", False)),
                }
            )
        return {"count": len(candidates), "actions": candidates}

    def _browser_extract_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        artifact: dict[str, Any] = {}
        if "fields" in payload:
            artifact["fields"] = payload.get("fields", {})
        if "objective" in payload:
            artifact["objective"] = str(payload.get("objective", ""))
        if "termination_reason" in payload:
            artifact["termination_reason"] = str(payload.get("termination_reason", ""))
        if "evidence_refs" in payload:
            artifact["evidence_refs"] = [str(item) for item in payload.get("evidence_refs", [])[:5]]
        if "missing_information" in payload:
            artifact["missing_information"] = str(payload.get("missing_information", ""))[:240]
        if "sources" in payload:
            artifact["sources"] = [
                {
                    "url": str(item.get("url", "")),
                    "title": str(item.get("title", ""))[:160],
                }
                for item in payload.get("sources", [])[:3]
                if isinstance(item, dict)
            ]
        return artifact

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
        prior_fingerprints = {
            str(item.get("text_fingerprint", ""))
            for item in prior_observations
            if item.get("text_fingerprint")
        }
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
