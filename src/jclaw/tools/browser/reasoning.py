from __future__ import annotations

import json
import re
from typing import Any, Callable


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


class BrowserReasoningMixin:
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
