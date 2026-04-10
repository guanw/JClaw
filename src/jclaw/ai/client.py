from __future__ import annotations

from collections.abc import Sequence
import json

import httpx

from jclaw.core.config import ProviderConfig


class OpenAICompatibleClient:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=config.timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        if not self.config.base_url or not self.config.model:
            raise RuntimeError("provider.base_url and provider.model must be configured before chatting")

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        response = self._client.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": self.config.model,
                "messages": list(messages),
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts).strip()
        if isinstance(content, str):
            return content.strip()
        return json.dumps(content)

    def health_check(self) -> str:
        if not self.config.base_url:
            return "provider.base_url is missing"
        response = self._client.get(f"{self.config.base_url.rstrip('/')}/models")
        if response.is_success:
            return "ok"
        return f"provider returned HTTP {response.status_code}"

