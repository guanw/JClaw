from __future__ import annotations

from jclaw.ai.agent import AssistantAgent


class LocalChannelHarness:
    def __init__(self, agent: AssistantAgent) -> None:
        self.agent = agent

    def send(self, chat_id: str, text: str, *, user_name: str = "cli") -> str:
        return self.agent.handle_text(chat_id, text, user_name=user_name)

