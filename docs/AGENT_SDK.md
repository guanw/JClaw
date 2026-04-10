# Agent Runtime Notes

This bootstrap uses a thin OpenAI-compatible client instead of a Claude-specific SDK so it can work with local or self-hosted providers that expose a chat-completions API.

Provider examples:

- DeepSeek: `https://api.deepseek.com/v1`
- GLM: `https://open.bigmodel.cn/api/paas/v4`

The daemon keeps the runtime simple:

- Telegram provides transport
- SQLite provides durable state
- the agent handles commands locally when possible
- only free-form tasks go to the configured model

