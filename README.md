# JClaw

JClaw is a lightweight AI assistant daemon for macOS that polls Telegram, stores state in SQLite, and talks to any OpenAI-compatible API endpoint such as DeepSeek or GLM.

The bootstrap in this repo is intentionally lean:

- Telegram long-polling instead of a heavier chat bridge
- SQLite for history, memory, and automation state
- A compact prompt pipeline to reduce token usage
- A `launchd` installer so it can stay alive on your MacBook

## Quick Start

1. Create a virtualenv and install the package:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

2. Generate a starter config:

```bash
jclaw init-config
```

3. Edit `~/.config/jclaw/config.toml` with:

- `telegram.bot_token`
- `provider.api_key`
- `provider.base_url`
- `provider.model`

4. Run a health check:

```bash
jclaw doctor
```

5. Run in the foreground:

```bash
jclaw run
```

6. Install as a background macOS daemon:

```bash
jclaw install-launchd
```

## Telegram Setup

Create a bot with `@BotFather`, copy the bot token into the config, and send the bot a message from Telegram. If you want to restrict access, set `telegram.allowed_chat_ids` after you learn your chat ID from the logs or database.

## Low-Token Design

JClaw keeps token usage down by:

- handling simple memory commands locally without an LLM call
- sending only a small recent history window
- retrieving only a few relevant memory snippets
- instructing the model to answer briefly by default

## Useful Inputs

These can be sent from Telegram or from the CLI with `jclaw send`:

- `/help`
- `/remember project = building a telegram-first assistant`
- `/memory`
- `/forget project`
- `Remind me every day at 9 AM to stretch`
- `Remind me to practice interview at 9 AM on April 27`
- `Remind me in 30 minutes to check the build`
