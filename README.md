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

## Prod Vs Dev

JClaw can run cleanly with one production Telegram bot and one development Telegram bot.

- `prod`: background `launchd` service, always-on bot, config at `~/.config/jclaw/config.toml`
- `dev`: foreground process for interactive testing, separate bot, config at `~/.config/jclaw/config.dev.toml`

The recommended pattern is:

1. Keep `~/.config/jclaw/config.toml` as the full production config.
2. Make `~/.config/jclaw/config.dev.toml` a small overlay that inherits from prod and overrides only the settings that should differ.

Example `config.dev.toml`:

```toml
extends = "/Users/guanw/.config/jclaw/config.toml"

[telegram]
bot_token = "YOUR_DEV_BOT_TOKEN"
allowed_chat_ids = []

[daemon]
state_dir = "/Users/guanw/Library/Application Support/JClawDev"
db_path = "/Users/guanw/Library/Application Support/JClawDev/jclaw.db"
stdout_log = "/Users/guanw/Library/Logs/JClawDev/stdout.log"
stderr_log = "/Users/guanw/Library/Logs/JClawDev/stderr.log"
launchd_label = "com.jclaw.dev"

[memory]
max_memory_items = 100
```

Notes:

- `extends = "..."` loads the base config first, then applies overrides from the current file.
- Use a different Telegram bot token for dev so dev and prod do not fight over `getUpdates`.
- Use a separate `state_dir` and `db_path` for dev if you want dev memory, traces, and offsets isolated from prod.

Start production:

```bash
scripts/bootstrap_local_prod.sh
```

Restart production after code changes:

```bash
scripts/bootstrap_local_prod.sh --restart-only
```

Start the dev Telegram loop in the foreground:

```bash
scripts/run_dev.sh
```

Run a local dev turn without starting a Telegram poller:

```bash
.venv/bin/python jclaw.py send --config ~/.config/jclaw/config.dev.toml "test prompt"
```

Important:

- In the current CLI, `--config` must come after the subcommand, for example `jclaw.py run --config ...`.
- `jclaw.py --config ... run` may silently fall back to the default prod config.

This split avoids the common failure mode where two `jclaw run` processes use the same Telegram bot token and trigger Telegram `409 Conflict` errors.

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
