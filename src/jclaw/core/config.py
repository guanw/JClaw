from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib


APP_NAME = "JClaw"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return Path.home() / ".config" / "jclaw" / "config.toml"


def default_state_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME


def default_log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / APP_NAME


def _expand_path(value: str | Path | None, fallback: Path) -> Path:
    if value in (None, ""):
        return fallback
    return Path(value).expanduser()


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(slots=True)
class ProviderConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 350
    temperature: float = 0.2
    timeout_seconds: float = 60.0
    system_prompt_files: tuple[str, ...] = (
        "SOUL.md",
        "IDENTITY.md",
        "BOOTSTRAP.md",
        "CLAUDE.md",
    )


@dataclass(slots=True)
class TelegramConfig:
    bot_token: str = ""
    base_url: str = "https://api.telegram.org/bot"
    poll_timeout_seconds: int = 20
    allowed_chat_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class DaemonConfig:
    state_dir: Path = field(default_factory=default_state_dir)
    db_path: Path = field(default_factory=lambda: default_state_dir() / "jclaw.db")
    stdout_log: Path = field(default_factory=lambda: default_log_dir() / "stdout.log")
    stderr_log: Path = field(default_factory=lambda: default_log_dir() / "stderr.log")
    launchd_label: str = "com.jclaw.daemon"
    idle_sleep_seconds: float = 2.0


@dataclass(slots=True)
class MemoryConfig:
    max_context_messages: int = 6
    max_memory_items: int = 4


@dataclass(slots=True)
class Config:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    config_path: Path = field(default_factory=default_config_path)
    repo_root: Path = field(default_factory=repo_root)

    def ensure_runtime_dirs(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.daemon.state_dir.mkdir(parents=True, exist_ok=True)
        self.daemon.stdout_log.parent.mkdir(parents=True, exist_ok=True)


def render_default_config() -> str:
    state_dir = default_state_dir()
    return f"""# JClaw configuration
# Replace the placeholder values below.

[provider]
# DeepSeek example: https://api.deepseek.com/v1
# GLM example: https://open.bigmodel.cn/api/paas/v4
api_key = ""
base_url = ""
model = ""
max_tokens = 350
temperature = 0.2
timeout_seconds = 60
system_prompt_files = ["SOUL.md", "IDENTITY.md", "BOOTSTRAP.md", "CLAUDE.md"]

[telegram]
bot_token = ""
base_url = "https://api.telegram.org/bot"
poll_timeout_seconds = 20
allowed_chat_ids = []

[daemon]
state_dir = "{state_dir}"
db_path = "{state_dir / "jclaw.db"}"
stdout_log = "{default_log_dir() / "stdout.log"}"
stderr_log = "{default_log_dir() / "stderr.log"}"
launchd_label = "com.jclaw.daemon"
idle_sleep_seconds = 2.0

[memory]
max_context_messages = 6
max_memory_items = 4
"""


def load_config(path: str | Path | None = None) -> Config:
    config_path = _expand_path(path, default_config_path())
    data: dict[str, object] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)

    provider_data = data.get("provider", {})
    telegram_data = data.get("telegram", {})
    daemon_data = data.get("daemon", {})
    memory_data = data.get("memory", {})

    provider = ProviderConfig(
        api_key=str(os.environ.get("JCLAW_API_KEY", provider_data.get("api_key", ""))),
        base_url=str(os.environ.get("JCLAW_BASE_URL", provider_data.get("base_url", ""))),
        model=str(os.environ.get("JCLAW_MODEL", provider_data.get("model", ""))),
        max_tokens=int(provider_data.get("max_tokens", 350)),
        temperature=float(provider_data.get("temperature", 0.2)),
        timeout_seconds=float(provider_data.get("timeout_seconds", 60)),
        system_prompt_files=tuple(
            provider_data.get("system_prompt_files", ("SOUL.md", "IDENTITY.md", "BOOTSTRAP.md", "CLAUDE.md"))
        ),
    )
    telegram = TelegramConfig(
        bot_token=str(os.environ.get("JCLAW_TELEGRAM_BOT_TOKEN", telegram_data.get("bot_token", ""))),
        base_url=str(telegram_data.get("base_url", "https://api.telegram.org/bot")),
        poll_timeout_seconds=int(telegram_data.get("poll_timeout_seconds", 20)),
        allowed_chat_ids=_env_list("JCLAW_TELEGRAM_ALLOWED_CHAT_IDS")
        or tuple(str(value) for value in telegram_data.get("allowed_chat_ids", [])),
    )
    daemon = DaemonConfig(
        state_dir=_expand_path(daemon_data.get("state_dir"), default_state_dir()),
        db_path=_expand_path(daemon_data.get("db_path"), default_state_dir() / "jclaw.db"),
        stdout_log=_expand_path(daemon_data.get("stdout_log"), default_log_dir() / "stdout.log"),
        stderr_log=_expand_path(daemon_data.get("stderr_log"), default_log_dir() / "stderr.log"),
        launchd_label=str(daemon_data.get("launchd_label", "com.jclaw.daemon")),
        idle_sleep_seconds=float(daemon_data.get("idle_sleep_seconds", 2.0)),
    )
    memory = MemoryConfig(
        max_context_messages=int(memory_data.get("max_context_messages", 6)),
        max_memory_items=int(memory_data.get("max_memory_items", 4)),
    )

    config = Config(
        provider=provider,
        telegram=telegram,
        daemon=daemon,
        memory=memory,
        config_path=config_path,
        repo_root=repo_root(),
    )
    config.ensure_runtime_dirs()
    return config
