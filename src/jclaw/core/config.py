from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib

from jclaw.core.defaults import (
    APP_NAME,
    AUTOMATION_ENABLED,
    BROWSER_CHANNEL,
    BROWSER_ENABLED,
    BROWSER_HEADLESS,
    BROWSER_MAX_OBJECTIVE_STEPS,
    BROWSER_MAX_RESEARCH_SOURCES,
    BROWSER_SLOW_MO_MS,
    BROWSER_VIEWPORT_HEIGHT,
    BROWSER_VIEWPORT_WIDTH,
    DAEMON_IDLE_SLEEP_SECONDS,
    DAEMON_LAUNCHD_LABEL,
    KNOWLEDGE_ENABLED,
    KNOWLEDGE_MAX_ANSWER_CITATIONS,
    KNOWLEDGE_MAX_CHUNKS_PER_FILE,
    KNOWLEDGE_MAX_FILE_READ_BYTES,
    KNOWLEDGE_MAX_FOLDER_SCAN_FILES,
    KNOWLEDGE_MAX_TOTAL_CHUNKS,
    KNOWLEDGE_TEXT_PREVIEW_CHARS,
    MEMORY_MAX_CONTEXT_MESSAGES,
    MEMORY_MAX_MEMORY_ITEMS,
    PROVIDER_MAX_TOKENS,
    PROVIDER_SYSTEM_PROMPT_FILES,
    PROVIDER_TEMPERATURE,
    PROVIDER_TIMEOUT_SECONDS,
    TELEGRAM_BASE_URL,
    TELEGRAM_POLL_TIMEOUT_SECONDS,
    WORKSPACE_ENABLED,
    WORKSPACE_MAX_FILES_PER_CHANGE,
    WORKSPACE_MAX_INTERNAL_READ_BYTES,
    WORKSPACE_MAX_PATH_ENTRIES,
    WORKSPACE_MAX_PREPARED_DIFF_BYTES,
    WORKSPACE_MAX_STEPS,
    WORKSPACE_SHELL_OUTPUT_CHARS,
    WORKSPACE_SHELL_TIMEOUT_SECONDS,
)


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
    max_tokens: int = PROVIDER_MAX_TOKENS
    temperature: float = PROVIDER_TEMPERATURE
    timeout_seconds: float = PROVIDER_TIMEOUT_SECONDS
    system_prompt_files: tuple[str, ...] = PROVIDER_SYSTEM_PROMPT_FILES


@dataclass(slots=True)
class TelegramConfig:
    bot_token: str = ""
    base_url: str = TELEGRAM_BASE_URL
    poll_timeout_seconds: int = TELEGRAM_POLL_TIMEOUT_SECONDS
    allowed_chat_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class DaemonConfig:
    state_dir: Path = field(default_factory=default_state_dir)
    db_path: Path = field(default_factory=lambda: default_state_dir() / "jclaw.db")
    stdout_log: Path = field(default_factory=lambda: default_log_dir() / "stdout.log")
    stderr_log: Path = field(default_factory=lambda: default_log_dir() / "stderr.log")
    launchd_label: str = DAEMON_LAUNCHD_LABEL
    idle_sleep_seconds: float = DAEMON_IDLE_SLEEP_SECONDS


@dataclass(slots=True)
class MemoryConfig:
    max_context_messages: int = MEMORY_MAX_CONTEXT_MESSAGES
    max_memory_items: int = MEMORY_MAX_MEMORY_ITEMS


@dataclass(slots=True)
class AutomationConfig:
    enabled: bool = AUTOMATION_ENABLED


@dataclass(slots=True)
class BrowserConfig:
    enabled: bool = BROWSER_ENABLED
    headless: bool = BROWSER_HEADLESS
    channel: str = BROWSER_CHANNEL
    slow_mo_ms: int = BROWSER_SLOW_MO_MS
    viewport_width: int = BROWSER_VIEWPORT_WIDTH
    viewport_height: int = BROWSER_VIEWPORT_HEIGHT
    max_objective_steps: int = BROWSER_MAX_OBJECTIVE_STEPS
    max_research_sources: int = BROWSER_MAX_RESEARCH_SOURCES


@dataclass(slots=True)
class WorkspaceConfig:
    enabled: bool = WORKSPACE_ENABLED
    max_steps: int = WORKSPACE_MAX_STEPS
    shell_timeout_seconds: int = WORKSPACE_SHELL_TIMEOUT_SECONDS
    shell_output_chars: int = WORKSPACE_SHELL_OUTPUT_CHARS
    max_prepared_diff_bytes: int = WORKSPACE_MAX_PREPARED_DIFF_BYTES
    max_files_per_change: int = WORKSPACE_MAX_FILES_PER_CHANGE
    max_path_entries: int = WORKSPACE_MAX_PATH_ENTRIES
    max_internal_read_bytes: int = WORKSPACE_MAX_INTERNAL_READ_BYTES


@dataclass(slots=True)
class KnowledgeConfig:
    enabled: bool = KNOWLEDGE_ENABLED
    max_file_read_bytes: int = KNOWLEDGE_MAX_FILE_READ_BYTES
    max_folder_scan_files: int = KNOWLEDGE_MAX_FOLDER_SCAN_FILES
    max_chunks_per_file: int = KNOWLEDGE_MAX_CHUNKS_PER_FILE
    max_total_chunks: int = KNOWLEDGE_MAX_TOTAL_CHUNKS
    text_preview_chars: int = KNOWLEDGE_TEXT_PREVIEW_CHARS
    max_answer_citations: int = KNOWLEDGE_MAX_ANSWER_CITATIONS


@dataclass(slots=True)
class Config:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
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
max_tokens = {PROVIDER_MAX_TOKENS}
temperature = {PROVIDER_TEMPERATURE}
timeout_seconds = {int(PROVIDER_TIMEOUT_SECONDS)}
system_prompt_files = ["SOUL.md", "IDENTITY.md", "BOOTSTRAP.md", "CLAUDE.md"]

[telegram]
bot_token = ""
base_url = "{TELEGRAM_BASE_URL}"
poll_timeout_seconds = {TELEGRAM_POLL_TIMEOUT_SECONDS}
allowed_chat_ids = []

[daemon]
state_dir = "{state_dir}"
db_path = "{state_dir / "jclaw.db"}"
stdout_log = "{default_log_dir() / "stdout.log"}"
stderr_log = "{default_log_dir() / "stderr.log"}"
launchd_label = "{DAEMON_LAUNCHD_LABEL}"
idle_sleep_seconds = {DAEMON_IDLE_SLEEP_SECONDS}

[memory]
max_context_messages = {MEMORY_MAX_CONTEXT_MESSAGES}
max_memory_items = {MEMORY_MAX_MEMORY_ITEMS}

[automation]
enabled = {str(AUTOMATION_ENABLED).lower()}

[browser]
enabled = {str(BROWSER_ENABLED).lower()}
headless = {str(BROWSER_HEADLESS).lower()}
channel = "{BROWSER_CHANNEL}"
slow_mo_ms = {BROWSER_SLOW_MO_MS}
viewport_width = {BROWSER_VIEWPORT_WIDTH}
viewport_height = {BROWSER_VIEWPORT_HEIGHT}
max_objective_steps = {BROWSER_MAX_OBJECTIVE_STEPS}
max_research_sources = {BROWSER_MAX_RESEARCH_SOURCES}

[workspace]
enabled = {str(WORKSPACE_ENABLED).lower()}
max_steps = {WORKSPACE_MAX_STEPS}
shell_timeout_seconds = {WORKSPACE_SHELL_TIMEOUT_SECONDS}
shell_output_chars = {WORKSPACE_SHELL_OUTPUT_CHARS}
max_prepared_diff_bytes = {WORKSPACE_MAX_PREPARED_DIFF_BYTES}
max_files_per_change = {WORKSPACE_MAX_FILES_PER_CHANGE}
max_path_entries = {WORKSPACE_MAX_PATH_ENTRIES}
max_internal_read_bytes = {WORKSPACE_MAX_INTERNAL_READ_BYTES}

[knowledge]
enabled = {str(KNOWLEDGE_ENABLED).lower()}
max_file_read_bytes = {KNOWLEDGE_MAX_FILE_READ_BYTES}
max_folder_scan_files = {KNOWLEDGE_MAX_FOLDER_SCAN_FILES}
max_chunks_per_file = {KNOWLEDGE_MAX_CHUNKS_PER_FILE}
max_total_chunks = {KNOWLEDGE_MAX_TOTAL_CHUNKS}
text_preview_chars = {KNOWLEDGE_TEXT_PREVIEW_CHARS}
max_answer_citations = {KNOWLEDGE_MAX_ANSWER_CITATIONS}
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
    automation_data = data.get("automation", {})
    browser_data = data.get("browser", {})
    workspace_data = data.get("workspace", {})
    knowledge_data = data.get("knowledge", {})

    provider = ProviderConfig(
        api_key=str(os.environ.get("JCLAW_API_KEY", provider_data.get("api_key", ""))),
        base_url=str(os.environ.get("JCLAW_BASE_URL", provider_data.get("base_url", ""))),
        model=str(os.environ.get("JCLAW_MODEL", provider_data.get("model", ""))),
        max_tokens=int(provider_data.get("max_tokens", PROVIDER_MAX_TOKENS)),
        temperature=float(provider_data.get("temperature", PROVIDER_TEMPERATURE)),
        timeout_seconds=float(provider_data.get("timeout_seconds", PROVIDER_TIMEOUT_SECONDS)),
        system_prompt_files=tuple(
            provider_data.get("system_prompt_files", PROVIDER_SYSTEM_PROMPT_FILES)
        ),
    )
    telegram = TelegramConfig(
        bot_token=str(os.environ.get("JCLAW_TELEGRAM_BOT_TOKEN", telegram_data.get("bot_token", ""))),
        base_url=str(telegram_data.get("base_url", TELEGRAM_BASE_URL)),
        poll_timeout_seconds=int(telegram_data.get("poll_timeout_seconds", TELEGRAM_POLL_TIMEOUT_SECONDS)),
        allowed_chat_ids=_env_list("JCLAW_TELEGRAM_ALLOWED_CHAT_IDS")
        or tuple(str(value) for value in telegram_data.get("allowed_chat_ids", [])),
    )
    daemon = DaemonConfig(
        state_dir=_expand_path(daemon_data.get("state_dir"), default_state_dir()),
        db_path=_expand_path(daemon_data.get("db_path"), default_state_dir() / "jclaw.db"),
        stdout_log=_expand_path(daemon_data.get("stdout_log"), default_log_dir() / "stdout.log"),
        stderr_log=_expand_path(daemon_data.get("stderr_log"), default_log_dir() / "stderr.log"),
        launchd_label=str(daemon_data.get("launchd_label", DAEMON_LAUNCHD_LABEL)),
        idle_sleep_seconds=float(daemon_data.get("idle_sleep_seconds", DAEMON_IDLE_SLEEP_SECONDS)),
    )
    memory = MemoryConfig(
        max_context_messages=int(memory_data.get("max_context_messages", MEMORY_MAX_CONTEXT_MESSAGES)),
        max_memory_items=int(memory_data.get("max_memory_items", MEMORY_MAX_MEMORY_ITEMS)),
    )
    automation = AutomationConfig(
        enabled=bool(automation_data.get("enabled", AUTOMATION_ENABLED)),
    )
    browser = BrowserConfig(
        enabled=bool(browser_data.get("enabled", BROWSER_ENABLED)),
        headless=bool(browser_data.get("headless", BROWSER_HEADLESS)),
        channel=str(browser_data.get("channel", BROWSER_CHANNEL)),
        slow_mo_ms=int(browser_data.get("slow_mo_ms", BROWSER_SLOW_MO_MS)),
        viewport_width=int(browser_data.get("viewport_width", BROWSER_VIEWPORT_WIDTH)),
        viewport_height=int(browser_data.get("viewport_height", BROWSER_VIEWPORT_HEIGHT)),
        max_objective_steps=int(browser_data.get("max_objective_steps", BROWSER_MAX_OBJECTIVE_STEPS)),
        max_research_sources=int(browser_data.get("max_research_sources", BROWSER_MAX_RESEARCH_SOURCES)),
    )
    workspace = WorkspaceConfig(
        enabled=bool(workspace_data.get("enabled", WORKSPACE_ENABLED)),
        max_steps=int(workspace_data.get("max_steps", WORKSPACE_MAX_STEPS)),
        shell_timeout_seconds=int(workspace_data.get("shell_timeout_seconds", WORKSPACE_SHELL_TIMEOUT_SECONDS)),
        shell_output_chars=int(workspace_data.get("shell_output_chars", WORKSPACE_SHELL_OUTPUT_CHARS)),
        max_prepared_diff_bytes=int(
            workspace_data.get("max_prepared_diff_bytes", WORKSPACE_MAX_PREPARED_DIFF_BYTES)
        ),
        max_files_per_change=int(workspace_data.get("max_files_per_change", WORKSPACE_MAX_FILES_PER_CHANGE)),
        max_path_entries=int(workspace_data.get("max_path_entries", WORKSPACE_MAX_PATH_ENTRIES)),
        max_internal_read_bytes=int(
            workspace_data.get("max_internal_read_bytes", WORKSPACE_MAX_INTERNAL_READ_BYTES)
        ),
    )
    knowledge = KnowledgeConfig(
        enabled=bool(knowledge_data.get("enabled", KNOWLEDGE_ENABLED)),
        max_file_read_bytes=int(
            knowledge_data.get("max_file_read_bytes", KNOWLEDGE_MAX_FILE_READ_BYTES)
        ),
        max_folder_scan_files=int(
            knowledge_data.get("max_folder_scan_files", KNOWLEDGE_MAX_FOLDER_SCAN_FILES)
        ),
        max_chunks_per_file=int(
            knowledge_data.get("max_chunks_per_file", KNOWLEDGE_MAX_CHUNKS_PER_FILE)
        ),
        max_total_chunks=int(
            knowledge_data.get("max_total_chunks", KNOWLEDGE_MAX_TOTAL_CHUNKS)
        ),
        text_preview_chars=int(
            knowledge_data.get("text_preview_chars", KNOWLEDGE_TEXT_PREVIEW_CHARS)
        ),
        max_answer_citations=int(
            knowledge_data.get("max_answer_citations", KNOWLEDGE_MAX_ANSWER_CITATIONS)
        ),
    )

    config = Config(
        provider=provider,
        telegram=telegram,
        daemon=daemon,
        memory=memory,
        automation=automation,
        browser=browser,
        workspace=workspace,
        knowledge=knowledge,
        config_path=config_path,
        repo_root=repo_root(),
    )
    config.ensure_runtime_dirs()
    return config
