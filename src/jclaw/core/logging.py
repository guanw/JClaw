from __future__ import annotations

from pathlib import Path
import logging


class JClawConsoleFormatter(logging.Formatter):
    RESET = "\033[0m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if record.name == "jclaw.ai.agent":
            if "tool initial planner raw response:" in rendered:
                return self._highlight(rendered, "tool initial planner raw response:", self.MAGENTA)
            if "tool continuation raw response:" in rendered:
                return self._highlight(rendered, "tool continuation raw response:", self.CYAN)
            if "tool initial planner selected" in rendered:
                return self._wrap(rendered, self.YELLOW)
            if "tool continuation selected" in rendered:
                return self._wrap(rendered, self.GREEN)
        return rendered

    def _highlight(self, text: str, marker: str, color: str) -> str:
        prefix, matched, suffix = text.partition(marker)
        if not matched:
            return text
        return f"{self.DIM}{prefix}{self.RESET}{color}{matched}{self.RESET}{suffix}"

    def _wrap(self, text: str, color: str) -> str:
        return f"{color}{text}{self.RESET}"


def configure_logging(log_path: str | Path | None = None, *, verbose: bool = False) -> None:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JClawConsoleFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handlers: list[logging.Handler] = [stream_handler]
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=handlers,
        force=True,
    )
