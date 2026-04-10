from __future__ import annotations

from pathlib import Path
import os
import plistlib
import subprocess
import sys

from jclaw.core.config import Config


def launch_agent_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def build_plist(config: Config) -> bytes:
    program_arguments = [
        sys.executable,
        str(config.repo_root / "jclaw.py"),
        "--config",
        str(config.config_path),
        "run",
    ]
    payload = {
        "Label": config.daemon.launchd_label,
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(config.repo_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(config.daemon.stdout_log),
        "StandardErrorPath": str(config.daemon.stderr_log),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
        },
    }
    return plistlib.dumps(payload)


def install_launch_agent(config: Config) -> Path:
    path = launch_agent_path(config.daemon.launchd_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_plist(config))
    _launchctl_bootout(path, label=config.daemon.launchd_label)
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{config.daemon.launchd_label}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def uninstall_launch_agent(config: Config) -> Path:
    path = launch_agent_path(config.daemon.launchd_label)
    _launchctl_bootout(path, label=config.daemon.launchd_label)
    if path.exists():
        path.unlink()
    return path


def _launchctl_bootout(path: Path, *, label: str) -> None:
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
