from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sqlite3

from jclaw.ai.agent import AssistantAgent
from jclaw.ai.client import OpenAICompatibleClient
from jclaw.channel.fake import LocalChannelHarness
from jclaw.channel.telegram import TelegramBotChannel
from jclaw.core.config import Config, default_config_path, load_config, render_default_config
from jclaw.core.db import Database
from jclaw.core.logging import configure_logging
from jclaw.daemon.launchd import install_launch_agent, launch_agent_path, uninstall_launch_agent
from jclaw.daemon.service import JClawDaemon


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="jclaw")
    parser.add_argument("--config", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_config_parser = subparsers.add_parser("init-config")
    init_config_parser.add_argument("--config", default=None)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--config", default=None)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default=None)

    install_launchd_parser = subparsers.add_parser("install-launchd")
    install_launchd_parser.add_argument("--config", default=None)

    uninstall_launchd_parser = subparsers.add_parser("uninstall-launchd")
    uninstall_launchd_parser.add_argument("--config", default=None)

    send = subparsers.add_parser("send")
    send.add_argument("--config", default=None)
    send.add_argument("message")
    send.add_argument("--chat-id", default="local-cli")
    send.add_argument("--user-name", default="cli")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config or str(default_config_path())

    if args.command == "init-config":
        return init_config(Path(config_path))

    config = load_config(config_path)
    configure_logging(config.daemon.stdout_log)

    if args.command == "doctor":
        return doctor(config)
    if args.command == "run":
        daemon = JClawDaemon(config)
        daemon.run_forever()
        return 0
    if args.command == "install-launchd":
        path = install_launch_agent(config)
        print(f"installed launch agent at {path}")
        return 0
    if args.command == "uninstall-launchd":
        path = uninstall_launch_agent(config)
        print(f"removed launch agent {path}")
        return 0
    if args.command == "send":
        return send_message(config, args)

    parser.error(f"unsupported command: {args.command}")
    return 2


def init_config(path: Path) -> int:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"config already exists at {path}")
        return 0
    path.write_text(render_default_config(), encoding="utf-8")
    print(f"wrote starter config to {path}")
    return 0


def doctor(config: Config) -> int:
    problems: list[str] = []

    if not config.telegram.bot_token:
        problems.append("telegram.bot_token is missing")
    if not config.provider.base_url:
        problems.append("provider.base_url is missing")
    if not config.provider.model:
        problems.append("provider.model is missing")

    db = Database(config.daemon.db_path)
    db.close()
    print(f"sqlite ok: {config.daemon.db_path}")

    if config.telegram.bot_token:
        try:
            channel = TelegramBotChannel(config.telegram)
            profile = channel.validate_token()
            channel.close()
            print(f"telegram ok: @{profile.get('username', 'unknown')}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"telegram check failed: {exc}")

    if config.provider.base_url:
        try:
            client = OpenAICompatibleClient(config.provider)
            status = client.health_check()
            client.close()
            print(f"provider status: {status}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"provider check failed: {exc}")

    print(f"launch agent path: {launch_agent_path(config.daemon.launchd_label)}")

    if problems:
        for problem in problems:
            print(f"doctor: {problem}")
        return 1

    print("doctor: ok")
    return 0


def send_message(config: Config, args: Namespace) -> int:
    db = Database(config.daemon.db_path)
    client = OpenAICompatibleClient(config.provider)
    try:
        agent = AssistantAgent(config, db, client)
        harness = LocalChannelHarness(agent)
        reply = harness.send(args.chat_id, args.message, user_name=args.user_name)
        print(reply)
        return 0
    finally:
        client.close()
        db.close()
