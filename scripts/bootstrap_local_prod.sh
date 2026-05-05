#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
CONFIG_PATH="${CONFIG_PATH:-$HOME/.config/jclaw/config.toml}"
PYTHON_BIN="$VENV_DIR/bin/python"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_local_prod.sh [--skip-browser] [--restart-only]

Bootstraps or refreshes a local JClaw production install:
  - creates .venv if needed
  - installs JClaw into the venv
  - optionally installs Playwright Chromium
  - runs doctor
  - installs or restarts the launchd agent

Environment overrides:
  CONFIG_PATH=/path/to/config.toml
  VENV_DIR=/path/to/venv
EOF
}

install_browser=1
restart_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-browser)
      install_browser=0
      shift
      ;;
    --restart-only)
      restart_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  echo "Create it first with: jclaw init-config" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ $restart_only -eq 0 ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    uv venv "$VENV_DIR"
  fi

  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e ".[dev]"

  if [[ $install_browser -eq 1 ]]; then
    "$PYTHON_BIN" -m playwright install chromium
  fi

  "$PYTHON_BIN" jclaw.py doctor --config "$CONFIG_PATH"
  "$PYTHON_BIN" jclaw.py install-launchd --config "$CONFIG_PATH"
else
  "$PYTHON_BIN" jclaw.py install-launchd --config "$CONFIG_PATH"
fi

echo
echo "JClaw launchd service refreshed."
echo "Config: $CONFIG_PATH"
echo "Logs:"
echo "  tail -f \"$HOME/Library/Logs/JClaw/stdout.log\""
echo "  tail -f \"$HOME/Library/Logs/JClaw/stderr.log\""
echo "Status:"
echo "  launchctl print gui/$(id -u)/com.jclaw.daemon"
