#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$HOME/.config/jclaw/config.dev.toml}"

cd "$REPO_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing virtualenv at $REPO_ROOT/.venv" >&2
  echo "Run scripts/bootstrap_local_prod.sh first or create the venv manually." >&2
  exit 1
fi

exec .venv/bin/python jclaw.py run --config "$CONFIG_PATH"
