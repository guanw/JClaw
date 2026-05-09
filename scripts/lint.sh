#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
  ARGS=("${ROOT_DIR}/src")
fi

if [ -x "$PYTHON_BIN" ]; then
  exec "$PYTHON_BIN" -m ruff check "${ARGS[@]}"
fi

exec python -m ruff check "${ARGS[@]}"
