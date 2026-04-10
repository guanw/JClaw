#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  .venv/bin/python jclaw.py install-launchd "$@"
else
  python3 jclaw.py install-launchd "$@"
fi
