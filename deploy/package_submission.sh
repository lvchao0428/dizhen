#!/bin/bash
set -euo pipefail
ROOT="${INSTALL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
[[ -d .venv ]] && source .venv/bin/activate
EVENT_ID="${1:-}"
if [[ -n "$EVENT_ID" ]]; then
  python -m monitor.package_submission --event "$EVENT_ID"
else
  python -m monitor.package_submission
fi
