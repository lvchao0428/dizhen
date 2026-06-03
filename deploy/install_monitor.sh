#!/bin/bash
set -euo pipefail

ROOT="${INSTALL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

echo "==> $ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -U pip
pip install -q -r solution/requirements.txt -r monitor/requirements.txt

if [[ ! -f solution/models_v3/models.pkl ]]; then
  echo "==> 训练 v3（约 1-2 分钟）"
  (cd solution && python predict_v3.py)
fi

[[ -f monitor/config.yaml ]] || cp monitor/config.example.yaml monitor/config.yaml
[[ -f monitor/.env ]] || cp monitor/.env.example monitor/.env

mkdir -p monitor/data/sequences monitor/ranking_output monitor/data/logs monitor/data/submissions

python -m monitor.run_once || true

TMP_UNIT="/tmp/dizhen-monitor.service"
sed "s|/home/charlie/project/dizhen|$ROOT|g" "$ROOT/deploy/dizhen-monitor.service" > "$TMP_UNIT"

if [[ "$(id -u)" -eq 0 ]]; then
  cp "$TMP_UNIT" /etc/systemd/system/dizhen-monitor.service
  systemctl daemon-reload && systemctl enable dizhen-monitor && systemctl restart dizhen-monitor
elif command -v sudo >/dev/null; then
  sudo cp "$TMP_UNIT" /etc/systemd/system/dizhen-monitor.service
  sudo systemctl daemon-reload && sudo systemctl enable dizhen-monitor && sudo systemctl restart dizhen-monitor
else
  echo "无 sudo，请手动: .venv/bin/python -m monitor.run_daemon"
fi

echo "==> 完成。日志: journalctl -u dizhen-monitor -f"
