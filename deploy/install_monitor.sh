#!/usr/bin/env bash
# 在远程服务器安装余震排名赛监控服务
set -euo pipefail

ROOT="${INSTALL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

echo "==> 仓库目录: $ROOT"

if [[ ! -d .venv ]]; then
  echo "==> 创建虚拟环境"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 安装依赖"
pip install -q -U pip
pip install -q -r solution/requirements.txt
pip install -q -r monitor/requirements.txt

if [[ ! -f solution/models_v3/models.pkl ]]; then
  echo "==> 训练 v3 模型（首次运行，约 1 分钟）"
  cd solution
  python predict_v3.py
  cd "$ROOT"
else
  echo "==> 模型已存在: solution/models_v3/models.pkl"
fi

if [[ ! -f monitor/config.yaml ]]; then
  echo "==> 复制 monitor/config.yaml"
  cp monitor/config.example.yaml monitor/config.yaml
fi

if [[ ! -f monitor/.env ]]; then
  echo "!! 请创建 monitor/.env 并设置 SERVERCHAN_SENDKEY"
  cp monitor/.env.example monitor/.env
fi

mkdir -p monitor/data/sequences monitor/ranking_output monitor/data/logs

echo "==> 试跑一轮 pipeline"
python -m monitor.run_once || true

SERVICE_SRC="$ROOT/deploy/dizhen-monitor.service"
SERVICE_DST="/etc/systemd/system/dizhen-monitor.service"

if [[ -w /etc/systemd/system ]] || [[ "$(id -u)" -eq 0 ]]; then
  echo "==> 安装 systemd 服务"
  sed "s|/home/charlie/project/dizhen|$ROOT|g" "$SERVICE_SRC" > /tmp/dizhen-monitor.service
  cp /tmp/dizhen-monitor.service "$SERVICE_DST"
  systemctl daemon-reload
  systemctl enable dizhen-monitor.service
  echo "==> 启动: sudo systemctl start dizhen-monitor"
  echo "==> 日志: journalctl -u dizhen-monitor -f"
else
  echo "==> 无 root 权限，跳过 systemd。可手动运行:"
  echo "    cd $ROOT && .venv/bin/python -m monitor.run_daemon"
fi

echo "==> 安装完成"
