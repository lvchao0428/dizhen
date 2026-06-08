#!/bin/bash
# 从远程拉取排名赛 CSV 到本机。用法: bash deploy/pull_ranking.sh [事件ID]
set -euo pipefail

REMOTE="${PULL_REMOTE:-charlie@charlie-ultra}"
REMOTE_ROOT="${PULL_REMOTE_ROOT:-~/project/dizhen}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVENT_ID="${1:-20260531213418}"

mkdir -p "$LOCAL_ROOT/ranking_output/$EVENT_ID"
mkdir -p "$LOCAL_ROOT/monitor/data/sequences"

echo "==> $REMOTE:$REMOTE_ROOT -> $LOCAL_ROOT"
rsync -avz \
  "$REMOTE:$REMOTE_ROOT/ranking_output/$EVENT_ID/" \
  "$LOCAL_ROOT/ranking_output/$EVENT_ID/"
rsync -avz \
  "$REMOTE:$REMOTE_ROOT/monitor/data/sequences/${EVENT_ID}_eq.csv" \
  "$LOCAL_ROOT/monitor/data/sequences/" 2>/dev/null || true

echo "==> 本地文件:"
ls -la "$LOCAL_ROOT/ranking_output/$EVENT_ID/"
echo ""
echo "检查: cd $LOCAL_ROOT && source .venv/bin/activate && python -m monitor.check_submission --event $EVENT_ID"
