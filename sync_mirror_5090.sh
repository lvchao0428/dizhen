#!/usr/bin/env bash
# Sync the current directory to the 5090 training box (RTX 5090).
# Audio/video-style files (mp4, gif, mp3, etc.) are excluded from rsync.
# Default: charlie@www.ultrapp.online:/home/charlie/project/dizhen/
#
# Requires: rsync on both ends (install: apt install rsync / brew install rsync).
# Pure scp cannot mirror cleanly (no --delete, slow); rsync over ssh is the usual approach.
#
# Usage:
#   ./sync_mirror_5090.sh
#   ./sync_mirror_5090.sh --dry-run
#   REMOTE_HOST=other.host ./sync_mirror_5090.sh
#   SYNC_ROOT=/path/to/repo ./sync_mirror_5090.sh

set -euo pipefail

ROOT="${SYNC_ROOT:-$(pwd)}"
ROOT="$(cd "$ROOT" && pwd)"

REMOTE_USER="${REMOTE_USER:-charlie}"
REMOTE_HOST="${REMOTE_HOST:-www.ultrapp.online}"
REMOTE_DIR="${REMOTE_DIR:-/home/charlie/project/dizhen}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    *)
      echo "Unknown option: $arg (only --dry-run / -n supported)" >&2
      exit 1
      ;;
  esac
done

RSYNC=(rsync -avz)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC+=(--dry-run)
fi

# Mirror: drop remote files not present locally (careful if you store extra files only on server).
RSYNC+=(--delete)

RSYNC+=(
  --human-readable
  --exclude ".venv/"
  --exclude "__pycache__/"
  --exclude "*.py[cod]"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".ruff_cache/"
  --exclude "*.egg-info/"
  --exclude ".DS_Store"
  --exclude "dist/*.tar.gz"
  # Training / generated artifacts: keep remote-only copies.
  --exclude "solution/models_v*/"
  --exclude "solution/output_v*/"
  --exclude "logs/"
  --exclude "data/*.npz"
  --exclude "*.pkl"
  --exclude "*.log"
  --exclude "*.zip"
  # Audio / video (do not mirror large media)
  --exclude "*.mp4"
  --exclude "*.webm"
  --exclude "*.avi"
  --exclude "*.mov"
  --exclude "*.mkv"
  --exclude "*.m4v"
  --exclude "*.wmv"
  --exclude "*.flv"
  --exclude "*.mpg"
  --exclude "*.mpeg"
  --exclude "*.3gp"
  --exclude "*.ogv"
  --exclude "*.mp3"
  --exclude "*.wav"
  --exclude "*.flac"
  --exclude "*.aac"
  --exclude "*.m4a"
  --exclude "*.ogg"
  --exclude "*.opus"
  --exclude "*.wma"
  --exclude "*.gif"
)

echo "Sync: $ROOT/ -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
"${RSYNC[@]}" "$ROOT/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
