"""加载 monitor 配置与环境变量。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

MONITOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = MONITOR_DIR.parent


def _load_yaml() -> Dict[str, Any]:
    for name in ("config.yaml", "config.example.yaml"):
        path = MONITOR_DIR / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def get_config() -> Dict[str, Any]:
    load_dotenv(MONITOR_DIR / ".env")
    cfg = _load_yaml()
    defaults = {
        "poll_interval_minutes": 5,
        "lookback_days": 7,
        "notify_max_age_hours": 48,
        "min_magnitude_submit": 6.0,
        "min_magnitude_watch": 5.8,
        "usgs_fetch_max_depth_km": 1000,
        "sequence_radius_km": 300,
        "sequence_hours": 168,
        "model_path": "solution/models_v3/models.pkl",
        "ranking_output_dir": "monitor/ranking_output",
        "data_dir": "monitor/data",
        "sequences_dir": "monitor/data/sequences",
        "tianchi_url": "https://tianchi.aliyun.com/competition/entrance/532460/submission",
        "serverchan_sendkey": "",
        "remind_t12_urgent_hours": 20,
        "remind_t3_prep_hours": 48,
        "remind_t3_urgent_hours": 68,
        "refresh_t2_hours": 23,
        "refresh_t3_hours": 71,
        "digest_lookback_days": 30,
        "digest_timezone": "Asia/Shanghai",
        "digest_cron_hour": 9,
        "digest_cron_minute": 0,
        "digest_backfill_on_daily": True,
        "digest_max_chars": 28000,
    }
    merged = {**defaults, **cfg}
    for key in ("model_path", "ranking_output_dir", "data_dir", "sequences_dir"):
        p = Path(merged[key])
        if not p.is_absolute():
            merged[key] = str(REPO_ROOT / p)
    merged["serverchan_sendkey"] = (
        os.environ.get("SERVERCHAN_SENDKEY") or merged.get("serverchan_sendkey") or ""
    )
    merged["repo_root"] = str(REPO_ROOT)
    return merged
