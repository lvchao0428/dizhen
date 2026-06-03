#!/usr/bin/env python3
"""常驻：5 分钟轮询 + 每日 9:00 日报（北京时间）。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from monitor.config import get_config
from monitor.digest import run_digest
from monitor.pipeline import run_pipeline, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    tz = cfg.get("digest_timezone", "Asia/Shanghai")
    sched = BlockingScheduler(timezone=tz)
    sched.add_job(run_pipeline, "interval", minutes=int(cfg.get("poll_interval_minutes", 5)), id="monitor")
    sched.add_job(
        lambda: run_digest(notify_user=True, backfill=bool(cfg.get("digest_backfill_on_daily", True))),
        "cron",
        hour=int(cfg.get("digest_cron_hour", 9)),
        minute=int(cfg.get("digest_cron_minute", 0)),
        id="digest",
    )
    logger.info("启动: 轮询 %s min, 日报 %02d:%02d %s", cfg.get("poll_interval_minutes"), cfg.get("digest_cron_hour", 9), cfg.get("digest_cron_minute", 0), tz)
    run_pipeline()
    sched.start()


if __name__ == "__main__":
    main()
