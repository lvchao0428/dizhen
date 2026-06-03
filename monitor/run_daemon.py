#!/usr/bin/env python3
"""APScheduler 常驻监控服务。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitor.config import get_config
from monitor.pipeline import run_pipeline, setup_logging

logger = logging.getLogger(__name__)


def job() -> None:
    try:
        result = run_pipeline()
        logger.info(
            "pipeline 完成: checked=%s new=%s reminders=%s",
            result.get("checked"),
            result.get("new_processed"),
            result.get("reminders"),
        )
    except Exception:
        logger.exception("pipeline 执行失败")


def main() -> None:
    cfg = get_config()
    setup_logging(cfg)
    interval = int(cfg.get("poll_interval_minutes", 5))

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(job, "interval", minutes=interval, id="eq_monitor")
    logger.info("余震排名赛监控启动，轮询间隔 %d 分钟", interval)
    job()
    scheduler.start()


if __name__ == "__main__":
    main()
