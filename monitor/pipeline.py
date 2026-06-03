"""监控 pipeline：检测 → 序列 → 预测 → 通知。"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from monitor.config import get_config
from monitor.deadlines import check_deadlines, hours_since_mainshock
from monitor.detector import detect_new_events
from monitor.live_predict import run_prediction
from monitor.notifier import format_new_event_message, notify
from monitor.sources import usgs
from monitor.store import EventStore

logger = logging.getLogger(__name__)


def setup_logging(cfg: Dict[str, Any]) -> None:
    log_dir = os.path.join(cfg["data_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "monitor.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _should_notify(ev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    max_age = cfg.get("notify_max_age_hours", 48)
    return hours_since_mainshock(ev) <= max_age


def process_new_event(
    ev: Dict[str, Any],
    store: EventStore,
    cfg: Dict[str, Any],
    models_bundle: Optional[Tuple] = None,
) -> None:
    """处理单个新检出事件。"""
    ev = dict(ev)
    ev["output_dir"] = os.path.join(cfg["ranking_output_dir"], ev["event_id"])

    if not store.insert_event(ev):
        logger.debug("事件已存在，跳过: %s", ev["event_id"])
        return

    logger.info("新事件: %s M%.1f %s", ev["event_id"], ev["mag"], ev.get("place", ""))
    do_notify = _should_notify(ev, cfg)

    # 构建序列并保存
    try:
        seq_df = usgs.fetch_sequence(
            mainshock_utc=ev["mainshock_utc"],
            lon=ev["lon"],
            lat=ev["lat"],
            radius_km=cfg.get("sequence_radius_km", 300),
            hours=cfg.get("sequence_hours", 168),
        )
        usgs.save_sequence(seq_df, ev["event_id"], cfg["sequences_dir"])
    except Exception as e:
        logger.error("序列拉取失败 %s: %s", ev["event_id"], e)

    pred_result = None
    try:
        pred_result = run_prediction(ev, cfg, models_bundle=models_bundle)
        store.update_status(ev["event_id"], "predicted_t1t2", pred_result["output_dir"])
    except Exception as e:
        logger.error("预测失败 %s: %s", ev["event_id"], e)
        if do_notify:
            store.update_status(ev["event_id"], "notified")
        else:
            store.update_status(ev["event_id"], "logged")

    if do_notify:
        title = f"[余震赛] M{ev['mag']:.1f} {ev.get('place', '')[:12]}"
        if ev.get("category") == "watch":
            title = f"[余震赛·关注] M{ev['mag']:.1f}"
        desp = format_new_event_message(ev, cfg, pred_result)
        notify(cfg, title, desp)
        row = store.get_event(ev["event_id"])
        if row and row["status"] in ("new", "logged"):
            store.update_status(ev["event_id"], "notified")
    else:
        logger.info("事件较旧，已入库但不推送: %s", ev["event_id"])
        store.update_status(ev["event_id"], "logged")


def refresh_event(ev: Dict[str, Any], store: EventStore, cfg: Dict[str, Any], kind: str) -> None:
    """重新拉序列并预测（T2/T3 刷新）。"""
    eid = ev["event_id"]
    logger.info("刷新预测 %s (%s)", eid, kind)

    try:
        seq_df = usgs.fetch_sequence(
            mainshock_utc=ev["mainshock_utc"],
            lon=ev["lon"],
            lat=ev["lat"],
            radius_km=cfg.get("sequence_radius_km", 300),
            hours=cfg.get("sequence_hours", 168),
        )
        usgs.save_sequence(seq_df, eid, cfg["sequences_dir"])
        pred_result = run_prediction(ev, cfg)
        store.update_status(eid, "predicted_t3", pred_result["output_dir"])

        desp = (
            f"## 预测已刷新 ({kind})\n"
            f"- 事件: `{eid}`\n"
            f"- 主震后: {hours_since_mainshock(ev):.1f}h\n\n"
            f"{pred_result.get('summary', '')}\n\n"
            f"目录: `{pred_result['output_dir']}`"
        )
        notify(cfg, f"[余震赛] 刷新 {eid}", desp)
    except Exception as e:
        logger.error("刷新失败 %s: %s", eid, e)


def run_pipeline(models_bundle: Optional[Tuple] = None) -> Dict[str, Any]:
    """执行一轮完整 pipeline。"""
    cfg = get_config()
    setup_logging(cfg)

    db_path = os.path.join(cfg["data_dir"], "events.sqlite")
    store = EventStore(db_path)

    os.makedirs(cfg["sequences_dir"], exist_ok=True)
    os.makedirs(cfg["ranking_output_dir"], exist_ok=True)

    new_events = detect_new_events(cfg)
    processed = []
    for ev in new_events:
        existing = store.get_event(ev["event_id"])
        if existing is None:
            process_new_event(ev, store, cfg, models_bundle=models_bundle)
            processed.append(ev["event_id"])

    def on_refresh(ev, kind):
        refresh_event(ev, store, cfg, kind)

    reminders = check_deadlines(store, cfg, on_refresh=on_refresh)

    return {
        "checked": len(new_events),
        "new_processed": processed,
        "reminders": reminders,
        "at": datetime.now(timezone.utc).isoformat(),
    }
