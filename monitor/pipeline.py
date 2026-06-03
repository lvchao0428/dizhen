"""监控 pipeline：检测 →（赛题有效则）序列+预测 → 按震级微信通知。"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from monitor.competition import is_competition_eligible
from monitor.config import get_config
from monitor.deadlines import check_deadlines, hours_since_mainshock
from monitor.detector import detect_new_events
from monitor.live_predict import run_prediction
from monitor.model_info import get_model_status
from monitor.notifier import format_new_event_message, notify
from monitor.sources import usgs
from monitor.store import EventStore
from monitor.time_utils import normalize_event

logger = logging.getLogger(__name__)


def setup_logging(cfg: Dict[str, Any]) -> None:
    log_dir = os.path.join(cfg["data_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "monitor.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _should_notify(ev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    return hours_since_mainshock(ev) <= cfg.get("notify_max_age_hours", 48)


def run_competition_pipeline(
    ev: Dict[str, Any], cfg: Dict[str, Any], models_bundle: Optional[Tuple] = None
) -> Dict[str, Any]:
    from monitor.window_compare import (
        build_window_comparison,
        format_comparison_markdown,
        format_comparison_wechat,
        save_comparison,
    )

    seq_df = usgs.fetch_sequence(
        mainshock_utc=ev["mainshock_utc"],
        lon=ev["lon"],
        lat=ev["lat"],
        radius_km=cfg.get("sequence_radius_km", 300),
        hours=cfg.get("sequence_hours", 168),
        mainshock_mag=float(ev["mag"]),
        mainshock_mag_type=str(ev.get("mag_type", "mw")),
    )
    usgs.save_sequence(seq_df, ev["event_id"], cfg["sequences_dir"])
    logger.info("序列 %s: %d 行", ev["event_id"], len(seq_df))

    pred_result = run_prediction(ev, cfg, models_bundle=models_bundle)
    comp = build_window_comparison(ev, cfg, predictions=pred_result["predictions"], seq_df=seq_df)
    comp_path = save_comparison(comp, pred_result["output_dir"])
    pred_result["window_comparison"] = comp
    pred_result["comparison_md"] = format_comparison_markdown(comp)
    pred_result["comparison_wechat"] = format_comparison_wechat(
        comp, model_ready=get_model_status(cfg)["ready"]
    )
    pred_result["comparison_path"] = comp_path
    return pred_result


def _notify_competition_ready(ev: Dict[str, Any], cfg: Dict[str, Any], pred_result: Dict) -> None:
    from monitor.notify_body import format_competition_ready_message

    notify(cfg, f"[余震赛] 已出CSV {ev['event_id']}", format_competition_ready_message(ev, cfg, pred_result))


def process_new_event(
    ev: Dict[str, Any], store: EventStore, cfg: Dict[str, Any], models_bundle: Optional[Tuple] = None
) -> None:
    ev = normalize_event(ev)
    ev["competition_eligible"] = ev.get("competition_eligible", is_competition_eligible(ev, cfg))
    ev["output_dir"] = os.path.join(cfg["ranking_output_dir"], ev["event_id"])

    if not store.insert_event(ev):
        return

    eligible = ev["competition_eligible"]
    do_notify = _should_notify(ev, cfg)
    pred_result = None

    if eligible:
        try:
            pred_result = run_competition_pipeline(ev, cfg, models_bundle=models_bundle)
            store.update_status(ev["event_id"], "predicted_t1t2", pred_result["output_dir"])
            if do_notify:
                _notify_competition_ready(ev, cfg, pred_result)
        except Exception as e:
            logger.error("赛题流水线失败 %s: %s", ev["event_id"], e)
            store.update_status(ev["event_id"], "pipeline_failed")
    else:
        store.update_status(ev["event_id"], "notify_only")

    if do_notify:
        title = f"[余震监测] M{ev['mag']:.1f}" if not eligible else f"[余震赛] M{ev['mag']:.1f}"
        notify(cfg, title, format_new_event_message(ev, cfg, pred_result))
        row = store.get_event(ev["event_id"])
        if row and row["status"] in ("new", "notify_only", "pipeline_failed"):
            store.update_status(ev["event_id"], "notified" if not eligible else row["status"])
    else:
        store.update_status(ev["event_id"], "logged")


def refresh_event(ev: Dict[str, Any], store: EventStore, cfg: Dict[str, Any], kind: str) -> None:
    ev = normalize_event(ev)
    if not ev.get("competition_eligible"):
        return
    try:
        pred_result = run_competition_pipeline(ev, cfg)
        store.update_status(ev["event_id"], "predicted_t3", pred_result["output_dir"])
        notify(
            cfg,
            f"[余震赛] 刷新 {ev['event_id']}",
            f"## 预测已刷新\n{pred_result.get('comparison_wechat', '')}\n\n`{pred_result['output_dir']}`",
        )
    except Exception as e:
        logger.error("刷新失败 %s: %s", ev["event_id"], e)


def run_pipeline(models_bundle: Optional[Tuple] = None) -> Dict[str, Any]:
    cfg = get_config()
    setup_logging(cfg)
    store = EventStore(os.path.join(cfg["data_dir"], "events.sqlite"))
    os.makedirs(cfg["sequences_dir"], exist_ok=True)
    os.makedirs(cfg["ranking_output_dir"], exist_ok=True)

    new_events = detect_new_events(cfg)
    processed = []
    for ev in new_events:
        if store.get_event(ev["event_id"]) is None:
            process_new_event(ev, store, cfg, models_bundle=models_bundle)
            processed.append(ev["event_id"])

    reminders = check_deadlines(
        store,
        cfg,
        on_refresh=lambda e, k: refresh_event(normalize_event(e), store, cfg, k),
    )
    return {
        "checked": len(new_events),
        "new_processed": processed,
        "reminders": reminders,
        "at": datetime.now(timezone.utc).isoformat(),
    }
