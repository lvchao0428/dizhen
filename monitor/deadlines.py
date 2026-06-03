"""提交截止与刷新提醒。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from monitor.notifier import notify, pd_timestamp_bj

logger = logging.getLogger(__name__)


def _parse_ms(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_mainshock(ev: Dict[str, Any]) -> float:
    from monitor.time_utils import to_utc_timestamp
    ms = to_utc_timestamp(ev["mainshock_utc"])
    now = pd.Timestamp.now(tz="UTC")
    return (now - ms).total_seconds() / 3600


def check_deadlines(
    store,
    cfg: Dict[str, Any],
    on_refresh=None,
) -> List[str]:
    """
    检查活跃事件的截止提醒与自动刷新。
    on_refresh(ev, kind) 可选回调：kind 为 t2|t3。
    返回已发送提醒类型列表。
    """
    sent = []
    for ev in store.list_active_events():
        if not ev.get("competition_eligible"):
            continue
        eid = ev["event_id"]
        h = hours_since_mainshock(ev)

        # T1-T2 紧急（20h）
        if h >= cfg.get("remind_t12_urgent_hours", 20) and not ev.get("reminded_t12_urgent"):
            left = max(0, 24 - h)
            desp = (
                f"## T1-T2 提交紧急提醒\n"
                f"- 事件: `{eid}` M{ev['mag']:.1f}\n"
                f"- 剩余约 **{left:.1f}h** 提交 T1-T2\n"
                f"- 输出: `{ev.get('output_dir', '')}`\n"
                f"- {cfg.get('tianchi_url', '')}"
            )
            if notify(cfg, f"[余震赛] T1-T2 还剩{left:.0f}h", desp):
                store.set_flag(eid, "reminded_t12_urgent")
                store.mark_reminder(eid, "t12_urgent")
                sent.append(f"{eid}:t12_urgent")

        # T3 准备（48h）
        if h >= cfg.get("remind_t3_prep_hours", 48) and not ev.get("reminded_t3_prep"):
            desp = (
                f"## T3 准备提醒\n"
                f"- 事件: `{eid}`\n"
                f"- 建议拉取 0-72h 余震序列并刷新 T3 预测\n"
                f"- 主震后已 **{h:.1f}h**"
            )
            if notify(cfg, f"[余震赛] 准备 T3 ({eid})", desp):
                store.set_flag(eid, "reminded_t3_prep")
                store.mark_reminder(eid, "t3_prep")
                sent.append(f"{eid}:t3_prep")
            if on_refresh and not ev.get("refreshed_t3"):
                on_refresh(ev, "t3_prep")

        # T3 紧急（68h）
        if h >= cfg.get("remind_t3_urgent_hours", 68) and not ev.get("reminded_t3_urgent"):
            left = max(0, 72 - h)
            desp = (
                f"## T3 提交紧急提醒\n"
                f"- 事件: `{eid}`\n"
                f"- 剩余约 **{left:.1f}h** 提交 T3\n"
                f"- 输出: `{ev.get('output_dir', '')}`"
            )
            if notify(cfg, f"[余震赛] T3 还剩{left:.0f}h", desp):
                store.set_flag(eid, "reminded_t3_urgent")
                store.mark_reminder(eid, "t3_urgent")
                sent.append(f"{eid}:t3_urgent")

        # 自动刷新 T2（23h）
        if (
            h >= cfg.get("refresh_t2_hours", 23)
            and h < 24
            and not ev.get("refreshed_t2")
            and on_refresh
        ):
            on_refresh(ev, "t2")
            store.set_flag(eid, "refreshed_t2")
            sent.append(f"{eid}:refresh_t2")

        # 自动刷新 T3（71h）
        if h >= cfg.get("refresh_t3_hours", 71) and not ev.get("refreshed_t3") and on_refresh:
            on_refresh(ev, "t3")
            store.set_flag(eid, "refreshed_t3")
            sent.append(f"{eid}:refresh_t3")

        # 超过 72h 标记完成
        if h >= 72 and ev.get("status") != "done":
            store.update_status(eid, "done")

    return sent
