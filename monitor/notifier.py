"""Server酱微信推送。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

SERVERCHAN_URL = "https://sctapi.ftqq.com/{sendkey}.send"


def send_serverchan(
    sendkey: str,
    title: str,
    desp: str,
    short: Optional[str] = None,
) -> bool:
    if not sendkey:
        logger.warning("未配置 SERVERCHAN_SENDKEY，跳过通知: %s", title)
        return False

    url = SERVERCHAN_URL.format(sendkey=sendkey.strip())
    payload = {"title": title[:32], "desp": desp}
    if short:
        payload["short"] = short[:64]

    try:
        resp = requests.post(url, data=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0 or data.get("data", {}).get("errno") == 0:
            logger.info("Server酱推送成功: %s", title)
            return True
        logger.error("Server酱返回异常: %s", data)
        return False
    except Exception as e:
        logger.error("Server酱推送失败: %s", e)
        return False


def format_new_event_message(ev: Dict[str, Any], cfg: Dict[str, Any], pred_result: Optional[Dict] = None) -> str:
    ms = ev["mainshock_utc"]
    if hasattr(ms, "strftime"):
        utc_str = ms.strftime("%Y-%m-%d %H:%M:%S UTC")
        bj = pd_timestamp_bj(ms)
    else:
        utc_str = str(ms)
        bj = utc_str

    cat = ev.get("category", "submit")
    cat_note = "（建议提交）" if cat == "watch" else ""

    lines = [
        f"## 新震 {cat_note}",
        f"- **事件ID**: `{ev['event_id']}`",
        f"- **震级**: M{ev['mag']:.1f} ({ev.get('mag_type', 'Mw')})",
        f"- **位置**: {ev.get('place', '')} ({ev['lat']:.2f}, {ev['lon']:.2f})",
        f"- **深度**: {ev.get('depth', 0):.1f} km",
        f"- **时间 UTC**: {utc_str}",
        f"- **时间 北京**: {bj}",
        f"- **T1-T2 截止**: 主震后 24h",
        f"- **T3 截止**: 主震后 72h",
        f"- **天池提交**: {cfg.get('tianchi_url', '')}",
    ]
    if pred_result:
        lines.append("\n### 预测结果")
        lines.append(pred_result.get("summary", ""))
        lines.append(f"\n**输出目录**: `{pred_result.get('output_dir', '')}`")
    return "\n".join(lines)


def pd_timestamp_bj(ms) -> str:
    import pandas as pd

    ts = pd.Timestamp(ms)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S 北京时间")


def notify(cfg: Dict[str, Any], title: str, desp: str) -> bool:
    return send_serverchan(cfg.get("serverchan_sendkey", ""), title, desp)
