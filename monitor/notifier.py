"""Server酱微信推送。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

SERVERCHAN_URL = "https://sctapi.ftqq.com/{sendkey}.send"


def send_serverchan(sendkey: str, title: str, desp: str, short: Optional[str] = None) -> bool:
    if not sendkey:
        logger.warning("未配置 SERVERCHAN_SENDKEY，跳过: %s", title)
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
            return True
        logger.error("Server酱异常: %s", data)
        return False
    except Exception as e:
        logger.error("Server酱失败: %s", e)
        return False


def format_new_event_message(ev: Dict[str, Any], cfg: Dict[str, Any], pred_result: Optional[Dict] = None) -> str:
    from monitor.notify_body import format_new_event_message as _fmt
    return _fmt(ev, cfg, pred_result)


def pd_timestamp_bj(ms) -> str:
    from monitor.time_utils import to_utc_timestamp
    return to_utc_timestamp(ms).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S 北京时间")


def notify(cfg: Dict[str, Any], title: str, desp: str) -> bool:
    return send_serverchan(cfg.get("serverchan_sendkey", ""), title, desp)
