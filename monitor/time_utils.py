"""统一 UTC 时间处理（避免 naive/aware 混用）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd


def to_utc_timestamp(value) -> pd.Timestamp:
    """任意时间输入 -> UTC aware Timestamp。"""
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def normalize_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """SQLite / USGS 事件 dict 统一主震时间为 UTC aware。"""
    out = dict(ev)
    out["mainshock_utc"] = to_utc_timestamp(ev["mainshock_utc"]).to_pydatetime()
    if "competition_eligible" in out:
        out["competition_eligible"] = bool(out["competition_eligible"])
    return out


def sequence_datetimes_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "datetime" not in out.columns:
        out["datetime"] = pd.to_datetime(out["Date"] + " " + out["Time"], utc=True)
    else:
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    return out
