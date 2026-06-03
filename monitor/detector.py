"""M6+ 事件检测与过滤。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from monitor.sources import usgs


def detect_new_events(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """拉取近期地震并过滤为待处理事件。"""
    lookback = cfg.get("lookback_days", 7)
    start = datetime.now(timezone.utc) - timedelta(days=lookback)
    min_mag = cfg.get("min_magnitude_watch", 5.8)
    max_depth = cfg.get("max_depth_km", 70)

    raw = usgs.fetch_events(
        starttime=start,
        minmagnitude=min_mag,
        maxdepth=max_depth,
    )

    min_submit = cfg.get("min_magnitude_submit", 6.0)
    results = []
    for ev in raw:
        if ev["depth"] > max_depth:
            continue
        if ev["category"] == "watch" and ev["mag"] < min_submit:
            ev["category"] = "watch"
        elif ev["mag"] >= min_submit:
            ev["category"] = "submit"
        else:
            continue
        results.append(ev)
    return results
