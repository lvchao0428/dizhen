"""地震检测：通知按震级；赛题流水线按 M≥6.0（不限深度）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from monitor.competition import is_competition_eligible
from monitor.sources import usgs


def _classify_magnitude(ev: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    min_submit = float(cfg.get("min_magnitude_submit", 6.0))
    min_watch = float(cfg.get("min_magnitude_watch", 5.8))
    mag = float(ev["mag"])
    if mag >= min_submit:
        return "submit"
    if mag >= min_watch:
        return "watch"
    return "skip"


def detect_new_events(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """拉取近期地震；通知不限深度，赛题标记 competition_eligible。"""
    lookback = cfg.get("lookback_days", 7)
    start = datetime.now(timezone.utc) - timedelta(days=lookback)
    min_mag = cfg.get("min_magnitude_watch", 5.8)
    usgs_maxdepth = float(cfg.get("usgs_fetch_max_depth_km", 1000))

    raw = usgs.fetch_events(
        starttime=start,
        minmagnitude=min_mag,
        maxdepth=usgs_maxdepth,
    )

    results = []
    for ev in raw:
        cat = _classify_magnitude(ev, cfg)
        if cat == "skip":
            continue
        ev = dict(ev)
        ev["category"] = cat
        ev["competition_eligible"] = is_competition_eligible(ev, cfg)
        results.append(ev)
    return results
