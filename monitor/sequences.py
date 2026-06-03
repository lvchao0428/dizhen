"""余震序列拉取与保存。"""

from __future__ import annotations

from typing import Any, Dict

from monitor.sources import usgs


def ensure_sequence(ev: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    """拉取/更新 USGS 余震序列 CSV，返回总行数（含主震）。"""
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
    return len(seq_df)
