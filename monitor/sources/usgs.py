"""USGS FDSN Event API 拉取与序列构建。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

MAG_PRIORITY = ["mww", "mw", "ms", "mwb", "mwc", "mwr", "ml", "mb", "md"]


def _parse_usgs_time(value) -> datetime:
    """USGS 时间：毫秒 epoch 或 ISO8601 -> UTC aware datetime。"""
    if isinstance(value, str):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC").to_pydatetime()
        return ts.tz_convert("UTC").to_pydatetime()

    # USGS GeoJSON 的 time 为毫秒级 Unix 时间戳（可能是 numpy 标量）
    ms = float(value)
    sec = ms / 1000.0 if ms > 1e12 else ms
    return datetime.fromtimestamp(sec, tz=timezone.utc)


def pick_magnitude(props: Dict[str, Any]) -> Tuple[float, str, str]:
    """返回 (mag, mag_type, category)：category 为 submit|watch|skip."""
    mag = props.get("mag")
    if mag is None:
        return 0.0, "mw", "skip"

    mags = {}
    for key in MAG_PRIORITY:
        val = props.get(key)
        if val is not None:
            mags[key] = float(val)

    if mags:
        for key in MAG_PRIORITY:
            if key in mags:
                best_type = key
                best_mag = mags[key]
                break
        else:
            best_type, best_mag = "mw", float(mag)
    else:
        best_type, best_mag = "mw", float(mag)

    if best_mag >= 6.0:
        cat = "submit"
    elif best_mag >= 5.8:
        cat = "watch"
    else:
        cat = "skip"

    return best_mag, best_type, cat


def event_id_from_dt(dt: datetime) -> str:
    """UTC 时间 -> YYYYMMDDHHMMSS。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_events(
    starttime: datetime,
    endtime: Optional[datetime] = None,
    minmagnitude: float = 5.8,
    maxdepth: float = 70,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    maxradiuskm: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """查询 USGS 事件列表。"""
    if endtime is None:
        endtime = datetime.now(timezone.utc)

    params = {
        "format": "geojson",
        "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": endtime.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": minmagnitude,
        "maxdepth": maxdepth,
        "orderby": "time",
    }
    if latitude is not None and longitude is not None and maxradiuskm is not None:
        params["latitude"] = latitude
        params["longitude"] = longitude
        params["maxradiuskm"] = maxradiuskm

    resp = requests.get(USGS_QUERY_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    events = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [0, 0, 0]
        lon, lat = float(coords[0]), float(coords[1])
        depth = float(coords[2]) if len(coords) > 2 else 10.0
        depth = abs(depth) if depth else 10.0

        mag, mag_type, category = pick_magnitude(props)
        if category == "skip":
            continue

        mainshock_utc = _parse_usgs_time(props["time"])
        events.append(
            {
                "usgs_id": feat.get("id"),
                "event_id": event_id_from_dt(mainshock_utc),
                "mainshock_utc": mainshock_utc,
                "lon": lon,
                "lat": lat,
                "depth": depth,
                "mag": mag,
                "mag_type": mag_type,
                "category": category,
                "place": props.get("place", ""),
                "source": "USGS",
                "url": props.get("url", ""),
            }
        )
    return events


def fetch_sequence(
    mainshock_utc: datetime,
    lon: float,
    lat: float,
    radius_km: float = 300,
    hours: float = 168,
    minmagnitude: float = 2.5,
) -> pd.DataFrame:
    """拉取主震后序列内事件，输出与 test_eq_data 相同列格式。"""
    end = datetime.now(timezone.utc)
    start = mainshock_utc

    raw = fetch_events(
        starttime=start,
        endtime=end,
        minmagnitude=minmagnitude,
        maxdepth=1000,
        latitude=lat,
        longitude=lon,
        maxradiuskm=radius_km,
    )

    rows = []
    for ev in raw:
        dt = ev["mainshock_utc"]
        if dt < mainshock_utc:
            continue
        hours_after = (dt - mainshock_utc).total_seconds() / 3600
        if hours_after > hours:
            continue
        rows.append(
            {
                "Date": dt.strftime("%Y-%m-%d"),
                "Time": dt.strftime("%H:%M:%S"),
                "Lon": ev["lon"],
                "Lat": ev["lat"],
                "Depth": ev["depth"],
                "Mag": ev["mag"],
                "MagType": ev["mag_type"],
                "Source": "USGS",
                "datetime": pd.Timestamp(dt).tz_localize(None),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["Date", "Time", "Lon", "Lat", "Depth", "Mag", "MagType", "Source"]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").drop_duplicates(
        subset=["Date", "Time", "Lon", "Lat", "Mag"], keep="first"
    )
    return df.drop(columns=["datetime"])


def save_sequence(df: pd.DataFrame, event_id: str, sequences_dir: str) -> str:
    import os

    os.makedirs(sequences_dir, exist_ok=True)
    path = os.path.join(sequences_dir, f"{event_id}_eq.csv")
    out = df[["Date", "Time", "Lon", "Lat", "Depth", "Mag", "MagType", "Source"]]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path
