#!/usr/bin/env python3
"""
从 USGS ComCat 抓取 2000-2025 年 M6.0+ 浅源地震，生成训练数据集。

输出：
  solution/usgs_train_data/catalog.csv          — 主震目录
  solution/usgs_train_data/sequences/{ts}_eq.csv — 余震序列

用法：
  python fetch_usgs_training_data.py                 # 全量抓取
  python fetch_usgs_training_data.py --year 2023     # 只抓某年
  python fetch_usgs_training_data.py --dry-run       # 只打印不下载序列
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "usgs_train_data")
SEQ_DIR = os.path.join(OUTPUT_DIR, "sequences")

EXISTING_CATALOG = os.path.join(SCRIPT_DIR, "..", "data_download", "test_eq_catalog.csv")

MAX_DEPTH_KM = 70.0
MIN_MAG = 6.0
SEQUENCE_RADIUS_KM = 300
SEQUENCE_HOURS = 168
SEQUENCE_MIN_MAG = 2.5

RATE_LIMIT_SECONDS = 1.5


def _load_existing_timestamps() -> set:
    """加载资格赛 20 个震例的 timestamp，用于去重。"""
    if not os.path.isfile(EXISTING_CATALOG):
        return set()
    df = pd.read_csv(EXISTING_CATALOG)
    stamps = set()
    for _, r in df.iterrows():
        ts = (
            f"{int(r['Year']):04d}{int(r['Month']):02d}{int(r['Day']):02d}"
            f"{int(r['Hour']):02d}{int(r['Minute']):02d}{int(r['Second']):02d}"
        )
        stamps.add(ts)
    return stamps


def fetch_mainshocks(year_start: int, year_end: int) -> List[dict]:
    """按年份分批拉取主震列表。"""
    all_events = []
    for year in range(year_start, year_end + 1):
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01" if year < year_end else f"{year}-12-31"
        params = {
            "format": "geojson",
            "starttime": start,
            "endtime": end,
            "minmagnitude": MIN_MAG,
            "maxdepth": MAX_DEPTH_KM,
            "orderby": "time",
        }
        print(f"  拉取 {year} ...", end=" ", flush=True)
        try:
            resp = requests.get(USGS_URL, params=params, timeout=90)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"失败: {e}")
            continue

        features = data.get("features", [])
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {}) or {}
            coords = geom.get("coordinates") or [0, 0, 0]
            lon, lat = float(coords[0]), float(coords[1])
            depth = abs(float(coords[2])) if len(coords) > 2 and coords[2] else 10.0

            mag = props.get("mag")
            if mag is None or float(mag) < MIN_MAG:
                continue
            mag = float(mag)

            mag_type = str(props.get("magType", "mw")).strip()
            dt = datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc)
            ts = dt.strftime("%Y%m%d%H%M%S")

            all_events.append({
                "timestamp": ts,
                "Year": dt.year,
                "Month": dt.month,
                "Day": dt.day,
                "Hour": dt.hour,
                "Minute": dt.minute,
                "Second": dt.second,
                "Lon": round(lon, 2),
                "Lat": round(lat, 2),
                "Depth": round(depth, 1),
                "Mag": round(mag, 1),
                "MagType": mag_type,
                "Source": "USGS",
                "place": props.get("place", ""),
                "usgs_id": feat.get("id", ""),
            })

        print(f"{len(features)} 条")
        time.sleep(RATE_LIMIT_SECONDS)

    return all_events


def fetch_one_sequence(ev: dict) -> pd.DataFrame:
    """拉取单个主震的余震序列。"""
    dt = datetime(
        ev["Year"], ev["Month"], ev["Day"],
        ev["Hour"], ev["Minute"], ev["Second"],
        tzinfo=timezone.utc,
    )
    end_dt = dt + timedelta(hours=SEQUENCE_HOURS)

    params = {
        "format": "geojson",
        "starttime": dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": ev["Lat"],
        "longitude": ev["Lon"],
        "maxradiuskm": SEQUENCE_RADIUS_KM,
        "minmagnitude": SEQUENCE_MIN_MAG,
        "maxdepth": 1000,
        "orderby": "time",
        "limit": 5000,
    }

    resp = requests.get(USGS_URL, params=params, timeout=90)
    resp.raise_for_status()
    data = resp.json()

    rows = [{
        "Date": dt.strftime("%Y-%m-%d"),
        "Time": dt.strftime("%H:%M:%S"),
        "Lon": ev["Lon"],
        "Lat": ev["Lat"],
        "Depth": ev["Depth"],
        "Mag": ev["Mag"],
        "MagType": ev["MagType"],
        "Source": "USGS",
    }]

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [0, 0, 0]

        emag = props.get("mag")
        if emag is None:
            continue
        edt = datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc)
        if edt < dt:
            continue

        rows.append({
            "Date": edt.strftime("%Y-%m-%d"),
            "Time": edt.strftime("%H:%M:%S"),
            "Lon": round(float(coords[0]), 2),
            "Lat": round(float(coords[1]), 2),
            "Depth": round(abs(float(coords[2])) if coords[2] else 10.0, 1),
            "Mag": round(float(emag), 1),
            "MagType": str(props.get("magType", "mw")).strip(),
            "Source": "USGS",
        })

    df = pd.DataFrame(rows)
    df["_dt"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("_dt").drop_duplicates(
        subset=["Date", "Time", "Lon", "Lat"], keep="first"
    )
    return df.drop(columns=["_dt"])


def deduplicate_events(events: List[dict], skip_ts: set) -> List[dict]:
    """去重：排除资格赛震例 + 同一秒内只保留最大震级。"""
    seen = {}
    for ev in events:
        ts = ev["timestamp"]
        if ts in skip_ts:
            continue
        ts_prefix = ts[:12]
        if ts_prefix not in seen or ev["Mag"] > seen[ts_prefix]["Mag"]:
            seen[ts_prefix] = ev
    return sorted(seen.values(), key=lambda x: x["timestamp"])


def main():
    parser = argparse.ArgumentParser(description="抓取 USGS 历史 M6+ 训练数据")
    parser.add_argument("--year", type=int, help="只抓某一年")
    parser.add_argument("--year-start", type=int, default=2000)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true", help="只下载目录不下载序列")
    args = parser.parse_args()

    if args.year:
        args.year_start = args.year
        args.year_end = args.year

    os.makedirs(SEQ_DIR, exist_ok=True)
    skip_ts = _load_existing_timestamps()
    print(f"资格赛震例 (排除): {len(skip_ts)} 个")

    print(f"\n[1/3] 抓取 {args.year_start}-{args.year_end} 年 M{MIN_MAG}+ 浅源 (≤{MAX_DEPTH_KM}km) 主震")
    raw_events = fetch_mainshocks(args.year_start, args.year_end)
    print(f"  原始: {len(raw_events)} 条")

    events = deduplicate_events(raw_events, skip_ts)
    print(f"  去重后: {len(events)} 条")

    catalog_df = pd.DataFrame(events)
    catalog_cols = [
        "Year", "Month", "Day", "Hour", "Minute", "Second",
        "Lon", "Lat", "Depth", "Mag", "MagType", "Source",
    ]
    catalog_path = os.path.join(OUTPUT_DIR, "catalog.csv")
    catalog_df[catalog_cols].to_csv(catalog_path, index=False)
    print(f"  目录已保存: {catalog_path}")

    if args.dry_run:
        print("\n[dry-run] 跳过序列下载")
        return

    print(f"\n[2/3] 抓取余震序列 ({len(events)} 个事件)")
    success = 0
    for i, ev in enumerate(events):
        ts = ev["timestamp"]
        seq_path = os.path.join(SEQ_DIR, f"{ts}_eq.csv")
        if os.path.isfile(seq_path):
            success += 1
            continue

        try:
            seq_df = fetch_one_sequence(ev)
            seq_df.to_csv(seq_path, index=False, encoding="utf-8-sig")
            success += 1
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(events)}: {ts} ({len(seq_df)} 行)")
        except Exception as e:
            print(f"  {ts} 失败: {e}")

        time.sleep(RATE_LIMIT_SECONDS)

    print(f"\n[3/3] 完成: {success}/{len(events)} 序列抓取成功")
    print(f"  目录: {catalog_path}")
    print(f"  序列: {SEQ_DIR}/")


if __name__ == "__main__":
    main()
