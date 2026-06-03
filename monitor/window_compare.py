"""比赛时间窗：模型预测 vs USGS 观测最大余震对比。"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLUTION_DIR = REPO_ROOT / "solution"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from features import WINDOW_BOUNDS  # noqa: E402
from inference_v3 import format_time, identify_mainshock_and_aftershocks, normalize_mag_type  # noqa: E402

WINDOW_NAMES = {
    "T1": "T1 主震后 0–24h",
    "T2": "T2 主震后 24–72h",
    "T3": "T3 主震后 72–168h",
}


def _mainshock_dt(ev: Dict[str, Any]) -> pd.Timestamp:
    ts = pd.Timestamp(ev["mainshock_utc"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def hours_since_mainshock(ev: Dict[str, Any]) -> float:
    ms = _mainshock_dt(ev)
    now = pd.Timestamp.now(tz="UTC")
    return (now - ms).total_seconds() / 3600


def load_sequence_df(event_id: str, sequences_dir: str) -> pd.DataFrame:
    path = os.path.join(sequences_dir, f"{event_id}_eq.csv")
    if not os.path.isfile(path):
        return pd.DataFrame(
            columns=["Date", "Time", "Lon", "Lat", "Depth", "Mag", "MagType", "Source"]
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], utc=True)
    return df


def observed_max_in_window(
    aftershocks: pd.DataFrame, t_start: float, t_end: float
) -> Optional[Dict[str, Any]]:
    if aftershocks is None or len(aftershocks) == 0:
        return None
    w = aftershocks[
        (aftershocks["hours_after"] >= t_start) & (aftershocks["hours_after"] < t_end)
    ]
    if len(w) == 0:
        return None
    row = w.loc[w["Mag"].idxmax()]
    return {
        "mag": float(row["Mag"]),
        "hours_after": float(row["hours_after"]),
        "datetime": row["datetime"],
        "count": int(len(w)),
    }


def window_phase(hours_after_ms: float, t_start: float, t_end: float) -> str:
    if hours_after_ms < t_start:
        return "未开始"
    if hours_after_ms < t_end:
        return "进行中"
    return "已结束"


def predictions_from_csv(event_id: str, ranking_output_dir: str) -> Optional[Dict[str, dict]]:
    out_dir = os.path.join(ranking_output_dir, event_id)
    p12 = os.path.join(out_dir, f"{event_id}-T1-T2.csv")
    if not os.path.isfile(p12):
        return None

    def _parse_line(line: str) -> dict:
        parts = line.split()
        mag = float(parts[4].split("(")[0])
        pred_time = pd.to_datetime(parts[-1], format="%Y%m%d%H", utc=True)
        return {"mag": mag, "time": pred_time}

    preds = {}
    with open(p12, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) >= 1:
        preds["T1"] = _parse_line(lines[0])
    if len(lines) >= 2:
        preds["T2"] = _parse_line(lines[1])
    p3 = os.path.join(out_dir, f"{event_id}-T3.csv")
    if os.path.isfile(p3):
        with open(p3, encoding="utf-8") as f:
            ln = f.readline().strip()
        if ln:
            preds["T3"] = _parse_line(ln)
    return preds or None


def build_window_comparison(
    ev: Dict[str, Any],
    cfg: Dict[str, Any],
    predictions: Optional[Dict[str, dict]] = None,
    seq_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    eid = ev["event_id"]
    if seq_df is None:
        seq_df = load_sequence_df(eid, cfg["sequences_dir"])

    ms = _mainshock_dt(ev)
    if len(seq_df) == 0:
        _, after = identify_mainshock_and_aftershocks(seq_df, ms, float(ev["mag"]))
        n_after = 0
    else:
        if "datetime" not in seq_df.columns:
            seq_df["datetime"] = pd.to_datetime(seq_df["Date"] + " " + seq_df["Time"], utc=True)
        _, after = identify_mainshock_and_aftershocks(seq_df, ms, float(ev["mag"]))
        n_after = len(after)

    if predictions is None:
        predictions = predictions_from_csv(eid, cfg["ranking_output_dir"]) or {}

    h_now = hours_since_mainshock(ev)
    windows_out = []

    for window in ("T1", "T2", "T3"):
        t_start, t_end = WINDOW_BOUNDS[window]
        phase = window_phase(h_now, t_start, t_end)
        obs = observed_max_in_window(after, t_start, t_end)
        pred = predictions.get(window) or {}
        pred_mag = pred.get("mag")
        pred_time = pred.get("time")

        row = {
            "window": window,
            "label": WINDOW_NAMES[window],
            "phase": phase,
            "predicted": None,
            "observed": None,
            "compare_note": "",
        }
        if pred_mag is not None:
            row["predicted"] = {
                "mag": pred_mag,
                "time": pred_time,
                "time_str": format_time(pred_time) if pred_time is not None else None,
            }
        if obs:
            obs_time = obs["datetime"]
            if obs_time.tzinfo is None:
                obs_time = obs_time.tz_localize("UTC")
            row["observed"] = {
                "mag": obs["mag"],
                "hours_after": obs["hours_after"],
                "time": obs_time,
                "time_str": format_time(obs_time),
                "count": obs["count"],
            }

        if row["predicted"] and row["observed"]:
            dm = row["predicted"]["mag"] - row["observed"]["mag"]
            row["compare_note"] = f"震级差(预测-官方) {dm:+.1f}; 窗内{row['observed']['count']}次"
        elif row["predicted"] and phase == "已结束":
            row["compare_note"] = "窗口已结束，官方(USGS)无记录"
        elif row["predicted"] and phase == "进行中":
            row["compare_note"] = "窗口进行中，官方为截至目前"
        elif row["predicted"]:
            row["compare_note"] = "仅有模型预估"
        elif row["observed"]:
            row["compare_note"] = "仅有官方观测"
        else:
            row["compare_note"] = "暂无数据"

        windows_out.append(row)

    return {
        "event_id": eid,
        "hours_since_mainshock": h_now,
        "sequence_rows": len(seq_df),
        "aftershock_count": n_after,
        "windows": windows_out,
    }


def format_comparison_wechat(comp: Dict[str, Any], model_ready: bool = True) -> str:
    lines = [
        "### 余震对比（各时间窗最大震）",
        f"- 主震后 **{comp['hours_since_mainshock']:.1f}h** | 目录余震 **{comp['aftershock_count']}** 条",
    ]
    if comp["aftershock_count"] == 0:
        lines.append("- **官方(USGS)余震**: 数据为空（截至目前无记录）")
    else:
        lines.append(f"- **官方(USGS)余震**: 已收录 {comp['aftershock_count']} 条")

    if not model_ready:
        lines.append("- **模型预估**: 不可用（请先训练 models.pkl）")

    for w in comp["windows"]:
        block = [f"**{w['window']}** {w['label']} · {w['phase']}"]
        pred, obs = w["predicted"], w["observed"]
        if model_ready:
            if pred:
                block.append(f"  - 模型预估: M{pred['mag']:.1f} @ {pred['time_str']}")
            else:
                block.append("  - 模型预估: —")
        if obs:
            block.append(
                f"  - 官方(USGS): M{obs['mag']:.1f} @ {obs['time_str']} "
                f"({obs['count']}次) [已有]"
            )
            if pred and model_ready:
                block.append(f"  - 差异: **ΔM {pred['mag'] - obs['mag']:+.1f}**")
        else:
            if w["phase"] == "已结束":
                block.append("  - 官方(USGS): — （窗口已结束，无记录）")
            elif w["phase"] == "进行中":
                block.append("  - 官方(USGS): — （进行中，暂无余震）")
            else:
                block.append("  - 官方(USGS): —")
        lines.extend(block)
    return "\n".join(lines)


def format_comparison_markdown(comp: Dict[str, Any]) -> str:
    lines = [
        "| 窗口 | 状态 | 模型预估 | 官方(USGS) | 说明 |",
        "|------|------|----------|------------|------|",
    ]
    for w in comp["windows"]:
        pred, obs = w["predicted"], w["observed"]
        pred_s = f"M{pred['mag']:.1f}@{pred['time_str']}" if pred else "—"
        obs_s = f"M{obs['mag']:.1f}@{obs['time_str']}({obs['count']}次)" if obs else "—"
        lines.append(f"| {w['window']} | {w['phase']} | {pred_s} | {obs_s} | {w['compare_note']} |")
    return "\n".join(lines)


def save_comparison(comp: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "window_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 窗口对比 `{comp['event_id']}`\n\n")
        f.write(format_comparison_markdown(comp))
    return path
