"""
v3 推理模块：单震例预测、提交 CSV 写入、模型加载。
供 predict_v3.py（训练）与 monitor（排名赛实时）共用。
"""

from __future__ import annotations

import os
import pickle
from datetime import timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from features import WINDOW_BOUNDS, build_features


def identify_mainshock_and_aftershocks(
    seq_df: pd.DataFrame, mainshock_time: pd.Timestamp, mainshock_mag: float
) -> Tuple[pd.Series, pd.DataFrame]:
    tol = pd.Timedelta(hours=2)
    if len(seq_df) == 0:
        empty = pd.DataFrame(columns=list(seq_df.columns) + ["hours_after"])
        return pd.Series(dtype=object), empty

    candidates = seq_df[
        (abs(seq_df["datetime"] - mainshock_time) < tol)
        & (seq_df["Mag"] >= mainshock_mag - 0.5)
    ]
    idx = candidates["Mag"].idxmax() if len(candidates) else seq_df["Mag"].idxmax()
    main = seq_df.loc[idx]
    after = seq_df[seq_df["datetime"] > main["datetime"]].copy()
    after["hours_after"] = (after["datetime"] - main["datetime"]).dt.total_seconds() / 3600
    return main, after


def clip_predictions(
    pred_mag: float, pred_hours: float, m_main: float, t_start: float, t_end: float
) -> Tuple[float, float]:
    pred_mag = min(pred_mag, m_main - 0.3)
    pred_mag = max(pred_mag, m_main - 3.0)
    pred_mag = max(pred_mag, 2.0)
    pred_hours = max(pred_hours, t_start + 0.5)
    pred_hours = min(pred_hours, t_end - 0.5)
    return round(pred_mag, 1), pred_hours

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_v3", "models.pkl")


def normalize_mag_type(mag_type: str) -> str:
    """排名赛提交仅允许 Mw / Ms。"""
    t = str(mag_type).strip().lower()
    if t in ("ms",):
        return "Ms"
    if t in ("mw", "mww", "mwc", "mwb", "mwr"):
        return "Mw"
    if t in ("m",):
        return "Mw"
    return "Mw"


def catalog_row_from_dict(
    event_id: str,
    dt: pd.Timestamp,
    lon: float,
    lat: float,
    depth: float,
    mag: float,
    mag_type: str = "Mw",
    source: str = "USGS",
) -> pd.Series:
    """由排名赛主震参数构造 catalog 单行。"""
    mag_type_norm = normalize_mag_type(mag_type)
    return pd.Series(
        {
            "timestamp": event_id,
            "Year": dt.year,
            "Month": dt.month,
            "Day": dt.day,
            "Hour": dt.hour,
            "Minute": dt.minute,
            "Second": dt.second,
            "Lon": lon,
            "Lat": lat,
            "Depth": depth if depth and depth > 0 else 10.0,
            "Mag": mag,
            "MagType": mag_type_norm,
            "Source": source,
            "datetime": dt,
        }
    )


def load_models(path: str = DEFAULT_MODEL_PATH) -> Tuple[Dict, Dict[str, float]]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["models"], bundle["w_ml"]


def save_models(models: Dict, w_ml: Dict[str, float], path: str = DEFAULT_MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"models": models, "w_ml": w_ml}, f)


def predict_event(
    catalog_row: pd.Series,
    seq_df: pd.DataFrame,
    models: Dict,
    w_ml: Dict[str, float],
) -> Dict[str, dict]:
    """对单个震例预测 T1/T2/T3。"""
    if seq_df is None or len(seq_df) == 0:
        seq_df = pd.DataFrame(
            columns=["Date", "Time", "Lon", "Lat", "Depth", "Mag", "MagType", "Source", "datetime"]
        )
    elif "datetime" not in seq_df.columns:
        seq_df = seq_df.copy()
        seq_df["datetime"] = pd.to_datetime(seq_df["Date"] + " " + seq_df["Time"])

    _, after = identify_mainshock_and_aftershocks(
        seq_df, catalog_row["datetime"], catalog_row["Mag"]
    )
    eq_pred = {}

    for window in ("T1", "T2", "T3"):
        t_start, t_end = WINDOW_BOUNDS[window]
        feat, stat = build_features(
            catalog_row["Mag"],
            catalog_row["Depth"],
            catalog_row["Lon"],
            catalog_row["Lat"],
            str(catalog_row["MagType"]),
            str(catalog_row["Source"]),
            after,
            window,
        )
        scaler = models[f"{window}_mag"]["scaler"]
        Xs = scaler.transform(feat.reshape(1, -1))
        ml_mag = models[f"{window}_mag"]["model"].predict(Xs)[0]
        ml_time = models[f"{window}_time"]["model"].predict(Xs)[0]

        w = w_ml[window]
        pred_mag = w * ml_mag + (1 - w) * stat["bath_mag"]
        pred_hours = w * ml_time + (1 - w) * stat["omori_time"]
        pred_mag, pred_hours = clip_predictions(
            pred_mag, pred_hours, catalog_row["Mag"], t_start, t_end
        )
        pred_time = catalog_row["datetime"] + timedelta(hours=pred_hours)
        eq_pred[window] = {"mag": pred_mag, "time": pred_time, "hours": pred_hours}

    return eq_pred


def format_time(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d%H")


def write_event_submission(
    catalog_row: pd.Series,
    predictions: Dict[str, dict],
    output_dir: str,
) -> Tuple[str, str]:
    """写入单个震例的 T1-T2 与 T3 文件，返回路径。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = catalog_row["timestamp"]
    mag_type = normalize_mag_type(catalog_row["MagType"])

    path_t12 = os.path.join(output_dir, f"{ts}-T1-T2.csv")
    path_t3 = os.path.join(output_dir, f"{ts}-T3.csv")

    t1, t2, t3 = predictions["T1"], predictions["T2"], predictions["T3"]
    with open(path_t12, "w", encoding="utf-8") as f:
        f.write(
            f"{ts} {catalog_row['Lon']:.2f} {catalog_row['Lat']:.2f} {catalog_row['Mag']:.1f} "
            f"{t1['mag']:.1f} ({mag_type}) {format_time(t1['time'])}\n"
        )
        f.write(
            f"{ts} {catalog_row['Lon']:.2f} {catalog_row['Lat']:.2f} {catalog_row['Mag']:.1f} "
            f"{t2['mag']:.1f} ({mag_type}) {format_time(t2['time'])}\n"
        )
    with open(path_t3, "w", encoding="utf-8") as f:
        f.write(
            f"{ts} {catalog_row['Lon']:.2f} {catalog_row['Lat']:.2f} {catalog_row['Mag']:.1f} "
            f"{t3['mag']:.1f} ({mag_type}) {format_time(t3['time'])}\n"
        )
    return path_t12, path_t3


def predictions_summary(predictions: Dict[str, dict]) -> str:
    lines = []
    for w in ("T1", "T2", "T3"):
        p = predictions[w]
        lines.append(f"- **{w}**: M{p['mag']:.1f} @ {format_time(p['time'])}")
    return "\n".join(lines)


def predict_all(
    catalog: pd.DataFrame,
    sequences: Dict[str, pd.DataFrame],
    models: Dict,
    w_ml: Dict[str, float],
) -> Dict[str, Dict[str, dict]]:
    predictions = {}
    for _, row in catalog.iterrows():
        ts = row["timestamp"]
        seq = sequences.get(ts, pd.DataFrame())
        predictions[ts] = predict_event(row, seq, models, w_ml)
    return predictions


def write_submission(catalog: pd.DataFrame, predictions: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for _, row in catalog.iterrows():
        ts = row["timestamp"]
        write_event_submission(row, predictions[ts], output_dir)
