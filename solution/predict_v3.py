#!/usr/bin/env python3
"""
余震预测资格赛 v3：100% ML 推理，无泄漏特征，LightGBM 分窗模型 + LOEO。
"""

from __future__ import annotations

import os
import pickle
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

from features import WINDOW_BOUNDS, WINDOW_CUTOFF, build_features

import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_download")
CATALOG_FILE = os.path.join(DATA_DIR, "test_eq_catalog.csv")
EQ_DATA_DIR = os.path.join(DATA_DIR, "test_eq_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_v3")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_v3")

W_ML_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]

LGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=2,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbosity=-1,
)


def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_FILE)
    df["timestamp"] = df.apply(
        lambda r: (
            f"{int(r['Year']):04d}{int(r['Month']):02d}{int(r['Day']):02d}"
            f"{int(r['Hour']):02d}{int(r['Minute']):02d}{int(r['Second']):02d}"
        ),
        axis=1,
    )
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute", "Second"]].rename(
            columns={
                "Year": "year",
                "Month": "month",
                "Day": "day",
                "Hour": "hour",
                "Minute": "minute",
                "Second": "second",
            }
        )
    )
    return df


def load_eq_sequence(timestamp: str) -> pd.DataFrame:
    path = os.path.join(EQ_DATA_DIR, f"{timestamp}_eq.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df


def identify_mainshock_and_aftershocks(
    seq_df: pd.DataFrame, mainshock_time: pd.Timestamp, mainshock_mag: float
) -> Tuple[pd.Series, pd.DataFrame]:
    tol = pd.Timedelta(hours=2)
    candidates = seq_df[
        (abs(seq_df["datetime"] - mainshock_time) < tol)
        & (seq_df["Mag"] >= mainshock_mag - 0.5)
    ]
    idx = candidates["Mag"].idxmax() if len(candidates) else seq_df["Mag"].idxmax()
    main = seq_df.loc[idx]
    after = seq_df[seq_df["datetime"] > main["datetime"]].copy()
    after["hours_after"] = (after["datetime"] - main["datetime"]).dt.total_seconds() / 3600
    return main, after


def label_in_window(aftershocks: pd.DataFrame, t_start: float, t_end: float) -> Tuple[Optional[float], Optional[float]]:
    w = aftershocks[(aftershocks["hours_after"] >= t_start) & (aftershocks["hours_after"] < t_end)]
    if len(w) == 0:
        return None, None
    row = w.loc[w["Mag"].idxmax()]
    return float(row["Mag"]), float(row["hours_after"])


def build_dataset(catalog: pd.DataFrame, sequences: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, row in catalog.iterrows():
        ts = row["timestamp"]
        seq = sequences[ts]
        _, after = identify_mainshock_and_aftershocks(seq, row["datetime"], row["Mag"])

        for window in ("T1", "T2", "T3"):
            t_start, t_end = WINDOW_BOUNDS[window]
            y_mag, y_time = label_in_window(after, t_start, t_end)
            if y_mag is None:
                y_mag = row["Mag"] - 1.2
                y_time = (t_start + t_end) / 2

            feat, stat = build_features(
                row["Mag"],
                row["Depth"],
                row["Lon"],
                row["Lat"],
                str(row["MagType"]),
                str(row["Source"]),
                after,
                window,
            )
            rows.append(
                {
                    "timestamp": ts,
                    "window": window,
                    "features": feat,
                    "y_mag": y_mag,
                    "y_time": y_time,
                    "M_main": row["Mag"],
                    "bath_mag": stat["bath_mag"],
                    "omori_time": stat["omori_time"],
                }
            )
    return pd.DataFrame(rows)


def _train_lgb(X: np.ndarray, y: np.ndarray) -> LGBMRegressor:
    model = LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    return model


def loeo_predict(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """留一震例交叉验证，搜索各窗口 w_ml。"""
    records = []
    best_w: Dict[str, float] = {}

    for window in ("T1", "T2", "T3"):
        sub = dataset[dataset["window"] == window].reset_index(drop=True)
        events = sub["timestamp"].unique()
        preds_mag, preds_time, true_mag, true_time = [], [], [], []
        stat_mag, stat_time = [], []

        for hold in events:
            train = sub[sub["timestamp"] != hold]
            test = sub[sub["timestamp"] == hold]
            X_train = np.vstack(train["features"].values)
            X_test = np.vstack(test["features"].values)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            m_mag = _train_lgb(X_train_s, train["y_mag"].values)
            m_time = _train_lgb(X_train_s, train["y_time"].values)

            ml_mag = m_mag.predict(X_test_s)[0]
            ml_time = m_time.predict(X_test_s)[0]
            preds_mag.append(ml_mag)
            preds_time.append(ml_time)
            true_mag.append(test["y_mag"].values[0])
            true_time.append(test["y_time"].values[0])
            stat_mag.append(test["bath_mag"].values[0])
            stat_time.append(test["omori_time"].values[0])

        best_score = float("inf")
        best_w_ml = 0.7
        for w_ml in W_ML_GRID:
            w_s = 1.0 - w_ml
            blend_mag = [w_ml * p + w_s * s for p, s in zip(preds_mag, stat_mag)]
            blend_time = [w_ml * p + w_s * s for p, s in zip(preds_time, stat_time)]
            mae_m = np.mean(np.abs(np.array(blend_mag) - np.array(true_mag)))
            mae_t = np.mean(np.abs(np.array(blend_time) - np.array(true_time)))
            score = mae_m + 0.05 * mae_t
            if score < best_score:
                best_score = score
                best_w_ml = w_ml

        best_w[window] = best_w_ml

        for i, ev in enumerate(events):
            w_ml = best_w_ml
            w_s = 1.0 - w_ml
            records.append(
                {
                    "timestamp": ev,
                    "window": window,
                    "pred_mag_loeo": w_ml * preds_mag[i] + w_s * stat_mag[i],
                    "pred_time_loeo": w_ml * preds_time[i] + w_s * stat_time[i],
                    "true_mag": true_mag[i],
                    "true_time": true_time[i],
                }
            )

    return pd.DataFrame(records), best_w


def train_final_models(dataset: pd.DataFrame) -> Dict:
    """全量训练 6 个模型 + scaler。"""
    models = {}
    for window in ("T1", "T2", "T3"):
        sub = dataset[dataset["window"] == window]
        X = np.vstack(sub["features"].values)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        models[f"{window}_mag"] = {"model": _train_lgb(Xs, sub["y_mag"].values), "scaler": scaler}
        models[f"{window}_time"] = {"model": _train_lgb(Xs, sub["y_time"].values), "scaler": scaler}
    return models


def clip_predictions(
    pred_mag: float, pred_hours: float, m_main: float, t_start: float, t_end: float
) -> Tuple[float, float]:
    pred_mag = min(pred_mag, m_main - 0.3)
    pred_mag = max(pred_mag, m_main - 3.0)
    pred_mag = max(pred_mag, 2.0)
    pred_hours = max(pred_hours, t_start + 0.5)
    pred_hours = min(pred_hours, t_end - 0.5)
    return round(pred_mag, 1), pred_hours


def predict_all(
    catalog: pd.DataFrame,
    sequences: Dict[str, pd.DataFrame],
    models: Dict,
    w_ml: Dict[str, float],
) -> Dict[str, Dict[str, dict]]:
    predictions = {}
    for _, row in catalog.iterrows():
        ts = row["timestamp"]
        seq = sequences[ts]
        _, after = identify_mainshock_and_aftershocks(seq, row["datetime"], row["Mag"])
        eq_pred = {}

        for window in ("T1", "T2", "T3"):
            t_start, t_end = WINDOW_BOUNDS[window]
            feat, stat = build_features(
                row["Mag"],
                row["Depth"],
                row["Lon"],
                row["Lat"],
                str(row["MagType"]),
                str(row["Source"]),
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
            pred_mag, pred_hours = clip_predictions(pred_mag, pred_hours, row["Mag"], t_start, t_end)

            pred_time = row["datetime"] + timedelta(hours=pred_hours)
            eq_pred[window] = {"mag": pred_mag, "time": pred_time, "hours": pred_hours}

        predictions[ts] = eq_pred
    return predictions


def format_time(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d%H")


def write_submission(catalog: pd.DataFrame, predictions: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for _, row in catalog.iterrows():
        ts = row["timestamp"]
        pred = predictions[ts]
        mag_type = row["MagType"]
        with open(os.path.join(output_dir, f"{ts}-T1-T2.csv"), "w", encoding="utf-8") as f:
            t1, t2 = pred["T1"], pred["T2"]
            f.write(
                f"{ts} {row['Lon']:.2f} {row['Lat']:.2f} {row['Mag']:.1f} "
                f"{t1['mag']:.1f} ({mag_type}) {format_time(t1['time'])}\n"
            )
            f.write(
                f"{ts} {row['Lon']:.2f} {row['Lat']:.2f} {row['Mag']:.1f} "
                f"{t2['mag']:.1f} ({mag_type}) {format_time(t2['time'])}\n"
            )
        with open(os.path.join(output_dir, f"{ts}-T3.csv"), "w", encoding="utf-8") as f:
            t3 = pred["T3"]
            f.write(
                f"{ts} {row['Lon']:.2f} {row['Lat']:.2f} {row['Mag']:.1f} "
                f"{t3['mag']:.1f} ({mag_type}) {format_time(t3['time'])}\n"
            )


def main() -> None:
    print("=" * 60)
    print("余震预测资格赛 v3 (ML-heavy, no leakage)")
    print("=" * 60)

    catalog = load_catalog()
    sequences = {row["timestamp"]: load_eq_sequence(row["timestamp"]) for _, row in catalog.iterrows()}
    dataset = build_dataset(catalog, sequences)
    print(f"样本数: {len(dataset)} (20 震例 x 3 窗口)")

    print("\n[LOEO 交叉验证 + w_ml 搜索]")
    loeo_df, w_ml = loeo_predict(dataset)
    for w in ("T1", "T2", "T3"):
        sub = loeo_df[loeo_df["window"] == w]
        mae_m = np.mean(np.abs(sub["pred_mag_loeo"] - sub["true_mag"]))
        mae_t = np.mean(np.abs(sub["pred_time_loeo"] - sub["true_time"]))
        print(f"  {w}: w_ml={w_ml[w]:.1f}, 震级MAE={mae_m:.3f}, 时间MAE={mae_t:.2f}h")

    print("\n[全量训练 + 生成提交]")
    models = train_final_models(dataset)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "models.pkl"), "wb") as f:
        pickle.dump({"models": models, "w_ml": w_ml}, f)

    predictions = predict_all(catalog, sequences, models, w_ml)
    write_submission(catalog, predictions, OUTPUT_DIR)
    print(f"输出目录: {OUTPUT_DIR}")
    print("完成。")


if __name__ == "__main__":
    main()
