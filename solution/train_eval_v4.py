#!/usr/bin/env python3
"""v3 vs v4 精度对比：资格赛 20 震例 LOEO + 排名赛回测。

用法:
  cd solution && python train_eval_v4.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from features import WINDOW_BOUNDS as V3_BOUNDS
from features import build_features as v3_build_features
from predict_v3 import (
    build_dataset as v3_build_dataset,
    load_catalog as v3_load_catalog,
    load_eq_sequence as v3_load_eq_sequence,
    loeo_predict as v3_loeo_predict,
)

from features_v4 import WINDOW_BOUNDS as V4_BOUNDS
from features_v4 import build_features as v4_build_features
from predict_v4 import (
    build_dataset as v4_build_dataset,
    load_all_data,
    loeo_base_predictions,
    train_meta_models,
    evaluate_loeo,
    DEFAULT_PARAMS,
    MODEL_NAMES,
)


def run_v3_loeo():
    """Run v3 LOEO on 20 qual events."""
    catalog = v3_load_catalog()
    sequences = {
        row["timestamp"]: v3_load_eq_sequence(row["timestamp"])
        for _, row in catalog.iterrows()
    }
    dataset = v3_build_dataset(catalog, sequences)
    loeo_df, w_ml = v3_loeo_predict(dataset)
    return loeo_df, w_ml


def run_v4_loeo(use_usgs: bool = True):
    """Run v4 LOEO with stacking."""
    catalog, sequences = load_all_data(use_usgs=use_usgs)
    dataset = v4_build_dataset(catalog, sequences)

    has_xgb = True
    has_cat = True
    try:
        import xgboost  # noqa
    except ImportError:
        has_xgb = False
    try:
        import catboost  # noqa
    except ImportError:
        has_cat = False

    active_models = ["lgb"]
    if has_xgb:
        active_models.append("xgb")
    if has_cat:
        active_models.append("cat")

    import predict_v4
    original_names = predict_v4.MODEL_NAMES
    predict_v4.MODEL_NAMES = active_models

    oof_df, _ = loeo_base_predictions(dataset, DEFAULT_PARAMS)
    meta_models = train_meta_models(oof_df)
    eval_df = evaluate_loeo(oof_df, meta_models)

    predict_v4.MODEL_NAMES = original_names
    return eval_df, dataset


def print_comparison(v3_loeo, v4_eval):
    qual_ts = set(v3_loeo["timestamp"].unique())
    v4_qual = v4_eval[v4_eval["timestamp"].isin(qual_ts)]

    print("\n" + "=" * 70)
    print("v3 vs v4 LOEO 精度对比 (资格赛 20 震例)")
    print("=" * 70)
    print(f"{'Window':<8} {'v3 MAE_mag':>12} {'v4 MAE_mag':>12} {'改进':>8}   "
          f"{'v3 MAE_time':>12} {'v4 MAE_time':>12} {'改进':>8}")
    print("-" * 70)

    for window in ("T1", "T2", "T3"):
        v3w = v3_loeo[v3_loeo["window"] == window]
        v4w = v4_qual[v4_qual["window"] == window]

        v3_mae_m = np.mean(np.abs(v3w["pred_mag_loeo"] - v3w["true_mag"]))
        v3_mae_t = np.mean(np.abs(v3w["pred_time_loeo"] - v3w["true_time"]))
        v4_mae_m = np.mean(np.abs(v4w["pred_mag"] - v4w["true_mag"]))
        v4_mae_t = np.mean(np.abs(v4w["pred_time"] - v4w["true_time"]))

        dm = v3_mae_m - v4_mae_m
        dt = v3_mae_t - v4_mae_t
        print(f"{window:<8} {v3_mae_m:>12.3f} {v4_mae_m:>12.3f} {dm:>+8.3f}   "
              f"{v3_mae_t:>11.2f}h {v4_mae_t:>11.2f}h {dt:>+7.2f}h")

    v3_m = np.mean(np.abs(v3_loeo["pred_mag_loeo"] - v3_loeo["true_mag"]))
    v3_t = np.mean(np.abs(v3_loeo["pred_time_loeo"] - v3_loeo["true_time"]))
    v4_m = np.mean(np.abs(v4_qual["pred_mag"] - v4_qual["true_mag"]))
    v4_t = np.mean(np.abs(v4_qual["pred_time"] - v4_qual["true_time"]))
    dm = v3_m - v4_m
    dt = v3_t - v4_t
    print("-" * 70)
    print(f"{'总体':<8} {v3_m:>12.3f} {v4_m:>12.3f} {dm:>+8.3f}   "
          f"{v3_t:>11.2f}h {v4_t:>11.2f}h {dt:>+7.2f}h")

    print(f"\nv4 全体 ({len(v4_eval['timestamp'].unique())} 震例) 10-fold CV:")
    for window in ("T1", "T2", "T3"):
        v4w = v4_eval[v4_eval["window"] == window]
        mae_m = np.mean(np.abs(v4w["pred_mag"] - v4w["true_mag"]))
        mae_t = np.mean(np.abs(v4w["pred_time"] - v4w["true_time"]))
        print(f"  {window}: 震级MAE={mae_m:.3f}, 时间MAE={mae_t:.2f}h")


def main():
    print("=" * 60)
    print("v3 vs v4 精度对比")
    print("=" * 60)

    print("\n[v3 LOEO]")
    v3_loeo, v3_wml = run_v3_loeo()
    for w in ("T1", "T2", "T3"):
        sub = v3_loeo[v3_loeo["window"] == w]
        mae_m = np.mean(np.abs(sub["pred_mag_loeo"] - sub["true_mag"]))
        mae_t = np.mean(np.abs(sub["pred_time_loeo"] - sub["true_time"]))
        print(f"  {w}: w_ml={v3_wml[w]:.1f}, 震级MAE={mae_m:.3f}, 时间MAE={mae_t:.2f}h")

    print("\n[v4 Stacking LOEO]")
    v4_eval, _ = run_v4_loeo(use_usgs=True)
    for w in ("T1", "T2", "T3"):
        sub = v4_eval[v4_eval["window"] == w]
        mae_m = np.mean(np.abs(sub["pred_mag"] - sub["true_mag"]))
        mae_t = np.mean(np.abs(sub["pred_time"] - sub["true_time"]))
        print(f"  {w}: 震级MAE={mae_m:.3f}, 时间MAE={mae_t:.2f}h")

    print_comparison(v3_loeo, v4_eval)


if __name__ == "__main__":
    main()
