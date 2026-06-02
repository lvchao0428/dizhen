#!/usr/bin/env python3
"""v2 查表 vs v3 LOEO 本地评估对比。"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from predict_v3 import (
    build_dataset,
    identify_mainshock_and_aftershocks,
    label_in_window,
    load_catalog,
    load_eq_sequence,
    loeo_predict,
)
from features import WINDOW_BOUNDS

# v2 查表逻辑（仅评估）
def v2_lookup_predictions(catalog, sequences):
    rows = []
    for _, row in catalog.iterrows():
        ts = row["timestamp"]
        _, after = identify_mainshock_and_aftershocks(sequences[ts], row["datetime"], row["Mag"])
        for window, (t_start, t_end) in WINDOW_BOUNDS.items():
            w = after[(after["hours_after"] >= t_start) & (after["hours_after"] < t_end)]
            if len(w) == 0:
                continue
            r = w.loc[w["Mag"].idxmax()]
            rows.append(
                {
                    "timestamp": ts,
                    "window": window,
                    "pred_mag": r["Mag"],
                    "pred_time_h": r["hours_after"],
                    "true_mag": r["Mag"],
                    "true_time_h": r["hours_after"],
                }
            )
    return pd.DataFrame(rows)


def main():
    catalog = load_catalog()
    sequences = {row["timestamp"]: load_eq_sequence(row["timestamp"]) for _, row in catalog.iterrows()}

    print("=" * 60)
    print("v2 查表方案 (有窗口数据即抄答案)")
    print("=" * 60)
    v2_df = v2_lookup_predictions(catalog, sequences)
    print(f"  震级 MAE: {0.0:.3f} (恒为 0，直接读标签)")
    print(f"  时间 MAE: {0.0:.2f} h")
    print(f"  覆盖样本: {len(v2_df)}/60")

    print("\n" + "=" * 60)
    print("v3 LOEO 方案 (100% ML + 统计融合，无泄漏特征)")
    print("=" * 60)
    dataset = build_dataset(catalog, sequences)
    loeo_df, w_ml = loeo_predict(dataset)

    for window in ("T1", "T2", "T3"):
        sub = loeo_df[loeo_df["window"] == window]
        mae_m = np.mean(np.abs(sub["pred_mag_loeo"] - sub["true_mag"]))
        mae_t = np.mean(np.abs(sub["pred_time_loeo"] - sub["true_time"]))
        print(f"  {window} (w_ml={w_ml[window]}): 震级 MAE={mae_m:.3f}, 时间 MAE={mae_t:.2f}h")

    mae_m_all = np.mean(np.abs(loeo_df["pred_mag_loeo"] - loeo_df["true_mag"]))
    mae_t_all = np.mean(np.abs(loeo_df["pred_time_loeo"] - loeo_df["true_time"]))
    print(f"\n  总体: 震级 MAE={mae_m_all:.3f}, 时间 MAE={mae_t_all:.2f}h")
    print("\n说明: v3 所有预测均经 LightGBM；v2 在资格赛数据上为 oracle 上界。")


if __name__ == "__main__":
    main()
