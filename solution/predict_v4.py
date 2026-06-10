#!/usr/bin/env python3
"""
余震预测 v4: 多模型 stacking + USGS 扩充训练集 + Optuna 超参优化。

Layer 0: LightGBM / XGBoost / CatBoost (各自 LOEO out-of-fold)
Layer 1: Ridge 回归 meta learner

用法:
  python predict_v4.py                       # 训练 + 生成提交
  python predict_v4.py --tune                # 含 Optuna 超参搜索
  python predict_v4.py --tune --trials 50    # 指定搜索轮数
  python predict_v4.py --no-usgs             # 仅用资格赛 20 个震例
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from features_v4 import WINDOW_BOUNDS, build_features
from inference_v4 import (
    identify_mainshock_and_aftershocks,
    predict_all,
    save_models,
    write_submission,
)

QUAL_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data_download")
QUAL_CATALOG = os.path.join(QUAL_DATA_DIR, "test_eq_catalog.csv")
QUAL_EQ_DIR = os.path.join(QUAL_DATA_DIR, "test_eq_data")

USGS_DATA_DIR = os.path.join(SCRIPT_DIR, "usgs_train_data")
USGS_CATALOG = os.path.join(USGS_DATA_DIR, "catalog.csv")
USGS_SEQ_DIR = os.path.join(USGS_DATA_DIR, "sequences")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_v4")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models_v4")

_ALL_MODEL_NAMES = ["lgb", "xgb", "cat"]

def _available_models():
    avail = ["lgb"]
    try:
        import xgboost  # noqa
        avail.append("xgb")
    except ImportError:
        pass
    try:
        import catboost  # noqa
        avail.append("cat")
    except ImportError:
        pass
    return avail

MODEL_NAMES = _available_models()

DEFAULT_PARAMS = {
    "lgb": dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        num_leaves=20, min_child_samples=3,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbosity=-1,
    ),
    "xgb": dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbosity=0,
    ),
    "cat": dict(
        iterations=300, depth=5, learning_rate=0.05,
        l2_leaf_reg=1.0, random_seed=42,
        verbose=0,
    ),
}


def _create_model(name: str, params: dict):
    if name == "lgb":
        return LGBMRegressor(**params)
    elif name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    elif name == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    raise ValueError(f"Unknown model: {name}")


# --------------- Data Loading ---------------

def load_qual_catalog() -> pd.DataFrame:
    df = pd.read_csv(QUAL_CATALOG)
    df["timestamp"] = df.apply(
        lambda r: (
            f"{int(r['Year']):04d}{int(r['Month']):02d}{int(r['Day']):02d}"
            f"{int(r['Hour']):02d}{int(r['Minute']):02d}{int(r['Second']):02d}"
        ), axis=1,
    )
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute", "Second"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day",
                     "Hour": "hour", "Minute": "minute", "Second": "second"}
        )
    )
    df["_source"] = "qual"
    return df


def load_usgs_catalog() -> pd.DataFrame:
    if not os.path.isfile(USGS_CATALOG):
        print("  警告: USGS 训练数据不存在，跳过扩充")
        return pd.DataFrame()
    df = pd.read_csv(USGS_CATALOG)
    df["timestamp"] = df.apply(
        lambda r: (
            f"{int(r['Year']):04d}{int(r['Month']):02d}{int(r['Day']):02d}"
            f"{int(r['Hour']):02d}{int(r['Minute']):02d}{int(r['Second']):02d}"
        ), axis=1,
    )
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute", "Second"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day",
                     "Hour": "hour", "Minute": "minute", "Second": "second"}
        )
    )
    df["_source"] = "usgs"
    return df


def load_eq_sequence(timestamp: str, source: str = "qual") -> Optional[pd.DataFrame]:
    if source == "qual":
        path = os.path.join(QUAL_EQ_DIR, f"{timestamp}_eq.csv")
    else:
        path = os.path.join(USGS_SEQ_DIR, f"{timestamp}_eq.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df


def load_all_data(use_usgs: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    qual_cat = load_qual_catalog()
    sequences = {}
    for _, row in qual_cat.iterrows():
        seq = load_eq_sequence(row["timestamp"], "qual")
        if seq is not None:
            sequences[row["timestamp"]] = seq

    catalog = qual_cat.copy()

    if use_usgs:
        usgs_cat = load_usgs_catalog()
        if len(usgs_cat) > 0:
            qual_ts = set(qual_cat["timestamp"])
            usgs_cat = usgs_cat[~usgs_cat["timestamp"].isin(qual_ts)]

            loaded = 0
            for _, row in usgs_cat.iterrows():
                seq = load_eq_sequence(row["timestamp"], "usgs")
                if seq is not None and len(seq) >= 2:
                    sequences[row["timestamp"]] = seq
                    loaded += 1
            usgs_cat = usgs_cat[usgs_cat["timestamp"].isin(sequences)]
            catalog = pd.concat([qual_cat, usgs_cat], ignore_index=True)
            print(f"  USGS 扩充: {loaded} 个有效序列")

    print(f"  总训练集: {len(catalog)} 个震例")
    return catalog, sequences


# --------------- Dataset ---------------

def label_in_window(aftershocks: pd.DataFrame, t_start: float, t_end: float):
    w = aftershocks[(aftershocks["hours_after"] >= t_start) & (aftershocks["hours_after"] < t_end)]
    if len(w) == 0:
        return None, None
    row = w.loc[w["Mag"].idxmax()]
    return float(row["Mag"]), float(row["hours_after"])


def build_dataset(catalog: pd.DataFrame, sequences: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, row in catalog.iterrows():
        ts = row["timestamp"]
        if ts not in sequences:
            continue
        seq = sequences[ts]
        _, after = identify_mainshock_and_aftershocks(seq, row["datetime"], row["Mag"])

        for window in ("T1", "T2", "T3"):
            t_start, t_end = WINDOW_BOUNDS[window]
            y_mag, y_time = label_in_window(after, t_start, t_end)
            if y_mag is None:
                y_mag = row["Mag"] - 1.2
                y_time = (t_start + t_end) / 2

            feat, stat = build_features(
                row["Mag"], row["Depth"],
                row["Lon"], row["Lat"],
                str(row["MagType"]), str(row["Source"]),
                after, window,
            )
            rows.append({
                "timestamp": ts,
                "window": window,
                "features": feat,
                "y_mag": y_mag,
                "y_time": y_time,
                "M_main": row["Mag"],
                "bath_mag": stat["bath_mag"],
                "omori_time": stat["omori_time"],
                "_source": row.get("_source", "qual"),
            })
    return pd.DataFrame(rows)


# --------------- Optuna Tuning ---------------

def tune_model_optuna(
    model_name: str,
    X_all: np.ndarray,
    y_mag_all: np.ndarray,
    y_time_all: np.ndarray,
    events: np.ndarray,
    stat_mag: np.ndarray,
    stat_time: np.ndarray,
    n_trials: int = 30,
) -> dict:
    """Optuna LOEO-based hyperparameter search for one model type."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_name == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "num_leaves": trial.suggest_int("num_leaves", 8, 40),
                "min_child_samples": trial.suggest_int("min_child_samples", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
                "random_state": 42, "verbosity": -1,
            }
        elif model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
                "random_state": 42, "verbosity": 0,
            }
        else:
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "random_seed": 42, "verbose": 0,
            }

        unique_events = np.unique(events)
        n_folds = min(10, len(unique_events))
        folds = _make_group_folds(events, n_folds=n_folds)

        preds_mag = np.full(len(X_all), np.nan)
        preds_time = np.full(len(X_all), np.nan)

        for train_ev, test_ev in folds:
            train_ev_set = set(train_ev)
            test_ev_set = set(test_ev)
            train_mask = np.array([e in train_ev_set for e in events])
            test_mask = np.array([e in test_ev_set for e in events])
            if not test_mask.any():
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_all[train_mask])
            X_te = scaler.transform(X_all[test_mask])

            try:
                m_mag = _create_model(model_name, params)
                m_mag.fit(X_tr, y_mag_all[train_mask])
                m_time = _create_model(model_name, params)
                m_time.fit(X_tr, y_time_all[train_mask])
            except Exception:
                return float("inf")

            preds_mag[test_mask] = m_mag.predict(X_te)
            preds_time[test_mask] = m_time.predict(X_te)

        valid = ~np.isnan(preds_mag)
        mae_m = np.mean(np.abs(preds_mag[valid] - y_mag_all[valid]))
        mae_t = np.mean(np.abs(preds_time[valid] - y_time_all[valid]))
        return mae_m + 0.05 * mae_t

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    if model_name == "lgb":
        best.update({"random_state": 42, "verbosity": -1})
    elif model_name == "xgb":
        best.update({"random_state": 42, "verbosity": 0})
    else:
        best.update({"random_seed": 42, "verbose": 0})

    return best


# --------------- LOEO + Stacking ---------------

def _make_group_folds(events: np.ndarray, n_folds: int = 10, seed: int = 42):
    """Split events into grouped folds. Returns list of (train_events, test_events)."""
    unique = np.unique(events)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    folds = np.array_split(unique, min(n_folds, len(unique)))
    splits = []
    for fold_events in folds:
        test_set = set(fold_events)
        train_ev = np.array([e for e in unique if e not in test_set])
        splits.append((train_ev, fold_events))
    return splits


MAX_LOEO_EVENTS = 60

def loeo_base_predictions(
    dataset: pd.DataFrame,
    model_params: Dict[str, dict],
) -> Tuple[pd.DataFrame, Dict]:
    """Grouped k-fold (or LOEO for small datasets) for all base models -> OOF predictions."""
    records = []
    trained_models = {}

    for window in ("T1", "T2", "T3"):
        sub = dataset[dataset["window"] == window].reset_index(drop=True)
        events = sub["timestamp"].values
        unique_events = np.unique(events)
        X_all = np.vstack(sub["features"].values)

        use_loeo = len(unique_events) <= MAX_LOEO_EVENTS
        if use_loeo:
            folds = [(np.array([e for e in unique_events if e != hold]), np.array([hold]))
                     for hold in unique_events]
            fold_desc = f"LOEO ({len(unique_events)} folds)"
        else:
            n_folds = 10
            folds = _make_group_folds(events, n_folds=n_folds)
            fold_desc = f"{n_folds}-fold grouped CV ({len(unique_events)} events)"
        print(f"  {window}: {fold_desc}")

        for model_name in MODEL_NAMES:
            params = model_params.get(model_name, DEFAULT_PARAMS[model_name])
            oof_mag = np.full(len(sub), np.nan)
            oof_time = np.full(len(sub), np.nan)

            for fi, (train_ev, test_ev) in enumerate(folds):
                train_ev_set = set(train_ev)
                test_ev_set = set(test_ev)
                train_mask = np.array([e in train_ev_set for e in events])
                test_mask = np.array([e in test_ev_set for e in events])

                if not test_mask.any():
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_all[train_mask])
                X_te = scaler.transform(X_all[test_mask])

                m_mag = _create_model(model_name, params)
                m_mag.fit(X_tr, sub.loc[train_mask, "y_mag"].values)
                m_time = _create_model(model_name, params)
                m_time.fit(X_tr, sub.loc[train_mask, "y_time"].values)

                oof_mag[test_mask] = m_mag.predict(X_te)
                oof_time[test_mask] = m_time.predict(X_te)

                if not use_loeo and (fi + 1) % 2 == 0:
                    print(f"    {model_name} fold {fi+1}/{len(folds)} done")

            for i, (_, row) in enumerate(sub.iterrows()):
                records.append({
                    "timestamp": row["timestamp"],
                    "window": window,
                    "model": model_name,
                    "oof_mag": oof_mag[i],
                    "oof_time": oof_time[i],
                    "true_mag": row["y_mag"],
                    "true_time": row["y_time"],
                    "bath_mag": row["bath_mag"],
                    "omori_time": row["omori_time"],
                })

    return pd.DataFrame(records), trained_models


def train_meta_models(oof_df: pd.DataFrame) -> Dict:
    """Train Ridge meta learners on OOF predictions."""
    meta = {}
    for window in ("T1", "T2", "T3"):
        wsub = oof_df[oof_df["window"] == window]
        events = wsub["timestamp"].unique()

        mag_features = []
        time_features = []
        y_mag_all = []
        y_time_all = []

        for ev in events:
            ev_data = wsub[wsub["timestamp"] == ev]
            mag_row = list(ev_data["oof_mag"].values) + [ev_data["bath_mag"].values[0]]
            time_row = list(ev_data["oof_time"].values) + [ev_data["omori_time"].values[0]]
            mag_features.append(mag_row)
            time_features.append(time_row)
            y_mag_all.append(ev_data["true_mag"].values[0])
            y_time_all.append(ev_data["true_time"].values[0])

        X_mag = np.array(mag_features)
        X_time = np.array(time_features)
        y_mag = np.array(y_mag_all)
        y_time = np.array(y_time_all)

        meta_mag = Ridge(alpha=1.0)
        meta_mag.fit(X_mag, y_mag)
        meta_time = Ridge(alpha=1.0)
        meta_time.fit(X_time, y_time)

        meta[f"{window}_mag"] = meta_mag
        meta[f"{window}_time"] = meta_time

    return meta


def train_final_base_models(dataset: pd.DataFrame, model_params: Dict[str, dict]) -> Dict:
    """Train all base models on full data."""
    base_models = {}
    for window in ("T1", "T2", "T3"):
        sub = dataset[dataset["window"] == window]
        X = np.vstack(sub["features"].values)

        for model_name in MODEL_NAMES:
            params = model_params.get(model_name, DEFAULT_PARAMS[model_name])
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            m_mag = _create_model(model_name, params)
            m_mag.fit(Xs, sub["y_mag"].values)
            m_time = _create_model(model_name, params)
            m_time.fit(Xs, sub["y_time"].values)

            base_models[f"{window}_{model_name}_mag"] = {"model": m_mag, "scaler": scaler}
            base_models[f"{window}_{model_name}_time"] = {"model": m_time, "scaler": scaler}

    return base_models


def evaluate_loeo(oof_df: pd.DataFrame, meta_models: Dict) -> pd.DataFrame:
    """Compute stacking LOEO predictions and MAE."""
    results = []
    for window in ("T1", "T2", "T3"):
        wsub = oof_df[oof_df["window"] == window]
        events = wsub["timestamp"].unique()

        meta_mag = meta_models[f"{window}_mag"]
        meta_time = meta_models[f"{window}_time"]

        for ev in events:
            ev_data = wsub[wsub["timestamp"] == ev]
            mag_row = np.array(list(ev_data["oof_mag"].values) + [ev_data["bath_mag"].values[0]]).reshape(1, -1)
            time_row = np.array(list(ev_data["oof_time"].values) + [ev_data["omori_time"].values[0]]).reshape(1, -1)

            pred_mag = meta_mag.predict(mag_row)[0]
            pred_time = meta_time.predict(time_row)[0]

            results.append({
                "timestamp": ev,
                "window": window,
                "pred_mag": pred_mag,
                "pred_time": pred_time,
                "true_mag": ev_data["true_mag"].values[0],
                "true_time": ev_data["true_time"].values[0],
            })

    return pd.DataFrame(results)


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(description="v4 多模型 stacking 训练")
    parser.add_argument("--tune", action="store_true", help="启用 Optuna 超参搜索")
    parser.add_argument("--trials", type=int, default=30, help="Optuna 搜索轮数")
    parser.add_argument("--no-usgs", action="store_true", help="仅用资格赛 20 个震例")
    args = parser.parse_args()

    print("=" * 60)
    print("余震预测 v4 (多模型 stacking + USGS 扩充)")
    print("=" * 60)

    print("\n[1/6] 加载数据")
    catalog, sequences = load_all_data(use_usgs=not args.no_usgs)
    dataset = build_dataset(catalog, sequences)
    print(f"  样本数: {len(dataset)} ({len(catalog)} 震例 x 3 窗口)")

    model_params = dict(DEFAULT_PARAMS)

    if args.tune:
        print(f"\n[2/6] Optuna 超参搜索 ({args.trials} trials/model/window)")
        for window in ("T1", "T2", "T3"):
            sub = dataset[dataset["window"] == window]
            X_all = np.vstack(sub["features"].values)
            events_arr = sub["timestamp"].values

            for model_name in MODEL_NAMES:
                print(f"  调优 {window}/{model_name} ...", end=" ", flush=True)
                best = tune_model_optuna(
                    model_name, X_all,
                    sub["y_mag"].values, sub["y_time"].values,
                    events_arr,
                    sub["bath_mag"].values, sub["omori_time"].values,
                    n_trials=args.trials,
                )
                model_params[model_name] = best
                print("done")
        print("  最优参数:")
        for name, params in model_params.items():
            if name in MODEL_NAMES:
                print(f"    {name}: {params}")
    else:
        print("\n[2/6] 使用默认超参 (可用 --tune 启用搜索)")

    print("\n[3/6] LOEO base model OOF 预测")
    oof_df, _ = loeo_base_predictions(dataset, model_params)

    for window in ("T1", "T2", "T3"):
        wsub = oof_df[oof_df["window"] == window]
        for mn in MODEL_NAMES:
            msub = wsub[wsub["model"] == mn]
            mae_m = np.mean(np.abs(msub["oof_mag"] - msub["true_mag"]))
            mae_t = np.mean(np.abs(msub["oof_time"] - msub["true_time"]))
            print(f"  {window}/{mn}: 震级MAE={mae_m:.3f}, 时间MAE={mae_t:.2f}h")

    print("\n[4/6] 训练 Ridge meta learner")
    meta_models = train_meta_models(oof_df)

    eval_df = evaluate_loeo(oof_df, meta_models)
    print("\n  Stacking LOEO 结果:")
    for window in ("T1", "T2", "T3"):
        wsub = eval_df[eval_df["window"] == window]
        mae_m = np.mean(np.abs(wsub["pred_mag"] - wsub["true_mag"]))
        mae_t = np.mean(np.abs(wsub["pred_time"] - wsub["true_time"]))
        print(f"  {window}: 震级MAE={mae_m:.3f}, 时间MAE={mae_t:.2f}h")

    mae_m_all = np.mean(np.abs(eval_df["pred_mag"] - eval_df["true_mag"]))
    mae_t_all = np.mean(np.abs(eval_df["pred_time"] - eval_df["true_time"]))
    print(f"  总体: 震级MAE={mae_m_all:.3f}, 时间MAE={mae_t_all:.2f}h")

    print("\n[5/6] 全量训练 base models + 保存")
    base_models = train_final_base_models(dataset, model_params)

    bundle = {
        "base_models": base_models,
        "meta_models": meta_models,
        "model_names": MODEL_NAMES,
        "w_ml": {"T1": 0.7, "T2": 0.7, "T3": 0.7},
        "model_params": {k: v for k, v in model_params.items() if k in MODEL_NAMES},
        "version": "v4",
        "n_train": len(catalog),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "models.pkl")
    save_models(bundle, model_path)
    print(f"  模型已保存: {model_path}")

    print("\n[6/6] 生成提交文件")
    qual_cat = catalog[catalog["_source"] == "qual"].copy() if "_source" in catalog.columns else catalog.copy()
    predictions = predict_all(qual_cat, sequences, bundle, bundle["w_ml"])
    write_submission(qual_cat, predictions, OUTPUT_DIR)
    print(f"  输出目录: {OUTPUT_DIR}")
    print("\n完成。")


if __name__ == "__main__":
    main()
