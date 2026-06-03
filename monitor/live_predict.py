"""排名赛实时预测：加载 v3 模型并写出提交 CSV。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLUTION_DIR = REPO_ROOT / "solution"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from inference_v3 import (  # noqa: E402
    catalog_row_from_dict,
    load_models,
    predict_event,
    predictions_summary,
    write_event_submission,
)
from monitor.time_utils import sequence_datetimes_utc, to_utc_timestamp  # noqa: E402


def load_sequence_csv(event_id: str, sequences_dir: str) -> pd.DataFrame:
    path = os.path.join(sequences_dir, f"{event_id}_eq.csv")
    if not os.path.isfile(path):
        return pd.DataFrame(
            columns=["Date", "Time", "Lon", "Lat", "Depth", "Mag", "MagType", "Source"]
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    return sequence_datetimes_utc(df)


def run_prediction(
    ev: Dict[str, Any],
    cfg: Dict[str, Any],
    models_bundle: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """
    对单个事件运行预测。
    返回 {predictions, output_dir, paths, summary}.
    """
    model_path = cfg["model_path"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n请先在 solution 目录运行: python predict_v3.py"
        )

    if models_bundle is None:
        models, w_ml = load_models(model_path)
    else:
        models, w_ml = models_bundle

    ms_utc = to_utc_timestamp(ev["mainshock_utc"])

    catalog_row = catalog_row_from_dict(
        event_id=ev["event_id"],
        dt=ms_utc,
        lon=ev["lon"],
        lat=ev["lat"],
        depth=ev.get("depth", 10),
        mag=ev["mag"],
        mag_type=ev.get("mag_type", "Mw"),
        source=ev.get("source", "USGS"),
    )

    seq_df = load_sequence_csv(ev["event_id"], cfg["sequences_dir"])
    predictions = predict_event(catalog_row, seq_df, models, w_ml)

    out_dir = os.path.join(cfg["ranking_output_dir"], ev["event_id"])
    path_t12, path_t3 = write_event_submission(catalog_row, predictions, out_dir)

    return {
        "predictions": predictions,
        "output_dir": out_dir,
        "path_t12": path_t12,
        "path_t3": path_t3,
        "summary": predictions_summary(predictions),
        "catalog_row": catalog_row,
    }
