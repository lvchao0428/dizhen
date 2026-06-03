"""v3 模型文件状态（供微信通知展示）。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict


def get_model_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = cfg.get("model_path", "")
    exists = bool(path and os.path.isfile(path))
    info = {
        "ready": exists,
        "path": path,
        "train_command": "cd solution && python predict_v3.py",
        "train_duration_hint": "约 1–2 分钟（20 震例 × 3 窗口，6 个 LightGBM）",
    }
    if exists:
        mtime = os.path.getmtime(path)
        info["trained_at_utc"] = datetime.fromtimestamp(
            mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        info["size_kb"] = round(os.path.getsize(path) / 1024, 1)
    return info


def format_model_status_line(cfg: Dict[str, Any]) -> str:
    st = get_model_status(cfg)
    if st["ready"]:
        return (
            f"- **模型**: 已就绪 v3 ({st['size_kb']}KB, 更新 {st['trained_at_utc']})"
        )
    return (
        f"- **模型**: 未就绪 — 执行 `{st['train_command']}` "
        f"（{st['train_duration_hint']}）"
    )
