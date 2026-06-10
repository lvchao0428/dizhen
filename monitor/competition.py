"""排名赛资格判定（与微信通知条件分离）。

赛题规则：全球 M≥6.0、震源深度 ≤70km 的浅源地震。
深度超过 70km 仍生成预测（备用），但标记为不符合评比条件。
"""

from __future__ import annotations

from typing import Any, Dict

MAX_DEPTH_KM = 70.0


def should_predict(ev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """M≥6.0 就跑预测（不限深度），深度超限的作为备用。"""
    min_mag = float(cfg.get("min_magnitude_submit", 6.0))
    return float(ev.get("mag", 0)) >= min_mag


def is_competition_eligible(ev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """M≥6.0 且震源深度 ≤70km 才算赛题有效事件（计入评比）。"""
    max_depth = float(cfg.get("max_depth_submit_km", MAX_DEPTH_KM))
    depth = float(ev.get("depth", 0))
    return should_predict(ev, cfg) and depth <= max_depth


def eligibility_reason(ev: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    min_mag = float(cfg.get("min_magnitude_submit", 6.0))
    max_depth = float(cfg.get("max_depth_submit_km", MAX_DEPTH_KM))
    mag = float(ev.get("mag", 0))
    depth = float(ev.get("depth", 0))
    if mag < min_mag:
        return "未达 M6.0，仅监测通知"
    if depth > max_depth:
        return f"震源深度 {depth:.1f}km > {max_depth:.0f}km，不符合评比（仍生成预测备用）"
    return "符合排名赛（M≥6.0，深度≤70km），应生成提交 CSV"
