"""排名赛资格判定（与微信通知条件分离）。"""

from __future__ import annotations

from typing import Any, Dict


def is_competition_eligible(ev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """M≥6.0 且深度≤70 km 才跑提交流水线。"""
    min_mag = float(cfg.get("min_magnitude_submit", 6.0))
    max_depth = float(cfg.get("competition_max_depth_km", 70))
    return float(ev.get("mag", 0)) >= min_mag and float(ev.get("depth", 0)) <= max_depth


def eligibility_reason(ev: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if is_competition_eligible(ev, cfg):
        return "符合排名赛（M≥6.0 且深度≤70km），应生成提交 CSV"
    mag = float(ev.get("mag", 0))
    depth = float(ev.get("depth", 0))
    max_depth = float(cfg.get("competition_max_depth_km", 70))
    if mag < float(cfg.get("min_magnitude_submit", 6.0)):
        return "未达 M6.0，仅监测通知"
    if depth > max_depth:
        return f"深度 {depth:.0f}km > {max_depth:.0f}km，仅微信通知，不生成比赛提交文件"
    return "不符合比赛处理条件"
