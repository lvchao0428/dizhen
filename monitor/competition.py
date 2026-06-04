"""排名赛资格判定（与微信通知条件分离）。

赛题规则：全球 M≥6.0 地震均需提交，不限深度。
"""

from __future__ import annotations

from typing import Any, Dict


def is_competition_eligible(ev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """M≥6.0 即跑提交流水线（赛题不限深度）。"""
    min_mag = float(cfg.get("min_magnitude_submit", 6.0))
    return float(ev.get("mag", 0)) >= min_mag


def eligibility_reason(ev: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if is_competition_eligible(ev, cfg):
        return "符合排名赛（M≥6.0），应生成提交 CSV"
    if float(ev.get("mag", 0)) < float(cfg.get("min_magnitude_submit", 6.0)):
        return "未达 M6.0，仅监测通知"
    return "不符合比赛处理条件"
