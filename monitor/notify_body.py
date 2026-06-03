"""组装微信通知正文（模型状态 + 余震对比）。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from monitor.competition import eligibility_reason
from monitor.model_info import format_model_status_line, get_model_status
from monitor.notifier import pd_timestamp_bj
from monitor.sequences import ensure_sequence
from monitor.window_compare import build_window_comparison, format_comparison_wechat


def build_aftershock_section(
    ev: Dict[str, Any],
    cfg: Dict[str, Any],
    pred_result: Optional[Dict[str, Any]] = None,
    refresh_sequence: bool = True,
) -> str:
    if not ev.get("competition_eligible"):
        return "- **余震对比**: 不满足浅源赛题，未跑模型"

    if refresh_sequence:
        ensure_sequence(ev, cfg)

    predictions = pred_result.get("predictions") if pred_result else None
    comp = build_window_comparison(ev, cfg, predictions=predictions)
    return format_comparison_wechat(comp, model_ready=get_model_status(cfg)["ready"])


def format_new_event_message(
    ev: Dict[str, Any], cfg: Dict[str, Any], pred_result: Optional[Dict] = None
) -> str:
    ms = ev["mainshock_utc"]
    bj = pd_timestamp_bj(ms) if hasattr(ms, "strftime") else str(ms)
    utc_str = ms.strftime("%Y-%m-%d %H:%M:%S UTC") if hasattr(ms, "strftime") else str(ms)

    eligible = ev.get("competition_eligible", False)
    lines = [
        "## 新震",
        format_model_status_line(cfg),
        f"- **赛题资格**: {eligibility_reason(ev, cfg)}",
        f"- **事件ID**: `{ev['event_id']}`",
        f"- **震级**: M{ev['mag']:.1f} ({ev.get('mag_type', 'Mw')})",
        f"- **位置**: {ev.get('place', '')}",
        f"- **深度**: {ev.get('depth', 0):.1f} km",
        f"- **时间**: {bj} / {utc_str}",
    ]
    if eligible:
        lines.extend(["- **T1-T2 截止**: 主震后 24h", "- **T3 截止**: 主震后 72h"])

    lines.append("")
    lines.append(build_aftershock_section(ev, cfg, pred_result, refresh_sequence=not bool(pred_result)))

    if pred_result:
        lines.extend(["", "### 提交文件", pred_result.get("summary", ""), f"`{pred_result.get('output_dir', '')}`"])
    elif eligible and not get_model_status(cfg)["ready"]:
        lines.append("", "- **提交 CSV**: 待训练模型后 `--backfill`")

    lines.append(f"\n**天池**: {cfg.get('tianchi_url', '')}")
    return "\n".join(lines)


def format_competition_ready_message(
    ev: Dict[str, Any], cfg: Dict[str, Any], pred_result: Dict[str, Any]
) -> str:
    return "\n".join(
        [
            "## 比赛提交文件已生成",
            format_model_status_line(cfg),
            f"- **事件**: `{ev['event_id']}` M{ev['mag']:.1f}",
            f"- **目录**: `{pred_result.get('output_dir', '')}`",
            "",
            build_aftershock_section(ev, cfg, pred_result, refresh_sequence=False),
            "",
            "### 提交摘要",
            pred_result.get("summary", ""),
            "",
            f"`python -m monitor.package_submission --event {ev['event_id']}`",
        ]
    )
