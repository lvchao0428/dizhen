"""地震汇总报告 + 每日微信日报。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from monitor.competition import eligibility_reason, is_competition_eligible
from monitor.config import get_config
from monitor.notifier import notify, pd_timestamp_bj
from monitor.pipeline import run_competition_pipeline, setup_logging
from monitor.sequences import ensure_sequence
from monitor.sources import usgs
from monitor.window_compare import build_window_comparison, format_comparison_markdown


def fetch_m6_plus_events(cfg: Dict[str, Any], days: int) -> List[Dict[str, Any]]:
    start = datetime.now(timezone.utc) - timedelta(days=days)
    raw = usgs.fetch_events(
        starttime=start,
        minmagnitude=float(cfg.get("min_magnitude_submit", 6.0)),
        maxdepth=float(cfg.get("usgs_fetch_max_depth_km", 1000)),
    )
    events = []
    for ev in raw:
        if float(ev["mag"]) < float(cfg.get("min_magnitude_submit", 6.0)):
            continue
        ev = dict(ev)
        ev["competition_eligible"] = is_competition_eligible(ev, cfg)
        ev["category"] = "submit"
        events.append(ev)
    return sorted(events, key=lambda x: x["mainshock_utc"], reverse=True)


def format_event_block(ev: Dict[str, Any], cfg: Dict[str, Any], index: int) -> str:
    from monitor.window_compare import format_comparison_wechat
    from monitor.model_info import get_model_status

    ms = ev["mainshock_utc"]
    bj = pd_timestamp_bj(ms)
    utc_str = ms.strftime("%Y-%m-%d %H:%M:%S UTC") if hasattr(ms, "strftime") else str(ms)
    eid = ev["event_id"]
    eligible = ev.get("competition_eligible", False)

    lines = [
        f"### {index}. `{eid}` | M{ev['mag']:.1f} | {ev.get('place', '')[:45]}",
        f"- **时间**: {bj} / {utc_str}",
        f"- **深度**: {ev['depth']:.0f} km | **赛题**: {eligibility_reason(ev, cfg)}",
    ]
    if eligible:
        n = ensure_sequence(ev, cfg)
        comp = build_window_comparison(ev, cfg)
        lines.append(f"- **序列**: {n} 行（余震 {max(0, n-1)} 条）")
        lines.append("")
        lines.append(format_comparison_wechat(comp, model_ready=get_model_status(cfg)["ready"]))
        out = os.path.join(cfg["ranking_output_dir"], eid)
        t12 = os.path.join(out, f"{eid}-T1-T2.csv")
        if os.path.isfile(t12):
            lines.append(f"\n- **提交 CSV**: `{out}`")
    return "\n".join(lines)


def build_digest_report(cfg: Dict[str, Any], days: Optional[int] = None, backfill: bool = False) -> Tuple[str, List, List]:
    days = days or int(cfg.get("digest_lookback_days", 30))
    events = fetch_m6_plus_events(cfg, days)
    backfilled = []
    if backfill:
        for ev in events:
            if not ev.get("competition_eligible"):
                continue
            t12 = os.path.join(cfg["ranking_output_dir"], ev["event_id"], f"{ev['event_id']}-T1-T2.csv")
            if not os.path.isfile(t12):
                try:
                    run_competition_pipeline(ev, cfg)
                    backfilled.append(ev["event_id"])
                except Exception:
                    pass

    header = [
        "## 全球 M6+ 地震汇总",
        f"- **区间**: 过去 {days} 天 | **生成**: {pd_timestamp_bj(datetime.now(timezone.utc))}",
        f"- **合计**: {len(events)} 起",
        "",
    ]
    if backfilled:
        header.append(f"- **本次补跑**: {', '.join(backfilled)}\n")

    body = "\n---\n\n".join(format_event_block(ev, cfg, i) for i, ev in enumerate(events, 1))
    report = "\n".join(header) + (body or "\n（无 M6+ 事件）")
    max_len = int(cfg.get("digest_max_chars", 28000))
    if len(report) > max_len:
        report = report[:max_len] + f"\n\n…（见 `{cfg['data_dir']}/logs/digest_latest.md`）"
    return report, events, backfilled


def save_digest_report(cfg: Dict[str, Any], text: str) -> str:
    log_dir = os.path.join(cfg["data_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "digest_latest.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run_digest(days: Optional[int] = None, notify_user: bool = False, backfill: bool = False) -> Dict:
    cfg = get_config()
    setup_logging(cfg)
    report, events, backfilled = build_digest_report(cfg, days, backfill)
    path = save_digest_report(cfg, report)
    ok = False
    if notify_user:
        ok = notify(cfg, f"[余震赛日报] {datetime.now(timezone.utc).date()} ({len(events)}起)", report)
    return {"events": len(events), "report_path": path, "backfilled": backfilled, "notified": ok}
