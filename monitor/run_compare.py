#!/usr/bin/env python3
"""单事件对比: python -m monitor.run_compare EVENT_ID [--predict]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from monitor.config import get_config
from monitor.detector import detect_new_events
from monitor.pipeline import run_competition_pipeline
from monitor.sequences import ensure_sequence
from monitor.window_compare import build_window_comparison, format_comparison_wechat, save_comparison


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("event_id")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args()

    cfg = get_config()
    cfg["lookback_days"] = args.days
    ev = next((e for e in detect_new_events(cfg) if e["event_id"] == args.event_id), None)
    if not ev:
        raise SystemExit(f"未找到 {args.event_id}")

    if args.predict:
        r = run_competition_pipeline(ev, cfg)
        print(r.get("comparison_wechat", ""))
        return

    ensure_sequence(ev, cfg)
    comp = build_window_comparison(ev, cfg)
    path = save_comparison(comp, os.path.join(cfg["ranking_output_dir"], ev["event_id"]))
    print(format_comparison_wechat(comp))
    print(f"\n-> {path}")


if __name__ == "__main__":
    import os
    main()
