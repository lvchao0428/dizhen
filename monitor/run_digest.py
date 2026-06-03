#!/usr/bin/env python3
"""手动 M6+ 汇总 / 发日报。用法: python -m monitor.run_digest [--notify] [--backfill] [--days 30]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from monitor.digest import build_digest_report, run_digest, save_digest_report
from monitor.config import get_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int)
    p.add_argument("--notify", action="store_true")
    p.add_argument("--backfill", action="store_true")
    p.add_argument("--print-only", action="store_true")
    args = p.parse_args()

    if args.print_only:
        cfg = get_config()
        report, events, bf = build_digest_report(cfg, args.days, args.backfill)
        print(report)
        save_digest_report(cfg, report)
        print(f"\n共 {len(events)} 起, 补跑 {len(bf)}", file=sys.stderr)
        return

    import json
    print(json.dumps(run_digest(args.days, args.notify, args.backfill), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
