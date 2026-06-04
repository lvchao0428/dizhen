#!/usr/bin/env python3
"""提交前检查：CSV 列表、大小、行数（天池 ≤100MB、格式为 CSV）。"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitor.config import get_config
from monitor.package_submission import collect_files

MAX_BYTES = 100 * 1024 * 1024  # 天池单文件上限 100MB


def _csv_stats(path: str) -> dict:
    p = Path(path)
    size = p.stat().st_size
    with open(p, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    return {"path": str(p), "name": p.name, "bytes": size, "lines": len(lines)}


def check(cfg: dict, event_id: str | None = None, zip_path: str | None = None) -> int:
    ranking_dir = cfg["ranking_output_dir"]
    pairs = collect_files(ranking_dir, event_id)
    csv_pairs = [(n, f) for n, f in pairs if n.endswith(".csv")]
    other = [(n, f) for n, f in pairs if not n.endswith(".csv")]

    print("=== 天池排名赛提交检查 ===\n")
    print("平台限制（以天池「提交结果」页为准）：")
    print("  - 格式：ZIP（内含 CSV 文件）")
    print("  - 单次上传不超过 100MB")
    print("  - 剩余提交次数有限（常见为 3 次，以最后一次为准）\n")

    if not csv_pairs:
        print(f"未找到 CSV: {ranking_dir}")
        return 1

    total = 0
    print("待上传 CSV：")
    for name, full in csv_pairs:
        st = _csv_stats(full)
        total += st["bytes"]
        expect = 2 if "-T1-T2" in name else 1
        ok = "✓" if st["lines"] == expect else f"⚠ 期望{expect}行"
        print(f"  {st['name']:40} {st['bytes']:>8} B  {st['lines']} 行  {ok}")

    if other:
        print("\n（以下非 CSV，一般不必上传天池）")
        for name, full in other:
            print(f"  {name}")

    print(f"\nCSV 合计: {total} B ({total / 1024 / 1024:.4f} MB)")
    if total > MAX_BYTES:
        print("✗ 超过 100MB 上限，请减少文件或联系组委会确认批量格式")
        return 2
    print("✓ 体积远低于 100MB 上限")

    if zip_path is not False:
        from monitor.package_submission import create_zip

        if zip_path:
            os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
        else:
            zip_path = os.path.join(cfg["data_dir"], "submissions", "check_latest.zip")
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        zip_path = create_zip(cfg, event_id, zip_path, csv_only=True)
        zsize = os.path.getsize(zip_path)
        print(f"\n已生成提交 ZIP: {zip_path} ({zsize} B)")
        if zsize > MAX_BYTES:
            print("✗ ZIP 超过 100MB")
            return 2
        print("✓ ZIP 体积远低于 100MB 上限")

    print("\n建议：")
    print("  1. 每次上传前在本机确认 T2/T3 已刷新（见 window_comparison.md）")
    print("  2. 仅剩 1～2 次机会时，优先在 T1-T2 / T3 截止并刷新后再提交")
    if zip_path and zip_path is not False:
        print(f"  3. 直接上传上述 ZIP 到天池: {cfg.get('tianchi_url', '')}")
    else:
        print(f"  3. 用 package_submission 打包后上传天池: {cfg.get('tianchi_url', '')}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="检查排名赛 CSV 并生成提交 ZIP")
    p.add_argument("--event", help="只检查单个事件 ID")
    p.add_argument("-z", "--zip", help="指定输出 ZIP 路径（默认自动生成）")
    p.add_argument("--no-zip", action="store_true", help="仅检查，不生成 ZIP")
    args = p.parse_args()
    cfg = get_config()
    zip_path = False if args.no_zip else args.zip
    raise SystemExit(check(cfg, args.event, zip_path))


if __name__ == "__main__":
    main()
