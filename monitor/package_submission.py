#!/usr/bin/env python3
"""打包排名赛提交 CSV 为 ZIP。"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitor.config import get_config


def collect_files(
    ranking_dir: str, event_id: str | None = None, *, include_aux: bool = True
) -> list[tuple[str, str]]:
    files = []
    root = Path(ranking_dir)
    dirs = [root / event_id] if event_id else sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*-T1-T2.csv")) + sorted(d.glob("*-T3.csv")):
            files.append((p.name, str(p)))
        if include_aux:
            comp = d / "window_comparison.md"
            if comp.is_file():
                files.append((f"{d.name}_window_comparison.md", str(comp)))
    return files


def collect_csv_only(ranking_dir: str, event_id: str | None = None) -> list[tuple[str, str]]:
    return collect_files(ranking_dir, event_id, include_aux=False)


def create_zip(cfg: dict, event_id: str | None = None, output_zip: str | None = None) -> str:
    pairs = collect_files(cfg["ranking_output_dir"], event_id)
    if not pairs:
        raise FileNotFoundError(f"无提交文件: {cfg['ranking_output_dir']}")
    pack_dir = os.path.join(cfg["data_dir"], "submissions")
    os.makedirs(pack_dir, exist_ok=True)
    if not output_zip:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        label = event_id or "all"
        output_zip = os.path.join(pack_dir, f"submission_{label}_{ts}.zip")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, full in pairs:
            zf.write(full, arcname=arc)
    return output_zip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--event")
    p.add_argument("-o", "--output")
    args = p.parse_args()
    cfg = get_config()
    path = create_zip(cfg, args.event, args.output)
    print(f"已打包 {len(collect_files(cfg['ranking_output_dir'], args.event))} 个文件 -> {path}")


if __name__ == "__main__":
    main()
