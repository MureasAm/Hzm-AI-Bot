#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量流水线：11 场直播 → 处境清单（一把跑完）。

串三层：
  ① scene_segment  每场切情景（含"能否搬到私聊"的筛子）
  ② scene_cluster  分批归并
  ③ scene_merge    全局归并 + 标「已覆盖/缺口/丢弃」
"""
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SCRIPTS = Path(__file__).resolve().parent


def run(script, args, label):
    print(f"\n{'#'*78}\n# {label}\n{'#'*78}", flush=True)
    r = subprocess.run([PY, str(SCRIPTS / script), *args], cwd=str(ROOT))
    if r.returncode != 0:
        print(f"⚠️ {label} 退出码 {r.returncode}", flush=True)


def main():
    # 跳过 lv3 那个（它是 0713 的重复转写，避免同一场算两次）
    files = sorted(p for p in glob.glob(str(ROOT / "outputs" / "clean" / "*.json"))
                   if "lv3" not in p and Path(p).name != "cleaned.json")
    if not files:
        print("❌ 没找到清洗稿")
        return
    print(f"共 {len(files)} 场：")
    for f in files:
        print("  ", Path(f).name)

    # ① 切分（一场一场来，避免并发把 API 打爆）
    for f in files:
        run("scene_segment.py", [f, "--window", "40", "--overlap", "8"],
            f"① 切情景：{Path(f).name}")

    # ② 聚类
    segs = sorted(glob.glob(str(ROOT / "outputs" / "scene_segments" / "*_scenes.json")))
    if segs:
        run("scene_cluster.py", segs, f"② 分批聚类：{len(segs)} 场")

    # ③ 全局归并
    cl = ROOT / "outputs" / "scene_clusters.json"
    if cl.exists():
        run("scene_merge.py", [str(cl)], "③ 全局归并 + 标缺口")

    print("\n\n全部完成。")


if __name__ == "__main__":
    sys.exit(main())
