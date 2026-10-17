#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12–19 块级开火统计：读运行时日志 `data/block_hits.jsonl`。

每条记录 = 一轮真实聊天实际注入了哪些块（由 `core.build_message_list` 写）。
用它回答"这块到底有没有在跑、开火率多少"，比拿存储反推靠谱。

用法：
    python scripts/block_hit_stats.py
    python scripts/block_hit_stats.py --days 7
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

HITS_FILE = ROOT / "data" / "block_hits.jsonl"
ALL_BLOCKS = ["commitment", "history", "same_request", "batch",
              "query_hint", "now", "mood", "daily"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=0, help="只看最近 N 天（0=全部）")
    args = ap.parse_args()
    _common.ensure_utf8_stdout()

    if not HITS_FILE.exists():
        print(f"还没有数据：{HITS_FILE}\n（先让 bot 跑一阵；只有真实聊天路径会写）")
        return 0
    cutoff = time.time() - args.days * 86400 if args.days else 0
    total = 0
    counts = Counter()
    for line in HITS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("t") or 0) < cutoff:
            continue
        total += 1
        counts.update(set(r.get("blocks") or []))

    if not total:
        print("时间窗内没有记录。")
        return 0
    print(f"记录 {total} 轮" + (f"（最近 {args.days:g} 天）" if args.days else ""))
    for name in ALL_BLOCKS:
        n = counts.get(name, 0)
        print(f"  {name:<12}{n:>6} 轮  {n / total * 100:5.1f}%")
    other = sum(v for k, v in counts.items() if k not in ALL_BLOCKS)
    if other:
        print(f"  （其它块 {other} 次）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
