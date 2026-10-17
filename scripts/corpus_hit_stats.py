#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus 注入命中统计：读运行时日志 `data/corpus_hits.jsonl`，按条目累计。

数据来源是 `core.build_message_list` 里 `log_corpus_hits()` 写的**真实注入**记录
（不是候选、不是判官保留，是最终进了提示词的）。

用法：
    python scripts/corpus_hit_stats.py            # 全量
    python scripts/corpus_hit_stats.py --days 7   # 只看最近 7 天
    python scripts/corpus_hit_stats.py --top 20
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

HITS_FILE = ROOT / "data" / "corpus_hits.jsonl"
OUT_MD = _common.OUTPUTS_DIR / "corpus_hit_stats.md"


def _load(days: float) -> list:
    if not HITS_FILE.exists():
        return []
    cutoff = time.time() - days * 86400 if days else 0
    out = []
    for line in HITS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("t") or 0) >= cutoff:
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=0, help="只看最近 N 天（0=全部）")
    ap.add_argument("--top", type=int, default=15, help="打印前 N 条")
    args = ap.parse_args()
    _common.ensure_utf8_stdout()

    recs = _load(args.days)
    statements = [x["statement"] for x in json.loads(
        _common.STATEMENT_FILE.read_text(encoding="utf-8"))["statements"]]
    hits = Counter()
    turns = 0
    for r in recs:
        items = r.get("items") or []
        if not items:
            continue
        turns += 1
        for it in items:
            hits[str(it.get("id"))] += 1

    total = sum(hits.values())
    print(f"日志：{HITS_FILE}")
    print(f"有注入的轮次：{turns} ｜ 总注入：{total} 次 ｜ 被用到条目：{len(hits)} / {len(statements)}")
    if not total:
        print("（还没有数据——先让 bot 跑一阵，命中日志会自然累积）")
        return 0
    top10 = sum(c for _, c in hits.most_common(10))
    print(f"前 10 条占全部注入：{top10 / total * 100:.0f}%")
    print("\n被注入最多的：")
    for i, c in hits.most_common(args.top):
        text = statements[int(i)] if i.isdigit() and int(i) < len(statements) else ""
        print(f"  ×{c:<3} #{i:<4}{text[:46]}")

    lines = ["# corpus 真实注入统计", "",
             f"> 数据：`data/corpus_hits.jsonl`（运行时日志，记录最终进提示词的条目）",
             f"> 轮次 {turns} ｜ 总注入 {total} 次 ｜ 被用到 {len(hits)} / {len(statements)} 条"
             + (f" ｜ 最近 {args.days:g} 天" if args.days else ""), "",
             "| # | 注入次数 | 条目 |", "|---|---|---|"]
    for i, text in enumerate(statements):
        lines.append(f"| {i} | {hits.get(str(i), 0)} | {text[:70].replace('|', '｜')} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 全量表格：{OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
