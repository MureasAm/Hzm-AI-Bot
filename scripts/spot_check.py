#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽查自动化：**从聊天记录里挑出最值得看的几条**，你只需要回答"有没有明显不对"。

## 为什么需要

`抽查清单.md` 有 13 项指标——**太重了，重到不会有人真的用**。
但完全不抽查也不行：组件级能测的都测完了，"整体到底行不行"只有真实对话能回答。

所以换个做法：**把"看哪条"自动化**。代码先按启发式筛出最可疑的几条，
你只看这几条，而且**只回答一个是/否**。

## 挑哪几条（按"最可能有问题"排）

| 信号 | 为什么可疑 |
|---|---|
| 回复带括号 | 她的括号是"例外不是常态"，出现说明可能在演 |
| 回复特别长/短 | 异常长度常伴随失控 |
| 她主动提起经历（"之前/那次/上次"） | **最容易出幻觉**——可能把某次的事当现在 |
| 用户问的是她（"你…"） | 涉及事实，最容易说错 |
| 同一话题被反复问 | 复读高发区 |

## 用法

    python scripts/spot_check.py                 # 挑 10 条存成 markdown
    python scripts/spot_check.py --n 20 --all    # 连测试会话一起
    python scripts/spot_check.py --recent 200    # 只看最近 200 轮

产出：`outputs/spot_check.md`——**扫一眼，每条给个是/否**。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "chat_log" / "chat.jsonl"
OUT = ROOT / "outputs" / "spot_check.md"

# 她主动提起经历的说法——**幻觉高发信号**（可能把某次的事当"现在"）
LOOKBACK = re.compile(r"(之前|那次|上次|以前|前几天|之前那次|曾经|当时)")
# 用户问的是"她"
ABOUT_HER = re.compile(r"(你|hzm|灰泽满|小满|满姐)")


def score(turn: dict) -> tuple:
    """返回 (可疑分, [命中的信号])。分越高越值得看。"""
    u = (turn.get("user") or "").strip()
    r = (turn.get("reply") or "").strip()
    sig, s = [], 0
    if "（" in r or "(" in r:
        sig.append("带括号"); s += 3
    if len(r) > 60:
        sig.append(f"回复长({len(r)}字)"); s += 2
    if len(r) < 6:
        sig.append("回复极短"); s += 1
    if LOOKBACK.search(r):
        sig.append("主动提起经历"); s += 4      # 幻觉最高发
    if ABOUT_HER.search(u):
        sig.append("问的是她"); s += 2
    if u.count("\n") >= 1:
        sig.append("连发多条"); s += 1
    return s, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="含测试会话")
    ap.add_argument("--recent", type=int, default=0, help="只看最近 N 轮")
    args = ap.parse_args()

    if not LOG.exists():
        print(f"❌ 没有聊天记录 {LOG}")
        return
    rows = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    turns = [t for t in rows if (t.get("user") or "").strip()
             and (args.all or str(t.get("session", "")).isdigit())]
    if args.recent:
        turns = turns[-args.recent:]

    # 同一会话里"同一话题被反复问"也算可疑
    dup = Counter((t["session"], (t["user"] or "").strip()[:12]) for t in turns)
    scored = []
    for t in turns:
        s, sig = score(t)
        if dup[(t["session"], (t.get("user") or "").strip()[:12])] >= 3:
            s += 2; sig.append("同话题反复问")
        scored.append((s, sig, t))
    scored.sort(key=lambda x: -x[0])
    top = [x for x in scored if x[0] > 0][: args.n]

    L = ["# 抽查清单（自动挑出来的）", "",
         f"> 从 {len(turns)} 条真实对话里挑出**最可疑的 {len(top)} 条**。",
         "> **每行只回答一个问题：「这条有没有明显不对？」**", "",
         "> 重点看两类（其他都是次要）：",
         "> 1. **她编了吗**——说了没有的事，或把某次的事当现在",
         "> 2. **像她吗**——行为/说话方式",
         "",
         "> 每条后面 `[ ] 没问题  [ ] 有问题：____`", "", "---", ""]
    for i, (s, sig, t) in enumerate(top, 1):
        L.append(f"### {i}. 「{(t.get('user') or '').strip()[:60]}」")
        L.append(f"{t.get('reply') or ''}")
        L.append("")
        L.append(f"`可疑信号：{'、'.join(sig)}`   `session={t.get('session')} kind={t.get('kind')}`")
        L.append("")
        L.append("**这条有没有明显不对？** [ ] 没有  [ ] 有：__________")
        L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"✅ 已挑 {len(top)} 条 → {OUT}")
    print("\n可疑信号分布:", dict(Counter(sg for _, sigs, _ in top for sg in sigs)))


if __name__ == "__main__":
    main()
