#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽查自动化：**从聊天记录里挑出最值得看的几条**，你只需要回答"有没有明显不对"。

## 为什么需要

`docs/操作手册.md` 有 13 项指标——**太重了，重到不会有人真的用**。
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

## ⭐ 勾完能变成用例（这一半才是重点）

在产出的 markdown 里把每条改成 `[x] 有：为什么不对`，然后跑：

    python scripts/run_tool.py problems --from-spot

`scripts/problem_cases.py` 会把这些条目读回来做**新用例的种子**。
（回读靠 `t=`——它是 `(session, t)` 里那个毫秒级时间戳，实测全库零重复。
**别用行号定位**：`chatlog.py` 超过 20MB 会轮转重命名、删旧文件，行号会漂。）

⚠️ **标记格式是约定，改这里要同步改 `problem_cases.py` 的回读正则。**
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import _common

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "spot_check.md"

# 她主动提起经历的说法——**幻觉高发信号**（可能把某次的事当"现在"）
LOOKBACK = re.compile(r"(之前|那次|上次|以前|前几天|之前那次|曾经|当时)")
# 用户问的是"她"
ABOUT_HER = re.compile(r"(你|hzm|灰泽满|小满|满姐)")

# 勾选标记——**`problem_cases.py` 按这个回读**，改这里要同步改那边。
# 判"有问题"只看紧挨 `有：` 前面那个 checkbox 是不是打了勾（`[ ] 没有` 里也有 checkbox，别误判）。
MARK = "**这条有没有明显不对？** [ ] 没有  [ ] 有：__________"
FLAG_RE = re.compile(r"\[[xX✓]\]\s*有：")

# 可疑信号 → 该对着 `docs/操作手册.md` 的哪几项看（"别每次全查一遍"，只给最相关的）
FOCUS = {
    "带括号": "B4 说话风格（括号是例外不是常态，出现说明可能在演）",
    "回复长": "B4/C2（她日常是短句；群聊里显得郑重）",
    "回复极短": "B4 说话风格",
    "主动提起经历": "★B2 知识幻觉（这事她有依据吗？会不会把某次的事当成现在）",
    "问的是她": "B1 知识准确（关于她自己的事实对不对）／B2",
    "连发多条": "A2 接不接得上",
    "同话题反复问": "C3 表达多样（有没有复读）",
}


def focus_of(sig: str) -> str:
    """信号 → 重点看哪几项（去掉 '回复长(62字)' 那种括号尾巴再查）。"""
    return FOCUS.get(sig.split("(")[0], "")


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
    # Windows 控制台默认 GBK，不强制 UTF-8 的话最后那行带 emoji 的 print 会抛
    # UnicodeEncodeError（文件其实已经写好了，但退出码是崩的）。
    _common.ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="含测试会话")
    ap.add_argument("--recent", type=int, default=0, help="只看最近 N 轮")
    args = ap.parse_args()

    if not LOG.exists():
        print(f"❌ 没有聊天记录 {LOG}")
        return
    rows = [json.loads(l) for l in _common.chatlog_lines()]
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
         "> 每条下面给了「重点看」——**只对着那一两项看就行**（13 项全表在 `docs/操作手册.md`）。",
         "> 勾完（把 `[ ] 有` 改成 `[x] 有：为什么不对`）跑 "
         "`python scripts/run_tool.py problems --from-spot` 就能变成回归用例。", "", "---", ""]
    for i, (s, sig, turn) in enumerate(top, 1):
        L.append(f"### {i}. 「{(turn.get('user') or '').strip()[:60]}」")
        L.append(f"{turn.get('reply') or ''}")
        L.append("")
        # `t=` 是回读用的**唯一键**（毫秒级时间戳）。⚠️ 别换成行号——轮转会改行号。
        L.append(f"`可疑信号：{'、'.join(sig)}`   "
                 f"`session={turn.get('session')} kind={turn.get('kind')} t={turn.get('t')!r}`")
        L.append("")
        focuses = []
        for sg in sig:
            f = focus_of(sg)
            if f and f not in focuses:
                focuses.append(f)
        if focuses:
            L.append("**重点看**：" + " ｜ ".join(focuses[:3]))
            L.append("")
        L.append(MARK)
        L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"✅ 已挑 {len(top)} 条 → {OUT}")
    print("\n可疑信号分布:", dict(Counter(sg for _, sigs, _ in top for sg in sigs)))


if __name__ == "__main__":
    main()
