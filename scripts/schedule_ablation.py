#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小范围实验：**【周表】那 136 字，值不值。**

## 为什么做这个（以及为什么可以用合成问题）

用户对"按需注入"的质疑很准：**那是拿"确定的 136 字"去换"一个新的、可能出错的判据"**。
所以真正该知道的不是"频率高不高"，而是：

    周表每次被用上时，它值多少？ → 由此算出"需要多高的频率才划算"

⚠️ **本实验用合成问法，不违反本仓那条"合成场景 = 循环论证"的警告**：
那条针对的是「**我造场景 + 我造判据**」（同一套假设自己证明自己）。
这里的判据是**硬事实**——她答的时间**是否等于 `schedule.json` 里的值**，与问题怎么造无关。

## 怎么测

对每个问法，**同一批 messages 生成两遍**（只差【灰泽满的周表】那一段）：
  ① 有周表  ② 没有周表
判据：**她答的时间对不对**（按问的星期查周表）。
  · 周三/周五/周日 = 20:00 ｜ 周四 = 14:00 ｜ 周一/周二 = 休
  · 接受几种说法：`20:00`/`20点`/`8点`/`晚8`；休 → `休`/`不播`/`没有`

## 用法
    python scripts/schedule_ablation.py            # 默认每个问法各跑 3 次（样本少会抖）
    python scripts/schedule_ablation.py --runs 5
"""
import argparse
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")

import src.plugins.chatbot.core as core            # noqa: E402
import src.plugins.chatbot.memory as mem           # noqa: E402

SCHEDULE_MARK = "【灰泽满的周表】"

# (问法, 该天的正确答案的正则)  —— 只用**答案唯一**的问法
CASES = [
    ("周三几点播呀", r"(20:00|20点|晚8点|8点|八点)"),
    ("周日有直播吗", r"(20:00|20点|晚8点|8点|八点)"),
    ("你周五播吗", r"(20:00|20点|晚8点|8点|八点)"),
    ("周四下午播吗", r"(14:00|14点|下午2点|2点|两点)"),
    ("周一你播吗", r"(休|不播|没有|休息)"),
    ("周二呢", r"(休|不播|没有|休息)"),
    ("这周你还播吗", r"(20:00|20点|晚8点|8点|周三|周四|周日)"),
    ("你平时几点开播", r"(20:00|20点|晚8点|8点|八点|14:00|14点)"),
]

_captured = {}
_ORIG = None
PLACEHOLDER = "（占位）"


async def _capture(messages):
    _captured["msgs"] = messages
    return PLACEHOLDER


def _strip_schedule(msgs):
    """只删【灰泽满的周表】那一段，其余一字不动。"""
    out = []
    for m in msgs:
        c = (m.get("content") or "").lstrip()
        if m.get("role") == "system" and c.startswith(SCHEDULE_MARK):
            continue
        out.append(m)
    return out


def _strip_traits(msgs):
    """删【性格基底】/【语言风格】（= messages[0] 里骨架之后的部分），用作对照。"""
    out = []
    for m in msgs:
        if m.get("role") == "system" and (m.get("content") or "").startswith(core.SYSTEM_PROMPT):
            m2 = dict(m)
            m2["content"] = core.SYSTEM_PROMPT
            out.append(m2)
        else:
            out.append(m)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="每个问法跑几次（防单次抖动）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="schedabl_")
    mem.use_storage(Path(tmp))
    global _ORIG
    _ORIG = core.generate_reply
    core.generate_reply = _capture

    print(f"📅【周表】消融：{len(CASES)} 个问法 × {args.runs} 次 × 2 个条件\n")
    rows = []
    for q, want in CASES:
        _captured.clear()
        await core.handle_chat("sched_demo", q)
        msgs = _captured.get("msgs")
        if not msgs:
            print(f"  [{q}] 取不到 messages，跳过"); continue
        has = any((m.get("content") or "").startswith(SCHEDULE_MARK)
                  for m in msgs if m.get("role") == "system")
        if not has:
            print(f"  ⚠️ [{q}] 这一轮 messages 里**没有**周表段，跳过"); continue
        pat = re.compile(want)
        for k in range(args.runs):
            r_with = await _ORIG([dict(m) for m in msgs])
            r_without = await _ORIG(_strip_schedule(msgs))
            rows.append({"q": q, "i": k, "with": r_with, "without": r_without,
                         "ok_with": bool(pat.search(r_with)), "ok_without": bool(pat.search(r_without))})
        w = sum(1 for r in rows if r["q"] == q and r["ok_with"])
        o = sum(1 for r in rows if r["q"] == q and r["ok_without"])
        print(f"  【{q}】有周表 {w}/{args.runs} 对，没有 {o}/{args.runs} 对")

    if not rows:
        print("\n❌ 一条都没跑成"); return 1
    n = len(rows)
    aw = sum(r["ok_with"] for r in rows)
    ao = sum(r["ok_without"] for r in rows)
    print(f"\n{'=' * 66}\n【结果】{n} 次生成\n")
    print(f"  有周表：答对 {aw}/{n} = {aw/n*100:.0f}%")
    print(f"  没周表：答对 {ao}/{n} = {ao/n*100:.0f}%")
    print(f"  → **周表的净作用 = +{(aw-ao)/n*100:.0f} 个百分点**")

    print("\n--- 答错的例子（看她是『不知道』还是『编一个』）---")
    bad = [r for r in rows if not r["ok_without"]][:4]
    for r in bad:
        print(f"  「{r['q']}」")
        print(f"     有周表：{r['with']}")
        print(f"     没周表：{r['without']}")

    out = ROOT / "outputs" / "eval" / "schedule_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": args.runs, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n✅ 明细已保存: {out}")
    print("\n⚠️ **怎么用这个数**：算『盈亏平衡频率』——周表每轮花 136 字（≈ base 的 3.8%）。")
    print("   若它每次被用上能挽回 +(X) 个百分点的答对率，则『每 N 轮被问到一次以上就划算』的 N 可以算出来；")
    print("   真实频率慢慢观察，**不用现在就赌**。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
