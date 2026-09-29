#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内容类注入段的"任务正确性"消融：**同一批 messages，只删掉那一段，看她还答不答得对。**

## 为什么用这套（而不是之前那些读数）

1. 前几版读数是「**字面重合**」（她有没有把那段的字抄进回复）—— 错。
   这些段的设计**就是"别抄"**（照抄才是故障），所以"差异≈0"是它们的正常状态。
2. 正确的问题是「**她那句话对不对**」，而**答案就在文件里** → **可以零成本自动判**。
3. **合成问法不构成循环论证**（本仓那条警告针对的是「我造场景 **+ 我造判据**」；
   这里判据是与问题无关的**客观事实**）。已由【周表】实验验证过（+88pp）。

## 判据怎么定（每个 case 一行）

    (问法, 正则)  —— 正则匹配她的回复即算"答对"。
    正则的词**全部来自那个文件的真实内容**（不是我编的期望）。

⚠️ **局限（先说清）**：
· 判据只测"**关键事实有没有出现**"，测不出"语气对不对"；
· 一条 case 只配对一段，**如果那段没被这一轮检索命中，会自动跳过并报出来**（不会假装测了）。

用法：
    python scripts/section_correctness.py                    # 全部段
    python scripts/section_correctness.py --section corpus
    python scripts/section_correctness.py --runs 3
"""
import argparse
import asyncio
import json
import re
import sys
import tempfile
from datetime import datetime
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

OUT = ROOT / "outputs" / "eval" / "section_correctness.json"

# ==================== 各段的 case 配置 ====================
# 正则里的词都来自对应文件的真实内容（**不是我编的期望**）。
_now = datetime.now()
_WD = "一二三四五六日"[_now.weekday()]
_H = _now.hour
# 正确的"现在时段"——她从【当前时间】拿到的就是澳洲当地时间（差 2 小时，见 context_probe）
_PERIOD = ("凌晨" if _H < 5 else "早上" if _H < 9 else "上午" if _H < 12
           else "中午" if _H < 14 else "下午" if _H < 18 else "晚上")

# ⚠️ **判据的铁律：它必须能失败。** 写之前先问：
#    "如果她答错了，这个正则会匹配吗？" —— 会匹配的，就不是判据，是噪声。
#    踩过的坑：第一版在正则里放了**话题词**（`猫|怕`、`绿冻`、`室友`），
#    于是"她答错也会通过"，两侧都 100%，看起来像"这段没用"。
#    修法：**只放"正确答案才会出现"的具体词**（具体食物名、定义要点、正确的时段）。
CASES = {
    "【当前时间】": [
        ("今天几号呀", rf"({_now.month}月{_now.day}[日号]|{_now.day}[日号])"),
        ("今天周几", rf"(周{_WD}|星期{_WD})"),
        # ⚠️ 她会写"七点多"这种**中文数字**，别只认阿拉伯数字（第一版就是这么漏的）
        ("现在几点了", rf"{_PERIOD}"),
    ],
    "【灰泽满的偏好】": [
        # 只认那两个**具体的**食物名（都在 preferences.json 的 text 里）
        ("你喜欢吃什么呀", r"(椰子鸡|火锅)"),
        ("你会玩宝可梦吗", r"(宝可梦|朱紫|剑盾|对战)"),
    ],
    "【她经历过的相关背景】": [
        ("你多高啊", r"(1米6|一米六)"),
        # 换成"事实唯一"的问法：corpus 里写的是**女**室友
        ("你室友是男的女的", r"(女|女生|姑娘)"),
    ],
    "【灰泽满的世界】": [
        # 定义要点：terms 里"绿冻"= 粉丝统称。**不放"绿冻"这个词**（两侧都会出现）
        ("绿冻是谁啊", r"(粉丝|统称|观众)"),
        ("乌色月是什么", r"(小说|她写|作品|写的)"),
    ],
}


def _strip(msgs: list, mark: str) -> list:
    """只删以该标题开头的那一段，其余一字不动。"""
    return [m for m in msgs
            if not (m.get("role") == "system" and (m.get("content") or "").lstrip().startswith(mark))]


_captured = {}
_ORIG = None


async def _capture(messages):
    _captured["msgs"] = messages
    return "（占位）"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default=None, choices=list(CASES))
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="seccorr_")
    mem.MEMORY_FILE = Path(tmp) / "short_term.json"
    global _ORIG
    _ORIG = core.generate_reply
    core.generate_reply = _capture

    names = [args.section] if args.section else list(CASES)
    all_rows, skipped = [], []
    for name in names:
        print(f"\n{'=' * 70}\n【{name}】")
        for q, want in CASES[name]:
            _captured.clear()
            try:
                await core.handle_chat("seccorr_demo", q)
            except Exception as e:
                print(f"  [{q}] 检索失败：{type(e).__name__}"); continue
            msgs = _captured.get("msgs")
            if not msgs:
                continue
            present = any((m.get("content") or "").lstrip().startswith(name)
                          for m in msgs if m.get("role") == "system")
            if not present:
                # ⚠️ 这一轮**没命中那一段** → 测不了，如实报出来（不许假装测过）
                print(f"  ⚠️「{q}」这一轮**没有**注入{name} → 跳过")
                skipped.append((name, q))
                continue
            pat = re.compile(want)
            for _ in range(args.runs):
                r_with = await _ORIG([dict(m) for m in msgs])
                r_without = await _ORIG(_strip(msgs, name))
                all_rows.append({"section": name, "q": q, "with": r_with, "without": r_without,
                                 "ok_with": bool(pat.search(r_with)),
                                 "ok_without": bool(pat.search(r_without))})
            same = [r for r in all_rows if r["section"] == name and r["q"] == q]
            print(f"  「{q}」有 {sum(r['ok_with'] for r in same)}/{args.runs}"
                  f"　没有 {sum(r['ok_without'] for r in same)}/{args.runs}")

    print(f"\n{'=' * 70}\n【汇总】\n")
    print(f"{'段':<22}{'次数':>6}{'有它答对':>10}{'没它答对':>10}{'净作用':>10}")
    print("-" * 60)
    for name in names:
        rs = [r for r in all_rows if r["section"] == name]
        if not rs:
            print(f"{name:<22}{'0':>6}　　— 这一轮一条都没命中，**没测到**")
            continue
        a = sum(r["ok_with"] for r in rs) / len(rs) * 100
        b = sum(r["ok_without"] for r in rs) / len(rs) * 100
        print(f"{name:<22}{len(rs):>6}{a:>9.0f}%{b:>9.0f}%{a-b:>+9.0f}pp")
    if skipped:
        print(f"\n⚠️ 因『这一轮没注入该段』而跳过的 {len(skipped)} 条（**它们没被测到**）：")
        for name, q in skipped:
            print(f"   · {name} ← 「{q}」")

    print("\n⚠️ 判据只测「关键事实有没有出现」，测不出语气；每条 case 只配对一段。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": all_rows, "skipped": skipped}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"✅ 明细已保存: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
