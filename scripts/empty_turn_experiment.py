#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""空轮次实验：**六路都搜不到的轮次，给她垫一条风格样本，会不会更像她。**

## 为什么做

实测**六路全空占 45% 的轮次**——将近一半的对话里，模型手里只有 2974 字的骨架、一条素材都没有。
那时她只能靠规则答，容易变成"一个遵守规则的人"而不是"她"。

`voice_samples` 本来有保底机制，但门槛 0.66（宁断档不错话题）→ 实测开火率只有 14%。
本实验问的是：**把门槛放开到"随便垫一条"，风格会不会更像她？代价是什么？**

## 怎么量"像不像"

"像"没有客观答案，所以用**她身上可量化的说话特征**当代理（这些都是她有明确特征的）：

| 指标 | 她的特征 | 判据 |
|---|---|---|
| **自称** | 习惯用「灰泽满/hzm」自称（第三人称自我保护），少用「我」 | 自称率↑ 好 |
| **长度** | 短句为主 | 别明显变长 |
| **括号** | 「例外不是常态」，日常闲聊基本不用 | 括号率↓ 好 |
| **感叹号** | 情绪克制 | 感叹号率↓ 好 |

⚠️ 这些只是**代理指标**，最终还是要人看。但代理能先看出方向。

用法：
    python scripts/empty_turn_experiment.py --n 16
    python scripts/empty_turn_experiment.py --q "今天在干嘛呀"
"""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from src.plugins.chatbot import core  # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules, build_global_persona_context  # noqa: E402
from src.plugins.chatbot.retrieval import load_voice_sample_vectors  # noqa: E402

SAMPLE_HEAD = ("【灰泽满的说话方式参考】以下是她真实的对话片段。只学其中的语气、断句、自称"
               "（灰泽满/hzm）和措辞。内容要针对当前话题，不要复述、也不要套用示例里的具体内容。")
RHYTHM = {"role": "system", "content": "【回复节奏】日常闲聊一句话说完就停，别硬凑第二、第三句长段。"}


def base_msgs(q: str) -> list:
    traits, styles, _ = load_persona_rules()
    gp = build_global_persona_context(traits, styles)
    return core.build_message_list(q, gp, [], "", [])


def with_samples(msgs: list, k: int, seed: int) -> list:
    """垫 k 条**随机**样本（不做相关性判断，纯风格示范）。"""
    pool = load_voice_sample_vectors()
    if not pool or k <= 0:
        return msgs
    rnd = random.Random(seed)
    picked = rnd.sample(pool, min(k, len(pool)))
    extra = [{"role": "system", "content": SAMPLE_HEAD}]
    for s in picked:
        extra += [{"role": "user", "content": s.get("user", "")},
                  {"role": "assistant", "content": s.get("reply", "")}]
    return msgs[:-1] + extra + [RHYTHM, msgs[-1]]


def metrics(texts: list) -> dict:
    n = len(texts)
    selfref = sum(1 for t in texts if ("灰泽满" in t or "hzm" in t))
    first = sum(1 for t in texts if "我" in t)
    paren = sum(1 for t in texts if "（" in t)
    bang = sum(1 for t in texts if "！" in t)
    avg = sum(len(t) for t in texts) / max(n, 1)
    return {"自称率": selfref / n, "「我」率": first / n, "括号率": paren / n,
            "感叹号率": bang / n, "平均长度": avg}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="今天在干嘛呀", help="用一个六路都搜不到的问题")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    conds = {"① 现状（不垫样本）": 0, "② 垫 1 条随机样本": 1, "③ 垫 2 条随机样本": 2}
    print(f"\n提问：「{args.q}」  各 {args.n} 次\n")
    results = {}
    for name, k in conds.items():
        sem = asyncio.Semaphore(5)

        async def one(idx):
            async with sem:
                return await core.generate_reply(with_samples(base_msgs(args.q), k, seed=idx))

        replies = list(await asyncio.gather(*[one(i) for i in range(args.n)]))
        m = metrics(replies)
        results[name] = {**m, "replies": replies}
        print(f"--- {name}")
        print(f"    自称率 {m['自称率']:.0%}  「我」率 {m['「我」率']:.0%}"
              f"  括号率 {m['括号率']:.0%}  感叹号率 {m['感叹号率']:.0%}  平均 {m['平均长度']:.1f} 字")
        for r in replies[:3]:
            print(f"      · {r[:52]}")

    print(f"\n{'条件':<22}{'自称率':>8}{'我率':>7}{'括号率':>8}{'叹号率':>8}{'均长':>7}")
    print("-" * 62)
    for k, v in results.items():
        print(f"{k:<22}{v['自称率']:>7.0%}{v['「我」率']:>7.0%}{v['括号率']:>8.0%}"
              f"{v['感叹号率']:>8.0%}{v['平均长度']:>7.1f}")

    out = ROOT / "outputs" / "_empty_turn_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"q": args.q, "n": args.n, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 原始回复已存 {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
