#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""位置对照实验：**同一段内容，放在 messages 的不同位置，会怎样。**

## 为什么做

`build_message_list` 里 18 段的**顺序是历史演化出来的，从没测过**——
没人知道【周表】放前面和放后面有没有区别、【背景记忆】该靠近用户那句还是远离。
外部资料（SillyTavern 的 World Info）有 `insertion order` / `@Depth` 这类旋钮，
说明位置确实起作用；但我们连"起不起作用"都没量过。

## 设计

- **只变位置这一个变量**：内容、形式、其余全部相同（都走真实链路 `build_message_list`）
- 两条通道各测一遍：
  - **system 通道**：同一段 system 文本，插在 **最前 / 中间 / 最后**
  - **assistant 通道**：同一个问答对，放在 **很早 / 很晚**（assistant turn 必须排在最后那句 user 之前）
- 观测量：**采用率**（进得去吗）+ **逐字率**（照抄吗）

用法：
    python scripts/position_experiment.py            # 默认 N=16
    python scripts/position_experiment.py --n 8
    python scripts/position_experiment.py --dry      # 只看会插在哪，不生成
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from src.plugins.chatbot import core  # noqa: E402

QUERY = "你最喜欢哪个数字啊"
WANT = ["7", "七"]
SIGNATURE = "一直排第一"

SYS_TEXT = "【关于灰泽满】被问到最喜欢的数字时，她会说 7，一直排第一。"
QA_USER, QA_REPLY = "你最喜欢哪个数字啊", "7 啊，一直排第一"
RHYTHM = {"role": "system", "content": "【回复节奏】日常闲聊一句话说完就停，别硬凑第二、第三句长段。"}


def base_msgs() -> list:
    """真实链路搭好的 messages（不含测试内容）。最后一条是用户那句话。"""
    return core.build_message_list(QUERY, [], "", [])


def at_system(where: str) -> list:
    msgs = base_msgs()
    layer = {"role": "system", "content": SYS_TEXT}
    body = msgs[:-1]                                   # 除最后那句 user 之外
    if where == "最前":
        return msgs[:1] + [layer] + body[1:] + [msgs[-1]]
    if where == "中间":
        mid = max(1, len(body) // 2)
        return body[:mid] + [layer] + body[mid:] + [msgs[-1]]
    return body + [layer, msgs[-1]]                    # 最后：紧挨用户那句


def at_assistant(where: str) -> list:
    msgs = base_msgs()
    pair = [{"role": "user", "content": QA_USER},
            {"role": "assistant", "content": QA_REPLY}, RHYTHM]
    body = msgs[:-1]
    if where == "很早":
        return msgs[:1] + pair + body[1:] + [msgs[-1]]
    return body + pair + [msgs[-1]]                    # 很晚：紧挨用户那句


CONDITIONS = {
    "system · 最前": lambda: at_system("最前"),
    "system · 中间": lambda: at_system("中间"),
    "system · 最后": lambda: at_system("最后"),
    "assistant · 很早": lambda: at_assistant("很早"),
    "assistant · 很晚": lambda: at_assistant("很晚"),
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.dry:
        for name, fn in CONDITIONS.items():
            msgs = fn()
            pos = next(i for i, m in enumerate(msgs) if SYS_TEXT in m["content"] or QA_REPLY in m["content"])
            print(f"{name:<18} 总 {len(msgs)} 段 → 插在第 {pos+1} 段（末尾是用户那句）")
        return

    print(f"\n提问：「{QUERY}」  每种位置各生成 {args.n} 次")
    print(f"内容：{SYS_TEXT}\n")
    results = {}
    for name, fn in CONDITIONS.items():
        sem = asyncio.Semaphore(5)

        async def one():
            async with sem:
                return await core.generate_reply(fn())

        replies = list(await asyncio.gather(*[one() for _ in range(args.n)]))
        used = sum(1 for r in replies if any(w in r for w in WANT))
        copied = sum(1 for r in replies if SIGNATURE in r)
        results[name] = {"used": used, "copied": copied, "replies": replies}
        print(f"--- {name}")
        print(f"    采用 {used}/{args.n}   逐字抄 {copied}/{args.n}")
        for r in replies[:3]:
            print(f"      · {r[:56]}")

    out = ROOT / "outputs" / "_position_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": args.n, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n{'位置':<18}{'采用率':>9}{'逐字率':>9}")
    print("-" * 36)
    for k, v in results.items():
        print(f"{k:<18}{v['used']}/{args.n:<7}{v['copied']}/{args.n}")
    print(f"\n✅ 原始回复已存 {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
