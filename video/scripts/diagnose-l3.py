#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01 诊断：L3 行为分类对**整批合成文本**判不出来时，看单条会判成什么。

规格 §10.3 预期「behavior：用户要求她做一件事/反复索要的情景」应当亮起，
但读秒窗口把三条合成一段文本后 L3 返回空。本脚本只做一次性诊断，不产出交付物。

    python video/scripts/diagnose-l3.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402

nonebot.init()

from scripts import _common  # noqa: E402
from src.plugins.chatbot import core, routing  # noqa: E402
from src.plugins.chatbot.config import _get_clients  # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules  # noqa: E402

BATCH = [
    "满神，我这边还在下雨，刚下班",
    "你今晚几点播来着？",
    "下播后念一段《乌色月》给我听呗，这都第三遍了",
]
HISTORY = (
    "用户：在吗满神\n灰泽满：在呢，怎么了\n"
    "用户：能念一段乌色月给我听吗\n灰泽满：……那个还没写完呢，念什么念。以后再说\n"
    "用户：可以念一段乌色月给我听吗\n灰泽满：灰泽满自己都没敢回头看那段，念出来多丢人"
)

CASES = [
    ("整批（换行连接，生产真实形态）", "\n".join(BATCH)),
    ("只第三条（反复索要那句）", BATCH[2]),
    ("只第三条（去掉遍数）", "下播后念一段《乌色月》给我听呗"),
    ("只第二条（问几点播）", BATCH[1]),
    ("只第一条（刚下班+下雨）", BATCH[0]),
    ("第一+第三条", "\n".join([BATCH[0], BATCH[2]])),
]


async def main() -> int:
    _common.ensure_utf8_stdout()
    behaviors = load_persona_rules()
    ds, _ = _get_clients()
    print(f"behaviors = {len(behaviors)} 条\n")
    for label, msg in CASES:
        l3 = await routing.classify_l3(ds, msg, HISTORY, behaviors)
        item = core.select_behavior_item(msg, l3["behavior"], behaviors)
        print(f"【{label}】")
        print(f"   输入 : {msg!r}")
        print(f"   LLM  : behavior={l3['behavior']!r}  same_request={l3.get('same_request')}")
        print(f"   兜底 : {'命中 ' + str(item.item_id) if item else '无'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
