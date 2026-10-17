#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""19_daily 的行为验收：同一天晚些时候问起早先的事，她能不能正确接住。

判据（`docs/下一步.md`）：
  · 通过率 ≥ 70%（回复正确提到早先的事件，且当作"今天早些时候"说）
  · **"把旧事当现在"必须为 0**

做法：用 `DailyStore` 在**受控时间**里种入当天事件 → 取 `context()` →
走 `build_message_list` 取真实提示词 → 主模型生成回复 → 便宜 LLM 判标签。
不读线上记忆、不写任何数据文件。

用法：
    python scripts/daily_acceptance.py
    python scripts/daily_acceptance.py --runs 1
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from src.plugins.chatbot.daily_memory import DailyStore  # noqa: E402
from src.plugins.chatbot.core import build_message_list, generate_reply  # noqa: E402
from src.plugins.chatbot.config import (  # noqa: E402
    _get_clients, _get_model_name, extract_chat_content, parse_json_block,
)
from src.plugins.chatbot.constants import THINKING_DISABLED  # noqa: E402

BASE = datetime(2026, 10, 17, 9, 0)

CASES = [
    ("d1", [("早上跑了三公里", 9), ("中午吃了炸鸡", 12)], 17, "我现在腿酸得下不了楼", "跑步"),
    ("d2", [("下午睡了很久", 15), ("中午吃了炸鸡", 12)], 22, "我晚上怎么一点都睡不着", "睡"),
    ("d3", [("早上被楼上装修吵得去楼下坐着", 9), ("中午点了外卖", 12)], 16, "我今天一直好困", "装修"),
    ("d4", [("中午吃了炸鸡", 12), ("下午睡得很舒服", 15)], 18, "我肚子有点不舒服", "炸鸡"),
    ("d5", [("早上去上课", 9), ("中午吃了炒乌冬面", 12)], 19, "我现在有点撑", "乌冬"),
    ("d6", [("早上去上课", 9), ("下午去上体育课", 15)], 20, "今天好累啊", "上课"),
    ("d7", [("上午和朋友单挑打游戏", 10), ("中午吃了兰州拉面", 12)], 17, "今天玩得好爽", "游戏"),
    ("d8", [("早上坐火车去武汉旅游", 8), ("下午到了武汉", 15)], 20, "我现在腰酸背痛", "火车"),
    ("d9", [("下午去上体育课", 15), ("中午吃了炸鸡", 12)], 19, "我腿好酸", "体育课"),
    ("d10", [("中午吃了兰州拉面", 12), ("下午睡了很久", 15)], 20, "我晚上不太饿", "拉面"),
]

JUDGE_PROMPT = """用户在**今天早些时候**说过这些事：{events}
现在（同一天晚些时候）用户问：{query}
灰泽满回复：{reply}

判断回复属于哪一种：
- correct：回复里**明确出现了**早先事件里的具体内容（食物/动作/地点等），
  并把它当作"用户今天早些时候"的事（把早上/中午/下午的时段说错一点也算 correct——
  重点是"过去"，不是时段精度）
- ignore：没有出现任何早先事件的具体内容
- wrong_now：把早先的事当成"现在正在发生/还没发生"，或把旧事当成本轮现实
- wrong_owner：把**用户**早先的事说成**灰泽满自己**的（如"灰泽满今天中午吃了..."）

只输出 JSON：{{"label": "correct|ignore|wrong_now|wrong_owner"}}"""


async def judge(client, events, query, reply):
    prompt = JUDGE_PROMPT.format(events="；".join(events), query=query, reply=reply)
    try:
        r = await client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=20, **THINKING_DISABLED,
        )
        data = parse_json_block(extract_chat_content(r))
        return str(data.get("label") or "ignore")
    except Exception:
        # 判官失败按 ignore 处理（不误报 correct）
        return "ignore"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    deepseek_client, _ = _get_clients()

    labels = []
    for case_id, events, hour, query, _expect in CASES:
        with TemporaryDirectory() as td:
            store = DailyStore(base_dir=Path(td))
            for text, h in events:
                store.add("u1", [text], now=BASE.replace(hour=h))
            late = BASE.replace(hour=hour, minute=30)
            ctx = store.context("u1", now=late)
            assert ctx, f"{case_id} 没有生成 daily context"

        messages = build_message_list(query, [], [], daily_context=ctx, log_hits=False)
        reply = await generate_reply(messages)
        label = await judge(deepseek_client, [t for t, _ in events], query, reply)
        labels.append(label)
        print(f"[{case_id}] {label:<9} {query}  →  {reply[:40]}")

    total = len(labels)
    correct = labels.count("correct")
    wrong = labels.count("wrong_now")
    owner = labels.count("wrong_owner")
    print(f"\n通过率 {correct}/{total} = {correct / total * 100:.0f}% ｜ "
          f"把旧事当现在 {wrong} 条 ｜ 把用户的事说成她自己的 {owner} 条")
    ok = correct / total >= 0.70 and wrong == 0 and owner == 0
    print("✅ 验收通过" if ok else "❌ 验收未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
