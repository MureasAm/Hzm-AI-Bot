#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""显式触发 → 系统侧预取（invocation policy ② 的 fail-open 部分）。

只认**言语行为**，不认话题：
- 回忆/身份
- 称呼协商
- 天气/冷暖
- 今日日常
- 精确时间/农历

命中后直接调 `recall`，把结果作为一条 `【预取】` system 消息注入；
不伪造 tool 消息，所以没有工具能力的降级层也能用。
"""
from __future__ import annotations

import json
import re

from . import recall as recall_mod

_RECALL_RE = re.compile(r"(我叫什么|我叫啥|我是谁|记得我|还记得我|你还记得|你记得我|我名字)")
_ADDRESS_RE = re.compile(r"(叫我|称呼我|怎么称呼|我想亲昵点|叫我什么|叫我啥)")
_WEATHER_RE = re.compile(
    r"(天气|几度|多少度|冷不冷|冷吗|热不热|热吗|凉不凉|凉吗|暖和吗|"
    r"下雨|下雪|刮风|风大|风很|降温|升温|气温|寒潮)(?<!角度)"
)
_TODAY_RE = re.compile(r"(我今天|我早上|我中午|我晚上|我今天干嘛|我今天干啥|吃了什么|干啥了|做了什么)")
_PERCEPTION_RE = re.compile(r"(现在几点|几点了|你那边几点|农历|初几|节气|什么日子)")


def _weather_subject(msg: str) -> str:
    if any(k in msg for k in ("你那边", "你那里", "你那", "澳洲", "墨尔本")):
        return "her"
    if any(k in msg for k in ("我这边", "我这里", "我这", "我们这")):
        return "user"
    return ""


def requests_for(msg: str) -> list[dict]:
    """返回本轮需要预取的 recall 请求；空 = 不是显式场景。"""
    m = msg or ""
    out: list[dict] = []
    if _RECALL_RE.search(m):
        out.append({"kind": "fact", "subject": "user", "query": "用户的名字、身份和基本情况"})
    if _ADDRESS_RE.search(m):
        out.append({"kind": "fact", "subject": "user", "query": "用户希望被怎么称呼"})
        out.append({"kind": "interaction_preference", "subject": "user",
                    "query": "称呼和相处方式"})
    if _WEATHER_RE.search(m):
        out.append({"kind": "weather", "subject": _weather_subject(m),
                    "when": "now", "query": "现在天气"})
    if _TODAY_RE.search(m):
        out.append({"kind": "daily", "subject": "user", "when": "today",
                    "query": "用户今天的日常事件"})
    if _PERCEPTION_RE.search(m):
        out.append({"kind": "perception", "when": "now", "query": "现在几点、农历日期"})
    return out


async def prefetch(*, user_id: str, msg: str, store, user_city: str,
                   deepseek_client=None, zhipu_client=None, is_group: bool = False,
                   now=None) -> tuple[str, list]:
    """命中显式场景就取数；返回 (渲染好的 system 文本, 条目)。失败/空返回空串。"""
    reqs = requests_for(msg)
    if not reqs:
        return "", []
    items, seen = [], set()
    for r in reqs:
        try:
            got = await recall_mod.recall(
                user_id=user_id, query=r.get("query") or "", kind=r.get("kind"),
                subject=r.get("subject"), when=r.get("when"), limit=3,
                now=now, store=store, user_city=user_city,
                deepseek_client=deepseek_client, zhipu_client=zhipu_client,
                is_group=is_group,
            )
        except Exception:
            continue
        for it in got:
            if it.id in seen:
                continue
            seen.add(it.id)
            items.append(it)
            if len(items) >= 3:
                break
        if len(items) >= 3:
            break
    if not items:
        return "", []
    body = json.dumps([it.as_dict() for it in items], ensure_ascii=False)
    text = (f"【预取】{body}\n"
            f"（这些是用户这一轮明确问到的，直接用；没有的不要编。）")
    return text[:400], items


__all__ = ["requests_for", "prefetch"]