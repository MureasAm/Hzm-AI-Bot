# -*- coding: utf-8 -*-
"""recall 工具后端：过滤、owner、时间、空结果、群聊范围。"""
import asyncio
from datetime import datetime

import pytest

from src.plugins.chatbot import recall as recall_mod


class FakeStore:
    def __init__(self, memories):
        self.memories = memories

    def _active_memories(self, user_id, now=None, session_id=None):
        return list(self.memories)


def _mem(kind, key, value, **kw):
    d = {"id": f"{kind}:{key}", "kind": kind, "key": key, "value": value,
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True}
    d.update(kw)
    return d


def _run(coro):
    return asyncio.run(coro)


def test_fact_name_by_intent():
    store = FakeStore([_mem("fact", "preferred_name", "小温")])
    items = _run(recall_mod.recall(user_id="u", query="我叫什么", kind="fact",
                                   subject="user", store=store))
    assert [it.text for it in items] == ["preferred_name：小温"]
    assert items[0].kind == "fact" and items[0].subject == "user"


def test_fact_without_hint_does_not_leak():
    store = FakeStore([_mem("fact", "preferred_name", "小温"),
                       _mem("fact", "favorite_color", "蓝")])
    items = _run(recall_mod.recall(user_id="u", query="今天好累", kind="fact",
                                   subject="user", store=store))
    assert items == []


def test_natural_recall_question_keeps_user_subject():
    """"你记得我叫什么"里的“你”不是在问她自己的名字，不能把 subject 切成 her。"""
    store = FakeStore([_mem("fact", "preferred_name", "小温")])
    items = _run(recall_mod.recall(user_id="u", query="你记得我叫什么", kind="fact",
                                   store=store))
    assert [it.text for it in items] == ["preferred_name：小温"]


def test_when_now_filters_stable_fact():
    store = FakeStore([_mem("fact", "preferred_name", "小温")])
    items = _run(recall_mod.recall(user_id="u", query="我叫什么", kind="fact",
                                   subject="user", when="now", store=store))
    assert items == []


def test_group_does_not_read_user_facts():
    store = FakeStore([_mem("fact", "preferred_name", "小温")])
    items = _run(recall_mod.recall(user_id="group", query="我叫什么", kind="fact",
                                   subject="user", store=store, is_group=True))
    assert items == []


def test_weather_requires_known_owner(monkeypatch):
    async def noop(city=""):
        return None
    monkeypatch.setattr(recall_mod.context_probe, "warm_weather", noop)
    monkeypatch.setattr(recall_mod.context_probe, "weather_now", lambda city="": "天气：晴 9℃")

    # 对方城市未知：不能拿她的城市冒充
    items = _run(recall_mod.recall(user_id="u", query="我这边冷吗", kind="weather",
                                   subject="user", user_city=""))
    assert items == []

    items = _run(recall_mod.recall(user_id="u", query="我这边冷吗", kind="weather",
                                   subject="user", user_city="杭州"))
    assert len(items) == 1 and items[0].subject == "user"

    items = _run(recall_mod.recall(user_id="u", query="你那边冷吗", kind="weather",
                                   subject="her"))
    assert len(items) == 1 and items[0].subject == "her"


def test_daily_items(monkeypatch):
    now = datetime(2026, 10, 27, 9, 0)
    monkeypatch.setattr(recall_mod.daily_memory, "recall_items",
                        lambda user_id, now=None: [{"t": now.isoformat(), "text": "跑了三公里"}])
    items = _run(recall_mod.recall(user_id="u", query="我今天干嘛了", kind="daily",
                                   subject="user", when="today", now=now))
    assert len(items) == 1
    assert "跑了三公里" in items[0].text


def test_term_lookup(monkeypatch):
    monkeypatch.setattr(recall_mod.persona, "load_terms", lambda: [
        {"keyword": "枪神8", "meaning": "一台游戏本", "priority": "on-demand"},
    ])
    items = _run(recall_mod.recall(user_id="u", query="枪神8怎么样", kind="term"))
    assert len(items) == 1 and "游戏本" in items[0].text


def test_preference_lookup(monkeypatch):
    monkeypatch.setattr(recall_mod.retrieval, "retrieve_preferences",
                        lambda q, **kw: [{"id": "food", "category": "食物",
                                          "text": "爱吃火锅", "score": 1.0}])
    items = _run(recall_mod.recall(user_id="u", query="你爱吃什么", kind="preference"))
    assert len(items) == 1 and "火锅" in items[0].text


def test_empty_tool_result_is_explicit():
    text, items = _run(recall_mod.recall_for_tool(
        {"query": ""}, user_id="u", store=FakeStore([]), user_city=""))
    assert items == []
    assert "没有查到" in text