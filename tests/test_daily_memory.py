# -*- coding: utf-8 -*-
"""每日记忆（第 20 块）：一天一批的用户日常。

盯的是设计时定死的那几条（见 `daily_memory.py` 模块说明）：
  ① 入库闸门在**代码**里（状态句、讲她自己的、段子起手一律不收），不靠模型自觉；
  ② 时点存**绝对时刻**、渲染成「今天/昨天 + 早上9点」；
  ③ 日界（3 点）只管**清理**，"今天/昨天"按**自然日**判——两者解耦；
  ④ 条目太少不注（宁可不提）。
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plugins.chatbot import daily_memory as dm  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return dm.DailyStore(base_dir=tmp_path)


# ---------------------------------------------------------------- 入库闸门

@pytest.mark.parametrize("text", [
    "中午吃了炸鸡",
    "下午的课取消了",
    "八点去上课",
    "晚上去操场跑了三公里",
    "凌晨四点才睡",
])
def test_gate_keeps_real_events(store, text):
    assert store.add("u1", [text], now=datetime(2026, 10, 17, 12, 0)) == 1


@pytest.mark.parametrize("text", [
    "他最近很累",              # 状态，归长期记忆
    "不喜欢吃辣",              # 喜好
    "和室友关系一般",          # 关系
    "灰泽满今天也在写作业",     # 讲的是她，不是用户
    "我重新介绍一下自己，我今年四十",  # 网上段子起手
    "",                        # 空
    "他挺好的",                # 没有事件
])
def test_gate_drops_non_events(store, text):
    assert store.add("u1", [text], now=datetime(2026, 10, 17, 12, 0)) == 0
    assert store.context("u1", now=datetime(2026, 10, 17, 12, 0)) == ""


def test_gate_dedupes_same_day(store):
    now = datetime(2026, 10, 17, 12, 0)
    assert store.add("u1", ["中午吃了炸鸡"], now=now) == 1
    assert store.add("u1", ["中午吃了炸鸡"], now=now) == 0


def test_entries_are_capped_per_day(store):
    now = datetime(2026, 10, 17, 22, 0)
    store.add("u1", [f"第{i}次去上课" for i in range(10)], now=now)
    raw = store._load("u1")["entries"]
    assert len(raw) == dm.MAX_ENTRIES_PER_DAY


# ---------------------------------------------------------------- 日界与清理

def test_day_key_puts_early_morning_on_previous_day():
    """日界是 3 点：凌晨 1 点算前一天，凌晨 4 点算新一天。"""
    assert dm._day_key(datetime(2026, 10, 17, 1, 0)) == dm._day_key(datetime(2026, 10, 16, 23, 0))
    assert dm._day_key(datetime(2026, 10, 17, 4, 0)) != dm._day_key(datetime(2026, 10, 16, 23, 0))


def test_cleanup_drops_previous_day_on_read(store):
    """跨过凌晨 3 点后读一次，前一天整批消失（惰性清理，不挂定时器）。"""
    store.add("u1", ["中午吃了炸鸡"], now=datetime(2026, 10, 16, 12, 0))
    store.add("u1", ["晚上去跑步了"], now=datetime(2026, 10, 16, 20, 0))
    assert store._load("u1")   # 写进去了
    # 10-17 凌晨 2 点：还属于 10-16 那一天，两条都在
    assert store.context("u1", now=datetime(2026, 10, 17, 2, 0)) != ""
    # 10-17 凌晨 3 点零一分：日界过了，10-16 整批清掉
    assert store.context("u1", now=datetime(2026, 10, 17, 3, 1)) == ""
    assert store._load("u1")["entries"] == []


def test_cleanup_keeps_same_day_early_morning_entries(store):
    """凌晨 1 点写的那条属于"前一天"那一批——上午 3 点前读还在，3 点后被清。"""
    store.add("u1", ["凌晨还在写作业"], now=datetime(2026, 10, 17, 1, 0))
    store.add("u1", ["早上吃了包子"], now=datetime(2026, 10, 17, 1, 30))
    assert store.context("u1", now=datetime(2026, 10, 17, 2, 0)) != ""
    assert store.context("u1", now=datetime(2026, 10, 17, 3, 30)) == ""


# ---------------------------------------------------------------- 渲染

def test_render_uses_absolute_time_never_relative_words():
    """条目存的是**绝对时刻**，渲染才换算——存相对词会随注入时刻漂移。"""
    now = datetime(2026, 10, 17, 20, 0)
    assert dm.render_entry({"t": "2026-10-17T09:00:00", "text": "早上吃了包子"}, now) \
        == "今天早上9点：早上吃了包子"
    assert dm.render_entry({"t": "2026-10-16T16:00:00", "text": "下午的课取消了"}, now) \
        == "昨天下午4点：下午的课取消了"


def test_day_label_is_natural_day_not_boundary():
    """措辞按**自然日**判：凌晨 1 点在当天上午看是"今天"，不是"昨天"。"""
    assert dm._day_label(datetime(2026, 10, 17, 1, 0), datetime(2026, 10, 17, 10, 0)) == "今天"


# ---------------------------------------------------------------- 注入条件

def test_too_few_entries_means_no_injection(store):
    """宁可不提：只有一条时不注。"""
    now = datetime(2026, 10, 17, 15, 0)
    store.add("u1", ["中午吃了炸鸡"], now=now)
    assert store.context("u1", now=now) == ""
    store.add("u1", ["下午去上了课"], now=now)
    assert store.context("u1", now=now) != ""


def test_context_is_char_bounded(store):
    now = datetime(2026, 10, 17, 22, 0)
    store.add("u1", [f"第{i}次去这个很长很长很长很长的地方上课" for i in range(6)], now=now)
    assert len(store.context("u1", now=now)) <= dm.MAX_CONTEXT_CHARS


def test_corrupt_file_is_not_a_crash(tmp_path):
    (tmp_path / "u1.json").write_text("{ not json", encoding="utf-8")
    store = dm.DailyStore(base_dir=tmp_path)
    assert store.context("u1") == ""
    assert store.add("u1", ["中午吃了炸鸡"]) == 1