# -*- coding: utf-8 -*-
"""成本优化闸门：只跳明显不值得提取的轮次。"""
from src.plugins.chatbot.memory_v2 import worth_extracting


def test_skip_short_chatter_and_emoji():
    for t in ("", "好的", "好耶", "晚安", "🐖", "[表情：可怜]", "我想你"):
        assert not worth_extracting(t), t


def test_keep_name_preference_event_fact():
    for t in ("我叫小温", "每天晚安都要有仪式感", "不学了，吃饭去",
              "我25岁", "我和室友吵架了", "我失业了", "在健身房看花礼哈雷直播"):
        assert worth_extracting(t), t


def test_long_message_always_kept():
    assert worth_extracting("今天发生了一件很复杂的事，我自己也说不清楚，反正就是很烦")
