# -*- coding: utf-8 -*-
"""私聊历史走 assistant 原生轮次（2026-11-19）。"""
import pytest

from src.plugins.chatbot import core, memory, self_speech


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path):
    memory.use_storage(tmp_path)
    yield


def test_native_turns_roles_and_labels():
    memory.append_user_history("u1", "在吗", "在呢")
    turns = core.build_native_history_turns("u1")
    assert turns == [
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": "在呢"},
    ]


def test_native_turns_skip_fallback_and_merge_same_role():
    memory.append_user_history("u1", "在吗", "")
    memory.append_bot_message("u1", core._FALLBACK_REPLY)
    memory.append_bot_message("u1", "刚发了个动态")
    turns = core.build_native_history_turns("u1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "小满现在不在" not in turns[-1]["content"]
    assert "刚发了个动态" in turns[-1]["content"]


def test_build_message_list_puts_native_turns_before_final_user():
    history = [
        {"role": "user", "content": "早"},
        {"role": "assistant", "content": "早啊"},
    ]
    msgs = core.build_message_list("今天也去跑操吗", [], [], history_turns=history)
    assert msgs[-1] == {"role": "user", "content": "今天也去跑操吗"}
    assert any(m["content"].startswith("【最近对话记录】")
               and "按时间顺序" in m["content"] for m in msgs
               if m["role"] == "system")
    assert {"role": "user", "content": "早"} in msgs
    assert {"role": "assistant", "content": "早啊"} in msgs
    assert any("【别自相矛盾】" in m["content"] for m in msgs
               if m["role"] == "system")


def test_history_flag_defaults_on_and_can_roll_back(monkeypatch):
    monkeypatch.delenv("HISTORY_AS_ASSISTANT", raising=False)
    assert core.history_as_assistant_enabled() is True
    monkeypatch.setenv("HISTORY_AS_ASSISTANT", "0")
    assert core.history_as_assistant_enabled() is False


async def test_annotate_quotes_without_quote_returns_unchanged():
    assert await self_speech.annotate_quotes("在吗") == "在吗"


def test_sub_quotes_rewrites_self_quote():
    msg = "[引用灰泽满说的：「少来」] 你又来了"
    out = self_speech._sub_quotes(msg, {"少来": "她让对方别来这套"})
    assert "对方引用了你之前说过的话" in out
    assert "少来" not in out
