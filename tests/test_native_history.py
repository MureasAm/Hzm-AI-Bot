# -*- coding: utf-8 -*-
"""私聊历史走 assistant 原生轮次（2026-11-19）。"""
from datetime import datetime

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


@pytest.mark.parametrize(("question", "old_reply"), [
    ("几点？", "凌晨3点了还不睡"),
    ("你那里现在几点几分？", "凌晨3点了还不睡"),
    ("现在几点了", "墨尔本现在凌晨2点34分"),
])
async def test_clock_question_uses_live_time_after_native_history(monkeypatch, question, old_reply):
    monkeypatch.setenv("INJECTION_V3", "1")
    monkeypatch.setenv("HISTORY_AS_ASSISTANT", "1")
    monkeypatch.setattr(core, "_get_clients", lambda: (None, None))
    monkeypatch.setattr(core.memory_v2, "injection_enabled", lambda: False)
    monkeypatch.setattr(core.memory_v2, "shadow_enabled", lambda: False)

    async def fake_probe(user_id, msg, *args, **kwargs):
        assert msg == question
        return msg, {}

    async def fake_retrieval(*args, **kwargs):
        return {"fused_items": [], "preference_items": [], "core_stories": []}

    async def no_weather(city):
        pass

    monkeypatch.setattr(core, "probe_session", fake_probe)
    monkeypatch.setattr(core, "gather_retrieval", fake_retrieval)
    monkeypatch.setattr(core.context_probe, "warm_weather", no_weather)
    monkeypatch.setattr(core.context_probe, "_cst", lambda now: datetime(
        2026, 11, 25, 23, 34, tzinfo=core.context_probe._CST))

    memory.append_user_history("u1", "还没睡", old_reply)
    requests = []

    async def fake_reply(messages):
        requests.append(messages)
        if len(requests) > 1:
            return "快去睡吧"
        return "墨尔本现在凌晨2点34分"

    monkeypatch.setattr(core, "generate_reply", fake_reply)
    timed_transcript = f"[北京时间 11月25日 23:33] {question}"
    reply = await core.handle_chat(
        "u1", question, user_transcript=question, timed_user_transcript=timed_transcript,
    )

    messages = requests[0]
    assistant_index = messages.index({"role": "assistant", "content": old_reply})
    assert messages[-1] == {"role": "user", "content": timed_transcript}
    assert any(m["role"] == "system" and "北京时间 11月25日 周三 23:34" in m["content"]
               and "墨尔本当地时间 11月26日 周四 02:34" in m["content"]
               and "报到分钟" in m["content"]
               for m in messages[assistant_index + 1:-1])
    assert len(requests) == 1
    assert reply == "墨尔本现在凌晨2点34分"
    assert memory.get_user_history("u1")[-2] == f"用户：{question}"


async def test_annotate_quotes_without_quote_returns_unchanged():
    assert await self_speech.annotate_quotes("在吗") == "在吗"


def test_sub_quotes_rewrites_self_quote():
    msg = "[引用灰泽满说的：「少来」] 你又来了"
    out = self_speech._sub_quotes(msg, {"少来": "她让对方别来这套"})
    assert "对方引用了你之前说过的话" in out
    assert "少来" not in out
