# -*- coding: utf-8 -*-
"""工具轮：一次模型调用是主回复，只有调工具才多一跳；失败可回落。"""
import asyncio
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import config, tool_loop


def _resp(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def _call(i, args='{"query":"名字"}'):
    return SimpleNamespace(id=f"call_{i}",
                           function=SimpleNamespace(name="recall", arguments=args))


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def _run(coro):
    return asyncio.run(coro)


def _kwargs():
    return dict(temperature=0.6, frequency_penalty=0.0, max_tokens=150)


def test_no_tool_call_is_one_call(monkeypatch):
    client = FakeClient([_resp(content="直接回答")])
    monkeypatch.setattr(tool_loop, "_get_clients", lambda: (client, None))
    monkeypatch.setattr(tool_loop, "log_cache_usage", lambda *a, **k: None)
    out = _run(tool_loop.generate_with_tools([{"role": "user", "content": "hi"}],
                                             executor=lambda *a: None, **_kwargs()))
    assert out == "直接回答"
    assert len(client.chat.completions.calls) == 1


def test_tool_call_adds_second_call_and_system_result(monkeypatch):
    client = FakeClient([_resp(tool_calls=[_call(1)]), _resp(content="好的")])
    monkeypatch.setattr(tool_loop, "_get_clients", lambda: (client, None))
    monkeypatch.setattr(tool_loop, "log_cache_usage", lambda *a, **k: None)
    seen = []

    async def executor(name, args, call_id):
        seen.append((name, args, call_id))
        return '{"items":[{"text":"小温"}]}'

    out = _run(tool_loop.generate_with_tools([{"role": "user", "content": "hi"}],
                                             executor=executor, **_kwargs()))
    assert out == "好的"
    assert seen == [("recall", '{"query":"名字"}', "call_1")]
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert "tools" in calls[0] and "tools" not in calls[1]
    msgs = calls[1]["messages"]
    # 结果走 system（Gemini 兼容层会忽略 role:"tool"），且放在用户消息前
    assert msgs[-1]["role"] == "user"
    assert msgs[-2]["role"] == "system"
    assert msgs[-2]["content"].startswith("【工具结果】")
    assert "小温" in msgs[-2]["content"]
    assert not any(m.get("role") == "tool" for m in msgs)


def test_tool_failure_still_generates(monkeypatch):
    client = FakeClient([_resp(tool_calls=[_call(1)]), _resp(content="查不到也没关系")])
    monkeypatch.setattr(tool_loop, "_get_clients", lambda: (client, None))
    monkeypatch.setattr(tool_loop, "log_cache_usage", lambda *a, **k: None)

    async def executor(name, args, call_id):
        raise TimeoutError()

    out = _run(tool_loop.generate_with_tools([{"role": "user", "content": "hi"}],
                                             executor=executor, **_kwargs()))
    assert out == "查不到也没关系"
    system_msgs = [m for m in client.chat.completions.calls[1]["messages"]
                   if m.get("role") == "system"]
    assert "没有查到" in system_msgs[-1]["content"]


def test_at_most_two_tool_calls(monkeypatch):
    client = FakeClient([_resp(tool_calls=[_call(1), _call(2), _call(3)]), _resp(content="ok")])
    monkeypatch.setattr(tool_loop, "_get_clients", lambda: (client, None))
    monkeypatch.setattr(tool_loop, "log_cache_usage", lambda *a, **k: None)
    seen = []

    async def executor(name, args, call_id):
        seen.append(call_id)
        return "ok"

    _run(tool_loop.generate_with_tools([{"role": "user", "content": "hi"}],
                                       executor=executor, **_kwargs()))
    assert seen == ["call_1", "call_2"]


def test_bad_shape_accepts_tool_calls():
    resp = _resp(content=None, tool_calls=[_call(1)])
    assert config._bad_shape_reason(resp) == ""


def test_tool_capable_whitelist(monkeypatch):
    tool_loop._capability_cache.clear()
    monkeypatch.setenv("TOOL_CAPABLE_MODELS", "model-a")
    assert tool_loop.tool_capable("model-a") is True
    assert tool_loop.tool_capable("model-b") is False
    tool_loop._capability_cache.clear()