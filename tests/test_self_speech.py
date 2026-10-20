# -*- coding: utf-8 -*-
"""自述转述：只改注入文本，不改落盘；失败必须原样返回。"""
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import self_speech


class _Completions:
    def __init__(self, content=None, exc=None):
        self.content = content
        self.exc = exc
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.content))])


def _client(content=None, exc=None):
    comp = _Completions(content, exc)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(self_speech, "_CACHE", {})
    monkeypatch.setattr(self_speech, "_CACHE_LOADED", True)
    monkeypatch.setattr(self_speech, "_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setenv("SELF_SPEECH_ANNOTATE", "1")
    monkeypatch.setattr(self_speech, "config", SimpleNamespace(
        _get_model_name=lambda: "fake",
        extract_chat_content=lambda r: r.choices[0].message.content,
    ), raising=False)


@pytest.mark.asyncio
async def test_rewrites_history_and_self_quote():
    hist = ["[3小时前] 用户：满神和绿冻是一对呀",
            "[3小时前] 灰泽满：少来，灰泽满才不和绿冻命中注定呢。"]
    msg = "[引用灰泽满自己说的：「少来，灰泽满才没忘呢。」] 那是什么关系"
    client, comp = _client('{"gists": ["她否认和对方命中注定。", "她没忘。"]}')
    new_hist, new_msg = await self_speech.annotate_turn(hist, msg, client)
    assert comp.calls == 1
    assert new_hist[0] == hist[0]                      # 用户行不动
    assert new_hist[1] == "[3小时前] 灰泽满（转述）：她否认和对方命中注定。"
    assert new_msg == "[对方引用了你之前说过的话，大意是：她没忘。] 那是什么关系"


@pytest.mark.asyncio
async def test_fail_open_on_api_error():
    hist = ["灰泽满：少来，灰泽满才不承认。"]
    msg = "你好"
    client, _ = _client(exc=RuntimeError("boom"))
    new_hist, new_msg = await self_speech.annotate_turn(hist, msg, client)
    assert new_hist == hist and new_msg == msg


@pytest.mark.asyncio
async def test_gist_keeping_formula_is_dropped():
    hist = ["灰泽满：少来，灰泽满才不承认。"]
    client, _ = _client('{"gists": ["她让对方少来这一套。"]}')
    new_hist, _ = await self_speech.annotate_turn(hist, "你好", client)
    assert new_hist == hist          # 转述仍带原句式 → 原样注入


@pytest.mark.asyncio
async def test_no_call_when_nothing_to_rewrite():
    hist = ["用户：今天吃什么", "灰泽满：你猜。"]
    client, comp = _client('{"gists": ["她让对方猜。"]}')
    new_hist, new_msg = await self_speech.annotate_turn(hist, "随便", client)
    # 历史里有一句她的话，所以会调用；但普通用户消息原样
    assert comp.calls == 1
    assert new_msg == "随便"
    assert new_hist[0] == hist[0]
    assert new_hist[1].startswith("灰泽满（转述）：")


@pytest.mark.asyncio
async def test_disabled_returns_original(monkeypatch):
    monkeypatch.setenv("SELF_SPEECH_ANNOTATE", "0")
    hist = ["灰泽满：少来。"]
    msg = "[引用灰泽满自己说的：「少来。」] 嗯"
    client, comp = _client('{"gists": ["她拒绝。"]}')
    new_hist, new_msg = await self_speech.annotate_turn(hist, msg, client)
    assert comp.calls == 0
    assert new_hist == hist and new_msg == msg
