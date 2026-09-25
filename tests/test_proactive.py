# -*- coding: utf-8 -*-
"""主动发言：把动态/微博转成"她会主动说的那句话"。

**这条路的兜底方向和前面两个都不一样**：
- 接话门判不出来 → 放行（该说时不说更难发现）
- 表情包判不出来 → 不发（发错比不发难看）
- 主动发言 → 返回 None，**由调用方退回原来的通知模板**（信息不能丢：
  粉丝本来就该知道她发动态了，只是形式换了）

所以这里要钉住的是：**任何失败都必须干净地返回 None**，绝不能吐出半截话
或者抛异常把整个监听循环带崩。
"""
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import proactive
from src.plugins.chatbot.proactive import compose_proactive, proactive_enabled


class _StubClient:
    def __init__(self, content=None, exc=None):
        self._content, self._exc = content, exc
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=self._content), finish_reason="stop")
        ])


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("PROACTIVE", raising=False)

    # 不打网络：客户端打桩，检索直接返回空（风格样本那条路本来就允许失败）
    monkeypatch.setattr(proactive, "_get_clients",
                        lambda: (_StubClient(content='{"unused": 1}'), object()))

    async def _no_embed(*a, **k):
        return [0.0]
    monkeypatch.setattr(proactive, "embed_query", _no_embed)
    monkeypatch.setattr(proactive, "retrieve_voice_samples", lambda *a, **k: [])


def _with_llm(monkeypatch, content=None, exc=None):
    monkeypatch.setattr(proactive, "_get_clients",
                        lambda: (_StubClient(content=content, exc=exc), object()))


class TestToggle:
    def test_enabled_by_default(self):
        assert proactive_enabled() is True

    def test_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("PROACTIVE", "0")
        assert proactive_enabled() is False

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("PROACTIVE", "0")
        _with_llm(monkeypatch, content="随便说点什么")
        assert await compose_proactive("今天发了个动态") is None


class TestGenerate:
    @pytest.mark.asyncio
    async def test_returns_her_sentence(self, monkeypatch):
        _with_llm(monkeypatch, content="灰泽满今天收拾到半夜，累死了。")
        out = await compose_proactive("收拾房间收拾到现在", "B站")
        assert out == "灰泽满今天收拾到半夜，累死了。"

    @pytest.mark.asyncio
    async def test_null_means_not_suitable(self, monkeypatch):
        # 纯转发/抽奖这类：模型判不适合，调用方据此退回通知模板
        _with_llm(monkeypatch, content="NULL")
        assert await compose_proactive("【转发抽奖】关注我抽一台手机") is None

    @pytest.mark.asyncio
    async def test_null_is_case_insensitive(self, monkeypatch):
        _with_llm(monkeypatch, content="null")
        assert await compose_proactive("转发微博") is None

    @pytest.mark.asyncio
    async def test_strips_quotes_and_wrapping(self, monkeypatch):
        # 模型爱给整句加引号；带引号发出去很怪
        _with_llm(monkeypatch, content="“你们在干嘛呢。”")
        assert await compose_proactive("随便发点啥") == "你们在干嘛呢。"

    @pytest.mark.asyncio
    async def test_keeps_only_first_line(self, monkeypatch):
        # 偶尔会多写一句解释——私聊里蹦出一大段很出戏
        _with_llm(monkeypatch, content="睡了，晚安。\n（这句是解释为什么要睡了）")
        assert await compose_proactive("发了个晚安动态") == "睡了，晚安。"

    @pytest.mark.asyncio
    async def test_empty_content_returns_none(self, monkeypatch):
        _with_llm(monkeypatch, content="有内容也不该走到这")
        assert await compose_proactive("   ") is None


class TestFailSafe:
    """失败必须干净地返回 None——监听循环不能被一次生成失败带崩。"""

    @pytest.mark.asyncio
    async def test_llm_error_returns_none(self, monkeypatch):
        _with_llm(monkeypatch, exc=RuntimeError("connection reset"))
        assert await compose_proactive("今天发了个动态") is None

    @pytest.mark.asyncio
    async def test_empty_body_returns_none(self, monkeypatch):
        # 思考模式吃光 max_tokens → content 空串 → extract_chat_content 抛错
        _with_llm(monkeypatch, content="")
        assert await compose_proactive("今天发了个动态") is None

    @pytest.mark.asyncio
    async def test_retrieval_failure_does_not_block(self, monkeypatch):
        """风格样本检索挂了，不该连累整条主动发言。"""
        _with_llm(monkeypatch, content="睡不着，随便说点啥。")

        async def _boom(*a, **k):
            raise RuntimeError("embedding 服务抖了")
        monkeypatch.setattr(proactive, "embed_query", _boom)

        assert await compose_proactive("深夜发了条动态") == "睡不着，随便说点啥。"


class TestPromptContent:
    def test_prompt_forbids_broadcast_tone(self):
        p = proactive.PROACTIVE_PROMPT
        assert "别像播报" in p          # 核心诉求：不是发公告
        assert "NULL" in p              # 不适合要能说出来
        assert "灰泽满" in p             # 自称习惯
