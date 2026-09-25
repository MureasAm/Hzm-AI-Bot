# -*- coding: utf-8 -*-
"""群聊接话门的测试。

为什么值得单独测：这是**行为大改**——灰泽满从"群里每条都接"变成"挑着接"。
判定错了有两种后果，方向相反：
- 判太松 → 退回"有话就接"，白做
- 判太严 / 判定失败就闭嘴 → **她变成哑巴**，而且这种故障很难被发现
所以"失败一律按接处理"这条必须有测试兜着。
"""
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import group_gate
from src.plugins.chatbot.group_gate import gate_enabled, should_reply_in_group


class _StubClient:
    """假的 DeepSeek 客户端：返回指定正文，或抛指定异常。"""

    def __init__(self, content=None, exc=None):
        self._content, self._exc = content, exc
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=self._content),
                            finish_reason="stop")
        ])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GROUP_GATE", raising=False)


class TestToggle:
    def test_enabled_by_default(self):
        assert gate_enabled() is True

    def test_can_be_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("GROUP_GATE", "0")
        assert gate_enabled() is False

    @pytest.mark.asyncio
    async def test_disabled_gate_always_replies(self, monkeypatch):
        # 关掉开关 = 退回旧行为（群里每条都接）
        monkeypatch.setenv("GROUP_GATE", "0")
        client = _StubClient(content='{"reply": false}')
        assert await should_reply_in_group(client, "历史", "随便说点什么") is True


class TestDecision:
    @pytest.mark.asyncio
    async def test_true_means_reply(self):
        client = _StubClient(content='{"reply": true, "why": "有人表达情绪"}')
        assert await should_reply_in_group(client, "历史", "我服了") is True

    @pytest.mark.asyncio
    async def test_false_means_stay_silent(self):
        client = _StubClient(content='{"reply": false, "why": "纯事务"}')
        assert await should_reply_in_group(client, "历史", "你扔点链接吧") is False

    @pytest.mark.asyncio
    async def test_code_fence_is_tolerated(self):
        client = _StubClient(content='```json\n{"reply": false}\n```')
        assert await should_reply_in_group(client, "历史", "哈哈") is False

    @pytest.mark.asyncio
    async def test_empty_batch_never_replies(self):
        client = _StubClient(content='{"reply": true}')
        assert await should_reply_in_group(client, "历史", "   ") is False


class TestFailOpen:
    """判定失败一律按"接"——宁滥勿缺。她该说时不说，比多说几句难发现得多。"""

    @pytest.mark.asyncio
    async def test_network_error_replies(self):
        client = _StubClient(exc=RuntimeError("connection reset"))
        assert await should_reply_in_group(client, "历史", "在吗") is True

    @pytest.mark.asyncio
    async def test_garbage_response_replies(self):
        client = _StubClient(content="我看不懂你在说什么")
        assert await should_reply_in_group(client, "历史", "在吗") is True

    @pytest.mark.asyncio
    async def test_empty_body_replies(self):
        # 上游思考模式吃光 max_tokens 时 content 会是空串 → extract_chat_content 抛错
        client = _StubClient(content="")
        assert await should_reply_in_group(client, "历史", "在吗") is True

    @pytest.mark.asyncio
    async def test_missing_reply_key_stays_silent(self):
        # 形状对但没给 reply 字段：按 false 处理（明确的 JSON 就是明确的表态）
        client = _StubClient(content='{"why": "不知道该不该"}')
        assert await should_reply_in_group(client, "历史", "在吗") is False


class TestPromptContent:
    """判据是用户标定的三条，别在改提示词时弄丢。"""

    def test_criteria_present(self):
        p = group_gate.GROUP_GATE_PROMPT
        assert "点她名" in p           # ① 明确问 → 必接
        assert "纯情绪也算" in p        # ② 情绪（纯情绪也算）
        assert "分享自己的经历" in p     # ③ 经历 → 偶尔
        assert "纯事务" in p           # ④ 事务/附和/收尾 → 不接

    def test_batch_not_single_message(self):
        # 判定粒度必须是"批"——逐条判会蹭余温（实测 0%→74%→7% 乱跳）
        assert "整批" in group_gate.GROUP_GATE_PROMPT
