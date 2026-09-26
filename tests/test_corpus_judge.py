# -*- coding: utf-8 -*-
"""corpus 语义判的测试。

为什么值得单独测：这一路决定了"她该不该想起某段经历"。
判错了有两种后果，方向相反：
- 判太松 → **她想起错的经历**，会理直气壮地说一件没发生过的事
  （"搬家"那个坑就是这么来的：用户问别的事，她答"搬家还没弄完"）
- 判太严 / 判定失败就不带经历 → 她照样按人设回，只是没想起这段

所以"**失败一律不带**"这条必须有测试兜着——它和 group_gate 的失败放行**相反**，
很容易在以后维护时被"顺手改成一致"。
"""
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import corpus_judge
from src.plugins.chatbot.corpus_judge import judge_corpus, judge_enabled
from src.plugins.chatbot.retrieval import RetrievalItem


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


def _cand(i: int, text: str) -> RetrievalItem:
    return RetrievalItem(source="corpus", item_id=str(i), score=0.5 - i * 0.01, text=text)


CANDS = [
    _cand(0, "灰泽满被问到身高，直接说1米6。"),
    _cand(1, "灰泽满说女室友作息很健康，早上九点自然醒。"),
    _cand(2, "灰泽满在直播里提到自己感冒时大舌头很明显。"),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CORPUS_JUDGE", raising=False)


class TestToggle:
    def test_enabled_by_default(self):
        assert judge_enabled() is True

    def test_can_be_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("CORPUS_JUDGE", "0")
        assert judge_enabled() is False

    @pytest.mark.asyncio
    async def test_disabled_keeps_nothing(self, monkeypatch):
        # 关掉开关 = corpus 完全退回"只靠关键词门放行"的老行为（判定的候选一条都不带）
        monkeypatch.setenv("CORPUS_JUDGE", "0")
        client = _StubClient(content='{"keep": [1]}')
        assert await judge_corpus(client, "你多高啊", CANDS) == []


class TestDecision:
    @pytest.mark.asyncio
    async def test_keep_one(self):
        client = _StubClient(content='{"keep": [1]}')
        kept = await judge_corpus(client, "你多高啊", CANDS)
        assert [k.text for k in kept] == [CANDS[0].text]

    @pytest.mark.asyncio
    async def test_empty_keep_returns_nothing(self):
        client = _StubClient(content='{"keep": []}')
        assert await judge_corpus(client, "明天几点开会", CANDS) == []

    @pytest.mark.asyncio
    async def test_code_fence_is_tolerated(self):
        client = _StubClient(content='```json\n{"keep": [2]}\n```')
        kept = await judge_corpus(client, "你室友是谁", CANDS)
        assert [k.text for k in kept] == [CANDS[1].text]

    @pytest.mark.asyncio
    async def test_fabricated_index_is_dropped(self):
        # 模型编了不存在的编号（"保留第 9 条"，候选只有 3 条）→ 直接丢弃，别让下游崩
        client = _StubClient(content='{"keep": [9]}')
        assert await judge_corpus(client, "你多高啊", CANDS) == []

    @pytest.mark.asyncio
    async def test_mixed_fabricated_and_real(self):
        client = _StubClient(content='{"keep": [1, 99]}')
        kept = await judge_corpus(client, "你多高啊", CANDS)
        assert [k.text for k in kept] == [CANDS[0].text]

    @pytest.mark.asyncio
    async def test_duplicate_indices_kept_once(self):
        client = _StubClient(content='{"keep": [1, 1]}')
        kept = await judge_corpus(client, "你多高啊", CANDS)
        assert len(kept) == 1

    @pytest.mark.asyncio
    async def test_keep_is_capped(self, monkeypatch):
        # 限爆破半径：判错也只是多说一句，不会一次灌一堆经历进来
        monkeypatch.setattr(corpus_judge, "CORPUS_JUDGE_MAX_KEEP", 2)
        client = _StubClient(content='{"keep": [1, 2, 3]}')
        kept = await judge_corpus(client, "你多高啊", CANDS)
        assert [k.text for k in kept] == [CANDS[0].text, CANDS[1].text]

    @pytest.mark.asyncio
    async def test_non_list_keep_returns_nothing(self):
        client = _StubClient(content='{"keep": 1}')
        assert await judge_corpus(client, "你多高啊", CANDS) == []


class TestShortCircuit:
    @pytest.mark.asyncio
    async def test_empty_candidates_never_calls_model(self):
        # 没有候选就不该花这次调用
        class _Boom:
            def __getattr__(self, _):
                raise AssertionError("不该调用模型")
        assert await judge_corpus(_Boom(), "你多高啊", []) == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_nothing(self):
        client = _StubClient(content='{"keep": [1]}')
        assert await judge_corpus(client, "   ", CANDS) == []

    @pytest.mark.asyncio
    async def test_candidates_without_text_are_skipped(self):
        client = _StubClient(content='{"keep": [1]}')
        assert await judge_corpus(client, "你多高啊", [_cand(0, "")]) == []


class TestFailClosed:
    """判定失败一律"不带经历"——宁可漏不可错（与 group_gate 的失败放行**相反**）。"""

    @pytest.mark.asyncio
    async def test_network_error_keeps_nothing(self):
        client = _StubClient(exc=RuntimeError("connection reset"))
        assert await judge_corpus(client, "你多高啊", CANDS) == []

    @pytest.mark.asyncio
    async def test_garbage_response_keeps_nothing(self):
        client = _StubClient(content="我看不懂你在说什么")
        assert await judge_corpus(client, "你多高啊", CANDS) == []

    @pytest.mark.asyncio
    async def test_empty_body_keeps_nothing(self):
        # 上游思考模式吃光 max_tokens 时 content 会是空串 → extract_chat_content 抛错
        client = _StubClient(content="")
        assert await judge_corpus(client, "你多高啊", CANDS) == []

    @pytest.mark.asyncio
    async def test_missing_keep_key_keeps_nothing(self):
        client = _StubClient(content='{"why": "拿不准"}')
        assert await judge_corpus(client, "你多高啊", CANDS) == []


class TestPromptContent:
    """判据措辞是这一路的核心（改词=改行为，实测过判据稍动结果就从 0% 跳到 74%）。
    这是**用户标定的取向**，改提示词时别弄丢。"""

    def test_criteria_present(self):
        p = corpus_judge.CORPUS_JUDGE_PROMPT
        assert "是不是在问她" in p          # 判的是"冲她来的"，不是"撞到同一个词"
        assert "感冒吃什么药" in p          # 真实反例：问药 ≠ 问她感冒过
        assert "拿不准" in p and "不保留" in p  # 严格偏向

    def test_clarifies_question_frame_collision(self):
        # 「明天几点开会 / 明天几点下课」——句式撞车但内容不同，是这个项目踩过的坑
        assert "明天几点开会" in corpus_judge.CORPUS_JUDGE_PROMPT


class TestPromptSent:
    @pytest.mark.asyncio
    async def test_candidates_are_numbered_and_query_included(self):
        seen = {}

        class _Capturing(_StubClient):
            async def create(self, **kwargs):
                seen["prompt"] = kwargs["messages"][0]["content"]
                return await super().create(**kwargs)

        await judge_corpus(_Capturing(content='{"keep": []}'), "你多高啊", CANDS)
        assert "你多高啊" in seen["prompt"]
        assert "1. " in seen["prompt"]
        assert "1米6" in seen["prompt"]
