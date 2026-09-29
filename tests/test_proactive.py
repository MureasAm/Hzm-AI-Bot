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
from src.plugins.chatbot import core
from src.plugins.chatbot.proactive import compose_proactive, proactive_enabled
from src.plugins.chatbot.constants import (
    PROACTIVE_SAMPLE_KEEP_ALL as KEEP_ALL, PROACTIVE_SAMPLE_TOP_N as SAMPLE_TOP_N,
)
from src.plugins.chatbot.retrieval import RetrievalItem


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
    monkeypatch.delenv("PROACTIVE_JUDGE", raising=False)

    # 不打网络：客户端打桩，检索直接返回空（风格样本那条路本来就允许失败）
    monkeypatch.setattr(proactive, "_get_clients",
                        lambda: (_StubClient(content='{"unused": 1}'), object()))

    # 检索现在在 core.gather_retrieval 里（主动发言走主链路），所以打桩打在 core 上
    async def _no_embed(*a, **k):
        return [0.0]
    monkeypatch.setattr(core, "embed_query", _no_embed)
    monkeypatch.setattr(core, "retrieve_voice_samples", lambda *a, **k: [])


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
        monkeypatch.setattr(core, "embed_query", _boom)

        assert await compose_proactive("深夜发了条动态") == "睡不着，随便说点啥。"


class TestPromptContent:
    def test_prompt_forbids_broadcast_tone(self):
        p = proactive.PROACTIVE_PROMPT
        assert "别像播报" in p          # 核心诉求：不是发公告
        assert "NULL" in p              # 不适合要能说出来
        assert "灰泽满" in p             # 自称习惯


class _TwoStepClient:
    """一次 compose 要调**两次** LLM：先判"说不说得清"，再生成。

    按 prompt 内容分派（判官问句里有"够不够撑起"），不靠调用顺序——
    以后中间再加一步也不会把用例测歪。
    """

    def __init__(self, judge='{"enough": true}', gen="明天来玩啊", judge_exc=None):
        self.judge, self.gen, self.judge_exc = judge, gen, judge_exc
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        is_judge = "够不够撑起" in prompt
        if is_judge and self.judge_exc:
            raise self.judge_exc
        content = self.judge if is_judge else self.gen
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
        ])


class TestContentJudge:
    """开口前的第一道闸：内容**说不说得清**。

    用户 2026-09-27 报：她微博写"写不完了"，直播里说的其实是"作业写不完了"——
    单独看连"写什么"都不知道，硬说一句只会说歪或说空，不如不说。
    """

    def _with(self, monkeypatch, judge, gen="明天来玩啊"):
        monkeypatch.setattr(proactive, "_get_clients",
                            lambda: (_TwoStepClient(judge=judge, gen=gen), object()))

    async def test_unclear_content_skips_generation(self, monkeypatch):
        self._with(monkeypatch, judge='{"enough": false, "why": "缺了写什么"}')
        assert await compose_proactive("写不完了", "微博") is None

    async def test_clear_content_still_generates(self, monkeypatch):
        # 短但完整（"今天好累"）不该被这条判据误杀
        self._with(monkeypatch, judge='{"enough": true}')
        assert await compose_proactive("今天好累", "微博") == "明天来玩啊"

    async def test_fenced_json_tolerated(self, monkeypatch):
        self._with(monkeypatch, judge='```json\n{"enough": false}\n```')
        assert await compose_proactive("写不完了") is None

    async def test_judge_failure_fails_open(self, monkeypatch):
        # 判官挂了 → 放行照常生成（"该说时不说"比"多说一句"难发现）
        monkeypatch.setattr(proactive, "_get_clients",
                            lambda: (_TwoStepClient(judge_exc=RuntimeError("boom")), object()))
        assert await compose_proactive("写不完了") == "明天来玩啊"

    async def test_judge_garbage_fails_open(self, monkeypatch):
        self._with(monkeypatch, judge="我看不出来")
        assert await compose_proactive("写不完了") == "明天来玩啊"

    def test_enabled_by_default(self):
        assert proactive.judge_enabled() is True

    async def test_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("PROACTIVE_JUDGE", "0")
        self._with(monkeypatch, judge='{"enough": false}')
        assert await compose_proactive("写不完了") == "明天来玩啊"

    async def test_pure_punctuation_skipped_without_llm(self, monkeypatch):
        """纯符号内容**确定性拦掉**，不该调 LLM 判——能确定的事别多一次判错的机会。

        实测（2026-09-27）：交给判官时 "..." 会被判成"够"。
        """
        calls = []

        class _Boom(_TwoStepClient):
            async def create(self, **kwargs):
                calls.append(1)
                raise AssertionError("纯符号不该调 LLM")

        monkeypatch.setattr(proactive, "_get_clients", lambda: (_Boom(), object()))
        for text in ("...", "。。。", "？？", "😭"):
            assert await compose_proactive(text) is None
        assert calls == []

    def test_prompt_separates_short_from_incomplete(self):
        p = proactive.PROACTIVE_JUDGE_PROMPT
        assert "短但完整" in p    # 短 ≠ 缺：别把"晚安，明天见"也拦了
        assert "放行" in p        # 拿不准放行——全拦等于这个功能没了


class TestDescribeFirstImage:
    """配图描述：给"⬇️/这个/图片里那个"当指代目标。"""

    async def test_no_image_returns_empty(self):
        assert await proactive._describe_first_image(object(), None) == ""
        assert await proactive._describe_first_image(object(), []) == ""

    async def test_describes_first_image(self, tmp_path, monkeypatch):
        p = tmp_path / "a.jpg"
        p.write_bytes(b"fake-image")
        seen = {}

        async def fake_desc(client, data):
            seen["data"] = data
            return "一张深色游戏图标"
        monkeypatch.setattr(proactive, "describe_image_bytes", fake_desc)
        assert await proactive._describe_first_image(object(), [p]) == "一张深色游戏图标"
        assert seen["data"] == b"fake-image"

    async def test_failure_returns_empty_not_raises(self, tmp_path, monkeypatch):
        # 与判官并行跑，抛异常会连累整条生成 → 必须自己吞掉
        async def boom(client, data):
            raise RuntimeError("视觉服务抖了")
        monkeypatch.setattr(proactive, "describe_image_bytes", boom)
        assert await proactive._describe_first_image(object(), [tmp_path / "nope.jpg"]) == ""


class TestImageBlock:
    """配图要真的进提示词——这次翻车就是因为它没进去。"""

    def _client(self, captured, gen="晚上来打游戏回回血"):
        class _Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            async def create(self, **kwargs):
                prompt = kwargs["messages"][-1]["content"]
                captured.append(prompt)
                content = ('{"enough": true}' if "够不够撑起" in prompt else gen)
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop")])
        return _Client()

    def _patch_desc(self, monkeypatch, desc):
        async def fake_desc(client, paths):
            return desc
        monkeypatch.setattr(proactive, "_describe_first_image", fake_desc)

    async def test_image_description_reaches_the_prompt(self, monkeypatch):
        captured = []
        monkeypatch.setattr(proactive, "_get_clients", lambda: (self._client(captured), object()))
        self._patch_desc(monkeypatch, "一个深色游戏图标，上面有白色文字")
        await compose_proactive("到这个⬇️里面鲨会儿人", "B站", image_paths=["x.jpg"])
        gen_prompt = captured[-1]
        assert "【这条的配图】" in gen_prompt
        assert "深色游戏图标" in gen_prompt
        assert "⬇️" in gen_prompt and "指的就是这张图" in gen_prompt

    async def test_no_image_no_block(self, monkeypatch):
        captured = []
        monkeypatch.setattr(proactive, "_get_clients", lambda: (self._client(captured), object()))
        self._patch_desc(monkeypatch, "不该出现")
        await compose_proactive("晚安，明天见", "B站")
        assert "【这条的配图】" not in captured[-1]

    async def test_no_image_read_when_content_does_not_point_at_it(self, monkeypatch):
        """内容没指配图 → **不读图、不注入**（避免往上下文里塞无关图的噪声）。"""
        calls = []

        async def spy(client, paths):
            calls.append(paths)
            return "一张图"
        monkeypatch.setattr(proactive, "_describe_first_image", spy)
        captured = []
        monkeypatch.setattr(proactive, "_get_clients", lambda: (self._client(captured), object()))

        await compose_proactive("今天下雨，出门记得带伞", "B站", image_paths=["x.jpg"])
        assert calls == []                       # 没读图
        assert "【这条的配图】" not in captured[-1]

    @pytest.mark.parametrize("text", [
        "到这个⬇️里面鲨会儿人", "图里那个是什么", "看这个配图", "👇的链接",
    ])
    async def test_image_read_when_content_points_at_it(self, monkeypatch, text):
        calls = []

        async def spy(client, paths):
            calls.append(paths)
            return "一张图"
        monkeypatch.setattr(proactive, "_describe_first_image", spy)
        captured = []
        monkeypatch.setattr(proactive, "_get_clients", lambda: (self._client(captured), object()))

        await compose_proactive(text, "B站", image_paths=["x.jpg"])
        assert calls == [["x.jpg"]]
        assert "【这条的配图】" in captured[-1]

    async def test_image_description_failure_still_generates(self, monkeypatch):
        captured = []
        monkeypatch.setattr(proactive, "_get_clients", lambda: (self._client(captured), object()))

        async def boom(client, paths):
            return ""          # _describe_first_image 失败时就是这个行为
        monkeypatch.setattr(proactive, "_describe_first_image", boom)
        out = await compose_proactive("写完了…终于写完了…", "B站", image_paths=["x.jpg"])
        assert out == "晚上来打游戏回回血"
        assert "【这条的配图】" not in captured[-1]

    def test_prompt_forbids_inventing_missing_info(self):
        p = proactive.PROACTIVE_PROMPT
        assert "一个字也别补" in p        # 不许编内容里没有的对象（"写稿"那次）
        assert "配图" in p               # 指代只指向配图
        assert "只挑一件说" in p          # 两件事揉不清就只说一件

    def test_prompt_does_not_override_self_reference(self):
        """提示词**不许规定自称**——那是人设文件的事（黄金律：素材层解决）。

        踩坑（2026-09-29）：为了压住"第三人称旁白"，我在这里写过
        「自称用"我"最自然」——等于把 system_prompt 的【自我称呼】覆盖掉，
        于是主动发言开始冒"陪我回回血"（正常聊天里她从不用"我"）。
        实测：把那段整段删掉后，只靠人设骨架 + 风格样本，8 次里 0 次出现"我"。
        """
        p = proactive.PROACTIVE_PROMPT
        assert '自称用"我"最自然' not in p     # 覆盖人设的那句必须没了
        assert "不要用" not in p               # 也不该反过来立一条硬规矩
        assert "人设" in p or "自我称呼" in p   # 只留一句"交给谁管"的指路


class _RecordingClient:
    """记录每次调用收到的 messages，便于断言注入了什么。"""

    def __init__(self, gen="陪灰泽满缓缓"):
        self.calls = []
        self.chat = self
        self.completions = self
        self.gen = gen

    async def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        prompt = kwargs["messages"][-1]["content"]
        content = ('{"enough": true}' if "够不够撑起" in prompt else self.gen)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")])


class TestSampleInjection:
    """"没有生活感"的根因：她自己的动态是陈述句，跟"弹幕问答"样本不同构。

    实测（2026-09-27）5 条真实动态：晚安 0.736 / 直播推迟 0.712 进得来，
    但「新视频发出来了…」0.532、「下午出去逛了逛」0.543 全部低于 0.66 阈值和保底门槛
    → **一条样本都进不来** → 只剩人设骨架干说。
    """

    def _patch(self, monkeypatch, client, items):
        monkeypatch.setattr(proactive, "_get_clients", lambda: (client, object()))

        async def _emb(*a, **k):
            return [0.0]
        monkeypatch.setattr(core, "embed_query", _emb)
        monkeypatch.setattr(core, "retrieve_voice_samples", lambda *a, **k: items)

    async def test_samples_requested_without_threshold(self, monkeypatch):
        seen = {}

        def fake_retrieve(text, vector, threshold=None, top_n=None):
            seen["threshold"], seen["top_n"] = threshold, top_n
            return []
        client = _RecordingClient()
        self._patch(monkeypatch, client, [])
        monkeypatch.setattr(core, "retrieve_voice_samples", fake_retrieve)

        await compose_proactive("下午出去逛了逛", "B站")
        assert seen["threshold"] == KEEP_ALL     # 不设阈值（她自己的陈述句跟样本不同构）
        assert seen["top_n"] == SAMPLE_TOP_N

    async def test_framing_message_keeps_samples_as_style_only(self, monkeypatch):
        # 不框一句的话，模型会把样本当成"最近说过的话"去接，甚至搬示例里的内容
        item = RetrievalItem(source="voice_sample", item_id="s1", score=0.5, text="",
                             extra={"user": "粉丝问：你忙吗", "reply": "灰泽满忙得很"})
        client = _RecordingClient()
        self._patch(monkeypatch, client, [item])

        await compose_proactive("今天好累", "B站")
        gen_msgs = client.calls[-1]
        assert any("说话方式参考" in m["content"] for m in gen_msgs)
        assert any(m["content"] == "灰泽满忙得很" for m in gen_msgs)

    async def test_no_samples_no_framing(self, monkeypatch):
        client = _RecordingClient()
        self._patch(monkeypatch, client, [])
        await compose_proactive("今天好累", "B站")
        assert not any("说话方式参考" in m["content"] for m in client.calls[-1])
