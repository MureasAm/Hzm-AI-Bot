"""`config.extract_chat_content`：LLM 响应的安全解析。

为什么值得单独测：以前 9 个调用点都直接写 `resp.choices[0].message.content.strip()`，
只在"响应形状永远正确"时才成立。实测踩过上游偶发返回 `choices: null`，
三个调用点**同时**报 `'NoneType' object is not subscriptable`——完全没法定位。
还有一个**静默**的坏形状：思考模式吃光 max_tokens → content 是空串、不报错，
功能悄悄失灵（记忆提取/行为分类拿到空串当"这次没结果"）。
"""
from types import SimpleNamespace
import json
import time

import pytest

from src.plugins.chatbot import config, constants
from src.plugins.chatbot.config import (extract_chat_content, _ClientProxy, _ChainCompletions,
                                        _mk_client)


class _Msg:
    def __init__(self, content, reasoning=None):
        self.content = content
        if reasoning is not None:
            self.reasoning_content = reasoning


class _Choice:
    def __init__(self, content, finish="stop", reasoning=None):
        self.message = _Msg(content, reasoning)
        self.finish_reason = finish


class _Resp:
    def __init__(self, choices):
        self.choices = choices


class TestExtractChatContent:
    def test_normal_content_returned_stripped(self):
        assert extract_chat_content(_Resp([_Choice("  在。  ")])) == "在。"

    def test_choices_none_raises_readable_error(self):
        # 上游偶发返回 choices: null —— 报错必须说人话，而不是 'NoneType' object is not subscriptable
        with pytest.raises(RuntimeError) as e:
            extract_chat_content(_Resp(None))
        assert "choices" in str(e.value)

    def test_choices_empty_list_raises(self):
        with pytest.raises(RuntimeError):
            extract_chat_content(_Resp([]))

    def test_content_none_raises(self):
        with pytest.raises(RuntimeError) as e:
            extract_chat_content(_Resp([_Choice(None)]))
        assert "content 为 None" in str(e.value)

    def test_empty_content_raises_with_thinking_hint(self):
        # 思考模式吃光 max_tokens 的典型形状：content=""、finish_reason=length、有 reasoning
        with pytest.raises(RuntimeError) as e:
            extract_chat_content(_Resp([_Choice("", finish="length", reasoning="想了很多" * 5)]))
        msg = str(e.value)
        assert "正文为空" in msg and "max_tokens" in msg

    def test_missing_message_attr_raises(self):
        class _Bare:
            choices = [object()]      # 没有 .message

        with pytest.raises(RuntimeError):
            extract_chat_content(_Bare())

    def test_missing_choices_attr_raises(self):
        with pytest.raises(RuntimeError):
            extract_chat_content(object())   # 完全没有 choices 属性

    def test_whitespace_only_content_raises(self):
        with pytest.raises(RuntimeError):
            extract_chat_content(_Resp([_Choice("   \n  ")]))


def _ok(text):
    """一个形状正常的假响应。"""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _text(resp):
    """从假响应里取正文——链现在返回的是**响应对象**，不是字符串。"""
    return resp.choices[0].message.content


class TestModelFallbackChain:
    """按顺序自动降级：**gemini → luna → deepseek**（2026-10-14）。

    起因：中转站的 gemini 线路整条 500 挂了 75 分钟，而 bot 只会一轮一轮地回兜底文案。
    端点/模型级故障是运维问题，不该表现成"她不在"。
    """

    @staticmethod
    def _client(result=None, error=None, calls=None):
        """一个长得像 AsyncOpenAI 的假客户端；`result` 传字符串会自动包成正常形状的响应。"""
        async def create(**kw):
            if calls is not None:
                calls.append(kw)
            if error is not None:
                raise error
            return _ok(result) if isinstance(result, str) else result
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def _chain(self, *specs, cooldown=60):
        """specs: (model, result, error[, calls])——按顺序当作第 1/2/3/… 层。"""
        layers = []
        for spec in specs:
            model, result, error = spec[0], spec[1], spec[2]
            calls = spec[3] if len(spec) > 3 else None
            layers.append((self._client(result, error, calls), model, model))
        return _ChainCompletions(layers, cooldown=cooldown)

    async def test_primary_ok_stops_at_first_layer(self):
        second = []
        chain = self._chain(("m1", "主的", None), ("m2", "二层的", None, second))
        assert _text(await chain.create(model="m1", messages=[])) == "主的"
        assert second == [], "第 1 层正常时不该动后面的"

    async def test_falls_through_in_order(self):
        second, third = [], []
        chain = self._chain(("m1", None, RuntimeError("500")),
                            ("m2", "二层的", None, second),
                            ("m3", "三层的", None, third))
        assert _text(await chain.create(model="m1", messages=[])) == "二层的"
        assert second[0]["model"] == "m2", "降级时要把 model 换成那一层的"
        assert third == []

    async def test_skips_to_third_when_second_also_down(self):
        chain = self._chain(("m1", None, RuntimeError("一层挂")),
                            ("m2", None, RuntimeError("二层挂")),
                            ("m3", "三层的", None))
        assert _text(await chain.create(model="m1", messages=[])) == "三层的"

    async def test_all_down_reraises_last_error(self):
        chain = self._chain(("m1", None, RuntimeError("一层挂")), ("m2", None, RuntimeError("二层挂")))
        with pytest.raises(RuntimeError, match="二层挂"):
            await chain.create(model="m1", messages=[])

    # ---- 形状坏也要往下走（2026-10-14 真实撞到）----

    async def test_no_choices_falls_through(self):
        """上游偶发返回 `choices` 为空的"200 响应"——HTTP 成功不等于能用的响应。

        链以前只认异常、把它当成功返回，下游 `extract_chat_content` 才发现形状不对，
        **那时已经在链外，不会再降级** → 整轮掉进"小满现在不在哦~"。
        """
        second = []
        chain = self._chain(("m1", SimpleNamespace(choices=None), None), ("m2", "二层的", None, second))
        assert _text(await chain.create(model="m1", messages=[])) == "二层的"
        assert second, "形状不对就该试下一层"

    async def test_empty_body_falls_through(self):
        empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  "))])
        chain = self._chain(("m1", empty, None), ("m2", "二层的", None))
        assert _text(await chain.create(model="m1", messages=[])) == "二层的"

    async def test_bad_shape_on_last_layer_is_returned_as_is(self):
        """最后一层也坏 → 原样返回，让调用方照旧报错（保留那句能看懂的日志）。"""
        bad = SimpleNamespace(choices=[])
        chain = self._chain(("m1", "好的", None), ("m2", bad, None))
        chain._down_until[0] = time.monotonic() + 60      # 逼它直接走最后一层
        assert await chain.create(model="m1", messages=[]) is bad

    # ---- 切换的粘性 ----

    async def test_failed_layer_is_skipped_within_the_window(self, monkeypatch):
        """一次失败就全线切走：窗口内不再去撞那一层（用户 2026-10-14 第一轮的要求）。

        原来"冷却 60 秒再试"对真实流量等于没做——聊天轮次本来就好几分钟一次，
        每次调用都还会去撞一次那条死线路、白等两三秒。
        """
        monkeypatch.setattr(config, "_CHAIN_RETRY_SEC", 600.0)
        first = []
        chain = _ChainCompletions([
            (self._client(None, RuntimeError("500"), first), "m1", "m1"),
            (self._client("二层的", calls=None), "m2", "m2"),
        ])
        await chain.create(model="m1", messages=[])
        n = len(first)
        await chain.create(model="m1", messages=[])
        assert len(first) == n, "窗口内不该再去撞那条死线路"

    async def test_default_layer_does_not_stick_forever(self, monkeypatch):
        """**但它必须会自己回来**（用户 2026-10-14 第二轮，当天就发现的问题）。

        中间那版是"永不回头"（`_down_until = inf`），用户当天的原话：
        "一旦滑到下一级模型，每次接受新消息都是以下一级模型进行回复了……一直在用 deepseek"。
        ——而**最贵的恰恰就是最后一层**（deepseek 官方按 token 计费，中转站是包月）。
        所以默认是 10 分钟回头探一次；探一次的代价只有 ~1s（`max_retries=0`）。
        """
        monkeypatch.setattr(config, "_CHAIN_RETRY_SEC", 600.0)
        chain = _ChainCompletions([
            (self._client(None, RuntimeError("500")), "m1", "m1"),
            (self._client("二层的"), "m2", "m2"),
        ])
        await chain.create(model="m1", messages=[])
        left = chain._down_until[0] - time.monotonic()
        assert left != float("inf"), "不能是「永远不回头」"
        assert 0 < left <= 601, f"该是 10 分钟量级的回头窗口，实际 {left:.0f}s"

    async def test_flaky_layer_gets_a_short_window(self, monkeypatch):
        """形状坏 = "抖了一下"，不是"挂了"：窗口短得多（默认 60s vs 600s）。

        这一层多半是好的（实测中转站 luna 6 次里 1 次卡死/空响应、其余毫秒级正常），
        用长窗口会让它白白离场十分钟，白花最后一层的钱；而它重试一次可能要等到超时（贵），
        所以也别短到"下一句就撞"。两档取值与理由见 `config._CHAIN_RETRY_SEC` 那段注释。
        """
        monkeypatch.setattr(config, "_CHAIN_RETRY_SEC", 600.0)
        monkeypatch.setattr(config, "_CHAIN_FLAKY_SEC", 60.0)
        chain = _ChainCompletions([(self._client(SimpleNamespace(choices=None)), "m1", "m1"),
                                   (self._client("二层的"), "m2", "m2")])
        await chain.create(model="m1", messages=[])
        left = chain._down_until[0] - time.monotonic()
        assert 0 < left <= 61, f"软失败该用短窗口，实际 {left:.0f}s"

    async def test_recovering_layer_is_announced(self, capsys):
        """**回升**也要报一句——不然"回到 gemini 了"在日志里看不出来（降级是黑箱）。"""
        chain = self._chain(("m1", "主的", None), ("m2", "二层的", None), cooldown=600)
        await chain.create(model="m1", messages=[])
        chain._down_until[0] = time.monotonic() + 600       # 逼它降到第 2 层
        await chain.create(model="m1", messages=[])
        chain._down_until.clear()                           # 第 1 层活了
        await chain.create(model="m1", messages=[])
        out = capsys.readouterr().out
        assert "降级到第 2 层" in out and "回到第 1 层：m1" in out

    async def test_retry_sec_makes_it_come_back(self, monkeypatch):
        """设了 `LLM_CHAIN_RETRY_SEC` 就变成"过一阵回头探一次"。"""
        monkeypatch.setattr(config, "_CHAIN_RETRY_SEC", 900.0)
        chain = _ChainCompletions([(self._client("主的"), "m1", "m1"),
                                   (self._client("二层的"), "m2", "m2")])
        chain._down_until[0] = time.monotonic() - 1        # 已经过了回头间隔
        assert _text(await chain.create(model="m1", messages=[])) == "主的"

    # ---- 冷却 ----

    async def test_failed_layer_is_skipped_while_cooling(self):
        """挂过的那层要冷却——否则主线路挂着时，**每次调用**都先白等它 2.3~3.4 秒。"""
        first = []
        chain = self._chain(("m1", "主的", None, first), ("m2", "二层的", None))
        chain._down_until[0] = time.monotonic() + 60
        assert _text(await chain.create(model="m1", messages=[])) == "二层的"
        assert first == [], "冷却期内不该再去试它"

    async def test_recovery_clears_cooldown(self):
        """冷却到点会再试主模型；成功就把冷却清掉（恢复了要自己回来）。"""
        chain = self._chain(("m1", "主的", None), ("m2", None, RuntimeError("二层")))
        chain._down_until[0] = time.monotonic() - 1        # 冷却已过
        assert _text(await chain.create(model="m1", messages=[])) == "主的"
        assert 0 not in chain._down_until, "主模型活了就该把它的冷却清掉"

    async def test_everything_cooling_still_tries_last_layer(self):
        chain = self._chain(("m1", None, RuntimeError("一")), ("m2", None, RuntimeError("二")),
                            ("m3", "三层的", None))
        chain._down_until.update({0: time.monotonic() + 60, 1: time.monotonic() + 60})
        assert _text(await chain.create(model="m1", messages=[])) == "三层的"

    # ---- 429 是"慢一点"，不是"坏了"（2026-10-14）----

    @staticmethod
    def _rate_limited():
        import httpx
        from openai import RateLimitError
        return RateLimitError("429", response=httpx.Response(
            429, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions")), body=None)

    async def test_rate_limit_uses_the_short_window(self, monkeypatch):
        """429 用**软失败**的短窗口，不是"线路挂了"的长窗口。

        实测（2026-10-14）：中转站 luna 连发 5 次后额度桶就空了，之后每个 429 都在 0.2s 内被拒。
        它表达的是"你问太快了"，不是"我坏了"——用 600s 的长窗口等于把一次限流
        当成线路报废，一小时里撞上两次这层就没了。
        """
        monkeypatch.setattr(config, "_CHAIN_RETRY_SEC", 600.0)
        monkeypatch.setattr(config, "_CHAIN_FLAKY_SEC", 60.0)
        chain = _ChainCompletions([(self._client(None, self._rate_limited()), "m1", "m1"),
                                   (self._client("二层的"), "m2", "m2")])
        await chain.create(model="m1", messages=[])
        left = chain._down_until[0] - time.monotonic()
        assert 0 < left <= 61, f"429 该用短窗口，实际 {left:.0f}s"

    async def test_rate_limit_is_logged_as_throttling(self, capsys):
        """"限流"和"报错"在日志上要分得开——不然会被当成线路挂了。"""
        chain = _ChainCompletions([(self._client(None, self._rate_limited()), "m1", "m1"),
                                   (self._client("二层的"), "m2", "m2")])
        await chain.create(model="m1", messages=[])
        assert "限流（慢一点，不是坏了）" in capsys.readouterr().out

    # ---- 按层给参数（不同模型"关思考"的写法不一样）----

    async def test_layer_extra_params_are_merged(self):
        """某一层可以自带参数覆盖（`constants.MODEL_EXTRA_PARAMS`）。

        实测（2026-10-14）：中转站的 `deepseek-v4.1-flash` **不认** `thinking=disabled`——
        它会照样思考，而且**思考计入 max_tokens**：主回复额度 150 被烧光 → 正文永远是空串
        (`finish_reason=length`)，整层看起来像坏的。它认的是 `budget_tokens=0`。
        """
        calls = []
        chain = _ChainCompletions([
            (self._client(None, RuntimeError("主层挂")), "m1", "m1", None),
            (self._client("二层的", calls=calls), "m2", "m2",
             {"thinking": {"type": "enabled", "budget_tokens": 0}}),
        ])
        await chain.create(model="m1", messages=[], extra_body={"thinking": {"type": "disabled"}})
        assert calls[0]["extra_body"] == {"thinking": {"type": "enabled", "budget_tokens": 0}}

    async def test_layers_without_override_keep_the_callers_params(self):
        """没有覆盖的层**不许动**调用方的参数——覆盖是"某一层特例"，不是全局换写法。"""
        calls = []
        chain = _ChainCompletions([(self._client("主的", calls=calls), None, "主", None)], cooldown=600)
        await chain.create(model="m1", messages=[], extra_body={"thinking": {"type": "disabled"}})
        assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_three_tuple_layers_are_tolerated(self):
        """老写法的三元组层照样收（`extra_params` 补 None）——测试和"没特例"的层都这么写。"""
        assert _ChainCompletions([("a", "b", "c")])._layers == [("a", "b", "c", None)]

    # ---- 参数传递 ----

    async def test_first_layer_keeps_callers_model_name(self):
        """第 1 层用 `model=None` 表示"用调用方传的那个"——**不能覆盖成 None**。

        踩坑（2026-10-14 当场撞到）：写成 `{**kwargs, "model": model}` 会把模型名覆盖成 None，
        上游收到 `model=null` 就报 400，而那条错误会被当成"主模型挂了"、白白降级。
        """
        first = []
        chain = _ChainCompletions([(self._client("主的", calls=first), None, "主层")], cooldown=60)
        await chain.create(model="真模型名", messages=[])
        assert first[0]["model"] == "真模型名"

    def test_cooldown_can_be_disabled(self):
        assert _ChainCompletions([("x",) * 3], cooldown=0)._cooldown == 0

    def test_chained_clients_disable_sdk_retries(self):
        """`max_retries=0` + 显式 timeout：SDK 那两个默认值是**降级链的隐形税**。

        - 自带重试 2 次 = 每次撞死线路白等 2.6s（实测同一条线路 3.7s → 2.0s → 1.1s）——
          我们的链本身就是重试机制，再让 SDK 重试一遍只是把延迟乘三。
        - timeout 默认 600s = 上游卡住不返回时，这一轮干等十分钟才轮到降级。
        """
        cli = _mk_client("k", "https://example.invalid/v1")
        assert cli.max_retries == 0
        assert float(cli.timeout) == float(constants.LLM_TIMEOUT_SEC)

    def test_other_attributes_pass_through(self):
        """只拦 chat.completions；别的属性（embeddings 之类）要原样透传。"""
        proxy = _ClientProxy(SimpleNamespace(embeddings="原样的", chat=SimpleNamespace(
            completions=SimpleNamespace(create=None))), None)
        assert proxy.embeddings == "原样的"


class TestChainLayerBuilding:
    """链怎么从配置拼出来——**三个不同端点**（2026-10-15 起，linkapi → krill → deepseek 官方）。"""

    @staticmethod
    def _cfg(**kw):
        base = dict(openai_api_key="k1", openai_api_base="https://linkapi.ai/v1",
                    openai_model="gemini-3.1-flash-lite", llm_fallback_models="",
                    fallback_api_key="k2", fallback_api_base="https://api-slb.krill-code.net/codex/v1",
                    fallback_model="gemini-3.8-flash",
                    fallback2_api_key="k3", fallback2_api_base="https://api.deepseek.com/v1",
                    fallback2_model="deepseek-flash")
        base.update(kw)
        return SimpleNamespace(**base)

    def test_endpoints_come_out_in_order(self):
        eps = config._more_endpoints(self._cfg())
        assert [e[2] for e in eps] == ["gemini-3.8-flash", "deepseek-flash"]
        assert eps[0][1] == "https://api-slb.krill-code.net/codex/v1"

    def test_unnumbered_fallback_still_works(self):
        """不带编号的 `FALLBACK_*` 仍当第 1 个端点读——**老配置不该因为这次改动失效**。"""
        cfg = self._cfg(fallback2_api_key=None, fallback2_api_base=None, fallback2_model=None)
        eps = config._more_endpoints(cfg)
        assert [e[2] for e in eps] == ["gemini-3.8-flash"]
        assert config._more_endpoints(self._cfg(fallback_api_key=None,
                                                fallback_api_base=None,
                                                fallback_model=None))[0][2] == "deepseek-flash"

    def test_fallback1_key_wins_over_unnumbered(self):
        eps = config._more_endpoints(self._cfg(fallback1_api_key="k1b", fallback1_model="第1个"))
        assert [e[2] for e in eps] == ["第1个", "deepseek-flash"]

    def test_layers_carry_their_own_params(self):
        """每层的参数覆盖按**它自己的模型名**取（gemini-3.8-flash 要 thinking_level=low）。"""
        layers = config._chain_layers(self._cfg())
        assert [l[2].split("（")[0] for l in layers] == \
            ["gemini-3.1-flash-lite", "gemini-3.8-flash", "deepseek-flash"]
        assert layers[1][3] == {"generation_config": {"thinking_level": "low"}}
        assert layers[0][3] is None and layers[2][3] is None


class TestServedByIsVisible:
    """三个端点之后，**日志和账本必须看得出这轮是哪个站答的**（2026-10-15）。

    起因：链变成 linkapi → krill → deepseek 官方之后，两个站可能挂着**同名不同价**的模型，
    光看模型名分不出来。第 1 层的标签原来还**只有模型名**（别的层都带端点），更看不出来。
    """

    def test_primary_layer_label_carries_the_host(self):
        layers = config._chain_layers(TestChainLayerBuilding._cfg())
        assert layers[0][2] == "gemini-3.1-flash-lite（linkapi.ai）", \
            "第 1 层也得带端点名，否则日志上分不出是哪个站"

    async def test_served_by_is_stamped_and_printed(self, capsys, monkeypatch):
        monkeypatch.setattr(config, "_SERVED_BY", config.contextvars.ContextVar("t", default=""))
        chain = _ChainCompletions([(TestModelFallbackChain._client("主的"), "m1", "m1（某站）", None)])
        resp = await chain.create(model="m1", messages=[])
        assert config._SERVED_BY.get() == "m1（某站）"
        config.log_cache_usage(
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
                            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1), model="m1"),
            "主回复")
        assert "[m1（某站）]" in capsys.readouterr().out, "日志要打层标签、不是光打模型名"

    async def test_ledger_row_records_the_layer(self, tmp_path, monkeypatch):
        """账本多一列 `layer`——**按模型定价时，同名不同价的两个站得靠它分**。"""
        monkeypatch.setattr(config, "USAGE_LOG_FILE", tmp_path / "u.jsonl")
        monkeypatch.delenv("LLM_USAGE", raising=False)
        monkeypatch.setattr(config, "_SERVED_BY", config.contextvars.ContextVar("t2", default=""))
        chain = _ChainCompletions([(TestModelFallbackChain._client("主的"), "m1", "m1（某站）", None)])
        resp = await chain.create(model="m1", messages=[])
        resp.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1)
        config.log_cache_usage(resp, "主回复")
        rec = json.loads((tmp_path / "u.jsonl").read_text(encoding="utf-8").strip())
        assert rec["layer"] == "m1（某站）"
