"""core 层：图片消息记忆记录 + 图片消息处理逻辑的单元测试。"""
from src.plugins.chatbot import core
from src.plugins.chatbot import rag
from src.plugins.chatbot import reply_style


class TestSplitReply:
    def test_short_reply_not_split(self):
        assert reply_style.split_reply("好的") == ["好的"]

    def test_single_sentence_not_split(self):
        assert reply_style.split_reply("今天天气真不错啊") == ["今天天气真不错啊"]

    def test_short_clauses_merge_to_one(self):
        # 短句+短句合并成一条自然消息，不再拆成碎条（"哦？"不该单独成条）
        assert reply_style.split_reply("今天好冷。你那边呢？") == ["今天好冷。你那边呢？"]

    def test_ellipsis_short_leadin_not_split(self):
        # 省略号前文太短（"唱拉了……"是犹豫前缀）不切，避免把"灰泽满……"单独发一条
        assert reply_style.split_reply("唱拉了……灰泽满先关上门悄悄听会儿。") == \
            ["唱拉了……灰泽满先关上门悄悄听会儿"]

    def test_trailing_short_text_merged(self):
        # 结尾短段（嗯）并入前段，不单独成条
        assert reply_style.split_reply("今天好冷。嗯", min_len=0) == ["今天好冷。嗯"]

    def test_micro_fragments_merge_to_natural(self):
        # 核心：短惊讶碎片（哦？/睡醒了？/那倒是稀奇……）合并成一条自然消息，不再戏剧节拍
        assert reply_style.split_reply(
            "哦？睡醒了？那倒是稀奇……灰泽满这个点刚准备睡，你却醒了", min_len=0
        ) == ["哦？睡醒了？那倒是稀奇……", "灰泽满这个点刚准备睡，你却醒了"]

    def test_min_len_respected(self):
        # 低于 min_len 不拆；末尾句号仍会被去掉
        assert reply_style.split_reply("好。", min_len=10) == ["好"]

    def test_comma_split_kept_when_long(self):
        # 长句的逗号拆分保留（只有短碎片才并回）
        assert reply_style.split_reply(
            "灰泽满今天特别想出去玩，但是作业还没写完，明天还得早起去上第一节课", min_len=0
        ) == ["灰泽满今天特别想出去玩，但是作业还没写完", "明天还得早起去上第一节课"]

    def test_ellipsis_hesitation_preserved(self):
        # 省略号表"无语/语气"（后面没新内容）不切
        assert reply_style.split_reply("啊……这……", min_len=0) == ["啊……这……"]

    def test_ellipsis_boundary_split_kept_when_both_long(self):
        # 省略号前后都有足够内容（都≥12字）时仍切分
        assert reply_style.split_reply(
            "灰泽满今天嗓子特别不舒服……明天就想早点下播休息一下", min_len=0
        ) == ["灰泽满今天嗓子特别不舒服……", "明天就想早点下播休息一下"]


class TestEchoReply:
    """复读机防护：检测新回复是否复读最近自己说过的话。"""

    def test_exact_duplicate(self):
        assert reply_style.is_echo_reply(
            "灰泽满刚醒，你倒是精神好", ["灰泽满刚醒，你倒是精神好"]
        ) is True

    def test_shared_tail(self):
        # "你这话说的……灰泽满刚醒" 复读后半句，最长公共子串覆盖 >60%
        assert reply_style.is_echo_reply(
            "你这话说的……灰泽满刚醒，你倒是精神好", ["灰泽满刚醒，你倒是精神好"]
        ) is True

    def test_reverse_order(self):
        # 完整复读句（>8字）在更长旧句里，也判为复读
        assert reply_style.is_echo_reply(
            "灰泽满刚醒，你倒是精神好", ["大早上的玩这个……灰泽满刚醒，你倒是精神好"]
        ) is True

    def test_short_not_flagged(self):
        # 太短不判（"晚安" 常见短句，避免误伤）
        assert reply_style.is_echo_reply("晚安", ["晚安，做个好梦"]) is False

    def test_different_not_flagged(self):
        assert reply_style.is_echo_reply("今天天气真不错", ["灰泽满刚醒，你倒是精神好"]) is False

    def test_empty_not_flagged(self):
        assert reply_style.is_echo_reply("", ["灰泽满刚醒"]) is False


class TestSummarizeBatch:
    async def test_single_message_skips(self, monkeypatch):
        def _fail(*a, **k):
            raise AssertionError("单条消息不应调用归纳")

        monkeypatch.setattr(core, "_get_clients", _fail)
        assert await core.summarize_batch([("在吗", "")]) == ""

    async def test_success_returns_summary(self, monkeypatch):
        class _FakeCompletions:
            async def create(self, **kwargs):
                msg = type("M", (), {"content": "用户约你明天见面"})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        fake_client = type("C", (), {"chat": type("Chat", (), {"completions": _FakeCompletions()})})()
        monkeypatch.setattr(core, "_get_clients", lambda: (fake_client, None))
        out = await core.summarize_batch([("在吗", ""), ("明天有空吗", "")])
        assert "明天" in out

    async def test_failure_returns_empty(self, monkeypatch):
        def _raise():
            raise RuntimeError("no api")

        monkeypatch.setattr(core, "_get_clients", _raise)
        assert await core.summarize_batch([("a", ""), ("b", "")]) == ""


class TestBatchSummaryInjection:
    def test_injected_as_system_before_user(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗\n明天有空吗", "p", [], "", [],
                                       batch_summary="用户约你明天见面")
        assert msgs[-1]["role"] == "user"
        assert any(m["role"] == "system" and "这批消息的归纳" in m["content"]
                   for m in msgs)

    def test_no_summary_no_injection(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", "p", [], "", [])
        assert not any("这批消息的归纳" in m["content"] for m in msgs)


class TestCleanReply:
    def test_strip_leading_prefix(self):
        assert reply_style.clean_reply("（咽口水）你这又是来投毒的吧……") == "你这又是来投毒的吧……"

    def test_strip_multiple_leading(self):
        assert reply_style.clean_reply("（心虚）（小声）灰泽满最近很忙……") == "灰泽满最近很忙……"

    def test_cap_to_one_parenthetical(self):
        # "a（小声）b（心虚）c" → 保留第一个（小声），去掉其余
        assert reply_style.clean_reply("a（小声）b（心虚）c") == "a（小声）bc"

    def test_keeps_single_inline(self):
        assert reply_style.clean_reply("刚啃完面包，别提了（咽口水）") == "刚啃完面包，别提了（咽口水）"

    def test_no_change_when_clean(self):
        assert reply_style.clean_reply("今天天气真不错啊") == "今天天气真不错啊"

    def test_all_stripped_returns_original(self):
        assert reply_style.clean_reply("（咽口水）") == "（咽口水）"

    def test_ellipsis_ascii_normalized(self):
        # 英文三点 ... 归一成中文省略号；开头"好冷……"短语+省略号犹豫 → 逗号
        assert reply_style.clean_reply("好冷……真的...手都僵了") == "好冷，真的……手都僵了"

    def test_ellipsis_preserves_double(self):
        # "啊……这……" 无语表达，2 次省略号在额度内，原样保留
        assert reply_style.clean_reply("啊……这……") == "啊……这……"

    def test_ellipsis_capped_keeps_two(self):
        # 刷屏式省略号，只保留前 2 个
        assert reply_style.clean_reply("a……b……c……d……") == "a……b……cd"

    def test_ellipsis_long_dots_normalized(self):
        assert reply_style.clean_reply("冷死了...........") == "冷死了……"


class TestSelfPronounCleanup:
    """clean_reply 的自指"她/他"（折中版）：
    只处理"回复里带自称名(灰泽满/hzm)、没别人名时，句读后复指自己的她/他"（去掉不堆名）；
    不带自称名的她/他（多半指对话里的别人）一律保留，不再全换成灰泽满。"""

    def test_without_self_name_third_party_preserved(self):
        # 没出现自称名，她/他指别人 → 原样保留（旧版会错改成灰泽满，这是本次修复点）
        assert reply_style.clean_reply("她昨天生日，我给她录了段祝福") == \
            "她昨天生日，我给她录了段祝福"

    def test_resumptive_she_after_self_name_dropped(self):
        # 带自称名且无别人名：句读后复指自己的"她"去掉，不重复堆名也不串成别人
        assert reply_style.clean_reply("灰泽满到点下播了，她准备睡了") == \
            "灰泽满到点下播了，准备睡了"

    def test_other_person_name_present_untouched(self):
        # 出现别的第三人称名（女同学/满妈），她/他指别人 → 整段不动
        assert reply_style.clean_reply("女同学递了巧克力，她笑了") == \
            "女同学递了巧克力，她笑了"
        assert reply_style.clean_reply("满妈说让她早点睡") == "满妈说让她早点睡"

    def test_plural_and_possessive_not_touched(self):
        # 她们/她的=复数/所有格，指别人或所有，不碰
        assert "她们" in reply_style.clean_reply("灰泽满放好耳机，她们接着复盘了")
        assert "她的" in reply_style.clean_reply("灰泽满说到她的猫就笑")

    def test_qita_not_corrupted(self):
        # "其他"里的"他"是词不是代词
        out = reply_style.clean_reply("其他的也别说了，就聊这个吧")
        assert "其他" in out
        assert "灰泽满" not in out

    def test_bare_self_she_no_longer_force_replaced(self):
        # 折中代价：不带自称名的裸"她"自述不再被强制改名，靠提示词约束模型
        out = reply_style.clean_reply("她转开视线，声音闷闷的：行了")
        assert "她" in out


class TestPreferences:
    def test_injects_retrieved_preferences(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", "p", [], "", [], preference_items=[
            {"category": "食物", "text": "爱吃椰子鸡", "score": 0.6},
            {"category": "作息", "text": "夜猫子", "score": 0.5}])
        injected = [m["content"] for m in msgs if "灰泽满的偏好" in m["content"]]
        assert injected and "爱吃椰子鸡" in injected[0] and "夜猫子" in injected[0]

    def test_no_preference_items_no_injection(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", "p", [], "", [])
        assert not any("灰泽满的偏好" in m["content"] for m in msgs)


class TestRetrievePreferences:
    def test_retrieves_only_above_threshold(self, monkeypatch):
        from src.plugins.chatbot import retrieval as rt
        entries = [
            {"id": "food", "category": "食物", "text": "爱吃椰子鸡", "vector": [1.0, 0.0]},
            {"id": "color", "category": "颜色", "text": "颜色蓝色", "vector": [0.0, 1.0]},
        ]
        monkeypatch.setattr(rt, "load_preference_vectors", lambda: entries)
        items = rt.retrieve_preferences("你爱吃什么", [1.0, 0.0], threshold=0.5, top_n=2)
        assert [i["id"] for i in items] == ["food"]
        assert items[0]["score"] >= 0.5


class TestLegendaryMemory:
    async def test_legendary_reply_records_memory(self, monkeypatch):
        captured = []

        def fake_append(uid, user, reply):  # append_user_history 是同步函数
            captured.append((user, reply))

        async def _noop(*a, **k):
            pass

        monkeypatch.setattr(core, "append_user_history", fake_append)
        monkeypatch.setattr(core, "get_user_memory", lambda uid: {})
        monkeypatch.setattr(core, "update_memory_task", _noop)

        trigger = next(iter(core.LEGENDARY_REPLIES))  # 取一个真实梗
        await core.handle_chat("u", trigger)
        assert captured, "梗匹配回复也应记入短期记忆"
        assert captured[0][0] == trigger


class TestClassifyBehavior:
    """L3：LLM 判定行为意图（embedding 猜意图不可靠，改 LLM 理解）。"""

    def _behaviors(self):
        return [
            {"name": "被夸时嘴硬否认", "trigger": "收到夸奖时", "response": "否认"},
            {"name": "被质疑时心虚辩解", "trigger": "被质问时", "response": "辩解"},
        ]

    def _fake(self, content):
        class _FakeCompletions:
            async def create(self, **kwargs):
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
        return type("C", (), {"chat": type("Chat", (), {"completions": _FakeCompletions()})()})

    async def test_returns_valid_behavior_name(self):
        out = await core.classify_behavior(self._fake('{"behavior": "被夸时嘴硬否认"}'),
                                           "你唱歌好好听", "", self._behaviors())
        assert out == "被夸时嘴硬否认"

    async def test_null_returns_empty(self):
        out = await core.classify_behavior(self._fake('{"behavior": null}'),
                                           "今天天气不错", "", self._behaviors())
        assert out == ""

    async def test_fabricated_name_rejected(self):
        out = await core.classify_behavior(self._fake('{"behavior": "不存在的行为"}'),
                                           "随便", "", self._behaviors())
        assert out == ""

    async def test_markdown_wrapped_json_parsed(self):
        out = await core.classify_behavior(self._fake('```json\n{"behavior": "被质疑时心虚辩解"}\n```'),
                                           "你又在骗人", "", self._behaviors())
        assert out == "被质疑时心虚辩解"

    async def test_failure_returns_empty(self):
        def _raise(*a, **k):
            raise RuntimeError("api down")
        fake = type("C", (), {"chat": type("Chat", (), {"completions": type("C2", (), {"create": _raise})()})()})
        out = await core.classify_behavior(fake, "你好", "", self._behaviors())
        assert out == ""

    async def test_no_behaviors_returns_empty(self):
        out = await core.classify_behavior(self._fake('{"behavior": "x"}'), "你好", "", [])
        assert out == ""


class TestRepairLlmJson:
    """DeepSeek 偶发的不规范 JSON 修复（记忆提取路径）。"""

    def test_bare_keys(self):
        assert core._repair_llm_json('{new_impression: "夜猫子"}') == '{"new_impression": "夜猫子"}'

    def test_single_quotes(self):
        assert core._repair_llm_json("{'a': 'x'}") == '{"a": "x"}'

    def test_trailing_comma(self):
        assert core._repair_llm_json('{"a": 1,}') == '{"a": 1}'

    def test_markdown_fence(self):
        assert core._repair_llm_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_leading_garbage(self):
        assert core._repair_llm_json('好的\n{"a": 1}') == '{"a": 1}'

    def test_empty_unchanged(self):
        assert core._repair_llm_json("") == ""


class TestParseMemoryExtract:
    def test_valid(self):
        assert core._parse_memory_extract('{"new_impression": "夜猫子"}') == {"new_impression": "夜猫子"}

    def test_null_returns_empty(self):
        assert core._parse_memory_extract("null") == {}

    def test_repaired(self):
        assert core._parse_memory_extract("{new_impression: '夜猫子'}") == {"new_impression": "夜猫子"}

    def test_invalid_raises(self):
        import pytest
        with pytest.raises(Exception):
            core._parse_memory_extract("{broken")


class TestUpdateMemoryTaskRetry:
    """记忆提取：首次 JSON 不规范 → 修复/重试，不丢记忆。"""

    def _fake_client(self, first_content, second_content):
        class _Seq:
            def __init__(self):
                self.n = 0
            async def create(self, **kwargs):
                self.n += 1
                content = first_content if self.n == 1 else second_content
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
        return type("C", (), {"chat": type("Chat", (), {"completions": _Seq()})})()

    async def test_retries_then_updates(self, monkeypatch):
        client = self._fake_client("{new_impression: '夜猫子'}", '{"new_impression": "夜猫子"}')
        monkeypatch.setattr(core, "_get_clients", lambda: (client, None))
        captured = {}
        monkeypatch.setattr(core, "update_user_memory", lambda uid, updates: captured.update(updates))
        await core.update_memory_task("u", "我是夜猫子，晚上不睡", "灰泽满也是夜猫子", {})
        assert captured.get("new_impression") == "夜猫子"

    async def test_valid_first_call_no_retry(self, monkeypatch):
        client = self._fake_client('{"new_impression": "夜猫子"}', "不应被调用")
        monkeypatch.setattr(core, "_get_clients", lambda: (client, None))
        captured = {}
        monkeypatch.setattr(core, "update_user_memory", lambda uid, updates: captured.update(updates))
        await core.update_memory_task("u", "我是夜猫子", "好巧", {})
        assert captured.get("new_impression") == "夜猫子"

    async def test_null_skips(self, monkeypatch):
        client = self._fake_client("null", "不应被调用")
        monkeypatch.setattr(core, "_get_clients", lambda: (client, None))
        called = []
        monkeypatch.setattr(core, "update_user_memory", lambda uid, updates: called.append(updates))
        await core.update_memory_task("u", "今天天气不错", "是啊", {})
        assert called == []  # 无新信息，不写卡


class TestVoiceSampleLabel:
    """few-shot 标签要防'内容抄袭'（模型把样本里的礼物/衣服/人物抄进回复）。"""

    def test_label_guards_against_content_copy(self, monkeypatch):
        from types import SimpleNamespace
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        sample = SimpleNamespace(
            source="voice_sample", item_id="peer_5",
            extra={"user": "粉丝问贺图", "reply": "是小雨前辈送的", "type": "daily"},
        )
        msgs = core.build_message_list("在吗", "p", [sample], "", [])
        label = [m["content"] for m in msgs if "灰泽满的说话方式参考" in m["content"]]
        assert label and "不要套用示例里的具体内容" in label[0]


class TestEmotionOnlyQuery:
    """纯情绪补全句检测：'可惜🤭'被 probe 补全成'用户发了个X的表情'时跳过语义检索。"""

    def test_emoji_expansion_detected(self):
        assert reply_style.is_emotion_only_query("用户发了个偷笑的表情") is True
        assert reply_style.is_emotion_only_query("用户发了个无语的表情") is True

    def test_normal_query_not_detected(self):
        assert reply_style.is_emotion_only_query("可惜🤭") is False
        assert reply_style.is_emotion_only_query("你唱歌好好听") is False
        assert reply_style.is_emotion_only_query("用户问灰泽满喜欢什么水果") is False
        assert reply_style.is_emotion_only_query("") is False


class TestSplitDelay:
    def test_bounds(self):
        for _ in range(50):
            d = reply_style.split_delay("今天天气不错")
            assert core.SPLIT_DELAY_MIN_MS / 1000 * 0.85 <= d <= core.SPLIT_DELAY_MAX_MS / 1000 * 1.15

    def test_longer_part_never_slower_than_min(self):
        # 长文本延迟应显著高于短文本的下限
        assert reply_style.split_delay("一" * 50) > core.SPLIT_DELAY_MIN_MS / 1000


class TestComposeRecordMsg:
    def test_image_only(self):
        assert core._compose_record_msg("", "动漫风灰发萌妹") == "[发送图片] 动漫风灰发萌妹"

    def test_text_with_image(self):
        assert core._compose_record_msg("看看这张", "一只猫") == "看看这张（附图：一只猫）"

    def test_no_vision(self):
        assert core._compose_record_msg("你好", "") == "你好"

    def test_blank_text_no_vision(self):
        assert core._compose_record_msg("", "") == ""


class TestEmbedQueryGuard:
    async def test_empty_text_returns_none(self):
        class _FakeEmbeddings:
            async def create(self, **kwargs):
                raise AssertionError("空输入不应调用 embedding API")

        class _FakeClient:
            embeddings = _FakeEmbeddings()

        assert await rag.embed_query(_FakeClient(), "") is None
        assert await rag.embed_query(_FakeClient(), "   ") is None


class TestBuildMessageListImage:
    def _now(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")

    def test_image_only_composes_user_message(self, monkeypatch):
        self._now(monkeypatch)
        msgs = core.build_message_list("", "p", [], "", [], vision_desc="一碗面")
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "[图片：一碗面]"

    def test_text_plus_image(self, monkeypatch):
        self._now(monkeypatch)
        msgs = core.build_message_list("看看这张", "p", [], "", [], vision_desc="一碗面")
        assert msgs[-1]["content"] == "看看这张\n[图片：一碗面]"


class TestHandleChatEmotionOnly:
    """纯情绪消息（'可惜🤭'→probe补全成'用户发了个偷笑的表情'）应跳过语义检索，
    防误命中无关样本（曾导致回'喜欢新衣服'）。"""

    async def test_emotion_only_skips_retrieval(self, monkeypatch):
        async def fake_probe(uid, msg, hist, client):
            return "用户发了个偷笑的表情"
        monkeypatch.setattr(core, "probe_session", fake_probe)

        def _fail(*a, **k):
            raise AssertionError("纯情绪消息不应触发检索/行为分类")

        monkeypatch.setattr(core, "embed_query", _fail)
        monkeypatch.setattr(core, "retrieve_corpus", _fail)
        monkeypatch.setattr(core, "retrieve_voice_samples", _fail)
        monkeypatch.setattr(core, "retrieve_phrases", _fail)
        monkeypatch.setattr(core, "classify_behavior", _fail)

        captured = {}

        async def fake_reply(messages):
            captured["msgs"] = messages
            return "可惜啥"

        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "append_user_history", lambda *a, **k: None)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")

        async def _noop(*a, **k):
            pass

        monkeypatch.setattr(core, "update_memory_task", _noop)
        reply = await core.handle_chat("u", "可惜🤭")
        assert reply == "可惜啥"
        # 语气提示仍注入（补全句给模型理解情绪）
        assert any("用户发了个偷笑的表情" in m.get("content", "") for m in captured["msgs"])


class TestHandleChatImageOnly:
    async def test_image_only_skips_retrieval_and_replies_to_image(self, monkeypatch):
        embed_called = []

        async def fake_embed(client, text):
            embed_called.append(text)
            return [0.1]

        def _fail(*a, **k):
            raise AssertionError("图片-only 消息不应触发检索")

        monkeypatch.setattr(core, "embed_query", fake_embed)
        monkeypatch.setattr(core, "retrieve_corpus", _fail)
        monkeypatch.setattr(core, "retrieve_voice_samples", _fail)
        monkeypatch.setattr(core, "classify_behavior", _fail)  # L3：图片-only 也不做行为意图分类
        monkeypatch.setattr(core, "retrieve_phrases", _fail)
        monkeypatch.setattr(core, "fuse_and_truncate", _fail)

        captured = {}

        async def fake_reply(messages):
            captured["user"] = messages[-1]["content"]
            return "这碗面看起来好香"

        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "append_user_history", lambda *a, **k: None)

        async def _noop(*a, **k):
            pass

        monkeypatch.setattr(core, "update_memory_task", _noop)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")

        reply = await core.handle_chat("u", "", vision_desc="一碗加煎蛋的面")
        assert embed_called == []                       # 图片-only 不做检索
        assert captured["user"].startswith("[图片：")     # 图片并入用户消息
        assert reply == "这碗面看起来好香"
