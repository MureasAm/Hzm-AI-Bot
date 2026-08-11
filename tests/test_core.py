"""core 层：图片消息记忆记录 + 图片消息处理逻辑的单元测试。"""
from src.plugins.chatbot import core
from src.plugins.chatbot import rag


class TestSplitReply:
    def test_short_reply_not_split(self):
        assert core.split_reply("好的") == ["好的"]

    def test_single_sentence_not_split(self):
        assert core.split_reply("今天天气真不错啊") == ["今天天气真不错啊"]

    def test_short_clauses_merge_to_one(self):
        # 短句+短句合并成一条自然消息，不再拆成碎条（"哦？"不该单独成条）
        assert core.split_reply("今天好冷。你那边呢？") == ["今天好冷。你那边呢？"]

    def test_ellipsis_short_leadin_not_split(self):
        # 省略号前文太短（"唱拉了……"是犹豫前缀）不切，避免把"灰泽满……"单独发一条
        assert core.split_reply("唱拉了……灰泽满先关上门悄悄听会儿。") == \
            ["唱拉了……灰泽满先关上门悄悄听会儿"]

    def test_trailing_short_text_merged(self):
        # 结尾短段（嗯）并入前段，不单独成条
        assert core.split_reply("今天好冷。嗯", min_len=0) == ["今天好冷。嗯"]

    def test_micro_fragments_merge_to_natural(self):
        # 核心：短惊讶碎片（哦？/睡醒了？/那倒是稀奇……）合并成一条自然消息，不再戏剧节拍
        assert core.split_reply(
            "哦？睡醒了？那倒是稀奇……灰泽满这个点刚准备睡，你却醒了", min_len=0
        ) == ["哦？睡醒了？那倒是稀奇……", "灰泽满这个点刚准备睡，你却醒了"]

    def test_min_len_respected(self):
        # 低于 min_len 不拆；末尾句号仍会被去掉
        assert core.split_reply("好。", min_len=10) == ["好"]

    def test_comma_split_kept_when_long(self):
        # 长句的逗号拆分保留（只有短碎片才并回）
        assert core.split_reply(
            "灰泽满今天特别想出去玩，但是作业还没写完，明天还得早起去上第一节课", min_len=0
        ) == ["灰泽满今天特别想出去玩，但是作业还没写完", "明天还得早起去上第一节课"]

    def test_ellipsis_hesitation_preserved(self):
        # 省略号表"无语/语气"（后面没新内容）不切
        assert core.split_reply("啊……这……", min_len=0) == ["啊……这……"]

    def test_ellipsis_boundary_split_kept_when_both_long(self):
        # 省略号前后都有足够内容（都≥12字）时仍切分
        assert core.split_reply(
            "灰泽满今天嗓子特别不舒服……明天就想早点下播休息一下", min_len=0
        ) == ["灰泽满今天嗓子特别不舒服……", "明天就想早点下播休息一下"]


class TestEchoReply:
    """复读机防护：检测新回复是否复读最近自己说过的话。"""

    def test_exact_duplicate(self):
        assert core._is_echo_reply(
            "灰泽满刚醒，你倒是精神好", ["灰泽满刚醒，你倒是精神好"]
        ) is True

    def test_shared_tail(self):
        # "你这话说的……灰泽满刚醒" 复读后半句，最长公共子串覆盖 >60%
        assert core._is_echo_reply(
            "你这话说的……灰泽满刚醒，你倒是精神好", ["灰泽满刚醒，你倒是精神好"]
        ) is True

    def test_reverse_order(self):
        # 完整复读句（>8字）在更长旧句里，也判为复读
        assert core._is_echo_reply(
            "灰泽满刚醒，你倒是精神好", ["大早上的玩这个……灰泽满刚醒，你倒是精神好"]
        ) is True

    def test_short_not_flagged(self):
        # 太短不判（"晚安" 常见短句，避免误伤）
        assert core._is_echo_reply("晚安", ["晚安，做个好梦"]) is False

    def test_different_not_flagged(self):
        assert core._is_echo_reply("今天天气真不错", ["灰泽满刚醒，你倒是精神好"]) is False

    def test_empty_not_flagged(self):
        assert core._is_echo_reply("", ["灰泽满刚醒"]) is False


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
        assert core.clean_reply("（咽口水）你这又是来投毒的吧……") == "你这又是来投毒的吧……"

    def test_strip_multiple_leading(self):
        assert core.clean_reply("（心虚）（小声）灰泽满最近很忙……") == "灰泽满最近很忙……"

    def test_cap_to_one_parenthetical(self):
        # "a（小声）b（心虚）c" → 保留第一个（小声），去掉其余
        assert core.clean_reply("a（小声）b（心虚）c") == "a（小声）bc"

    def test_keeps_single_inline(self):
        assert core.clean_reply("刚啃完面包，别提了（咽口水）") == "刚啃完面包，别提了（咽口水）"

    def test_no_change_when_clean(self):
        assert core.clean_reply("今天天气真不错啊") == "今天天气真不错啊"

    def test_all_stripped_returns_original(self):
        assert core.clean_reply("（咽口水）") == "（咽口水）"

    def test_ellipsis_ascii_normalized(self):
        # 英文三点 ... 归一成中文省略号 ……
        assert core.clean_reply("好冷……真的...手都僵了") == "好冷……真的……手都僵了"

    def test_ellipsis_preserves_double(self):
        # "啊……这……" 无语表达，2 次省略号在额度内，原样保留
        assert core.clean_reply("啊……这……") == "啊……这……"

    def test_ellipsis_capped_keeps_two(self):
        # 刷屏式省略号，只保留前 2 个
        assert core.clean_reply("a……b……c……d……") == "a……b……cd"

    def test_ellipsis_long_dots_normalized(self):
        assert core.clean_reply("冷死了...........") == "冷死了……"


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


class TestSplitDelay:
    def test_bounds(self):
        for _ in range(50):
            d = core.split_delay("今天天气不错")
            assert core.SPLIT_DELAY_MIN_MS / 1000 * 0.85 <= d <= core.SPLIT_DELAY_MAX_MS / 1000 * 1.15

    def test_longer_part_never_slower_than_min(self):
        # 长文本延迟应显著高于短文本的下限
        assert core.split_delay("一" * 50) > core.SPLIT_DELAY_MIN_MS / 1000


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
        monkeypatch.setattr(core, "retrieve_behaviors", _fail)
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
