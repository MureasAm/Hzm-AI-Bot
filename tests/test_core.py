"""core 层：图片消息记忆记录 + 图片消息处理逻辑的单元测试。"""
import asyncio

from src.plugins.chatbot import core
from src.plugins.chatbot import memory as mem
from src.plugins.chatbot import rag
from src.plugins.chatbot import reply_style
from src.plugins.chatbot.reply_style import call_name_once, maybe_call_name
from src.plugins.chatbot import routing


class TestSplitReply:
    def test_short_reply_not_split(self):
        assert reply_style.split_reply("好的") == ["好的"]

    def test_single_sentence_not_split(self):
        assert reply_style.split_reply("今天天气真不错啊") == ["今天天气真不错啊"]

    def test_short_sentences_split_apart(self):
        # 逗号分段模式：短句也各自成条（真人就是这样连发的），不再焊回一条
        assert reply_style.split_reply("今天好冷。你那边呢？") == ["今天好冷", "你那边呢？"]

    def test_ellipsis_short_leadin_not_split(self):
        # 省略号前文太短（"唱拉了……"是犹豫前缀）不切，避免把"灰泽满……"单独发一条
        assert reply_style.split_reply("唱拉了……灰泽满先关上门悄悄听会儿。") == \
            ["唱拉了……灰泽满先关上门悄悄听会儿"]

    def test_trailing_short_text_kept_as_message(self):
        # 结尾的“嗯”单独一条也像真人；只在开头 1-2 字碎片才并入下一段
        assert reply_style.split_reply("今天好冷。嗯", min_len=0) == ["今天好冷", "嗯"]

    def test_micro_fragments_merge_into_next(self):
        # 问号不切；逗号两边不足 5 字时保留逗号、不切
        assert reply_style.split_reply(
            "哦？睡醒了？那倒是稀奇……灰泽满这个点刚准备睡，你却醒了", min_len=0
        ) == ["哦？睡醒了？那倒是稀奇……", "灰泽满这个点刚准备睡，你却醒了"]

    def test_min_len_respected(self):
        # 低于 min_len 不拆；末尾句号仍会被去掉
        assert reply_style.split_reply("好。", min_len=10) == ["好"]

    def test_commas_become_message_boundaries(self):
        # 逗号即分段：长句里的逗号切成多条，逗号本身不再出现在消息里
        assert reply_style.split_reply(
            "灰泽满今天特别想出去玩，但是作业还没写完，明天还得早起去上第一节课", min_len=0
        ) == ["灰泽满今天特别想出去玩", "但是作业还没写完", "明天还得早起去上第一节课"]

    def test_ellipsis_hesitation_preserved(self):
        # 省略号表"无语/语气"（后面没新内容）不切
        assert reply_style.split_reply("啊……这……", min_len=0) == ["啊……这……"]

    def test_ellipsis_boundary_split_kept_when_both_long(self):
        # 省略号前后都有足够内容（都≥12字）时仍切分
        assert reply_style.split_reply(
            "灰泽满今天嗓子特别不舒服……明天就想早点下播休息一下", min_len=0
        ) == ["灰泽满今天嗓子特别不舒服……", "明天就想早点下播休息一下"]

    def test_no_newline_leaks_into_parts(self):
        # 报告的案例，走真实链路（chat_window: clean_reply → split_reply）：
        # 短首句被并回后 split_reply 走 `return [text]` 原样返回，换行曾就此漏进消息
        reply = reply_style.clean_reply("没有。\n\n（小声）mua又不是糖，说给就给的")
        parts = reply_style.split_reply(reply)
        assert parts == ["没有（小声）mua又不是糖", "说给就给的"]
        assert not any("\n" in p for p in parts)

    def test_interjection_merges_into_next(self):
        # 多字语气词不单独发一条：“好啦，知道了” → “好啦知道了”；“嗯”跟下一句
        assert reply_style.split_reply("好啦，知道了。", min_len=0) == ["好啦知道了"]
        assert reply_style.split_reply("行了行了，别说了。", min_len=0) == ["行了行了别说了"]
        assert reply_style.split_reply("嗯，知道了。", min_len=0) == ["嗯，知道了"]

    def test_no_comma_no_runon(self):
        # 逗号分段后消息里不再有逗号；超限用空格接，不焊成 run-on
        parts = reply_style.split_reply(
            "真的是这样。明天还要早起呢。我先去睡了啊。你也早点休息。晚安。", min_len=0)
        assert not any("\n" in p for p in parts)
        assert not any("，" in p for p in parts)
        assert parts[0] == "真的是这样"
        assert parts[-1] == "你也早点休息 晚安"

    def test_overflow_parts_joined_without_newline(self):
        # 超限分段（>max_parts）接成一条，用空格不用换行、也不用逗号
        long_reply = "灰泽满今天特别想出去玩，但是作业还没写完。" * 6
        parts = reply_style.split_reply(long_reply, min_len=0, max_parts=2)
        assert len(parts) == 2
        assert not any("\n" in p for p in parts)
        assert " " in parts[1] and "，" not in parts[1]


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

    def test_find_repeat_word_removed(self):
        # 防措辞固化已删除（2026-09-12）：实测 75% 的触发在拦她自己的自称"灰泽满"。
        # 这里留一条守卫，防止有人翻回来又把那个机制加回去而不看 reason。
        assert not hasattr(reply_style, "find_repeat_word")
        assert not hasattr(reply_style, "_REPEAT_STOP")


    def test_window_beyond_default_caught(self):
        # 隔几句前说过的话被重复：默认只看最近 3 条会漏，放宽 window 才拦（回归：隔句复读）
        old = "灰泽满刚醒，你倒是精神好"
        reply = "你这话说的……灰泽满刚醒，你倒是精神好"
        recent = [f"旧话{i}" for i in range(4)] + [old] + [f"后话{i}" for i in range(3)]
        assert reply_style.is_echo_reply(reply, recent) is False            # 默认(3)漏网
        assert reply_style.is_echo_reply(reply, recent, window=8) is True   # 放宽拦住


class TestTimeHook:
    """「时间钩子」判据：只认时间词，别的口癖一概不碰（防误伤，见函数注释）。"""

    def test_detects_night_time_phrases(self):
        assert reply_style.uses_time_hook("凌晨三点了还不睡")
        assert reply_style.uses_time_hook("大半夜的说这个干嘛")
        assert reply_style.uses_time_hook("3点了你咋还没睡")
        assert reply_style.uses_time_hook("") is False

    def test_does_not_fire_on_her_other_tics(self):
        # 她的自称/嘴硬等口癖绝不能触发——历史教训：防措辞固化就是死在误伤自称"灰泽满"
        assert reply_style.uses_time_hook("灰泽满才没有呢") is False
        assert reply_style.uses_time_hook("行行行，你说了算") is False
        assert reply_style.uses_time_hook("明天直播记得来") is False


class TestHollowOpener:
    """「看看」空壳起手判据：只认"回复拿看看起头"，别的口癖一概不碰。"""

    def test_detects_the_real_shapes(self):
        # 全是该用户真实出现过的形态（从有宾语退化到没宾语）
        assert reply_style.uses_hollow_opener("灰泽满看看时间，这都快下午两点了吧")
        assert reply_style.uses_hollow_opener("灰泽满看看这个mua，又来了")
        assert reply_style.uses_hollow_opener("灰泽满看看，企鹅笔盒？")
        assert reply_style.uses_hollow_opener("（心虚）灰泽满看看你，追着问两轮了")
        assert reply_style.uses_hollow_opener("看看你这话说的")

    def test_not_at_start_is_fine(self):
        assert reply_style.uses_hollow_opener("你这话灰泽满看看再说") is False
        assert reply_style.uses_hollow_opener("行行行") is False
        assert reply_style.uses_hollow_opener("") is False

    def test_does_not_fire_on_normal_openers(self):
        # 通用"起手重复"实测会打中 18% 的正常回复；这条窄判据不能碰它们
        for s in ("今天过得怎么样呀", "嗯，知道了", "灰泽满哪能睡啊", "哦，就这个啊"):
            assert reply_style.uses_hollow_opener(s) is False, s


class TestFormulaOpener:
    """`灰泽满才没…` / `突然说这个…` 的模板判据（2026-10-20 A/B：模型级口癖）。"""

    def test_detects_two_known_templates(self):
        assert reply_style.formula_opener("灰泽满才没感动呢") == "才不/才没"
        assert reply_style.formula_opener("灰泽满才不呢") == "才不/才没"
        assert reply_style.formula_opener("我才不要") == "才不/才没"
        assert reply_style.formula_opener("才没有呢") == "才不/才没"
        assert reply_style.formula_opener("你这话灰泽满才没说过") == "才不/才没"
        assert reply_style.formula_opener("突然说这个干嘛") == "突然说这个"
        assert reply_style.formula_opener("啊，突然说这个干嘛") == "突然说这个"
        assert reply_style.formula_opener("（心虚）灰泽满才没有") == "才不/才没"

    def test_normal_openers_are_untouched(self):
        for s in ("今天过得怎么样呀", "嗯，知道了", "灰泽满哪能睡啊",
                  "才刚睡醒", "现在是两个人"):
            assert reply_style.formula_opener(s) == "", s

    def test_only_fires_when_recent_replies_repeat_it(self):
        recent = ["灰泽满才没感动", "别的", "我才不要呢"]
        assert reply_style.repeated_formula("灰泽满才不担心你", recent) == "才不/才没"
        assert reply_style.repeated_formula("灰泽满才没担心你", ["灰泽满才没感动", "别的", "别的"]) == ""
        assert reply_style.repeated_formula("今天天气不错", recent) == ""


class TestWeatherCity:
    """天气城市从 V2 的 location 事实取（v1 卡的 weather_city 回落已于 2026-10-12 撤掉）。"""

    def _patch(self, monkeypatch, memories):
        fake = type("S", (), {"snapshot": lambda self, uid: {"memories": memories}})()
        monkeypatch.setattr(core, "_MEMORY_V2_STORE", fake)

    def test_prefers_v2_location(self, monkeypatch):
        self._patch(monkeypatch, [
            {"kind": "fact", "key": "location", "value": "在曼谷", "status": "confirmed"}])
        assert core._weather_city_for("u") == "曼谷"

    def test_no_location_returns_empty(self, monkeypatch):
        # 取不到城市 → 空串（天气用默认城市），不再回落 v1 卡
        self._patch(monkeypatch, [])
        assert core._weather_city_for("u") == ""

    def test_cleans_dirty_value(self, monkeypatch):
        self._patch(monkeypatch, [
            {"kind": "fact", "key": "location", "value": "在泰国留学，但觉得选错了地方",
             "status": "confirmed"}])
        assert core._weather_city_for("u") == "泰国"

    def test_too_long_value_ignored(self, monkeypatch):
        self._patch(monkeypatch, [
            {"kind": "fact", "key": "location", "value": "一个非常非常长的地名描述啊",
             "status": "confirmed"}])
        assert core._weather_city_for("u") == ""

    def test_superseded_location_ignored(self, monkeypatch):
        self._patch(monkeypatch, [
            {"kind": "fact", "key": "location", "value": "在曼谷", "status": "superseded"}])
        assert core._weather_city_for("u") == ""


class TestNameOpener:
    """名字开头判据：只认"这条回复是不是拿用户的名字起手"，别的口癖一概不碰。"""

    def test_detects_name_at_start(self):
        assert reply_style.starts_with_name("草草子哥，你说啥", "草草子哥")
        assert reply_style.starts_with_name("（心虚）草草子哥……", "草草子哥")

    def test_name_not_at_start_is_fine(self):
        assert reply_style.starts_with_name("你说啥呢草草子哥", "草草子哥") is False
        assert reply_style.starts_with_name("行行行", "草草子哥") is False

    def test_empty_name_never_fires(self):
        assert reply_style.starts_with_name("草草子哥，你说啥", "") is False

    def test_strip_wrappers_gets_bare_name(self):
        """把 V2 里那些整句值剥成裸名字——**剥不干净守卫就会静默失效**。

        2026-10-12 实测：名单里漏了"希望被叫"，小温那条返回整句 → 守卫一次没拦，
        她 5/5 回复都用"小温，"开头；修好提取后 5/5 → 1/5。
        下面这些值都是**线上真实存过的**（`memory_v2_shadow.json`）。
        """
        cases = {
            "希望被叫小温": "小温",
            "用户自称四时小路": "四时小路",
            "用户自称阿梓": "阿梓",
            "用户要求被称为“爸爸”": "爸爸",
            "用户希望被称呼为草草子，不叫hikami": "草草子",
            "许流於": "许流於",          # 裸名字：原样返回
            "LG": "LG",
            "绿冻": "绿冻",
        }
        for raw, want in cases.items():
            got = core._bare_name_from(raw)
            assert got == want, f"{raw!r} → {got!r}（该是 {want!r}）"


class TestSummarizeBatch:
    async def test_single_message_skips(self, monkeypatch):
        def _fail(*a, **k):
            raise AssertionError("单条消息不应调用归纳")

        monkeypatch.setattr(core, "_get_clients", _fail)
        assert await core.summarize_batch([("在吗", "")]) == ""

    async def test_only_one_text_plus_image_skips(self, monkeypatch):
        """一条文字 + 一条图片不归纳：图片不该成为这批"的主题（2026-10-19）。"""
        def _fail(*a, **k):
            raise AssertionError("一条文字+图片不应调用归纳")

        monkeypatch.setattr(core, "_get_clients", _fail)
        assert await core.summarize_batch([("怎么评价", "[表情包]\n一只虫子")]) == ""

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


class TestSummarizeGroupBatch:
    """群聊对话线归纳：**带发言人**（谁在对谁说、停在哪），不是只归纳“聊了什么”。"""

    async def test_single_message_skips(self, monkeypatch):
        def _fail(*a, **k):
            raise AssertionError("单条群消息不用归纳")

        monkeypatch.setattr(core, "_get_clients", _fail)
        assert await core.summarize_group_batch([("甲", "今天下雨了", "")]) == ""

    async def test_prompt_carries_speakers(self, monkeypatch):
        seen = {}

        class _FakeCompletions:
            async def create(self, **kwargs):
                seen["prompt"] = kwargs["messages"][0]["content"]
                msg = type("M", (), {"content": "甲说下雨，乙在追问。"})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        fake_client = type("C", (), {
            "chat": type("Chat", (), {"completions": _FakeCompletions()})()})()
        monkeypatch.setattr(core, "_get_clients", lambda: (fake_client, None))
        out = await core.summarize_group_batch([
            ("甲", "今天下雨了", ""),
            ("乙", "真的假的", "[截图]天预报"),
        ])
        assert "甲：今天下雨了" in seen["prompt"]
        assert "乙：真的假的 [图片：[截图]天预报]" in seen["prompt"]
        assert "谁在对谁" in seen["prompt"]
        assert out == "甲说下雨，乙在追问。"

    async def test_failure_returns_empty(self, monkeypatch):
        def _raise():
            raise RuntimeError("no api")

        monkeypatch.setattr(core, "_get_clients", _raise)
        assert await core.summarize_group_batch([("甲", "a", ""), ("乙", "b", "")]) == ""


class TestBatchSummaryInjection:
    def test_injected_as_system_before_user(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗\n明天有空吗", [], [],
                                       batch_summary="用户约你明天见面")
        assert msgs[-1]["role"] == "user"
        assert any(m["role"] == "system" and "这批消息的归纳" in m["content"]
                   for m in msgs)

    def test_no_summary_no_injection(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", [], [])
        assert not any("这批消息的归纳" in m["content"] for m in msgs)


class TestCallName:
    """称呼替换（2026-10-30）：把一次安全的"你"换成对方要的名字。

    背景：用户 2151983431 的回复要么每句都叫"温宝"、要么一句都不叫（工具化检索之后）。
    现在在已经说出的话上换一个词，落在中间；频率由 NAME_CALL_RATE 控制。
    """

    def test_replaces_first_you(self):
        assert call_name_once("明天还来？那灰泽满提前祝你玩到手抽筋。", "温宝")             == "明天还来？那灰泽满提前祝温宝玩到手抽筋。"

    def test_skips_plural_and_greeting(self):
        assert call_name_once("你们今天好吵。", "温宝") == "你们今天好吵。"
        assert call_name_once("你好。", "温宝") == "你好。"

    def test_skips_quoted(self):
        assert call_name_once("灰泽满说了「你别闹」。", "温宝") == "灰泽满说了「你别闹」。"
        # 括号里的"你"多半是心里话/舞台说明，不动；括号外照常换
        assert call_name_once("（你别闹）灰泽满知道了。", "温宝") == "（你别闹）灰泽满知道了。"
        assert call_name_once("（心虚）你别闹。", "温宝") == "（心虚）温宝别闹。"

    def test_noop_when_name_already_there(self):
        assert call_name_once("谢谢温宝。", "温宝") == "谢谢温宝。"

    def test_noop_without_you(self):
        assert call_name_once("灰泽满知道了。", "温宝") == "灰泽满知道了。"

    def test_rate_zero_and_one(self):
        assert maybe_call_name("谢谢你。", "温宝", rate=0) == "谢谢你。"
        assert maybe_call_name("谢谢你。", "温宝", rate=1) == "谢谢温宝。"

    def test_empty_inputs(self):
        assert call_name_once("", "温宝") == ""
        assert call_name_once("谢谢你。", "") == "谢谢你。"


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


class TestNewlineCleanup:
    """换行归一：模型偶尔写 RP 式分段，一条消息里带多行（QQ 显示成剧本感，语音读断）。"""

    def test_newline_after_punctuation_joined(self):
        # 断行跟在句末标点后 → 直接接上（报告的原始案例）
        assert reply_style.clean_reply("没有。\n\n（小声）mua又不是糖，说给就给的") == \
            "没有。（小声）mua又不是糖，说给就给的"

    def test_single_newline_after_punctuation(self):
        assert reply_style.clean_reply("嗯。\n（心虚）") == "嗯。（心虚）"

    def test_newline_without_punctuation_becomes_comma(self):
        # 前面是实词 → 补逗号，别把两句焊成 run-on
        assert reply_style.clean_reply("今天真冷\n明天还得早起") == "今天真冷，明天还得早起"

    def test_leading_paren_after_newline_stripped(self):
        # 换行归一在"剥开头括号前缀"之前生效：原本因换行挡着没剥掉的前缀要剥掉
        assert reply_style.clean_reply("\n\n（小声）mua又不是糖，说给就给的") == \
            "mua又不是糖，说给就给的"

    def test_no_newline_leaks_when_all_stripped(self):
        # 剥光括号的兜底路径也不能把换行带出去
        assert "\n" not in reply_style.clean_reply("（心虚）\n（小声）")

    def test_no_newline_in_clean_output(self):
        assert "\n" not in reply_style.clean_reply(
            "灰泽满今天有点累。\n\n明天还得早起去上第一节课。\n不过还行。")


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


class TestSelfAsObjectFix:
    """「真拿灰泽满没办法」→「真拿你没办法」（2026-10-23）。

    她的第三人称自称偶尔会挤进“真拿你没办法”的**对象槽**，把“你”挤掉，语义直接反转。
    只改**无主语**的“真(是)拿…”；带主语的“你也拿灰泽满没办法”是对的（对方拿她没办法）。
    """

    def test_fixes_subjectless_true_frame(self):
        assert reply_style.clean_reply("真拿灰泽满没办法，别哭了啊……") == \
            "真拿你没办法，别哭了啊……"

    def test_fixes_variants(self):
        assert reply_style.clean_reply("可真拿灰泽满没辙了是吧") == "可真拿你没辙了是吧"
        assert reply_style.clean_reply("嗯，真是拿灰泽满没办法。") == "嗯，真是拿你没办法。"

    def test_keeps_correct_object_usage(self):
        # 带主语：对方拿她没办法，是对的
        assert reply_style.clean_reply("反正你也拿灰泽满没办法。") == \
            "反正你也拿灰泽满没办法。"
        assert reply_style.clean_reply("就吃你们拿灰泽满没办法这套。") == \
            "就吃你们拿灰泽满没办法这套。"

    def test_keeps_ambiguous_bare_form(self):
        # 句首裸露的“拿灰泽满没办法”也可能是“（你）拿她没办法”，有歧义，不动
        assert reply_style.clean_reply("拿灰泽满没办法呢，真受不了你。") == \
            "拿灰泽满没办法呢，真受不了你。"


class TestSelfReferenceMetrics:
    """自称方式改写的体检指标（skeleton-ab --check 用的那套）。"""

    def _m(self, replies):
        return reply_style.self_reference_metrics(replies)

    def test_person_and_negation_rates(self):
        m = self._m(["灰泽满才不去呢", "我觉得不行", "我们走吧", "好"])
        assert m["第一人称率"] == 0.25          # “我们”不算第一人称自称
        assert m["第三人称自指率"] == 0.25
        assert m["才不才没率"] == 0.25

    def test_name_initial_ignores_leading_paren(self):
        m = self._m(["（心虚）灰泽满才没有", "hzm不知道", "行吧"])
        assert m["名字起手率"] == round(2 / 3, 3)

    def test_pronoun_slip_counts_only_subjectless(self):
        m = self._m(["真拿灰泽满没办法，别哭了",
                     "反正你也拿灰泽满没办法", "别哭了"])
        assert m["代词错位率"] == round(1 / 3, 3)

    def test_opener_homogeneity(self):
        m = self._m(["好啊好啊", "好啊好啊", "行吧"])
        assert m["开头同质率"] == round(2 / 3, 3)

    def test_empty(self):
        assert self._m([]) == {}


class TestSelfReferenceCheck:
    """`skeleton-ab --check` 的闸门是纯函数，单测覆盖“基线通过 / 恶化失败 / 相对提醒”。"""

    BASE = {"第一人称率": 0.0, "第三人称自指率": 0.90, "名字起手率": 0.34,
            "才不才没率": 0.24, "代词错位率": 0.0, "开头同质率": 0.06}
    MAX = {"第一人称率": 0.10, "名字起手率": 0.45, "才不才没率": 0.55,
           "代词错位率": 0.0, "开头同质率": 0.30}
    MIN = {"第三人称自指率": 0.60}

    def _check(self, cand):
        return reply_style.self_reference_check(self.BASE, cand, self.MAX, self.MIN)

    def test_baseline_noise_passes(self):
        # 2026-10-23 实测：同一版骨架两次独立采样（名字起手 34%/38%、才不 24%/42%）
        cand = {**self.BASE, "第三人称自指率": 0.96,
                "名字起手率": 0.38, "才不才没率": 0.42}
        fails, warns = self._check(cand)
        assert fails == []
        # 才不 24%→42% 会触发相对提醒（>15%），但**不判死**——噪声范围内
        assert any("才不才没率" in w for w in warns)

    def test_regressions_fail(self):
        cand = {**self.BASE, "名字起手率": 0.60, "代词错位率": 0.02,
                "第三人称自指率": 0.50, "第一人称率": 0.20}
        fails, _ = self._check(cand)
        text = " ".join(fails)
        assert "名字起手率" in text
        assert "代词错位率" in text
        assert "第三人称自指率" in text
        assert "第一人称率" in text

    def test_relative_jump_warns_not_fails(self):
        cand = {**self.BASE, "才不才没率": 0.45}   # 45% 未超 55% 上限，但比基线高 21%
        fails, warns = self._check(cand)
        assert fails == []
        assert any("才不才没率" in w for w in warns)


class TestHistoryGapNote:
    """时间不再以「一行汇总」拼在历史顶部，而是**每条消息自带 [多久前]**。

    改因（2026-10-05）：旧做法下 gap=10秒 与 gap=3小时 注入的历史**长得一样**，
    她分不出「刚说完」和「三小时前说的」。现在由 `memory.get_user_history_timed`
    在每行前渲染 `[3小时前] `，注入层原样拼即可（渲染逻辑见 TestTimedHistory）。
    """

    def test_gap_note_is_not_prepended_anymore(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list(
            "在吗", [], ["用户：在吗", "灰泽满：在呢"], history_gap_note="3天")
        block = [m["content"] for m in msgs if m["content"].startswith("【最近对话记录】")]
        assert block
        # 旧的一行汇总**不该**再出现（否则等于同一件事说两遍）
        assert "距离上一轮对话" not in block[0]

    def test_no_note_when_empty(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list(
            "在吗", [], ["用户：在吗"], history_gap_note="")
        block = [m["content"] for m in msgs if m["content"].startswith("【最近对话记录】")]
        assert block and "距离上一轮对话" not in block[0]

    def test_nickname_hint_is_injected_near_user(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list(
            "灰宝在吗", [], [],
            nickname_hint="【称呼提示】用户这一轮叫你「灰宝」。",
        )
        assert any(m["content"].startswith("【称呼提示】") for m in msgs)

    def test_user_transcript_overrides_trailing_vision_block(self, monkeypatch):
        """私聊 batch 用按顺序拼好的 transcript；不要再把图片描述追加成尾部块。"""
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list(
            "怎么评价", [], [],
            vision_desc="[表情包]\n一只虫子",
            user_transcript="怎么看《空中杀人事件》\n怎么评价",
        )
        assert msgs[-1]["content"] == "怎么看《空中杀人事件》\n怎么评价"
        assert "[图片：" not in msgs[-1]["content"]


class TestLengthGuard:
    """超长 → 带提醒"重说一遍，短一半"（2026-10-14）。

    为什么它是模型内的可靠解法：骨架里"默认往短里走"只能定基调（换模型后尤其明显——
    gemini 默认 36 字，而她历史水平是 ~29）；`max_tokens` 硬压会腰斩半句。
    实测阈值 40：只触发约 1/4 轮次，gemini 35.7 → 25.4 字（≈历史水平），ds 完全不触发。
    """

    async def _run(self, monkeypatch, replies, guard="1"):
        monkeypatch.setenv("LENGTH_GUARD", guard)   # ⚠️ 别在这里 delenv——那会把用例设的值删掉
        seq = list(replies)
        calls = []

        async def fake_reply(messages):
            calls.append(len(messages))
            return seq.pop(0) if seq else "在呢"

        async def fake_probe(uid, msg, hist, client, debug=False):
            _v = '用户发了个偷笑的表情'
            return (_v, {}) if debug else _v       # 纯情绪 → 跳过检索，最小 harness

        async def _async_noop(*a, **k):
            return None

        def _noop(*a, **k):
            return None

        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", _noop)
        monkeypatch.setattr(core, "update_memory_v2_shadow_task", _async_noop)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        out = await core.handle_chat("u", "可惜🤭")
        await asyncio.sleep(0)
        return out, calls

    async def test_long_reply_gets_regenerated(self, monkeypatch):
        long_reply = "字" * (core.LENGTH_GUARD_MAX_CHARS + 5)
        out, calls = await self._run(monkeypatch, [long_reply, "在呢"])
        assert out == "在呢", "重说的那句要生效"
        assert len(calls) == 2, "超长应该触发一次重说"

    async def test_short_reply_untouched(self, monkeypatch):
        out, calls = await self._run(monkeypatch, ["在呢"])
        assert out == "在呢" and len(calls) == 1, "没超长就不该多调一次"

    async def test_can_be_disabled(self, monkeypatch):
        long_reply = "字" * (core.LENGTH_GUARD_MAX_CHARS + 5)
        out, calls = await self._run(monkeypatch, [long_reply], guard="0")
        assert out == long_reply and len(calls) == 1


class TestFallbackReplyNotRemembered:
    """生成失败的兜底文案**不能进记忆**。

    踩坑（实测发生）：上游偶发 `choices: null` 那一轮，兜底文案被当成"她的回复"存进短期记忆，
    她的历史里就永久留着一句 `哎呀，hzm脑子卡了一下……（错误: 'NoneType' ...）`
    ——AI 腔的故障信息变成"她说过的话"，还会作为 few-shot 喂回去。一次上游抖动 = 一份永久污染。

    2026-10-14 换过文案：现在是"小满现在不在哦~"（体面的缺席，不把故障摆上台面）。
    **老文案仍要认得**——历史记录里就存着。
    """

    def test_is_fallback_reply_detects_current(self):
        assert core.is_fallback_reply("小满现在不在哦~")
        assert core.is_fallback_reply("  小满现在不在哦~  ")

    def test_is_fallback_reply_still_detects_legacy(self):
        assert core.is_fallback_reply("哎呀，hzm脑子卡了一下……")
        assert core.is_fallback_reply("……（沉默，可能是信号不好）")

    def test_normal_reply_is_not_fallback(self):
        assert not core.is_fallback_reply("在呢，你说")
        assert not core.is_fallback_reply("")

    async def _run(self, monkeypatch, reply_text):
        captured = {"short": 0, "long": 0}

        async def fake_probe(uid, msg, hist, client, debug=False):
            _v = '用户发了个偷笑的表情'
            return (_v, {}) if debug else _v   # 纯情绪 → 跳过检索，最小 harness

        async def fake_reply(messages):
            return reply_text

        def fake_append(*a, **k):
            captured["short"] += 1

        async def fake_update(*a, **k):
            captured["long"] += 1

        async def _noop(*a, **k):
            pass

        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", fake_append)
        # 长期记忆已换成 V2（2026-10-10）：数它的提取任务即可
        monkeypatch.setattr(core, "update_memory_v2_shadow_task", fake_update)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")

        out = await core.handle_chat("u", "可惜🤭")
        await asyncio.sleep(0)   # 长期记忆走 asyncio.create_task，让调度器跑一拍
        return out, captured

    async def test_fallback_reply_not_stored(self, monkeypatch):
        out, captured = await self._run(monkeypatch, "小满现在不在哦~")
        assert out == "小满现在不在哦~", "兜底文案照常回给用户"
        assert captured["short"] == 0, "兜底文案不该进短期记忆"
        assert captured["long"] == 0, "兜底文案也不该触发长期记忆提取"

    async def test_normal_reply_still_stored(self, monkeypatch):
        out, captured = await self._run(monkeypatch, "在呢，你说")
        assert captured["short"] == 1
        assert captured["long"] == 1


class TestConsistencyRuleWording:
    """【一致性规则】只管事实、不管说法。

    旧文案是"借口要与之前保持一致"——**直接教它复读**：第一次用了某个借口
    （"威严"/"搬家"）就被规则锁死，错误只固化不自我纠正。
    这里把判据钉住，防止有人翻回去又写成"保持一致"。
    """

    def _block(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list("你怎么又鸽了", [], ["用户：你又鸽了", "灰泽满：在忙"])
        return next(m["content"] for m in msgs if m["content"].startswith("【最近对话记录】"))

    def test_scoped_to_facts_not_wording(self, monkeypatch):
        b = self._block(monkeypatch)
        assert "【别自相矛盾】" in b
        assert "只管**事实**" in b or "只管事实" in b

    def test_no_longer_tells_her_to_keep_the_same_excuse(self, monkeypatch):
        b = self._block(monkeypatch)
        assert "借口要与之前保持一致" not in b, "这句会把它锁死在第一次的说法上"

    def test_her_own_lines_are_injected(self, monkeypatch):
        """2026-10-18 用户定：短期记忆恢复完整上下文——用户 + 她自己最近 5 轮。

        2026-10-14 曾过滤掉她自己的话（去自反馈）；用户明确说那不是他要的"正反馈"。
        现在两边都注，`contradiction` 那条仍只约束事实、不约束说法。
        """
        b = self._block(monkeypatch)
        assert "灰泽满：在忙" in b
        assert "用户：你又鸽了" in b

    def test_only_bot_history_still_injects(self, monkeypatch):
        """整段都是她说的话时（如刚推完动态还没人回），现在也要注完整上下文。"""
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list("在吗", [], ["[3小时前] 灰泽满：睡了"])
        block = [m["content"] for m in msgs if m["content"].startswith("【最近对话记录】")]
        assert block and "灰泽满：睡了" in block[0]

    def test_timed_prefix_does_not_hide_her_lines(self, monkeypatch):
        """带 `[3小时前] ` 前缀的行同样是"她说的"，别被前缀骗过去。"""
        assert mem.is_bot_line("[3小时前] 灰泽满：睡了")
        assert not mem.is_bot_line("[3小时前] 用户：在吗")
        assert not mem.is_bot_line("灰泽满这个名字好听")     # 用户在说"灰泽满"不算

    def test_repeat_question_no_longer_exempts_anti_repeat(self, monkeypatch):
        # 旧【防复读】写"用户没有主动追问时"才不重复 → 反复追问正好是复读高发场景却豁免了
        b = self._block(monkeypatch)
        assert "用户没有主动追问时" not in b


class TestScheduleInjection:
    """周表注入：只讲"什么时候播"。

    曾有的「近况」（自由文本+TTL）已整条删除——它治不了"用旧记忆回答当下"，
    真正的漏点是 voice_samples 那条 assistant turn 通道。见 constants.py 的说明。
    """

    def _inject(self, monkeypatch, sched):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: sched)
        msgs = core.build_message_list("你怎么又鸽了", [], [])
        block = [m["content"] for m in msgs if "周表" in m["content"]]
        return block[0] if block else ""

    def test_weekly_injected(self, monkeypatch):
        text = self._inject(monkeypatch, {"weekly": [{"day": "周六", "time": "19:00"}]})
        assert "周六 19:00" in text

    def test_note_field_no_longer_injected(self, monkeypatch):
        # 防回归：就算有人把「近况」写回 schedule.json，也不该被注入
        text = self._inject(monkeypatch, {
            "weekly": [{"day": "周一", "time": "20:00"}],
            "近况": "这周在收拾搬家",
        })
        assert "搬家" not in text, "被删除的近况字段不该再被注入"
        assert "周一 20:00" in text, "周表（weekday 制，不过期）要照常注入"


class TestGroupEventAge:
    """群近况事件存了 t，注入时要带相对时间（以前只取 text → 上周的事被当"现在"）。"""

    async def test_event_injected_with_relative_age(self, monkeypatch):
        import time as _t
        now = _t.time()
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core.group_memory, "get_group", lambda gid: {
            "members": {"1": "绿冻A"},
            "events": [{"t": now - 3 * 86400, "text": "群里在聊新游戏"},
                       {"t": now - 60, "text": "刚有人发了张图"}],
        })
        monkeypatch.setattr(core, "load_schedule", lambda: {})

        async def fake_probe(uid, msg, hist, client, debug=False):
            return (msg, {}) if debug else msg

        monkeypatch.setattr(core, "probe_session", fake_probe)

        async def _no_intent(*a, **k):   # classify_l3 是 await 的
            return {"behavior": "", "phrases": []}

        monkeypatch.setattr(core, "classify_l3", _no_intent)

        captured = {}

        async def fake_reply(messages):
            captured["msgs"] = messages
            return "嗯"

        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "append_user_history", lambda *a, **k: None)

        async def _noop(*a, **k):
            pass

        await core.handle_chat("12345", "群里有啥好玩的", is_group=True)

        # ⚠️ 必须按**段头**匹配：骨架里也写着"【群聊现场】"（告诉她群里会多一块），
        #    用 `in` 会先命中骨架那条 —— 断言就变成在测骨架、永远测不到群近况（2026-10-15 踩到）。
        block = next((m["content"] for m in captured["msgs"]
                      if m["content"].startswith("【群聊现场】")), "")
        assert "群里在聊新游戏（3天前）" in block, f"群近况该带相对时间，实际：{block}"
        # 刚发生的事不带时间前缀（humanize_gap 对 <90s 返回空串）——读起来就是"现在"，
        # 标个"1分钟前"反而是噪音。
        assert "刚有人发了张图" in block and "发了张图（" not in block


class TestPreferences:
    def test_injects_retrieved_preferences(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", [], [], preference_items=[
            {"category": "食物", "text": "爱吃椰子鸡", "score": 0.6},
            {"category": "作息", "text": "夜猫子", "score": 0.5}])
        injected = [m["content"] for m in msgs if "灰泽满的偏好" in m["content"]]
        assert injected and "爱吃椰子鸡" in injected[0] and "夜猫子" in injected[0]

    def test_no_preference_items_no_injection(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", [], [])
        assert not any("灰泽满的偏好" in m["content"] for m in msgs)


class TestMemoryV2Segments:
    """Memory V2 三段注入：靠前相处方式 / 中段约定 / 末尾情绪（docs/记忆v2设计.md §三）。"""

    def _msgs(self, monkeypatch, **kwargs):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        return core.build_message_list("在吗", [], [], **kwargs)

    def test_absent_by_default(self, monkeypatch):
        msgs = self._msgs(monkeypatch)
        assert not any(
            any(tag in m["content"] for tag in ("相处方式", "未完成约定", "此刻的情绪"))
            for m in msgs
        )

    def test_segments_land_in_their_positions(self, monkeypatch):
        msgs = self._msgs(
            monkeypatch,
            profile_context="【和这个绿冻的相处方式】\n回复尽量短。",
            commitment_context="【相关未完成约定】下次给你唱一段",
            emotion_context="【用户此刻的情绪】因为考试很失落；本轮优先照顾当前情绪。",
        )
        roles = [m["role"] for m in msgs]
        assert roles[-1] == "user"

        profile_i = next(i for i, m in enumerate(msgs) if "相处方式" in m["content"])
        commit_i = next(i for i, m in enumerate(msgs) if "未完成约定" in m["content"])
        emotion_i = next(i for i, m in enumerate(msgs) if "此刻的情绪" in m["content"])
        user_i = len(msgs) - 1

        # 顺序：相处方式靠前 < 约定中段 < 情绪紧挨用户消息（只输给末尾的自称一句）。
        assert profile_i < commit_i < emotion_i
        assert emotion_i == user_i - 2
        assert msgs[user_i - 1]["role"] == "system"
        assert "需要自称就用" in msgs[user_i - 1]["content"]


class TestPrevSessionNote:
    """上一场会话（已过期）**单独一条**注入，不能并进【当前会话】。

    并进去就坏了：【当前会话】那句写的是"你们这一场对话的调性"，
    几天前的旧事塞进那个标签下会被当成现在正在聊的。
    """

    def test_injects_when_present(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", [], [],
                                       prev_session_note="【上次聊过】3天前你们聊到「香水」。")
        hits = [m["content"] for m in msgs if "【上次聊过】" in m["content"]]
        assert hits and "3天前" in hits[0]

    def test_absent_by_default(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        msgs = core.build_message_list("在吗", [], [])
        assert not any("【上次聊过】" in m["content"] for m in msgs)


class TestCorpusTimePrefix:
    """corpus 每条**自己带**时间定性前缀（2026-10-15）。

    段头那句总说明（"都是曾经发生的"）实测管不住模型——见 `docs/架构.md` §4.5「包装语无效」：
    说明写在整块外面，模型读单条时不会往回看。所以把定性**挂到每条自己身上**。
    ⚠️ 走的是**注入层**：数据层（statement_final）加前缀会让 421 条向量都多同一个常数偏移、
    稀释条间区分度，还得重算全库——**别改成数据层**。
    """

    def _msgs(self, monkeypatch, texts):
        from src.plugins.chatbot.retrieval import RetrievalItem
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        items = [RetrievalItem(source="corpus", item_id=str(i), score=0.9, text=t)
                 for i, t in enumerate(texts)]
        return core.build_message_list("你最近忙不忙", items, [])

    def test_every_corpus_line_carries_the_prefix(self, monkeypatch):
        msgs = self._msgs(monkeypatch, ["她第二天要去拿搬家的钥匙。", "她上周熬夜改稿到三点。"])
        block = [m["content"] for m in msgs if m["content"].startswith("【她经历过的相关背景】")]
        assert block, "corpus 块没注入"
        assert block[0].count("- 【这是过去发生过的事】") == 2
        # 前缀和原句必须**同一条线**上（分开写就等于又变成"块级说明"了）
        assert "- 【这是过去发生过的事】她第二天要去拿搬家的钥匙。" in block[0]

    def test_behavior_lines_do_not_carry_it(self, monkeypatch):
        """行为指令是"现在该怎么做"，不是过去的事——不能带这个前缀。"""
        from src.plugins.chatbot.retrieval import RetrievalItem
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        items = [RetrievalItem(source="behavior", item_id="被夸时把夸奖调小", score=1.0,
                               text="【被夸时把夸奖调小】\n回应模式：压低评价。")]
        msgs = core.build_message_list("你好厉害", items, [])
        joined = "\n".join(m["content"] for m in msgs)
        assert "【当前情境下的行为指令】" in joined
        assert "【这是过去发生过的事】" not in joined


    def test_hit_log_records_only_items_that_enter_prompt(self, monkeypatch):
        """留痕点必须在**渲染之后**、只记真正进提示词的 corpus 条目。

        2026-10-18：这条日志是"库利用率"的唯一真值来源——记候选或记判官保留都不对，
        因为 RRF/字符预算还可能再截掉一部分。
        """
        from types import SimpleNamespace
        from src.plugins.chatbot.retrieval import RetrievalItem
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        recorded = {}
        monkeypatch.setattr(core, "get_llm_route", lambda: "私")
        monkeypatch.setattr(core, "log_corpus_hits",
                            lambda q, items: recorded.update(
                                query=q, ids=[it.item_id for it in items]))
        monkeypatch.setattr(core, "log_block_hits",
                            lambda blocks: recorded.update(blocks=list(blocks)))
        items = [
            RetrievalItem(source="corpus", item_id="42", score=0.9, text="她以前被问过身高。"),
            RetrievalItem(source="corpus", item_id="43", score=0.8, text=""),  # 空文本不渲染
            SimpleNamespace(source="behavior", item_id="被夸", score=1.0, text="【被夸】压低评价。"),
        ]
        core.build_message_list("你多高啊", items, [])
        assert recorded == {}, "离线/实验路径默认不写命中日志"
        core.build_message_list(
            "你多高啊", items, ["用户：在吗"],
            commitment_context="【相关未完成约定】测试",
            same_request={"count": 2, "what": "要语音"},
            batch_summary="测试摘要", query_hint="测试扩充",
            emotion_context="【用户此刻的情绪】测试",
            daily_context="今天中午吃了炸鸡",
            log_hits=True,
        )
        assert recorded["query"] == "你多高啊"
        assert recorded["ids"] == ["42"]
        assert set(recorded["blocks"]) == {
            "commitment", "history", "same_request", "batch",
            "query_hint", "now", "mood", "daily", "selfref",
        }


class TestInjectionTrace:
    """条目级注入轨迹（`docs/数据迭代.md` §2）：只记**实际进提示词**的块与条目。

    为什么要有：只有块名时，模型诊断只能猜"可能是 behaviors"；条目级才能定位到
    "哪一条素材/记忆/词条在这一轮里出现过"。候选和判官保留都不算——只记最终渲染进提示词的。
    """

    def _build(self, monkeypatch, trace=None):
        from src.plugins.chatbot.retrieval import RetrievalItem
        monkeypatch.setattr(core.context_probe, "get_now_context",
                            lambda city="": "【当前时间】测试上午十点\n第二行不记")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        monkeypatch.setattr(core, "load_terms", lambda: [
            {"keyword": "绿冻", "meaning": "她的粉丝", "priority": "always"},
            {"keyword": "鸽子", "meaning": "爱放鸽子的人", "aliases": ["咕咕"]},
        ])
        items = [
            RetrievalItem(source="corpus", item_id="42", score=0.9, text="她以前被问过身高。"),
            RetrievalItem(source="corpus", item_id="43", score=0.8, text=""),  # 空文本不渲染
            RetrievalItem(source="behavior", item_id="被夸时把夸奖调小", score=1.0,
                          text="【被夸时把夸奖调小】\n回应模式：压低评价。"),
        ]
        kwargs = dict(
            preference_items=[{"id": "food", "category": "食物", "text": "爱椰子鸡"}],
            core_stories=[{"id": "good_student_38", "category": "好学生", "text": "她当年…"}],
            commitment_context="【相关未完成约定】测试",
            same_request={"count": 2, "what": "要语音"},
            batch_summary="测试摘要", query_hint="测试扩充",
            emotion_context="【用户此刻的情绪】测试",
            daily_context="今天中午吃了炸鸡",
        )
        if trace is not None:
            kwargs["trace"] = trace
        return core.build_message_list("灰泽满你怎么又咕咕了", items, ["用户：在吗"], **kwargs)

    def test_blocks_follow_actual_injection_order(self, monkeypatch):
        trace = {}
        self._build(monkeypatch, trace)
        assert trace["blocks"] == [
            "preferences", "core_stories", "terms", "behaviors", "corpus",
            "commitment", "history", "same_request", "batch", "query_hint",
            "now", "mood", "daily", "selfref",
        ]

    def test_items_only_rendered_entries(self, monkeypatch):
        trace = {}
        self._build(monkeypatch, trace)
        assert trace["items"] == {
            "preferences": ["food"],
            "core_stories": ["good_student_38"],
            "terms": ["绿冻", "鸽子"],          # always + 别名命中
            "behaviors": ["被夸时把夸奖调小"],
            "corpus": ["42"],                   # "43" 文本为空、没渲染，不能记
        }

    def test_values_record_compact_perception(self, monkeypatch):
        trace = {}
        self._build(monkeypatch, trace)
        assert trace["values"]["now"] == "【当前时间】测试上午十点"   # 只取首行
        assert trace["values"]["mood"] == "【用户此刻的情绪】测试"
        assert trace["values"]["daily"] is True

    def test_absent_blocks_are_not_marked(self, monkeypatch):
        trace = {}
        self._build(monkeypatch, trace)
        assert "nickname" not in trace["blocks"] and "group" not in trace["blocks"]

    def test_no_trace_arg_is_silent(self, monkeypatch):
        """离线/实验路径不传 trace 时，构建照旧、不会因为轨迹而报错。"""
        msgs = self._build(monkeypatch, None)
        assert msgs and msgs[-1]["role"] == "user"


class TestInjectionTraceWrite:
    """`handle_chat` 只在真实聊天通道（私/群）落盘轨迹——离线回放不能混进周报。"""

    async def _run(self, monkeypatch, route):
        recorded = []

        async def fake_probe(uid, msg, hist, client, debug=False):
            _v = '用户发了个偷笑的表情'
            return (_v, {}) if debug else _v   # 纯情绪 → 跳过检索，最小 harness

        async def fake_reply(messages):
            return "在呢"

        async def _anoop(*a, **k):
            return None

        def _noop(*a, **k):
            return None

        monkeypatch.setenv("MEMORY_V2_INJECT", "0")
        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", _noop)
        monkeypatch.setattr(core, "update_memory_v2_shadow_task", _anoop)
        monkeypatch.setattr(core, "get_llm_route", lambda: route)
        monkeypatch.setattr(core, "log_corpus_hits", _noop)
        monkeypatch.setattr(core, "log_block_hits", _noop)
        monkeypatch.setattr(core, "log_injection_trace", recorded.append)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        await core.handle_chat("u", "可惜🤭")
        return recorded

    async def test_private_route_records_trace(self, monkeypatch):
        rec = await self._run(monkeypatch, "私")
        assert len(rec) == 1
        assert rec[0]["session"] == "u" and rec[0]["route"] == "私"
        assert "now" in rec[0]["blocks"]
        assert rec[0]["reply_used_nickname"] is False

    async def test_offline_route_records_nothing(self, monkeypatch):
        assert await self._run(monkeypatch, "") == []


class TestMetaSilence:
    """未点名群聊批次的“元回复”识别（2026-10-20 群 901907410 实测）。"""

    def test_detects_known_meta_replies(self):
        for t in ("没点灰泽满，灰泽满才不管。",
                  "没点我，才不管呢",
                  "（群里没人 @ 灰泽满，保持潜水状态，没有回复。）",
                  "（不说话，静静看着群里聊天）",
                  "没人叫我，不插话"):
            assert reply_style.is_meta_silence(t), t

    def test_normal_group_chat_not_flagged(self):
        for t in ("灰泽满才不管你钱包空不空呢，有得买就不错啦。",
                  "群里聊天呢，翁大帅怎么突然关心起这个了。",
                  "灰泽满没点开也知道是那种视频。",
                  "大疲？这是哪儿的方言。"):
            assert not reply_style.is_meta_silence(t), t


class TestGroupMetaSilenceNotSent:
    """未点名的群聊批次：元回复不发、只记用户侧；点名批次不拦。"""

    async def _run(self, monkeypatch, reply_text, addressed):
        recorded = []

        async def fake_probe(uid, msg, hist, client, debug=False):
            return ("用户发了个无语的表情", {})   # 纯情绪 → 跳过检索

        async def fake_reply(messages):
            return reply_text

        def fake_append(uid, user, reply, **kwargs):
            recorded.append((user, reply))

        async def _anoop(*a, **k):
            return None

        def _noop(*a, **k):
            return None

        monkeypatch.setenv("MEMORY_V2_INJECT", "0")
        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", fake_append)
        monkeypatch.setattr(core, "update_memory_v2_shadow_task", _anoop)
        monkeypatch.setattr(core, "get_llm_route", lambda: "")
        monkeypatch.setattr(core, "log_corpus_hits", _noop)
        monkeypatch.setattr(core, "log_block_hits", _noop)
        monkeypatch.setattr(core.group_memory, "get_group", lambda gid: {})
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        out = await core.handle_chat("g1", "大疲", is_group=True, group_addressed=addressed)
        return out, recorded

    async def test_meta_reply_suppressed_when_not_addressed(self, monkeypatch):
        out, recorded = await self._run(monkeypatch, "没点灰泽满，灰泽满才不管。", addressed=False)
        assert out == ""
        assert recorded == [("大疲", "")], "只记用户侧，不记那句元回复"

    async def test_meta_reply_kept_when_addressed(self, monkeypatch):
        out, recorded = await self._run(monkeypatch, "没点灰泽满，灰泽满才不管。", addressed=True)
        assert out == "没点灰泽满，灰泽满才不管。"
        assert recorded == [("大疲", "没点灰泽满，灰泽满才不管。")]


class TestGroupHistoryLabels:
    """群聊短期记忆带说话人标签：`群聊：甲：…` / `灰泽满（回甲）：…`（2026-10-23）。"""

    async def test_group_reply_records_reply_to(self, monkeypatch):
        recorded = []

        async def fake_probe(uid, msg, hist, client, debug=False):
            return ("用户发了个无语的表情", {})   # 纯情绪 → 跳过检索

        async def fake_reply(messages):
            return "在呢"

        def fake_append(uid, user, reply, user_label="用户", reply_label="灰泽满"):
            recorded.append((user_label, user, reply_label, reply))

        async def _anoop(*a, **k):
            return None

        def _noop(*a, **k):
            return None

        monkeypatch.setenv("MEMORY_V2_INJECT", "0")
        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", fake_append)
        monkeypatch.setattr(core, "update_memory_v2_shadow_task", _anoop)
        monkeypatch.setattr(core, "get_llm_route", lambda: "")
        monkeypatch.setattr(core, "log_corpus_hits", _noop)
        monkeypatch.setattr(core, "log_block_hits", _noop)
        monkeypatch.setattr(core.group_memory, "get_group", lambda gid: {})
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})

        await core.handle_chat("g1", "甲：大疲", is_group=True, group_addressed=True,
                               record_text="甲：大疲", reply_to="甲")

        assert recorded == [("群聊", "甲：大疲", "灰泽满（回甲）", "在呢")]

    async def test_group_skips_name_budget(self, monkeypatch):
        """群聊不注入称呼预算：群里没有“单独对某个人称呼”的语境。"""
        seen = {}

        def fake_build(*a, **k):
            seen.update(k)
            return [{"role": "system", "content": "x"}]

        async def fake_probe(uid, msg, hist, client, debug=False):
            return ("用户发了个无语的表情", {})

        async def fake_reply(messages):
            return "在呢"

        async def _anoop(*a, **k):
            return None

        def _noop(*a, **k):
            return None

        monkeypatch.setenv("MEMORY_V2_INJECT", "0")
        monkeypatch.setattr(core, "build_message_list", fake_build)
        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history",
                            lambda uid: ["灰泽满（回甲）：温温在呢", "灰泽满（回甲）：温温早"])
        monkeypatch.setattr(core, "_user_name_for", lambda uid: "温温")
        monkeypatch.setattr(core, "get_user_history_timed", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", _noop)
        monkeypatch.setattr(core, "update_memory_v2_shadow_task", _anoop)
        monkeypatch.setattr(core, "get_llm_route", lambda: "")
        monkeypatch.setattr(core, "log_corpus_hits", _noop)
        monkeypatch.setattr(core, "log_block_hits", _noop)
        monkeypatch.setattr(core.group_memory, "get_group", lambda gid: {})
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})

        await core.handle_chat("g1", "甲：大疲", is_group=True, group_addressed=True,
                               record_text="甲：大疲", reply_to="甲")

        assert seen.get("name_budget") == ""


class TestNameBudget:
    """称呼预算（2026-10-23）：最近 5 轮她叫对方名字太多次 → 这轮前置提醒。"""

    def _note(self, monkeypatch, hist, name="温温", min_calls=None):
        monkeypatch.setattr(core, "_user_name_for", lambda uid: name)
        monkeypatch.setattr(core, "get_user_history", lambda uid: hist)
        if min_calls is not None:
            monkeypatch.setenv("NAME_BUDGET_MIN_CALLS", str(min_calls))
        return core.name_budget_note("u1")

    def test_no_name_returns_empty(self, monkeypatch):
        note = self._note(monkeypatch, ["灰泽满：在呢"], name="")
        assert note == ""

    def test_no_call_returns_empty(self, monkeypatch):
        note = self._note(monkeypatch, ["用户：温温在吗", "灰泽满：在呢"])
        assert note == ""

    def test_single_call_fires_by_default(self, monkeypatch):
        """默认档 = 5 轮内出现过就提醒（A/B 里最有效的一档，2026-10-23）。"""
        note = self._note(monkeypatch, ["灰泽满：温温在呢", "用户：嗯"])
        assert "温温" in note and "1 次" in note
        assert "尽量别再叫" in note

    def test_only_her_lines_count(self, monkeypatch):
        """用户自己提名字不算她称呼滥用。"""
        note = self._note(monkeypatch, [
            "用户：温温在吗", "用户：温温？", "灰泽满：在呢",
        ])
        assert note == ""

    def test_group_style_bot_lines_count(self, monkeypatch):
        """群聊历史行是「灰泽满（回甲）：…」，也认（函数本身与标签无关）。"""
        note = self._note(monkeypatch, [
            "灰泽满（回甲）：温温早", "灰泽满（回甲）：温温吃了吗",
        ])
        assert "温温" in note and "2 次" in note

    def test_window_is_last_five_rounds(self, monkeypatch):
        """只数最近 10 行（5 轮）；更早的叫过不算。"""
        old = ["灰泽满：温温早"] * 4
        recent = ["用户：嗯", "灰泽满：在呢"] * 5
        assert self._note(monkeypatch, old + recent) == ""

    def test_env_threshold_override(self, monkeypatch):
        hist = ["灰泽满：温温在呢"]
        assert self._note(monkeypatch, hist, min_calls=2) == ""
        assert self._note(monkeypatch, hist + ["灰泽满：温温早"], min_calls=2) != ""
        assert self._note(monkeypatch, hist, min_calls=0) == ""

    def test_injection_order_after_profile(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list(
            "在吗", [], [],
            profile_context="【和这个绿冻的相处方式】测试",
            name_budget="【称呼】最近 5 轮里已经叫过他「温温」2 次，这轮尽量别再叫名字，直接说事。",
        )
        texts = [m["content"] for m in msgs if m["role"] == "system"]
        idx_profile = next(i for i, t in enumerate(texts)
                           if t.startswith("【和这个绿冻的相处方式】"))
        idx_name = next(i for i, t in enumerate(texts) if t.startswith("【称呼】"))
        assert idx_profile < idx_name

    def test_not_injected_when_empty(self, monkeypatch):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list("在吗", [], [], name_budget="")
        assert not any(m["content"].startswith("【称呼】") for m in msgs)


class TestSceneBlock:
    """私聊/群聊场景（00_scene）：「现在在哪儿、对面是谁」只按 is_group 二选一。

    真值只有 NapCat 的 message_type（经 handle_chat 传 is_group）——
    别从文本、group_context 是否为空、to_me 推断。
    """

    def _msgs(self, monkeypatch, is_group):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        return core.build_message_list("在吗", [], [], is_group=is_group)

    def test_private_scene_lives_in_skeleton(self, monkeypatch):
        msgs = self._msgs(monkeypatch, False)
        joined = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        # 私聊不再额外注入场景块；一对一场景就是骨架原句
        assert "现在是群聊" not in joined
        assert msgs[0]["content"] == core.SYSTEM_PROMPT
        assert "正在 QQ 上与绿冻聊天" in msgs[0]["content"]
        assert "你是在直接与对方聊天" in msgs[0]["content"]

    def test_group_scene_only(self, monkeypatch):
        msgs = self._msgs(monkeypatch, True)
        joined = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        assert "现在是群聊" in joined
        assert msgs[0]["content"].startswith(core.SYSTEM_PROMPT)
        assert "现在是群聊" in msgs[0]["content"]
        assert "也可以顺着话题接一句" in joined, "没点名也可能接一句，文案不能写成“只有点名”"

    def test_skeleton_keeps_one_on_one_scene(self):
        """私聊场景移回骨架：一对一场景句保留在骨架里（2026-10-21 用户定）。"""
        text = core.SYSTEM_PROMPT
        assert "正在 QQ 上与绿冻聊天" in text
        assert "你是在直接与对方聊天" in text
        assert "你就是灰泽满本人" in text
        assert "不是在撰写关于灰泽满的旁白" in text

    def test_self_reference_moved_out_of_skeleton(self):
        """自称方式移出骨架，只保留注入末尾一句弱约束（2026-10-27）。

        实测（60 条真实消息，2.5-lite）：名字自指 28%、第一人称 7%、才不 13%、
        “名字+才” 7%、名字起手 2%。整段“不要让名字当第一个词”的规则会逼出
        “名字+才”的固定否认；这里允许少量“我”，换取更自然的句式。不要改回整段。
        """
        from src.plugins.chatbot import injection as inj
        text = core.SYSTEM_PROMPT
        assert "自称方式固定" not in text
        assert "不要让“灰泽满”当一句话的第一个词" not in text
        assert inj.T("selfref") == "回复时，需要自称就用“灰泽满”或“hzm”；偶尔用一次“我”也没关系。"


class TestAssociationInjection:
    """【关联背景】注入 + 轨迹：证明算出来的东西真的进了提示词。"""

    def _build(self, monkeypatch, trace=None, **kwargs):
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        return core.build_message_list("你看过空中杀人事件吗", [], [], trace=trace, **kwargs)

    def test_header_injected_after_terms(self, monkeypatch):
        msgs = self._build(monkeypatch,
                           association_context="《空中杀人现场》是东野圭吾的作品。",
                           association_entities=["空中杀人现场", "东野圭吾"])
        idx = [i for i, m in enumerate(msgs) if m["content"].startswith("【关联背景】")]
        assert idx, "【关联背景】没注入"
        assert "《空中杀人现场》是东野圭吾的作品。" in msgs[idx[0]]["content"]
        terms_i = [i for i, m in enumerate(msgs) if m["content"].startswith("【灰泽满的世界】")]
        assert not terms_i or terms_i[0] < idx[0], "关联背景应在名词库之后"

    def test_trace_records_entities(self, monkeypatch):
        trace = {}
        self._build(monkeypatch, trace=trace,
                    association_context="《空中杀人现场》是东野圭吾的作品。",
                    association_entities=["空中杀人现场", "东野圭吾"])
        assert "associations" in trace["blocks"]
        assert trace["items"]["associations"] == ["空中杀人现场", "东野圭吾"]

    def test_absent_by_default(self, monkeypatch):
        msgs = self._build(monkeypatch)
        assert not any(m["content"].startswith("【关联背景】") for m in msgs)

    def test_group_address_note_is_system_only(self, monkeypatch):
        """群聊点名情况走系统元信息，用户消息保持干净（模型没有标记可抄）。"""
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list(
            "甲：大疲", [], [], is_group=True,
            user_transcript="甲：大疲",
            group_context="在场成员：甲",
            group_address_note="甲说「大疲」",
        )
        hit = [m for m in msgs if m["content"].startswith("【有人在跟你说话】")]
        assert hit and "甲说「大疲」" in hit[0]["content"]
        assert msgs[-1]["content"] == "甲：大疲"
        assert "[未点名]" not in msgs[-1]["content"]

    def test_no_address_note_when_nobody_addressed(self, monkeypatch):
        """没人点名 → **一个字都不注**（“本批无人点名”本身会被模型复述成元回复）。"""
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list(
            "甲：今天下雨了", [], [], is_group=True,
            user_transcript="甲：今天下雨了",
            group_context="在场成员：甲",
            group_address_note="",
        )
        joined = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        assert "【有人在跟你说话】" not in joined
        assert "本批无人点名" not in joined

    def test_group_thread_note_injected(self, monkeypatch):
        """群聊对话线（谁在对谁说）紧跟群块注入，让模型接在同一条线上。"""
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list(
            "甲：今天下雨了\n乙：真的假的", [], [], is_group=True,
            user_transcript="甲：今天下雨了\n乙：真的假的",
            group_context="在场成员：甲、乙",
            group_thread_note="甲说下雨，乙在追问是不是真的。",
        )
        joined = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        assert "【群聊对话线】甲说下雨，乙在追问是不是真的。" in joined


class TestAssociationWiring:
    """handle_chat 要把 probe 识别出的专名交给关联解析——不然未知作品永远接不上。"""

    async def test_probe_entities_reach_resolver(self, monkeypatch):
        seen = {}

        async def fake_probe(uid, msg, hist, client, debug=False):
            # 返回"纯情绪"扩充句 → gather_retrieval 直接跳过检索（最小 harness，不调外部）
            return ("用户发了个无语的表情", {"entities": ["东野圭吾"]})

        def fake_resolve(text, extra=None):
            seen["text"] = text
            seen["extra"] = extra
            return {"entities": [], "note": "", "preference_ids": []}

        async def fake_reply(messages):
            return "在呢"

        async def _anoop(*a, **k):
            return None

        def _noop(*a, **k):
            return None

        monkeypatch.setenv("MEMORY_V2_INJECT", "0")
        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "resolve_associations", fake_resolve)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", _noop)
        monkeypatch.setattr(core, "update_memory_v2_shadow_task", _anoop)
        monkeypatch.setattr(core, "get_llm_route", lambda: "")
        monkeypatch.setattr(core, "log_corpus_hits", _noop)
        monkeypatch.setattr(core, "log_block_hits", _noop)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        await core.handle_chat("u", "那本小说怎么样")

        assert seen["text"] == "那本小说怎么样"
        assert seen["extra"] == ["东野圭吾"]


class TestRetrievePreferences:
    """偏好已从"向量语义检索"改成"keywords 子串命中"（2026-09-26）。

    原因：噪声地板 0.565 插在正例（0.501/0.552/0.590）**正中间**，余弦分不开；
    而偏好本来就是"类别 + 用户能直接说出的词"，跟 terms 同构 → 确定性子串命中。
    """

    def test_hit_by_keyword(self, monkeypatch):
        from src.plugins.chatbot import retrieval as rt
        entries = [
            {"id": "food", "category": "食物", "text": "爱吃椰子鸡", "keywords": ["爱吃", "火锅"]},
            {"id": "color", "category": "颜色", "text": "喜欢蓝色", "keywords": ["颜色", "蓝色"]},
        ]
        monkeypatch.setattr(rt, "load_preferences", lambda: entries)
        assert [i["id"] for i in rt.retrieve_preferences("你爱吃什么呀")] == ["food"]
        assert [i["id"] for i in rt.retrieve_preferences("你喜欢什么颜色")] == ["color"]

    def test_no_keyword_no_hit(self, monkeypatch):
        from src.plugins.chatbot import retrieval as rt
        entries = [{"id": "food", "category": "食物", "text": "爱吃椰子鸡", "keywords": ["爱吃"]}]
        monkeypatch.setattr(rt, "load_preferences", lambda: entries)
        assert rt.retrieve_preferences("明天几点开会") == []

    def test_english_keyword_is_case_insensitive(self, monkeypatch):
        from src.plugins.chatbot import retrieval as rt
        entries = [{"id": "games", "category": "游戏", "text": "喜欢 Switch", "keywords": ["switch"]}]
        monkeypatch.setattr(rt, "load_preferences", lambda: entries)
        assert [i["id"] for i in rt.retrieve_preferences("你 Switch 玩得多吗")] == ["games"]

    def test_pattern_hit(self, monkeypatch):
        # pattern（正则）命中——与 terms 同一套机制，治"子串做不到"的场景
        from src.plugins.chatbot import retrieval as rt
        entries = [{"id": "food", "category": "食物", "text": "爱吃椰子鸡",
                    "keywords": ["爱吃"], "pattern": r"吃(?![^，。！？]{0,2}(药|瓜))"}]
        monkeypatch.setattr(rt, "load_preferences", lambda: entries)
        assert [i["id"] for i in rt.retrieve_preferences("你吃月饼了吗")] == ["food"]
        assert rt.retrieve_preferences("感冒吃什么药") == []      # 子串没中、pattern 排除
        assert rt.retrieve_preferences("我要吃药了") == []

    def test_broken_pattern_does_not_crash(self, monkeypatch):
        # 写坏的正则不该炸掉整条路（只当没配 pattern）
        from src.plugins.chatbot import retrieval as rt
        entries = [{"id": "x", "category": "x", "text": "t",
                    "keywords": ["命中"], "pattern": "([unclosed"}]
        monkeypatch.setattr(rt, "load_preferences", lambda: entries)
        assert [i["id"] for i in rt.retrieve_preferences("这个词命中")] == ["x"]

    def test_top_n_caps(self, monkeypatch):
        from src.plugins.chatbot import retrieval as rt
        entries = [{"id": f"e{i}", "category": "x", "text": "t", "keywords": ["吃"]} for i in range(5)]
        monkeypatch.setattr(rt, "load_preferences", lambda: entries)
        assert len(rt.retrieve_preferences("吃什么", top_n=2)) == 2


class TestLegendaryMemory:
    async def test_legendary_reply_records_memory(self, monkeypatch):
        captured = []

        def fake_append(uid, user, reply, **kwargs):  # append_user_history 是同步函数
            captured.append((user, reply))

        async def _noop(*a, **k):
            pass

        monkeypatch.setattr(core, "append_user_history", fake_append)

        trigger = next(iter(core.LEGENDARY_REPLIES))  # 取一个真实梗
        await core.handle_chat("u", trigger)
        assert captured, "梗匹配回复也应记入短期记忆"
        assert captured[0][0] == trigger


class TestLegendaryConfirmTemplates:
    """每条 confirm 模板都必须**把消息喂给模型**、并**要求它只答是/否**。

    踩坑（2026-10-11）：8 条里有 5 条既没有 `{msg}`（模型看不到要判的那句话，只能瞎猜）
    也没有「只回复：是 或 否」（而 routing 是按"回答以'是'开头"来判的）→ 这几条**必然判错**。
    用户看到的「到底爱不爱绿冻被判成不是梗」就是这么来的，而当时没有任何测试能抓到。
    """

    def test_every_confirm_feeds_msg_and_asks_yes_no(self):
        for key, tpl in core.LEGENDARY_CONFIRMS.items():
            assert "{msg}" in tpl, f"{key} 的 confirm 没给模型消息（缺 {{msg}}）"
            assert "只回复" in tpl, f"{key} 的 confirm 没要求只答是/否"

    def test_asking_for_context_means_injecting_it(self):
        for key, tpl in core.LEGENDARY_CONFIRMS.items():
            if "结合最近对话" in tpl:
                assert "{context}" in tpl, f"{key} 要求结合最近对话，却没注入 {{context}}"


class TestMemoryStoresCleanedReply:
    """记忆里存的是**清洗后**的她。

    否则她带换行/多括号的原始输出会进记忆，再作为"她自己怎么说话"的 few-shot 喂回去，
    形成自我强化回路——显示端每次洗掉了，源头却一直被喂养。
    """

    async def test_memory_gets_cleaned_reply(self, monkeypatch):
        async def fake_probe(uid, msg, hist, client, debug=False):
            _v = '用户发了个偷笑的表情'
            return (_v, {}) if debug else _v   # 纯情绪 → 跳过检索，最小 harness

        captured = {}

        def fake_append(uid, user, reply, **kwargs):
            captured["reply"] = reply

        async def fake_reply(messages):
            return "没有。\n\n（小声）mua又不是糖，说给就给的（心虚）"

        async def _noop(*a, **k):
            pass

        monkeypatch.setattr(core, "probe_session", fake_probe)
        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "get_user_history", lambda uid: [])
        monkeypatch.setattr(core, "get_last_turn_gap_seconds", lambda uid: None)
        monkeypatch.setattr(core, "append_user_history", fake_append)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")

        await core.handle_chat("u", "可惜🤭")

        assert "\n" not in captured["reply"], "记忆里不该再有换行"
        assert captured["reply"].count("（") <= 1, "记忆里不该再有多个括号"


class TestClassifyBehavior:
    """L3：LLM 判定行为意图 + 措辞组（embedding 猜意图不可靠，改 LLM 理解）。

    2026-09-26：**行为与措辞共用这一次调用**（同一个问题的两半：该怎么做 / 该用哪些词）。
    `classify_behavior` 保留为只取行为名的薄包装，逻辑在 routing.classify_l3。
    """

    def _behaviors(self):
        return [
            {"name": "被夸时把夸奖调小", "trigger": "收到夸奖时", "response": "否认"},
            {"name": "被质疑时心虚辩解", "trigger": "被质问时", "response": "辩解"},
        ]

    def _phrases(self):
        return [
            {"id": "brag_deny", "trigger": "被夸奖、被称赞时", "meaning": "被夸时的否认", "phrases": ["也没有啦"]},
            {"id": "caught_deny", "trigger": "被指出问题、被戳穿时", "meaning": "被戳穿时的否认", "phrases": ["不是不是"]},
        ]

    def _fake(self, content):
        class _FakeCompletions:
            async def create(self, **kwargs):
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
        return type("C", (), {"chat": type("Chat", (), {"completions": _FakeCompletions()})()})

    async def test_returns_valid_behavior_name(self):
        out = await routing.classify_behavior(self._fake('{"behavior": "被夸时把夸奖调小"}'),
                                              "你唱歌好好听", "", self._behaviors())
        assert out == "被夸时把夸奖调小"

    async def test_null_returns_empty(self):
        out = await routing.classify_behavior(self._fake('{"behavior": null}'),
                                              "今天天气不错", "", self._behaviors())
        assert out == ""

    async def test_fabricated_name_rejected(self):
        out = await routing.classify_behavior(self._fake('{"behavior": "不存在的行为"}'),
                                              "随便", "", self._behaviors())
        assert out == ""

    async def test_markdown_wrapped_json_parsed(self):
        out = await routing.classify_behavior(self._fake('```json\n{"behavior": "被质疑时心虚辩解"}\n```'),
                                              "你又在骗人", "", self._behaviors())
        assert out == "被质疑时心虚辩解"

    async def test_failure_returns_empty(self):
        def _raise(*a, **k):
            raise RuntimeError("api down")
        fake = type("C", (), {"chat": type("Chat", (), {"completions": type("C2", (), {"create": _raise})()})()})
        out = await routing.classify_behavior(fake, "你好", "", self._behaviors())
        assert out == ""

    async def test_no_behaviors_returns_empty(self):
        out = await routing.classify_behavior(self._fake('{"behavior": "x"}'), "你好", "", [])
        assert out == ""



class TestRepairLlmJson:
    """DeepSeek 偶发的不规范 JSON 修复。

    `_repair_llm_json` 住在 `routing.py`（L3 分类在用）。原 core 里那条 v1 记忆提取路径
    （`_parse_memory_extract` / `update_memory_task`）已随 V2 转正删掉，所以这里直接测 routing。
    """

    def test_bare_keys(self):
        assert routing._repair_llm_json('{new_impression: "夜猫子"}') == '{"new_impression": "夜猫子"}'

    def test_single_quotes(self):
        assert routing._repair_llm_json("{'a': 'x'}") == '{"a": "x"}'

    def test_trailing_comma(self):
        assert routing._repair_llm_json('{"a": 1,}') == '{"a": 1}'

    def test_markdown_fence(self):
        assert routing._repair_llm_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_leading_garbage(self):
        assert routing._repair_llm_json('好的\n{"a": 1}') == '{"a": 1}'

    def test_empty_unchanged(self):
        assert routing._repair_llm_json("") == ""


class TestMemoryV2ShadowTask:
    async def test_disabled_flag_does_not_initialize_client(self, monkeypatch):
        monkeypatch.setattr(core.memory_v2, "shadow_enabled", lambda: False)

        def fail_clients():
            raise AssertionError("影子开关关闭时不应初始化客户端")

        monkeypatch.setattr(core, "_get_clients", fail_clients)
        result = await core.update_memory_v2_shadow_task(
            "u", "我叫小明", "知道了", "scope-a"
        )
        assert result == []

    async def test_enabled_flag_passes_private_turn_to_extractor(self, monkeypatch):
        monkeypatch.setattr(core.memory_v2, "shadow_enabled", lambda: True)
        fake_client = object()
        monkeypatch.setattr(core, "_get_clients", lambda: (fake_client, None))
        monkeypatch.setattr(core, "_get_model_name", lambda: "test-model")
        captured = {}

        async def fake_extract(**kwargs):
            captured.update(kwargs)
            return [{"kind": "fact"}]

        monkeypatch.setattr(core.memory_v2, "extract_and_ingest", fake_extract)
        result = await core.update_memory_v2_shadow_task(
            "u", "我叫小明", "知道了", "scope-a"
        )

        assert result == [{"kind": "fact"}]
        assert captured["client"] is fake_client
        assert captured["model"] == "test-model"
        assert captured["user_id"] == "u"
        assert captured["session_id"] == "scope-a"

    async def test_pure_emoji_is_skipped(self, monkeypatch):
        monkeypatch.setattr(core.memory_v2, "shadow_enabled", lambda: True)

        def fail_clients():
            raise AssertionError("纯表情不应产生记忆提取调用")

        monkeypatch.setattr(core, "_get_clients", fail_clients)
        assert await core.update_memory_v2_shadow_task("u", "😭", "咋了", "scope-a") == []


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
        msgs = core.build_message_list("", [], [], vision_desc="一碗面")
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "[图片：一碗面]"

    def test_text_plus_image(self, monkeypatch):
        self._now(monkeypatch)
        msgs = core.build_message_list("看看这张", [], [], vision_desc="一碗面")
        assert msgs[-1]["content"] == "看看这张\n[图片：一碗面]"


class TestHandleChatEmotionOnly:
    """纯情绪消息（'可惜🤭'→probe补全成'用户发了个偷笑的表情'）应跳过语义检索，
    防误命中无关样本（曾导致回'喜欢新衣服'）。"""

    async def test_emotion_only_skips_retrieval(self, monkeypatch):
        async def fake_probe(uid, msg, hist, client, debug=False):
            _v = '用户发了个偷笑的表情'
            return (_v, {}) if debug else _v
        monkeypatch.setattr(core, "probe_session", fake_probe)

        def _fail(*a, **k):
            raise AssertionError("纯情绪消息不应触发检索/行为分类")

        monkeypatch.setattr(core, "embed_query", _fail)
        monkeypatch.setattr(core, "retrieve_corpus_candidates", _fail)
        monkeypatch.setattr(core, "classify_l3", _fail)

        captured = {}

        async def fake_reply(messages):
            captured["msgs"] = messages
            return "可惜啥"

        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "append_user_history", lambda *a, **k: None)
        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")

        async def _noop(*a, **k):
            pass

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
        monkeypatch.setattr(core, "retrieve_corpus_candidates", _fail)
        monkeypatch.setattr(core, "classify_l3", _fail)  # L3：图片-only 也不做行为/措辞分类
        monkeypatch.setattr(core, "fuse_and_truncate", _fail)

        captured = {}

        async def fake_reply(messages):
            captured["user"] = messages[-1]["content"]
            return "这碗面看起来好香"

        monkeypatch.setattr(core, "generate_reply", fake_reply)
        monkeypatch.setattr(core, "append_user_history", lambda *a, **k: None)

        async def _noop(*a, **k):
            pass

        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "【当前时间】测试")

        reply = await core.handle_chat("u", "", vision_desc="一碗加煎蛋的面")
        assert embed_called == []                       # 图片-only 不做检索
        assert captured["user"].startswith("[图片：")     # 图片并入用户消息
        assert reply == "这碗面看起来好香"


class TestL3AndEmbedConcurrent:
    """L3 和 embedding 互不依赖，必须**并行**发（省一次串行往返）；结果和先后 await 一样。"""

    async def test_l3_and_embed_overlap(self, monkeypatch):
        started, both = [], asyncio.Event()

        async def _arrive(name):
            started.append(name)
            if len(started) == 2:
                both.set()
            # 串行的话第一个永远等不到第二个 → 超时报错
            await asyncio.wait_for(both.wait(), 1.0)

        async def fake_l3(client, query, history, behaviors):
            await _arrive("l3")
            return {"behavior": "", "same_request": {"what": "", "count": 1}}

        embedded = []

        async def fake_embed(client, text):
            await _arrive("embed")
            embedded.append(text)
            return [0.1, 0.2]

        async def fake_judge(client, query, candidates):
            return []

        monkeypatch.setattr(core, "classify_l3", fake_l3)
        monkeypatch.setattr(core, "embed_query", fake_embed)
        monkeypatch.setattr(core, "retrieve_corpus_candidates", lambda q, v: [])
        monkeypatch.setattr(core, "judge_corpus", fake_judge)

        result = {"fused_items": [], "preference_items": [], "core_stories": [], "same_request": None}
        await core._fill_retrieval(result, "你今天干嘛了", "用户问她今天做了什么", "",
                                   None, None, [], True)
        assert sorted(started) == ["embed", "l3"]
        assert embedded == ["用户问她今天做了什么"], "embedding 用的还是检索句（probe 扩充后的）"

    async def test_non_user_msg_skips_l3(self, monkeypatch):
        def _fail(*a, **k):
            raise AssertionError("主动发言不做 L3")

        async def fake_embed(client, text):
            return [0.1]

        monkeypatch.setattr(core, "classify_l3", _fail)
        monkeypatch.setattr(core, "embed_query", fake_embed)
        result = {"fused_items": [], "preference_items": [], "core_stories": [], "same_request": None}
        await core._fill_retrieval(result, "今天直播好累", "", "", None, None, [], False)
