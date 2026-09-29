# -*- coding: utf-8 -*-
"""合并转发：取回后的解析测试。

背景：用户转发一条"聊天记录"时，QQ 给的是一个 `forward` 段 + 一个 id；
内容要用 `get_forward_msg` 取回。而**各家 OneBot 实现返回的形状不一样**
（字符串 / 段数组 / Message 对象），所以解析必须防御式——认不出就返回空串，
绝不能把整条消息处理搞崩。

**只注入摘要、不注入原文**：转发几十条几千字，原文会挤爆上下文
（见 core.summarize_forward）。这里测的是取回之后、摘要之前的解析层。
"""
from types import SimpleNamespace

import nonebot  # noqa: F401  （conftest 已 init，这里只为类型导入稳妥）
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from src.plugins.chatbot import _render_forward, _flatten_forward_content


class TestFlatten:
    def test_plain_string(self):
        assert _flatten_forward_content("你好") == "你好"

    def test_segment_array(self):
        segs = [{"type": "text", "data": {"text": "看这个"}},
                {"type": "image", "data": {}},
                {"type": "face", "data": {}}]
        assert _flatten_forward_content(segs) == "看这个[图片][表情]"

    def test_nonebot_message_object(self):
        m = Message([MessageSegment.text("喏"), MessageSegment.image(file="x.png")])
        assert _flatten_forward_content(m) == "喏"

    def test_unknown_shape_is_empty(self):
        assert _flatten_forward_content(None) == ""
        assert _flatten_forward_content(12345) == ""
        assert _flatten_forward_content([{"type": "video", "data": {}}]) == ""


class TestRender:
    def test_standard_shape(self):
        resp = {"message": [
            {"type": "node", "data": {"nickname": "小李", "content": [
                {"type": "text", "data": {"text": "在吗"}}]}},
            {"type": "node", "data": {"nickname": "小王", "content": [
                {"type": "text", "data": {"text": "在"}}]}},
        ]}
        assert _render_forward(resp) == "小李：在吗\n小王：在"

    def test_content_as_plain_string(self):
        resp = {"message": [{"data": {"nickname": "小李", "content": "直接是字符串"}}]}
        assert _render_forward(resp) == "小李：直接是字符串"

    def test_name_field_fallback(self):
        # 有的实现给 name 不给 nickname
        resp = {"message": [{"data": {"name": "小王", "content": "嗨"}}]}
        assert _render_forward(resp) == "小王：嗨"

    def test_bare_list_response(self):
        assert _render_forward([{"nickname": "A", "content": "x"}]) == "A：x"

    def test_missing_nickname_still_keeps_text(self):
        # 没昵称也要留住内容——总比整条丢掉好
        resp = {"message": [{"data": {"content": "无名的发言"}}]}
        assert _render_forward(resp) == "无名的发言"

    def test_empty_or_broken_shapes(self):
        assert _render_forward({}) == ""
        assert _render_forward(None) == ""
        assert _render_forward("给了个字符串") == ""

    def test_image_only_node_is_kept_as_placeholder(self):
        # 只有图的一条也要留住——"小李发了张图"本身就是信息，丢掉反而失真
        resp = {"message": [
            {"data": {"nickname": "小李", "content": [{"type": "image", "data": {}}]}},
            {"data": {"nickname": "小王", "content": "有字"}},
        ]}
        assert _render_forward(resp) == "小李：[图片]\n小王：有字"

    def test_truly_empty_node_is_skipped(self):
        resp = {"message": [
            {"data": {"nickname": "小李", "content": ""}},
            {"data": {"nickname": "小王", "content": "有字"}},
        ]}
        assert _render_forward(resp) == "小王：有字"


# ==================== 转发/卡片"不能静默消失" ====================
# 踩坑（2026-09-29 群 901907410）：豆老湿先发了一条内容（疑似卡片/转发），
# 3 分钟后才问"能不能给灰泽满调成这样"。那条先发的**没进任何一批**——
# 她永远不知道"这样"指什么，只能回一句"调成哪样啊"。

import pytest  # noqa: E402

import src.plugins.chatbot as pkg  # noqa: E402


def _event(msg, self_id="10000", user_id="20000", message_type="group"):
    return SimpleNamespace(
        message=msg, self_id=self_id, message_type=message_type, group_id="901907410",
        raw_message="", to_me=False,
        get_message=lambda: msg, get_user_id=lambda: user_id,
    )


class TestUnrecognizedSegmentsNotDropped:
    """只含"我们没解析的段"的消息（卡片/文件/视频…）不能一声不响地消失。

    这类消息 `extract_plain_text()` 是空的、又没有图/表情，会掉进
    `if not user_msg and not image_url and not image_file: return` —— 整条丢弃，
    **连日志都没有**，事后完全查不出来。
    """

    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        self.enqueued = []
        monkeypatch.setattr(pkg.chat_window, "enqueue",
                            lambda *a, **k: self.enqueued.append((a, k)) or None)

    async def _run(self, msg):
        await pkg._handle_chat(bot=SimpleNamespace(), event=_event(msg))

    @pytest.mark.asyncio
    async def test_json_card_is_not_dropped(self):
        await self._run(Message([MessageSegment("json", {"data": "{}"})]))
        assert len(self.enqueued) == 1
        text = self.enqueued[0][0][2]
        assert "没能解析出来" in text and "卡片" in text

    @pytest.mark.asyncio
    async def test_video_segment_is_not_dropped(self):
        await self._run(Message([MessageSegment("video", {"file": "x.mp4"})]))
        assert len(self.enqueued) == 1
        assert "视频" in self.enqueued[0][0][2]

    @pytest.mark.asyncio
    async def test_plain_text_unaffected(self):
        await self._run(Message([MessageSegment.text("在吗")]))
        assert self.enqueued[0][0][2] == "在吗"

    @pytest.mark.asyncio
    async def test_truly_empty_still_dropped(self):
        # 真空消息（只有空白文字）还是不该回
        await self._run(Message([MessageSegment.text("   ")]))
        assert self.enqueued == []

    @pytest.mark.asyncio
    async def test_known_segments_do_not_trigger_placeholder(self):
        # 有图的消息走正常路径，不该被占位提示顶掉
        await self._run(Message([MessageSegment.image(file="x.png")]))
        assert len(self.enqueued) == 1
        assert "没能解析出来" not in self.enqueued[0][0][2]


class TestQuotedForwardDetail:
    """引用的那条**自己是一条转发卡片**时，把里面的内容取回来。

    转发卡片在 `extract_plain_text()` 里是空的 → 旧代码遇到这种引用 `return ""`，
    转发内容整段消失（"这样""那个"全部失去指向）。
    """

    def _ev(self, fwd_id="fid1"):
        reply_msg = Message([MessageSegment("forward", {"id": fwd_id})])
        reply = SimpleNamespace(message=reply_msg,
                                raw_message=f"[CQ:forward,id={fwd_id}]",
                                sender=SimpleNamespace(user_id="30000", nickname="豆子"))
        return SimpleNamespace(reply=reply, self_id="10000",
                               get_user_id=lambda: "20000", raw_message="")

    async def test_fetches_and_summarizes(self, monkeypatch):
        nodes = {"message": [
            {"nickname": "大肥鱼", "content": "怎么把AI调成那种性格"},
            {"nickname": "豆子", "content": "这也太暴力了"},
        ]}

        async def fake_get(id):
            return nodes
        monkeypatch.setattr(pkg, "_render_forward", lambda resp: "大肥鱼：怎么把AI调成那种性格\n豆子：这也太暴力了")

        async def fake_summarize(raw):
            return "一段关于怎么把 AI 调成那种性格的对话"
        import src.plugins.chatbot.core as core_mod
        monkeypatch.setattr(core_mod, "summarize_forward", fake_summarize)

        bot = SimpleNamespace(get_forward_msg=fake_get)
        detail = await pkg._quoted_forward_detail(self._ev(), bot)
        assert "调成那种性格" in detail and "2 条" in detail

    async def test_fetch_failure_says_so(self, monkeypatch):
        async def boom(fid):
            raise RuntimeError("NapCat 抖了")
        detail = await pkg._quoted_forward_detail(self._ev(), SimpleNamespace(get_forward_msg=boom))
        assert "没取到内容" in detail

    async def test_non_forward_quote_returns_empty(self):
        reply = SimpleNamespace(message=Message([MessageSegment.text("普通引用")]),
                                raw_message="普通引用",
                                sender=SimpleNamespace(user_id="30000", nickname=""))
        ev = SimpleNamespace(reply=reply, self_id="10000",
                             get_user_id=lambda: "20000", raw_message="")
        assert await pkg._quoted_forward_detail(ev, SimpleNamespace()) == ""

    def test_detail_lands_in_quote_label(self):
        # 取回的内容要真的出现在注入文本里，否则等于白取
        reply = SimpleNamespace(message=Message([MessageSegment("forward", {"id": "fid1"})]),
                                raw_message="[CQ:forward,id=fid1]",
                                sender=SimpleNamespace(user_id="30000", nickname="豆子"))
        ev = SimpleNamespace(reply=reply, self_id="10000",
                             get_user_id=lambda: "20000", raw_message="")
        out = pkg._extract_quote_text(ev, "共 8 条：一段关于调 AI 性格的对话")
        assert "转发聊天记录" in out and "调 AI 性格" in out


class TestNestedForwardExpansion:
    """转发里**还套着转发**时，内层内容必须取回来展开。

    实测（2026-09-29）：用户转发的记录里还套着一条记录，而"蓝泽大肥鱼被拉进群后的
    暴力发言"就在里层。旧实现遇到内层 forward 段直接跳过 → 摘要只剩
    "有人在群里发图并接话，提到自己这儿还有'攻击的'"——**真正的内容一个字没进来**。
    """

    def test_nested_ids_from_segment_list(self):
        content = [{"type": "text", "data": {"text": "看这个"}},
                   {"type": "forward", "data": {"id": "inner1"}}]
        assert pkg._nested_forward_ids(content) == ["inner1"]

    def test_nested_ids_from_cq_string(self):
        assert pkg._nested_forward_ids("[CQ:forward,id=abc] 哈哈") == ["abc"]
        assert pkg._nested_forward_ids("普通文字") == []

    def test_flatten_keeps_a_trace_for_forward_segment(self):
        # 即使没展开（展开失败/层数到顶），也不能什么都不留
        out = pkg._flatten_forward_content([{"type": "forward", "data": {"id": "x"}}])
        assert "转发" in out

    async def test_render_deep_expands_inner(self):
        inner = {"message": [{"nickname": "蓝泽大肥鱼",
                             "content": [{"type": "text", "data": {"text": "怎么把AI调成那种性格"}}]}]}
        outer = {"message": [
            {"nickname": "豆子", "content": [{"type": "text", "data": {"text": "看这个"}},
                                            {"type": "forward", "data": {"id": "inner1"}}]},
        ]}

        class _Bot:
            async def get_forward_msg(self, id):
                return inner if id == "inner1" else {}

        out = await pkg._render_forward_deep(_Bot(), outer)
        assert "调成那种性格" in out           # 内层内容进来了
        assert "内层转发" in out               # 而且标明是内层
        assert "豆子" in out

    async def test_depth_cap_leaves_placeholder(self):
        """层数到顶时留占位说明，不能无限递归、也不能留空。"""
        lvl2 = {"message": [{"nickname": "x", "content": [{"type": "forward", "data": {"id": "L2"}}]}]}
        lvl1 = {"message": [{"nickname": "x", "content": [{"type": "forward", "data": {"id": "L1"}}]}]}

        class _Bot:
            async def get_forward_msg(self, id):
                return {"L1": lvl1, "L2": lvl2}[id]

        out = await pkg._render_forward_deep(_Bot(), {"message": [{"nickname": "x", "content": [
            {"type": "forward", "data": {"id": "L1"}}]}]})
        assert "转发" in out      # 有痕迹，不是空

    async def test_inner_fetch_failure_is_visible(self):
        class _Boom:
            async def get_forward_msg(self, id):
                raise RuntimeError("抖了")

        content = [{"type": "forward", "data": {"id": "x"}}]
        out = await pkg._expand_nested_forwards(_Boom(), content)
        assert "没取到内容" in out[0]["data"]["text"]
