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
