# -*- coding: utf-8 -*-
"""一条消息多张图的抽取测试。

踩坑（2026-10-18，用户报）：真实聊天里对方"文字+图片一起发"时，
攒批回复只读到一张图。根因是 `_extract_image_source` 只 `return` 第一张——
QQ 的一条消息可以带多个 image 段，其余图片**连源都没进攒批缓冲**，
视觉模型自然永远看不到它们。

这里钉住两件事：
- 一条消息里的**所有** image 段都要被抽出来（不是只取第一张）；
- 上限按 QQ 客户端自己的 9 张收，异常消息不至于打爆视觉调用。
"""
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from src.plugins.chatbot import MAX_IMAGES_PER_MESSAGE, _extract_image_sources


def test_extracts_all_images_in_one_message():
    msg = Message([
        MessageSegment.text("你看这三张"),
        MessageSegment.image("file:///a.jpg"),
        MessageSegment.image("file:///b.jpg"),
        MessageSegment.image("file:///c.jpg"),
    ])
    srcs = _extract_image_sources(msg)
    assert [f for _, f in srcs] == ["file:///a.jpg", "file:///b.jpg", "file:///c.jpg"]


def test_caps_at_qq_image_limit():
    msg = Message([MessageSegment.image(f"file:///{i}.jpg") for i in range(12)])
    assert len(_extract_image_sources(msg)) == MAX_IMAGES_PER_MESSAGE
