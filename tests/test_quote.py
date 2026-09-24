# -*- coding: utf-8 -*-
"""引用回复（QQ 引用某条消息再回）的解析测试。

背景：之前 `extract_plain_text()` 会把 `reply` 段整个丢掉，于是"用户引用了哪条"对模型
完全不可见。适配器其实已经做了取回（`Bot.handle_event` → `_check_reply` → `get_msg`
→ `event.reply`），我们只是没用。这里测的就是"把它用起来"的那层。

关键是**要带上"是谁说的"**：引用她自己的话 vs 引用另一个群友的话，她的反应完全不同。
"""
from types import SimpleNamespace

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from src.plugins.chatbot import _extract_quote_text, QUOTE_TEXT_MAX

BOT_ID = "10000"


def _event(reply, user_id="20000"):
    return SimpleNamespace(reply=reply, self_id=BOT_ID, get_user_id=lambda: user_id)


def _reply(text="", *, sender_id="10000", nickname="", image=False, face=None):
    segs = []
    if image:
        segs.append(MessageSegment.image(file="x.png"))
    if face is not None:
        segs.append(MessageSegment("face", {"id": str(face)}))
    if text:
        segs.append(MessageSegment.text(text))
    msg = Message(segs)
    sender = SimpleNamespace(user_id=sender_id, nickname=nickname)
    return SimpleNamespace(message=msg, raw_message=text, sender=sender)


class TestNoQuote:
    def test_no_reply_returns_empty(self):
        assert _extract_quote_text(_event(None)) == ""

    def test_empty_quoted_message_returns_empty(self):
        # 引用了一条空消息：没什么可指路的
        assert _extract_quote_text(_event(_reply(""))) == ""


class TestWhoSaidIt:
    def test_quoting_her_own_words(self):
        # 引用的是她自己说过的话 —— 最常见的情况（用户指着她上一句问"这什么意思"）
        out = _extract_quote_text(_event(_reply("此八点非彼八点", sender_id=BOT_ID)))
        assert "引用灰泽满自己说的" in out
        assert "此八点非彼八点" in out

    def test_quoting_the_user_own_words(self):
        out = _extract_quote_text(_event(_reply("我昨天说的", sender_id="20000")))
        assert "引用对方自己说的" in out

    def test_quoting_a_third_party(self):
        # 群聊里引用另一个群友的话 —— 必须和"引用她自己"区分开
        out = _extract_quote_text(
            _event(_reply("群友说的", sender_id="30000", nickname="小明"))
        )
        assert "引用另一个绿冻小明说的" in out

    def test_third_party_without_nickname(self):
        out = _extract_quote_text(_event(_reply("群友说的", sender_id="30000")))
        assert "引用另一个绿冻说的" in out


class TestTruncation:
    def test_long_quote_is_truncated(self):
        out = _extract_quote_text(_event(_reply("字" * 300)))
        assert "…" in out
        # 引用是"指路"不是搬运，正文不该超上限
        assert out.count("字") == QUOTE_TEXT_MAX

    def test_short_quote_not_truncated(self):
        out = _extract_quote_text(_event(_reply("短")))
        assert "…" not in out


class TestQuotedMedia:
    def test_image_marker(self):
        out = _extract_quote_text(_event(_reply("看这个", image=True)))
        assert "（带图）" in out

    def test_face_marker_uses_qq_face_table(self):
        # QQ 内置表情：含义就是名字，不该让模型猜
        out = _extract_quote_text(_event(_reply("你好", face=6)))
        assert "表情" in out

    def test_pure_image_quote_is_not_lost(self):
        # 引用一条纯图片：没有文字，但引用不能整个丢掉
        out = _extract_quote_text(_event(_reply("", image=True)))
        assert out.startswith("[引用")
        assert "发的" in out and "图" in out

    def test_pure_face_quote_is_not_lost(self):
        out = _extract_quote_text(_event(_reply("", face=6)))
        assert out.startswith("[引用")
        assert "表情" in out


class TestRobustness:
    def test_broken_reply_does_not_raise(self):
        # 适配器拿不到被引用消息时 event.reply 可能是残缺对象，别让整条消息处理崩掉
        broken = SimpleNamespace(message=SimpleNamespace(extract_plain_text=lambda: ""),
                                 raw_message=None,
                                 sender=SimpleNamespace(user_id=None, nickname=None))
        assert isinstance(_extract_quote_text(_event(broken)), str)

    def test_missing_sender_does_not_raise(self):
        broken = SimpleNamespace(message=Message([MessageSegment.text("x")]),
                                 raw_message="x", sender=None)
        assert isinstance(_extract_quote_text(_event(broken)), str)

    def test_event_without_reply_attr(self):
        assert _extract_quote_text(SimpleNamespace(self_id=BOT_ID)) == ""
