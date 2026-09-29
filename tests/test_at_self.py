# -*- coding: utf-8 -*-
"""群里 @ 她本人的解析测试。

**为什么单独测**：@ 是一个**独立消息段**（at），`extract_plain_text()` 只收 text 段——
所以「@灰泽满 在吗」走到后面就只剩"在吗"，群里"被点名必接"的确定性判据命中不了。
踩坑（2026-09-26）：用户报"@了她还是不接话"，根因就在这。

两条要钉住的：
- @ 她 → 消息前插 `AT_SELF_MARK`（接话门靠它确定性放行）
- **只 @ 不带字** → 不能当空消息丢掉（以前直接 return，她永远收不到）
"""
from types import SimpleNamespace

import pytest
from nonebot.adapters.onebot.v11 import Message, MessageSegment

import src.plugins.chatbot as pkg
from src.plugins.chatbot import _extract_at_self
from src.plugins.chatbot.constants import AT_SELF_MARK

BOT_ID = "10000"
GROUP_ID = "555"


def _msg(*segs) -> Message:
    return Message(list(segs))


def _event(msg, self_id=BOT_ID, user_id="20000", to_me=False, message_type="group"):
    return SimpleNamespace(
        message=msg,
        self_id=self_id,
        message_type=message_type,
        group_id=GROUP_ID,
        to_me=to_me,
        get_message=lambda: msg,
        get_user_id=lambda: user_id,
    )


class TestToMeSignal:
    """适配器把"@在开头/结尾"的 at 段**删掉了**，只能靠 `event.to_me`。

    踩坑（2026-09-27）：只加了段检测，"@她"还是不接话。看适配器源码才知道
    `_check_at_me` 遇到首/尾的 @ 会 `event.message.pop(0)` 并置 `to_me=True`——
    `@灰泽满 在吗` 正是最常见的写法，轮到我们处理时 at 段已经不存在了。
    """

    def test_to_me_counts_in_group(self):
        # 模拟适配器处理后的样子：at 段没了，只剩正文，但 to_me=True
        msg = _msg(MessageSegment.text("在吗"))
        assert _extract_at_self(msg, _event(msg, to_me=True)) is True

    def test_to_me_ignored_in_private(self):
        # 私聊的 to_me 是适配器**无条件**置的 True（不代表被 @），别在私聊乱放行
        msg = _msg(MessageSegment.text("在吗"))
        ev = _event(msg, to_me=True, message_type="private")
        assert _extract_at_self(msg, ev, is_group=False) is False

    def test_mid_message_at_still_caught_by_segment(self):
        # @ 在**中间**时适配器不动它 → 段还在，靠段检测
        msg = _msg(MessageSegment.text("你们看"), MessageSegment.at(BOT_ID),
                   MessageSegment.text("这个"))
        assert _extract_at_self(msg, _event(msg)) is True

    def test_plain_group_message_not_addressed(self):
        msg = _msg(MessageSegment.text("你们吃了吗"))
        assert _extract_at_self(msg, _event(msg)) is False


class TestExtractAtSelf:
    def test_at_self_detected(self):
        msg = _msg(MessageSegment.at(BOT_ID), MessageSegment.text(" 在吗"))
        assert _extract_at_self(msg, _event(msg)) is True

    def test_at_other_is_not_her(self):
        # 群里 @ 的是别人 —— 不能把她叫出来
        msg = _msg(MessageSegment.at("30000"), MessageSegment.text(" 你看这个"))
        assert _extract_at_self(msg, _event(msg)) is False

    def test_at_all_is_not_her(self):
        # @全体成员 是 @ 所有人，不是点她
        msg = _msg(MessageSegment.at("all"), MessageSegment.text(" 通知"))
        assert _extract_at_self(msg, _event(msg)) is False

    def test_plain_text_without_at(self):
        msg = _msg(MessageSegment.text("你们吃了吗"))
        assert _extract_at_self(msg, _event(msg)) is False

    def test_no_self_id_is_safe(self):
        # 取不到 bot 自己的 QQ 号时，宁可判"不是"（后面还有名字表兜底）
        msg = _msg(MessageSegment.at(BOT_ID))
        assert _extract_at_self(msg, _event(msg, self_id="")) is False

    def test_multiple_at_segments(self):
        msg = _msg(MessageSegment.at("30000"), MessageSegment.at(BOT_ID))
        assert _extract_at_self(msg, _event(msg)) is True


class TestHandleChatMarks:
    """`_handle_chat` 要把 @ 变成文本里的标记，并让"纯 @"活下来。"""

    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        self.enqueued = []
        monkeypatch.setattr(
            pkg.chat_window, "enqueue",
            lambda *a, **k: self.enqueued.append((a, k)) or None)

    async def _run(self, msg, **kw):
        await pkg._handle_chat(bot=SimpleNamespace(), event=_event(msg, **kw))

    @pytest.mark.asyncio
    async def test_adapter_stripped_at_still_marks(self):
        """最要紧的一条：@ 被适配器删掉后，光靠 to_me 也要能标记上。"""
        await self._run(_msg(MessageSegment.text("今天怎么没播")), to_me=True)
        assert len(self.enqueued) == 1
        text = self.enqueued[0][0][2]
        assert text.startswith(AT_SELF_MARK) and "今天怎么没播" in text

    @pytest.mark.asyncio
    async def test_private_to_me_does_not_mark(self):
        # 私聊不该因为 to_me（恒 True）就带上"有人@了你"的标记
        await self._run(_msg(MessageSegment.text("在吗")), to_me=True,
                        message_type="private")
        assert len(self.enqueued) == 1
        assert AT_SELF_MARK not in self.enqueued[0][0][2]

    @pytest.mark.asyncio
    async def test_at_plus_text_gets_mark(self):
        await self._run(_msg(MessageSegment.at(BOT_ID), MessageSegment.text(" 今天怎么没播")))
        assert len(self.enqueued) == 1
        text = self.enqueued[0][0][2]
        assert text.startswith(AT_SELF_MARK) and "今天怎么没播" in text

    @pytest.mark.asyncio
    async def test_bare_at_is_not_dropped(self):
        # 踩坑：纯 @ 进来时 user_msg 是空串 → 被"空消息"分支 return 掉，她永远收不到
        await self._run(_msg(MessageSegment.at(BOT_ID)))
        assert len(self.enqueued) == 1
        assert self.enqueued[0][0][2] == AT_SELF_MARK

    @pytest.mark.asyncio
    async def test_at_other_groupmate_not_marked(self):
        # @ 的是别人 → 不加标记，照常当普通群聊消息过门
        await self._run(_msg(MessageSegment.at("30000"), MessageSegment.text(" 看这个")))
        assert len(self.enqueued) == 1
        text = self.enqueued[0][0][2]
        assert AT_SELF_MARK not in text
