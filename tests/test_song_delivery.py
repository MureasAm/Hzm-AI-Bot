"""唱歌动作从主回复到 QQ 回执、短期对话的集成测试。"""
import asyncio
from types import SimpleNamespace

import pytest
from nonebot.adapters.onebot.v11 import Message

from src.plugins import chatbot
from src.plugins.chatbot import chat_window as cw
from src.plugins.chatbot import core, reply_evidence, session_memory


async def test_song_intent_preserves_ingress_order_when_same_second_quote_is_slow(monkeypatch):
    user = "song-order"
    quote_started = asyncio.Event()
    release_quote = asyncio.Event()
    captured = {}

    class Event:
        message_type = "private"
        time = 100

        def __init__(self, message_id, text):
            self.message_id = message_id
            self.message = Message(text)

        def get_message(self):
            return self.message

        def get_user_id(self):
            return user

    async def slow_quote(event, bot):
        if event.message_id == "first":
            quote_started.set()
            await release_quote.wait()
        return ""

    async def fake_handle(user_id, text, *, authored_text, **kwargs):
        captured.update(text=text, authored_text=authored_text)
        return ""

    async def no_op(*args, **kwargs):
        return ""

    monkeypatch.setattr(chatbot, "_quoted_forward_detail", slow_quote)
    monkeypatch.setattr(cw, "_windows", {})
    monkeypatch.setattr(cw, "_process", no_op)
    monkeypatch.setattr(cw, "summarize_batch", no_op)
    monkeypatch.setattr(cw, "handle_chat", fake_handle)

    first = asyncio.create_task(chatbot._handle_chat(None, Event("first", "给我唱两句")))
    await quote_started.wait()
    try:
        await chatbot._handle_chat(None, Event("second", "算了"))
    finally:
        release_quote.set()
        await first
    win = cw._windows[user]
    await win.task
    await cw._flush(win)

    assert captured == {"text": "算了\n给我唱两句", "authored_text": "给我唱两句\n算了"}


@pytest.mark.parametrize(("mode", "as_file", "reply_text", "label"), [
    ("clip", False, "", "[唱歌片段]《星星》"),
    ("full", False, "给你。", "[唱歌完整版]《星星》"),
    ("full", True, "给你。", "[歌曲文件]《星星》"),
])
async def test_confirmed_song_is_delivered_and_available_to_the_next_turn(
    monkeypatch, tmp_path, mode, as_file, reply_text, label,
):
    sent = []
    user = "song-test"
    delivery = SimpleNamespace(title="星星", mode=mode, path=tmp_path / "song.wav", as_file=as_file)
    first = True

    async def fake_handle(user_id, text, *, authored_text, **kwargs):
        nonlocal first
        if first:
            first = False
            assert text == "给我唱这首\n[引用：别唱]"
            assert authored_text == "给我唱这首"
            reply = core.Reply(reply_text, {"song": delivery, "voice": True})
        else:
            assert session_memory.last_song(user_id) == "星星"
            reply = core.Reply("")
        reply.memory_context = {"user_text": text, "session_id": "song-session"}
        return reply

    async def fake_send(win, content):
        sent.append(content)
        return "text-id"

    async def fake_song(bot, target_id, request):
        assert target_id == user
        assert request is delivery
        sent.append("歌")
        return True

    async def no_op(*a, **kw):
        pass

    def no_tts(*a, **kw):
        pytest.fail("唱歌的同轮文字不走合成语音")

    monkeypatch.setattr(cw, "_windows", {})
    monkeypatch.setattr(cw, "_process", no_op)
    monkeypatch.setattr(cw, "summarize_batch", no_op)
    monkeypatch.setattr(cw, "_maybe_send_sticker", no_op)
    monkeypatch.setattr(cw, "update_memory_v2_shadow_task", no_op)
    monkeypatch.setattr(cw, "handle_chat", fake_handle)
    monkeypatch.setattr(cw, "_send", fake_send)
    monkeypatch.setattr(cw, "send_song", fake_song, raising=False)
    monkeypatch.setattr(cw, "should_voice", no_tts)
    monkeypatch.setattr(cw, "SPLIT_REPLY_ENABLED", False)

    cw.enqueue(user, user, "给我唱这首\n[引用：别唱]", [], None, True, authored_text="给我唱这首")
    win = cw._windows[user]
    await win.task
    await cw._flush(win)

    assert sent == ([reply_text] if reply_text else []) + ["歌"]
    assert cw.get_user_history(user)[-1].endswith(label)
    [turn] = reply_evidence.recall_turns(user, terms=["星星"])
    assert turn["parts"][-1]["modality"] == "song"
    assert turn["parts"][-1]["text"] == label
    assert session_memory.last_song(user) == "星星"

    win.pending = [(user, "接着聊点别的", [])]
    await cw._flush(win)
    assert not session_memory.last_song(user)


@pytest.mark.parametrize("overtaken", [False, True])
async def test_song_failure_or_newer_message_never_leaves_success_receipt(monkeypatch, tmp_path, overtaken):
    user = "song-test"
    win = cw._UserWindow(user)
    win.pending = [(user, "给我唱两句", [])]
    delivery = SimpleNamespace(title="星星", mode="clip", path=tmp_path / "song.wav", as_file=False)
    sent = []

    async def fake_handle(user_id, text, **kwargs):
        if overtaken:
            win.pending.append((user, "算了", []))
            win.generation += 1
        reply = core.Reply("", {"song": delivery})
        reply.memory_context = {"user_text": text, "session_id": "song-session"}
        return reply

    async def fake_song(*args):
        sent.append("尝试发送")
        raise RuntimeError("QQ 拒绝了这个文件")

    async def fake_send(win, content):
        sent.append(content)
        return "text-id"

    async def no_op(*a, **kw):
        pass

    monkeypatch.setattr(cw, "handle_chat", fake_handle)
    monkeypatch.setattr(cw, "summarize_batch", no_op)
    monkeypatch.setattr(cw, "update_memory_v2_shadow_task", no_op)
    monkeypatch.setattr(cw, "_send", fake_send)
    monkeypatch.setattr(cw, "send_song", fake_song, raising=False)

    await cw._flush(win)

    if overtaken:
        assert sent == []
        assert [pending[1] for pending in win.pending] == ["给我唱两句", "算了"]
    else:
        assert len(sent) == 2 and sent[0] == "尝试发送"
        assert "没发出去" in sent[1]
    assert not session_memory.last_song(user)
    assert not reply_evidence.recall_turns(user, terms=["星星"])
