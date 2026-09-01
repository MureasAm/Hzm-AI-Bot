"""读秒窗口（chat_window）攒批/读图/归纳/分批发送逻辑的单元测试。"""
import asyncio

from src.plugins.chatbot import chat_window as cw


class TestCombineText:
    def test_joins_texts_ignores_image_src(self):
        assert cw._combine_text([("在吗", "", ""), ("天气咋样", "", ""), ("", "http://img", "")]) == "在吗\n天气咋样"

    def test_empty(self):
        assert cw._combine_text([]) == ""


class TestFlush:
    async def test_flush_combines_and_splits(self, monkeypatch):
        win = cw._UserWindow("u1")
        win.pending = [("u1", "在吗", "", ""), ("u1", "天气咋样", "", "")]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False):
            captured["text"] = text
            captured["vision"] = vision_desc
            captured["summary"] = batch_summary
            return "灰泽满今天嗓子特别不舒服……明天就想早点下播休息一下"

        async def fake_summarize(msgs):
            return "用户在问候并问天气"

        async def fake_describe(bot, url, file):
            raise AssertionError("无图消息不应调用视觉")

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        sent = []

        async def fake_send(win, content):
            sent.append(content)

        monkeypatch.setattr(cw, "_send", fake_send)

        await cw._flush(win)
        assert captured["text"] == "在吗\n天气咋样"
        assert captured["vision"] == ""
        assert captured["summary"] == "用户在问候并问天气"
        assert sent == ["灰泽满今天嗓子特别不舒服……", "明天就想早点下播休息一下"]  # 拆分 + 去句号

    async def test_flush_reads_image_at_reply_time(self, monkeypatch):
        win = cw._UserWindow("u2")
        win.pending = [("u2", "", "http://img1", "img1.image")]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False):
            captured["text"] = text
            captured["vision"] = vision_desc
            captured["summary"] = batch_summary
            return "这碗面好香"

        async def fake_summarize(msgs):
            return ""

        async def fake_describe(bot, url, file):
            return "一碗加煎蛋的面"  # 视觉在回复前才解析

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        sent = []

        async def fake_send(win, content):
            sent.append(content)

        monkeypatch.setattr(cw, "_send", fake_send)

        await cw._flush(win)
        assert captured["text"] == ""
        assert captured["vision"] == "一碗加煎蛋的面"
        assert sent == ["这碗面好香"]

    async def test_flush_vision_fallback_keeps_reply(self, monkeypatch):
        win = cw._UserWindow("u3")
        win.pending = [("u3", "", "http://img1", "")]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False):
            captured["vision"] = vision_desc
            return "嗯？没看清"

        async def fake_summarize(msgs):
            return ""

        async def fake_describe(bot, url, file):
            return ""  # 视觉失败

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        sent = []

        async def fake_send(win, content):
            sent.append(content)

        monkeypatch.setattr(cw, "_send", fake_send)

        await cw._flush(win)
        assert captured["vision"] == "灰泽满收到一张图片，但暂时没看清里面的内容"
        assert sent == ["嗯？没看清"]  # 兜底保证有回复


class TestVoiceFlush:
    """_flush 语音分支：短回复(should_voice=True) 补语音；长/含内心戏括号只走文字。"""

    async def _run_flush(self, monkeypatch, reply_text):
        win = cw._UserWindow("u4")
        win.pending = [("u4", "在吗", "", "")]
        voice_calls = []

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False):
            return reply_text

        async def fake_summarize(msgs):
            return ""

        async def fake_describe(bot, url, file):
            raise AssertionError("无图消息不应调用视觉")

        async def fake_send_voice(bot, target_id, is_private, reply_text_):
            voice_calls.append((target_id, is_private, reply_text_))

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        monkeypatch.setattr(cw, "send_voice", fake_send_voice)
        sent = []

        async def fake_send(win, content):
            sent.append(content)

        monkeypatch.setattr(cw, "_send", fake_send)

        await cw._flush(win)
        await asyncio.sleep(0)  # 让后台语音任务跑起来
        return sent, voice_calls

    async def test_short_reply_spawns_voice(self, monkeypatch):
        sent, voice_calls = await self._run_flush(monkeypatch, "在呢")
        assert sent == ["在呢"]
        assert voice_calls == [("u4", True, "在呢")]  # 短回复 → 补一条语音条

    async def test_long_reply_no_voice(self, monkeypatch):
        sent, voice_calls = await self._run_flush(monkeypatch, "今天的直播真的特别特别顺利，感觉大家也都玩得很开心呀")
        assert sent  # 文字照发
        assert voice_calls == []  # 长回复只走文字，不双发

    async def test_inner_paren_reply_no_voice(self, monkeypatch):
        sent, voice_calls = await self._run_flush(monkeypatch, "嗯（小声）")
        assert sent == ["嗯（小声）"]
        assert voice_calls == []  # 内心戏括号是文字专属表达，TTS 表达不了


class TestEnqueue:
    async def test_enqueue_accumulates_and_new_task(self):
        cw._windows.clear()
        bot = object()
        cw.enqueue("u1", "u1", "a", "", "", bot, True)
        w = cw._windows["u1"]
        t1 = w.task
        gen1 = w.generation
        cw.enqueue("u1", "u1", "b", "", "", bot, True)
        assert w.pending == [("u1", "a", "", ""), ("u1", "b", "", "")]   # 攒批
        assert w.task is not t1                       # 插话 → 新任务
        assert w.generation == gen1 + 1               # 代际递增
        # 清理：取消并让循环处理；不 await，避免"未启动即取消"的任务抛 CancelledError
        for t in (t1, w.task):
            t.cancel()
        await asyncio.sleep(0)
        cw._windows.clear()

    def test_enqueue_ignores_empty(self):
        cw._windows.clear()
        bot = object()
        cw.enqueue("u1", "u1", "", "", "", bot, True)
        assert "u1" not in cw._windows  # 空消息不入缓冲
