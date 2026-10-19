"""读秒窗口（chat_window）攒批/读图/归纳/分批发送逻辑的单元测试。"""
import asyncio

import pytest

from src.plugins.chatbot import chat_window as cw


class TestCombineText:
    def test_joins_texts_ignores_image_src(self):
        assert cw._combine_text([("在吗", []), ("天气咋样", []), ("", [("http://img", "")])]) == "在吗\n天气咋样"

    def test_empty(self):
        assert cw._combine_text([]) == ""


class TestKeepImageDesc:
    """文字+表情包时是否保留视觉描述（2026-10-19）。"""

    def test_question_text_drops_sticker(self):
        assert cw._keep_image_desc("怎么评价", "[表情包]\n一只虫子") is False

    def test_short_name_or_greeting_keeps_sticker(self):
        assert cw._keep_image_desc("云子", "[表情包]\n躺在床上睡觉") is True
        assert cw._keep_image_desc("晚安", "[表情包]\n睡觉") is True

    def test_explicit_image_reference_keeps_sticker(self):
        assert cw._keep_image_desc("这个表情包是什么", "[表情包]\n一只虫子") is True

    def test_empty_text_keeps_sticker(self):
        assert cw._keep_image_desc("", "[表情包]\n一只虫子") is True

    def test_screenshot_always_kept(self):
        assert cw._keep_image_desc("怎么评价", "[截图]\n小说封面") is True


class TestFlush:
    async def test_flush_combines_and_splits(self, monkeypatch):
        win = cw._UserWindow("u1")
        win.pending = [("u1", "在吗", []), ("u1", "天气咋样", [])]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False, user_transcript=""):
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

        async def fake_send_voice(bot, target_id, is_private, reply_text):
            return False  # 本测试只验文字拆分路径：语音一律失败回退文字

        monkeypatch.setattr(cw, "_send", fake_send)
        monkeypatch.setattr(cw, "send_voice", fake_send_voice)

        await cw._flush(win)
        assert captured["text"] == "在吗\n天气咋样"
        assert captured["vision"] == ""
        assert captured["summary"] == "用户在问候并问天气"
        assert sent == ["灰泽满今天嗓子特别不舒服……", "明天就想早点下播休息一下"]  # 拆分 + 去句号

    async def test_flush_reads_image_at_reply_time(self, monkeypatch):
        win = cw._UserWindow("u2")
        win.pending = [("u2", "", [("http://img1", "img1.image")])]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False, user_transcript=""):
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

    async def test_flush_reads_all_images_of_one_message(self, monkeypatch):
        """一条消息带多张图：每张都要解析，且归纳层也拿到完整描述。"""
        win = cw._UserWindow("u10")
        win.pending = [("u10", "你看", [("http://a", "fa"), ("http://b", "fb")])]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False, user_transcript=""):
            captured["vision"] = vision_desc
            return "好"

        async def fake_summarize(msgs):
            captured["summary_msgs"] = msgs
            return ""

        async def fake_describe(bot, url, file):
            return {"http://a": "一张猫", "http://b": "一张狗"}[url]

        async def fake_send(win_, content):
            pass

        async def fake_send_voice(*args, **kwargs):
            return False

        async def fake_pick(client, reply, avoid_groups=None, **kwargs):
            return None

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        monkeypatch.setattr(cw, "_send", fake_send)
        monkeypatch.setattr(cw, "send_voice", fake_send_voice)
        monkeypatch.setattr(cw, "pick_sticker", fake_pick)
        monkeypatch.setattr(cw, "_sticker_state", {})

        await cw._flush(win)
        assert captured["vision"] == "一张猫；一张狗"
        assert captured["summary_msgs"] == [("你看", "一张猫；一张狗")]

    async def test_sticker_description_dropped_when_text_has_topic(self, monkeypatch):
        """文字+表情包：表情包只是反应，描述不进当前批次（2807992783 的翻车点）。"""
        win = cw._UserWindow("u11")
        win.pending = [("u11", "怎么评价", [("http://img", "f")])]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False,
                              user_transcript=""):
            captured.update(text=text, vision=vision_desc, transcript=user_transcript)
            return "好"

        async def fake_describe(bot, url, file):
            return "[表情包]\n一只绿色虫子在地上蠕动"

        async def fake_summarize(msgs):
            return ""

        async def fake_send(win_, content):
            pass

        async def fake_send_voice(*a, **k):
            return False

        async def fake_pick(client, reply, avoid_groups=None, **kwargs):
            return None

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_send", fake_send)
        monkeypatch.setattr(cw, "send_voice", fake_send_voice)
        monkeypatch.setattr(cw, "pick_sticker", fake_pick)
        monkeypatch.setattr(cw, "_sticker_state", {})

        await cw._flush(win)
        assert captured["vision"] == ""
        assert captured["transcript"] == "怎么评价"

    async def test_screenshot_description_kept(self, monkeypatch):
        """截图可能就是话题，不能因为带了文字就丢。"""
        win = cw._UserWindow("u12")
        win.pending = [("u12", "怎么评价", [("http://img", "f")])]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False,
                              user_transcript=""):
            captured.update(vision=vision_desc, transcript=user_transcript)
            return "好"

        async def fake_describe(bot, url, file):
            return "[截图]\n小说《空中杀人事件》的封面和简介"

        async def fake_summarize(msgs):
            return ""

        async def fake_send(win_, content):
            pass

        async def fake_send_voice(*a, **k):
            return False

        async def fake_pick(client, reply, avoid_groups=None, **kwargs):
            return None

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_send", fake_send)
        monkeypatch.setattr(cw, "send_voice", fake_send_voice)
        monkeypatch.setattr(cw, "pick_sticker", fake_pick)
        monkeypatch.setattr(cw, "_sticker_state", {})

        await cw._flush(win)
        assert "小说《空中杀人事件》" in captured["vision"]
        assert "[图片：[截图]" in captured["transcript"]

    async def test_flush_vision_fallback_keeps_reply(self, monkeypatch):
        win = cw._UserWindow("u3")
        win.pending = [("u3", "", [("http://img1", "")])]
        captured = {}

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False, user_transcript=""):
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
    """_flush 语音优先：成句回复朗读语音、成功即不发文字；短敷衍/含内心戏括号只走文字。"""

    SENTENCE = "灰泽满今天直播聊得特别开心，下次有空我们再一起好好玩呀，明天也要来哦"  # 34 字成句

    async def _run_flush(self, monkeypatch, reply_text, voice_ok=True):
        win = cw._UserWindow("u4")
        win.pending = [("u4", "在吗", [])]
        voice_calls = []

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False, user_transcript=""):
            return reply_text

        async def fake_summarize(msgs):
            return ""

        async def fake_describe(bot, url, file):
            raise AssertionError("无图消息不应调用视觉")

        async def fake_send_voice(bot, target_id, is_private, reply_text_):
            voice_calls.append((target_id, is_private, reply_text_))
            return voice_ok  # 模拟合成/发送成败

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "_describe_image_src", fake_describe)
        monkeypatch.setattr(cw, "send_voice", fake_send_voice)

        # 语音成功之后还会试着配一张表情包（2026-10-05 新行为）→ 这里必须打桩，
        # 否则会真调 LLM 判定（网络），整个测试挂住。
        self.pick_calls = []

        async def fake_pick(client, reply, avoid_groups=None, **kwargs):
            self.pick_calls.append(reply)
            return None
        monkeypatch.setattr(cw, "pick_sticker", fake_pick)
        monkeypatch.setattr(cw, "_sticker_state", {})   # 冷却状态隔离，别跨用例串味
        sent = []

        async def fake_send(win, content):
            sent.append(content)

        monkeypatch.setattr(cw, "_send", fake_send)

        await cw._flush(win)
        return sent, voice_calls

    async def test_sentence_voice_replaces_text(self, monkeypatch):
        sent, voice_calls = await self._run_flush(monkeypatch, self.SENTENCE, voice_ok=True)
        assert voice_calls == [("u4", True, self.SENTENCE)]  # 成句 → 尝试语音
        assert sent == []  # 语音成功，不再刷文字（互斥不双发）

    async def test_voice_failure_falls_back_to_text(self, monkeypatch):
        sent, voice_calls = await self._run_flush(monkeypatch, self.SENTENCE, voice_ok=False)
        assert voice_calls == [("u4", True, self.SENTENCE)]  # 尝试过语音
        assert sent  # 失败回退文字分段，绝不影响收到回复

    async def test_short_ack_text_only(self, monkeypatch):
        sent, voice_calls = await self._run_flush(monkeypatch, "在呢")
        assert sent == ["在呢"]
        assert voice_calls == []  # 短敷衍词 <20 字 → 打字

    async def test_inner_paren_sentence_no_voice(self, monkeypatch):
        sent, voice_calls = await self._run_flush(monkeypatch, "今天直播聊得特别开心（小声）下次再一起")
        assert voice_calls == []  # 内心戏括号是文字专属表达，TTS 表达不了
        assert sent

    async def test_voice_path_also_tries_sticker(self, monkeypatch):
        """语音那条路原来直接 return，表情包永远是语音轮的盲区（用户 2026-10-05）。"""
        sent, voice_calls = await self._run_flush(monkeypatch, self.SENTENCE, voice_ok=True)
        assert voice_calls and sent == []       # 语音成功、没刷文字
        assert self.pick_calls == [self.SENTENCE]   # 但照样试着配了一张


class TestVoiceSurvivesInterrupt:
    """插话取消不该让"已经合成好的语音"卡在中间发不出去。

    实测症状：TTS 转出来了、消息没发出去，且**时有时无**。原因是 send_voice 里
    await _synthesize() 要几秒（GPU），用户在这几秒插话 → enqueue 的 win.task.cancel()
    打断在 await 上的它 → wav 已生成、没发出，且文字兜底也不会走
    （CancelledError 是 BaseException，except Exception 抓不到）。
    """

    SENTENCE = TestVoiceFlush.SENTENCE

    async def _setup(self, monkeypatch, supersede_during_handle=False):
        win = cw._UserWindow("u9")
        win.pending = [("u9", "在吗", [])]
        sent, voice_done = [], []

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False, user_transcript=""):
            if supersede_during_handle:
                win.generation += 1   # 模拟"她还在组织语言时，用户又发了一条"
            return self.SENTENCE

        async def fake_summarize(msgs):
            return ""

        started = asyncio.Event()

        async def slow_send_voice(bot, target_id, is_private, reply_text_):
            started.set()
            await asyncio.sleep(0.05)          # 模拟 TTS 合成的耗时
            voice_done.append(reply_text_)
            return True

        async def fake_send(win_, content):
            sent.append(content)

        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "should_voice", lambda t, force=False: True)

        async def fake_pick(client, reply, avoid_groups=None, **kwargs):
            return None
        monkeypatch.setattr(cw, "pick_sticker", fake_pick)
        monkeypatch.setattr(cw, "_sticker_state", {})
        monkeypatch.setattr(cw, "send_voice", slow_send_voice)
        monkeypatch.setattr(cw, "_send", fake_send)
        return win, sent, voice_done, started

    async def test_cancel_during_synthesis_still_delivers_voice(self, monkeypatch):
        win, sent, voice_done, started = await self._setup(monkeypatch)
        task = asyncio.create_task(cw._flush(win))
        await started.wait()          # 合成已开始
        task.cancel()                 # 用户此时插话
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.15)     # 让 shield 的内层跑完
        assert voice_done, "语音已经合成好了，插话取消不该让它发不出去（用户会什么都收不到）"

    async def test_superseded_before_synthesis_skips_voice(self, monkeypatch):
        """起跑后已被新消息取代 → 不占 GPU 合成（合成完也送不出去）。"""
        win, sent, voice_done, started = await self._setup(
            monkeypatch, supersede_during_handle=True)
        await cw._flush(win)
        assert voice_done == [], "已被取代就不该再去合成"
        assert sent == [], "这轮回复已作废，别再把旧内容刷出去"


class TestEnqueue:
    async def test_enqueue_accumulates_and_new_task(self):
        cw._windows.clear()
        bot = object()
        cw.enqueue("u1", "u1", "a", [], bot, True)
        w = cw._windows["u1"]
        t1 = w.task
        gen1 = w.generation
        cw.enqueue("u1", "u1", "b", [], bot, True)
        assert w.pending == [("u1", "a", []), ("u1", "b", [])]   # 攒批
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
        cw.enqueue("u1", "u1", "", [], bot, True)
        assert "u1" not in cw._windows  # 空消息不入缓冲
