"""bili_bridge 开播/动态去重 + 白名单推送逻辑的单元测试。

所有外部调用（B站接口、好友列表、私聊、状态写入）全部 mock/打桩。
"""
import asyncio

import pytest

from src.plugins.chatbot import _bridge_common as _bc
from src.plugins.chatbot import bili_bridge as bb


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """动态推送现在会先试着转成"她会主动说的那句话"（要调 LLM）。

    **单测一律让它返回 None**（走回退的通知模板）——否则每个用例都会真的请求一次，
    又慢又贵又不稳，而且测的就不是推送逻辑了。要测主动发言那条路，用例自己再打桩。
    """
    async def _none(*a, **k):
        return None
    monkeypatch.setattr(bb, "compose_proactive", _none)


class FakeBot:
    """模拟 OneBot Bot：记录私聊发送，返回固定好友列表。"""

    def __init__(self, friends=("111", "222", "333")):
        self.friends = [{"user_id": f} for f in friends]
        self.sent = []

    async def get_friend_list(self):
        return self.friends

    async def send_private_msg(self, user_id=None, message=None):
        self.sent.append((user_id, message))


def _make_monitor(**attrs):
    m = bb.BiliMonitor()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class TestLiveTransition:
    async def test_offline_to_live_pushes_once(self, monkeypatch):
        m = _make_monitor(uid="1298779265", state={"last_live_status": False})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)

        async def _live():
            return {"live_status": 1, "title": "测试直播"}

        monkeypatch.setattr(m, "_fetch_live_status", _live)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        bot = FakeBot()
        await m._check_live(bot)
        assert len(pushed) == 1
        assert m.state["last_live_status"] is True

        # 仍在直播 → 不重复推
        await m._check_live(bot)
        assert len(pushed) == 1

    async def test_offline_to_live_with_cover_downloads(self, monkeypatch):
        """开播且带封面时，下载封面并传给 _push（失败不阻塞）。"""
        m = _make_monitor(uid="1298779265", state={"last_live_status": False})
        pushed = []
        download_called = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append((content, image_paths))

        async def fake_download(url):
            download_called.append(url)
            from pathlib import Path
            return Path("fake_cover.jpg")

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(bb, "_download_image", fake_download)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        async def _live():
            return {"live_status": 1, "title": "测试直播", "cover": "https://x.com/c.jpg"}

        monkeypatch.setattr(m, "_fetch_live_status", _live)

        await m._check_live(FakeBot())
        assert download_called == ["https://x.com/c.jpg"]  # 封面被下载
        assert pushed and pushed[0][1] is not None  # image_path 传给了 _push

    async def test_offline_to_live_cover_download_failure_ok(self, monkeypatch):
        """封面下载失败不阻塞，仍发文字推送。"""
        m = _make_monitor(uid="1298779265", state={"last_live_status": False})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append((content, image_paths))

        async def fake_download(url):
            raise RuntimeError("网络错误")

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(bb, "_download_image", fake_download)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        async def _live():
            return {"live_status": 1, "title": "测试直播", "cover": "https://x.com/c.jpg"}

        monkeypatch.setattr(m, "_fetch_live_status", _live)

        await m._check_live(FakeBot())
        assert len(pushed) == 1
        assert pushed[0][1] == []  # 下载失败 → 无图，仍推文字

    async def test_already_live_no_push(self, monkeypatch):
        m = _make_monitor(uid="1", state={"last_live_status": True})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)

        async def _live():
            return {"live_status": 1, "title": ""}

        monkeypatch.setattr(m, "_fetch_live_status", _live)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        await m._check_live(FakeBot())
        assert pushed == []

    async def test_stay_offline_no_push(self, monkeypatch):
        m = _make_monitor(uid="1", state={"last_live_status": False})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)

        async def _live():
            return {"live_status": 0, "title": ""}

        monkeypatch.setattr(m, "_fetch_live_status", _live)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        await m._check_live(FakeBot())
        assert pushed == []


class TestExtractDynamic:
    def test_opus_text_via_summary(self):
        item = {"modules": {"module_dynamic": {
            "major": {"type": "MAJOR_TYPE_OPUS",
                      "opus": {"summary": {"text": "现在已经是煮饭高手了"}}}}}}
        assert bb._extract_dynamic_text(item) == "现在已经是煮饭高手了"

    def test_opus_images_via_pics(self):
        item = {"modules": {"module_dynamic": {
            "major": {"type": "MAJOR_TYPE_OPUS",
                      "opus": {"pics": [{"url": "http://a.jpg"}, {"url": "http://b.jpg"}]}}}}}
        assert bb._extract_dynamic_images(item) == ["http://a.jpg", "http://b.jpg"]

    def test_desc_text_legacy(self):
        item = {"modules": {"module_dynamic": {"desc": {"text": "唱拉了 关上门悄悄听"}}}}
        assert bb._extract_dynamic_text(item) == "唱拉了 关上门悄悄听"

    def test_archive_pic(self):
        item = {"modules": {"module_dynamic": {
            "major": {"type": "MAJOR_TYPE_ARCHIVE", "archive": {"pic": "http://v.jpg"}}}}}
        assert bb._extract_dynamic_images(item) == ["http://v.jpg"]


class TestLiveMessage:
    def test_format_with_room(self):
        msg = bb._live_open_message("测试直播", 1775719573)
        assert msg == (
            "灰泽满宣布开播！\n\n"
            "今天的内容是：测试直播\n\n"
            "https://live.bilibili.com/1775719573"
        )

    def test_format_fallback_url(self):
        msg = bb._live_open_message("测试直播")
        assert "https://live.bilibili.com" in msg
        assert "今天的内容是：测试直播" in msg


class TestProactiveDynamic:
    """动态推**两条**：先原文通知，再"她会主动说的那句话"。

    锁的是"原文不许被主动发言顶掉"——曾经写的是 `said or notice` 二选一，
    主动发言一成功，**原文就整条消失**（粉丝只剩一条没头没尾的私聊）。
    """

    def _setup(self, monkeypatch, said):
        m = _make_monitor(uid="1", sessdata="sess", state={"last_dynamic_id": "old"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(m, "_format_dynamic_push", lambda text: f"通知:{text}")

        async def _dyn():
            return {"id": "new", "text": "正文", "image_urls": []}

        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        async def _compose(*a, **k):
            return said
        monkeypatch.setattr(bb, "compose_proactive", _compose)
        return m, pushed

    async def test_notice_first_then_her_sentence(self, monkeypatch):
        # 两条都要有，且**原文在前**（先"转发给你"，再补一句）
        m, pushed = self._setup(monkeypatch, "灰泽满今天收拾到半夜，累死了。")
        await m._check_dynamic(FakeBot())
        assert pushed == ["通知:正文", "灰泽满今天收拾到半夜，累死了。"]

    async def test_falls_back_to_notice_when_not_suitable(self, monkeypatch):
        # 不适合主动说（纯转发/抽奖）→ 只发原文通知，**信息不能丢**
        m, pushed = self._setup(monkeypatch, None)
        await m._check_dynamic(FakeBot())
        assert pushed == ["通知:正文"]

    async def test_images_go_with_the_notice_not_her_sentence(self, monkeypatch):
        """配图只跟着原文那条走——她那句是纯文字，否则同一批图会发两遍。"""
        m, _ = self._setup(monkeypatch, "明天来玩啊")
        calls = []

        async def fake_push(bot, content, image_paths=None, record=True):
            calls.append((content, image_paths))

        async def fake_download(url):
            from pathlib import Path
            return Path("fake_pic.jpg")

        async def _dyn():
            return {"id": "new", "text": "正文", "image_urls": ["https://x.com/a.jpg"]}

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(bb, "_download_image", fake_download)
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)

        await m._check_dynamic(FakeBot())
        assert len(calls) == 2
        assert calls[0][0] == "通知:正文" and calls[0][1]        # 原文那条带图
        assert calls[1][0] == "明天来玩啊" and not calls[1][1]    # 她那句不带图


class TestDynamic:
    async def test_new_dynamic_pushes_and_records_id(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="sess", state={"last_dynamic_id": "old"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(m, "_format_dynamic_push", lambda text: f"转述:{text}")

        async def _dyn():
            return {"id": "new", "text": "正文", "image_urls": []}

        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        bot = FakeBot()
        await m._check_dynamic(bot)
        assert pushed == ["转述:正文"]
        assert m.state["last_dynamic_id"] == "new"

        # 同一 id → 不再推
        await m._check_dynamic(bot)
        assert len(pushed) == 1

    async def test_no_sessdata_skips(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="", state={})
        monkeypatch.setattr(bb, "get_bili_sessdata", lambda: "")  # 未配置 → 跳过
        called = []

        async def fake_fetch():
            called.append(1)
            return {"id": "x", "text": "", "image_urls": []}

        monkeypatch.setattr(m, "_fetch_latest_dynamic", fake_fetch)
        await m._check_dynamic(FakeBot())
        assert called == []  # 未配置 sessdata 不应尝试请求


class TestDynamicRetraction:
    """撤回/置顶导致的"最新退回旧动态"不该重推（水位线判据）。

    踩坑：状态里只记一条 id、判据是"和上一条不同"。她撤回最新动态后，接口的
    "最新"退回上一条（更旧、id 更小），与记的不同 → 被判成新动态 → 把昨天的又推一遍。
    """

    def _monitor(self, monkeypatch, dyn_id, state):
        m = _make_monitor(uid="1298779265", state=state, _primed=True, _warned_no_sessdata=True)
        monkeypatch.setattr(bb, "get_bili_sessdata", lambda: "SESSDATA=x")
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        monkeypatch.setattr(bb, "_download_image", lambda u: None)
        monkeypatch.setattr(bb, "_resize_emote_if_large", lambda p, n: p)

        async def fake_fetch():
            return {"id": dyn_id, "text": "晚安🌙", "image_urls": [], "emote_urls": [],
                    "local_emotes": []}

        monkeypatch.setattr(m, "_fetch_latest_dynamic", fake_fetch)
        return m

    async def test_newer_dynamic_pushes(self, monkeypatch):
        m = self._monitor(monkeypatch, "1246839524441456643", {"last_dynamic_id": "1246839524441456000"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m._check_dynamic(FakeBot())
        assert len(pushed) == 1
        assert m.state["last_dynamic_id"] == "1246839524441456643"

    async def test_retracted_newest_does_not_repush(self, monkeypatch):
        # 撤回最新那条 → 接口"最新"退回更旧的 → 不该重推
        m = self._monitor(monkeypatch, "1246311690117578769", {"last_dynamic_id": "1246839524441456643"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m._check_dynamic(FakeBot())
        assert pushed == [], "撤回后回退到的旧动态不该重推"
        # 水位线不能被调低
        assert m.state["last_dynamic_id"] == "1246839524441456643"

    async def test_same_id_does_not_push(self, monkeypatch):
        m = self._monitor(monkeypatch, "1246839524441456643", {"last_dynamic_id": "1246839524441456643"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m._check_dynamic(FakeBot())
        assert pushed == []


class TestIsNewerId:
    """水位线判据本身（bili/weibo 共用，见 _bridge_common.is_newer_id）。"""

    def test_larger_is_newer(self):
        assert _bc.is_newer_id("1246839524441456643", "1246311690117578769") is True

    def test_smaller_is_not_newer(self):
        assert _bc.is_newer_id("1246311690117578769", "1246839524441456643") is False

    def test_equal_is_not_newer(self):
        assert _bc.is_newer_id("123", "123") is False

    def test_no_previous_is_newer(self):
        assert _bc.is_newer_id("123", "") is True

    def test_empty_new_is_not_newer(self):
        assert _bc.is_newer_id("", "123") is False

    def test_different_length_compares_numerically(self):
        # 位数不同也要按数值比，不能按字符串比（'999' > '1000' 是错的）
        assert _bc.is_newer_id("1000", "999") is True
        assert _bc.is_newer_id("999", "1000") is False

    def test_non_numeric_falls_back_to_inequality(self):
        # 异常数据（非数字 id）退回老行为：不同即新
        assert _bc.is_newer_id("abc", "def") is True
        assert _bc.is_newer_id("abc", "abc") is False


class TestPrime:
    async def test_first_poll_establishes_baseline_without_push(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="sess", state={})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        async def _live():
            return {"live_status": 1, "title": "直播中"}

        async def _dyn():
            return {"id": "new", "text": "正文", "image_urls": []}

        saved = {}
        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(m, "_fetch_live_status", _live)
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)
        monkeypatch.setattr(bb, "_save_state", lambda s: saved.update(s))

        bot = FakeBot()
        await m.poll_once(bot)
        # 首次只建基线，不推送
        assert pushed == []
        assert m._primed is True  # 基线是实例标志
        assert m.state["last_live_status"] is True
        assert m.state["last_dynamic_id"] == "new"

        # 第二次：状态未变，依然不推
        await m.poll_once(bot)
        assert pushed == []

    async def test_restart_does_not_push_downtime_events(self, monkeypatch):
        """重启后（_primed 实例标志重置为 False）：停机期间已开播/发动态不推送。

        回归：此前 _primed 持久化在 state 文件，重启后不重新建基线，
        导致停机期间的开播/动态被当新事件推送。
        """
        m = _make_monitor(uid="1", sessdata="sess", state={
            "last_live_status": False, "last_dynamic_id": "old"})
        # 模拟重启：实例 _primed 默认 False
        assert m._primed is False
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        # 停机期间灰泽满开播了 + 发了新动态（都比 state 里的旧记录新）
        async def _live():
            return {"live_status": 1, "title": "直播中"}

        async def _dyn():
            return {"id": "new", "text": "停机期间发的", "image_urls": []}

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(m, "_fetch_live_status", _live)
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        await m.poll_once(FakeBot())
        # 重启后第一次 poll 只对齐基线，不推送停机期间的事件
        assert pushed == []
        assert m.state["last_live_status"] is True  # 基线已对齐为直播中
        assert m.state["last_dynamic_id"] == "new"

    async def test_after_baseline_new_event_pushes(self, monkeypatch):
        # _primed 是实例标志（进程内）；已建基线后，新动态才推送
        m = _make_monitor(uid="1", sessdata="sess", _primed=True, state={
            "last_live_status": False, "last_dynamic_id": "old"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)

        async def _live():
            return {"live_status": 0, "title": ""}

        async def _dyn():
            return {"id": "new", "text": "新动态", "image_urls": []}

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(m, "_fetch_live_status", _live)
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)
        monkeypatch.setattr(m, "_format_dynamic_push", lambda text: f"转述:{text}")
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        bot = FakeBot()
        await m.poll_once(bot)
        assert pushed == ["转述:新动态"]  # 新动态触发推送


    async def test_prime_does_not_lower_watermark(self, monkeypatch):
        """停机期间她撤回/置顶，基线也不能被调低（水位线只升不降）。"""
        m = _make_monitor(uid="1", state={"last_dynamic_id": "1246839524441456643"})
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        async def fake_dyn():
            return {"id": "1246311690117578769", "text": "旧的"}

        async def fake_live():
            return {"live_status": 0, "title": "", "room_id": 0, "cover": ""}

        monkeypatch.setattr(m, "_fetch_latest_dynamic", fake_dyn)
        monkeypatch.setattr(m, "_fetch_live_status", fake_live)
        await m._prime()
        assert m.state["last_dynamic_id"] == "1246839524441456643"

    async def test_prime_raises_watermark_to_newest(self, monkeypatch):
        """停机期间发了新动态 → 基线对齐到它（保持"不补推停机期间事件"的语义）。"""
        m = _make_monitor(uid="1", state={"last_dynamic_id": "1246311690117578769"})
        monkeypatch.setattr(bb, "_save_state", lambda s: None)

        async def fake_dyn():
            return {"id": "1246839524441456643", "text": "新的"}

        async def fake_live():
            return {"live_status": 0, "title": "", "room_id": 0, "cover": ""}

        monkeypatch.setattr(m, "_fetch_latest_dynamic", fake_dyn)
        monkeypatch.setattr(m, "_fetch_live_status", fake_live)
        await m._prime()
        assert m.state["last_dynamic_id"] == "1246839524441456643"


class TestPush:
    async def test_whitelist_filters(self, monkeypatch):
        m = _make_monitor(state={})
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: ["111", "333"])
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        bot = FakeBot()
        await m._push(bot, "内容")
        sent_ids = [uid for uid, _ in bot.sent]
        assert sorted(sent_ids) == ["111", "333"]

    async def test_push_all_when_no_whitelist(self, monkeypatch):
        m = _make_monitor(state={})
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: [])
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        bot = FakeBot()
        await m._push(bot, "内容")
        assert len(bot.sent) == 3

    async def test_push_uses_fresh_friends(self, monkeypatch):
        # 旧缓存应被忽略：每次推送都拉最新好友，确保新加好友能收到
        m = _make_monitor(state={"friends": [{"user_id": "111"}],
                                 "friends_ts": 9999999999})
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: [])
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        bot = FakeBot(friends=("111", "222"))
        await m._push(bot, "内容")
        assert [uid for uid, _ in bot.sent] == ["111", "222"]

    async def test_push_with_image_attaches_segment(self, monkeypatch):
        m = _make_monitor(state={})
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: ["111"])
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        bot = FakeBot()
        await m._push(bot, "内容", image_paths=["C:/fake/dyn.jpg"])
        assert len(bot.sent) == 1
        assert "CQ:image" in str(bot.sent[0][1])  # 消息含图片段


class TestPushRecordsIntoMemory:
    """推出去的消息要写进**那个好友**的短期记忆。

    2026-09-29 用户反馈：主动发出去的消息是机器人自己用 API 发的，不会作为事件回来，
    从没进过记忆 → 粉丝顺着它回她（"这个好玩吗"）时，她不知道对方在说什么。
    """

    @pytest.fixture(autouse=True)
    def _spy(self, monkeypatch):
        self.recorded = []
        monkeypatch.setattr(bb, "append_bot_message",
                            lambda uid, text: self.recorded.append((uid, text)))

    async def test_success_records_per_friend(self, monkeypatch):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: [])
        await m._push(FakeBot(friends=("111", "222")),
                      "灰泽满刚刚发了动态哦！\n\n动态内容：写完了\n\nhttps://space.bilibili.com/x/dynamic")
        # 记的是**整理过的人话**（剥掉播报腔和链接），不是原样的通知模板
        assert self.recorded == [("111", "（刚发了条动态：写完了）"),
                                 ("222", "（刚发了条动态：写完了）")]

    async def test_failed_send_not_recorded(self, monkeypatch):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: [])

        class FlakyBot(FakeBot):
            async def send_private_msg(self, user_id=None, message=None):
                raise RuntimeError("发不出去")
        await m._push(FlakyBot(friends=("111",)), "内容")
        assert self.recorded == []      # 没发出去就别写进记忆

    async def test_whitelist_still_records_only_targets(self, monkeypatch):
        m = _make_monitor(uid="1", state={})
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: ["222"])
        await m._push(FakeBot(friends=("111", "222")), "内容")
        assert self.recorded == [("222", "内容")]


class TestToMemoryLine:
    """通知模板是播报腔，写进记忆前要整理成人话。

    照原样记会变成「灰泽满：灰泽满刚刚发了动态哦！」——名字两遍、像旁白，
    而且短期记忆会作为 few-shot 喂回去，播报句会被她学走。
    """

    def test_dynamic_notice_becomes_plain(self):
        out = _bc.to_memory_line("灰泽满刚刚发了动态哦！\n\n动态内容：写完了…终于写完了…\n\n"
                                 "https://space.bilibili.com/1298779265/dynamic")
        assert out == "（刚发了条动态：写完了…终于写完了…）"
        assert "http" not in out and "灰泽满刚刚" not in out

    def test_weibo_notice(self):
        out = _bc.to_memory_line("灰泽满刚刚发了微博哦！\n\n微博内容：今天好累\n\nhttps://weibo.com/1/2")
        assert out == "（刚发了条微博：今天好累）"

    def test_live_notice(self):
        out = _bc.to_memory_line("灰泽满宣布开播！\n\n今天的内容是：游玩下BanG Dream\n\n"
                                 "https://live.bilibili.com/1713546334")
        assert out == "（刚宣布开播：游玩下BanG Dream）"

    def test_her_own_sentence_kept_verbatim(self):
        assert _bc.to_memory_line("终于写完了，人快没了…晚上来打游戏") == "终于写完了，人快没了…晚上来打游戏"

    def test_empty_body_still_readable(self):
        assert _bc.to_memory_line("灰泽满刚刚发了动态哦！\n\n动态内容：（图片动态）") == "（刚发了条动态：（图片动态））"


class TestNoticeOnlyRecordedWhenNoProactive:
    """通知只在"她那句没发出去"时写进记忆（用户 2026-09-29 的建议）。

    她那句本来就覆盖了同一件事，通知只是机器复述——再记一条是白占窗口。
    但她那句没发出去时（判官拦住/生成失败/PROACTIVE=0），粉丝回的就是通知本身，
    那时必须留痕。
    """

    @pytest.fixture(autouse=True)
    def _spy(self, monkeypatch):
        self.recorded = []
        monkeypatch.setattr(bb, "append_bot_message",
                            lambda uid, text: self.recorded.append((uid, text)))
        monkeypatch.setattr(bb, "get_notify_whitelist", lambda: [])

    def _dyn(self, monkeypatch, said):
        m = _make_monitor(uid="1", sessdata="sess", state={"last_dynamic_id": "old"})

        async def _d():
            return {"id": "new", "text": "正文", "image_urls": []}
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _d)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        monkeypatch.setattr(m, "_format_dynamic_push", lambda text: f"灰泽满刚刚发了动态哦！\n\n动态内容：{text}")

        async def _compose(*a, **k):
            return said
        monkeypatch.setattr(bb, "compose_proactive", _compose)
        return m

    async def test_only_her_sentence_recorded(self, monkeypatch):
        m = self._dyn(monkeypatch, "明天来玩啊")
        await m._check_dynamic(FakeBot(friends=("111",)))
        assert self.recorded == [("111", "明天来玩啊")]      # 通知没记

    async def test_notice_recorded_when_no_sentence(self, monkeypatch):
        m = self._dyn(monkeypatch, None)
        await m._check_dynamic(FakeBot(friends=("111",)))
        assert self.recorded == [("111", "（刚发了条动态：正文）")]   # 通知兜底留痕

    async def test_live_notice_always_recorded(self, monkeypatch):
        # 开播没有"她那句"，通知就是唯一留痕
        m = _make_monitor(uid="1", state={"last_live_status": False})

        async def _live():
            return {"live_status": 1, "title": "测试直播", "room_id": 1}
        monkeypatch.setattr(m, "_fetch_live_status", _live)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        await m._check_live(FakeBot(friends=("111",)))
        assert self.recorded and self.recorded[0][0] == "111"
        assert "开播" in self.recorded[0][1]


class TestRepostDynamicSkipped:
    """转发别人的动态：**整条不推**（用户 2026-09-29 决定）。

    原因：转一条视频时原动态的标题/封面在接口层取不全，正文只剩她自己那半句
    （实测：`_extract_dynamic_text` 给 '看这个'），粉丝看不懂；
    转出来的主动发言也容易张冠李戴。
    """

    def test_is_repost_flag(self, monkeypatch):
        from types import SimpleNamespace
        m = _make_monitor(uid="1", sessdata="sess", state={})
        monkeypatch.setattr(bb, "_get_emote_map", lambda: _async({}))

        async def _page(*a, **k):
            return {"items": [{"id_str": "9", "orig": {"id_str": "1"},
                               "modules": {"module_dynamic": {"desc": {"text": "看这个"}}}}]}
        import bilibili_api.dynamic as _dyn
        monkeypatch.setattr(_dyn, "get_dynamic_page_info", _page)
        import bilibili_api.utils.network as _net
        monkeypatch.setattr(_net, "Credential", lambda **k: object())
        dyn = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            m._fetch_latest_dynamic())
        assert dyn["is_repost"] is True
        assert dyn["text"] == "看这个"

    async def test_repost_not_pushed_but_watermark_advances(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="sess", state={"last_dynamic_id": "old"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)
        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        monkeypatch.setattr(m, "_format_dynamic_push", lambda t: f"通知:{t}")

        async def _dyn():
            return {"id": "new", "text": "看这个", "image_urls": [], "is_repost": True}
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)

        await m._check_dynamic(FakeBot())
        assert pushed == []                              # 一条都不发
        assert m.state["last_dynamic_id"] == "new"       # 但水位线要前进，否则每轮重判

    async def test_normal_dynamic_still_pushed(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="sess", state={"last_dynamic_id": "old"})
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True):
            pushed.append(content)
        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(bb, "_save_state", lambda s: None)
        monkeypatch.setattr(m, "_format_dynamic_push", lambda t: f"通知:{t}")

        async def _dyn():
            return {"id": "new", "text": "正文", "image_urls": [], "is_repost": False}
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)

        await m._check_dynamic(FakeBot())
        assert pushed == ["通知:正文"]


async def _async(v):
    return v
