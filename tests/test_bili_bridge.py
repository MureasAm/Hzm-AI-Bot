"""bili_bridge 开播/动态去重 + 白名单推送逻辑的单元测试。

所有外部调用（B站接口、好友列表、私聊、状态写入）全部 mock/打桩。
"""
import pytest

from src.plugins.chatbot import bili_bridge as bb


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

        async def fake_push(bot, content):
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

    async def test_already_live_no_push(self, monkeypatch):
        m = _make_monitor(uid="1", state={"last_live_status": True})
        pushed = []

        async def fake_push(bot, content):
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

        async def fake_push(bot, content):
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


class TestDynamic:
    async def test_new_dynamic_pushes_and_records_id(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="sess", state={"last_dynamic_id": "old"})
        pushed = []

        async def fake_push(bot, content, image_path=None):
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
        called = []

        async def fake_fetch():
            called.append(1)
            return {"id": "x", "text": "", "image_urls": []}

        monkeypatch.setattr(m, "_fetch_latest_dynamic", fake_fetch)
        await m._check_dynamic(FakeBot())
        assert called == []  # 未配置 sessdata 不应尝试请求


class TestPrime:
    async def test_first_poll_establishes_baseline_without_push(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="sess", state={})
        pushed = []

        async def fake_push(bot, content):
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
        assert m.state["_primed"] is True
        assert m.state["last_live_status"] is True
        assert m.state["last_dynamic_id"] == "new"

        # 第二次：状态未变，依然不推
        await m.poll_once(bot)
        assert pushed == []

    async def test_after_baseline_new_event_pushes(self, monkeypatch):
        m = _make_monitor(uid="1", sessdata="sess", state={
            "_primed": True, "last_live_status": False, "last_dynamic_id": "old"})
        pushed = []

        async def fake_push(bot, content, image_path=None):
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
        await m._push(bot, "内容", image_path="C:/fake/dyn.jpg")
        assert len(bot.sent) == 1
        assert "CQ:image" in str(bot.sent[0][1])  # 消息含图片段
