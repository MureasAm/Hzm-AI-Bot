"""置顶动态里的周表图：换图/公告两条触发，同一张图只推一次。

2026-11-09 实测：置顶项 `module_tag.text == "置顶"`，周表是图集第 4 张
（882×1080，`SCHEDULE 11月9日-11月15日`）。换图 = CDN 文件名哈希变，
pub_ts 不变（编辑更新），所以不能靠动态 id/时间判。
"""
from pathlib import Path

import pytest

from src.plugins.chatbot import bili_bridge as bb


class FakeBot:
    async def get_friend_list(self):
        return []

    async def send_private_msg(self, user_id=None, message=None):
        pass


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    async def _none(*a, **k):
        return None
    monkeypatch.setattr(bb, "compose_proactive", _none)
    monkeypatch.setattr(bb, "_ITEMS_CACHE", {})
    monkeypatch.setattr(bb, "_save_state", lambda s: None)


def _make(state):
    m = bb.BiliMonitor()
    m.uid = "1"
    m.state = dict(state)
    return m


def _pinned(hashes, pid="p1"):
    return {"id": pid, "text": "置顶",
            "image_urls": [f"https://i0.hdslb.com/bfs/new_dyn/{h}.png" for h in hashes],
            "pic_hashes": list(hashes)}


def _setup(monkeypatch, pinned, state, download_ok=True):
    m = _make(state)
    pushed = []

    async def fake_push(bot, content, image_paths=None, record=True, voice=False):
        pushed.append((content, list(image_paths or []), voice))

    async def fake_pinned(uid, items=None, use_network=True):
        return pinned

    async def fake_dl(url):
        if not download_ok:
            raise RuntimeError("下载失败")
        return Path("week.png")

    monkeypatch.setattr(m, "_push", fake_push)
    monkeypatch.setattr(bb, "fetch_pinned_dynamic", fake_pinned)
    monkeypatch.setattr(bb, "_download_image", fake_dl)
    return m, pushed


class TestWeekTable:
    async def test_first_seen_records_baseline_without_push(self, monkeypatch):
        m, pushed = _setup(monkeypatch, _pinned(["a", "b", "c", "w1"]), state={})
        handled = await m._check_week_table(FakeBot())
        assert handled is False and pushed == []
        assert m.state["pinned_pic_hashes"] == ["a", "b", "c", "w1"]
        assert m.state["week_pic_index"] == 3

    async def test_changed_schedule_slot_pushes_once(self, monkeypatch):
        state = {"pinned_id": "p1", "pinned_pic_hashes": ["a", "b", "c", "w1"],
                 "week_pic_index": 3}
        m, pushed = _setup(monkeypatch, _pinned(["a", "b", "c", "w2"]), state)
        handled = await m._check_week_table(FakeBot())
        assert handled is True
        assert len(pushed) == 1
        assert pushed[0][1] and pushed[0][1][0].name == "week.png"
        assert m.state["last_pushed_week_pic"] == "w2"
        # 同一张图再轮询：不重复推
        pushed.clear()
        handled = await m._check_week_table(FakeBot())
        assert handled is False and pushed == []

    async def test_unrelated_pinned_image_change_is_ignored(self, monkeypatch):
        """月度大航海礼物/立绘换了，不是周表槽位 → 不推（除非公告同时到）。"""
        state = {"pinned_id": "p1",
                 "pinned_pic_hashes": ["a", "b", "c", "w1", "gift1"],
                 "week_pic_index": 3}
        m, pushed = _setup(
            monkeypatch, _pinned(["a", "b", "c", "w1", "gift2"]), state)
        await m._check_week_table(FakeBot())
        assert pushed == []
        assert m.state["pinned_pic_hashes"][-1] == "gift2"

    async def test_announcement_pushes_current_schedule_even_without_change(self, monkeypatch):
        state = {"pinned_id": "p1", "pinned_pic_hashes": ["a", "b", "c", "w1"],
                 "week_pic_index": 3}
        m, pushed = _setup(monkeypatch, _pinned(["a", "b", "c", "w1"]), state)
        ann = {"id": "new", "text": "你好 本周周表已更新"}
        handled = await m._check_week_table(FakeBot(), announcement=ann)
        assert handled is True and len(pushed) == 1
        assert m.state["last_pushed_week_pic"] == "w1"

    async def test_announcement_after_already_pushed_is_handled_without_duplicate(self, monkeypatch):
        state = {"pinned_id": "p1", "pinned_pic_hashes": ["a", "b", "c", "w1"],
                 "week_pic_index": 3, "last_pushed_week_pic": "w1"}
        m, pushed = _setup(monkeypatch, _pinned(["a", "b", "c", "w1"]), state)
        handled = await m._check_week_table(
            FakeBot(), announcement={"id": "new", "text": "本周周表已更新"})
        assert handled is True and pushed == []

    async def test_slot_shift_after_insert_uses_vision(self, monkeypatch):
        """她在图集前面插了一张图 → 周表槽位后移，视觉重找。"""
        state = {"pinned_id": "p1", "pinned_pic_hashes": ["a", "b", "c", "w1"],
                 "week_pic_index": 3}
        pinned = _pinned(["x", "a", "b", "c", "w2"])
        m, pushed = _setup(monkeypatch, pinned, state)

        async def fake_is_week(url):
            return "w2" in url

        monkeypatch.setattr(bb, "_image_is_week_table", fake_is_week)
        handled = await m._check_week_table(
            FakeBot(), announcement={"id": "new", "text": "本周周表已更新"})
        assert handled is True and len(pushed) == 1
        assert m.state["week_pic_index"] == 4
        assert m.state["last_pushed_week_pic"] == "w2"

    async def test_download_failure_does_not_suppress_announcement(self, monkeypatch):
        state = {"pinned_id": "p1", "pinned_pic_hashes": ["a", "b", "c", "w1"],
                 "week_pic_index": 3}
        m, pushed = _setup(
            monkeypatch, _pinned(["a", "b", "c", "w2"]), state, download_ok=False)
        handled = await m._check_week_table(
            FakeBot(), announcement={"id": "new", "text": "本周周表已更新"})
        assert handled is False and pushed == []

    async def test_dynamic_announcement_is_suppressed_when_week_pic_pushed(self, monkeypatch):
        """公告动态本身不再推文字，只推周表图 + 记水位线。"""
        m = _make({
            "last_dynamic_id": "old",
            "pinned_id": "p1", "pinned_pic_hashes": ["a", "b", "c", "w1"],
            "week_pic_index": 3,
        })
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True, voice=False):
            pushed.append(content)

        async def _dyn():
            return {"id": "new", "text": "你好 本周周表已更新[UPOWER_1_emm]晚上去吃下🍢",
                    "image_urls": [], "is_repost": False, "is_live_notice": False}

        async def fake_pinned(uid, items=None, use_network=True):
            return _pinned(["a", "b", "c", "w2"])

        async def fake_dl(url):
            return Path("week.png")

        monkeypatch.setattr(m, "_push", fake_push)
        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)
        monkeypatch.setattr(bb, "fetch_pinned_dynamic", fake_pinned)
        monkeypatch.setattr(bb, "_download_image", fake_dl)

        await m._check_dynamic(FakeBot())
        assert len(pushed) == 1                    # 只有周表那条
        assert "周表更新了" in pushed[0]
        assert m.state["last_dynamic_id"] == "new"

    async def test_plain_dynamic_still_pushes_normally(self, monkeypatch):
        m = _make({"last_dynamic_id": "old"})

        async def _dyn():
            return {"id": "new", "text": "普通动态", "image_urls": [],
                    "is_repost": False, "is_live_notice": False}

        async def fake_pinned(uid, items=None, use_network=True):
            return None

        monkeypatch.setattr(m, "_fetch_latest_dynamic", _dyn)
        monkeypatch.setattr(bb, "fetch_pinned_dynamic", fake_pinned)
        monkeypatch.setattr(m, "_format_dynamic_push", lambda t: f"通知:{t}")
        pushed = []

        async def fake_push(bot, content, image_paths=None, record=True, voice=False):
            pushed.append(content)

        monkeypatch.setattr(m, "_push", fake_push)
        await m._check_dynamic(FakeBot())
        assert pushed == ["通知:普通动态"]


class TestWeekTableHelpers:
    def test_pic_hash_ignores_host_and_extension(self):
        assert bb._pic_hash("http://i0.hdslb.com/bfs/new_dyn/abc123.png") == "abc123"
        assert bb._pic_hash("https://i1.hdslb.com/bfs/new_dyn/abc123.jpg?x=1") == "abc123"

    def test_announcement_pattern(self):
        assert bb.is_week_table_announcement("你好 本周周表已更新[UPOWER_1_emm]")
        assert bb.is_week_table_announcement("这周的直播安排更新了")
        assert not bb.is_week_table_announcement("今天出去吃饭")
        assert not bb.is_week_table_announcement("周表真好看")

    def test_find_pinned_item_requires_tag(self):
        items = [
            {"id_str": "1", "modules": {"module_tag": None}},
            {"id_str": "2", "modules": {"module_tag": {"text": "置顶"}}},
        ]
        assert bb._find_pinned_item(items)["id_str"] == "2"
        assert bb._find_pinned_item(items[:1]) is None
