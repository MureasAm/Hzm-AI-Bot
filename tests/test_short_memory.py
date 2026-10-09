"""短期记忆时间戳：老数据（裸字符串）兼容 + **每条消息带「多久前」**。

背景：她以前分不清"刚刚说的"和"三天前说的"。
2026-10-05 改：不再是"一行汇总拼在历史顶部"，而是**每行自己带 `[3小时前] `**
——因为旧做法下 gap=10秒 和 gap=3小时 注入出来长得一样，她分不出刚说完还是隔了很久。
"""
import json
import time

import pytest

from src.plugins.chatbot import memory


@pytest.fixture
def tmp_memory(tmp_path, monkeypatch):
    """把短期记忆指向临时目录，别碰线上 user_memory/。

    ⚠️ **两个都要改**（MEMORY_DIR + MEMORY_FILE）：只改 DIR 的话，
    `_ensure_migrated()` 会去动真实那份 short_term.json（拆完还改名）——测试会毁线上记忆。
    `_migrated` 也要复位，否则同一个进程里第一个用例跑完就再不迁移了。
    """
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path / "short_term")
    legacy = tmp_path / "short_term.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", legacy)
    monkeypatch.setattr(memory, "_migrated", False)
    return legacy


def _raw(uid):
    """读某个会话的落盘内容（拆分后：一个会话一个文件，内容就是行列表）。"""
    return json.loads(memory._file_for(uid).read_text(encoding="utf-8"))


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestHistoryBackCompat:
    """线上 short_term.json 已经有一批裸字符串老数据，读取端必须两种都认。"""

    def test_legacy_plain_strings_readable(self, tmp_memory):
        _write(tmp_memory, {"u1": ["用户：在吗", "灰泽满：在呢"]})
        assert memory.get_user_history("u1") == ["用户：在吗", "灰泽满：在呢"]

    def test_new_timestamped_entries_readable_as_text(self, tmp_memory):
        _write(tmp_memory, {"u1": [{"t": 1.0, "text": "用户：在吗"}]})
        assert memory.get_user_history("u1") == ["用户：在吗"]

    def test_unknown_user_is_empty(self, tmp_memory):
        assert memory.get_user_history("nobody") == []

    def test_append_writes_timestamps(self, tmp_memory):
        memory.append_user_history("u1", "在吗", "在呢")
        raw = _raw("u1")
        assert len(raw) == 2
        assert all(isinstance(it, dict) and "t" in it for it in raw)
        # 对外仍是字符串行——core 的 ln[4:] 剥"灰泽满："前缀依赖这个
        assert memory.get_user_history("u1") == ["用户：在吗", "灰泽满：在呢"]

    def test_append_still_caps_length(self, tmp_memory):
        for i in range(10):
            memory.append_user_history("u1", f"消息{i}", f"回复{i}")
        raw = _raw("u1")
        assert len(raw) == memory.SHORT_MEMORY_LINES
        assert memory.get_user_history("u1")[-1] == "灰泽满：回复9"

    def test_append_tolerates_legacy_string_value(self, tmp_memory):
        # 老式写法：整个 key 存成一个字符串（历史遗留形态）
        _write(tmp_memory, {"u1": "用户：在吗"})
        memory.append_user_history("u1", "还来", "来了")
        assert memory.get_user_history("u1") == ["用户：在吗", "用户：还来", "灰泽满：来了"]

    def test_empty_reply_records_only_her_side(self, tmp_memory):
        """群聊接话门判定"不接"时走这条：她看到了、只是没说。

        只记"群里说了什么"，**不记一条空的"灰泽满："**；而且要留得下
        ——下一轮判定和生成都得看得到上下文，否则她会开始"装作知道上文"。
        """
        memory.append_user_history("g1", "小李：我服了\n小王：哈哈", "")
        assert memory.get_user_history("g1") == ["用户：小李：我服了\n小王：哈哈"]

    def test_silent_batch_still_caps_length(self, tmp_memory):
        for i in range(12):
            memory.append_user_history("g1", f"群消息{i}", "")
        raw = _raw("g1")
        assert len(raw) == memory.SHORT_MEMORY_LINES


class TestTimedHistory:
    """`get_user_history_timed`：每行带「距上一条多久」，没时间戳的老数据不加前缀。"""

    def test_each_line_gets_gap_prefix(self, tmp_memory):
        now = time.time()
        t0 = now - 5 * 3600           # 5 小时前
        _write(tmp_memory, {"u": [
            {"t": t0, "text": "用户：在吗"},
            {"t": t0 + 600, "text": "灰泽满：在呢"},        # 距上一条 10 分钟
            {"t": t0 + 600 + 1800, "text": "用户：那个事你还记得吗"},   # 距上一条 30 分钟
        ]})
        lines = memory.get_user_history_timed("u")
        assert len(lines) == 3
        # 每行的时间是**相对上一条**的间隔；第一行相对"现在"。
        assert lines[0].startswith("[5小时前] ")
        assert lines[1].startswith("[10分钟前] ")
        assert lines[2].startswith("[30分钟前] ")
        assert lines[0].endswith("用户：在吗")

    def test_legacy_plain_strings_get_no_prefix(self, tmp_memory):
        _write(tmp_memory, {"u": ["用户：在吗", "灰泽满：在呢"]})
        lines = memory.get_user_history_timed("u")
        assert lines == ["用户：在吗", "灰泽满：在呢"]

    def test_short_gap_gets_no_prefix(self, tmp_memory):
        """同一场对话内（<90秒）不加前缀——否则每行都挂个 [刚刚]，纯噪音。"""
        now = time.time()
        _write(tmp_memory, {"u": [
            {"t": now - 20, "text": "用户：在吗"},
            {"t": now - 10, "text": "灰泽满：在呢"},
        ]})
        lines = memory.get_user_history_timed("u")
        assert lines == ["用户：在吗", "灰泽满：在呢"]


class TestLastTurnGap:
    def test_no_history_is_none(self, tmp_memory):
        assert memory.get_last_turn_gap_seconds("nobody") is None

    def test_legacy_entries_have_no_timestamp(self, tmp_memory):
        # 老数据没时间戳 → 这轮先不注入；等她再说一句就补上了
        _write(tmp_memory, {"u1": ["用户：在吗"]})
        assert memory.get_last_turn_gap_seconds("u1") is None

    def test_gap_measured_from_last_entry(self, tmp_memory):
        _write(tmp_memory, {"u1": [
            {"t": time.time() - 10 * 86400, "text": "用户：旧"},
            {"t": time.time() - 3 * 86400, "text": "灰泽满：新"},
        ]})
        assert memory.get_last_turn_gap_seconds("u1") == pytest.approx(3 * 86400, abs=10)

    def test_malformed_timestamp_is_none(self, tmp_memory):
        _write(tmp_memory, {"u1": [{"t": "不是数字", "text": "用户：在吗"}]})
        assert memory.get_last_turn_gap_seconds("u1") is None


class TestHumanizeGap:
    def test_same_conversation_is_silent(self):
        assert memory.humanize_gap(30) == ""
        assert memory.humanize_gap(89) == ""

    @pytest.mark.parametrize("sec,expect", [
        (90, "1分钟"),
        (600, "10分钟"),
        (3600, "1小时"),
        (7200, "2小时"),
        (86400, "1天"),
        (3 * 86400, "3天"),
        (29 * 86400, "29天"),
        (30 * 86400, "1个月"),
        (90 * 86400, "3个月"),
    ])
    def test_buckets(self, sec, expect):
        assert memory.humanize_gap(sec) == expect

    def test_bad_input_is_silent(self):
        assert memory.humanize_gap(None) == ""
        assert memory.humanize_gap("abc") == ""


class TestAppendBotMessage:
    """主动发言/推送写进记忆——**只记她那一句，没有对应的用户消息**。

    2026-09-29 用户反馈：主动发出去的消息是机器人自己用 API 发的，不会作为事件回来，
    所以从没进过记忆；粉丝顺着那句回她时，她的【最近对话记录】里没有那一句，
    完全不知道对方在说什么。
    """

    def test_records_only_her_line(self, tmp_memory):
        memory.append_bot_message("u1", "明天晚上八点来玩音游啊")
        lines = memory.get_user_history("u1")
        assert lines == ["灰泽满：明天晚上八点来玩音游啊"]
        assert not any(ln.startswith("用户：") for ln in lines)

    def test_has_timestamp(self, tmp_memory):
        memory.append_bot_message("u1", "晚安")
        raw = _raw("u1")
        assert "t" in raw[0] and raw[0]["text"] == "灰泽满：晚安"

    def test_appends_after_existing_history(self, tmp_memory):
        memory.append_user_history("u1", "在吗", "在")
        memory.append_bot_message("u1", "刚发了个动态，来看看")
        assert memory.get_user_history("u1") == ["用户：在吗", "灰泽满：在", "灰泽满：刚发了个动态，来看看"]

    def test_still_caps_length(self, tmp_memory):
        for i in range(20):
            memory.append_bot_message("u1", f"第{i}条")
        assert len(memory.get_user_history("u1")) == memory.SHORT_MEMORY_LINES

    def test_empty_text_writes_nothing(self, tmp_memory):
        memory.append_bot_message("u1", "   ")
        assert memory.get_user_history("u1") == []


class TestPerSessionFiles:
    """按会话拆文件（2026-10-08）：一次读写只碰一个会话，删某人的历史＝删一个文件。"""

    def test_each_session_gets_its_own_file(self, tmp_memory):
        memory.append_user_history("111", "在吗", "在")
        memory.append_user_history("222", "hello", "hi")
        assert memory._file_for("111") != memory._file_for("222")
        assert memory._file_for("111").exists() and memory._file_for("222").exists()
        assert memory.get_user_history("111") == ["用户：在吗", "灰泽满：在"]
        assert memory.get_user_history("222") == ["用户：hello", "灰泽满：hi"]

    def test_writing_one_session_does_not_touch_another(self, tmp_memory):
        memory.append_user_history("111", "第一次", "好")
        before = memory._file_for("222").stat().st_mtime if memory._file_for("222").exists() else None
        memory.append_user_history("111", "第二次", "嗯")
        after = memory._file_for("222").stat().st_mtime if memory._file_for("222").exists() else None
        assert before == after          # 别人的文件根本没被碰

    def test_user_id_is_sanitised(self, tmp_memory):
        # id 只可能是 QQ 号/群号，仍然过滤一遍防路径穿越
        f = memory._file_for("../evil")
        assert f.parent == memory.MEMORY_DIR

    def test_legacy_single_file_is_split_once(self, tmp_memory):
        """老单文件自动拆分，原件改名留下（**不删数据**）。"""
        _write(tmp_memory, {"u1": [{"t": 1.0, "text": "用户：在吗"}], "u2": ["用户：喂"]})
        assert memory.get_user_history("u1") == ["用户：在吗"]     # 先触发迁移
        assert memory._file_for("u1").exists() and memory._file_for("u2").exists()
        assert not tmp_memory.exists()                            # 原件已改名
        assert tmp_memory.with_suffix(".json.migrated").exists()

    def test_broken_legacy_file_does_not_break_chat(self, tmp_memory):
        tmp_memory.write_text("{坏掉的 json", encoding="utf-8")
        memory.append_user_history("u1", "在吗", "在")
        assert memory.get_user_history("u1") == ["用户：在吗", "灰泽满：在"]
