"""短期记忆时间戳：老数据（裸字符串）兼容 + "距上一轮对话"注入。

背景：她以前分不清"刚刚说的"和"三天前说的"。现在每轮存时间戳，
生成前把间隔作为**一行汇总**注入【最近对话记录】顶部（不是每行都打时间戳，
免得模型去复述"3天前"）。
"""
import json
import time

import pytest

from src.plugins.chatbot import memory


@pytest.fixture
def tmp_memory(tmp_path, monkeypatch):
    """把短期记忆指向临时文件，别碰线上 user_memory/short_term.json。"""
    f = tmp_path / "short_term.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", f)
    return f


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
        raw = json.loads(tmp_memory.read_text(encoding="utf-8"))["u1"]
        assert len(raw) == 2
        assert all(isinstance(it, dict) and "t" in it for it in raw)
        # 对外仍是字符串行——core 的 ln[4:] 剥"灰泽满："前缀依赖这个
        assert memory.get_user_history("u1") == ["用户：在吗", "灰泽满：在呢"]

    def test_append_still_caps_length(self, tmp_memory):
        for i in range(10):
            memory.append_user_history("u1", f"消息{i}", f"回复{i}")
        raw = json.loads(tmp_memory.read_text(encoding="utf-8"))["u1"]
        assert len(raw) == memory.SHORT_MEMORY_LINES
        assert memory.get_user_history("u1")[-1] == "灰泽满：回复9"

    def test_append_tolerates_legacy_string_value(self, tmp_memory):
        # 老式写法：整个 key 存成一个字符串（历史遗留形态）
        _write(tmp_memory, {"u1": "用户：在吗"})
        memory.append_user_history("u1", "还来", "来了")
        assert memory.get_user_history("u1") == ["用户：在吗", "用户：还来", "灰泽满：来了"]


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
