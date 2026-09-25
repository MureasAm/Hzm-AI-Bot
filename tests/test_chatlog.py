# -*- coding: utf-8 -*-
"""聊天落盘的测试。

这是**分析用的旁路**——所以它唯一的硬要求是：
1. 记的东西对（用户说了什么 / 她回了什么 / 私聊还是群）
2. **写失败绝不能影响主流程**（聊天不能被一个日志搞挂）
3. 关得掉，且**真实对话不落进公开仓库**（.gitignore 那道另有测试盯 `.gitignore` 本身，
   这里测开关与健壮性）
"""
import json
from pathlib import Path

import pytest

from src.plugins.chatbot import chatlog
from src.plugins.chatbot.chatlog import chatlog_enabled, log_turn


@pytest.fixture(autouse=True)
def _tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(chatlog, "LOG_DIR", tmp_path / "chat_log")
    monkeypatch.delenv("CHATLOG", raising=False)
    return tmp_path / "chat_log" / "chat.jsonl"


def _read(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class TestToggle:
    def test_enabled_by_default(self):
        assert chatlog_enabled() is True

    def test_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("CHATLOG", "0")
        assert chatlog_enabled() is False

    def test_disabled_writes_nothing(self, monkeypatch, _tmp_log):
        monkeypatch.setenv("CHATLOG", "0")
        log_turn("u1", False, "在吗", "在")
        assert not _tmp_log.exists()


class TestRecord:
    def test_records_private_turn(self, _tmp_log):
        log_turn("u1", False, "在吗", "在呢")
        rec = _read(_tmp_log)
        assert len(rec) == 1
        assert rec[0]["session"] == "u1"
        assert rec[0]["kind"] == "private"
        assert rec[0]["user"] == "在吗"
        assert rec[0]["reply"] == "在呢"
        assert isinstance(rec[0]["t"], float)

    def test_records_group_turn(self, _tmp_log):
        log_turn("123456", True, "小李：在吗\n小王：在", "在在在")
        assert _read(_tmp_log)[0]["kind"] == "group"

    def test_appends_not_overwrites(self, _tmp_log):
        log_turn("u1", False, "第一句", "回一")
        log_turn("u1", False, "第二句", "回二")
        assert [r["user"] for r in _read(_tmp_log)] == ["第一句", "第二句"]

    def test_strips_whitespace(self, _tmp_log):
        log_turn("u1", False, "  在吗  ", "  在呢  ")
        r = _read(_tmp_log)[0]
        assert r["user"] == "在吗" and r["reply"] == "在呢"


class TestRobustness:
    def test_write_failure_never_raises(self, monkeypatch, tmp_path):
        """分析用的旁路，**绝不能因为它写不进去而影响聊天**。"""
        bad = tmp_path / "not_a_dir"
        bad.write_text("我是文件不是目录", encoding="utf-8")
        monkeypatch.setattr(chatlog, "LOG_DIR", bad / "chat_log")
        log_turn("u1", False, "在吗", "在")   # 不该抛

    def test_rotation_does_not_crash(self, monkeypatch, _tmp_log):
        monkeypatch.setattr(chatlog, "MAX_BYTES", 1)   # 逼它每次都轮转
        for i in range(3):
            log_turn("u1", False, f"第{i}句", f"回{i}")
        assert _tmp_log.exists()   # 轮转后新文件仍在写
