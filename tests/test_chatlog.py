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
    """指向临时目录（2026-10-08 起**按会话分文件**，所以给的是目录不是文件）。

    `_migrated` 也要复位：不然同一个进程里第一个用例跑完，后面的用例就不再走迁移分支。
    """
    monkeypatch.setattr(chatlog, "LOG_DIR", tmp_path / "chat_log")
    monkeypatch.setattr(chatlog, "_migrated", False)
    monkeypatch.delenv("CHATLOG", raising=False)
    return tmp_path / "chat_log"


def _read(path: Path) -> list:
    """读记录：给目录就把所有会话的文件合并读（拆分后的常态）。"""
    if path.is_dir():
        return list(chatlog.iter_records())
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


class TestPerSessionFiles:
    """聊天落盘按会话分文件（2026-10-08）：一个人一个文件才"能看"。"""

    def test_each_session_own_file(self, _tmp_log):
        log_turn("111", False, "在吗", "在")
        log_turn("222", False, "hello", "hi")
        assert (_tmp_log / "111.jsonl").exists() and (_tmp_log / "222.jsonl").exists()
        assert not (_tmp_log / "chat.jsonl").exists()

    def test_iter_records_spans_all_sessions(self, _tmp_log):
        log_turn("111", False, "在吗", "在")
        log_turn("222", False, "hello", "hi")
        recs = list(chatlog.iter_records())
        assert len(recs) == 2
        assert {r["session"] for r in recs} == {"111", "222"}
        assert all({"t", "kind", "user", "reply"} <= set(r) for r in recs)

    def test_session_id_is_sanitised(self, _tmp_log):
        log_turn("../evil", False, "x", "y")
        assert chatlog._file_for("../evil").parent == _tmp_log

    def test_legacy_single_file_is_split_once(self, _tmp_log):
        """旧 chat.jsonl 自动按会话拆分，**原件改名存档不删**。"""
        _tmp_log.mkdir(parents=True, exist_ok=True)
        (_tmp_log / "chat.jsonl").write_text(
            json.dumps({"t": 1.0, "session": "111", "kind": "private", "user": "a", "reply": "b"},
                       ensure_ascii=False) + "\n" +
            json.dumps({"t": 2.0, "session": "222", "kind": "group", "user": "c", "reply": "d"},
                       ensure_ascii=False) + "\n", encoding="utf-8")
        assert len(list(chatlog.iter_records())) == 2          # 触发拆分
        assert (_tmp_log / "111.jsonl").exists() and (_tmp_log / "222.jsonl").exists()
        assert not (_tmp_log / "chat.jsonl").exists()
        assert (_tmp_log / "chat.jsonl.migrated").exists()  # 原件留着

    def test_broken_lines_are_skipped(self, _tmp_log):
        _tmp_log.mkdir(parents=True, exist_ok=True)
        (_tmp_log / "111.jsonl").write_text("{坏行\n" + json.dumps({"session": "111"}) + "\n",
                                            encoding="utf-8")
        assert len(list(chatlog.iter_records())) == 1
