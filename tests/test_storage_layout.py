# -*- coding: utf-8 -*-
"""四个存储的落盘布局：**新用户一定会自动建文件**（用户 2026-10-09 特别问过）。

2026-10-09 起四层都是"一个会话一个文件"：
    短期记忆 user_memory/short_term/<id>.json
    会话记忆 user_memory/session/<id>.json
    长期记忆 user_memory/long_term/<id>.json
    聊天落盘 data/chat_log/<id>.jsonl

这条最容易出错的不是"拆"，而是"**新用户第一次来时目录/文件建不建得出来**"
（新建父目录漏了 mkdir 就会抛异常，而异常点在聊天主链路上）。
所以这里从一个**空的临时根**出发，四个存储各写一个新 QQ 号，逐个断言文件出现、内容读得回来。
"""
from pathlib import Path

import pytest

from src.plugins.chatbot import chatlog, memory, session_memory

NEW = "4000000099"


@pytest.fixture
def fresh_root(tmp_path):
    """把四个存储一起指到一个**全新的空根**（conftest 已做同样的事，这里显式再来一遍，
    保证这个用例不依赖别的隔离）。"""
    memory.use_storage(tmp_path)
    session_memory.use_storage(tmp_path)
    memory._mm.use_storage(tmp_path)
    chatlog.LOG_DIR = tmp_path / "data" / "chat_log"
    chatlog._migrated = True
    return tmp_path


def test_new_user_gets_all_four_files(fresh_root):
    assert not list(fresh_root.iterdir()), "起点必须是空根（否则测不到'自动创建'）"

    memory.append_user_history(NEW, "你好呀", "你好，绿冻")
    session_memory._write_json(session_memory._file_for(NEW),
                               {"topic": "打招呼", "events": [], "last_active": ""})
    memory._mm.update_user_memory(NEW, {"new_impression": "新来的"})
    chatlog.log_turn(NEW, False, "你好呀", "你好，绿冻")

    files = {
        "短期记忆": fresh_root / "short_term" / f"{NEW}.json",
        "会话记忆": fresh_root / "session" / f"{NEW}.json",
        "长期记忆": fresh_root / "long_term" / f"{NEW}.json",
        "聊天落盘": fresh_root / "data" / "chat_log" / f"{NEW}.jsonl",
    }
    for name, f in files.items():
        assert f.exists(), f"{name} 没给新用户建出文件：{f}"


def test_new_user_content_reads_back(fresh_root):
    memory.append_user_history(NEW, "在吗", "在")
    assert memory.get_user_history(NEW) == ["用户：在吗", "灰泽满：在"]

    memory._mm.update_user_memory(NEW, {"new_impression": "爱熬夜"})
    assert memory.get_user_memory(NEW)["impressions"][0]["tag"] == "爱熬夜"

    chatlog.log_turn(NEW, False, "在吗", "在")
    assert [r["user"] for r in chatlog.iter_records()] == ["在吗"]


def test_second_turn_appends_without_losing_first(fresh_root):
    memory.append_user_history(NEW, "第一句", "第一答")
    memory.append_user_history(NEW, "第二句", "第二答")
    assert memory.get_user_history(NEW) == ["用户：第一句", "灰泽满：第一答",
                                           "用户：第二句", "灰泽满：第二答"]

    memory._mm.update_user_memory(NEW, {"new_impression": "爱熬夜"})
    memory._mm.update_user_memory(NEW, {"new_impression": "爱熬夜"})
    card = memory.get_user_memory(NEW)
    assert card["total_interactions"] == 2          # 累加
    assert len(card["impressions"]) == 1            # 同名标签不重复堆


def test_sessions_do_not_share_files(fresh_root):
    memory.append_user_history("111", "甲", "a")
    memory.append_user_history("222", "乙", "b")
    assert memory.get_user_history("111") == ["用户：甲", "灰泽满：a"]
    assert memory.get_user_history("222") == ["用户：乙", "灰泽满：b"]
    assert len(list((fresh_root / "short_term").glob("*.json"))) == 2
