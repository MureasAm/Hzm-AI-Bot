# -*- coding: utf-8 -*-
"""表情包使用统计的测试。

这套东西和 `llm_cost` 同族：**分析用的旁路**。所以硬要求一样——
1. 记账绝不能影响聊天（写不进去就静默）；
2. 一行坏数据不能拖垮统计（跳过它，别整个报错）；
3. 聚合口径与写侧一致（事件名、`避免重复类别` 的归因）；
4. 没数据时输出要能读，不能崩。
"""
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sticker_stats  # noqa: E402

from src.plugins.chatbot import stickers  # noqa: E402


def _ev(event, sid="", group="", why="", t=None, session="u1"):
    return {"t": t if t is not None else time.time(), "session": session,
            "kind": "private", "event": event, "id": sid, "group": group, "why": why}


LIB = [
    {"id": "st_a", "file": "README.md", "desc": "甲", "group": "吐槽", "use_when": "吐槽时"},
    {"id": "st_b", "file": "README.md", "desc": "乙", "group": "吐槽", "use_when": "吐槽时"},
    {"id": "st_c", "file": "不存在的文件.png", "desc": "丙", "group": "夸赞", "use_when": "夸人时"},
]


# ---------------------------------------------------------------- 读库

def test_load_library_splits_by_file_existence(tmp_path):
    p = tmp_path / "lib.json"
    p.write_text(json.dumps({"stickers": LIB}, ensure_ascii=False), encoding="utf-8")
    ok, missing = sticker_stats.load_library(p)
    assert [s["id"] for s in ok] == ["st_a", "st_b"]
    assert [s["id"] for s in missing] == ["st_c"]      # 图不在磁盘上 → 单独归类


def test_load_library_tolerates_broken_file(tmp_path):
    p = tmp_path / "lib.json"
    p.write_text("{ 这不是 json", encoding="utf-8")
    assert sticker_stats.load_library(p) == ([], [])   # 坏文件当作空库，不抛


# ---------------------------------------------------------------- 读流水

def test_load_events_skips_bad_lines_and_filters_since(tmp_path):
    p = tmp_path / "log.jsonl"
    now = time.time()
    lines = [
        json.dumps(_ev("sent", "st_a", t=now - 100)),
        "{ 坏行",
        "",                                             # 空行
        json.dumps({"t": "不是数字", "event": "sent"}),  # 时间戳坏的
        json.dumps(_ev("sent", "st_b", t=now - 40 * 86400)),   # 40 天前
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    # 5 行里只有 2 行是好的：坏 json、空行、时间戳不是数字的三行都跳过
    assert len(sticker_stats.load_events(0, p)) == 2
    assert [e["id"] for e in sticker_stats.load_events(10, p)] == ["st_a"]


def test_load_events_missing_file_is_empty(tmp_path):
    assert sticker_stats.load_events(0, tmp_path / "没有这个文件.jsonl") == []


# ---------------------------------------------------------------- 聚合

def test_summarize_counts_and_unused():
    events = [
        _ev("cooldown"), _ev("cooldown"),
        _ev("none", why=""),                       # 模型说 null
        _ev("sent", "st_a", "吐槽"),
        _ev("sent", "st_a", "吐槽"),
        _ev("sent", "st_b", "吐槽"),
        _ev("sent_failed", "st_b", "吐槽"),
    ]
    st = sticker_stats.summarize(LIB, events)
    assert st["by_event"] == {"cooldown": 2, "none": 1, "sent": 3, "sent_failed": 1}
    # 发送率的分母只算"真的问了模型的"，冷却轮不算
    assert st["asked"] == 5 and st["sent"] == 3
    assert st["send_rate"] == 3 / 5
    assert st["per_id"] == {"st_a": 2, "st_b": 1}
    assert st["per_group"] == {"吐槽": 3}
    assert st["unused"] == ["st_c"]                # 库里有、从没发过


def test_summarize_classifies_none_reasons():
    events = [
        _ev("none", why=""),
        _ev("none", why="bad_id:st_zzz"),
        _ev("none", why="avoid_group", sid="st_a", group="吐槽"),
        _ev("none", why="error:boom"),
    ]
    r = sticker_stats.summarize(LIB, events)["none_reasons"]
    assert r["模型判定没有十分对应的（正常）"] == 1
    assert r["模型编了不存在的 id"] == 1
    assert r["选中了最近发过的类别，被代码拦下"] == 1
    assert r["判定调用出错"] == 1


def test_summarize_on_empty_events():
    st = sticker_stats.summarize(LIB, [])
    assert st["events_total"] == 0 and st["send_rate"] == 0.0
    assert st["unused"] == ["st_a", "st_b", "st_c"]


# ---------------------------------------------------------------- 打印

def test_render_empty_says_no_data_and_lists_library():
    st = sticker_stats.summarize(LIB, [])
    out = sticker_stats.render(st, LIB, [])
    assert "还没有数据" in out
    assert "吐槽" in out and "st_a" not in out.split("还没有数据")[0]


def test_render_full_report_mentions_counts_and_unused():
    events = [_ev("cooldown"), _ev("sent", "st_a", "吐槽")]
    st = sticker_stats.summarize(LIB, events)
    out = sticker_stats.render(st, LIB, [])
    assert "使用情况" in out
    assert "冷却" in out
    assert "st_c" in out                            # 零使用榜里要出现
    assert "平均每张" in out


# ---------------------------------------------------------------- 写侧

def test_log_event_appends_and_never_raises(tmp_path, monkeypatch):
    p = tmp_path / "sub" / "log.jsonl"              # 目录还不存在
    monkeypatch.setattr(stickers, "STICKER_LOG", p)
    stickers.log_event("sent", session="u1", kind="private",
                       sticker={"id": "st_a", "group": "吐槽"})
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["event"] == "sent" and rec["id"] == "st_a" and rec["session"] == "u1"

    # 写不进去也必须静默（统计绝不能拖垮发送）
    monkeypatch.setattr(stickers, "STICKER_LOG", tmp_path)   # 目标是目录 → 必然失败
    stickers.log_event("sent", sticker={"id": "st_b"})