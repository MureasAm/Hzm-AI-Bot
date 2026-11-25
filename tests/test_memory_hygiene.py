# -*- coding: utf-8 -*-
"""记忆库卫生脚本的确定性修复。"""
from scripts import memory_hygiene as h


def _mem(**kw):
    base = {"kind": "fact", "status": "confirmed", "explicit": True,
            "temporal_scope": "timeless", "created_at": "2026-11-01T00:00:00",
            "last_observed_at": "2026-11-01T00:00:00"}
    base.update(kw)
    return base


def test_bare_habit_renamed_with_metadata():
    state = {"users": {"u": {"memories": [
        _mem(key="habit", value="每天早餐后按医嘱吃药"),
    ]}}}
    report = h.repair(state)
    m = state["users"]["u"]["memories"][0]
    assert m["key"] == "habit_medication"
    assert m["habit"]["type"] == "routine"
    assert any(c["action"] == "bare_habit_renamed" for c in report["changes"])


def test_same_key_keeps_newest_only():
    state = {"users": {"u": {"memories": [
        _mem(kind="commitment", key="send_voice", value="明天发语音",
             created_at="2026-11-01T00:00:00", last_observed_at="2026-11-01T00:00:00"),
        _mem(kind="commitment", key="send_voice", value="明天一定发语音",
             created_at="2026-11-02T00:00:00", last_observed_at="2026-11-02T00:00:00"),
        _mem(key="her_nickname", value="姐姐", temporal_scope="current"),
        _mem(key="her_nickname", value="妈妈", temporal_scope="current"),
    ]}}}
    h.repair(state)
    statuses = [m["status"] for m in state["users"]["u"]["memories"]]
    assert statuses == ["superseded", "confirmed", "confirmed", "confirmed"]
