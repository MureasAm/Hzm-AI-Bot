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
        _mem(key="habit", value="药一般是中午吃饭前吃和晚上睡前吃助眠"),
    ]}}}
    report = h.repair(state)
    m = state["users"]["u"]["memories"][0]
    assert m["key"] == "habit_medication"
    assert m["habit"]["type"] == "routine"
    assert any(c["action"] == "bare_habit_renamed" for c in report["changes"])


def test_stale_one_off_superseded_and_habit_restored():
    state = {"users": {"2151983431": {"memories": [
        _mem(key="habit_user_bedtime_comfort_seeking", value="已经洗完澡躺床上了"),
        _mem(key="habit_user_bedtime_comfort_seeking",
             value="用户习惯在睡前或聊天时频繁向灰泽满索要摸摸和抚慰",
             status="superseded", invalidated_at="2026-10-27T00:00:00"),
    ]}}}
    h.repair(state)
    stale, proper = state["users"]["2151983431"]["memories"]
    assert stale["status"] == "superseded"
    assert proper["status"] == "confirmed"


def test_same_key_keeps_newest_only():
    state = {"users": {"u": {"memories": [
        _mem(kind="commitment", key="send_voice", value="明天发语音",
             created_at="2026-11-01T00:00:00", last_observed_at="2026-11-01T00:00:00"),
        _mem(kind="commitment", key="send_voice", value="明天一定发语音",
             created_at="2026-11-02T00:00:00", last_observed_at="2026-11-02T00:00:00"),
    ]}}}
    h.repair(state)
    statuses = [m["status"] for m in state["users"]["u"]["memories"]]
    assert statuses == ["superseded", "confirmed"]
