"""群级记忆（group_memory.py）单元测试。"""
import json

from src.plugins.chatbot import group_memory as gm


def _reset(tmp_path, monkeypatch):
    f = tmp_path / "groups.json"
    f.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gm, "GROUP_MEMORY_FILE", f)
    return f


def test_upsert_members_and_get(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    gm.upsert_members("g1", {"10001": "A", "10002": "B"})
    g = gm.get_group("g1")
    assert g["members"] == {"10001": "A", "10002": "B"}


def test_add_event_dedup_and_cap(tmp_path, monkeypatch):
    f = _reset(tmp_path, monkeypatch)
    gm.add_event("g1", "大家在聊周表")
    gm.add_event("g1", "大家在聊周表")     # 相邻重复 → 跳过
    gm.add_event("g1", "又聊到搬家了")
    g = gm.get_group("g1")
    assert [e["text"] for e in g["events"]] == ["又聊到搬家了", "大家在聊周表"]

    # 封顶：注入超上限，events 不超过 GROUP_EVENT_MAX
    for i in range(gm.GROUP_EVENT_MAX + 5):
        gm.add_event("g1", f"事件{i}")
    g = gm.get_group("g1")
    assert len(g["events"]) <= gm.GROUP_EVENT_MAX


def test_empty_group_returns_empty_shell(tmp_path, monkeypatch):
    f = _reset(tmp_path, monkeypatch)
    assert gm.get_group("不存在群") == {"members": {}, "events": []}
