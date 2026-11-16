# -*- coding: utf-8 -*-
"""定时提醒：L3 解析 + commitment due_at 存取（不联网）。"""
import datetime

from src.plugins.chatbot.core import _reminder_request_note
from src.plugins.chatbot.memory_v2 import ShadowMemoryStore
from src.plugins.chatbot.routing import _parse_reminder

NOW = datetime.datetime(2026, 11, 15, 18, 8)
TZ8 = datetime.timezone(datetime.timedelta(hours=8))


def test_parse_reminder_accepts_absolute_due():
    out = _parse_reminder({
        "what": "提醒他扔垃圾", "when_hint": "18:20",
        "due_at": "2026-11-15T18:20:00+08:00", "ambiguous": False,
    }, NOW.replace(tzinfo=TZ8))
    assert out["ambiguous"] is False
    assert out["due_at"].startswith("2026-11-15T18:20:00")


def test_parse_reminder_marks_bad_or_missing_due_ambiguous():
    out = _parse_reminder({"what": "扔垃圾", "when_hint": "20的时候",
                           "due_at": "", "ambiguous": True,
                           "clarify": "是20分钟后还是20:20？"},
                          NOW.replace(tzinfo=TZ8))
    assert out["ambiguous"] is True
    assert out["due_at"] == ""
    assert "20" in out["clarify"]


def test_parse_reminder_rejects_past_and_empty_what():
    out = _parse_reminder({"what": "扔垃圾", "due_at": "2026-11-15T17:00:00+08:00"},
                          NOW.replace(tzinfo=TZ8))
    assert out["ambiguous"] is True and out["due_at"] == ""
    assert _parse_reminder({"what": ""}, NOW.replace(tzinfo=TZ8)) is None


def test_reminder_note_asks_clarify_when_ambiguous():
    note = _reminder_request_note({"what": "扔垃圾", "ambiguous": True,
                                   "clarify": "是20分钟后还是20:20？"})
    assert "反问" in note and "20" in note
    note = _reminder_request_note({"what": "扔垃圾", "ambiguous": False,
                                   "when_hint": "18:20", "due_at": "2026-11-15T18:20:00"})
    assert "18:20" in note and "确认" in note


def test_commitment_due_at_roundtrip(tmp_path):
    store = ShadowMemoryStore(state_file=tmp_path / "state.json",
                              audit_file=tmp_path / "audit.jsonl")
    due = (NOW + datetime.timedelta(minutes=12)).isoformat()
    source = {
        "session_id": "2151983431", "observed_at": NOW.isoformat(),
        "user_text": "18:20提醒我扔垃圾", "assistant_text": "好，18:20提醒你",
    }
    accepted = store.ingest("2151983431", [{
        "kind": "commitment", "key": "reminder_trash", "value": "扔垃圾",
        "action": "upsert", "temporal_scope": "future", "valid_for": "none",
        "explicit": True, "confidence": 0.9, "actor": "assistant", "due_at": due,
    }], source=source, now=NOW)
    assert accepted and accepted[0]["due_at"] == due

    due_list = store.due_commitments(NOW + datetime.timedelta(minutes=13))
    assert len(due_list) == 1
    assert due_list[0]["value"] == "扔垃圾"
    assert due_list[0]["user_id"] == "2151983431"

    assert store.mark_reminder_delivered("2151983431", due_list[0]["id"],
                                         NOW + datetime.timedelta(minutes=13)) is True
    assert store.due_commitments(NOW + datetime.timedelta(minutes=14)) == []
    assert store.snapshot("2151983431")["memories"][0]["status"] == "done"
