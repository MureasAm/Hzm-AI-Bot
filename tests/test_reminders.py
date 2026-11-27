# -*- coding: utf-8 -*-
"""定时提醒：L3 解析 + commitment due_at 存取（不联网）。"""
import datetime
import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import core, reminders, reply_evidence
from src.plugins.chatbot.core import _reminder_request_note
from src.plugins.chatbot.memory_v2 import ShadowMemoryStore
from src.plugins.chatbot.routing import _parse_reminder

NOW = datetime.datetime(2026, 11, 15, 18, 8)
TZ8 = datetime.timezone(datetime.timedelta(hours=8))


def test_parse_reminder_accepts_absolute_due():
    out = _parse_reminder({
        "what": "提醒他浇花", "when_hint": "18:20",
        "due_at": "2026-11-15T18:20:00+08:00", "ambiguous": False,
    }, NOW.replace(tzinfo=TZ8))
    assert out["ambiguous"] is False
    assert out["due_at"].startswith("2026-11-15T18:20:00")


def test_parse_reminder_marks_bad_or_missing_due_ambiguous():
    out = _parse_reminder({"what": "浇花", "when_hint": "十几分",
                           "due_at": "", "ambiguous": True,
                           "clarify": "是20分钟后还是20:20？"},
                          NOW.replace(tzinfo=TZ8))
    assert out["ambiguous"] is True
    assert out["due_at"] == ""
    assert "20" in out["clarify"]


def test_parse_reminder_rejects_past_and_empty_what():
    out = _parse_reminder({"what": "浇花", "due_at": "2026-11-15T17:00:00+08:00"},
                          NOW.replace(tzinfo=TZ8))
    assert out["ambiguous"] is True and out["due_at"] == ""
    assert _parse_reminder({"what": ""}, NOW.replace(tzinfo=TZ8)) is None


def test_reminder_note_asks_clarify_when_ambiguous():
    note = _reminder_request_note({"what": "浇花", "ambiguous": True,
                                   "clarify": "是20分钟后还是20:20？"})
    assert "反问" in note and "20" in note
    note = _reminder_request_note({"what": "浇花", "ambiguous": False,
                                   "when_hint": "18:20", "due_at": "2026-11-15T18:20:00"})
    assert "18:20" in note and "确认" in note


@pytest.fixture
def scheduled_commitment(tmp_path):
    store = ShadowMemoryStore(state_file=tmp_path / "state.json",
                              audit_file=tmp_path / "audit.jsonl")
    due = (NOW + datetime.timedelta(minutes=12)).isoformat()
    source = {
        "session_id": "900000001", "observed_at": NOW.isoformat(),
        "user_text": "18:20提醒我浇花", "assistant_text": "好，18:20提醒你",
    }
    accepted = store.ingest("900000001", [{
        "kind": "commitment", "key": "reminder_plants", "value": "浇花",
        "action": "upsert", "temporal_scope": "future", "valid_for": "none",
        "explicit": True, "confidence": 0.9, "actor": "assistant", "due_at": due,
    }], source=source, now=NOW)
    assert accepted and accepted[0]["due_at"] == due
    return store, accepted[0]["id"]


def test_commitment_due_at_roundtrip(scheduled_commitment):
    store, memory_id = scheduled_commitment
    due_list = store.due_commitments(NOW + datetime.timedelta(minutes=13))
    assert len(due_list) == 1
    assert due_list[0]["value"] == "浇花"
    assert due_list[0]["user_id"] == "900000001"

    assert store.mark_reminder_delivered("900000001", memory_id,
                                         NOW + datetime.timedelta(minutes=13)) is True
    assert store.due_commitments(NOW + datetime.timedelta(minutes=14)) == []
    assert store.snapshot("900000001")["memories"][0]["status"] == "done"
    assert store.commitment_history("900000001", "浇花")[0]["status"] == "done"

    state = json.loads(store.state_file.read_text(encoding="utf-8"))
    del state["users"]["900000001"]["memories"][0]["reminder_delivery_state"]
    store.state_file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    assert store.snapshot("900000001")["memories"][0]["status"] == "done"
    assert store.commitment_history("900000001", "浇花")[0]["status"] == "legacy_done"


async def test_tick_archives_reminder_only_after_qq_confirms(scheduled_commitment, monkeypatch):
    store, _ = scheduled_commitment
    attempts = []
    msg = "到点啦，垃圾记得拿下去。"

    async def render(*args):
        return msg

    async def send_private_msg(user_id, message):
        assert user_id == 900000001 and str(message) == msg
        assert not reply_evidence.recall_turns(str(user_id))
        attempts.append(user_id)
        if len(attempts) == 1:
            raise ConnectionError("QQ 未确认")
        return {"message_id": 12345}

    monkeypatch.setattr(core, "_MEMORY_V2_STORE", store)
    monkeypatch.setattr(reminders, "_render", render)
    monkeypatch.setattr(reminders, "get_bot", lambda: SimpleNamespace(send_private_msg=send_private_msg))
    now = NOW + datetime.timedelta(minutes=13)
    assert await reminders.tick(now) == 0
    assert store.due_commitments(now)
    assert await reminders.tick(now) == 1
    assert not store.due_commitments(now)

    [turn] = reply_evidence.recall_turns("900000001", turn_ids=("12345",))
    assert turn["user"] == ""
    assert turn["reply"] == msg and turn["verified"]
    [item] = store.commitment_history("900000001", "浇花")
    assert item["status"] == "done" and item["reminder_turn_id"] == turn["turn_id"]

    new_due = now + datetime.timedelta(days=1)
    [renewed] = store.ingest("900000001", [{
        "kind": "commitment", "key": item["key"], "value": item["value"],
        "actor": "assistant", "explicit": True, "confidence": 0.9, "due_at": new_due.isoformat(),
    }], source={
        "session_id": "another-session", "user_text": "明天也提醒我浇花",
        "assistant_text": "好，明天也提醒你。",
    }, now=now)
    assert renewed["status"] == "active" and "reminder_turn_id" not in renewed
    assert store.due_commitments(new_due)


async def test_late_reminder_is_missed_and_never_sent(scheduled_commitment, monkeypatch):
    store, _ = scheduled_commitment

    async def unexpected_send(**kwargs):
        raise AssertionError("超过补发期限的提醒不能发送")

    monkeypatch.setattr(core, "_MEMORY_V2_STORE", store)
    monkeypatch.setattr(reminders, "get_bot", lambda: SimpleNamespace(send_private_msg=unexpected_send))
    now = NOW + datetime.timedelta(hours=7)
    assert await reminders.tick(now) == 0
    assert not store.due_commitments(now)
    assert not reply_evidence.recall_turns("900000001")
    [item] = store.commitment_history("900000001", "浇花", now=now)
    assert item["status"] == "missed"
    assert "reminded_at" not in item and "reminder_turn_id" not in item
