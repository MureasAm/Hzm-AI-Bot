# -*- coding: utf-8 -*-
"""事件记忆：存/更新/检索/渲染。"""
import asyncio
import json

from src.plugins.chatbot import event_memory as em


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "EVENT_DIR", tmp_path)


def _event(**kw):
    base = {
        "date": "2026-11-03", "subject": "user", "source": "user_told",
        "what": "他捡了一只小猫并收养了", "status": "ongoing", "type": "pet",
        "entities": ["猫"], "confidence": 0.9,
    }
    base.update(kw)
    return base


def test_add_and_render(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    got = em.add_events("u1", [_event()])
    assert len(got) == 1 and got[0]["id"].startswith("evt_")
    text = em.render_context(got)
    assert text.startswith("[2026-11-03 | 进行中 | 他说的]")
    assert "小猫" in text


def test_same_event_updates_status(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    first = em.add_events("u1", [_event(what="他出了车祸，现在住院", type="health",
                                       entities=["车祸"], status="ongoing")])[0]
    em.add_events("u1", [_event(update_of=first["id"], what="他车祸的伤已经好了",
                                type="health", entities=["车祸"], status="ended")])
    events = em._load_events("u1")
    assert len(events) == 1
    assert events[0]["status"] == "ended"
    assert events[0]["history"][0]["status"] == "ongoing"


def test_template_or_joke_rejected(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    got = em.add_events("u1", [
        _event(what="丫头，我重新介绍一下自己，我今年四十，跑豫粤线大货"),
        _event(what="今天我给大家讲一个段子，哈哈哈哈"),
    ])
    assert got == []


def test_no_date_is_not_fabricated(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    got = em.add_events("u1", [_event(date="")])[0]
    assert got["date"] == ""


def test_retrieve_by_entity(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    em.add_events("u1", [_event()])
    async def run():
        return await em.retrieve("u1", "我家猫今天拆家了", entities=["猫"])
    got = asyncio.run(run())
    assert len(got) == 1 and "猫" in got[0]["what"]


def test_parse_payload_includes_events():
    from src.plugins.chatbot.memory_v2 import parse_extraction_payload
    payload = json.dumps({"items": [], "daily": [], "events": [_event()]}, ensure_ascii=False)
    parsed = parse_extraction_payload(payload)
    assert parsed["events"] and parsed["events"][0]["what"].startswith("他捡了")