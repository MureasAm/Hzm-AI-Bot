# -*- coding: utf-8 -*-
"""长期请求（request）：存储、渲染、过期、同 target 顶替。"""
import json
from datetime import datetime, timedelta

from src.plugins.chatbot import memory_v2


def _store(tmp_path):
    return memory_v2.ShadowMemoryStore(
        state_file=tmp_path / "state.json", audit_file=tmp_path / "audit.jsonl")


def _source(now):
    return {"session_id": "s1", "observed_at": now.isoformat(),
            "user_text": "以后多叫叫我名字", "assistant_text": "行吧。"}


def _item(**kw):
    d = {"kind": "request", "key": "request_address",
         "value": "他希望以后多叫他的名字", "target": "address",
         "scope": "standing", "action": "upsert", "explicit": True,
         "confidence": 1.0, "temporal_scope": "timeless", "actor": "user"}
    d.update(kw)
    return d


def test_request_roundtrip_and_name(tmp_path):
    now = datetime(2026, 10, 27, 12, 0)
    store = _store(tmp_path)
    store.ingest("u", [{
        "kind": "fact", "key": "preferred_name", "value": "小温",
        "action": "upsert", "explicit": True, "confidence": 1.0,
        "temporal_scope": "timeless",
    }], source=_source(now), now=now)
    accepted = store.ingest("u", [_item()], source=_source(now), now=now)
    assert len(accepted) == 1 and accepted[0]["status"] == "confirmed"
    text = store.build_request_context("u", now=now)
    assert "【他对你的长期请求】" in text
    assert "多叫他的名字" in text and "小温" in text


def test_request_same_target_supersedes(tmp_path):
    now = datetime(2026, 10, 27, 12, 0)
    store = _store(tmp_path)
    store.ingest("u", [_item()], source=_source(now), now=now)
    store.ingest("u", [_item(value="他希望以后每天多叫几次名字")],
                 source=_source(now), now=now)
    text = store.build_request_context("u", now=now)
    assert "每天多叫几次" in text
    assert "多叫他的名字" not in text


def test_session_request_expires(tmp_path):
    now = datetime(2026, 10, 27, 12, 0)
    store = _store(tmp_path)
    store.ingest("u", [_item(scope="session", value="他本场希望多叫名字")],
                 source=_source(now), now=now)
    assert "本场希望" in store.build_request_context("u", now=now)
    assert store.build_request_context("u", now=now + timedelta(hours=4)) == ""


def test_invalid_target_rejected(tmp_path):
    now = datetime(2026, 10, 27, 12, 0)
    store = _store(tmp_path)
    accepted = store.ingest("u", [_item(key="request_frequency", target="frequency")],
                            source=_source(now), now=now)
    assert accepted == []


def test_prompt_has_request_channel():
    assert "request_<target>" in memory_v2.MEMORY_V2_EXTRACT_PROMPT
    assert "一次性请求不是 request" in memory_v2.MEMORY_V2_EXTRACT_PROMPT


def test_name_budget_relaxes_with_address_request(tmp_path, monkeypatch):
    from src.plugins.chatbot import core
    now = datetime(2026, 10, 27, 12, 0)
    store = _store(tmp_path)
    store.ingest("u", [{
        "kind": "fact", "key": "preferred_name", "value": "小温",
        "action": "upsert", "explicit": True, "confidence": 1.0,
        "temporal_scope": "timeless",
    }], source=_source(now), now=now)
    monkeypatch.setattr(core, "_MEMORY_V2_STORE", store)
    monkeypatch.setattr(core, "get_user_history",
                        lambda uid: ["灰泽满：小温，晚安。", "用户：晚安"])
    # 没有 request：最近叫过 1 次就提醒
    assert core.name_budget_note("u") != ""
    store.ingest("u", [_item()], source=_source(now), now=now)
    # 有 request_address：阈值放宽到 3 次
    assert core.name_budget_note("u") == ""