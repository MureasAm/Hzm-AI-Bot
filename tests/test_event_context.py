# -*- coding: utf-8 -*-
"""事件候选必须按当前命题与近期对话选择，不能凭实体同词放行。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import event_memory as em


def _event(**changes):
    event = {
        "date": "2026-11-03", "subject": "user", "source": "user_told",
        "what": "他捡了一只小猫并收养了", "status": "ongoing", "type": "pet",
        "entities": ["猫"], "confidence": 0.9,
    }
    event.update(changes)
    return event


def _seed(tmp_path, monkeypatch, **changes):
    monkeypatch.setattr(em, "EVENT_DIR", tmp_path)
    return em.add_events("u1", [_event(**changes)])[0]


def _client(answer):
    prompts = []

    async def create(**kwargs):
        prompts.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=answer))])

    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))), prompts


def _context(query, **kwargs):
    from src.plugins.chatbot.retrieval_context import RetrievalContext
    return RetrievalContext(raw_text=query, **kwargs)


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_recall_survives_recent_entity_mention(tmp_path, monkeypatch, explicit):
    event = _seed(tmp_path, monkeypatch)
    result = asyncio.run(em.retrieve(
        "u1", "你还记得我收养猫那次吗", entities=["猫"],
        avoid_text="用户：我的猫刚才睡着了", explicit=explicit))
    assert [row["id"] for row in result] == [event["id"]]


def test_contextual_recall_uses_recent_history_without_entity_exclusion(tmp_path, monkeypatch):
    event = _seed(tmp_path, monkeypatch)
    client, prompts = _client('{"keep": [1]}')
    query = "你还记得我收养猫那次吗"
    history = "用户：我的猫刚才睡着了\n灰泽满：睡得可香了"
    result = asyncio.run(em.retrieve(
        "u1", query, context=_context(query, history=history, speech_act="recall"),
        entities=["猫"], explicit=True, avoid_text=history,
        judge_client=client, judge_model="offline-test"))
    assert [row["id"] for row in result] == [event["id"]]
    assert json.loads(prompts[0][-1]["content"])["本轮语境"]["最近对话"] == history


def test_negation_with_same_entity_still_requires_contextual_judgment(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client, prompts = _client('{"keep": []}')
    query = "我不是在说我收养过猫，是同事家的猫"
    result = asyncio.run(em.retrieve(
        "u1", query, context=_context(query, target="同事家的猫", speech_act="correction"),
        entities=["猫"], judge_client=client, judge_model="offline-test"))
    assert result == []
    assert prompts and query in str(prompts)


def test_contextual_entity_hit_fails_closed_without_judge(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    query = "我家猫今天拆家了"
    assert asyncio.run(em.retrieve(
        "u1", query, entities=["猫"], context=_context(query))) == []


def test_contextual_ordinary_event_is_selected(tmp_path, monkeypatch):
    event = _seed(tmp_path, monkeypatch)
    client, _ = _client('{"keep": [1]}')
    query = "我家猫今天拆家了"
    result = asyncio.run(em.retrieve(
        "u1", query, entities=["猫"], context=_context(query, topic="养猫"),
        judge_client=client, judge_model="offline-test"))
    assert [row["id"] for row in result] == [event["id"]]
    assert "小猫" in em.render_context(result)


def test_contextual_resolved_reference_recalls_an_event(tmp_path, monkeypatch):
    event = _seed(tmp_path, monkeypatch)
    client, prompts = _client('{"keep": [1]}')
    query = "后来怎么样了"
    resolved = "之前我捡了小猫并收养的那次，后来怎么样了"
    result = asyncio.run(em.retrieve(
        "u1", query, context=_context(query, resolved_text=resolved,
                                     history="用户：说起之前捡到小猫的事情"),
        judge_client=client, judge_model="offline-test"))
    assert [row["id"] for row in result] == [event["id"]]
    assert resolved in str(prompts)


def test_contextual_disputed_record_is_historical_without_persisting_flag(tmp_path, monkeypatch):
    event = _seed(tmp_path, monkeypatch, what="他出了车祸，现在住院", type="health",
                  entities=["车祸"], status="ongoing")
    client, _ = _client('{"keep": [1], "disputed": [1]}')
    query = "你记错了，我没有出车祸，住院的是朋友"
    result = asyncio.run(em.retrieve(
        "u1", query, context=_context(query, speech_act="correction", target="朋友"),
        entities=["车祸"], judge_client=client, judge_model="offline-test"))
    assert [row["id"] for row in result] == [event["id"]]
    rendered = em.render_context(result)
    assert "历史" in rendered and "待核对" in rendered
    assert "进行中" not in rendered
    assert em._load_events("u1") == [event]


def test_disputed_evidence_has_priority_when_output_limit_is_one(tmp_path, monkeypatch):
    first = _seed(tmp_path, monkeypatch, what="他家有一只成年猫")
    second = em.add_events("u1", [_event(what="他说猫在宠物医院", type="health")])[0]
    client, _ = _client('{"keep": [1,2], "disputed": [2]}')
    query = "我家猫没去医院，你把别人的事记我头上了"
    result = asyncio.run(em.retrieve("u1", query, context=_context(query),
                                    entities=["猫"], limit=1, judge_client=client, judge_model="offline-test"))
    assert len(result) == 1 and result[0].get('_retrieval_disputed')


@pytest.mark.parametrize("answer", [
    '{"keep": [true]}', '{"keep": "1"}', '{"keep": [1.0]}',
    '{"keep": [1],', '{"keep": [0, -1, 99]}',
    '{"keep": [1]}\n{"keep": [2]}',
])
def test_judge_rejects_malformed_or_non_integer_indices(tmp_path, monkeypatch, answer):
    event = _seed(tmp_path, monkeypatch)
    client, _ = _client(answer)
    assert asyncio.run(em._judge(client, "offline-test", "我想起收养那次", [event])) == []


@pytest.mark.parametrize("answer", [
    '```json\n{"keep": [1]}\n```',
    '```\n{"keep": [1]}\n```',
    '分析结束。\n{"keep": [1]}',
    '分析结束。\n{"keep": [1]}\n仅保留这条历史事件。',
])
def test_judge_accepts_fenced_or_explained_json(tmp_path, monkeypatch, answer):
    event = _seed(tmp_path, monkeypatch)
    client, _ = _client(answer)
    assert asyncio.run(em._judge(client, "offline-test", "我想起收养那次", [event])) == [event]
