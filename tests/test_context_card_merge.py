"""争议事件只能压下对应卡片值；不能丢掉同栏其它事实或残留整张旧卡。"""
import copy

import pytest

from src.plugins.chatbot import core, memory_v2


def test_disputed_merge_preserves_other_values_in_the_same_section():
    card = "本名：小明\n家人：用户收养了一只猫；有个妹妹在读高中\n兴趣：喜欢天文"
    event = {"what": "用户收养了一只猫", "_retrieval_disputed": True,
             "date": "2026-11-01", "status": "ongoing"}
    original = copy.deepcopy(event)

    merged, kept = memory_v2.merge_events_into_card(card, [event])

    assert merged == "本名：小明\n家人：有个妹妹在读高中\n兴趣：喜欢天文"
    assert kept == [event] and event == original


def test_disputed_merge_removes_empty_section_and_keeps_evidence():
    event = {"what": "用户收养了一只猫", "_retrieval_disputed": True}
    merged, kept = memory_v2.merge_events_into_card("家人：用户收养了一只猫", [event])

    assert merged == ""
    assert kept == [event]


def test_uncontested_merge_still_adds_date_and_deduplicates_event():
    event = {"what": "用户收养了一只猫", "date": "2026-11-01", "status": "ended"}
    merged, kept = memory_v2.merge_events_into_card("家人：用户收养了一只猫；有个妹妹", [event])

    assert "2026-11-01，已结束" in merged and "有个妹妹" in merged
    assert kept == []


@pytest.mark.parametrize("card,remaining", [
    ("家人：用户收养了一只猫", ""),
    ("本名：小明\n家人：用户收养了一只猫；有个妹妹在读高中", "有个妹妹在读高中"),
])
async def test_handle_chat_renders_only_the_card_remaining_after_disputed_merge(
        monkeypatch, tmp_path, card, remaining):
    store = memory_v2.ShadowMemoryStore(state_file=tmp_path / "v2.json",
                                       audit_file=tmp_path / "v2_audit.jsonl")
    monkeypatch.setattr(store, "build_state_card", lambda *args, **kwargs: card)
    monkeypatch.setattr(core, "_MEMORY_V2_STORE", store)
    monkeypatch.setattr(core, "_get_clients", lambda: (None, None))
    monkeypatch.setattr(core, "aux_clients_or", lambda fallback: (None, None))
    monkeypatch.setattr(core, "injection_v3_enabled", lambda: True)
    monkeypatch.setattr(core.memory_v2, "injection_enabled", lambda: True)
    monkeypatch.setattr(core.self_speech, "enabled", lambda: False)
    monkeypatch.setattr(core, "_weather_city_for", lambda uid: "")
    monkeypatch.setattr(core, "_user_name_for", lambda uid: "")
    monkeypatch.setattr(core, "get_llm_route", lambda: "")
    monkeypatch.setattr(core, "LEGENDARY_REPLIES", {})
    monkeypatch.setattr(core, "load_terms", lambda: [])
    monkeypatch.setattr(core, "spawn", lambda coro: coro.close())
    captured = {}

    async def probe(*args, **kwargs):
        return "我没养过猫，你记错了", {"entities": ["猫"]}

    async def retrieve(*args, **kwargs):
        return {"fused_items": [], "preference_items": [], "core_stories": [],
                "denied_terms": set(), "approved_terms": set()}

    async def events(*args, **kwargs):
        return [{"what": "用户收养了一只猫", "subject": "user", "source": "user_told",
                 "_retrieval_disputed": True}]

    async def prefetch(*args, **kwargs):
        return "", []

    async def no_hints(*args, **kwargs):
        return {}

    class Rendered(Exception):
        pass

    def capture(*args, **kwargs):
        captured.update(kwargs)
        raise Rendered

    monkeypatch.setattr(core, "probe_session", probe)
    monkeypatch.setattr(core, "gather_retrieval", retrieve)
    monkeypatch.setattr(core.event_memory, "retrieve", events)
    monkeypatch.setattr(core.need_triggers, "prefetch", prefetch)
    monkeypatch.setattr(core, "filter_memory_hints", no_hints)
    monkeypatch.setattr(core, "build_message_list", capture)

    with pytest.raises(Rendered):
        await core.handle_chat("disputed-card-case", "我没养过猫，你记错了")

    stable = captured["state_stable"]
    assert "用户收养了一只猫" not in stable
    if remaining:
        assert remaining in stable and "本名：小明" in stable
    else:
        assert stable == ""
    assert "用户收养了一只猫" in captured["event_context"]
