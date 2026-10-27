# -*- coding: utf-8 -*-
"""memory-reflect 的候选→记忆条目映射（纯函数，不调模型）。"""
from scripts import memory_reflect as mr


def test_ritual_maps_with_cue():
    item, why = mr.candidate_to_item({
        "kind": "ritual", "key": "bedtime_blessing",
        "value": "晚安：晚安+喜欢绿冻+爱绿冻+mua", "days": 8, "count": 10,
    })
    assert why == ""
    assert item["kind"] == "interaction_preference"
    assert item["key"] == "ritual_bedtime_blessing"
    assert item["temporal_scope"] == "timeless"


def test_ritual_without_cue_rejected():
    item, why = mr.candidate_to_item({"kind": "ritual", "key": "x", "value": "晚安喜欢你"})
    assert item is None and why == "ritual_needs_cue"


def test_habit_gets_prefix_and_structured_meta():
    item, why = mr.candidate_to_item({
        "kind": "habit", "key": "morning_running",
        "value": "每天早上起床后都要跑操（学校要求）",
    })
    assert why == ""
    assert item["kind"] == "fact" and item["key"] == "habit_morning_running"
    assert item["habit"]["type"] == "routine"
    assert item["habit"]["cadence"] == "daily"
    assert item["habit"]["period"] == "morning"
    assert "跑操" in item["habit"]["cues"]


def test_habit_structured_fields_pass_through():
    item, why = mr.candidate_to_item({
        "kind": "habit", "key": "sticker", "value": "每天都会用表情包撒娇",
        "habit": {"type": "interaction", "cadence": "daily",
                  "period": "any", "cues": ["表情包", "撒娇"]},
    })
    assert why == ""
    assert item["habit"] == {"type": "interaction", "cadence": "daily",
                             "period": "any", "cues": ["表情包", "撒娇"]}


def test_habit_without_frequency_rejected():
    item, why = mr.candidate_to_item({
        "kind": "habit", "key": "morning_running",
        "value": "用户有早起跑步或早操的习惯",
    })
    assert item is None and why == "habit_missing_frequency"


def test_preference_nickname_maps_to_preferred_name():
    item, why = mr.candidate_to_item({
        "kind": "preference", "key": "user_call_me_wenbao",
        "value": "用户要求灰泽满称呼自己为“温宝”（早期曾叫“小温”）"})
    assert why == ""
    assert item["kind"] == "fact" and item["key"] == "preferred_name"
    assert item["value"] == "温宝"


def test_unknown_preference_with_frequency_becomes_habit():
    item, why = mr.candidate_to_item({
        "kind": "preference", "key": "user_seek_headpats",
        "value": "用户喜欢索要摸摸，频率为高频日常"})
    assert why == ""
    assert item["kind"] == "fact" and item["key"] == "habit_user_seek_headpats"


def test_habit_cadence_with_daily_life_accepted():
    item, why = mr.candidate_to_item({
        "kind": "habit", "key": "night_drawing",
        "value": "用户习惯在晚自习写机械制图，频率为有晚自习的日常"})
    assert why == "" and item["key"] == "habit_night_drawing"


def test_preference_must_be_known():
    item, why = mr.candidate_to_item({
        "kind": "preference", "key": "reply_length", "value": "short"})
    assert why == "" and item["kind"] == "interaction_preference"
    item, why = mr.candidate_to_item({
        "kind": "preference", "key": "reply_length", "value": "epic"})
    assert item is None and why == "invalid_preference_value"
    item, why = mr.candidate_to_item({
        "kind": "preference", "key": "unknown_thing", "value": "x"})
    assert item is None and why == "unknown_preference_key"


def test_fact_and_boundary_map_to_fact():
    for kind in ("fact", "boundary"):
        item, why = mr.candidate_to_item({"kind": kind, "key": "Some Key!", "value": "x"})
        assert why == "" and item["kind"] == "fact" and item["key"] == "some_key"


def test_unsupported_kind_and_empty_value():
    item, why = mr.candidate_to_item({"kind": "emotion", "key": "x", "value": "y"})
    assert item is None and why == "unsupported_kind"
    item, why = mr.candidate_to_item({"kind": "fact", "key": "x", "value": "  "})
    assert item is None and why == "empty_value"


def test_apply_mapped_writes_only_mapped(tmp_path, monkeypatch):
    """--from-file：只写 review 过的 mapped，不碰 candidates/rejected。"""
    import json

    from src.plugins.chatbot import memory_v2

    class _FakeStore:
        def __init__(self):
            self.state_file = tmp_path / "nope.json"
            self.calls = []

        def ingest(self, user, items, source):
            self.calls.append((user, items, source))
            return items

    fake = _FakeStore()
    monkeypatch.setattr(memory_v2, "ShadowMemoryStore", lambda: fake)
    path = tmp_path / "cand.json"
    path.write_text(json.dumps({
        "mapped": [{"candidate": {"evidence": "每天跑操"},
                    "item": {"kind": "fact", "key": "habit_run", "value": "每天跑操"}}],
        "rejected": [{"candidate": {"key": "x"}, "reason": "unknown"}],
    }, ensure_ascii=False), encoding="utf-8")

    out = mr.apply_mapped("u1", str(path))
    assert out["mapped"] == 1
    assert len(fake.calls) == 1
    assert fake.calls[0][1][0]["key"] == "habit_run"
