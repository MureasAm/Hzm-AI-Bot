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


def test_habit_gets_prefix():
    item, why = mr.candidate_to_item({
        "kind": "habit", "key": "morning_running",
        "value": "每天早上起床后都要跑操（学校要求）",
    })
    assert why == ""
    assert item["kind"] == "fact" and item["key"] == "habit_morning_running"


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
