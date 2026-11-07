# -*- coding: utf-8 -*-
"""v3 frame：时间拆分、预取消息、相处方式不含事实。"""
import json

from src.plugins.chatbot import context_probe, core, memory_v2


def test_frame_now_does_not_contain_exact_time_or_weather():
    text = context_probe.frame_now()
    assert text.startswith("【当前时间】")
    assert "℃" not in text          # 不注实况天气
    assert "农历" not in text
    # 不出现 HH:MM（精确时间走 recall）
    assert ":" not in text.split("【当前时间】")[1].split("。")[0]


def test_session_time_hint_only_on_trigger_or_new_session():
    assert context_probe.session_time_hint(user_msg="今天好累") == ""
    assert context_probe.session_time_hint(user_msg="晚安") != ""
    assert context_probe.session_time_hint(user_msg="在吗", is_new_session=True) != ""


def test_perception_now_has_time_and_lunar():
    text = context_probe.perception_now()
    assert "北京时间" in text and "农历" in text


def test_build_message_list_uses_v3_contexts():
    msgs = core.build_message_list(
        "你好", [], [],
        state_stable="【关于这个绿冻】\n称呼：小温",
        state_dynamic="【当前状态】\n10月27日 周二，灰泽满现在没有在直播。",
        time_hint="【本场时段】现在是深夜。",
        prefetch_context="【预取】[{\"text\":\"名字是小温\"}]",
    )
    joined = "\n".join(m["content"] for m in msgs if isinstance(m.get("content"), str))
    assert "【关于这个绿冻】" in joined
    assert "【当前状态】" in joined
    assert "【本场时段】" in joined
    assert "【预取】" in joined
    assert msgs[-1]["role"] == "user"
    assert "需要自称就用" in msgs[-2]["content"]
    # 不再出现旧的完整感知行（骨架正文里提到这个词不算）
    assert not any(m.get("content", "").startswith("【当前时间】") for m in msgs)


def test_state_card_has_stable_slots(tmp_path):
    state = {"users": {"u": {"memories": [
        {"id": "a", "kind": "fact", "key": "preferred_name", "value": "学长",
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
        {"id": "b", "kind": "fact", "key": "location", "value": "在广东",
         "status": "confirmed", "temporal_scope": "current", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
        {"id": "c", "kind": "interaction_preference", "key": "reply_length", "value": "long",
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
        {"id": "d", "kind": "fact", "key": "interest_books", "value": "东野圭吾",
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
        {"id": "e", "kind": "fact", "key": "real_name", "value": "胡洋",
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
        {"id": "f", "kind": "fact", "key": "online_name", "value": "千纸鸢",
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
        {"id": "g", "kind": "fact", "key": "address_terms", "value": "学长、机长",
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
    ]}}}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    store = memory_v2.ShadowMemoryStore(state_file=p)
    card = store.build_state_card("u")
    assert "本名：胡洋" in card
    assert "网名：千纸鸢" in card
    assert "称呼：学长、机长" in card
    assert "地点：在广东" in card
    assert "多说几句" in card
    assert "兴趣：东野圭吾" in card


def test_profile_context_can_exclude_facts(tmp_path):
    state = {"users": {"u": {"memories": [
        {"id": "a", "kind": "interaction_preference", "key": "support_style",
         "value": "comfort", "status": "confirmed", "temporal_scope": "timeless",
         "explicit": True, "evidence_count": 1, "confidence": 0.9},
        {"id": "b", "kind": "fact", "key": "preferred_name", "value": "小温",
         "status": "confirmed", "temporal_scope": "timeless", "explicit": True,
         "evidence_count": 1, "confidence": 0.9},
    ]}}}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    store = memory_v2.ShadowMemoryStore(state_file=p)

    style_only = store.build_preference_context("u", "今天好累", include_facts=False)
    assert "接住情绪" in style_only
    assert "小温" not in style_only

    with_facts = store.build_preference_context("u", "今天好累", include_facts=True)
    assert "小温" in with_facts