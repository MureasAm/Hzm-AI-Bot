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
        now_context="【现在】10月27日 周二，灰泽满现在没有在直播。",
        time_hint="【本场时段】现在是深夜。",
        prefetch_context="【预取】[{\"text\":\"名字是小温\"}]",
        tool_note="【工具】需要时调用 recall；不确定就不调用。",
    )
    joined = "\n".join(m["content"] for m in msgs if isinstance(m.get("content"), str))
    assert "【现在】" in joined
    assert "【本场时段】" in joined
    assert "【预取】" in joined
    assert "【工具】" in joined
    # 工具提示在用户消息之前，自称一句在最后
    assert msgs[-1]["role"] == "user"
    assert msgs[-3]["content"].startswith("【工具】")
    assert "需要自称就用" in msgs[-2]["content"]
    # 不再出现旧的完整感知行（骨架正文里提到这个词不算）
    assert not any(m.get("content", "").startswith("【当前时间】") for m in msgs)


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