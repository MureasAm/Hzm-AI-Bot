# -*- coding: utf-8 -*-
"""注入 v3 registry：20 个 block 必须都被登记，policy/顺序/映射不能漂。"""
from src.plugins.chatbot import injection as inj
from src.plugins.chatbot import injection_registry as reg


def test_every_block_is_claimed_by_a_kind():
    blocks = set(inj.load_blocks())
    assert reg.validate_blocks(blocks) == [], "有 block.json 没被任何 KindSpec 认领"


def test_frame_order_only_uses_visible_policies():
    for spec in reg.frame_specs():
        assert spec.policy in {"always", "context", "conditional"}, spec.kind


def test_tool_kinds_are_the_expected_set():
    assert set(reg.tool_kinds()) >= {
        "fact", "preference", "core_story", "corpus", "daily", "weather", "perception",
    }
    assert "behaviors" not in reg.tool_kinds()
    assert "identity" not in reg.tool_kinds()


def test_no_duplicate_kind_and_required_metadata():
    for kind, spec in reg.REGISTRY.items():
        assert spec.kind == kind
        assert spec.policy in reg.POLICIES
        assert spec.subject in reg.SUBJECTS
        assert spec.validity in reg.VALIDITIES
        assert spec.authority in reg.AUTHORITIES
        assert spec.fail_mode in reg.FAIL_MODES


def test_identity_and_schedule_are_always():
    assert reg.policy("identity") == "always"
    assert reg.policy("schedule") == "always"
    assert reg.policy("frame_now") == "always"


def test_commitment_and_style_are_conditional_not_tool():
    assert reg.policy("commitment") == "conditional"
    assert reg.policy("behaviors") == "conditional"
    assert reg.policy("interaction_preference") == "conditional"
    assert reg.policy("habit") == "conditional"