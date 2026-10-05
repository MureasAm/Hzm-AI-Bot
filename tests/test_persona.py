"""persona.py 人格加载的单元测试。"""
import pytest

from src.plugins.chatbot.persona import (
    load_persona_rules,
    build_global_persona_context,
)


class TestLoadPersonaRules:
    """traits / styles 已于 2026-10-04 拆掉（内容分进了 behaviors 与骨架）。

    保存的期望：这两个来源可以**为空**（文件不存在 → 空列表，不报错）；
    behaviors 必须存在。旧断言"三个都非空"已废。
    """

    def test_traits_and_styles_may_be_empty(self):
        traits, styles, behaviors = load_persona_rules()
        assert isinstance(traits, list)
        assert isinstance(styles, list)
        assert isinstance(behaviors, list) and len(behaviors) >= 1

    def test_behaviors_have_trigger_and_response(self):
        _, _, behaviors = load_persona_rules()
        for b in behaviors:
            assert b.get("trigger"), "behavior 必须有 trigger"
            assert b.get("response"), "behavior 必须有 response"


class TestScheduleWeekly:
    """周表：只回答"什么时候播"，是永不过期的固定表。

    曾有的「近况」字段（自由文本+TTL）已整条删除——它治不了"用旧记忆回答当下"
    （真正的漏点是 voice_samples 那条 assistant turn 通道），且手动维护必烂。
    见 src/plugins/chatbot/constants.py 的说明与 docs/历史/待办清单.md 的「本土化」条。
    """

    def test_loads_weekly(self):
        from src.plugins.chatbot.persona import load_schedule
        sched = load_schedule()
        assert isinstance(sched, dict)
        weekly = sched.get("weekly")
        assert isinstance(weekly, list) and len(weekly) == 7, "周表该有周一到周日 7 条"
        for x in weekly:
            assert x.get("day") and x.get("time")

    def test_note_field_is_gone(self):
        # 防回归：别再把这套会腐烂的字段加回来（它曾被当"现在为什么没播"的理由用了一周）
        from src.plugins.chatbot.persona import load_schedule
        sched = load_schedule()
        assert "近况" not in sched
        assert "近况_updated" not in sched


class TestBuildGlobalPersonaContext:
    def test_empty_inputs_return_empty(self):
        assert build_global_persona_context([], []) == ""

    def test_builds_sections(self):
        ctx = build_global_persona_context(["嘴硬"], ["括号自嘲"])
        assert "性格基底" in ctx
        assert "语言风格" in ctx
        assert "嘴硬" in ctx
        assert "括号自嘲" in ctx
