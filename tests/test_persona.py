"""persona.py 人格加载的单元测试。"""
from datetime import date, timedelta

import pytest

from src.plugins.chatbot.persona import (
    load_persona_rules,
    build_global_persona_context,
    schedule_note_age_days,
    schedule_note_is_fresh,
)


class TestLoadPersonaRules:
    def test_loads_all_three_files(self):
        traits, styles, behaviors = load_persona_rules()
        assert isinstance(traits, list) and len(traits) >= 1
        assert isinstance(styles, list) and len(styles) >= 1
        assert isinstance(behaviors, list) and len(behaviors) >= 1

    def test_behaviors_have_trigger_and_response(self):
        _, _, behaviors = load_persona_rules()
        for b in behaviors:
            assert b.get("trigger"), "behavior 必须有 trigger"
            assert b.get("response"), "behavior 必须有 response"


class TestScheduleNoteFreshness:
    """近况时效：踩坑是"这周在收拾搬家"（更新于 09-05）被当当前理由用了一周。

    为什么必须代码判、不交给模型：模型不知道阈值、也不会拿"更新于 X"去和当前时间做减法，
    而【一致性规则】会把第一次用错的借口锁死（错误只固化、不自我纠正）。
    """

    def _sched(self, days_ago):
        return {"近况": "这周在收拾搬家", "近况_updated": str(date.today() - timedelta(days=days_ago))}

    def test_today_is_fresh(self):
        assert schedule_note_age_days(self._sched(0)) == 0
        assert schedule_note_is_fresh(self._sched(0)) is True

    def test_within_ttl_is_fresh(self):
        assert schedule_note_is_fresh(self._sched(6)) is True

    def test_at_ttl_is_stale(self):
        # 7 天 = 边界：近况自己写"这周…"，一周就是它的自然边界，满 7 天即过期
        assert schedule_note_is_fresh(self._sched(7)) is False

    def test_long_overdue_is_stale(self):
        assert schedule_note_is_fresh(self._sched(30)) is False

    def test_missing_date_is_untrusted(self):
        # 没有更新日期 = 无法判断时效 → 当作不可信，别当当前事实用
        assert schedule_note_age_days({"近况": "在忙"}) is None
        assert schedule_note_is_fresh({"近况": "在忙"}) is False

    def test_broken_date_is_untrusted(self):
        assert schedule_note_age_days({"近况_updated": "不是日期"}) is None
        assert schedule_note_is_fresh({"近况_updated": "上周"}) is False

    def test_future_date_is_untrusted(self):
        # 时钟回拨/写错成未来 → 不可信，不当"刚更新过"放行
        future = {"近况": "在忙", "近况_updated": str(date.today() + timedelta(days=3))}
        assert schedule_note_age_days(future) is None
        assert schedule_note_is_fresh(future) is False

    def test_non_dict_is_untrusted(self):
        assert schedule_note_age_days(None) is None
        assert schedule_note_is_fresh({}) is False

    def test_iso_with_time_also_parsed(self):
        # 容忍带时间的 ISO（其他通道写的是 full isoformat）
        s = {"近况_updated": f"{date.today() - timedelta(days=2)}T10:00:00"}
        assert schedule_note_age_days(s) == 2
        assert schedule_note_is_fresh(s) is True


class TestBuildGlobalPersonaContext:
    def test_empty_inputs_return_empty(self):
        assert build_global_persona_context([], []) == ""

    def test_builds_sections(self):
        ctx = build_global_persona_context(["嘴硬"], ["括号自嘲"])
        assert "性格基底" in ctx
        assert "语言风格" in ctx
        assert "嘴硬" in ctx
        assert "括号自嘲" in ctx
