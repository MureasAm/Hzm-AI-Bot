"""persona.py 人格加载的单元测试。"""
from src.plugins.chatbot.persona import load_persona_rules


class TestLoadPersonaRules:
    """behaviors 是人格层唯一还在的数据文件。

    traits / styles 两个文件已于 2026-10-04 拆掉（情景部分并进 behaviors，其余并进骨架），
    读取它们的代码也一并删了——留着只会读到不存在的文件、恒返回空，
    还会让骨架里那句"见注入的【性格基底】"指向一个永不出现在 messages 里的块。
    """

    def test_behaviors_loaded(self):
        behaviors = load_persona_rules()
        assert isinstance(behaviors, list) and len(behaviors) >= 1

    def test_behaviors_have_trigger_and_response(self):
        for b in load_persona_rules():
            assert b.get("trigger"), "behavior 必须有 trigger"
            assert b.get("response"), "behavior 必须有 response"


class TestScheduleWeekly:
    """周表：只回答"什么时候播"，是永不过期的固定表。

    曾有的「近况」字段（自由文本+TTL）已整条删除——它治不了"用旧记忆回答当下"，
    且手动维护必烂。理由与实测数据见 docs/历史/已删机制登记.md。
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
