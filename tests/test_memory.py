"""memory_manager 记忆卡合并逻辑的单元测试（纯函数，不触 IO）。"""
from datetime import date, datetime, timedelta

import pytest

from memory_manager import merge_memory_card, update_user_memory, build_memory_context


class TestMergeMemoryCard:
    def test_first_interaction_sets_basics(self):
        card = merge_memory_card({}, {})
        assert card["total_interactions"] == 1
        assert "last_seen" in card

    def test_interaction_counter_increments(self):
        card = {"total_interactions": 5}
        card = merge_memory_card(card, {})
        assert card["total_interactions"] == 6

    def test_new_impression_added(self):
        card = merge_memory_card({}, {"new_impression": "夜猫子"})
        assert card["impressions"][0]["tag"] == "夜猫子"
        assert card["impressions"][0]["confidence"] == pytest.approx(0.8)

    def test_duplicate_impression_boosts_confidence(self):
        card = merge_memory_card({}, {"new_impression": "夜猫子"})
        card = merge_memory_card(card, {"new_impression": "夜猫子"})
        imps = card["impressions"]
        assert len(imps) == 1
        assert imps[0]["confidence"] == pytest.approx(0.9)

    def test_user_name_stored_and_overwritten(self):
        card = merge_memory_card({}, {"new_name": "小明"})
        assert card["user_name"] == "小明"
        card = merge_memory_card(card, {"new_name": "阿伟"})
        assert card["user_name"] == "阿伟"  # 改名覆盖
        ctx = build_memory_context(card)
        assert "阿伟" in ctx  # 注入时带上名字

    def test_supersede_removes_old_facts(self):
        card = merge_memory_card({}, {"new_impression": "上班族", "new_user_fact": "在XX公司做设计"})
        assert "上班族" in build_memory_context(card)
        card = merge_memory_card(card, {"new_impression": "自由职业", "supersede": ["上班族", "在XX公司做设计"]})
        ctx = build_memory_context(card)
        assert "自由职业" in ctx
        assert "上班族" not in ctx  # 旧标签被作废
        assert "在XX公司做设计" not in ctx  # 旧事实被作废

    def test_user_fact_dedup(self):
        card = merge_memory_card({}, {"new_user_fact": "在考研"})
        card = merge_memory_card(card, {"new_user_fact": "在考研"})
        assert len(card["user_facts"]) == 1
        card = merge_memory_card(card, {"new_user_fact": "养猫"})
        assert len(card["user_facts"]) == 2

    def test_does_not_mutate_input(self):
        original = {"total_interactions": 0}
        result = merge_memory_card(original, {"new_impression": "x"})
        assert original == {"total_interactions": 0}  # 输入未被污染
        assert result["total_interactions"] == 1

    # ---- null 字符串清洗（提取模型常把"无"写成字符串 "null" 而非 JSON null） ----

    def test_null_string_impression_not_stored(self):
        card = merge_memory_card({}, {"new_impression": "null"})
        assert "impressions" not in card or card["impressions"] == []

    def test_null_string_all_fields_not_stored(self):
        updates = {
            "new_impression": "null",
            "new_user_fact": "null",
            "new_self_fact": "None",
            "new_promise": "null",
            "new_moment": "null",
            "new_city": "null",
            "new_name": "null",
        }
        card = merge_memory_card({}, updates)
        for key in ("impressions", "user_facts", "self_facts", "promises",
                    "significant_moments", "weather_city", "user_name"):
            assert key not in card or not card[key]

    def test_supersede_ignores_null_entries(self):
        card = merge_memory_card({}, {"new_impression": "上班族"})
        card = merge_memory_card(card, {"new_impression": "自由职业", "supersede": ["上班族", "null"]})
        assert "上班族" not in build_memory_context(card)
        assert "自由职业" in build_memory_context(card)

    def test_legacy_null_polluted_card_not_injected(self):
        """历史遗留的 'null' 数据不应注入上下文（兜底在注入边界过滤）。"""
        card = {
            "user_name": "null",
            "weather_city": "null",
            "impressions": [{"tag": "null", "confidence": 0.9}],
            "user_facts": [{"fact": "null"}],
            "self_facts": [{"fact": "null"}],
            "significant_moments": [{"summary": "null"}],
            "promises": [{"promise": "null"}],
        }
        ctx = build_memory_context(card)
        assert ctx == ""
        assert "null" not in ctx


class TestLastSeenInjection:
    """last_seen 一直在存，以前从不注入——模型因此分不清"刚才"和"半个月前"。"""

    def test_recent_gap_injected(self):
        card = merge_memory_card({}, {})
        card["last_seen"] = (datetime.now() - timedelta(days=3)).isoformat()
        assert "3天" in build_memory_context(card)

    def test_fresh_card_says_nothing(self):
        # 刚聊完（last_seen≈现在）不该冒出"距上次 0 分钟"，同一场对话里这话很怪
        card = merge_memory_card({}, {})
        assert "距上次" not in build_memory_context(card)

    def test_missing_last_seen_says_nothing(self):
        assert "距上次" not in build_memory_context({"user_name": "小明"})

    def test_malformed_last_seen_ignored(self):
        assert "距上次" not in build_memory_context({"last_seen": "不是时间"})

    def test_future_timestamp_ignored(self):
        # 时钟回拨/脏数据不该注入"负几天"
        card = {"last_seen": (datetime.now() + timedelta(days=5)).isoformat()}
        assert "距上次" not in build_memory_context(card)


class TestSignificantMomentAge:
    """"最近的记忆"一直存着 date 却从不注入 → 几个月前的事也被说成"最近的"。

    和 schedule 近况 / 群近况同一个坑：**存了时间戳、注入时丢掉**。
    """

    def test_moment_injected_with_age(self):
        card = {"significant_moments": [
            {"date": (date.today() - timedelta(days=3)).isoformat(), "summary": "一起看了流星"}]}
        ctx = build_memory_context(card)
        assert "一起看了流星" in ctx
        assert "3天前" in ctx

    def test_recent_moment_has_no_age_prefix(self):
        # 今天的事不用标"今天前"——humanize_gap 对 <90s 返回空串，这里日期差 0 天
        card = {"significant_moments": [
            {"date": date.today().isoformat(), "summary": "刚说的事"}]}
        ctx = build_memory_context(card)
        assert "刚说的事" in ctx
        assert "前）" not in ctx

    def test_moment_without_date_still_injected(self):
        # 旧数据可能没有 date——不能因为它没时间就不注入内容
        ctx = build_memory_context({"significant_moments": [{"summary": "一起看了流星"}]})
        assert "一起看了流星" in ctx
        assert "前）" not in ctx

    def test_broken_date_ignored(self):
        ctx = build_memory_context({"significant_moments": [{"date": "不是日期", "summary": "某事"}]})
        assert "某事" in ctx
        assert "前）" not in ctx
