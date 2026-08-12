"""memory_manager 记忆卡合并逻辑的单元测试（纯函数，不触 IO）。"""
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
