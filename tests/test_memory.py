"""memory_manager 记忆卡合并逻辑的单元测试（纯函数，不触 IO）。"""
import pytest

from memory_manager import merge_memory_card, update_user_memory


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
