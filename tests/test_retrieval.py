"""retrieval.py：三路检索 + RRF 融合 + 预算截断的单元测试。"""
import os

import pytest

from src.plugins.chatbot.retrieval import (
    RetrievalItem,
    retrieve_corpus,
    retrieve_voice_samples,
    retrieve_behaviors,
    select_behavior_item,
    rrf_fuse,
    truncate_by_budget,
    fuse_and_truncate,
    load_voice_sample_vectors,
    _score_candidates,
)


# ==================== RRF 融合 ====================

class TestRRFFuse:
    def test_empty_inputs_returns_empty(self):
        assert rrf_fuse([]) == []

    def test_behavior_weight_puts_instruction_first(self):
        # 同 rank：behavior 权重 1.5 > corpus/voice 1.0
        behavior = RetrievalItem(source="behavior", item_id="b", score=0.8)
        corpus = RetrievalItem(source="corpus", item_id="c", score=0.9)
        fused = rrf_fuse([[corpus], [behavior]])
        assert fused[0].source == "behavior"
        assert fused[1].source == "corpus"

    def test_fusion_score_decreases_with_rank(self):
        a = RetrievalItem(source="corpus", item_id="a", score=0.8)
        b = RetrievalItem(source="corpus", item_id="b", score=0.7)
        fused = rrf_fuse([[a, b]])
        assert fused[0].fusion_score > fused[1].fusion_score
        assert fused[0].rank == 1
        assert fused[1].rank == 2

    def test_tie_broken_by_raw_score(self):
        # 同一路内同权重，rank 不同已经区分；跨路同 rank 时按 score
        c1 = RetrievalItem(source="corpus", item_id="c1", score=0.9)
        v1 = RetrievalItem(source="voice_sample", item_id="v1", score=0.5)
        fused = rrf_fuse([[c1], [v1]])
        # 同 rank(1) 同权重，融合分相等 → 按 score 降序 → c1 在前
        assert fused[0].item_id == "c1"


# ==================== 预算截断 ====================

class TestTruncateByBudget:
    def test_within_budget_keeps_all_in_order(self):
        items = [
            RetrievalItem(source="corpus", item_id="a", score=1.0, text="x" * 10),
            RetrievalItem(source="corpus", item_id="b", score=0.9, text="y" * 10),
        ]
        kept = truncate_by_budget(items, budget_chars=100, max_item_chars=300)
        assert [i.item_id for i in kept] == ["a", "b"]

    def test_over_budget_drops_last(self):
        items = [
            RetrievalItem(source="corpus", item_id="a", score=1.0, text="x" * 10),
            RetrievalItem(source="corpus", item_id="b", score=0.9, text="y" * 10),
        ]
        kept = truncate_by_budget(items, budget_chars=15, max_item_chars=300)
        assert [i.item_id for i in kept] == ["a"]

    def test_voice_sample_cost_uses_extra(self):
        items = [
            RetrievalItem(source="voice_sample", item_id="v", score=1.0,
                          extra={"user": "问", "reply": "答" * 20}),
        ]
        kept = truncate_by_budget(items, budget_chars=10, max_item_chars=300)
        # user+reply = 1+20 = 21 字符 > 10 预算 → 丢弃
        assert kept == []

    def test_single_item_capped_at_max_item_chars(self):
        items = [RetrievalItem(source="corpus", item_id="a", score=1.0, text="x" * 500)]
        kept = truncate_by_budget(items, budget_chars=400, max_item_chars=300)
        # 单条按 300 计 > 400? no, 300 < 400 → 保留
        assert [i.item_id for i in kept] == ["a"]


# ==================== 三路 retriever ====================

def _vec(dim=4, fill=1.0):
    return [fill] * dim


class TestRetrieveCorpus:
    def test_no_query_vector_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db",
                            lambda: [{"text": "A", "vector": _vec()}])
        assert retrieve_corpus("q", None) == []

    def test_threshold_filters_and_top_n(self, monkeypatch):
        # V6 关键词门：语义达标 + 有实质词重叠才保留；无词重叠（即使语义高）滤掉
        db = [
            {"text": "灰泽满和女同学一起上学", "vector": [1, 0, 0, 0]},   # 相关且有词重叠
            {"text": "灰泽满喜欢看小说", "vector": [1, 1, 0, 0]},        # 语义达标但无词重叠
            {"text": "今天天气不错", "vector": [0, 0, 1, 0]},            # 正交
        ]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        items = retrieve_corpus("你和女同学一起上学的事", [1, 0, 0, 0], threshold=0.48, top_n=3)
        assert len(items) == 1
        assert items[0].text == db[0]["text"]
        assert all(i.source == "corpus" for i in items)

    def test_empty_db_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: [])
        assert retrieve_corpus("q", _vec()) == []


class TestCorpusKeywordGate:
    """V6：corpus 关键词门（语义降阈值 + 区分性词重叠，防'名词撞车但不相关'乱锁）。"""

    def test_positive_relevant_kept(self, monkeypatch):
        db = [{"text": "灰泽满和女同学一起上学，幸亏有女同学陪", "vector": [1, 0, 0, 0]}]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        items = retrieve_corpus("你和女同学一起上学的事", [0.5, 0.5, 0, 0], threshold=0.48, top_n=3)
        assert len(items) == 1
        assert items[0].text == db[0]["text"]

    def test_strong_keyword_overrides_low_sim(self, monkeypatch):
        # 专名"乌色月"在 query 和 statement 都有 → 语义低（0）也放行
        db = [{"text": "灰泽满写的小说《乌色月》", "vector": [1, 0, 0, 0]}]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        items = retrieve_corpus("乌色月是什么", [0, 0, 0, 0], threshold=0.48, top_n=3)
        assert len(items) == 1

    def test_noun_collision_rejected(self, monkeypatch):
        # "灰泽满是不是很懒"与直播陈述只共享名字"灰泽"，无实质词重叠 → 拒绝
        db = [{"text": "灰泽满当时提到有人问可不可以直播，她说网速不确定", "vector": [1, 0, 0, 0]}]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        items = retrieve_corpus("灰泽满是不是很懒", [1, 0, 0, 0], threshold=0.48, top_n=3)
        assert items == []

    def test_time_query_rejected(self, monkeypatch):
        # "9点是你那边"与"约了8点直播"无实质词重叠 → 拒绝（V5.3 当年伪关联回归案例）
        db = [{"text": "灰泽满说约了8点聊天，家里磨叽到8点后推迟到10点", "vector": [1, 0, 0, 0]}]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        items = retrieve_corpus("9点是你那边", [1, 0, 0, 0], threshold=0.48, top_n=3)
        assert items == []

    def test_no_query_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db",
                            lambda: [{"text": "A", "vector": [1, 0, 0, 0]}])
        assert retrieve_corpus("", [1, 0, 0, 0]) == []


class TestRetrieveVoiceSamples:
    def _fake_samples(self):
        return [
            {"id": "a", "user": "问A", "reply": "答A", "type": "daily", "vector": [1, 0, 0, 0]},
            {"id": "b", "user": "问B", "reply": "答B", "type": "emotion", "vector": [0, 1, 0, 0]},
        ]

    def test_no_vector_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_voice_sample_vectors", lambda: [])
        assert retrieve_voice_samples("q", _vec()) == []

    def test_keepalive_skips_when_all_below_threshold(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_voice_sample_vectors",
                            self._fake_samples)
        # query 与两个样本都正交 → 全低于阈值，且低于保底门槛 → 不注入（宁断档不错话题）
        items = retrieve_voice_samples("q", [0, 0, 1, 0], threshold=0.9)
        assert items == []

    def test_keepalive_injects_when_high_similarity(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_voice_sample_vectors",
                            self._fake_samples)
        # query 与样本 a 高相关（cos=1.0），但主阈值设很高 → 走保底注入 1 条
        items = retrieve_voice_samples("q", [1, 0, 0, 0], threshold=0.99)
        assert len(items) >= 1
        assert all(i.source == "voice_sample" for i in items)

    def test_returns_related_sample(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_voice_sample_vectors",
                            self._fake_samples)
        items = retrieve_voice_samples("q", [1, 0, 0, 0], threshold=0.5, top_n=2)
        assert len(items) == 1
        assert items[0].item_id == "a"
        assert items[0].extra["reply"] == "答A"


class TestRetrieveBehaviors:
    def test_no_behaviors_returns_empty(self):
        assert retrieve_behaviors("q", _vec(), []) == []

    def test_below_threshold_returns_empty(self, monkeypatch):
        behaviors = [{"name": "B", "trigger": "情境", "response": "反应"}]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_trigger_vectors",
                            lambda: {"情境": [0, 1, 0, 0]})
        # query 正交于 trigger → 相似度 0 < 0.65 → 空
        assert retrieve_behaviors("q", [1, 0, 0, 0], behaviors, threshold=0.65) == []

    def test_returns_best_behavior(self, monkeypatch):
        behaviors = [
            {"name": "B1", "trigger": "情境1", "response": "反应1"},
            {"name": "B2", "trigger": "情境2", "response": "反应2"},
        ]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_trigger_vectors",
                            lambda: {"情境1": [1, 0, 0, 0], "情境2": [0, 1, 0, 0]})
        items = retrieve_behaviors("q", [1, 0, 0, 0], behaviors, threshold=0.5)
        assert len(items) == 1
        assert items[0].item_id == "B1"
        assert "B1" in items[0].text
        assert "反应1" in items[0].text

    def test_keyword_fallback_forces_behavior(self):
        # 口语质问词（敷衍/鸽/迟到）→ 关键词兜底强制命中，即使 query 语义 miss（None）
        behaviors = [
            {"name": "被质疑时心虚辩解", "trigger": "质疑", "response": "先否认再心虚"},
            {"name": "失约被催时认栽滑跪", "trigger": "迟到", "response": "认栽滑跪"},
        ]
        items = retrieve_behaviors("你是不是在敷衍我", None, behaviors)
        assert len(items) == 1 and items[0].item_id == "被质疑时心虚辩解"
        items2 = retrieve_behaviors("说好的八点直播呢？你又鸽了", None, behaviors)
        assert len(items2) == 1 and items2[0].item_id == "失约被催时认栽滑跪"


class TestSelectBehaviorItem:
    """L3：LLM 判定的意图 → 行为注入项（关键词兜底）。"""

    def test_llm_intent_wins(self):
        behaviors = [
            {"name": "被质疑时心虚辩解", "trigger": "质疑", "response": "先否认"},
            {"name": "被夸时嘴硬否认", "trigger": "夸奖", "response": "否认"},
        ]
        item = select_behavior_item("你唱歌好好听", "被夸时嘴硬否认", behaviors)
        assert item is not None
        assert item.item_id == "被夸时嘴硬否认"
        assert "否认" in item.text

    def test_keyword_fallback_when_llm_null(self):
        behaviors = [
            {"name": "被质疑时心虚辩解", "trigger": "质疑", "response": "先否认"},
            {"name": "失约被催时认栽滑跪", "trigger": "迟到", "response": "滑跪"},
        ]
        # LLM 拿不准（空）但有判别词 → 关键词兜底命中
        item = select_behavior_item("说好的八点直播呢", "", behaviors)
        assert item is not None and item.item_id == "失约被催时认栽滑跪"

    def test_yuejie_keyword_fallback(self):
        # 领域黑话（黄桃/擦边）是判别词：LLM 对"造个黄桃吧"判 null，关键词兜底可靠
        behaviors = [{"name": "被越界时冷静推开", "trigger": "越界", "response": "推开"}]
        item = select_behavior_item("给我们造个黄桃吧", "", behaviors)
        assert item is not None and item.item_id == "被越界时冷静推开"
        assert select_behavior_item("说点擦边的听听", "", behaviors).item_id == "被越界时冷静推开"

    def test_keyword_does_not_hit_unrelated(self):
        behaviors = [{"name": "被质疑时心虚辩解", "trigger": "质疑", "response": "先否认"}]
        # "被骗"不在关键词表（敷衍/又骗/在骗…），不兜底
        assert select_behavior_item("我是不是被骗了", "", behaviors) is None

    def test_no_match_returns_none(self):
        behaviors = [{"name": "被质疑时心虚辩解", "trigger": "质疑", "response": "先否认"}]
        assert select_behavior_item("今天天气不错", "", behaviors) is None

    def test_empty_input_returns_none(self):
        assert select_behavior_item("", "", [{"name": "A", "trigger": "T", "response": "R"}]) is None


# ==================== 环境变量开关 ====================

class TestVoiceSampleEnvSwitch:
    def test_env_off_returns_empty(self, monkeypatch):
        os.environ["VOICE_SAMPLES"] = "0"
        try:
            monkeypatch.setattr("src.plugins.chatbot.retrieval._sample_vectors", None)
            assert load_voice_sample_vectors() == []
        finally:
            os.environ.pop("VOICE_SAMPLES", None)
            monkeypatch.setattr("src.plugins.chatbot.retrieval._sample_vectors", None)


# ==================== 融合流程 ====================

class TestFuseAndTruncate:
    def test_full_pipeline(self):
        corpus = [RetrievalItem(source="corpus", item_id="c", score=0.8, text="记忆")]
        sample = [RetrievalItem(source="voice_sample", item_id="v", score=0.7,
                                extra={"user": "问", "reply": "答"})]
        behavior = [RetrievalItem(source="behavior", item_id="b", score=0.9, text="指令")]
        fused = fuse_and_truncate(corpus, sample, behavior)
        assert fused[0].source == "behavior"  # 行为指令优先
        assert len(fused) <= 5

    def test_empty_all_returns_empty(self):
        assert fuse_and_truncate([], [], []) == []
