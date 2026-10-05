"""retrieval.py：六路检索 + RRF 融合 + 预算截断的单元测试。"""
import os

import pytest

from src.plugins.chatbot.retrieval import (
    RetrievalItem,
    retrieve_corpus, retrieve_corpus_candidates,
    select_behavior_item,
    rrf_fuse,
    truncate_by_budget,
    fuse_and_truncate,
    _score_candidates,
    _corpus_keyword_overlap,
)
from src.plugins.chatbot.constants import CORPUS_KEYWORD_FLOOR


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


class TestRetrieveCorpusCandidates:
    """候选召回：**不过阈值、不过关键词门**，只按余弦取 top-N。

    它存在的理由：实测正例对正确那条的**排名**是对的（「你多高啊」→ 身高那条排第 1），
    只是分数（0.443）低于噪声地板——**排序有用、阈值没用**。打分交给排序，
    "相关不相关"交给 LLM 判（corpus_judge）。
    """

    def test_low_sim_still_returned(self, monkeypatch):
        # 门槛会滤掉的那条（0.443 < 0.48），候选召回必须给出来
        db = [{"text": "灰泽满被问到身高，直接说1米6", "vector": [0.443, 0.9, 0, 0]}]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        assert retrieve_corpus("你多高啊", [1, 0, 0, 0], threshold=0.48) == []
        cands = retrieve_corpus_candidates("你多高啊", [1, 0, 0, 0], top_n=3)
        assert [c.text for c in cands] == [db[0]["text"]]

    def test_sorted_by_score_and_top_n(self, monkeypatch):
        db = [
            {"text": "A", "vector": [0.2, 1, 0, 0]},
            {"text": "B", "vector": [0.9, 0, 0, 0]},
            {"text": "C", "vector": [0.5, 0, 0, 0]},
        ]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        cands = retrieve_corpus_candidates("q", [1, 0, 0, 0], top_n=2)
        assert [c.text for c in cands] == ["B", "C"]
        assert all(c.source == "corpus" for c in cands)

    def test_no_query_or_vector_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db",
                            lambda: [{"text": "A", "vector": [1, 0, 0, 0]}])
        assert retrieve_corpus_candidates("q", None) == []
        assert retrieve_corpus_candidates("", [1, 0, 0, 0]) == []
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: [])
        assert retrieve_corpus_candidates("q", [1, 0, 0, 0]) == []


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

    # ---- 2026-09-29：取消"门放行直通"时补的（此前这个门只有上面几条）----

    def test_误报的重叠度比真阳性还高_所以不能靠调阈值(self):
        """⚠️ 把这条事实钉住，免得后人再试一次"提高 FLOOR 挡误报"。

        两者只差"**说的是谁**"：误报是用户在说**自己**，真阳性是在问**她**。
        词重叠（甚至 bigram 文档频率）都拿不到这个信息 —— 实测试过 0.5：
        n7 挡住了，co9/co10 一起被打回红。
        """
        fp = _corpus_keyword_overlap(
            "宝宝我明天要早起我先去睡了",
            "粉丝问她明天会不会早起，她先表示可能性很低，又说其实很想早起")
        tp = _corpus_keyword_overlap(
            "你不是以前有喜欢的男学霸吗，你直播里说的",
            "灰泽满聊到学生时代时，提到自己曾被女同学质问是不是女同，情急之下谎称喜欢一个男同学")
        assert fp > tp, (
            f"前提变了（误报 ov={fp:.3f} 不再 > 真阳性 ov={tp:.3f}）—— "
            "若真是这样，那时再回来讨论阈值这个杠杆")

    def test_真提到同一件事仍然放行(self):
        ov = _corpus_keyword_overlap("你和女同学一起上学的事",
                                     "灰泽满和女同学一起上学，幸亏有女同学陪")
        assert ov >= CORPUS_KEYWORD_FLOOR

    def test_语义分低但词面沾边的条目也要递给LLM(self, monkeypatch):
        """co9 的救法，而且必须是**语义分低也照递**。

        踩过的坑：第一版补位沿用了 `_corpus_gate_pass`（要求 sim≥0.48）——
        而要救的恰恰是语义分低的那批（co9 的目标条 sim 不到 0.48），
        结果实测"候选 7 条 → 保留 0 条"，照样红。**补位只能看词面。**
        """
        target = {"text": "灰泽满聊到学生时代时，提到自己曾被女同学质问是不是女同，"
                          "情急之下谎称喜欢一个男同学",
                  "vector": [0.1, 0.995, 0, 0]}           # sim≈0.1：**远低于** 0.48
        filler = [{"text": f"无关的第{i}条", "vector": [1, 0, 0, 0]} for i in range(2)]
        db = [target] + filler
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_corpus_keywords", lambda: {})
        cands = retrieve_corpus_candidates("你不是以前有喜欢的男学霸吗，你直播里说的",
                                          [1, 0, 0, 0], top_n=1)
        assert "0" in [c.item_id for c in cands], "语义分低但词面沾边的条目没被递过去 → 等于丢召回"

    def test_没沾边的条目不会被塞进候选池(self, monkeypatch):
        # 反面：候选池别被灌爆（判定质量会被稀释 —— 判据对比实验量过"候选给多了反而更差"）
        db = [{"text": "灰泽满提到明天要搬家", "vector": [1, 0, 0, 0]},
              {"text": "完全无关的一条", "vector": [0.5, 0.86, 0, 0]}]
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_vector_db", lambda: db)
        monkeypatch.setattr("src.plugins.chatbot.retrieval.load_corpus_keywords", lambda: {})
        cands = retrieve_corpus_candidates("今天天气不错啊", [1, 0, 0, 0], top_n=1)
        assert [c.item_id for c in cands] == ["0"]




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

class TestFuseAndTruncate:
    def test_full_pipeline(self):
        corpus = [RetrievalItem(source="corpus", item_id="c", score=0.8, text="记忆")]
        behavior = [RetrievalItem(source="behavior", item_id="b", score=0.9, text="指令")]
        fused = fuse_and_truncate(corpus, behavior)
        assert fused[0].source == "behavior"  # 行为指令优先
        assert len(fused) <= 5

    def test_empty_all_returns_empty(self):
        assert fuse_and_truncate([], []) == []
