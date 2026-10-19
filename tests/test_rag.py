"""rag.py：余弦相似度的单元测试。"""
import json

import pytest

from src.plugins.chatbot import rag
from src.plugins.chatbot.rag import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0
        assert cosine_similarity([1, 1], [0, 0]) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_similar_vectors_positive(self):
        # (1,1) 与 (1,0) 夹角 45°
        assert cosine_similarity([1, 1], [1, 0]) == pytest.approx(2 ** 0.5 / 2)


class TestCosineScores:
    """整库打分 `cosine_scores` 必须和逐条 `cosine_similarity` **逐位相同**（排序、阈值都不能变）。"""

    DB = [{"vector": [1, 2, 3]}, {"vector": [0, 0, 0]}, {"vector": [-1.5, 0.25]},
          {"vector": [3.1, 1.7, 4.2, 1.9]}]

    def test_matches_pairwise_bit_for_bit(self):
        q = [0.3, -0.7, 0.2]
        assert rag.cosine_scores(q, self.DB) == [rag.cosine_similarity(q, it["vector"]) for it in self.DB]

    def test_zero_query_scores_zero(self):
        assert rag.cosine_scores([0, 0, 0], self.DB) == [0.0] * len(self.DB)

    def test_cached_db_uses_precomputed_norms_same_result(self, tmp_path, monkeypatch):
        f = tmp_path / "vectors.json"
        f.write_text(json.dumps([{"text": t, "vector": v["vector"]} for t, v in zip("abcd", self.DB)]),
                     encoding="utf-8")
        monkeypatch.setattr(rag, "VECTOR_FILE", f)
        monkeypatch.setattr(rag, "_vector_db_cache", None)
        db = rag.load_vector_db()
        q = [0.9, 0.1, -0.4]
        assert rag.cosine_scores(q, db) == [rag.cosine_similarity(q, it["vector"]) for it in db]


class TestVectorDbCache:
    """16MB 的向量库不能每轮重读；但文件一改要自动重读（"改完不重启也生效"的老行为不变）。"""

    @pytest.fixture
    def vec_file(self, tmp_path, monkeypatch):
        f = tmp_path / "vectors.json"
        monkeypatch.setattr(rag, "VECTOR_FILE", f)
        monkeypatch.setattr(rag, "_vector_db_cache", None)
        return f

    def test_second_call_hits_cache(self, vec_file):
        vec_file.write_text(json.dumps([{"text": "a", "vector": [1, 0]}]), encoding="utf-8")
        first = rag.load_vector_db()
        assert rag.load_vector_db() is first

    def test_reloads_after_file_changes(self, vec_file):
        vec_file.write_text(json.dumps([{"text": "a", "vector": [1, 0]}]), encoding="utf-8")
        rag.load_vector_db()
        vec_file.write_text(json.dumps([{"text": "b", "vector": [0, 1]},
                                        {"text": "c", "vector": [1, 1]}]), encoding="utf-8")
        assert [it["text"] for it in rag.load_vector_db()] == ["b", "c"]

    def test_missing_or_broken_file_returns_empty_and_is_not_cached(self, vec_file):
        assert rag.load_vector_db() == []
        vec_file.write_text("{broken", encoding="utf-8")
        assert rag.load_vector_db() == []
        assert rag._vector_db_cache is None
