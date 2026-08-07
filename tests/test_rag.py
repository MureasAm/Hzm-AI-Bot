"""rag.py：余弦相似度 + RAG 检索过滤的单元测试。"""
import asyncio

import pytest

from src.plugins.chatbot.constants import PROJECT_ROOT
from src.plugins.chatbot.rag import cosine_similarity, retrieve_semantic_contexts


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


class TestRetrieveSemanticContexts:
    def test_empty_db_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.plugins.chatbot.rag.VECTOR_FILE", PROJECT_ROOT / "no_such_file.json")
        result = asyncio.run(retrieve_semantic_contexts("你好", [1, 0, 0]))
        assert result == ""

    def test_no_result_above_threshold(self, monkeypatch):
        # 所有片段相似度都低于阈值 0.35
        fake_db = [{"text": "A", "vector": [1, 0, 0]}, {"text": "B", "vector": [0, 1, 0]}]
        monkeypatch.setattr("src.plugins.chatbot.rag.load_vector_db", lambda: fake_db)
        query = [-1.0, -1.0]  # 与 A、B 都接近正交，相似度≈0
        result = asyncio.run(retrieve_semantic_contexts("q", query))
        assert result == ""

    def test_top_k_picks_best(self, monkeypatch):
        fake_db = [
            {"text": "低相关", "vector": [1, 0]},
            {"text": "高相关", "vector": [1.0, 1.0]},
            {"text": "中相关", "vector": [1, 0.5]},
        ]
        monkeypatch.setattr("src.plugins.chatbot.rag.load_vector_db", lambda: fake_db)
        query = [1.0, 1.0]
        result = asyncio.run(retrieve_semantic_contexts("q", query, top_k=1))
        assert "高相关" in result
        assert "中相关" not in result

    def test_threshold_filters_low(self, monkeypatch):
        # 高相关在阈值内，低相关（相似度为负）被滤掉
        fake_db = [
            {"text": "相关片段", "vector": [1.0, 1.0]},
            {"text": "无关片段", "vector": [-1.0, -1.0]},
        ]
        monkeypatch.setattr("src.plugins.chatbot.rag.load_vector_db", lambda: fake_db)
        query = [1.0, 1.0]
        result = asyncio.run(retrieve_semantic_contexts("q", query, top_k=2))
        assert "相关片段" in result
        assert "无关片段" not in result
