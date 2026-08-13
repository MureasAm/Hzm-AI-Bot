"""rag.py：余弦相似度的单元测试。"""
import pytest

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
