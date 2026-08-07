"""persona.py 行为匹配与人格加载的单元测试。"""
import asyncio
import pytest

from src.plugins.chatbot.persona import (
    load_persona_rules,
    build_global_persona_context,
    match_behaviors_semantic,
    load_trigger_vectors,
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


class TestBuildGlobalPersonaContext:
    def test_empty_inputs_return_empty(self):
        assert build_global_persona_context([], []) == ""

    def test_builds_sections(self):
        ctx = build_global_persona_context(["嘴硬"], ["括号自嘲"])
        assert "性格基底" in ctx
        assert "语言风格" in ctx
        assert "嘴硬" in ctx
        assert "括号自嘲" in ctx


class TestLoadTriggerVectors:
    def test_cache_file_exists_and_populated(self):
        vectors = load_trigger_vectors()
        assert isinstance(vectors, dict) and len(vectors) >= 1
        # 与 behaviors 的 trigger 一一对应
        _, _, behaviors = load_persona_rules()
        for b in behaviors:
            assert b["trigger"] in vectors


class TestMatchBehaviorsSemantic:
    def test_empty_behaviors_returns_empty(self):
        result = asyncio.run(match_behaviors_semantic("你好", [1, 0, 0], []))
        assert result == ""

    def test_no_query_vector_returns_empty(self):
        _, _, behaviors = load_persona_rules()
        result = asyncio.run(match_behaviors_semantic("你好", None, behaviors))
        assert result == ""

    def test_high_similarity_returns_rule(self):
        _, _, behaviors = load_persona_rules()
        trigger_vectors = load_trigger_vectors()
        # 用第一个 trigger 的向量作为 query，应命中对应规则
        target = behaviors[0]
        query_vec = trigger_vectors[target["trigger"]]
        result = asyncio.run(match_behaviors_semantic("q", query_vec, behaviors))
        assert target["name"] in result

    def test_below_threshold_returns_empty(self):
        _, _, behaviors = load_persona_rules()
        trigger_vectors = load_trigger_vectors()
        query_vec = trigger_vectors[behaviors[0]["trigger"]]
        # 余弦相似度最大为 1.0，阈值取 1.01 保证任何匹配都无法达标
        result = asyncio.run(match_behaviors_semantic(
            "q", query_vec, behaviors, threshold=1.01
        ))
        assert result == ""

    def test_best_match_selected(self):
        _, _, behaviors = load_persona_rules()
        trigger_vectors = load_trigger_vectors()
        query_vec = trigger_vectors[behaviors[0]["trigger"]]
        result = asyncio.run(match_behaviors_semantic("q", query_vec, behaviors))
        # 命中的应该是相似度最高的行为（第一个 trigger 自身）
        assert behaviors[0]["name"] in result
