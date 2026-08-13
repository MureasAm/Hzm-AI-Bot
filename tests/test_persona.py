"""persona.py 人格加载的单元测试。"""
import pytest

from src.plugins.chatbot.persona import (
    load_persona_rules,
    build_global_persona_context,
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
