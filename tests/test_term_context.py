"""术语词形只产生候选，含义仍需按整轮语境裁决。"""
import pytest

from src.plugins.chatbot import term_hints as th


@pytest.mark.parametrize(
    ("text", "expected"),
    [("你也太考了", (True, False)), ("考试复习", (False, False)),
     ("思考问题", (False, False)), ("你也太孝了", (False, True))],
)
def test_term_match_separates_pattern_only_direct_and_concept_hits(text, expected):
    term = {"keyword": "考", "aliases": ["考子"], "pattern_only": True,
            "pattern": "太考|考子", "usage_triggers": ["太孝"]}
    assert th.term_match(term, text) == expected


@pytest.mark.parametrize("pattern_only", [False, True])
@pytest.mark.parametrize("pattern", [None, "["])
def test_invalid_term_regex_is_safe_and_does_not_bypass_pattern_only(pattern_only, pattern):
    term = {"keyword": "考", "pattern": pattern, "pattern_only": pattern_only,
            "usage_triggers": ["孝"]}
    assert th.term_match(term, "考试") == (not pattern_only, False)
    assert th.term_match(term, "孝") == (False, True)


def test_term_candidates_keep_full_metadata_and_direct_hits_first(monkeypatch):
    concept = {"keyword": "棉花糖", "meaning": "匿名提问", "usage_triggers": ["匿名提问"],
               "usage": "按提问场景理解", "extra": {"source": "persona"}}
    direct = {"keyword": "绿冻", "aliases": ["粉丝称呼"], "meaning": "粉丝名",
              "usage_triggers": ["粉丝群体"]}
    unrelated = {"keyword": "别的词", "meaning": "无关含义"}
    monkeypatch.setattr(th, "load_terms", lambda: [concept, direct, unrelated, direct])

    candidates = th.term_candidates("粉丝称呼和匿名提问")

    assert candidates == [direct, concept]
    assert candidates[1]["usage_triggers"] == ["匿名提问"]
    assert candidates[1]["extra"] == {"source": "persona"}


def test_auxiliary_term_hints_mark_meanings_as_unconfirmed_candidates(monkeypatch):
    monkeypatch.setattr(th, "load_terms", lambda: [
        {"keyword": "棉花糖", "meaning": "匿名提问服务", "usage": "读提问时使用"},
    ])

    hint = th.terms_hint("这是普通棉花糖，不是匿名提问")
    block = th.terms_hint_block("这是普通棉花糖，不是匿名提问")

    assert "候选释义" in hint and "匿名提问服务" in hint
    assert "候选释义" in block and "上下文" in block
    assert "不能据此确认" in block
    assert th.term_candidates("普通闲聊") == []
    assert th.terms_hint_block("普通闲聊") == ""
