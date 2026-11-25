"""Shared LLM JSON response parsing and candidate-index validation."""
import pytest

from src.plugins.chatbot.llm_json import parse_json_object, selected_indices


@pytest.mark.parametrize("text", [
    '{"keep": [2]}',
    '\ufeff{"keep": [2]}',
    '  \ufeff{"keep": [2]}  ',
    '```json\n{"keep": [2]}\n```',
    '```JSON\n{"keep": [2]}\n```',
    '```\n{"keep": [2]}\n```',
])
def test_parses_plain_bom_and_fenced_objects(text):
    assert parse_json_object(text) == {"keep": [2]}


def test_fence_and_json_words_inside_strings_are_preserved():
    text = '```json\n{"reason": "literal ``` and json text", "keep": [1]}\n```'
    assert parse_json_object(text) == {
        "reason": "literal ``` and json text", "keep": [1],
    }


@pytest.mark.parametrize("text", [
    '下面是判断结果。\n{"keep": [2]}',
    '{"keep": [2]}\n仅保留第二条。',
    '先说明筛选理由。\n{"keep": [2]}\n判定完成。',
    '先说明筛选理由。\n```json\n{"keep": [2]}\n```\n判定完成。',
    '先说明筛选理由。\n```\n{"keep": [2]}\n```',
    '```JSON\n{"keep": [2]}\n```\n判定完成。',
    '\ufeff先说明筛选理由。\n```json\n\ufeff{"keep": [2]}\n```',
])
def test_single_explicit_object_can_have_plain_explanation_and_fences(text):
    assert parse_json_object(text) == {"keep": [2]}


def test_explained_object_preserves_brackets_and_fences_inside_json_strings():
    text = '解释如下。\n```json\n{"reason": "literal ``` and { [ ] }", "keep": [1]}\n```\n完成。'
    assert parse_json_object(text) == {"reason": "literal ``` and { [ ] }", "keep": [1]}


@pytest.mark.parametrize("text", [
    '解释：{"outer": {"keep": [1]}',
    '解释：{broken: {"keep": [1]}}',
    '解释：{broken}\n{"keep": [1]}',
    '解释：{"keep": [1,]}',
    '解释：{"keep": [1]} {"keep": [2]}',
    '{"keep": [1]}\n{"keep": [2]}',
    '解释：{"keep": [1]} 后文还有 [附注]',
    '解释：{"keep": [1]} 后文还有一个 }',
    '解释：[{"keep": [1]}]',
    '[broken, {"keep": [1]}]',
    'null\n{"keep": [1]}',
    'true\n{"keep": [1]}',
    '12\n{"keep": [1]}',
    '"{"keep": [1]}"',
    "'{\"keep\": [1]}'",
    '解释：{"value": NaN}',
    '解释：{"value": Infinity}',
    '解释：{"value": -Infinity}',
    '解释：```python\n{"keep": [1]}\n```',
    '解释：```json\n{"keep": [1]}',
])
def test_fallback_never_selects_nested_ambiguous_or_malformed_json(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


@pytest.mark.parametrize("text", ['{"value": 1e999}', '解释：{"value": -1e999}'])
def test_overflowing_json_float_cannot_create_infinity(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


@pytest.mark.parametrize("text", [
    "", "not JSON", "[]", "null", "true", '"text"', "12",
    "{'keep': [1]}", '{"keep": [1,]}',
    '{"value": NaN}', '{"value": Infinity}',
    '```python\n{"keep": [1]}\n```',
    '```json\n{"keep": [1]}',
])
def test_invalid_json_or_non_object_raises_value_error(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_indices_are_unique_in_model_order_and_lossless_integers():
    data = {"keep": [3, True, 1, 3, 1.0, "2", False, 0, -1, 4, 2]}
    assert selected_indices(data, 3) == [3, 1, 2]


def test_live_model_payload_accepts_canonical_decimal_string_indices():
    data = parse_json_object('{"keep":["1","2","8"],"disputed":[],"mentions":{}}')
    assert selected_indices(data, 8) == [1, 2, 8]


def test_canonical_strings_are_trimmed_deduplicated_and_limited_after_validation():
    data = {"keep": [" 2 ", 2, "1", "3"]}
    assert selected_indices(data, 3, limit=2) == [2, 1]


@pytest.mark.parametrize("value", [
    "", " ", "0", "00", "01", "1.0", "1e0", "-1", "+1",
    "True", "item1", "1 2", "１", "١", "9",
])
def test_noncanonical_or_out_of_range_index_strings_are_rejected(value):
    assert selected_indices({"keep": [value]}, 8) == []


@pytest.mark.parametrize("data", [{}, {"keep": None}, {"keep": 1}, {"keep": "1"}])
def test_missing_or_non_list_indices_select_nothing(data):
    assert selected_indices(data, 3) == []


def test_custom_key_and_limit_apply_after_validation():
    data = {"selected": [99, 2, 2, False, 1, 3]}
    assert selected_indices(data, 3, key="selected", limit=2) == [2, 1]


def test_zero_limit_selects_nothing():
    assert selected_indices({"keep": [1, 2]}, 2, limit=0) == []


def test_no_candidates_selects_nothing():
    assert selected_indices({"keep": [1]}, 0) == []
