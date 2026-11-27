"""Bounded, immutable understanding shared by all material decisions."""
from dataclasses import FrozenInstanceError

import pytest

from src.plugins.chatbot.turn_understanding import (
    GroundedParticipant, TurnFocus, TurnUnderstanding, parse_understanding, render_understanding,
)


def _value(**updates):
    value = {
        'focus': 'character', 'focus_text': '正在与用户交谈的角色',
        'request': '解释提到的称呼，不查询她本人的经历',
        'exclusions': ['她本人的亲历'], 'premises': ['她曾参与该活动（待核实）'],
        'third_parties': [],
    }
    value.update(updates)
    return value


def test_understanding_round_trips_without_losing_exclusions_or_unverified_premises():
    value = _value(third_parties=[{'name': '阿青', 'evidence': '阿青问你去过吗'}])
    result = parse_understanding(value, '阿青问你去过吗，不是问你的亲历，是问称呼含义', '')
    assert isinstance(result, TurnUnderstanding)
    assert result.focus == TurnFocus.CHARACTER and result.focus == 'character'
    assert result.third_parties == (GroundedParticipant('阿青', '阿青问你去过吗'),)
    assert result.to_dict() == value
    with pytest.raises(FrozenInstanceError):
        result.request = '改写任务'
    with pytest.raises(FrozenInstanceError):
        result.third_parties[0].name = '另一个人'


def test_third_party_evidence_can_come_from_history_but_not_join_separate_passages():
    value = _value(third_parties=[{'name': '阿青', 'evidence': '阿青是她的旧室友'}])
    assert parse_understanding(value, '那她后来怎么样了', '用户：阿青是她的旧室友') is not None
    forged = _value(third_parties=[{'name': '阿青', 'evidence': '阿青后来送了礼物'}])
    assert parse_understanding(forged, '后来送了礼物', '阿青') is None


@pytest.mark.parametrize('focus', ['character', 'user', 'third_party', 'shared', 'general', 'unclear'])
def test_focus_enum_accepts_only_declared_values(focus):
    result = parse_understanding(_value(focus=focus), '原话', '')
    assert result is not None and result.to_dict()['focus'] == focus


@pytest.mark.parametrize('updates', [
    {'focus': 'someone'}, {'focus': True}, {'focus_text': ''}, {'focus_text': 1},
    {'request': ''}, {'request': None}, {'exclusions': '不是问经历'},
    {'exclusions': [True]}, {'premises': None}, {'premises': ['']},
    {'third_parties': {}}, {'third_parties': [None]},
    {'third_parties': [{'name': '阿青', 'evidence': ''}]},
    {'third_parties': [{'name': '阿青', 'evidence': '原话没有这个人'}]},
    {'third_parties': [{'name': '捏造的人', 'evidence': '原话'}]},
    {'third_parties': [{'name': True, 'evidence': '原话'}]},
    {'request': '长' * 321}, {'focus_text': '长' * 161},
    {'exclusions': ['长' * 161]}, {'premises': ['前提'] * 7},
    {'third_parties': [{'name': '原话', 'evidence': '原话'}] * 7},
])
def test_invalid_or_unbounded_objects_are_rejected(updates):
    assert parse_understanding(_value(**updates), '原话', '历史') is None


@pytest.mark.parametrize('value', [None, [], 'character', {}, {'focus': 'character'}])
def test_missing_contract_is_not_invented(value):
    assert parse_understanding(value, '原话', '') is None


def test_arrays_are_trimmed_and_deduplicated_without_mutating_model_payload():
    value = _value(exclusions=[' 亲历 ', '亲历'], premises=[' 尚未确认 '])
    result = parse_understanding(value, '原话', '')
    assert result.exclusions == ('亲历',) and result.premises == ('尚未确认',)
    assert value['exclusions'] == [' 亲历 ', '亲历']


def test_generation_note_preserves_boundaries_without_repeating_evidence():
    result = parse_understanding(_value(third_parties=[{
        'name': '阿青', 'evidence': '阿青在私聊里说了一段仅用于核对身份的话',
    }]), '阿青在私聊里说了一段仅用于核对身份的话', '')
    note = render_understanding(result)
    assert all(text in note for text in ('理解参考', '原话优先', '实际需求', '排除范围',
                                         '待核实前提', '不能当事实', '阿青'))
    assert result.third_parties[0].evidence not in note
    assert result.request in note and result.exclusions[0] in note and result.premises[0] in note
    assert render_understanding(None) == ''
