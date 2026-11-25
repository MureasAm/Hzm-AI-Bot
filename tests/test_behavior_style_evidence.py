"""行为可以用有来源的短语示范语气，而不把整段同题答案交给生成器。"""
from copy import deepcopy

from src.plugins.chatbot.persona import _format_behavior_rule


def _rule():
    return {'name': '失约时', 'trigger': '对方催约定', 'response': '嘴硬后认下',
            'samples': [{'user': '昨天没有通知呀', 'reply': '应该提前说的，对不起对不起。'},
                        {'user': '这周还播吗', 'reply': '播的播的，别催了嘛。'}]}


def test_short_style_evidence_keeps_source_voice_without_complete_answers():
    rule = _rule()
    rule['style_fragments'] = ['对不起对不起。', '别催了嘛。']
    before = deepcopy(rule)
    text = _format_behavior_rule(rule)
    assert all(s in text for s in rule['style_fragments'])
    assert rule['response'] in text and rule['trigger'] in text
    assert not any(s['reply'] in text or s['user'] in text for s in rule['samples'])
    assert rule == before


def test_unconfigured_behaviors_keep_existing_real_question_answer_examples():
    rule = _rule()
    text = _format_behavior_rule(rule)
    assert all(s['reply'] in text and s['user'] in text for s in rule['samples'])


def test_style_evidence_needs_an_exact_source_and_cannot_be_a_whole_answer():
    rule = _rule()
    rule['style_fragments'] = ['凭空编的口头禅', rule['samples'][0]['reply'], '对不起对不起。',
                               '对不起对不起。', {'text': '假字段'}, '']
    text = _format_behavior_rule(rule)
    assert text.count('对不起对不起。') == 1
    assert '凭空编的' not in text and '应该提前说的' not in text and '假字段' not in text


def test_invalid_explicit_evidence_keeps_strategy_without_falling_back_to_the_full_template():
    rule = _rule(); rule['style_fragments'] = ['无来源的台词']
    text = _format_behavior_rule(rule)
    assert rule['response'] in text
    assert not any(s['reply'] in text for s in rule['samples'])


def test_configured_fragments_are_bounded_and_keep_source_order():
    rule = _rule(); rule['samples'] = [{'user': '以前说什么', 'reply': '甲片段，乙片段，丙片段，丁片段，戊片段。'}]
    rule['style_fragments'] = ['乙片段', '甲片段', '丙片段', '丁片段', '戊片段']
    text = _format_behavior_rule(rule)
    assert text.index('乙片段') < text.index('甲片段')
    assert '丁片段' in text and '戊片段' not in text
