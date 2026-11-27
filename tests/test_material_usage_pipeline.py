from src.plugins.chatbot import core
from src.plugins.chatbot.material_usage import MaterialUse, render_material_use
from src.plugins.chatbot.retrieval import RetrievalItem, truncate_by_budget
from src.plugins.chatbot.turn_understanding import parse_understanding
import pytest


def test_generation_receives_each_surviving_source_boundary(monkeypatch):
    monkeypatch.setattr(core, 'load_terms', lambda: [
        {'keyword': '纸鹤', 'category': 'world', 'meaning': '折纸做成的鹤'}])
    uses = [
        MaterialUse('preference', 'craft', 'background', limits=('PREF_SCOPE',)),
        MaterialUse('core_story', 'school', 'past_example', limits=('STORY_SCOPE',)),
        MaterialUse('term', '纸鹤', 'definition', limits=('TERM_SCOPE',)),
        MaterialUse('association', '折纸', 'background', limits=('RELATION_SCOPE',)),
    ]
    messages = core.build_message_list(
        '纸鹤和折纸有什么关系', [], [], material_uses=uses,
        preference_items=[{'id': 'craft', 'category': '手工', 'text': '喜欢折纸'}],
        core_stories=[{'id': 'school', 'category': '学校', 'text': '以前折过纸鹤'}],
        selected_terms={'纸鹤'}, association_context='纸鹤是一种折纸作品',
        association_use_ids=['折纸'])
    text = '\n'.join(m['content'] for m in messages)
    for marker in ('PREF_SCOPE', 'STORY_SCOPE', 'TERM_SCOPE', 'RELATION_SCOPE'):
        assert marker in text
    assert all(word in text for word in ('喜欢折纸', '以前折过纸鹤', '折纸做成的鹤'))


def test_dropped_sources_cannot_leave_usage_notes_in_generation(monkeypatch):
    monkeypatch.setattr(core, 'load_terms', lambda: [
        {'keyword': '纸鹤', 'meaning': '折纸作品'}])
    uses = [MaterialUse(source, 'dropped', purpose, limits=('ORPHAN_' + source,))
            for source, purpose in [('preference', 'background'), ('core_story', 'past_example'),
                                    ('term', 'definition'), ('association', 'background'),
                                    ('corpus', 'past_example')]]
    messages = core.build_message_list('普通聊天', [], [], material_uses=uses,
                                       selected_terms=set(), association_use_ids=[])
    assert 'ORPHAN_' not in '\n'.join(m['content'] for m in messages)


def test_association_usage_is_filtered_by_actual_relation_budget():
    uses = [MaterialUse('association', 'kept', 'background', limits=('KEPT_RELATION',)),
            MaterialUse('association', 'dropped', 'background', limits=('DROPPED_RELATION',))]
    messages = core.build_message_list('聊聊作品', [], [], material_uses=uses,
                                       association_context='留下的关系', association_use_ids=['kept'])
    text = '\n'.join(m['content'] for m in messages)
    assert 'KEPT_RELATION' in text and 'DROPPED_RELATION' not in text


def test_corpus_usage_is_counted_before_budget_and_preserves_canonical_source():
    item = RetrievalItem('corpus', 'old', 1, text='她过去参加过合唱排练。')
    use = MaterialUse('corpus', 'old', 'past_example', fact='她曾参加排练', limits=('OLD_ONLY',))
    annotated = core._corpus_with_uses([item], [use])[0]
    assert annotated is not item and item.text == '她过去参加过合唱排练。'
    assert annotated.extra['source_text'] == item.text
    assert render_material_use(use) in annotated.text and item.text in annotated.text
    assert truncate_by_budget([annotated], budget_chars=len(item.text) + 2) == []
    assert truncate_by_budget([item], budget_chars=len(item.text) + 2) == [item]
    messages = core.build_message_list('你以前参加过排练吗', [], [], material_uses=[use])
    assert 'OLD_ONLY' not in '\n'.join(m['content'] for m in messages)


def test_unapproved_usage_cannot_annotate_a_different_candidate():
    item = RetrievalItem('corpus', 'original', 1, text='原始事实')
    use = MaterialUse('corpus', 'other', 'correction', fact='另一个事实')
    assert core._corpus_with_uses([item], [use]) == [item]


def test_generation_receives_request_exclusions_and_unverified_premise():
    understanding = parse_understanding({
        'focus': 'character', 'focus_text': '角色以前参加评选的经历',
        'request': '核对是否拿过奖', 'exclusions': ['不查询今天的活动'],
        'premises': ['用户以为角色拿过奖，尚未确认'], 'third_parties': [],
    }, '你是不是拿过奖？不是问今天的活动。', '')
    messages = core.build_message_list('你是不是拿过奖', [], [], turn_understanding=understanding)
    text = '\n'.join(m['content'] for m in messages)
    assert all(s in text for s in ('核对是否拿过奖', '不查询今天的活动',
                                   '尚未确认', '待核实前提不能当事实'))


@pytest.mark.parametrize('old_field', ['query_hint', 'pragmatic_hint'])
def test_valid_understanding_replaces_older_interpretation_in_generation(old_field):
    understanding = parse_understanding({
        'focus': 'user', 'focus_text': '用户本人', 'request': '回应用户被朋友放鸽子的抱怨',
        'exclusions': ['不是角色失约'], 'premises': [], 'third_parties': [],
    }, '我朋友放我鸽子，不是在说你', '')
    kwargs = {old_field: 'OLD_WRONG_INTERPRETATION：角色本人失约'}
    messages = core.build_message_list('我朋友放我鸽子，不是在说你', [], [],
                                       turn_understanding=understanding, **kwargs)
    text = '\n'.join(m['content'] for m in messages)
    assert 'OLD_WRONG_INTERPRETATION' not in text
    assert understanding.request in text and messages[-1]['content'] == '我朋友放我鸽子，不是在说你'
    legacy = core.build_message_list('那个呢', [], [], **kwargs)
    assert 'OLD_WRONG_INTERPRETATION' in '\n'.join(m['content'] for m in legacy)


@pytest.mark.asyncio
async def test_same_understanding_reaches_all_material_decisions(monkeypatch):
    understanding = parse_understanding({
        'focus': 'character', 'focus_text': '角色', 'request': '询问过去的排练活动',
        'exclusions': [], 'premises': [], 'third_parties': [],
    }, '你以前参加过排练吗', '')
    seen = {}

    async def classify(*args, **kwargs):
        return {'behavior': '', 'classification_ok': True, 'needs_ok': True,
                'needs': ['corpus'], 'task': '兼容字段里另一份含糊摘要', 'scope': 'conversation',
                'understanding': understanding, 'understanding_ok': True}

    async def embed(*args):
        return [1.0]

    async def materials(context, *args, **kwargs):
        seen['understanding'] = context.understanding
        return {'corpus_items': [], 'preference_items': [], 'core_stories': [],
                'behavior_items': [], 'selection_trace': {'available': True}}

    monkeypatch.setattr(core, 'classify_l3', classify)
    monkeypatch.setattr(core, 'embed_query', embed)
    monkeypatch.setattr(core, 'retrieve_contextual_materials', materials)
    result = await core.gather_retrieval('你以前参加过排练吗', '你以前参加过排练吗', '', None, None)
    assert seen['understanding'] is understanding
    assert result['context'].understanding is understanding
    assert result['context'].response_task == understanding.request


@pytest.mark.asyncio
async def test_handle_chat_forwards_the_confirmed_usage_and_understanding(monkeypatch):
    from src.plugins.chatbot.retrieval_context import RetrievalContext
    understanding = parse_understanding({
        'focus': 'character', 'focus_text': '角色', 'request': '核对以前参加过的活动',
        'exclusions': ['不推断当前安排'], 'premises': [], 'third_parties': [],
    }, '你以前参加过什么活动', '')
    context = RetrievalContext('你以前参加过什么活动', understanding=understanding)
    uses = [MaterialUse('core_story', 'school', 'past_example')]
    for name, value in [('_get_clients', lambda: (None, None)),
                        ('aux_clients_or', lambda fallback: (None, None)),
                        ('injection_v3_enabled', lambda: False),
                        ('_weather_city_for', lambda uid: ''), ('_user_name_for', lambda uid: ''),
                        ('get_llm_route', lambda: ''), ('spawn', lambda coro: coro.close())]:
        monkeypatch.setattr(core, name, value)
    monkeypatch.setattr(core.memory_v2, 'injection_enabled', lambda: False)
    monkeypatch.setattr(core.self_speech, 'enabled', lambda: False)
    monkeypatch.setattr(core, 'LEGENDARY_REPLIES', {})
    monkeypatch.setattr(core.context_probe, 'get_now_context', lambda city='': '')

    async def probe(*args, **kwargs):
        return context.raw_text, {}

    async def retrieve(*args, **kwargs):
        return {'fused_items': [], 'preference_items': [], 'core_stories': [],
                'denied_terms': set(), 'approved_terms': set(), 'context': context,
                'material_uses': uses, 'association_use_ids': ['school']}

    async def events(*args, **kwargs):
        assert kwargs['context'] is context
        return []

    class Rendered(Exception):
        pass

    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        raise Rendered

    monkeypatch.setattr(core, 'probe_session', probe)
    monkeypatch.setattr(core, 'gather_retrieval', retrieve)
    monkeypatch.setattr(core.event_memory, 'retrieve', events)
    monkeypatch.setattr(core, 'build_message_list', capture)
    with pytest.raises(Rendered):
        await core.handle_chat('usage-boundary-forwarding', context.raw_text)
    assert captured['material_uses'] is uses
    assert captured['turn_understanding'] is understanding
    assert captured['association_use_ids'] == ['school']
