from types import SimpleNamespace

import pytest

from src.plugins.chatbot import core
from src.plugins.chatbot.retrieval import RetrievalItem
from src.plugins.chatbot.retrieval_context import RetrievalContext


@pytest.mark.parametrize('ok,intent,expected_count', [(True, '', 0), (False, '', 0), (True, '模式', 1)])
async def test_l3_decision_and_fallback_share_context(monkeypatch, ok, intent, expected_count):
    context = RetrievalContext('他又鸽了，不是在催你', resolved_text='用户说另一个人失约', history='用户：我在等朋友')
    seen = {}
    monkeypatch.setattr(core, 'load_persona_rules', lambda: [{'name': '模式'}])

    async def classify(client, raw, history, behaviors, **kwargs):
        seen['l3'] = kwargs.get('context')
        return {'behavior': intent, 'same_request': {'what': '', 'count': 1}, 'classification_ok': ok}

    async def embed(client, query):
        seen['embedded'] = query
        return [1.0]

    def behavior(raw, name, behaviors, *, allow_fallback=True):
        seen['fallback_allowed'] = allow_fallback
        return RetrievalItem('behavior', '模式', 1, text='完整样本') if name or allow_fallback else None

    async def retrieve(ctx, vector, client, **kwargs):
        seen['materials'] = ctx
        seen['fallback_candidate'] = kwargs.get('behavior_candidate')
        return {'corpus_items': [], 'preference_items': [], 'core_stories': [], 'behavior_items': [],
                'denied_terms': set(), 'association_context': '', 'association_entities': [],
                'wiki_entities': [], 'entity_mentions': {}, 'selection_trace': {'available': True}}

    monkeypatch.setattr(core, 'classify_l3', classify)
    monkeypatch.setattr(core, 'embed_query', embed)
    monkeypatch.setattr(core, 'select_behavior_item', behavior)
    monkeypatch.setattr(core, 'retrieve_contextual_materials', retrieve)
    result = await core.gather_retrieval(context.raw_text, context.query, context.history, None, None, context=context)
    assert seen['l3'] is context and seen['materials'] is context
    assert seen['embedded'] == context.query
    assert seen['fallback_allowed'] is (not ok)
    assert bool(seen['fallback_candidate']) is (not ok)
    assert len(result['fused_items']) == expected_count
    assert result['embedding_available'] and result['l3']['classification_ok'] is ok


def test_context_approved_referential_term_is_rendered_without_raw_keyword(monkeypatch):
    monkeypatch.setattr(core, 'load_terms', lambda: [
        {'keyword': '考', 'meaning': '特定圈内义项', 'usage': '相应表达规则'},
        {'keyword': '养', 'meaning': '另一个义项'},
    ])
    hits = []
    note = core.build_terms_note('那个词是什么意思', selected_terms={'考'}, hit_out=hits)
    assert '特定圈内义项' in note and '另一个义项' not in note
    assert hits == ['考']


@pytest.mark.parametrize('query,confirmed,expected', [
    ('我今天16岁了，祝我生日快乐', False, '这是正常生成的回答'),
    ('电影里这人几岁啊', False, '这是正常生成的回答'),
    ('你到底几岁啊', True, '永远16岁'),
])
async def test_age_route_never_bypasses_context_confirmation(monkeypatch, query, confirmed, expected):
    seen = {}
    monkeypatch.setattr(core, '_get_clients', lambda: (None, None))
    monkeypatch.setattr(core, 'aux_clients_or', lambda fallback: (None, None))
    monkeypatch.setattr(core, 'injection_v3_enabled', lambda: False)
    monkeypatch.setattr(core.memory_v2, 'injection_enabled', lambda: False)
    monkeypatch.setattr(core.self_speech, 'enabled', lambda: False)
    monkeypatch.setattr(core, '_weather_city_for', lambda uid: '')
    monkeypatch.setattr(core, '_user_name_for', lambda uid: '')
    monkeypatch.setattr(core, 'get_llm_route', lambda: '')
    monkeypatch.setattr(core, 'LEGENDARY_REPLIES', {'年龄': {'pattern': '几岁|年龄|16岁', 'replies': ['永远16岁']}})
    monkeypatch.setattr(core, 'LEGENDARY_CONFIRMS', {})
    monkeypatch.setattr(core.context_probe, 'get_now_context', lambda city='': '')
    monkeypatch.setattr(core, 'spawn', lambda coro: coro.close())

    async def probe(*args, **kwargs):
        return query, {'target': '本轮实际对象', 'speech_act': 'literal_question'}

    async def confirm(raw, template, history='', context=None):
        seen['confirm'] = context
        return confirmed

    async def retrieve(raw, resolved, history, *clients, context=None, **kwargs):
        seen['retrieve'] = context
        return {'fused_items': [], 'preference_items': [], 'core_stories': [],
                'denied_terms': set(), 'approved_terms': set()}

    async def events(*args, **kwargs):
        seen['events'] = kwargs.get('context')
        return []

    async def no_weather(*args, **kwargs):
        return None

    async def reply(messages):
        return '这是正常生成的回答'

    monkeypatch.setattr(core, 'probe_session', probe)
    monkeypatch.setattr(core, 'legendary_confirmed', confirm)
    monkeypatch.setattr(core, 'gather_retrieval', retrieve)
    monkeypatch.setattr(core.event_memory, 'retrieve', events)
    monkeypatch.setattr(core.context_probe, 'warm_weather', no_weather)
    monkeypatch.setattr(core, 'generate_reply', reply)
    assert await core.handle_chat('context-case', query) == expected
    assert seen['confirm'].raw_text == query
    if not confirmed:
        assert seen['retrieve'] is seen['events'] is seen['confirm']


async def test_broadcast_keeps_public_content_materials_without_reply_context_gate(monkeypatch):
    async def embed(client, query):
        return [1.0]
    async def unexpected(*args, **kwargs):
        raise AssertionError('broadcast must not use user-reply contextual decisions')
    monkeypatch.setattr(core, 'embed_query', embed)
    monkeypatch.setattr(core, 'classify_l3', unexpected)
    monkeypatch.setattr(core, 'retrieve_contextual_materials', unexpected)
    monkeypatch.setattr(core, 'retrieve_preferences', lambda query, **kwargs: [{'id': 'food', 'text': '公共人格偏好'}])
    monkeypatch.setattr(core, 'retrieve_core_stories', lambda query, vector: [{'id': 'public', 'text': '公共往事'}])
    result = await core.gather_retrieval('她自己的动态', '她自己的动态', '', None, None, is_user_msg=False)
    assert result['preference_items'][0]['id'] == 'food'
    assert result['core_stories'][0]['id'] == 'public'
    assert not result.get('denied_terms') and result.get('approved_terms') is None


@pytest.mark.parametrize('needs_ok,needs,expected', [(True, ['preference'], ('preference',)),
                                                  (False, ['corpus'], ())])
async def test_l3_needs_are_shared_before_candidate_selection(monkeypatch, needs_ok, needs, expected):
    seen = {}
    async def classify(*args, **kwargs):
        return {'behavior': '', 'classification_ok': True, 'needs': needs, 'needs_ok': needs_ok,
                'task': '本轮独立理解的任务', 'scope': 'general_task'}
    async def embed(*args):
        return [1.0]
    async def retrieve(context, *args, **kwargs):
        seen['context'] = context
        return {'corpus_items': [], 'behavior_items': [], 'preference_items': [], 'core_stories': []}
    monkeypatch.setattr(core, 'classify_l3', classify)
    monkeypatch.setattr(core, 'embed_query', embed)
    monkeypatch.setattr(core, 'retrieve_contextual_materials', retrieve)
    result = await core.gather_retrieval('本轮原话', '本轮完整含义', '前文', None, None)
    assert seen['context'].needs == expected
    assert seen['context'].response_task == ('本轮独立理解的任务' if needs_ok else '')
    assert seen['context'].response_scope == ('general_task' if needs_ok else '')
    assert result['context'] is seen['context']


@pytest.mark.parametrize('confirmed_behavior', [False, True])
async def test_retrieval_exception_clears_material_permission_for_events_and_private_hints(monkeypatch, confirmed_behavior):
    from src.plugins.chatbot import event_memory, memory_context

    context = RetrievalContext('我没养过猫，你记错了')
    approved_behavior = RetrievalItem('behavior', '模式', 1, text='已独立确认的行为')
    unverified_behavior = RetrievalItem('behavior', '其他', 1, text='未经确认的行为')

    async def broken(result, *args, **kwargs):
        result.update(
            l3={'classification_ok': confirmed_behavior, 'behavior': '模式'},
            fused_items=[approved_behavior, unverified_behavior, RetrievalItem('corpus', 'old', 1, text='旧经历')],
            preference_items=[{'id': 'food', 'text': '偏好'}], core_stories=[{'id': 'story', 'text': '往事'}],
            approved_terms={'词'}, association_context='关联背景', association_entities=['实体'],
            wiki_entities=['百科实体'], entity_mentions={'百科实体': '简称'},
        )
        raise RuntimeError('failure before retrieval completes')

    class NoDecision:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise AssertionError('a retrieval exception must not let candidates invent permission')

    monkeypatch.setattr(core, '_fill_retrieval', broken)
    result = await core.gather_retrieval(context.raw_text, context.query, '', None, None, context=context)
    assert result['context'].needs == ()
    assert result['fused_items'] == ([approved_behavior] if confirmed_behavior else [])
    assert result['approved_terms'] == set()
    assert all(not result[key] for key in ('preference_items', 'core_stories', 'association_context',
                                          'association_entities', 'wiki_entities', 'entity_mentions'))
    assert not result['selection_trace']['available']

    event = {'id': 'old-cat', 'what': '用户以前说养猫', 'entities': ['猫'], 'date': '2026-11-01'}
    monkeypatch.setattr(event_memory, '_active_events', lambda *args, **kwargs: [event])
    monkeypatch.setattr(event_memory, '_load', lambda *args, **kwargs: {})
    client = NoDecision()
    assert await event_memory.retrieve('offline-user', context.raw_text, entities=['猫'],
                                      context=result['context'], judge_client=client) == []
    assert await memory_context.filter_memory_hints(client, result['context'],
                                                    {'habit_hint': '【习惯提示】用户养猫'}) == {}
    assert client.calls == 0
