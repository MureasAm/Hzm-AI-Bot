import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot.retrieval_context import (
    MaterialCandidate, RetrievalContext, select_materials,
)


class Client:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.content, Exception):
            raise self.content
        try:
            payload = json.loads(kwargs['messages'][-1]['content'])
        except ValueError:
            payload = {}
        content = (json.dumps({'checks': [{'index': row['index'], 'entailed': True, 'answers': True}
                                         for row in payload['待核对事实']]})
                   if '待核对事实' in payload else self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)


def test_shared_context_preserves_original_and_reference_interpretation():
    ctx = RetrievalContext.from_probe(
        '我只是引用“你几岁”，不是问你', '用户在引用别人的年龄问题',
        '用户：电影里的人物看起来很年轻',
        {'target': '电影人物', 'speech_act': 'factual_statement', 'pragmatic_confidence': .9},
        is_group=True, addressed=False,
    )
    assert ctx.raw_text == '我只是引用“你几岁”，不是问你'
    assert ctx.query == '用户在引用别人的年龄问题'
    note = ctx.note()
    assert all(text in note for text in ('不是问你', '电影人物', '电影里的人物', 'factual_statement'))
    assert ctx.is_group and not ctx.addressed


def test_independent_task_description_is_shared_with_material_decisions():
    ctx = RetrievalContext('这个应该怎么处理', response_task='用户在咨询通用操作方法，不是在问她的经历',
                           response_scope='general_task')
    assert json.loads(ctx.note())['独立任务理解'] == ctx.response_task
    assert json.loads(ctx.note())['回答范围参考'] == 'general_task'


async def test_one_selection_distinguishes_channels_and_uses_full_context():
    ctx = RetrievalContext('那后来呢', resolved_text='追问灰泽满室友后来如何', history='用户：你室友上次那件事呢')
    client = Client('{"keep":[2,3,4],"disputed":[]}')
    candidates = [MaterialCandidate('preference', 'food', '她喜欢甜食'),
                  MaterialCandidate('corpus', '42', '她的室友向她问起醉酒'),
                  MaterialCandidate('core_story', 'school', '她当时上学的经历'),
                  MaterialCandidate('term', '女室友', '女室友的圈内含义')]
    selected = await select_materials(client, ctx, candidates, model='fake')
    assert selected.available
    assert selected.selected == frozenset({('corpus', '42'), ('core_story', 'school'), ('term', '女室友')})
    assert len(client.calls) == 1
    assert client.calls[0]['response_format'] == {'type': 'json_object'}
    prompt = json.dumps(client.calls[0]['messages'], ensure_ascii=False)
    assert all(text in prompt for text in ('那后来呢', '追问灰泽满室友后来如何', '上次那件事', 'preference', 'corpus'))


async def test_limits_and_disputed_evidence_are_validated_per_source():
    client = Client('```json\n{"keep":[true,0,1,1,2,3,4,5,6,99],"disputed":[1,3,4]}\n```')
    candidates = [MaterialCandidate('event', 'old-cat', '用户以前说养猫'),
                  MaterialCandidate('core_story', 'a', '经历甲'), MaterialCandidate('core_story', 'b', '经历乙'),
                  MaterialCandidate('corpus', 'a', '背景甲'), MaterialCandidate('corpus', 'b', '背景乙'),
                  MaterialCandidate('corpus', 'c', '背景丙')]
    selected = await select_materials(client, RetrievalContext('我没养过猫，你记错了'), candidates, model='fake')
    assert selected.selected == frozenset({('event', 'old-cat'), ('core_story', 'a'), ('corpus', 'a'), ('corpus', 'b')})
    assert selected.disputed == frozenset({('event', 'old-cat')})


async def test_entity_surface_mapping_must_be_anchored_in_raw_message():
    client = Client('{"keep":[1,2],"mentions":{"东野圭吾":"东野","初音未来":"另一个名字"}}')
    selected = await select_materials(client, RetrievalContext('东野写过什么'), [
        MaterialCandidate('wiki', '东野圭吾', '作者东野圭吾'),
        MaterialCandidate('wiki', '初音未来', '虚拟歌手初音未来')], model='fake')
    assert selected.entity_mentions == {'东野圭吾': '东野'}


@pytest.mark.parametrize('content', [RuntimeError('offline'), '{"keep":[', '[]', '{"wrong":[]}',
                                      '{"keep":[true,99,"unknown"]}'])
async def test_unusable_decision_does_not_authorize_keyword_candidates(content):
    selected = await select_materials(Client(content), RetrievalContext('这句话狗屁不通'), [
        MaterialCandidate('preference', 'animals', '她喜欢动物')], model='fake')
    assert not selected.available
    assert not selected.selected


async def test_empty_pool_needs_no_model_and_is_not_a_failure():
    client = Client(AssertionError('must not call'))
    selected = await select_materials(client, RetrievalContext('你好'), [], model='fake')
    assert selected.available and not selected.selected and not client.calls


async def test_disputed_event_alone_is_valid_evidence_not_silently_lost():
    selected = await select_materials(Client('{"keep":[],"disputed":[1]}'),
        RetrievalContext('我没养过猫'), [MaterialCandidate('event', 'old', '用户以前说养猫')], model='fake')
    assert selected.selected == selected.disputed == frozenset({('event', 'old')})


@pytest.mark.parametrize('disputed', [None, '1', 1, True, {}, {'index': 1}])
async def test_malformed_disputed_field_cannot_downgrade_an_event_to_confirmed_fact(disputed):
    response = json.dumps({'keep': [1], 'disputed': disputed})
    selected = await select_materials(Client(response), RetrievalContext('这件事你记错了'), [
        MaterialCandidate('event', 'old', '用户以前说养猫')], model='fake')
    assert not selected.available
    assert not selected.selected and not selected.disputed


@pytest.mark.parametrize('disputed', [[True], ['unknown'], [99], [0, -1]])
async def test_nonempty_disputed_with_no_valid_event_index_is_unusable(disputed):
    response = json.dumps({'keep': [1], 'disputed': disputed})
    selected = await select_materials(Client(response), RetrievalContext('这件事你记错了'), [
        MaterialCandidate('event', 'old', '用户以前说养猫')], model='fake')
    assert not selected.available and not selected.selected


async def test_legacy_event_response_without_disputed_field_remains_compatible():
    selected = await select_materials(Client('{"keep":[1]}'), RetrievalContext('还记得那次吗'), [
        MaterialCandidate('event', 'old', '用户以前说养猫')], model='fake')
    assert selected.available and selected.selected == frozenset({('event', 'old')})
    assert not selected.disputed


async def test_independent_needs_prevent_candidates_from_redefining_the_task():
    client = Client('{"keep":[1]}')
    context = RetrievalContext('只问她喜欢吃什么', needs=('preference',))
    selected = await select_materials(client, context, [
        MaterialCandidate('corpus', 'old', '她过去生病吃药的经历'),
        MaterialCandidate('preference', 'food', '她的食物偏好')], model='fake')
    assert selected.selected == frozenset({('preference', 'food')})
    offered = json.loads(client.calls[0]['messages'][-1]['content'])['候选材料']
    assert [candidate['来源'] for candidate in offered] == ['preference']


async def test_explicit_no_material_need_requires_no_candidate_judge():
    client = Client(AssertionError('must not let offered stories invent a need'))
    selected = await select_materials(client, RetrievalContext('请改写这句引文', needs=()), [
        MaterialCandidate('corpus', 'old', '她过去的经历'),
        MaterialCandidate('memory_hint', 'ritual', '用户的晚安约定')], model='fake')
    assert selected.available and not selected.selected and not client.calls


class ResponseSequence(Client):
    def __init__(self, responses):
        super().__init__('')
        self.responses = responses

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)


@pytest.mark.parametrize('decision', [
    {'keep': [1]},
    {'keep': [1], 'evidence': []},
    {'keep': [1], 'evidence': [{'index': 1, 'fact': '她见过海豹', 'quote': '她见过活海豹', 'relation': 'direct'}]},
    {'keep': [1], 'evidence': [{'index': 1, 'fact': '', 'quote': '她在码头看过海豹', 'relation': 'direct'}]},
])
async def test_corpus_without_valid_source_evidence_cannot_authorize_a_fact(decision):
    selected = await select_materials(Client(json.dumps(decision)), RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'seal', '她在码头看过海豹。')], model='fake', require_evidence=True)
    assert not selected.available and not selected.selected


@pytest.mark.parametrize('query,text,fact,quote', [
    ('你见过海豹吗', '她在码头看过一只海豹。', '她看过海豹', '在码头看过一只海豹'),
    ('你学过钢琴吗', '她说自己从来没学过钢琴。', '她没有学过钢琴', '从来没学过钢琴'),
    ('你和室友关系怎么样', '她此前说室友是多年的好朋友。', '她此前称室友是多年好友', '室友是多年的好朋友'),
    ('你有多高', '她被问到身高时自报一米六。', '她此前自报身高一米六', '自报一米六'),
])
async def test_corpus_direct_evidence_can_support_recorded_positive_or_negative_fact(query, text, fact, quote):
    client = Client(json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': fact, 'quote': quote, 'relation': 'direct'}]}))
    selected = await select_materials(client, RetrievalContext(query), [
        MaterialCandidate('corpus', 'fact', text)], model='fake', require_evidence=True)
    assert selected.available and selected.selected == frozenset({('corpus', 'fact')})
    assert selected.evidence[('corpus', 'fact')]['quote'] == quote
    assert len(client.calls) == selected.attempts == 2


@pytest.mark.parametrize('query,text,fact,quote,relation', [
    ('你见过海豹吗', '她闻过朋友买的海豹玩偶。', '她闻过玩偶', '闻过朋友买的海豹玩偶', 'topic_only'),
    ('你在音乐节上唱过歌吗', '她去参加了音乐节。', '她没有在音乐节上唱歌', '参加了音乐节', 'unsupported'),
])
async def test_shared_object_word_or_unrecorded_negation_is_not_direct_fact_support(query, text, fact, quote, relation):
    client = Client(json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': fact, 'quote': quote, 'relation': relation}]}))
    selected = await select_materials(client, RetrievalContext(query), [
        MaterialCandidate('corpus', 'nearby', text)], model='fake', require_evidence=True)
    assert selected.available and not selected.selected
    assert len(client.calls) == 1


async def test_evidence_quote_cannot_be_borrowed_from_another_candidate():
    selected = await select_materials(Client(json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': '她见过海豹', 'quote': '她见过一只海豹', 'relation': 'direct'}]})),
        RetrievalContext('你见过海豹吗'), [MaterialCandidate('corpus', 'toy', '她买过海豹玩偶。'),
                                         MaterialCandidate('corpus', 'animal', '她见过一只海豹。')],
        model='fake', require_evidence=True)
    assert not selected.available and not selected.selected


@pytest.mark.parametrize('quote', ['她在码头见过海豹', '原文后半段才有这句话'])
async def test_quote_must_be_contiguous_in_the_text_actually_offered_to_the_model(quote):
    text = '她在\n码头见过海豹。' + '填' * 650 + '原文后半段才有这句话'
    decision = {'keep': [1], 'evidence': [
        {'index': 1, 'fact': '她见过海豹', 'quote': quote, 'relation': 'direct'}]}
    selected = await select_materials(Client(json.dumps(decision)), RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'seal', text)], model='fake', require_evidence=True)
    assert not selected.available and not selected.selected


async def test_duplicate_evidence_cannot_choose_between_conflicting_relations():
    decision = {'keep': [1], 'evidence': [
        {'index': 1, 'fact': '她买过玩偶', 'quote': '她买过海豹玩偶', 'relation': 'topic_only'},
        {'index': '1', 'fact': '她见过活海豹', 'quote': '她买过海豹玩偶', 'relation': 'direct'}]}
    selected = await select_materials(Client(json.dumps(decision)), RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'seal', '她买过海豹玩偶。')], model='fake', require_evidence=True)
    assert not selected.available and not selected.selected


async def test_canonical_string_index_is_bound_to_the_same_evidence_source():
    decision = {'keep': ['1'], 'evidence': [
        {'index': '1', 'fact': '她见过海豹', 'quote': '她见过海豹', 'relation': 'direct'}]}
    selected = await select_materials(Client(json.dumps(decision)), RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'seal', '她见过海豹。')], model='fake', require_evidence=True)
    assert selected.available and selected.selected == frozenset({('corpus', 'seal')})


async def test_schema_repair_is_bounded_and_each_response_is_metered(monkeypatch):
    from src.plugins.chatbot import retrieval_context as context_module
    usage = []
    monkeypatch.setattr(context_module, 'log_cache_usage', lambda response, tag: usage.append(tag))
    valid = json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': '她见过海豹', 'quote': '她见过海豹', 'relation': 'direct'}]})
    client = ResponseSequence(['{"keep": [1]}', valid,
                               '{"checks":[{"index":1,"entailed":true,"answers":true}]}'])
    selected = await select_materials(client, RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'seal', '她见过海豹。')], model='fake', require_evidence=True)
    assert selected.available and selected.selected
    assert len(client.calls) == len(usage) == selected.attempts == 3
    assert '补救' in usage[1] and '事实复核' in usage[2]


async def test_bad_schema_stops_after_one_repair_and_valid_empty_never_retries():
    bad = ResponseSequence(['{"keep":"1"}'])
    candidates = [MaterialCandidate('corpus', 'seal', '她见过海豹。')]
    failed = await select_materials(bad, RetrievalContext('你见过海豹吗'), candidates,
                                    model='fake', require_evidence=True)
    assert not failed.available and not failed.selected and len(bad.calls) == 2
    empty = ResponseSequence(['{"keep":[]}'])
    selected = await select_materials(empty, RetrievalContext('你见过海豹吗'), candidates,
                                      model='fake', require_evidence=True)
    assert selected.available and not selected.selected and len(empty.calls) == 1


async def test_preferences_can_support_recommendations_without_corpus_fact_evidence():
    selected = await select_materials(Client('{"keep":[1]}'), RetrievalContext('按你的口味推荐一本小说'), [
        MaterialCandidate('preference', 'genre', '她喜欢悬疑推理小说。')], model='fake')
    assert selected.available and selected.selected == frozenset({('preference', 'genre')})


class RawResponseSequence(Client):
    def __init__(self, responses):
        super().__init__('')
        self.responses = responses

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize('first_response', [
    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=''))]),
    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
    SimpleNamespace(choices=None), SimpleNamespace(choices=[]), SimpleNamespace(choices=[None]),
])
async def test_unusable_response_body_gets_at_most_one_format_repair(first_response):
    valid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
        'keep': [1], 'evidence': [{'index': 1, 'fact': '她见过海豹',
                                 'quote': '她见过海豹', 'relation': 'direct'}]})))])
    reviewed = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"checks":[{"index":1,"entailed":true,"answers":true}]}'))])
    client = RawResponseSequence([first_response, valid, reviewed])
    selected = await select_materials(client, RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'seal', '她见过海豹。')], model='fake', require_evidence=True)
    assert selected.available and selected.selected == frozenset({('corpus', 'seal')})
    assert selected.attempts == len(client.calls) == 3


async def test_repeated_missing_body_stops_after_one_repair_but_api_error_does_not_retry():
    candidates = [MaterialCandidate('preference', 'genre', '她喜欢推理小说')]
    bad = RawResponseSequence([SimpleNamespace(choices=None)])
    result = await select_materials(bad, RetrievalContext('按你的口味推荐小说'), candidates, model='fake')
    assert not result.available and not result.selected
    assert result.attempts == len(bad.calls) == 2
    offline = RawResponseSequence([RuntimeError('HTTP 503'), SimpleNamespace(choices=[])])
    result = await select_materials(offline, RetrievalContext('按你的口味推荐小说'), candidates, model='fake')
    assert not result.available and result.attempts == len(offline.calls) == 1


@pytest.mark.parametrize('query,text,fact,entailed,answers', [
    ('你今天和室友见面了吗', '她此前说室友是多年的好朋友。', '她此前说室友是多年好友', True, False),
    ('你自己见过海豹吗', '她听朋友讲过在港口见海豹的事。', '她自己没有见过海豹', False, False),
])
async def test_independent_review_can_remove_direct_labeled_fact_with_genuine_quote(query, text, fact, entailed, answers):
    initial = json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': fact, 'quote': text, 'relation': 'direct'}]})
    review = json.dumps({'checks': [{'index': 1, 'entailed': entailed, 'answers': answers}]})
    client = ResponseSequence([initial, review])
    selected = await select_materials(client, RetrievalContext(query), [
        MaterialCandidate('corpus', 'fact', text)], model='fake', require_evidence=True)
    assert selected.available and not selected.selected and not selected.evidence
    assert selected.attempts == len(client.calls) == 2
    assert selected.verification[('corpus', 'fact')] == {'entailed': entailed, 'answers': answers}
    second = client.calls[1]['messages']
    assert len(second) == 2 and all(message['role'] != 'assistant' for message in second)
    payload = json.loads(second[-1]['content'])
    assert payload['原话'] == query
    assert payload['待核对事实'][0]['fact'] == fact and payload['待核对事实'][0]['原文'] == text
    assert 'relation' not in json.dumps(payload)


async def test_review_preserves_supported_limited_answer_and_does_not_rewrite_fact(monkeypatch):
    from src.plugins.chatbot import retrieval_context as module
    usage = []
    monkeypatch.setattr(module, 'log_cache_usage', lambda response, tag: usage.append(tag))
    source = '她说那次没有登台唱歌。'
    fact = '她此前说那次没有登台唱歌'
    initial = json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': fact, 'quote': source, 'relation': 'direct'}]})
    client = ResponseSequence([initial, '{"checks":[{"index":1,"entailed":true,"answers":true,"fact":"新增的事"}]}'])
    selected = await select_materials(client, RetrievalContext('那次你上台唱了哪首歌'), [
        MaterialCandidate('corpus', 'fact', source)], model='fake', require_evidence=True)
    assert selected.selected == frozenset({('corpus', 'fact')})
    assert selected.evidence[('corpus', 'fact')]['fact'] == fact
    assert selected.attempts == len(usage) == 2 and '事实复核' in usage[-1]


@pytest.mark.parametrize('review', [
    {}, {'checks': []}, {'checks': {}},
    {'checks': [{'index': 1, 'entailed': 'true', 'answers': True}]},
    {'checks': [{'index': 1, 'entailed': 1, 'answers': True}]},
    {'checks': [{'index': 1, 'entailed': True}]},
    {'checks': [{'index': True, 'entailed': True, 'answers': True}]},
    {'checks': [{'index': 1.0, 'entailed': True, 'answers': True}]},
    {'checks': [{'index': 2, 'entailed': True, 'answers': True}]},
    {'checks': [{'index': 1, 'entailed': True, 'answers': True},
                {'index': '1', 'entailed': False, 'answers': False}]},
])
async def test_invalid_review_cannot_add_facts_skip_items_or_retry(review):
    initial = json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': '她见过海豹', 'quote': '她见过海豹', 'relation': 'direct'}]})
    client = ResponseSequence([initial, json.dumps(review)])
    selected = await select_materials(client, RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'first', '她见过海豹。'),
        MaterialCandidate('corpus', 'not-selected', '她还见过鲸鱼。')], model='fake', require_evidence=True)
    assert not selected.available and not selected.selected
    assert selected.attempts == len(client.calls) == 2


async def test_review_http_error_does_not_start_another_call():
    initial = json.dumps({'keep': [1], 'evidence': [
        {'index': 1, 'fact': '她见过海豹', 'quote': '她见过海豹', 'relation': 'direct'}]})
    client = ResponseSequence([initial, RuntimeError('HTTP 503')])
    selected = await select_materials(client, RetrievalContext('你见过海豹吗'), [
        MaterialCandidate('corpus', 'seal', '她见过海豹。')], model='fake', require_evidence=True)
    assert not selected.available and not selected.selected
    assert selected.attempts == len(client.calls) == 2


async def test_review_must_cover_every_selected_item_and_keep_original_indices():
    decision = {'keep': [1, 2], 'evidence': [
        {'index': 1, 'fact': '她去过北京', 'quote': '她去过北京', 'relation': 'direct'},
        {'index': 2, 'fact': '她去过上海', 'quote': '她去过上海', 'relation': 'direct'}]}
    candidates = [MaterialCandidate('corpus', 'beijing', '她去过北京。'),
                  MaterialCandidate('corpus', 'shanghai', '她去过上海。')]
    client = ResponseSequence([json.dumps(decision), '{"checks":[{"index":1,"entailed":true,"answers":true}]}'])
    selected = await select_materials(client, RetrievalContext('你去过哪些城市'), candidates,
                                      model='fake', require_evidence=True)
    assert not selected.available and not selected.selected and len(client.calls) == 2
    decision['keep'] = [2]
    decision['evidence'] = decision['evidence'][1:]
    client = ResponseSequence([json.dumps(decision), '{"checks":[{"index":"2","entailed":true,"answers":true}]}'])
    selected = await select_materials(client, RetrievalContext('你去过哪些城市'), candidates,
                                      model='fake', require_evidence=True)
    assert selected.available and selected.selected == frozenset({('corpus', 'shanghai')})
