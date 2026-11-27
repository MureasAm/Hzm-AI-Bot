import asyncio
import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import material_retrieval as materials
from src.plugins.chatbot.retrieval import RetrievalItem
from src.plugins.chatbot.retrieval_context import RetrievalContext


class ChoosingClient:
    def __init__(self, allowed):
        self.allowed = allowed
        self.seen = []
        self.requests = []
        self.verifications = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs['messages'][-1]['content'])
        if '待核对事实' in payload:
            self.verifications.append(payload)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                'checks': [{'index': row['index'], 'entailed': True, 'answers': True}
                           for row in payload['待核对事实']]})))], usage=None)
        self.seen.append(payload)
        keep = [c['编号'] for c in payload['候选材料'] if (c['来源'], c['id']) in self.allowed]
        evidence = [{'index': c['编号'], 'fact': c['内容'], 'quote': c['内容'], 'relation': 'direct'}
                    for c in payload['候选材料'] if c['编号'] in keep and c['来源'] == 'corpus']
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'keep': keep, 'evidence': evidence})))], usage=None)


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(materials, 'judge_enabled', lambda: True)
    monkeypatch.setattr(materials, 'retrieve_corpus_candidates', lambda q, v: [RetrievalItem('corpus', 'past', .8, text='她过去养猫')])
    monkeypatch.setattr(materials, 'retrieve_core_stories', lambda q, v: [{'id': 'story', 'text': '她的一段往事'}])
    monkeypatch.setattr(materials, 'load_preferences', lambda: [{'id': 'animals'}])
    monkeypatch.setattr(materials, 'retrieve_preferences', lambda *a, **k: [{'id': 'animals', 'category': '动物', 'text': '她喜欢猫'}])
    monkeypatch.setattr(materials, 'term_candidates', lambda q: [{'keyword': '猫', 'category': 'world', 'meaning': '圈内词义', 'usage_triggers': ['宠物话题']}])
    monkeypatch.setattr(materials, 'load_terms', lambda: [{'keyword': '猫'}, {'keyword': '另一个词'}])
    monkeypatch.setattr(materials, 'match_entities', lambda *a: ['书名'])
    monkeypatch.setattr(materials, 'resolve_associations', lambda *a: {'entities': ['书名', '作者'], 'note': '书名由作者创作', 'preference_ids': []})


async def test_keyword_hits_from_every_source_need_context_approval(pool):
    client = ChoosingClient(set())
    result = await materials.retrieve_contextual_materials(RetrievalContext('只是引用猫这个字'), [1], client)
    assert not result['corpus_items'] and not result['preference_items'] and not result['core_stories']
    assert result['denied_terms'] == {'猫', '另一个词'}
    assert not result['association_context'] and not result['wiki_entities']
    assert len(client.seen) == 2
    batches = [{c['来源'] for c in payload['候选材料']} for payload in client.seen]
    assert {'corpus'} in batches
    assert {'preference', 'core_story', 'term', 'association'} in batches


async def test_positive_context_keeps_only_approved_sources(pool):
    client = ChoosingClient({('preference', 'animals'), ('term', '猫'), ('association', '书名')})
    result = await materials.retrieve_contextual_materials(RetrievalContext('你喜欢猫吗'), [1], client)
    assert [p['id'] for p in result['preference_items']] == ['animals']
    assert result['denied_terms'] == {'另一个词'}
    assert result['association_context'] == '书名由作者创作'
    assert not result['core_stories'] and not result['corpus_items']
    candidates = [candidate for payload in client.seen for candidate in payload['候选材料']]
    assert next(c for c in candidates if c['来源'] == 'preference')['属性']['category'] == '动物'
    assert next(c for c in candidates if c['来源'] == 'term')['属性']['usage_triggers'] == ['宠物话题']
    assert next(c for c in candidates if c['来源'] == 'term')['属性']['category'] == 'world'


async def test_failed_gate_cannot_promote_local_fallbacks(pool):
    fallback = RetrievalItem('behavior', '催播', 1, text='认栽')
    result = await materials.retrieve_contextual_materials(RetrievalContext('他又鸽了，不是在催你'), None, None,
                                                          behavior_candidate=fallback)
    assert not result['behavior_items'] and not result['preference_items']
    assert not result['selection_trace']['available']


async def test_explicit_entities_need_selection_before_wiki_lookup(pool):
    client = ChoosingClient({('wiki', '初音未来')})
    result = await materials.retrieve_contextual_materials(
        RetrievalContext('初音未来是谁', entities=('初音未来',)), [1], client)
    assert result['wiki_entities'] == ['初音未来']


async def test_proactive_materials_do_not_use_user_experience_criterion(pool, monkeypatch):
    def unexpected(*args):
        raise AssertionError('proactive content is not a user query about her past')
    monkeypatch.setattr(materials, 'retrieve_corpus_candidates', unexpected)
    client = ChoosingClient({('preference', 'animals')})
    result = await materials.retrieve_contextual_materials(
        RetrievalContext('她发布了猫的照片', is_user_msg=False), [1], client)
    assert result['preference_items'] and not result['corpus_items']


async def test_preference_only_relation_still_supplies_candidates(pool, monkeypatch):
    monkeypatch.setattr(materials, 'resolve_associations', lambda *a: {
        'entities': [], 'note': '', 'preference_ids': ['singer']})
    seen = {}
    def preferences(*args, **kwargs):
        seen.update(kwargs)
        return []
    monkeypatch.setattr(materials, 'retrieve_preferences', preferences)
    await materials.retrieve_contextual_materials(RetrievalContext('周董呢'), [1], ChoosingClient(set()))
    assert seen['extra_ids'] == ['singer']


async def test_reference_resolution_can_supply_a_term_candidate(pool, monkeypatch):
    seen = []
    def terms(query):
        seen.append(query)
        return [{'keyword': '考', 'meaning': '上下文对应的圈内义项'}] if '考' in query else []
    monkeypatch.setattr(materials, 'term_candidates', terms)
    result = await materials.retrieve_contextual_materials(
        RetrievalContext('那个词呢', resolved_text='用户追问考的意思'), [1], ChoosingClient({('term', '考')}))
    assert '考' in seen[0]
    assert result['approved_terms'] == {'考'}


async def test_corpus_batch_has_one_use_contract_and_the_shared_context(pool):
    context = RetrievalContext('那后来呢', history='用户在问她过去的那件事',
                               response_task='追问她本人的过往', response_scope='conversation')
    client = ChoosingClient({('corpus', 'past')})
    result = await materials.retrieve_contextual_materials(context, [1], client)
    request = next(request for request in client.requests
                   if {c['来源'] for c in json.loads(request['messages'][-1]['content']).get('候选材料', [])} == {'corpus'})
    policy = request['messages'][0]['content']
    assert '只输出 JSON：{"keep": [编号' not in policy
    assert '"evidence"' in policy and policy.count('只输出JSON：') == 1
    assert '独立任务' in policy and '候选' in policy
    assert all(purpose in policy for purpose in ('answer', 'past_example', 'correction'))
    assert [item.item_id for item in result['corpus_items']] == ['past']
    assert all(payload['本轮语境'] == json.loads(context.note()) for payload in client.seen)


async def test_corpus_integration_cannot_authorize_bare_keep_without_fact_evidence(pool):
    class MissingEvidenceClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"keep":[1]}'))], usage=None)

    client = MissingEvidenceClient()
    result = await materials.retrieve_contextual_materials(
        RetrievalContext('你以前养过猫吗', needs=('corpus',)), [1], client)
    assert not result['corpus_items']
    assert not result['selection_trace']['batches']['corpus']
    assert client.calls == 2


async def test_corpus_trace_keeps_only_public_source_evidence_and_attempt_counts(pool):
    context = RetrievalContext('用户私人暗号XYZ，只问你养过猫吗', needs=('corpus', 'preference'))
    client = ChoosingClient({('corpus', 'past'), ('preference', 'animals')})
    result = await materials.retrieve_contextual_materials(context, [1], client)
    trace = result['selection_trace']
    assert trace['evidence'] == {'past': {'quote': '她过去养猫', 'relation': 'direct'}}
    assert trace['attempts'] == {'corpus': 2, 'materials': 1}
    assert trace['verification'] == {'past': {'entailed': True, 'answers': True}}
    assert '用户私人暗号XYZ' not in json.dumps(trace, ensure_ascii=False)
    assert result['preference_items']
    material_request = next(request for request in client.requests
                            if any(c['来源'] == 'preference'
                                   for c in json.loads(request['messages'][-1]['content']).get('候选材料', [])))
    assert '【事实证据核对】' not in material_request['messages'][0]['content']


async def test_corpus_and_other_material_decisions_start_in_parallel(pool):
    both_started = asyncio.Event()

    class ParallelClient(ChoosingClient):
        def __init__(self):
            super().__init__({('corpus', 'past'), ('preference', 'animals')})
            self.started = 0

        async def create(self, **kwargs):
            self.started += 1
            if self.started == 2:
                both_started.set()
            await both_started.wait()
            return await super().create(**kwargs)

    client = ParallelClient()
    result = await asyncio.wait_for(
        materials.retrieve_contextual_materials(RetrievalContext('说说那件事'), [1], client), timeout=1)
    assert client.started == 3
    assert result['corpus_items'] and result['preference_items']


async def test_no_corpus_pool_needs_only_one_material_call(pool, monkeypatch):
    monkeypatch.setattr(materials, 'retrieve_corpus_candidates', lambda *args: [])
    client = ChoosingClient({('preference', 'animals')})
    result = await materials.retrieve_contextual_materials(RetrievalContext('你喜欢猫吗'), [1], client)
    assert len(client.seen) == 1 and result['preference_items'] and not result['corpus_items']


async def test_empty_candidate_pools_need_no_api_call(pool, monkeypatch):
    for name in ('retrieve_corpus_candidates', 'retrieve_core_stories', 'retrieve_preferences',
                 'term_candidates', 'match_entities'):
        monkeypatch.setattr(materials, name, lambda *args, **kwargs: [])
    client = ChoosingClient(set())
    result = await materials.retrieve_contextual_materials(RetrievalContext('普通话头'), [1], client)
    assert not client.seen and result['selection_trace']['available']


@pytest.mark.parametrize('failed_batch', ['corpus', 'materials'])
async def test_failed_batch_does_not_authorize_itself_or_discard_the_successful_batch(pool, failed_batch):
    class PartialFailureClient(ChoosingClient):
        async def create(self, **kwargs):
            payload = json.loads(kwargs['messages'][-1]['content'])
            if '待核对事实' in payload:
                return await super().create(**kwargs)
            offered = payload['候选材料']
            is_corpus = {candidate['来源'] for candidate in offered} == {'corpus'}
            if is_corpus == (failed_batch == 'corpus'):
                raise RuntimeError('one batch offline')
            return await super().create(**kwargs)

    result = await materials.retrieve_contextual_materials(
        RetrievalContext('看看相关材料'), [1], PartialFailureClient({('corpus', 'past'), ('preference', 'animals')}))
    assert bool(result['corpus_items']) is (failed_batch != 'corpus')
    assert bool(result['preference_items']) is (failed_batch != 'materials')
    assert result['selection_trace']['available'] is False
    assert result['selection_trace']['batches'] == {
        'corpus': failed_batch != 'corpus', 'materials': failed_batch != 'materials'}


async def test_verification_api_failure_only_closes_corpus_batch(pool):
    class FailedReviewClient(ChoosingClient):
        async def create(self, **kwargs):
            if '待核对事实' in json.loads(kwargs['messages'][-1]['content']):
                raise RuntimeError('HTTP 503')
            return await super().create(**kwargs)

    result = await materials.retrieve_contextual_materials(
        RetrievalContext('你过去养过猫吗'), [1], FailedReviewClient({('corpus', 'past'), ('preference', 'animals')}))
    assert not result['corpus_items'] and result['preference_items']
    assert not result['selection_trace']['available']
    assert result['selection_trace']['batches'] == {'corpus': False, 'materials': True}
    assert result['selection_trace']['attempts'] == {'corpus': 2, 'materials': 1}


async def test_valid_negative_review_deletes_fact_without_degrading_other_materials(pool):
    class RejectReviewClient(ChoosingClient):
        async def create(self, **kwargs):
            payload = json.loads(kwargs['messages'][-1]['content'])
            if '待核对事实' in payload:
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    'checks': [{'index': row['index'], 'entailed': True, 'answers': False}
                               for row in payload['待核对事实']]})))], usage=None)
            return await super().create(**kwargs)

    result = await materials.retrieve_contextual_materials(
        RetrievalContext('用户自己的新话题'), [1], RejectReviewClient({('corpus', 'past'), ('preference', 'animals')}))
    assert not result['corpus_items'] and result['preference_items']
    assert result['selection_trace']['available']
    assert result['selection_trace']['verification'] == {'past': {'entailed': True, 'answers': False}}
    assert [(use.source, use.item_id) for use in result['material_uses']] == [('preference', 'animals')]


async def test_approved_material_uses_keep_verified_fact_and_distinct_source_purposes(pool):
    client = ChoosingClient({('corpus', 'past'), ('core_story', 'story'), ('preference', 'animals'),
                             ('term', '猫'), ('association', '书名')})
    result = await materials.retrieve_contextual_materials(RetrievalContext('说说你和猫的往事'), [1], client)
    uses = {(use.source, use.item_id): use for use in result['material_uses']}
    assert set(uses) == {('corpus', 'past'), ('core_story', 'story'), ('preference', 'animals'),
                         ('term', '猫'), ('association', '书名')}
    assert uses[('corpus', 'past')].purpose == 'past_example'
    assert uses[('corpus', 'past')].fact == '她过去养猫'
    assert uses[('corpus', 'past')].quote == '她过去养猫'
    assert uses[('core_story', 'story')].purpose == 'past_example'
    assert uses[('preference', 'animals')].purpose == 'background'
    assert uses[('term', '猫')].purpose == 'definition'
    assert uses[('association', '书名')].purpose == 'background'
    assert result['corpus_items'][0].text == '她过去养猫'
    assert result['association_use_ids'] == ['书名']


async def test_rejected_or_failed_review_never_leaves_material_use(pool):
    class RejectedCorpus(ChoosingClient):
        async def create(self, **kwargs):
            payload = json.loads(kwargs['messages'][-1]['content'])
            if '待核对事实' in payload:
                raise RuntimeError('unavailable verifier')
            return await super().create(**kwargs)

    result = await materials.retrieve_contextual_materials(RetrievalContext('你养过猫吗'), [1],
        RejectedCorpus({('corpus', 'past'), ('preference', 'animals')}))
    assert [(use.source, use.item_id) for use in result['material_uses']] == [('preference', 'animals')]
    empty = await materials.retrieve_contextual_materials(RetrievalContext('只改写这句话'), [1], ChoosingClient(set()))
    assert empty['material_uses'] == [] and empty['association_use_ids'] == []


async def test_association_use_ids_only_describe_groups_surviving_join_budget(pool, monkeypatch):
    monkeypatch.setattr(materials, 'match_entities', lambda *args: ['大段关系', '短关系'])
    monkeypatch.setattr(materials, 'resolve_associations', lambda query, seeds: {
        'note': '长' * 301 if seeds[0] == '大段关系' else '作者写过这本书', 'entities': [], 'preference_ids': []})
    result = await materials.retrieve_contextual_materials(RetrievalContext('那本书呢'), [1],
        ChoosingClient({('association', '大段关系'), ('association', '短关系')}))
    assert result['association_context'] == '作者写过这本书'
    assert result['association_use_ids'] == ['短关系']


async def test_correction_use_keeps_qualified_fact_without_rewriting_original_material(pool, monkeypatch):
    source = '她回忆学生时代，为应付盘问临时说喜欢一个同学，但并未说他是什么身份。'
    fact = '她此前为应付盘问临时说喜欢同学，未说明其身份'
    quote = '为应付盘问临时说喜欢一个同学，但并未说他是什么身份'
    monkeypatch.setattr(materials, 'retrieve_corpus_candidates', lambda *args: [
        RetrievalItem('corpus', 'excuse', .8, text=source)])

    class CorrectionClient(ChoosingClient):
        async def create(self, **kwargs):
            response = await super().create(**kwargs)
            payload = json.loads(response.choices[0].message.content)
            for proof in payload.get('evidence', []):
                proof.update(fact=fact, quote=quote, purpose='correction')
            response.choices[0].message.content = json.dumps(payload)
            return response

    client = CorrectionClient({('corpus', 'excuse')})
    result = await materials.retrieve_contextual_materials(
        RetrievalContext('你以前喜欢的那位是不是班长'), [1], client)
    use, = result['material_uses']
    assert (use.source, use.item_id, use.purpose) == ('corpus', 'excuse', 'correction')
    assert use.fact == fact and use.quote == quote
    assert result['corpus_items'][0].text == source
    assert client.verifications[0]['待核对事实'][0]['purpose'] == 'correction'


async def test_wiki_and_fallback_behavior_also_return_only_their_approved_uses(pool):
    behavior = RetrievalItem('behavior', '短回应', 1, text='当前互动的语气参考')
    result = await materials.retrieve_contextual_materials(
        RetrievalContext('陌生作者是谁', entities=('陌生作者',)), [1],
        ChoosingClient({('wiki', '陌生作者'), ('behavior', '短回应')}), behavior_candidate=behavior)
    assert {(use.source, use.item_id, use.purpose) for use in result['material_uses']} == {
        ('wiki', '陌生作者', 'background'), ('behavior', '短回应', 'interaction')}


async def test_association_budget_includes_actual_usage_note_and_skips_whole_overbudget_relation(pool, monkeypatch):
    from src.plugins.chatbot.material_usage import render_material_use
    long_relation = '长' * 270
    short_relation = '作者写过这本书'
    monkeypatch.setattr(materials, 'match_entities', lambda *args: ['长关系', '短关系'])
    monkeypatch.setattr(materials, 'resolve_associations', lambda query, seeds: {
        'note': long_relation if seeds[0] == '长关系' else short_relation,
        'entities': [], 'preference_ids': []})
    result = await materials.retrieve_contextual_materials(RetrievalContext('这本书呢'), [1],
        ChoosingClient({('association', '长关系'), ('association', '短关系')}))
    uses = [use for use in result['material_uses'] if use.source == 'association']
    notes = list(dict.fromkeys(render_material_use(use) for use in uses))
    rendered = '\n'.join([*notes, result['association_context']])
    assert len(rendered) <= 300
    assert result['association_context'] == short_relation
    assert result['association_use_ids'] == ['短关系']
    assert [use.item_id for use in uses] == ['短关系']
    monkeypatch.setattr(materials, 'match_entities', lambda *args: ['长关系'])
    empty = await materials.retrieve_contextual_materials(RetrievalContext('这本书呢'), [1],
        ChoosingClient({('association', '长关系')}))
    assert empty['association_context'] == '' and empty['association_use_ids'] == []
    assert not [use for use in empty['material_uses'] if use.source == 'association']
