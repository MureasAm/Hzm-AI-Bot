"""Shared context and classification availability at routing boundaries."""
import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import routing
from src.plugins.chatbot import retrieval_context
from src.plugins.chatbot.retrieval_context import RetrievalContext


class _Client:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.content), finish_reason="stop",
        )])


BEHAVIORS = [{"name": "被夸", "trigger": "用户夸她本人"}]
CONTEXT = RetrievalContext(
    raw_text="你唱得好听", resolved_text="在夸另一个人唱歌", history="共享历史",
    target="另一位群友", speech_act="夸奖", pragmatic_note="辅助解释可能有误",
    is_group=True, addressed=False,
)
EMPTY = {"behavior": "", "same_request": {"what": "", "count": 1}, "reminder": None}


class _SequenceClient(_Client):
    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=response), finish_reason='stop',
        )])


@pytest.mark.asyncio
@pytest.mark.parametrize('song_text', ['', '唱两句'])
async def test_contextual_prompt_has_one_complete_output_contract(song_text):
    client = _Client('{"behavior":null,"same_request":null,"reminder":null,"task":"当前任务","scope":"conversation","song_request":null}')
    await routing.classify_l3(client, '当前原话', '前文', BEHAVIORS, context=CONTEXT, song_text=song_text)
    prompt = client.calls[0]['messages'][0]['content']
    assert prompt.count('【唯一输出合同】') == 1
    contract = json.loads(prompt.split('【唯一输出合同】\n', 1)[1].split('\n', 1)[0])
    expected = {'behavior', 'same_request', 'reminder', 'task', 'scope', 'understanding'}
    if song_text:
        expected.add('song_request')
    assert set(contract) == expected
    assert '只输出 JSON：{"behavior"' not in prompt


@pytest.mark.asyncio
async def test_format_repair_fills_missing_task_without_overriding_valid_first_behavior(monkeypatch):
    client = _SequenceClient([
        '{"behavior":"被夸","same_request":{"what":"要一个答复","count":2},"reminder":null}',
        '{"behavior":"编造的行为","same_request":null,"reminder":null,"task":"询问本轮实际话题","scope":"conversation"}',
    ])
    usage_tags = []
    monkeypatch.setattr(routing, 'log_cache_usage', lambda response, tag: usage_tags.append(tag))
    result = await routing.classify_l3(client, '你唱得好听', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 2 and len(usage_tags) == 2
    assert result['classification_ok'] is True and result['behavior'] == '被夸'
    assert result['same_request'] == {'what': '要一个答复', 'count': 2}
    assert result['needs_ok'] is True and result['task'] == '询问本轮实际话题'
    assert '格式补救' in usage_tags[1]
    assert CONTEXT.note() in client.calls[1]['messages'][0]['content']


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', ['', None, 'not JSON', '{"behavior":', '{"behavior":null,"task":"任务","scope":"invalid"}'])
async def test_one_format_repair_can_recover_broken_json_or_invalid_scope(bad):
    client = _SequenceClient([bad, '{"behavior":null,"task":"已修复的任务","scope":"general_task"}'])
    result = await routing.classify_l3(client, '本轮原话', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 2 and result['needs_ok'] is True and result['classification_ok'] is True
    assert result['scope'] == 'general_task'


@pytest.mark.asyncio
async def test_second_bad_response_stops_and_preserves_the_first_valid_null():
    client = _SequenceClient(['{"behavior":null}', 'still not JSON'])
    result = await routing.classify_l3(client, '本轮原话', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 2
    assert result['classification_ok'] is True and result['behavior'] == ''
    assert result['needs_ok'] is False and result['needs'] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [PermissionError('forbidden'), RuntimeError('HTTP 503')])
async def test_transport_or_permission_error_does_not_start_format_repair(error):
    client = _SequenceClient([error])
    result = await routing.classify_l3(client, '本轮原话', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 1 and not result['classification_ok'] and not result['needs_ok']


@pytest.mark.asyncio
async def test_transport_failure_during_format_repair_preserves_confirmed_first_result():
    client = _SequenceClient(['{"behavior":"被夸"}', PermissionError('forbidden')])
    result = await routing.classify_l3(client, '本轮原话', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 2 and result['classification_ok'] and result['behavior'] == '被夸'
    assert not result['needs_ok']


@pytest.mark.asyncio
async def test_unknown_behavior_does_not_trigger_retry_or_invalidate_valid_task():
    client = _SequenceClient(['{"behavior":"不存在","task":"有效任务","scope":"conversation"}'])
    result = await routing.classify_l3(client, '本轮原话', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 1 and not result['classification_ok'] and result['needs_ok']


@pytest.mark.asyncio
async def test_data_trigger_and_actual_participants_define_behavior_eligibility():
    client = _Client('{"behavior":null,"task":"用户在与她交谈","scope":"conversation"}')
    await routing.classify_l3(client, '你今天忙什么', '', BEHAVIORS, context=CONTEXT)
    prompt = client.calls[0]['messages'][0]['content']
    assert '谁在说话、评价谁、要求谁做什么' in prompt
    assert '不能凭熟悉语气补出未出现的第三人' in prompt
    assert '普通闲聊、提问、寒暄、表情、玩梗、单纯道贺、用户聊自己的事 → null' not in prompt
    assert '普通闲聊或玩梗也可能符合某条trigger' in prompt


@pytest.mark.asyncio
async def test_legacy_bad_json_still_uses_one_call():
    client = _SequenceClient(['not JSON'])
    assert await routing.classify_l3(client, '原话', '', BEHAVIORS) == EMPTY
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_classification_receives_context_and_retains_original_inputs():
    client = _Client('{"behavior": "被夸"}')
    result = await routing.classify_l3(
        client, "原始用户消息", "原始最近对话", BEHAVIORS, context=CONTEXT,
    )
    prompt = client.calls[0]["messages"][0]["content"]
    assert "用户消息：原始用户消息" in prompt
    assert "最近对话：\n原始最近对话" in prompt
    assert CONTEXT.note() in prompt
    assert result["behavior"] == "被夸"
    assert result["classification_ok"] is True
    assert client.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_explicit_null_is_a_successful_no_behavior_decision():
    result = await routing.classify_l3(
        _Client('{"behavior": null}'), "你好", "", BEHAVIORS, context=CONTEXT,
    )
    assert result == {**EMPTY, "classification_ok": True, "needs": [], "needs_ok": False, "task": "", "scope": "", "understanding": None, "understanding_ok": False}


@pytest.mark.asyncio
@pytest.mark.parametrize('empty_label', [None, 'null', ' null '])
async def test_vendor_null_label_is_an_explicit_no_behavior_not_a_classifier_failure(empty_label):
    client = _Client(json.dumps({'behavior': empty_label, 'task': '引用同事的催问，并明确不是催她',
                                'scope': 'conversation'}))
    result = await routing.classify_l3(client, '同事又鸽了，不是在催你', '', BEHAVIORS, context=CONTEXT)
    assert result['classification_ok'] is True
    assert result['behavior'] == '' and result['needs_ok'] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    '{}', '{"behavior": ""}', '{"behavior": "编造的行为"}',
    '{"behavior": false}', '{"behavior": 1}', '{"behavior": []}',
    '[]', 'null', 'not JSON',
])
async def test_invalid_classification_is_distinct_from_explicit_null(content):
    result = await routing.classify_l3(
        _Client(content), "你好", "", BEHAVIORS, context=CONTEXT,
    )
    assert result["behavior"] == ""
    assert result["classification_ok"] is False


@pytest.mark.asyncio
async def test_network_failure_marks_classification_unavailable():
    result = await routing.classify_l3(
        _Client(error=RuntimeError("offline")), "你好", "", BEHAVIORS, context=CONTEXT,
    )
    assert result == {**EMPTY, "classification_ok": False, "needs": [], "needs_ok": False, "task": "", "scope": "", "understanding": None, "understanding_ok": False}


@pytest.mark.asyncio
async def test_empty_input_marks_classification_unavailable():
    client = _Client('{"behavior": null}')
    result = await routing.classify_l3(client, "", "", BEHAVIORS, context=CONTEXT)
    assert result == {**EMPTY, "classification_ok": False, "needs": [], "needs_ok": False, "task": "", "scope": "", "understanding": None, "understanding_ok": False}
    assert client.calls == []


@pytest.mark.asyncio
async def test_legacy_classification_result_shape_is_unchanged():
    client = _Client('{"behavior": null}')
    result = await routing.classify_l3(client, "你好", "", BEHAVIORS)
    assert result == EMPTY
    assert "response_format" not in client.calls[0]
    failed = await routing.classify_l3(
        _Client(error=RuntimeError("offline")), "你好", "", BEHAVIORS,
    )
    assert failed == EMPTY


@pytest.mark.asyncio
@pytest.mark.parametrize('scope,expected', [
    ('conversation', {'corpus', 'core_story', 'preference', 'term', 'association', 'wiki', 'event', 'memory_hint'}),
    ('general_task', {'corpus', 'core_story', 'preference', 'term', 'association', 'wiki', 'event', 'memory_hint'}),
])
async def test_answer_scope_expands_without_guessing_material_sources(scope, expected):
    client = _Client(json.dumps({"behavior": None, "task": "  用户本轮真正想解决的问题  ", "scope": scope}))
    result = await routing.classify_l3(client, "你上次提过哪件事", "最近原话", BEHAVIORS, context=CONTEXT)
    assert set(result["needs"]) == expected
    assert len(result["needs"]) == len(expected)
    assert result["needs_ok"] is True
    assert result["classification_ok"] is True
    assert result['task'] == '用户本轮真正想解决的问题' and result['scope'] == scope
    prompt = client.calls[0]["messages"][0]["content"]
    assert '"task"' in prompt and '"scope"' in prompt and "不依赖候选" in prompt
    assert '"needs"' not in prompt
    assert '含义不清' in prompt and '知识问句' in prompt
    assert CONTEXT.note() in prompt


@pytest.mark.asyncio
async def test_general_task_does_not_guess_where_a_useful_fact_is_stored():
    result = await routing.classify_l3(
        _Client('{"behavior": "被夸", "task": "完成独立通用操作", "scope": "general_task", "needs": ["corpus"]}'),
        "完成这个操作", "", BEHAVIORS, context=CONTEXT,
    )
    assert set(result["needs"]) == retrieval_context.MATERIAL_SOURCES and result["needs_ok"] is True
    assert result['task'] == '完成独立通用操作' and result['scope'] == 'general_task'
    assert result["classification_ok"] is True and result["behavior"] == "被夸"


@pytest.mark.asyncio
@pytest.mark.parametrize('fields', [
    {}, {'task': '实际任务'}, {'scope': 'conversation'},
    {'task': '', 'scope': 'conversation'}, {'task': '  ', 'scope': 'conversation'},
    {'task': None, 'scope': 'conversation'}, {'task': [], 'scope': 'conversation'},
    {'task': True, 'scope': 'conversation'}, {'task': '实际任务', 'scope': None},
    {'task': '实际任务', 'scope': []}, {'task': '实际任务', 'scope': 'unknown'},
    {'task': '实际任务', 'scope': 'persona'}, {'needs': ['corpus']}, {'needs': ['persona']},
])
async def test_invalid_task_or_scope_does_not_authorize_materials_or_invalidate_behavior(fields):
    result = await routing.classify_l3(
        _Client(json.dumps({"behavior": "被夸", **fields})), "你唱得好听", "", BEHAVIORS, context=CONTEXT,
    )
    assert result["needs"] == [] and result["needs_ok"] is False
    assert result['task'] == result['scope'] == ''
    assert result["classification_ok"] is True and result["behavior"] == "被夸"


@pytest.mark.asyncio
async def test_valid_task_remains_independent_when_behavior_is_invalid():
    result = await routing.classify_l3(
        _Client('{"behavior": "编造的行为", "task": "询问她的日常选择", "scope": "conversation"}'),
        "你喜欢哪些水果", "", BEHAVIORS, context=CONTEXT,
    )
    assert result["classification_ok"] is False and result["behavior"] == ""
    assert result["needs_ok"] is True and set(result["needs"]) == retrieval_context.MATERIAL_SOURCES
    assert result['task'] == '询问她的日常选择' and result['scope'] == 'conversation'


@pytest.mark.asyncio
async def test_legacy_call_does_not_request_or_return_material_needs():
    client = _Client('{"behavior": null, "task": "当前任务", "scope": "conversation"}')
    result = await routing.classify_l3(client, "你好", "", BEHAVIORS)
    assert result == EMPTY
    prompt = client.calls[0]["messages"][0]["content"]
    assert all(field not in prompt for field in ('"needs"', '"task"', '"scope"'))


@pytest.mark.asyncio
async def test_existing_json_repairs_still_apply_with_context():
    result = await routing.classify_l3(
        _Client("{behavior: '被夸',}"), "你好", "", BEHAVIORS, context=CONTEXT,
    )
    assert result["behavior"] == "被夸"
    assert result["classification_ok"] is True


@pytest.mark.asyncio
async def test_json_string_fences_do_not_truncate_classification():
    result = await routing.classify_l3(
        _Client('```JSON\n{"behavior": null, "same_request": {"what": "说 ```", "count": 2}}\n```'),
        "你好", "", BEHAVIORS, context=CONTEXT,
    )
    assert result["classification_ok"] is True
    assert result["same_request"] == {"what": "说 ```", "count": 2}


@pytest.mark.asyncio
async def test_legendary_confirmation_receives_the_same_context(monkeypatch):
    client = _Client("是")
    monkeypatch.setattr(routing, "aux_clients_or", lambda _: (client, None))
    result = await routing.legendary_confirmed(
        "原始消息", "最近对话：{context}\n用户：{msg}\n只回复：是 或 否",
        history="原始历史", context=CONTEXT,
    )
    prompt = client.calls[0]["messages"][0]["content"]
    assert "最近对话：原始历史" in prompt
    assert "用户：原始消息" in prompt
    assert CONTEXT.note() in prompt
    assert result is True


@pytest.mark.asyncio
async def test_contextual_legendary_failure_degrades_to_normal_generation(monkeypatch):
    client = _Client(error=RuntimeError("offline"))
    monkeypatch.setattr(routing, "aux_clients_or", lambda _: (client, None))
    assert await routing.legendary_confirmed("你几岁", "{msg}", context=CONTEXT) is False


@pytest.mark.asyncio
async def test_legacy_legendary_failure_strategy_is_preserved(monkeypatch):
    client = _Client(error=RuntimeError("offline"))
    monkeypatch.setattr(routing, "aux_clients_or", lambda _: (client, None))
    assert await routing.legendary_confirmed("你几岁", "{msg}") is True


@pytest.mark.asyncio
async def test_contextual_legendary_invalid_output_is_not_confirmed(monkeypatch):
    client = _Client("拿不准")
    monkeypatch.setattr(routing, "aux_clients_or", lambda _: (client, None))
    assert await routing.legendary_confirmed("你几岁", "{msg}", context=CONTEXT) is False


@pytest.mark.asyncio
@pytest.mark.parametrize('answer,expected', [
    ('是', True), (' 是。 ', True), ('是！', True), ('是.', True),
    ('是否', False), ('是或否', False), ('是吗?', False), ('是？', False),
    ('是，还是不是？', False),
])
async def test_contextual_legendary_requires_an_exact_affirmative_label(monkeypatch, answer, expected):
    client = _Client(answer)
    monkeypatch.setattr(routing, 'aux_clients_or', lambda _: (client, None))
    assert await routing.legendary_confirmed('你几岁', '{msg}', context=CONTEXT) is expected


@pytest.mark.asyncio
async def test_legacy_legendary_prefix_parsing_is_unchanged(monkeypatch):
    client = _Client('是否')
    monkeypatch.setattr(routing, 'aux_clients_or', lambda _: (client, None))
    assert await routing.legendary_confirmed('你几岁', '{msg}') is True


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["classification", "legendary", "materials"])
async def test_shared_semantic_guidance_reaches_each_contextual_decision(step, monkeypatch):
    guidance = retrieval_context.CONTEXT_GUIDANCE
    client = _Client('{"behavior": null}' if step == "classification" else
                     ('是' if step == "legendary" else '{"keep": [1]}'))
    if step == "classification":
        result = await routing.classify_l3(
            client, "你好", "", BEHAVIORS, context=CONTEXT,
        )
        assert result["classification_ok"] is True
    elif step == "legendary":
        monkeypatch.setattr(routing, "aux_clients_or", lambda _: (client, None))
        assert await routing.legendary_confirmed("你好", "{msg}", context=CONTEXT) is True
    else:
        result = await retrieval_context.select_materials(client, CONTEXT, [
            retrieval_context.MaterialCandidate("preference", "food", "她喜欢水果"),
        ], model="fake")
        assert result.selected == frozenset({("preference", "food")})
    prompt = "\n".join(message["content"] for message in client.calls[0]["messages"])
    assert prompt.count(guidance) == 1
    assert all(concept in guidance for concept in (
        "主体", "互动对象", "实际意图", "引用", "否定", "假设", "过去/现在",
        "可能出错", "用户自述", "第三人", "真实互动",
    ))


def _understanding(**updates):
    value = {'focus': 'character', 'focus_text': '对话中的角色', 'request': '回应用户对她的评价',
             'exclusions': [], 'premises': ['对方提到的表现尚未核实'], 'third_parties': []}
    value.update(updates)
    return value


def _understood_response(**updates):
    value = {'behavior': None, 'same_request': None, 'reminder': None,
             'task': '回应评价，不把问题前提当事实', 'scope': 'conversation',
             'understanding': _understanding()}
    value.update(updates)
    return json.dumps(value, ensure_ascii=False)


@pytest.mark.asyncio
async def test_contextual_understanding_is_typed_and_prompt_binds_people_before_task():
    from src.plugins.chatbot.turn_understanding import TurnUnderstanding
    client = _Client(_understood_response())
    result = await routing.classify_l3(client, '今天你在小群里成冷场大王了', '', BEHAVIORS, context=CONTEXT)
    assert result['understanding_ok'] is True and isinstance(result['understanding'], TurnUnderstanding)
    assert result['understanding'].to_dict() == _understanding()
    prompt = client.calls[0]['messages'][0]['content']
    assert all(text in prompt for text in ('先绑定人物', '待核实前提', '不能当作既成事实',
                                         '只约束被排除的用途', '连续原文'))
    assert client.calls[0]['max_tokens'] >= 700 and len(client.calls) == 1


@pytest.mark.asyncio
async def test_missing_understanding_is_compatible_without_a_semantic_retry():
    client = _Client('{"behavior":null,"task":"原有有效任务","scope":"conversation"}')
    result = await routing.classify_l3(client, '原话', '', BEHAVIORS, context=CONTEXT)
    assert result['needs_ok'] and result['classification_ok']
    assert result['understanding'] is None and result['understanding_ok'] is False
    assert len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', [None, {}, {'focus': 'invalid'}, _understanding(
    third_parties=[{'name': '未出现的人', 'evidence': '本轮原话'}])])
async def test_invalid_present_understanding_uses_one_format_repair_preserving_valid_fields(invalid):
    client = _SequenceClient([
        _understood_response(behavior='被夸', understanding=invalid),
        _understood_response(behavior='不存在', task='不覆盖第一轮任务'),
    ])
    result = await routing.classify_l3(client, '本轮原话', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 2
    assert result['classification_ok'] and result['behavior'] == '被夸'
    assert result['task'] == '回应评价，不把问题前提当事实'
    assert result['understanding_ok'] and result['understanding'].request == '回应用户对她的评价'


@pytest.mark.asyncio
async def test_second_invalid_understanding_does_not_erase_independent_behavior_or_task():
    client = _SequenceClient([_understood_response(behavior='被夸', understanding={}),
                              _understood_response(understanding=[])])
    result = await routing.classify_l3(client, '原话', '', BEHAVIORS, context=CONTEXT)
    assert len(client.calls) == 2
    assert result['needs_ok'] and result['classification_ok'] and result['behavior'] == '被夸'
    assert result['understanding'] is None and not result['understanding_ok']


@pytest.mark.asyncio
@pytest.mark.parametrize('response_understanding', ['missing', _understanding()])
async def test_relation_behavior_without_grounded_third_party_is_valid_null(response_understanding):
    behavior = {'name': '谈及熟人', 'trigger': '聊到她熟的人', 'requires_third_party': True}
    data = json.loads(_understood_response(behavior='谈及熟人'))
    if response_understanding == 'missing':
        del data['understanding']
    else:
        data['understanding'] = response_understanding
    client = _Client(json.dumps(data))
    result = await routing.classify_l3(client, '你今天表现怎么样', '', [behavior], context=CONTEXT)
    assert result['classification_ok'] is True and result['behavior'] == ''
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_relation_behavior_allows_grounded_history_participant_after_format_repair():
    behavior = {'name': '谈及熟人', 'trigger': '聊到她熟的人', 'requires_third_party': True}
    client = _SequenceClient([
        _understood_response(behavior='谈及熟人', understanding={}),
        _understood_response(behavior=None, understanding=_understanding(third_parties=[
            {'name': '阿青', 'evidence': '阿青是她的旧室友'},
        ])),
    ])
    result = await routing.classify_l3(client, '那她后来呢', '阿青是她的旧室友', [behavior], context=CONTEXT)
    assert result['classification_ok'] and result['behavior'] == '谈及熟人'
    assert result['understanding_ok'] and len(client.calls) == 2


@pytest.mark.asyncio
async def test_legacy_relation_behavior_is_unchanged_and_has_no_new_fields():
    behavior = {'name': '谈及熟人', 'trigger': '聊到她熟的人', 'requires_third_party': True}
    client = _Client(_understood_response(behavior='谈及熟人'))
    result = await routing.classify_l3(client, '原话', '', [behavior])
    assert result == {**EMPTY, 'behavior': '谈及熟人'}
    assert 'understanding' not in client.calls[0]['messages'][0]['content']
    assert client.calls[0]['max_tokens'] == 420


def test_relation_requirements_live_in_behavior_data():
    from src.plugins.chatbot.persona import load_persona_rules
    rules = {row['name']: row for row in load_persona_rules()}
    for name in ('能嘴熟络同期是亲近', '被套话点名排名他人时端水'):
        assert rules[name]['requires_third_party'] is True
