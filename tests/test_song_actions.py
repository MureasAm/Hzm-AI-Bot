import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.plugins.chatbot import core, reply_style, routing, song_requests
from src.plugins.chatbot.songs import Song


@pytest.mark.asyncio
async def test_classification_checks_authored_text_for_song_requests():
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"behavior": null, "song_request": null}'),
        finish_reason="stop",
    )]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = await routing.classify_l3(
        client, '[引用：可以给我唱首歌吗] 那段是真好听', '', [],
        song_text='那段是真好听',
    )
    assert result.get('song_request') is None
    assert '那段是真好听' in create.call_args.kwargs['messages'][0]['content'].split('【点歌判定】')[-1]


@pytest.mark.parametrize(("mode", "authored", "flag", "expected_file"), [
    ("full", "发《小星星》完整版", True, False),
    ("full", "发《小星星》的完整版音频文件", "true", False),
    ("full", "发《小星星》的完整版音频文件", True, True),
    ("recent_full", "把刚才那首完整版作为附件发我", True, True),
    ("clip", "给我唱两句《小星星》，用音频文件发", True, False),
])
async def test_only_an_explicit_full_song_can_use_file_delivery(mode, authored, flag, expected_file):
    title = "" if mode == "recent_full" else "小星星"
    content = json.dumps({"behavior": None, "song_request": {
        "mode": mode, "title": title, "as_file": flag,
    }})
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason="stop",
    )]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = await routing.classify_l3(client, authored, "", [], song_text=authored)
    catalog = {"小星星": Song("小星星", Path("clip.wav"), Path("full.m4a"))}
    note, options = song_requests.plan_song(result["song_request"], catalog, "小星星")

    assert options["1"].as_file is expected_file
    assert ("音频文件" if expected_file else "QQ 语音条") in note


def test_missing_named_song_offers_a_choice_without_authorizing_other_audio():
    catalog = {'小星星': Song('小星星', Path('song.wav'), Path('song.mp3'))}
    note, options = song_requests.plan_song({'mode': 'clip', 'title': '另一首'}, catalog, '')
    assert '另一首' in note and '小星星' in note
    assert not options


def test_full_followup_uses_only_the_preceding_confirmed_song():
    catalog = {'小星星': Song('小星星', Path('song.wav'), Path('song.mp3'))}
    request = {'mode': 'recent_full', 'title': ''}
    assert not song_requests.plan_song(request, catalog, '')[1]
    _, options = song_requests.plan_song(request, catalog, '小星星')
    assert [(opt.title, opt.mode, opt.path) for opt in options.values()] == [
        ('小星星', 'full', Path('song.mp3')),
    ]


def test_confirmed_named_full_request_overrides_previous_song():
    catalog = {
        '小星星': Song('小星星', None, Path('star.mp3')),
        '月亮': Song('月亮', None, Path('moon.mp3')),
    }
    _, options = song_requests.plan_song({'mode': 'recent_full', 'title': '月亮'}, catalog, '小星星')
    assert [item.title for item in options.values()] == ['月亮']


@pytest.mark.parametrize('body,previous,should_classify', [
    ('好', '要发《小星星》完整版吗？', True),
    ('想听你唱歌了', '', False),
    ('给我唱两句吧\n等等，不用唱了', '', False),
    ('你唱歌太好听了', '', False),
    ('你唱歌太好听了，再给我唱首歌吧', '', True),
])
def test_song_candidate_gate_checks_the_actual_request_and_later_cancellation(
        body, previous, should_classify):
    assert song_requests.may_request_song(body, {}, previous) is should_classify


def test_action_block_lists_singing_only_where_the_chat_can_play_it():
    messages = core.build_message_list('嗨', [], [], show_actions=True,
                                       song_context='唱歌能力测试')
    assert any('唱歌能力测试' in m['content'] for m in messages if isinstance(m['content'], str))
    messages = core.build_message_list('嗨', [], [], show_actions=True, is_group=True,
                                       song_context='唱歌能力测试')
    assert not any('唱歌能力测试' in m['content'] for m in messages if isinstance(m['content'], str))


def test_singing_tag_can_be_the_whole_reply():
    text, actions = reply_style.extract_action_tags('[[唱歌:2]]')
    assert text == '' and actions['song'] == '2'


@pytest.mark.parametrize('is_group,song_request,reply,selected', [
    (False, {'mode': 'clip', 'title': '小星星'}, '[[唱歌:1]]', True),
    (False, {'mode': 'clip', 'title': '小星星'}, '今天先饶了灰泽满吧', False),
    (False, {'mode': 'clip', 'title': '小星星'}, '灰泽满才不唱呢，又不是点歌台。[[唱歌:1]]', False),
    (False, {'mode': 'clip', 'title': '小星星'}, '整首就不唱了，给你哼两句。[[唱歌:1]]', True),
    (False, None, '夸大了啊 [[唱歌:1]]', False),
    (False, None, '[[唱歌:1]]', False),
    (True, None, '先歇会儿 [[唱歌:1]]', False),
])
@pytest.mark.asyncio
async def test_main_reply_supplies_and_enforces_its_song_options(
        monkeypatch, is_group, song_request, reply, selected):
    text = '给我唱两句《小星星》' if song_request else '你唱歌很好听'
    catalog = {'小星星': Song('小星星', Path('song.wav'), Path('song.mp3'))}
    classify = AsyncMock(return_value={
        'classification_ok': True, 'behavior': '被推着要求时打太极' if song_request else '',
        'song_request': song_request,
    })
    generate = AsyncMock(return_value=reply)
    empty_materials = {
        'corpus_items': [], 'preference_items': [], 'core_stories': [], 'behavior_items': [],
        'denied_terms': set(), 'association_context': '', 'association_entities': [],
        'wiki_entities': [], 'entity_mentions': {}, 'selection_trace': {'available': True},
    }
    monkeypatch.setattr(core, '_get_clients', lambda: (None, None))
    monkeypatch.setattr(core, 'aux_clients_or', lambda fallback: (None, None))
    monkeypatch.setattr(core, 'injection_v3_enabled', lambda: False)
    monkeypatch.setattr(core.memory_v2, 'injection_enabled', lambda: False)
    monkeypatch.setattr(core.self_speech, 'enabled', lambda: False)
    monkeypatch.setattr(core, '_weather_city_for', lambda uid: '')
    monkeypatch.setattr(core, 'get_flag', lambda name, default='': default)
    monkeypatch.setattr(core, 'LEGENDARY_REPLIES', {})
    monkeypatch.setattr(core, 'spawn', lambda coro: coro.close())
    monkeypatch.setattr(core, 'probe_session', AsyncMock(return_value=(text, {})))
    monkeypatch.setattr(core.context_probe, 'warm_weather', AsyncMock())
    monkeypatch.setattr(core.context_probe, 'get_now_context', lambda city='': '')
    monkeypatch.setattr(core.event_memory, 'retrieve', AsyncMock(return_value=[]))
    monkeypatch.setattr(core, 'classify_l3', classify)
    monkeypatch.setattr(core, 'embed_query', AsyncMock(return_value=None))
    monkeypatch.setattr(core, 'retrieve_contextual_materials', AsyncMock(return_value=empty_materials))
    monkeypatch.setattr(core, 'generate_reply', generate)
    monkeypatch.setattr(core.songs, 'load_songs', lambda: catalog)

    response = await core.handle_chat(
        'song-test', text, is_group=is_group, defer_memory=not is_group, song_text=text)
    prompt = '\n'.join(m['content'] for m in generate.call_args.args[0]
                       if isinstance(m['content'], str))
    assert ('【唱歌】' in prompt) is not is_group
    assert ('1 "小星星"（经典两句）' in prompt) is bool(song_request)
    assert '被推着要求时打太极' not in prompt
    assert classify.call_args.kwargs.get('song_text') == (text if song_request else None)
    assert '[[唱歌' not in response
    delivery = response.actions.get('song')
    if selected:
        assert (delivery.title, delivery.mode, delivery.path) == ('小星星', 'clip', Path('song.wav'))
    else:
        assert delivery is None
    expected_text = reply_style.extract_action_tags(reply)[0]
    if not song_request and not expected_text:
        assert response
    else:
        assert response == expected_text
