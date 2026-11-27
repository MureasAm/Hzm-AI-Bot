import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.plugins.chatbot import chat_window as cw
from src.plugins.chatbot import core, memory, reply_evidence, reply_style, routing, song_requests
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
    ("full", "发《小星星》完整版语音，别发文件", True, False),
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


def test_spacing_in_confirmed_english_title_does_not_hide_a_catalog_song():
    catalog = {'always online': Song('Always Online', Path('song.wav'), None)}
    _, options = song_requests.plan_song({'mode': 'clip', 'title': 'AlwaysOnline'}, catalog, '')
    assert options['1'].title == 'Always Online'


@pytest.mark.parametrize('title,spoken', [
    ('Make it!', 'MAKE　IT！'),
    ('小悪魔だってかまわない！', '小悪魔だってかまわない!'),
    ('ドライフラワー（干花）', 'ドライフラワー(干花)'),
    ('热爱105℃的你', '热爱105°C的你'),
])
def test_equivalent_title_characters_use_the_canonical_filename(title, spoken):
    song = Song(title, Path('clip.wav'), None)
    assert song_requests.find_song(spoken, {song_requests.song_key(title): song}) == song


def test_compact_lookup_rejects_collisions_and_prefers_an_exact_title():
    a, b = (Song(title, Path(f'{title}.wav'), None) for title in ('A BC', 'AB C'))
    catalog = {song_requests.song_key(s.title): s for s in (a, b)}
    assert song_requests.find_song('ABC', catalog) is None
    assert song_requests.find_song('a bc', catalog) == a


def test_legacy_offer_uses_received_order_even_with_split_boundaries_and_title_punctuation():
    titles = ('Always Online', 'First Love', 'Make it!')
    catalog = {song_requests.song_key(title): Song(title, Path(f'{title}.wav'), None)
               for title in titles}
    for i, title in enumerate(titles):
        assert song_requests.confirmed_choice(
            i, catalog, '想听哪首？\nAlways Online、\nFirst Love、\nMake it!。'
        ) == {'mode': 'clip', 'title': title}


def test_confirmed_choice_keeps_the_offered_full_version():
    catalog = {'月亮': Song('月亮', None, Path('moon.m4a')),
               '小星星': Song('小星星', Path('star.wav'), Path('star.m4a'))}
    request = song_requests.confirmed_choice(1, catalog, '要听哪首完整版？《月亮》、小星星')
    _, options = song_requests.plan_song(request, catalog, '')
    assert (options['1'].title, options['1'].mode, options['1'].as_file) == ('小星星', 'full', False)


@pytest.mark.parametrize('is_group,song_request,reply,selected', [
    (False, {'mode': 'clip', 'title': '小星星'}, '[[唱歌:1]]', True),
    (False, {'mode': 'clip', 'title': '小星星'}, '今天先饶了灰泽满吧', False),
    (False, {'mode': 'clip', 'title': '小星星'}, '灰泽满才不唱呢，又不是点歌台。[[唱歌:1]]', False),
    (False, {'mode': 'clip', 'title': '小星星'}, '这首先不唱啦。[[唱歌:1]]', False),
    (False, {'mode': 'clip', 'title': '小星星'}, '改天给你唱两句', False),
    (False, {'mode': 'clip', 'title': '小星星'}, '现在就唱吗？', False),
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
    classify, generate = _mock_song_turn(monkeypatch, text, catalog, song_request, reply)

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


def _mock_song_turn(monkeypatch, text, catalog, song_request, reply, history=()):
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
    monkeypatch.setattr(core, 'get_user_history_timed', lambda _: list(history))
    return classify, generate


@pytest.mark.parametrize('reply,expected_song', [
    ('AlwaysOnline，唱完就得回去了哈', True),
    ('[[唱歌:1]]', True),
    ('这就唱 [[唱歌:999]]', False),
    ('今天先饶了灰泽满吧', False),
])
@pytest.mark.asyncio
async def test_main_reply_resolves_the_rank_in_the_delivered_song_offer(
        monkeypatch, reply, expected_song):
    text = '第一首吧'
    catalog = {song_requests.song_key(title): Song(title, Path(f'{title}.wav'), None)
               for title in ('未推荐的排首曲目', 'Bad Guy', 'First Love', 'Always Online')}
    _, generate = _mock_song_turn(
        monkeypatch, text, catalog, None, reply,
        history=['[语音] 灰泽满：哈吉米没收录，换一首？AlwaysOnline、BadGuy还是FirstLove'],
    )
    response = await core.handle_chat('song-test', text, defer_memory=True, song_text=text)
    prompt = '\n'.join(m['content'] for m in generate.call_args.args[0]
                       if isinstance(m['content'], str))
    assert '1 "Always Online"（经典两句）' in prompt
    assert '2 "Bad Guy"（经典两句）' not in prompt
    if expected_song:
        assert response.actions['song'].title == 'Always Online'
        assert response.actions['song'].mode == 'clip'
    else:
        assert 'song' not in response.actions
        assert response == ('这次先不唱啦。' if '999' in reply else reply)


@pytest.mark.asyncio
async def test_main_reply_does_not_promise_an_unavailable_song(monkeypatch):
    text = '给我唱哈吉米'
    catalog = {'小星星': Song('小星星', Path('song.wav'), None)}
    _mock_song_turn(monkeypatch, text, catalog, {'mode': 'clip', 'title': '哈吉米'},
                    '灰泽满这就唱 [[唱歌:1]]')
    response = await core.handle_chat('song-test', text, defer_memory=True, song_text=text)
    assert 'song' not in response.actions
    assert '这就唱' not in response
    assert '哈吉米' in response


@pytest.mark.parametrize('text,previous,age', [
    ('第一首吧', '先聊哪一首？小星星、月亮', 0),
    ('第一首吧', '换一首？小星星、月亮', 601),
    ('第一首吧', '换一首？小星星、没收录的曲目', 0),
    ('第四首吧', '换一首？小星星、月亮', 0),
    ('第一首还是第二首', '换一首？小星星、月亮', 0),
    ('第一首吧\n算了', '换一首？小星星、月亮', 0),
])
@pytest.mark.asyncio
async def test_main_reply_does_not_use_an_unconfirmed_ordinal_to_authorize_a_song(
        monkeypatch, text, previous, age):
    catalog = {title: Song(title, Path(f'{title}.wav'), None) for title in ('小星星', '月亮')}
    classify, _ = _mock_song_turn(monkeypatch, text, catalog, {'mode': 'clip', 'title': ''},
                                  '[[唱歌:1]]', history=[f'灰泽满：{previous}'])
    monkeypatch.setattr(core, 'get_last_turn_gap_seconds', lambda _: age)
    response = await core.handle_chat('song-test', text, defer_memory=True, song_text=text)
    assert classify.call_args.kwargs.get('song_text') is None
    assert 'song' not in response.actions


@pytest.mark.asyncio
async def test_main_reply_blocks_an_immediate_singing_promise_without_a_confirmed_request(monkeypatch):
    text, reply = '那为什么还不唱', '在呢，这就唱。'
    _mock_song_turn(monkeypatch, text, {}, None, reply)
    response = await core.handle_chat('song-test', text, defer_memory=True, song_text=text)
    assert reply_style.clean_reply(response) == '在呢，这次先不唱啦。'
    assert 'song' not in response.actions


@pytest.mark.asyncio
async def test_main_reply_honors_refusal_to_send_the_full_version(monkeypatch):
    text, reply = '发小星星完整版', '我不发完整版，给你哼两句。'
    catalog = {'小星星': Song('小星星', Path('song.wav'), Path('song.m4a'))}
    _mock_song_turn(monkeypatch, text, catalog, {'mode': 'full', 'title': '小星星'}, reply)
    response = await core.handle_chat('song-test', text, defer_memory=True, song_text=text)
    assert reply_style.clean_reply(response) == '我不发完整版，想听经典两句吗？'
    assert 'song' not in response.actions


@pytest.mark.asyncio
async def test_accepting_song_without_tag_preserves_unrelated_reply_text(monkeypatch):
    text = '给我唱小星星'
    catalog = {'小星星': Song('小星星', Path('clip.wav'), None)}
    _mock_song_turn(monkeypatch, text, catalog, {'mode': 'clip', 'title': '小星星'},
                    '明天会下雨，记得带伞。这就唱。')
    response = await core.handle_chat('song-test', text, defer_memory=True, song_text=text)
    assert response == '明天会下雨，记得带伞。'
    assert response.actions['song'].title == '小星星'


@pytest.mark.parametrize('offer,number,mode,as_file,chosen', [
    ('换一首？"Always Online"、"Bad Guy"、"カタオモイ"', '第一首吧',
     'clip', False, 'Always Online'),
    ('换一首? Always Online, Make it!, カタオモイ', '第二首吧\n嗯嗯',
     'clip', False, 'Make it!'),
    ('想听哪首？Always Online，Bad Guy，カタオモイ', '第三首吧',
     'clip', False, 'カタオモイ'),
    ('换一首？《ドライフラワー（干花）》、《歌名（现场）》', '第二首吧',
     'clip', False, '歌名（现场）'),
    ('要听哪首完整版？Always Online、Bad Guy', '第二首吧',
     'full', False, 'Bad Guy'),
    ('要发哪首完整版文件？Always Online、Bad Guy', '第二首吧',
     'full', True, 'Bad Guy'),
])
@pytest.mark.asyncio
async def test_two_turn_song_offer_uses_the_actual_delivery_and_real_history(
        monkeypatch, tmp_path, offer, number, mode, as_file, chosen):
    first = '发哈吉米的完整版文件' if as_file else '给我唱哈吉米'
    catalog = {song_requests.song_key(title): Song(title, tmp_path / f'{title}.wav', tmp_path / f'{title}.m4a')
               for title in ('Always Online', 'Bad Guy', 'Make it!', 'カタオモイ',
                             'ドライフラワー（干花）', '歌名（现场）')}
    expected = catalog[song_requests.song_key(chosen)]
    audio = expected.full if mode == 'full' else expected.clip
    audio.write_bytes(b'audio')
    _, generate = _mock_song_turn(
        monkeypatch, first, catalog, {'mode': mode, 'title': '哈吉米', 'as_file': as_file},
        f'哈吉米没收录，{offer} [[语音]]',
    )
    generate.side_effect = [f'哈吉米没收录，{offer} [[语音]]', '[[唱歌:1]]']
    monkeypatch.setattr(core, 'get_user_history_timed', memory.get_user_history_timed)
    monkeypatch.setattr(cw, 'should_voice', lambda *_args, **_kw: True)
    monkeypatch.setattr(cw, 'SPLIT_REPLY_ENABLED', True)
    monkeypatch.setattr(cw, 'split_delay', lambda _text: 0)
    for name in ('summarize_batch', '_maybe_send_sticker', 'update_memory_v2_shadow_task'):
        monkeypatch.setattr(cw, name, AsyncMock())
    send_voice = AsyncMock(return_value=True)
    monkeypatch.setattr(cw, 'send_voice', send_voice)

    user = 'song-offer-roundtrip'
    win = cw._UserWindow(user)
    win.bot = AsyncMock()
    win.bot.send_private_msg.return_value = {'message_id': 123}
    win.pending = [(user, first, [])]
    await cw._flush(win)
    win.bot.send_private_msg.assert_awaited_once()
    win.bot.upload_private_file.assert_not_awaited()
    [menu] = win.bot.send_private_msg.await_args.kwargs['message']
    assert menu.type == 'text'
    assert f'《{chosen}》' in menu.data['text']
    win.pending = [(user, part, []) for part in number.splitlines()]
    await cw._flush(win)
    if as_file:
        win.bot.upload_private_file.assert_awaited_once_with(
            user_id=user, file=audio.as_uri(), name=audio.name,
        )
        win.bot.send_private_msg.assert_awaited_once()
    else:
        assert win.bot.send_private_msg.await_count == 2
        record = win.bot.send_private_msg.await_args.kwargs['message']
        assert record.type == 'record'
        assert record.data['file'] == audio.as_uri()
        win.bot.upload_private_file.assert_not_awaited()
    send_voice.assert_not_awaited()
    played = [part for turn in reply_evidence.recall_turns(user, terms=[chosen])
              for part in turn['parts'] if part['modality'] == 'song']
    assert len(played) == 1 and f'《{chosen}》' in played[0]['text']
