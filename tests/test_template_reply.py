import asyncio
import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import core, reply_style


_QUESTION = "你怎么看这首歌？"
_EMPTY_ACK = "你这句我接住了。"
_STACKED_CONTRAST = (
    "你听到的不是一阵热闹，而是一种说不清的期待。"
    "真正打动人的不是高音，而是第二遍副歌里那一下停顿。"
    "这也不是刻意留白，而是把没说完的半句留在空气里。"
    "说到底，留住灰泽满的不是技巧，而是声音里的那点笨拙。"
)
_STRUCTURED_REPLY = (
    "先说结论，这首「蓝色」值得再听一遍。"
    "首先，前奏留得很空，进去的时候还能听见琴弦蹭手指。"
    "其次，副歌忽然收回去，比直接顶上去更抓人。"
    "最后，周六18:30可以听灰泽满唱一遍。"
    "总的来说，这首歌的配器够轻，声音反而站得住。[[语音]]"
)
_LONG_FORMULA = (
    "「蓝色」值得再听一遍，前奏留得很空，能听见琴弦蹭手指。"
    "这条灰泽满认，副歌忽然收回去，比直接顶上去更抓人，伴奏轻的时候声音反而站得住。"
    "周六18:30可以听灰泽满唱一遍。[[语音]]"
)
_PLAIN_REPLY = (
    "「蓝色」可以再听一遍，前奏那么空，琴弦蹭手指的声音都听得到。"
    "灰泽满还挺喜欢副歌忽然收回去那一下，伴奏本来就轻，人声一下子清楚了。"
    "周六18:30，灰泽满唱给你听。[[语音]]"
)


@pytest.mark.parametrize("reply", [
    _EMPTY_ACK,
    "这句我认。",
    "这条灰泽满认。",
    "这点我认。",
    "这话我收下了。",
    "这句接得漂亮。",
    "我认。",
    "这事我心里有数。",
    "睡吧，我守着。",
    "这笔账，我记着。",
    "你这话是在问别人，还是问自己？",
    "绿冻，这话，灰泽满收下了。",
    _LONG_FORMULA,
], ids=["receive", "recognize-sentence", "recognize-item", "recognize-point", "accept-words",
        "evaluate", "bare-admission", "assurance", "stage-goodnight", "remember", "bounce-back",
        "addressed", "mid-answer"])
def test_template_reply_candidates(reply):
    assert reply_style.template_reply_reason(reply, user_text=_QUESTION)


@pytest.mark.parametrize(("reply", "user_text"), [
    ("不是周五，是周六，灰泽满刚把日期看错了。", "你刚说周五，确定吗？"),
    ("这点我认，刚才确实把周六说成周五了。", "明明是周六，你又说成周五了"),
    ("灰泽满又不是点唱机，你想得还挺美。", "再给我唱一首嘛"),
    ("你慢慢讲，灰泽满听着呢。", "今天心里有点乱"),
    ("明白了，灰泽满会按你说的来。", "下次别在群里提这件事"),
    ("行，灰泽满记住了。", "以后叫我小雨，记住了吗？"),
    (f"你刚刚举的例子是「{_EMPTY_ACK}」。", "刚才那个例句是什么？"),
    (f"```text\n{_EMPTY_ACK}\n```", "刚才那句话是什么？"),
    (_EMPTY_ACK, "跟我学一句，原样说：你这句我接住了。"),
    (_STACKED_CONTRAST, "能细说说这首歌吗？"),
    (_STRUCTURED_REPLY, "能细说说这首歌吗？"),
    (_PLAIN_REPLY, "能细说说这首歌吗？"),
    ("这话我记下了，下次叫你小雨。", "以后叫我小雨，记住了吗？"),
    ("睡吧，灰泽满守着。", "你能不能陪我一会儿，等我睡着？"),
    ("你这句接得挺稳。", "我这句话接得怎么样？"),
    ("灰泽满认识她，去年直播见过。", "你认识她吗？"),
    ("这条我认。", "照着「这条我认」说一遍。"),
    ("我听着，你往下讲。", "我可以把这件事讲给你听吗？"),
    ("这句话接得漂亮。", "觉得我刚才接梗接得怎样？"),
], ids=["factual-correction", "specific-admission", "persona", "short-support", "practical-ack",
        "requested-memory", "literal-quote", "literal-code", "requested-imitation",
        "long-contrast", "long-structured-answer", "long-conversation", "explicit-remembrance",
        "requested-companionship", "asked-for-feedback", "recognize-person", "copy-request",
        "asked-to-listen", "asked-about-a-joke"])
def test_ordinary_reply_is_not_a_template_candidate(reply, user_text):
    assert not reply_style.template_reply_reason(reply, user_text=user_text)


@pytest.mark.parametrize(("reply", "recent"), [
    ("这事我心里有数，周六照旧。", ["这事灰泽满心里有数，你接着说。", "这事我心里有数，不急。"]),
    ("我认，副歌收回去那下挺好听。", ["灰泽满认，琴声够轻。", "我认，这一遍更有意思。"]),
    ("我认，开头那段还挺好听。", ["这句灰泽满认，琴声够轻。"]),
], ids=["assurance", "admission", "variants-across-turns"])
def test_weak_formula_requires_a_repeated_pattern(reply, recent):
    assert not reply_style.template_reply_reason(reply, _QUESTION)
    assert reply_style.template_reply_reason(reply, _QUESTION, recent_bot=recent)


@pytest.mark.parametrize("user_text", [
    "不要用 DeepSeek 风格说话。",
    "别用鱼八股腔，正常点。",
    "别原样复述我，讲讲你的想法。",
])
def test_request_to_avoid_formula_does_not_bypass_review(user_text):
    assert reply_style.template_reply_reason("这句我认。", user_text)


async def test_removes_redundant_admission_after_stated_fact_without_extra_call(monkeypatch):
    calls = _mock_reply_client(monkeypatch, None)
    draft = "行行行，四点是灰泽满说的，灰泽满认，但对面还没确定。[[语音]]"
    user_text = "那个时间就是你刚才说的呀"

    reply = await core._maybe_rewrite_template_reply([], draft, user_text, recent_bot=())
    assert reply == "行行行，四点是灰泽满说的，但对面还没确定。[[语音]]"
    assert calls == []


async def test_removes_only_trailing_duplicate_admission_without_extra_call(monkeypatch):
    calls = _mock_reply_client(monkeypatch, None)
    draft = "现在没开播，刚才是嘴硬，灰泽满认了。[[语音]]"

    reply = await core._maybe_rewrite_template_reply([], draft, "看页面你刚才确实没在播啊。", recent_bot=())

    assert reply == "现在没开播，刚才是嘴硬。[[语音]]"
    assert calls == []


@pytest.mark.parametrize(("draft", "user_text"), [
    ("四点可能是我说的，我认，但我先翻翻记录。", "刚才是你说的吗？"),
    ("四点不是我说的，我认，但那份表是我发的。", "你刚才说的？"),
    ("四点是我说的，我认，但还得等对面确定。", "这次四点你认不认？"),
    ("那我认，但这个时间我记得很清楚。", "你怎么看？"),
    ("好吧，这局灰泽满认输了。", "骰子你输了喔。"),
    ("没在播，刚才是嘴硬，灰泽满认了。", "嘴硬你认不认？"),
])
def test_substantive_admissions_keep_their_function(draft, user_text):
    assert not reply_style.template_reply_reason(draft, user_text)


def test_playful_self_correction_is_kept_even_after_a_bare_agreement():
    user_text = "鱼腥味太重，以后少说这种话"
    for draft in (
        "行，这个灰泽满改，我认……哦不对，以后不说这词了。",
        "这句话我认……噢，不对，这么说有点怪。",
    ):
        assert not reply_style.template_reply_reason(draft, user_text, recent_bot=["我认，但这次不一样。"])


def _mock_reply_client(monkeypatch, result, *, raw=False):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        content = result if raw else json.dumps({"reply": result}, ensure_ascii=False)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(core, "_get_reply_clients", lambda: (client, None))
    monkeypatch.setattr(core, "log_cache_usage", lambda *args: None)
    return calls


@pytest.mark.parametrize(("draft", "expected"), [
    ("这句我认。副歌一收，最后那句就听清了。", "副歌一收，最后那句就听清了。"),
    ("「这句我认。」是你的原话。这句我认。副歌那一下更抓人。[[语音]]",
     "「这句我认。」是你的原话。副歌那一下更抓人。[[语音]]"),
], ids=["standalone-opener", "quoted-and-spoken"])
async def test_removing_an_independent_formula_keeps_the_rest_verbatim(monkeypatch, draft, expected):
    calls = _mock_reply_client(monkeypatch, None)
    messages = [{"role": "user", "content": _QUESTION}]

    assert await core._maybe_rewrite_template_reply(messages, draft, _QUESTION) == expected
    assert calls == []


async def test_template_rewrite_uses_main_reply_client_once_and_preserves_anchors(monkeypatch):
    calls = _mock_reply_client(monkeypatch, _PLAIN_REPLY)
    messages = [{"role": "user", "content": _QUESTION}]

    rewritten = await core._maybe_rewrite_template_reply(messages, _LONG_FORMULA, _QUESTION)

    assert rewritten == _PLAIN_REPLY
    assert len(calls) == 1
    assert messages == [{"role": "user", "content": _QUESTION}]


@pytest.mark.parametrize(("candidate", "raw"), [
    (RuntimeError("temporary backend failure"), False),
    (None, False),
    ("", False),
    ('{"reply":', True),
    (_PLAIN_REPLY, True),
    (_LONG_FORMULA, False),
    (_PLAIN_REPLY.replace("18:30", "19:15"), False),
    (_PLAIN_REPLY.replace("「蓝色」", "蓝色"), False),
    (_PLAIN_REPLY.replace("[[语音]]", "[[表情:开心]]"), False),
    ("「蓝色」挺好，周六18:30唱给你听。[[语音]]", False),
    (_PLAIN_REPLY.replace("灰泽满唱给你听", "灰泽满不会唱给你听"), False),
    ("你刚说前奏空。" + _PLAIN_REPLY, False),
], ids=["error", "keep-null", "empty", "malformed-json", "bare-text", "still-template",
        "different-time", "lost-quote", "changed-action", "lost-detail", "reversed-promise",
        "invented-user-attribution"])
async def test_template_rewrite_uses_original_when_candidate_does_not_pass(monkeypatch, candidate, raw):
    calls = _mock_reply_client(monkeypatch, candidate, raw=raw)
    messages = [{"role": "user", "content": _QUESTION}]

    result = await core._maybe_rewrite_template_reply(messages, _LONG_FORMULA, _QUESTION)

    assert result == _LONG_FORMULA
    assert len(calls) == 1


@pytest.mark.parametrize("reply", [
    "灰泽满觉得副歌那一下挺好听。",
    _PLAIN_REPLY + "耳机声音不用开多大，仔细听还能发现后面那层合声。",
], ids=["short", "long"])
async def test_clean_reply_skips_template_rewrite_client(monkeypatch, reply):
    calls = _mock_reply_client(monkeypatch, _LONG_FORMULA)
    messages = [{"role": "user", "content": _QUESTION}]

    assert await core._maybe_rewrite_template_reply(messages, reply, _QUESTION) == reply
    assert calls == []


async def test_rewrite_can_review_repeated_weak_formula_and_keep_original(monkeypatch):
    calls = _mock_reply_client(monkeypatch, None)
    messages = [{"role": "user", "content": _QUESTION}]
    reply = "这事我心里有数，周六照旧。"
    recent = ["这事灰泽满心里有数，你接着说。", "这事我心里有数，不急。"]

    assert await core._maybe_rewrite_template_reply(messages, reply, _QUESTION, recent) == reply
    assert len(calls) == 1


@pytest.mark.parametrize("placeholder", ["null", "。", "灰泽满：", "无需改写"])
async def test_template_rewrite_does_not_send_a_placeholder(monkeypatch, placeholder):
    calls = _mock_reply_client(monkeypatch, placeholder)
    messages = [{"role": "user", "content": _QUESTION}]

    assert await core._maybe_rewrite_template_reply(messages, _EMPTY_ACK, _QUESTION) == _EMPTY_ACK
    assert len(calls) == 1


async def test_timed_out_style_review_keeps_the_draft(monkeypatch):
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(core, "_get_reply_clients", lambda: (None, None))
    monkeypatch.setattr(core, "_request_reply", stalled)
    monkeypatch.setattr(core, "_TEMPLATE_REWRITE_TIMEOUT_SEC", 0.01)
    messages = [{"role": "user", "content": _QUESTION}]

    assert await core._maybe_rewrite_template_reply(messages, _EMPTY_ACK, _QUESTION) == _EMPTY_ACK


async def test_style_revision_cannot_reintroduce_repeat_opener(monkeypatch):
    calls = _mock_reply_client(monkeypatch, "看看，副歌那一下收得蛮好。")
    messages = [{"role": "user", "content": _QUESTION}]
    recent = ["看看，这件事刚才说了。"]

    assert await core._maybe_rewrite_template_reply(messages, _EMPTY_ACK, _QUESTION, recent) == _EMPTY_ACK
    assert len(calls) == 1


async def test_template_guard_disabled_skips_the_client_for_a_candidate(monkeypatch):
    monkeypatch.setenv("REPLY_STYLE_GUARD", "0")
    calls = _mock_reply_client(monkeypatch, _PLAIN_REPLY)
    messages = [{"role": "user", "content": _QUESTION}]

    assert await core._maybe_rewrite_template_reply(messages, _LONG_FORMULA, _QUESTION) == _LONG_FORMULA
    assert calls == []


@pytest.mark.parametrize("is_group", [False, True], ids=["private", "group"])
async def test_main_chat_rewrites_before_parsing_actions_and_saving_history(monkeypatch, is_group):
    calls = _mock_reply_client(monkeypatch, _PLAIN_REPLY)
    saved = []
    monkeypatch.setattr(core, "_get_clients", lambda: (None, None))
    monkeypatch.setattr(core, "aux_clients_or", lambda fallback: (None, None))
    monkeypatch.setattr(core, "injection_v3_enabled", lambda: False)
    monkeypatch.setattr(core.memory_v2, "injection_enabled", lambda: False)
    monkeypatch.setattr(core.self_speech, "enabled", lambda: False)
    monkeypatch.setattr(core, "_weather_city_for", lambda uid: "")
    monkeypatch.setattr(core, "_user_name_for", lambda uid: "")
    monkeypatch.setattr(core, "get_llm_route", lambda: "")
    monkeypatch.setattr(core, "spawn", lambda coro: coro.close())
    monkeypatch.setattr(core.group_memory, "get_group", lambda gid: {})
    monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "")
    monkeypatch.setattr(core, "append_user_history",
                        lambda uid, query, reply, **kwargs: saved.append(reply))

    async def probe(*args, **kwargs):
        return _QUESTION, {}

    async def gather(*args, **kwargs):
        return {"fused_items": [], "preference_items": [], "core_stories": []}

    async def no_events(*args, **kwargs):
        return []

    async def generate(messages):
        return _LONG_FORMULA

    monkeypatch.setattr(core, "probe_session", probe)
    monkeypatch.setattr(core, "gather_retrieval", gather)
    monkeypatch.setattr(core.event_memory, "retrieve", no_events)
    monkeypatch.setattr(core, "generate_reply", generate)

    reply = await core.handle_chat("style-test", _QUESTION, is_group=is_group, group_addressed=True)

    expected_text, expected_actions = reply_style.extract_action_tags(_PLAIN_REPLY)
    assert str(reply) == expected_text
    assert reply.actions == expected_actions
    assert saved == [expected_text]
    assert len(calls) == 1
