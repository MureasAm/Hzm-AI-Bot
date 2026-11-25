"""逐轮记忆提示必须经过同一份语境的明确许可；匹配原字符串不构成许可。"""
import importlib
import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot.retrieval_context import RetrievalContext, Selection


async def test_memory_hint_reaches_real_shared_selector_and_preserves_purpose():
    prompts = []
    async def create(**kwargs):
        prompts.append(kwargs['messages'])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"keep":["1"]}'))], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    selected = await _module().filter_memory_hints(client, RetrievalContext('晚安'), {'ritual_hint': '睡前的固定约定'})
    assert selected == {'ritual_hint': '睡前的固定约定'}
    candidate = json.loads(prompts[0][-1]['content'])['候选材料'][0]
    assert candidate['来源'] == 'memory_hint' and candidate['属性']['purpose'] == 'ritual_hint'


def _module():
    return importlib.import_module("src.plugins.chatbot.memory_context")


@pytest.mark.parametrize("hints", [{}, {"habit_hint": "", "ritual_hint": " \n "}])
async def test_empty_memory_hints_need_no_selection(monkeypatch, hints):
    module = _module()
    calls = []

    async def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("empty hints must not invoke selection")

    monkeypatch.setattr(module, "select_materials", unexpected)
    assert await module.filter_memory_hints(object(), RetrievalContext("你好"), hints) == {}
    assert calls == []


async def test_memory_hints_share_original_context_and_preserve_approved_text(monkeypatch):
    module = _module()
    context = RetrievalContext(
        "电影里那句‘姐姐晚安’挺有意思", resolved_text="讨论电影中的称呼和台词",
        history="用户：我们刚才说到电影人物，不是在说我和你的约定",
        target="电影人物", speech_act="quotation", proposition="引用电影台词",
    )
    hints = {"ritual_hint": "【固定期待】给用户说晚安", "habit_hint": "  【习惯】用户喜欢聊电影\n",
             "nickname_hint": "【称呼线索】用户以前叫过姐姐", "recall_context": ""}
    original = dict(hints)
    calls = []

    async def choose(client, passed_context, candidates):
        calls.append((client, passed_context, candidates))
        return Selection(selected=frozenset({("memory_hint", "habit_hint")}))

    client = object()
    monkeypatch.setattr(module, "select_materials", choose)
    selected = await module.filter_memory_hints(client, context, hints)

    assert selected == {"habit_hint": original["habit_hint"]}
    assert hints == original
    assert len(calls) == 1 and calls[0][:2] == (client, context)
    note = json.loads(calls[0][1].note())
    assert note["原话"] == context.raw_text and note["最近对话"] == context.history
    assert note["指向对象"] == "电影人物" and note["言语行为"] == "quotation"
    candidates = calls[0][2]
    assert [(c.source, c.item_id, c.text, c.metadata) for c in candidates] == [
        ("memory_hint", key, value, {"purpose": key})
        for key, value in original.items() if value.strip()
    ]


async def test_unavailable_selection_clears_even_a_named_but_unverified_hint(monkeypatch):
    module = _module()

    async def unavailable(*args):
        return Selection(selected=frozenset({("memory_hint", "commitment_context")}), available=False)

    monkeypatch.setattr(module, "select_materials", unavailable)
    assert await module.filter_memory_hints(object(), RetrievalContext("引用别人说的‘记得提醒我’"),
                                            {"commitment_context": "【未完成约定】提醒用户休息"}) == {}


@pytest.mark.parametrize("text", ["【固定期待】晚安仪式", "【称呼线索】姐姐", "【你记得的事】早上跑操"])
async def test_nonempty_legacy_hint_strings_are_not_automatically_authorized(monkeypatch, text):
    module = _module()

    async def reject(*args):
        return Selection()

    monkeypatch.setattr(module, "select_materials", reject)
    assert await module.filter_memory_hints(object(), RetrievalContext("朋友转发了一句‘早安姐姐’"),
                                            {"habit_hint": text}) == {}


async def test_other_sources_or_invented_hint_keys_cannot_authorize_output(monkeypatch):
    module = _module()

    async def mismatched(*args):
        return Selection(selected=frozenset({("event", "nickname_hint"),
                                             ("memory_hint", "invented_key")}))

    monkeypatch.setattr(module, "select_materials", mismatched)
    assert await module.filter_memory_hints(object(), RetrievalContext("姐姐这两个字怎么用"),
                                            {"nickname_hint": "用户过去叫过姐姐"}) == {}
