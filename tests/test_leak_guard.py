# -*- coding: utf-8 -*-
"""规划式泄露兜底的判据测试（使用虚构文本）。"""
import pytest

from src.plugins.chatbot.reply_style import planning_leak_reason


def test_production_planning_leak_is_flagged():
    # 虚构草稿，覆盖生成规划泄漏的标记组合。
    text = (
        "可以先接住这句，系统要求用角色口吻。灰泽满可以回得柔和一点，"
        "也可以直接回答，注意默认短回复。"

    )
    assert planning_leak_reason(text)


def test_reference_tone_planning_is_flagged():
    # 2026-11-14 批量复现样本（节选）
    text = (
        "灰泽满又得把之前说过的“不展开、不追问、不陪着沉下去”拿出来用了，"
        "不过这次对方是问“如果是你会怎么处理”，不是诉苦，是请求建议，可以稍微多说两句。"
        "但保持她的风格：不端，不灌鸡汤，带点自嘲和嘴硬。\n\n参考腔调：\n"
        "- “第二选择怎么了？”\n- “懂你。……香蕉吗？”"
    )
    assert planning_leak_reason(text)


def test_third_person_narration_is_flagged():
    assert planning_leak_reason("灰泽满没忍住笑出了声，带着点欠揍的语气回：行行行，记住了。")
    assert planning_leak_reason("灰泽满盯着屏幕看了两秒，突然有点想笑。")


def test_english_reasoning_is_flagged():
    assert planning_leak_reason("The user is asking for advice. Let me keep the reply short.")


def test_bullet_list_planning_is_flagged():
    text = "可以这样回：\n- 先安慰一句\n- 再转开话题"
    assert planning_leak_reason(text)


def test_list_with_internal_reply_notes_is_flagged():
    # 保留 11 月 22 日真实泄露的结构，示例内容使用合成文本。
    text = (
        "面前的问题要先顺着主题接一句。\n当前用户在问晚饭。\n\n"
        "1. 语气：偏随意。\n2. 限制：不替对方决定。\n3. 示例参考：吃面也行呀。\n\n"
        "吃面也行呀。"
    )
    assert planning_leak_reason(text) == "列表式草稿"
    markdown = text.replace("当前用户", "**当前用户**")
    for number in (1, 2, 3):
        markdown = markdown.replace(f"{number}. ", f"{number}) ")
    assert planning_leak_reason(markdown) == "列表式草稿"


def test_normal_replies_are_not_flagged():
    normal = [
        "嗯，好多了就好。那早点睡吧，明天周六在家玩一整天。",
        "吵架嘛，灰泽满会先冷下来，别气头上说话。等心情缓过来了，看看是谁的错。",
        "你应该早点休息，别熬太晚了。",
        "要么先去睡，要么再聊一会儿，你自己选。",
        "灰泽满觉得你这话说得有点重了，但他应该不是故意的。",
        "那就好。……不过下次别一个人憋着。",
        "可以这样回：\n- 「谢谢你告诉我。」\n- 「明天聊也行，你先忙。」",
        "可以这样回复：\n- 先安慰一句，告诉对方你愿意听。\n- 再换个话题，给他一点缓冲。",
        "约人见面可以参考这三点。\n1. 语气：轻松。\n2. 限制：说清时间。\n3. 示例参考：周五见？",
    ]
    for text in normal:
        assert planning_leak_reason(text) == "", text


def test_empty_and_short_text():
    assert planning_leak_reason("") == ""
    assert planning_leak_reason("嗯。") == ""


# ==================== 兜底流程（generate_reply） ====================

class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeCompletions:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return _FakeResp(self._contents.pop(0))


class _FakeClient:
    def __init__(self, contents):
        self.chat = type("C", (), {"completions": _FakeCompletions(contents)})()


def _patch_core(monkeypatch, contents):
    from src.plugins.chatbot import core
    fake = _FakeClient(contents)
    monkeypatch.setattr(core, "_get_clients", lambda: (fake, None))
    monkeypatch.setattr(core, "_get_reply_clients", lambda: (fake, None))
    monkeypatch.setattr(core, "_get_model_name", lambda: "fake")
    monkeypatch.setattr(core, "_get_reply_model", lambda: "fake")
    monkeypatch.setattr(core, "log_cache_usage", lambda *a, **k: None)
    monkeypatch.setattr(core, "extract_chat_content", lambda r: r.content)
    monkeypatch.setattr(core, "log_leak_guard", lambda *a, **k: None)
    return core, fake


LEAK = "这句先接住，系统要求尽量别再叫名字，灰泽满可以回：那就好。"
CLEAN = "嗯，那就好，早点睡吧。"


def test_generate_reply_rewrites_leak(monkeypatch):
    core, fake = _patch_core(monkeypatch, [LEAK, CLEAN])
    out = __import__("asyncio").run(core.generate_reply([{"role": "user", "content": "hi"}]))
    assert out == CLEAN
    assert fake.chat.completions.calls == 2


def test_generate_reply_falls_back_when_rewrite_still_leaks(monkeypatch, capsys):
    core, fake = _patch_core(monkeypatch, [LEAK, LEAK])
    out = __import__("asyncio").run(core.generate_reply([{"role": "user", "content": "hi"}]))
    assert out == core._FALLBACK_REPLY
    assert fake.chat.completions.calls == 2
    assert "主回复重写后仍混入规划/旁白，使用兜底回复" in capsys.readouterr().out


def test_generate_reply_does_not_retry_clean_reply(monkeypatch):
    core, fake = _patch_core(monkeypatch, [CLEAN])
    out = __import__("asyncio").run(core.generate_reply([{"role": "user", "content": "hi"}]))
    assert out == CLEAN
    assert fake.chat.completions.calls == 1


@pytest.mark.parametrize("body", [
    "1. 把电脑关掉。\n2. 拔掉电源再拆外壳。",
    "- 去之前先打个电话。\n- 到了记得带上取件码。",
], ids=["numbered", "bulleted"])
@pytest.mark.parametrize("first_leaks", [False, True], ids=["first-draft", "after-rewrite"])
async def test_generate_reply_accepts_plain_list_content(monkeypatch, body, first_leaks):
    contents = [LEAK, body] if first_leaks else [body, CLEAN]
    core, fake = _patch_core(monkeypatch, contents)

    result = await core.generate_reply([{"role": "user", "content": "分两点讲就好"}])

    assert result == body
    assert fake.chat.completions.calls == (2 if first_leaks else 1)
