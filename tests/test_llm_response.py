"""`config.extract_chat_content`：LLM 响应的安全解析。

为什么值得单独测：以前 9 个调用点都直接写 `resp.choices[0].message.content.strip()`，
只在"响应形状永远正确"时才成立。实测踩过上游偶发返回 `choices: null`，
三个调用点**同时**报 `'NoneType' object is not subscriptable`——完全没法定位。
还有一个**静默**的坏形状：思考模式吃光 max_tokens → content 是空串、不报错，
功能悄悄失灵（记忆提取/行为分类拿到空串当"这次没结果"）。
"""
import pytest

from src.plugins.chatbot.config import extract_chat_content


class _Msg:
    def __init__(self, content, reasoning=None):
        self.content = content
        if reasoning is not None:
            self.reasoning_content = reasoning


class _Choice:
    def __init__(self, content, finish="stop", reasoning=None):
        self.message = _Msg(content, reasoning)
        self.finish_reason = finish


class _Resp:
    def __init__(self, choices):
        self.choices = choices


class TestExtractChatContent:
    def test_normal_content_returned_stripped(self):
        assert extract_chat_content(_Resp([_Choice("  在。  ")])) == "在。"

    def test_choices_none_raises_readable_error(self):
        # 上游偶发返回 choices: null —— 报错必须说人话，而不是 'NoneType' object is not subscriptable
        with pytest.raises(RuntimeError) as e:
            extract_chat_content(_Resp(None))
        assert "choices" in str(e.value)

    def test_choices_empty_list_raises(self):
        with pytest.raises(RuntimeError):
            extract_chat_content(_Resp([]))

    def test_content_none_raises(self):
        with pytest.raises(RuntimeError) as e:
            extract_chat_content(_Resp([_Choice(None)]))
        assert "content 为 None" in str(e.value)

    def test_empty_content_raises_with_thinking_hint(self):
        # 思考模式吃光 max_tokens 的典型形状：content=""、finish_reason=length、有 reasoning
        with pytest.raises(RuntimeError) as e:
            extract_chat_content(_Resp([_Choice("", finish="length", reasoning="想了很多" * 5)]))
        msg = str(e.value)
        assert "正文为空" in msg and "max_tokens" in msg

    def test_missing_message_attr_raises(self):
        class _Bare:
            choices = [object()]      # 没有 .message

        with pytest.raises(RuntimeError):
            extract_chat_content(_Bare())

    def test_missing_choices_attr_raises(self):
        with pytest.raises(RuntimeError):
            extract_chat_content(object())   # 完全没有 choices 属性

    def test_whitespace_only_content_raises(self):
        with pytest.raises(RuntimeError):
            extract_chat_content(_Resp([_Choice("   \n  ")]))
