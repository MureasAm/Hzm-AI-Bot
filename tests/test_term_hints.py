"""辅助链路术语提示：命中才带、梗词不被字面误伤。"""
from src.plugins.chatbot import term_hints as th


def test_kao_meme_hits():
    assert "考" in th.terms_hint("你也太考了")
    assert "考" in th.terms_hint("考子报道")
    assert "考" in th.terms_hint("又开考了")


def test_kao_does_not_hit_exam_words():
    assert th.terms_hint("考试") == ""
    assert th.terms_hint("思考") == ""
    assert th.terms_hint("考据") == ""


def test_kao_hits_from_filial_concept():
    assert "考" in th.terms_hint("你也太孝了")


def test_marshmallow_hits():
    assert "棉花糖" in th.terms_hint("今晚棉花糖回")
    assert "棉花糖" in th.terms_hint("读棉花糖环节")


def test_block_has_header_only_on_hit():
    assert th.terms_hint_block("考子报道").startswith("\n\n【术语表")
    assert th.terms_hint_block("普通闲聊") == ""