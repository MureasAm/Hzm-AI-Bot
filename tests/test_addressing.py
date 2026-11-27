import pytest

from src.plugins.chatbot import addressing


@pytest.mark.parametrize(("text", "name"), [
    ("要是害羞可以叫我绿冻宝宝", "绿冻宝宝"),
    ("我的本名是「小明」，网名是「风铃」", "风铃"),
    ("总之，你经常叫我学长或者机长，这是我们之间的称呼", "学长、机长"),
    ("我的网名是千纸鸢，你常叫我的外号是“学长”“机长”", "学长、机长"),
    ("以后叫我学长就行，外号可以叫机长", "学长、机长"),
    ("姐姐，早\n我叫小明", "小明"),
    ("你能叫我小鱼吗？", "小鱼"),
])
def test_name_is_bound_to_the_user_in_the_original_clause(text, name):
    assert addressing.user_address_matches(text, name)


@pytest.mark.parametrize(("text", "name"), [
    ("我可以叫你姐姐吗？", "姐姐"),
    ("姐姐，早\n我叫小明", "姐姐"),
    ("他说：『以后叫我小鱼』", "小鱼"),
    ("[引用灰泽满自己说的：「你可以叫我姐姐」] 哈哈", "姐姐"),
    ("她的名字叫阿梨", "阿梨"),
    ("我朋友挺可爱的，名字叫阿梨", "阿梨"),
    ("我叫小明。我朋友到了，名字叫阿梨", "阿梨"),
    ("我朋友说，叫我姐姐", "姐姐"),
    ("我叫小鱼吗？", "小鱼"),
    ("以后叫我阿狗", "阿猫"),
    ("如果你叫我姐姐的话，气氛会很奇怪", "姐姐"),
    ("以后别叫我小鱼", "小鱼"),
    ("你从前叫我姐姐吗？", "姐姐"),
])
def test_crossed_quoted_and_hypothetical_names_have_no_binding(text, name):
    assert not addressing.user_address_matches(text, name)


def test_alias_definition_is_current_assertion_not_a_guess():
    assert addressing.her_alias_definition("嗯嗯，姐姐就是我的宝宝") == "宝宝"
    assert addressing.her_alias_definition("我说的xny就是你啊") == "xny"
    assert not addressing.her_alias_definition("宝宝是指hzm嘛")
    assert not addressing.her_alias_definition("我姐姐就是我的宝宝")
    assert not addressing.her_alias_definition("那她就是我的宝宝")
    assert not addressing.her_alias_definition("他说：「宝宝就是你」")


def test_current_note_keeps_the_speaker_and_target_in_generation_view():
    assert "用户想念灰泽满" in addressing.perspective_note(
        "那姐姐我想你啦怎么办，我想你夸夸我"
    )
    assert "灰泽满是否喜欢用户" in addressing.perspective_note("嗯嗯，姐姐喜欢我嘛")
    assert "「宝宝」指灰泽满" in addressing.perspective_note("嗯嗯，姐姐就是我的宝宝")
    assert not addressing.perspective_note("[引用灰泽满自己说的：「我想你啦」] 我看到了")
    assert not addressing.perspective_note("如果我想你怎么办")
    assert not addressing.perspective_note("我姐姐喜欢我做的菜")
    assert not addressing.perspective_note("灰泽满喜欢我。")
    assert not addressing.perspective_note("我爱你？")
