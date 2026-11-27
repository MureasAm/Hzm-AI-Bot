from dataclasses import FrozenInstanceError

import pytest

from src.plugins.chatbot.material_usage import MaterialUse, render_material_use


def test_historical_fact_keeps_source_boundary_and_custom_limit_in_rendered_note():
    use = MaterialUse('corpus', 'activity', 'past_example',
                      fact='她此前说曾熬夜做模型', quote='聊到以前熬夜做模型',
                      limits=('不能说成每天都这样',))
    note = render_material_use(use)
    assert '过去例子' in note and '她此前说曾熬夜做模型' in note
    assert '不证明当前状态' in note and '不能说成每天都这样' in note
    assert '未记载不代表未发生' in note
    assert '聊到以前熬夜做模型' not in note
    with pytest.raises(FrozenInstanceError):
        use.purpose = 'answer'


@pytest.mark.parametrize('source,purpose,required', [
    ('core_story', 'past_example', ('原文记载的过程', '不补成结果')),
    ('preference', 'background', ('喜好和习惯', '不证明某次行为')),
    ('term', 'definition', ('定义', '不证明角色亲历')),
    ('association', 'background', ('已有关系', '不补造其他关系')),
])
def test_other_source_notes_keep_their_own_evidence_scope(source, purpose, required):
    use = MaterialUse(source, 'item', purpose)
    note = render_material_use(use)
    assert all(text in note for text in required)
    assert '仅限随附原文' in note and len(note) < 150


def test_custom_limits_cannot_remove_default_and_correction_does_not_confirm_premise():
    use = MaterialUse('corpus', 'excuse', 'correction', fact='她曾为脱身谎称熟识作家',
                      limits=('不确认作家身份', '历史记录不证明当前状态'))
    note = render_material_use(use)
    assert '纠正前提' in note and '谎称熟识' in note and '不确认作家身份' in note
    assert note.count('历史记录不证明当前状态') == 1
    assert '未记载不代表未发生' in use.limits


def test_note_keeps_complete_approved_fact_instead_of_cutting_off_its_negation():
    fact = '她此前提到' + '那次旅行需要分几段说明，' * 20 + '但没有去过那个城市。'
    assert fact in render_material_use(MaterialUse('corpus', 'trip', 'correction', fact=fact))
