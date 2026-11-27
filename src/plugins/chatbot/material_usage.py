"""材料用途与来源边界；选择通过不等于可以扩写原文没有的事实。"""
from dataclasses import dataclass


_SOURCE_LIMITS = {
    'corpus': ('历史记录不证明当前状态', '未记载不代表未发生'),
    'core_story': ('只讲原文记载的过程和过去，不补成结果或现状',),
    'preference': ('喜好和习惯不证明某次行为或未记载的具体对象',),
    'term': ('定义不证明角色亲历',),
    'association': ('只使用原文已有关系，不补造其他关系或亲历',),
    'wiki': ('客观资料不证明角色亲历',),
    'behavior': ('仅用于已确认的互动情境，不补造经历',),
    'event': ('用户历史不证明现状，待核对记录不作已确认事实',),
    'memory_hint': ('仅用于对应用户和本轮情境，不替代事实核对',),
}
_PURPOSE_LABELS = {
    'answer': '直接回答', 'past_example': '过去例子', 'correction': '纠正前提',
    'background': '背景', 'definition': '定义', 'interaction': '互动参考',
}


@dataclass(frozen=True)
class MaterialUse:
    source: str
    item_id: str
    purpose: str
    fact: str = ''
    quote: str = ''
    limits: tuple[str, ...] = ()

    def __post_init__(self):
        # 调用方可以增加限制，不能移除来源的默认边界。
        limits = (*_SOURCE_LIMITS.get(self.source, ()), *self.limits)
        object.__setattr__(self, 'limits', tuple(dict.fromkeys(limits)))


def render_material_use(use: MaterialUse) -> str:
    """短注只声明用途、获准事实及边界；完整来源正文仍由原出口提供。"""
    purpose = _PURPOSE_LABELS.get(use.purpose, use.purpose)
    content = f'允许事实：{use.fact}' if use.fact else '允许内容：仅限随附原文'
    boundary = '；边界：' + '；'.join(use.limits) if use.limits else ''
    return f'【材料用途：{purpose}】{content}{boundary}'
