"""回复链逐轮记忆提示的适用性过滤；不读写记忆，不用于主动广播。"""
from .retrieval_context import MaterialCandidate, RetrievalContext, select_materials


async def filter_memory_hints(client, context: RetrievalContext,
                              hints: dict[str, str]) -> dict[str, str]:
    """按共用语境一次裁决提示，只返回明确获准项的原文。

    仅处理本轮动作/回查提示；稳定身份卡、称呼频率预算不由此层过滤。
    模型请求中的每项正文由共享选择器统一限到 650 字，返回文本不改写。
    """
    candidates = [
        MaterialCandidate(source="memory_hint", item_id=key, text=value,
                          metadata={"purpose": key})
        for key, value in hints.items() if isinstance(value, str) and value.strip()
    ]
    if not candidates:
        return {}
    selection = await select_materials(client, context, candidates)
    if not selection.available:
        return {}
    return {candidate.item_id: candidate.text for candidate in candidates
            if ("memory_hint", candidate.item_id) in selection.selected}
