"""给辅助模型看的压缩术语提示。

主回复链路有 `07_terms` 注入；probe / 行为分类 / 群聊门 / 语料判定 / 自述转述 / 记忆提取
这些辅助调用以前只看原文，遇到「考」「棉花糖」这类圈内词会按字面理解。
这里只挑**当前文本里真的命中的词**，压缩成一段短提示，供各辅助 prompt 附加。
"""
import re

from .persona import load_terms


def terms_hint(text: str, max_terms: int = 3, max_chars: int = 320) -> str:
    """返回「词：含义；用法：…」形式的短提示；没有命中返回空串。"""
    msg = str(text or "")
    if not msg.strip():
        return ""
    direct: list[str] = []
    concept: list[str] = []
    seen: set[str] = set()
    for t in load_terms():
        kw = str(t.get("keyword") or "").strip()
        if not kw or kw in seen:
            continue
        pattern = t.get("pattern")
        keys = [kw] + [str(a) for a in t.get("aliases") or [] if a]
        if t.get("pattern_only") and pattern:
            hit = bool(re.search(pattern, msg))
        else:
            hit = any(k in msg for k in keys) or bool(pattern and re.search(pattern, msg))
        concept_hit = any(w in msg for w in (t.get("usage_triggers") or []))
        if not (hit or concept_hit):
            continue
        seen.add(kw)
        parts = [str(t.get("meaning") or "").strip()]
        usage = str(t.get("usage") or "").strip()
        if usage:
            parts.append(f"用法：{usage}")
        line = f"{kw}：{'；'.join(p for p in parts if p)}"
        (direct if hit else concept).append(line)

    out = "；".join((direct + concept)[:max_terms])
    if len(out) > max_chars:
        out = out[:max_chars].rstrip("，,、；; ") + "…"
    return out


def terms_hint_block(text: str, max_terms: int = 3, max_chars: int = 320) -> str:
    """带段头的版本，直接拼到辅助 prompt 末尾。"""
    hint = terms_hint(text, max_terms=max_terms, max_chars=max_chars)
    if not hint:
        return ""
    return f"\n\n【术语表·只用于理解，不要照抄】\n{hint}"