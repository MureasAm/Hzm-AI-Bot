"""给辅助模型看的压缩术语提示。

主回复链路有 `07_terms` 注入；probe / 行为分类 / 群聊门 / 语料判定 / 自述转述 / 记忆提取
这些辅助调用以前只看原文，遇到「考」「棉花糖」这类圈内词会按字面理解。
这里只挑当前文本里命中词形或场景线索的候选，供辅助 prompt 结合上下文辨义。
"""
import re

from .persona import load_terms


def term_match(term: dict, text: str) -> tuple[bool, bool]:
    """纯词形匹配，返回（直接命中，场景线索命中）；均不代表该义项已确认。"""
    msg = str(text or "")
    keyword = str(term.get("keyword") or "").strip()
    if not msg.strip() or not keyword:
        return False, False
    pattern_hit = False
    pattern = term.get("pattern")
    if pattern:
        try:
            pattern_hit = bool(re.search(pattern, msg))
        except (re.error, TypeError):
            # 坏模式不影响其它术语，也不能绕开 pattern_only 的限制。
            pass
    keys = [keyword] + [str(a).strip() for a in term.get("aliases") or [] if a]
    direct = pattern_hit if term.get("pattern_only") else (
        pattern_hit or any(key and key in msg for key in keys))
    concept = any(str(word).strip() and str(word).strip() in msg
                  for word in term.get("usage_triggers") or [] if word)
    return bool(direct), bool(concept)


def term_candidates(text: str) -> list[dict]:
    """返回全部匹配术语及完整属性，直接命中优先；不在此处裁决实际词义。"""
    direct: list[dict] = []
    concept: list[dict] = []
    seen: set[str] = set()
    for term in load_terms():
        kw = str(term.get("keyword") or "").strip()
        if not kw or kw in seen:
            continue
        hit, concept_hit = term_match(term, text)
        if not (hit or concept_hit):
            continue
        seen.add(kw)
        (direct if hit else concept).append(dict(term))
    return direct + concept


def terms_hint(text: str, max_terms: int = 3, max_chars: int = 320) -> str:
    """返回「词（候选释义）：含义；用法：…」短提示；没有命中返回空串。"""
    lines = []
    for t in term_candidates(text)[:max_terms]:
        kw = str(t.get("keyword") or "").strip()
        parts = [str(t.get("meaning") or "").strip()]
        usage = str(t.get("usage") or "").strip()
        if usage:
            parts.append(f"用法：{usage}")
        lines.append(f"{kw}（候选释义）：{'；'.join(p for p in parts if p)}")

    out = "；".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip("，,、；; ") + "…"
    return out


def terms_hint_block(text: str, max_terms: int = 3, max_chars: int = 320) -> str:
    """带段头的版本，直接拼到辅助 prompt 末尾。"""
    hint = terms_hint(text, max_terms=max_terms, max_chars=max_chars)
    if not hint:
        return ""
    return ("\n\n【术语表·候选释义，只用于理解，不要照抄】\n"
            "词形或场景线索命中不能据此确认用户采用了该含义，须结合原话上下文判断。\n"
            f"{hint}")
