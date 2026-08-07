"""人格规则加载与语义行为匹配。

行为匹配的 trigger 向量已离线预计算到 trigger_vectors.json，
运行时只对用户消息计算 1 次 embedding，避免逐条调用 embedding API。
"""
import json

from .constants import (
    TRAITS_FILE, STYLES_FILE, BEHAVIORS_FILE,
    TRIGGER_VECTOR_FILE, BEHAVIOR_MATCH_THRESHOLD,
)
from .rag import cosine_similarity


def load_persona_rules():
    """读取人格规则三件套：traits / styles / behaviors。"""
    traits_text = []
    styles_text = []
    behaviors = []

    if TRAITS_FILE.exists():
        try:
            with open(TRAITS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    name = item.get("name", "")
                    desc = item.get("description", "")
                    if name or desc:
                        traits_text.append(f"{name}: {desc}" if name else desc)
        except Exception as e:
            print(f"⚠️ 读取 traits 失败: {e}")

    if STYLES_FILE.exists():
        try:
            with open(STYLES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    name = item.get("name", "")
                    desc = item.get("description", "")
                    if name or desc:
                        styles_text.append(f"{name}: {desc}" if name else desc)
        except Exception as e:
            print(f"⚠️ 读取 styles 失败: {e}")

    if BEHAVIORS_FILE.exists():
        try:
            with open(BEHAVIORS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    behaviors = data
                elif isinstance(data, dict):
                    behaviors = [data]
        except Exception as e:
            print(f"⚠️ 读取 behaviors 失败: {e}")

    return traits_text, styles_text, behaviors


def build_global_persona_context(traits, styles):
    context_parts = []
    if traits:
        context_parts.append("【性格基底】\n" + "\n".join([f"- {t}" for t in traits]))
    if styles:
        context_parts.append("【语言风格】\n" + "\n".join([f"- {s}" for s in styles]))
    return "\n".join(context_parts) if context_parts else ""


# 缓存的 trigger → 向量 映射（模块级，只加载一次）
_trigger_vectors = None


def load_trigger_vectors() -> dict:
    """加载预计算的 trigger 向量缓存。文件缺失时返回空字典。"""
    global _trigger_vectors
    if _trigger_vectors is not None:
        return _trigger_vectors
    if not TRIGGER_VECTOR_FILE.exists():
        _trigger_vectors = {}
        return _trigger_vectors
    try:
        with open(TRIGGER_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _trigger_vectors = data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        _trigger_vectors = {}
    return _trigger_vectors


async def match_behaviors_semantic(user_query: str, query_vector, behaviors: list,
                                   threshold: float = BEHAVIOR_MATCH_THRESHOLD) -> str:
    """基于缓存的 trigger 向量匹配行为规则。

    query_vector 由调用方传入（已在 RAG 阶段算好），此处不再调用 embedding API。
    """
    if not behaviors or query_vector is None:
        return ""

    trigger_vectors = load_trigger_vectors()

    best_sim = -1
    best_rule = None
    for b in behaviors:
        t = b.get("trigger", "")
        if not t:
            continue
        trigger_vec = trigger_vectors.get(t)
        if trigger_vec is None:
            continue  # 该 trigger 未预计算，跳过
        sim = cosine_similarity(query_vector, trigger_vec)
        if sim > best_sim:
            best_sim = sim
            best_rule = b

    if best_rule and best_sim >= threshold:
        return _format_behavior_rule(best_rule)
    return ""


def _format_behavior_rule(rule: dict) -> str:
    """将一条行为规则格式化为注入文本。供 match_behaviors_semantic 与检索层共用。"""
    name = rule.get("name", "")
    desc = rule.get("response", "")
    trigger_desc = rule.get("trigger", "")
    parts = []
    if name:
        parts.append(f"【{name}】")
    if trigger_desc:
        parts.append(f"触发情境：{trigger_desc}")
    if desc:
        parts.append(f"回应模式：{desc}")
    return "\n".join(parts)
