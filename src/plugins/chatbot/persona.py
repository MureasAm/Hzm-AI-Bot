"""人格规则加载：traits/styles/behaviors + trigger 向量缓存。

trigger 向量已离线预计算到 trigger_vectors.json，运行时读缓存。
"""
import json

from .constants import (
    TRAITS_FILE, STYLES_FILE, BEHAVIORS_FILE,
    TRIGGER_VECTOR_FILE,
)


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


def _format_behavior_rule(rule: dict) -> str:
    """将一条行为规则格式化为注入文本。供检索层（select_behavior_item）使用。

    新版格式含 samples（真人原话示范，few-shot）：
    - response 给行为指令
    - samples 给"她真实怎么说话"的示范（必须来自素材原文，不得改写）
    """
    name = rule.get("name", "")
    desc = rule.get("response", "")
    trigger_desc = rule.get("trigger", "")
    samples = rule.get("samples", [])
    parts = []
    if name:
        parts.append(f"【{name}】")
    if trigger_desc:
        parts.append(f"触发情境：{trigger_desc}")
    if desc:
        parts.append(f"回应模式：{desc}")
    if samples:
        # 真人原话示范：模型照这个腔调学，不自己发明
        sample_lines = []
        for s in samples:
            u = s.get("user", "")
            r = s.get("reply", "")
            if u and r:
                sample_lines.append(f"  粉丝说：{u} → 灰泽满：{r}")
        if sample_lines:
            parts.append("她这么说过（照着学腔调，不自己发明）：\n" + "\n".join(sample_lines))
    return "\n".join(parts)
