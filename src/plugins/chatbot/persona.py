"""人格规则加载：traits/styles/behaviors + 周表 + terms 名词库。

behaviors **不走向量**（L3 改为「判别词 + LLM 意图分类」，见 retrieval.select_behavior_item），
所以改 behaviors.json 后不需要重算任何向量（旧文档提的 trigger_vectors.json 已废弃）。
"""
import json

from .constants import (
    TRAITS_FILE, STYLES_FILE, BEHAVIORS_FILE,
    TERMS_FILE, SCHEDULE_FILE,
)


_terms_cache = None
_schedule_cache = None


def load_terms() -> list:
    """加载 persona/world/terms.json 名词库（模块级缓存）。"""
    global _terms_cache
    if _terms_cache is not None:
        return _terms_cache
    if not TERMS_FILE.exists():
        _terms_cache = []
        return _terms_cache
    try:
        data = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
        _terms_cache = data.get("terms", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, OSError):
        _terms_cache = []
    return _terms_cache


def load_schedule():
    """加载 persona/world/schedule.json（只有周表，手动维护）。

    周表是回答"明天/这周/几点播"类问题的**地面真值**——记忆里带"明天/下周"的话
    是过去某场直播当时的说法，可能早已过期，不能当现在的安排答。
    weekly 是 weekday 制的固定表，到周自动对，**永远不会过期**，这是它比
    "近况"可靠的地方（后者已删除，见 待办清单.md）。
    """
    global _schedule_cache
    if _schedule_cache is not None:
        return _schedule_cache
    _schedule_cache = {}
    if SCHEDULE_FILE.exists():
        try:
            data = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
            _schedule_cache = data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            _schedule_cache = {}
    return _schedule_cache


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


def _format_behavior_rule(rule: dict) -> str:
    """将一条行为规则格式化为注入文本。供检索层（select_behavior_item）使用。

    格式含 samples（真人原话示范，few-shot）：
    - response 描述"**做什么**"（什么情境怎么反应），**不规定"用哪个词"**
      ——措辞是台词，按黄金律第 1 条归 samples 承担。曾把"'呃…'起头"写进 response，
      配上注入端的"请严格按此模式回应"，导致同一情境反复触发时每次都同一个开头
      （实测 14 条回复里 8 条以"呃，"开头，而 86 条真实素材里 0 条这么开头）。
    - samples 给"她真实怎么说话"的示范。**来自素材，但允许本土化同义改写**
      （素材是直播弹幕语境、这里是私聊，语境变了就要适配；这不是"素材掺假"）。
      判据见 待办清单.md 的「本土化」条：携带**某一回的具体情境**的样本会被模型当
      **现在**复用（"明天要搬家"→ 三个月后还在说搬家），要改写成"她一贯如何"。
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
