"""人格规则加载：traits/styles/behaviors + trigger 向量缓存 + terms 名词库。

trigger 向量已离线预计算到 trigger_vectors.json，运行时读缓存。
"""
import json
from datetime import date, datetime

from .constants import (
    TRAITS_FILE, STYLES_FILE, BEHAVIORS_FILE,
    TERMS_FILE, SCHEDULE_FILE, SCHEDULE_NOTE_TTL_DAYS,
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
    """加载 persona/world/schedule.json（周表+近况，手动维护）。

    周表是回答"明天/这周/几点播"类问题的**地面真值**——记忆里带"明天/下周"的话
    是过去某场直播当时的说法，可能早已过期，不能当现在的安排答。
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


def schedule_note_age_days(sched: dict) -> float | None:
    """近况（"这周在收拾搬家"这类临时事实）距今多少天。

    返回 None = 没有近况或日期缺失/解析不了（调用方当作"不可信"，别当当前事实用）。
    """
    if not isinstance(sched, dict):
        return None
    raw = str(sched.get("近况_updated") or "").strip()
    if not raw:
        return None
    try:
        d = date.fromisoformat(raw) if len(raw) == 10 else datetime.fromisoformat(raw).date()
    except (TypeError, ValueError):
        return None
    delta = (date.today() - d).days
    return float(delta) if delta >= 0 else None   # 未来日期 = 脏数据，不可信


def schedule_note_is_fresh(sched: dict, ttl_days: int = SCHEDULE_NOTE_TTL_DAYS) -> bool:
    """近况是否还在有效期内。**超期就该整条不注入**——过期的事实等于不存在。

    为什么在代码里判、不交给模型：模型不知道阈值、也不会拿"更新于 X"去和当前时间
    做减法；更要命的是【一致性规则】会把第一次用错的借口锁死，错误只固化不自我纠正。
    实测踩坑：近况"这周在收拾搬家"（更新于 09-05）被当当前理由用了一周。
    """
    age = schedule_note_age_days(sched)
    if age is None:
        return False
    return age < ttl_days


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
