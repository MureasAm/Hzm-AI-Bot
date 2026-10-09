"""Memory V2 shadow store.

This module deliberately does not affect live replies.  It validates typed LLM
extractions, consolidates evidence, persists a per-user shadow profile, and can
render the bounded context that *would* be injected after the shadow rollout is
approved.

Runtime data stays under ``user_memory/`` (gitignored).  Model output is treated
as untrusted input and must pass the allowlists below before it is stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT, THINKING_DISABLED


DEFAULT_STATE_FILE = PROJECT_ROOT / "user_memory" / "memory_v2_shadow.json"
DEFAULT_AUDIT_FILE = PROJECT_ROOT / "user_memory" / "memory_v2_audit.jsonl"

SCHEMA_VERSION = 2
MAX_SOURCE_CHARS = 240
MAX_VALUE_CHARS = 160
MAX_USER_ID_CHARS = 128
MAX_SESSION_ID_CHARS = 96

# 注入分段预算（字符）。偏好最大——它是针对这个人的适配，四层里优先级最高。
MAX_PREFERENCE_ITEMS = 6
PREFERENCE_BUDGET_CHARS = 360
COMMITMENT_BUDGET_CHARS = 220
# 时效性近况（fact）的期限档位 → 天数；"none" 不过期（长期事实/明确无期限）。
_VALID_FOR_DAYS = {"3d": 3, "14d": 14, "90d": 90}
# 这些 key 的事实**不管聊没聊到都带上**——名字/地点/时区/身份是"不矛盾的底线"。
# 换掉了原来那个对不上模型产出的旧名单（preferred_name/name/city）：
# 模型实际会产 location / timezone / identity 这些，旧名单等于没生效。
_ALWAYS_ON_FACT_KEYS = {
    "preferred_name", "name", "nickname", "location", "city",
    "timezone", "identity", "occupation", "background",
}

_KINDS = {"fact", "interaction_preference", "commitment"}
_ACTIONS = {"upsert", "retract"}
_TEMPORAL_SCOPES = {"past", "current", "future", "timeless"}
# 每个维度都是"枚举值 → 行为句"，不存印象标签（黄金律第 2 条）。
# 8 个维度：低落时怎么接 / 玩笑尺度 / 拒绝方式 / 主动关心阈值（原有 4 个）
#         + 称呼偏好 / 回复长度 / 话题雷区 / 她怎么自称（2026-10-10 扩到 8 个）。
_PREFERENCE_VALUES = {
    "support_style": {"comfort", "problem_solving", "balanced"},
    "banter_tolerance": {"low", "light", "high"},
    "directness": {"gentle", "direct", "balanced"},
    "care_initiative": {"proactive", "reserved", "balanced"},
    "address_style": {"nickname", "plain", "casual"},
    "reply_length": {"short", "medium", "long"},
    "topic_preference": {"engage", "avoid"},
    "self_reference": {"name", "first_person", "nickname"},
}
_SENSITIVE_KEYS = {
    "phone", "phone_number", "email", "exact_address", "home_address",
    "id_number", "identity_number", "bank_account", "finance_account",
    "medical_diagnosis", "health_diagnosis", "password", "credential",
}
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:password|passwd|pwd|密码)\s*[:：=]?\s*\S+"
)
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")

# 关于机器人自己（不是用户）的 fact 一律拒收。这类永远不改变她的回复，却是实测最顽固的
# 误记（用户贴更新日志 → fact/bot_version=V8.0；引用她的动态 → fact/identity=灰泽满自己是…）。
# 提示词已写进反例清单，这里再加一道确定性闸门（模型输出本就当不可信输入校验）。
_BOT_SELF_KEY_PREFIXES = ("bot", "hzm")
_BOT_SELF_KEYS = {"version", "changelog", "release_notes", "bot_info"}
_BOT_SELF_VALUE_SUBJECTS = ("灰泽满", "hzm", "机器人", "assistant")

_PREFERENCE_BEHAVIORS = {
    ("support_style", "comfort"): "用户低落时先接住情绪，不要马上讲道理或给方案。",
    ("support_style", "problem_solving"): "用户求助时可以更快给出具体办法，但先简短确认其感受。",
    ("support_style", "balanced"): "用户低落时先回应感受，再视情况给一个简短办法。",
    ("banter_tolerance", "low"): "减少针对用户的互怼和挤兑，保留灰泽满自己的嘴硬语气即可。",
    ("banter_tolerance", "light"): "普通闲聊可以轻度调侃，碰到认真或低落内容时收住。",
    ("banter_tolerance", "high"): "普通闲聊可以更强地互怼；用户明显低落时暂停。",
    ("directness", "gentle"): "涉及拒绝或纠正时说得委婉一点，但不要虚假附和。",
    ("directness", "direct"): "可以直接说结论，不必层层铺垫。",
    ("directness", "balanced"): "结论说清楚，同时保留一点缓冲。",
    ("care_initiative", "proactive"): "察觉用户状态变化时可以主动关心一句，但不要连续追问。",
    ("care_initiative", "reserved"): "用户没有主动展开时不要追问隐私或情绪原因。",
    ("care_initiative", "balanced"): "可以关心一次；用户不展开就自然换回当前话题。",
    ("address_style", "nickname"): "称呼用户时用他给出的昵称或称呼，别一上来就泛泛地叫'你'。",
    ("address_style", "plain"): "别用亲昵称谓（'宝宝''亲爱的'这类），直接叫名字或干脆不称呼。",
    ("address_style", "casual"): "称呼随意点就好，用'你''喂''诶'这类，不必刻意恭维或套近乎。",
    ("reply_length", "short"): "回复尽量短，一两句就够，别铺陈解释。",
    ("reply_length", "medium"): "回复控制在一两句话到一小段，别长篇大论。",
    ("reply_length", "long"): "用户聊得起劲时可以多说几句，别急着收尾。",
    ("topic_preference", "engage"): "用户主动聊起他喜欢的话题时多顺着聊两句，别急着收尾或硬转台。",
    ("topic_preference", "avoid"): "用户明显不想聊某个话题时不要追问，被带到就自然岔开去别处。",
    ("self_reference", "name"): "自称时用'灰泽满'这个名字就行。",
    ("self_reference", "first_person"): "自称时用'我'就好，少报全名。",
    ("self_reference", "nickname"): "自称时用用户给她的昵称或简称。",
}


# ⚠️ **变量一律放最后**（2026-10-10 修）：DeepSeek 缓存按**前缀**匹配——把 {observed_at} /
#    {current_summary} / 对话放到前面，会让**它后面的整段静态指令永远 miss**。原来第一个变量
#    在第 130 字（共 2630 字）→ 只有 5% 可缓存；跑几千次重建就攒出 400 万+ 未命中 token。
#    改成"指令在前、数据在后"后有 ~95% 可缓存（v1 的提取提示词一直是这个排法）。
MEMORY_V2_EXTRACT_PROMPT = """你是灰泽满私聊机器人的记忆候选提取器。你的输出不会直接成为事实，
还会经过代码校验和影子审计。只根据下面这一次真实对话提取，不要脑补。

【唯一的准入标准】
把这条信息抽出来，**会不会改变灰泽满下一次回复**？不会的，一律不记。

只允许三种 kind（**情绪不在长期记忆里**——它住在会话记忆，跟话题一起 3 小时消散）：
1. fact：关于**用户本人**的、会影响回应的稳定事实——称呼/昵称、地域时区、身份（学生/社畜）、
   正在经历的大事（搬家/考试/生病）、明确喜好（"月饼一般，芒果味的可以"）。
   **必须是关于"用户本人"的**；讲灰泽满/机器人自己的内容一律不算 fact。
   · key 用**具体到事**的小写英文（`location`/`preferred_name`/`timezone`/`exam_status`…），
     **别用 `current_situation`/`past_experience`/`info` 这类装什么都行的通用键**——
     通用键会让不同的事互相顶掉。
   · temporal_scope：**经历**用 past（可叠加，不倒旧的）；**地域/时区/身份**这类长期成立用 timeless；
     **当前状态**（在准备考试 / 在搬家）用 current（**同一 key 的新值会顶掉旧的**）。
   · valid_for：**只有"时效性近况"**（在准备考试 / 这周军训 / 最近感冒）才填其大概期限，
     从用户话里的时间词读（"这几天"→3d、"最近/这一阵"→14d、"这学期"→90d）；
     没有明确期限、或不是近况，一律 "none"。
   explicit 必须为 true。一次性状态、推断身份不提取。
2. interaction_preference：用户想被怎样对待（针对这个人的相处方式，不是性格标签）。key/value 只能是：
   support_style=comfort|problem_solving|balanced    低落时怎么接
   banter_tolerance=low|light|high                    玩笑尺度
   directness=gentle|direct|balanced                  拒绝/纠正的方式
   care_initiative=proactive|reserved|balanced        主动关心阈值
   address_style=nickname|plain|casual                称呼偏好
   reply_length=short|medium|long                     回复长度偏好
   topic_preference=engage|avoid                      话题偏好与雷区
   self_reference=name|first_person|nickname          她该怎么自称
   用户明确说喜欢/不喜欢时 explicit=true；仅从互动表现推断时 explicit=false。
3. commitment：灰泽满本轮明确答应用户、需要以后兑现的事。承诺只从灰泽满回复中提取，
   用户自己的计划不是灰泽满的承诺；actor 必须是 assistant，explicit 必须为 true。

【反例清单——抽到这些一律不记】
- 灰泽满自己/机器人自身的事：版本号、功能、更新日志、"你是不是 AI"、她自己的身份设定。
  **不管用户是告知、询问还是吐槽，只要讲的是灰泽满/机器人自己，都不是"关于用户的事实"**。
- 单次闲聊、寒暄、玩梗（"在吗""哈哈哈""晚安"）。
- 只出现一次、不会影响下次回应的状态（"他刚打了个哈欠""他刚发了个表情"）。
- 已能从别处拿到的：时间/星期/农历、天气、直播场次状态——那是感知模块的活。
- 用户自己的计划/打算——那是用户的事，不是她的承诺。
- 敏感信息：电话、邮箱、精确地址、证件、账户、密码、医疗诊断。

隐私铁律：不要提取电话、邮箱、精确地址、证件、账户、密码、医疗诊断等敏感信息。
时间铁律：保留“以前/现在/打算/已经结束”等状态，不把过去事实写成当前事实。
temporal_scope 必须是 past/current/future/timeless：用户过去经历用 past，当前仍成立用 current，
无时间属性的稳定偏好用 timeless，承诺或尚未发生的计划用 future。
纠正铁律：用户明确否定或纠正旧信息时 action=retract；其他新增/确认使用 action=upsert。
不确定就不提取。不要为了显得有记忆而凑内容。

只输出严格 JSON：
{{
  "items": [
    {{
      "kind": "fact|interaction_preference|commitment",
      "key": "小写英文键；fact 要具体到事，别用通用键",
      "value": "简洁、保留限定词的中文内容",
      "action": "upsert|retract",
      "temporal_scope": "past|current|future|timeless",
      "valid_for": "none|3d|14d|90d（仅时效性近况填，其余 none）",
      "explicit": true,
      "confidence": 0.0,
      "actor": "assistant"
    }}
  ]
}}
没有候选时返回 {{"items":[]}}。不要 markdown，不要额外文字。

【本轮对话】
用户：{user_text}
灰泽满：{assistant_text}

【观测时间与会话】
观测时间：{observed_at}
证据会话：{session_id}

【当前已知记忆摘要（避免重复提取）】
{current_summary}
"""


def shadow_enabled() -> bool:
    """V2 提取**默认开启**（2026-10-10 转正：真实聊天已用它重建过一整库）。

    要临时关掉（回退排查）在 `.env.prod` 设 `MEMORY_V2_SHADOW=0`，不用改代码。
    """
    return os.environ.get("MEMORY_V2_SHADOW", "1") != "0"


def injection_enabled() -> bool:
    """V2 注入**默认开启**（同上转正）。临时关掉设 `MEMORY_V2_INJECT=0`。

    注入的都是长期适配（相处方式靠前 / 未完成约定中段）；情绪走会话记忆，不在这里。
    """
    return os.environ.get("MEMORY_V2_INJECT", "1") != "0"


def parse_extraction_payload(content: str) -> dict:
    """Parse model output without repairing or guessing malformed data."""
    raw = (content or "").strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    if not raw or raw == "null":
        return {"items": []}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"items": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return {"items": []}
    return {"items": payload["items"][:20]}


def build_extraction_prompt(
    *,
    user_text: str,
    assistant_text: str,
    current_summary: str,
    session_id: str,
    observed_at: datetime,
) -> str:
    """Render the strict write-path prompt with bounded untrusted text."""
    return MEMORY_V2_EXTRACT_PROMPT.format(
        observed_at=observed_at.isoformat(),
        session_id=_bounded_text(session_id, MAX_SESSION_ID_CHARS),
        current_summary=_bounded_text(current_summary, 1200) or "（无）",
        user_text=_bounded_text(user_text, 1000),
        assistant_text=_bounded_text(assistant_text, 1000),
    )


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(getattr(part, "text", ""))
            for part in content
        )
    return ""


def _extraction_summary(store: "ShadowMemoryStore", user_id: str) -> str:
    memories = store.snapshot(user_id).get("memories", [])
    visible = [
        f"{m.get('kind')}/{m.get('key')}={m.get('value')}({m.get('status')})"
        for m in memories
        if m.get("status") not in {"superseded", "expired"}
    ]
    return "；".join(visible)[:1200] or "（无）"


async def extract_and_ingest(
    *,
    client: Any,
    model: str,
    store: "ShadowMemoryStore",
    user_id: str,
    user_text: str,
    assistant_text: str,
    session_id: str,
    observed_at: datetime | None = None,
) -> list[dict]:
    """Run one strict shadow extraction; failures never escape to chat flow."""
    observed_at = observed_at or datetime.now()
    prompt = build_extraction_prompt(
        user_text=user_text,
        assistant_text=assistant_text,
        current_summary=_extraction_summary(store, user_id),
        session_id=session_id,
        observed_at=observed_at,
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            **THINKING_DISABLED,
        )
    except Exception:
        return []
    payload = parse_extraction_payload(_response_text(response))
    return store.ingest(
        user_id,
        payload["items"],
        source={
            "session_id": session_id,
            "observed_at": observed_at.isoformat(),
            "user_text": user_text,
            "assistant_text": assistant_text,
        },
        now=observed_at,
    )


def _iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _contains_sensitive_value(value: str) -> bool:
    return (
        any(pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS)
        or bool(_SECRET_ASSIGNMENT_RE.search(value))
    )


def _redact_sensitive_excerpt(value: str) -> str:
    redacted = value
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return _SECRET_ASSIGNMENT_RE.sub("[redacted]", redacted)


def _is_bot_self_fact(key: str, value: str) -> bool:
    """这条 fact 讲的是机器人/她自己，而不是用户本人？"""
    lowered = key.lower()
    if lowered in _BOT_SELF_KEYS or lowered.startswith(_BOT_SELF_KEY_PREFIXES):
        return True
    return value.strip().lower().startswith(_BOT_SELF_VALUE_SUBJECTS)


def _fact_expires_at(item: dict, now: datetime) -> str | None:
    """时效性近况（fact）的过期时刻；长期事实 / 偏好 / 承诺不过期 → None。"""
    if item.get("kind") != "fact":
        return None
    days = _VALID_FOR_DAYS.get(str(item.get("valid_for") or ""))
    if not days:
        return None
    return (now + timedelta(days=days)).isoformat()


def _ngrams(text: str) -> set[str]:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "").lower()
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i:i + 2] for i in range(len(normalized) - 1)}


def _relevance(query: str, record: dict) -> float:
    query_terms = _ngrams(query)
    memory_terms = _ngrams(f"{record.get('key', '')}{record.get('value', '')}")
    if not query_terms or not memory_terms:
        return 0.0
    return len(query_terms & memory_terms) / max(1, min(len(query_terms), len(memory_terms)))


class ShadowMemoryStore:
    """Validate, consolidate, and inspect Memory V2 shadow state.

    The interface is intentionally small: callers ingest model candidates,
    request a hypothetical context, inspect a user partition, or delete it.
    """

    def __init__(
        self,
        state_file: Path = DEFAULT_STATE_FILE,
        audit_file: Path = DEFAULT_AUDIT_FILE,
        context_budget_chars: int = 700,
    ) -> None:
        self.state_file = Path(state_file)
        self.audit_file = Path(audit_file)
        self.context_budget_chars = max(100, int(context_budget_chars))
        self._lock = threading.Lock()

    def _empty_state(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "users": {}}

    def _load(self) -> dict:
        if not self.state_file.exists():
            return self._empty_state()
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_state()
        if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
            return self._empty_state()
        return data

    def _save(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.state_file)

    def _audit(self, entry: dict) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _normalize_source(self, source: dict, role: str, now: datetime) -> dict | None:
        if not isinstance(source, dict):
            return None
        session_id = _bounded_text(source.get("session_id"), MAX_SESSION_ID_CHARS)
        if not session_id:
            return None
        observed = _iso(source.get("observed_at")) or now
        raw = source.get("assistant_text") if role == "assistant" else source.get("user_text")
        excerpt = _redact_sensitive_excerpt(_bounded_text(raw, MAX_SOURCE_CHARS))
        if not excerpt:
            return None
        return {
            "role": role,
            "excerpt": excerpt,
            "session_id": session_id,
            "observed_at": observed.isoformat(),
        }

    def _normalize_item(self, item: dict, source: dict, now: datetime) -> tuple[dict | None, str]:
        if not isinstance(item, dict):
            return None, "item_not_object"
        kind = _bounded_text(item.get("kind"), 40)
        key = _bounded_text(item.get("key"), 48)
        value = _bounded_text(item.get("value"), MAX_VALUE_CHARS)
        action = _bounded_text(item.get("action") or "upsert", 16)
        explicit = item.get("explicit")
        confidence = item.get("confidence")
        temporal_scope = _bounded_text(item.get("temporal_scope"), 16)

        if kind not in _KINDS or action not in _ACTIONS:
            return None, "unsupported_kind_or_action"
        if not _KEY_RE.fullmatch(key) or not value:
            return None, "invalid_key_or_value"
        if not isinstance(explicit, bool):
            return None, "explicit_must_be_boolean"
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return None, "confidence_must_be_number"
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            return None, "confidence_out_of_range"
        if key in _SENSITIVE_KEYS or _contains_sensitive_value(value):
            return None, "sensitive_data_blocked"
        if kind == "fact" and _is_bot_self_fact(key, value):
            return None, "bot_self_fact_blocked"

        default_scope = {
            "fact": "current",
            "interaction_preference": "timeless",
            "commitment": "future",
        }[kind]
        temporal_scope = temporal_scope or default_scope
        if temporal_scope not in _TEMPORAL_SCOPES:
            return None, "invalid_temporal_scope"
        allowed_scopes = {
            "fact": {"past", "current", "timeless"},
            "interaction_preference": {"current", "timeless"},
            "commitment": {"future", "current"},
        }[kind]
        if temporal_scope not in allowed_scopes:
            return None, "temporal_scope_mismatch"

        # 时效性近况的期限档位（只有 fact 用；其余一律 none）。
        valid_for = _bounded_text(item.get("valid_for"), 8).lower()
        if valid_for not in _VALID_FOR_DAYS:
            valid_for = "none"

        role = "assistant" if kind == "commitment" else "user"
        normalized_source = self._normalize_source(source, role, now)
        if normalized_source is None:
            return None, "invalid_source"

        if kind == "fact" and not explicit:
            return None, "inferred_fact_blocked"
        if kind == "interaction_preference":
            if key not in _PREFERENCE_VALUES or value not in _PREFERENCE_VALUES[key]:
                return None, "invalid_preference_dimension"
        if kind == "commitment":
            if item.get("actor") != "assistant" or not explicit:
                return None, "assistant_commitment_required"

        normalized = {
            "kind": kind,
            "key": key,
            "value": value,
            "action": action,
            "explicit": explicit,
            "confidence": confidence,
            "temporal_scope": temporal_scope,
            "valid_for": valid_for,
            "source": normalized_source,
        }
        return normalized, "accepted"

    @staticmethod
    def _memory_id(kind: str, key: str, value: str) -> str:
        raw = f"{kind}\0{key}\0{value}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def _evidence_id(source: dict) -> str:
        raw = "\0".join(str(source.get(key) or "") for key in (
            "role", "excerpt", "session_id", "observed_at",
        )).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    def ingest(self, user_id: str, items: list, *, source: dict, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        user_id = _bounded_text(user_id, MAX_USER_ID_CHARS)
        if not user_id or not isinstance(items, list):
            return []

        accepted: list[dict] = []
        with self._lock:
            state = self._load()
            user = state["users"].setdefault(user_id, {"memories": [], "updated_at": now.isoformat()})
            memories = user.setdefault("memories", [])

            for raw_item in items[:20]:
                item, reason = self._normalize_item(raw_item, source, now)
                if item is None:
                    self._audit({
                        "t": now.isoformat(), "user_id": user_id,
                        "decision": "rejected", "reason": reason,
                    })
                    continue

                if item["action"] == "retract":
                    changed = False
                    for memory in memories:
                        if (memory.get("kind") == item["kind"] and memory.get("key") == item["key"]
                                and memory.get("status") not in {"superseded", "expired"}
                                and (not item["value"] or memory.get("value") == item["value"])):
                            memory["status"] = "superseded"
                            memory["invalidated_at"] = now.isoformat()
                            changed = True
                    self._audit({
                        "t": now.isoformat(), "user_id": user_id,
                        "decision": "retracted" if changed else "ignored",
                        "kind": item["kind"], "key": item["key"],
                    })
                    continue

                memory_id = self._memory_id(item["kind"], item["key"], item["value"])
                evidence_id = self._evidence_id(item["source"])
                existing = next((m for m in memories if m.get("id") == memory_id), None)
                if existing is None:
                    if item["explicit"] and item["kind"] in {"fact", "interaction_preference"}:
                        status = "confirmed"
                    elif item["kind"] == "commitment":
                        status = "active"
                    else:
                        status = "candidate"
                    existing = {
                        "id": memory_id,
                        "kind": item["kind"],
                        "key": item["key"],
                        "value": item["value"],
                        "status": status,
                        "explicit": item["explicit"],
                        "confidence": item["confidence"],
                        "temporal_scope": item["temporal_scope"],
                        "valid_for": item["valid_for"],
                        "evidence_count": 1,
                        "evidence_ids": [evidence_id],
                        "session_ids": [item["source"]["session_id"]],
                        "session_count": 1,
                        "created_at": now.isoformat(),
                        "last_observed_at": now.isoformat(),
                        "source": item["source"],
                    }
                    expires_at = _fact_expires_at(item, now)
                    if expires_at:
                        existing["expires_at"] = expires_at
                    memories.append(existing)
                else:
                    evidence_ids = list(existing.get("evidence_ids") or [])
                    if not evidence_ids and isinstance(existing.get("source"), dict):
                        evidence_ids.append(self._evidence_id(existing["source"]))
                    if evidence_id in evidence_ids:
                        self._audit({
                            "t": now.isoformat(), "user_id": user_id,
                            "decision": "duplicate_evidence",
                            "memory_id": existing["id"],
                        })
                        continue
                    evidence_ids.append(evidence_id)
                    existing["evidence_ids"] = evidence_ids[-50:]
                    existing["evidence_count"] = int(existing.get("evidence_count", 1)) + 1
                    session_ids = list(existing.get("session_ids") or [])
                    if item["source"]["session_id"] not in session_ids:
                        session_ids.append(item["source"]["session_id"])
                    existing["session_ids"] = session_ids[-20:]
                    existing["session_count"] = len(existing["session_ids"])
                    existing["last_observed_at"] = now.isoformat()
                    existing["source"] = item["source"]
                    existing["confidence"] = max(float(existing.get("confidence", 0)), item["confidence"])
                    existing["temporal_scope"] = item["temporal_scope"]
                    existing["valid_for"] = item["valid_for"]
                    expires_at = _fact_expires_at(item, now)
                    if expires_at:
                        existing["expires_at"] = expires_at   # 近况被再次提到 → 续期
                    if item["explicit"]:
                        existing["explicit"] = True
                        existing["status"] = "active" if item["kind"] == "commitment" else "confirmed"
                    elif (item["kind"] == "interaction_preference"
                          and existing["evidence_count"] >= 3
                          and existing["session_count"] >= 2):
                        existing["status"] = "confirmed"

                # 只有"单值槽"才顶替：偏好（每个维度只应有一个值），以及 temporal_scope=current
                # 的 fact（地点/身份/近况这类"当前只有一个值"的）。**past / timeless 的 fact 累积，
                # 不倒旧的**——否则"小时候被霸凌"会被后来的"高考车祸"顶掉（实测真丢过 3 条）。
                if item["explicit"] and (
                    item["kind"] == "interaction_preference"
                    or (item["kind"] == "fact" and item["temporal_scope"] == "current")
                ):
                    for other in memories:
                        if (other is not existing and other.get("kind") == item["kind"]
                                and other.get("key") == item["key"]
                                and other.get("status") not in {"superseded", "expired"}):
                            other["status"] = "superseded"
                            other["invalidated_at"] = now.isoformat()

                accepted.append(json.loads(json.dumps(existing, ensure_ascii=False)))
                self._audit({
                    "t": now.isoformat(), "user_id": user_id, "decision": "accepted",
                    "memory_id": existing["id"], "kind": existing["kind"],
                    "key": existing["key"], "status": existing["status"],
                })

            user["updated_at"] = now.isoformat()
            self._save(state)
        return accepted

    def snapshot(self, user_id: str) -> dict:
        with self._lock:
            user = self._load().get("users", {}).get(str(user_id), {})
        if not isinstance(user, dict):
            return {"memories": []}
        result = json.loads(json.dumps(user, ensure_ascii=False))
        result.setdefault("memories", [])
        return result

    def _active_memories(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """存活的条目：confirmed/active，且没过 TTL（时效性近况到期就不算）。

        session_id 参数保留仅为兼容旧调用；情绪已搬去会话记忆，这里不再按会话过滤。
        """
        now = now or datetime.now()
        memories = self.snapshot(user_id).get("memories", [])
        active = []
        for memory in memories:
            if memory.get("status") not in {"confirmed", "active"}:
                continue
            expires = _iso(memory.get("expires_at"))
            if expires is not None and expires <= now:
                continue
            active.append(memory)
        return active

    def _select_facts(self, active: list[dict], query: str) -> list[dict]:
        facts = [m for m in active if m.get("kind") == "fact"]
        facts.sort(key=lambda m: (m.get("key") not in _ALWAYS_ON_FACT_KEYS, -_relevance(query, m)))
        return [
            m for m in facts
            if (
                (m.get("key") in _ALWAYS_ON_FACT_KEYS and m.get("temporal_scope") != "past")
                or _relevance(query, m) > 0
                or (m.get("temporal_scope") == "past" and any(k in query for k in ("以前", "过去", "曾经")))
            )
        ][:4]

    def _preference_lines(self, active: list[dict], query: str) -> list[str]:
        """靠前段：相处方式（行为句）+ 话题相关事实。"""
        preferences = [m for m in active if m.get("kind") == "interaction_preference"]
        lines: list[str] = []
        if preferences:
            lines.append("【和这个绿冻的相处方式】只调整相处方式，不改变灰泽满的核心性格、事实和边界。")
            for memory in preferences[:MAX_PREFERENCE_ITEMS]:
                behavior = _PREFERENCE_BEHAVIORS.get((memory.get("key"), memory.get("value")))
                if behavior:
                    lines.append(behavior)
        facts = self._select_facts(active, query)
        if facts:
            rendered = "；".join(
                f"{'过去事实/' if m.get('temporal_scope') == 'past' else ''}{m.get('key')}：{m.get('value')}"
                for m in facts
            )
            lines.append(f"【相关用户事实】{rendered}")
        return lines

    def _commitment_lines(self, active: list[dict], query: str) -> list[str]:
        """中段：未完成约定——别忘、要兑现。"""
        commitments = [m for m in active if m.get("kind") == "commitment" and _relevance(query, m) > 0]
        if not commitments:
            return []
        rendered = "；".join(str(m.get("value")) for m in commitments[:2])
        return [f"【相关未完成约定】{rendered}"]

    def _fit(self, lines: list[str], budget: int) -> str:
        if not lines:
            return ""
        output: list[str] = []
        used = 0
        for line in lines:
            separator = 1 if output else 0
            remaining = budget - used - separator
            if remaining <= 0:
                break
            piece = line[:remaining]
            if piece:
                output.append(piece)
                used += len(piece) + separator
            if len(piece) < len(line):
                break
        return "\n".join(output)

    def build_preference_context(
        self,
        user_id: str,
        query: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
    ) -> str:
        """靠前注入：相处方式行为句 + 话题相关事实（system，个体适配）。"""
        active = self._active_memories(user_id, now=now, session_id=session_id)
        return self._fit(self._preference_lines(active, query or ""), PREFERENCE_BUDGET_CHARS)

    def build_commitment_context(
        self,
        user_id: str,
        query: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
    ) -> str:
        """中段注入：未完成约定（system，中段——别忘、要兑现）。"""
        active = self._active_memories(user_id, now=now, session_id=session_id)
        return self._fit(self._commitment_lines(active, query or ""), COMMITMENT_BUDGET_CHARS)

    def build_context(
        self,
        user_id: str,
        query: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
    ) -> str:
        """聚合视图（两段拼接、共享同一预算）——审计/测试用，注入走各自的方法。

        情绪不在这里：它住在会话记忆（`session_memory.mood_note`），跟话题一起 3h 消散。
        """
        active = self._active_memories(user_id, now=now, session_id=session_id)
        question = query or ""
        lines = (
            self._preference_lines(active, question)
            + self._commitment_lines(active, question)
        )
        return self._fit(lines, self.context_budget_chars)

    def delete_user(self, user_id: str, *, now: datetime | None = None) -> None:
        now = now or datetime.now()
        user_id = _bounded_text(user_id, MAX_USER_ID_CHARS)
        if not user_id:
            return
        with self._lock:
            state = self._load()
            existed = state.get("users", {}).pop(user_id, None) is not None
            self._save(state)
            self._audit({
                "t": now.isoformat(), "user_id": user_id,
                "decision": "deleted" if existed else "delete_noop",
            })


__all__ = [
    "ShadowMemoryStore",
    "DEFAULT_STATE_FILE",
    "DEFAULT_AUDIT_FILE",
    "MEMORY_V2_EXTRACT_PROMPT",
    "shadow_enabled",
    "injection_enabled",
    "parse_extraction_payload",
    "build_extraction_prompt",
    "extract_and_ingest",
]
