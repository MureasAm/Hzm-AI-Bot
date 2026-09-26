"""会话级记忆（episodic memory）：记录"当前聊什么话题 + 本场关键事件"。

短期记忆只有最近 5 轮原文（碎片），长期记忆只有用户画像（事实），
都没有"这一场的调性/话题线"。会话级记忆补上这一层：

- 每轮对话后，用 LLM 判断这轮聊了什么话题、是否转话题、有无关键事件
- 同一话题持续累积事件；检测到转话题时更替当前话题
- 下一轮注入【当前会话】给模型，让它接得住会话调性
- **隔太久（>SESSION_STALE_SECONDS）的上一场**：不再当【当前会话】，
  但也不丢——降级成 `previous_session_note()` 的「上次聊过（3天前）…」，
  并明说"这不是现在的话题"。时间戳在这里真正参与提示，而不是当一道静默的闸。
- 短 query（≤4字）先在会话语境里扩充成完整句，再做 embedding+检索，
  避免短向量"糊"导致误命中不相关样本

数据：user_memory/session.json，按 user_id 存 {"topic","events","last_active"}
"""
import json
import threading
from pathlib import Path
from datetime import datetime

from .constants import PROJECT_ROOT, THINKING_DISABLED
from .config import _get_model_name, extract_chat_content
# 时长措辞（"3天"/"2小时"）与长期/短期记忆共用一份，别各写一套（memory re-export 自根模块）
from .memory import humanize_gap

SESSION_MEMORY_FILE = PROJECT_ROOT / "user_memory" / "session.json"

# 短消息阈值：≤4 字视为"口语回应"，需要上下文扩充
SHORT_QUERY_MAX_CHARS = 4
# 单用户保留的最大事件数（防无限膨胀）
MAX_EVENTS_PER_SESSION = 6
# 话题无活动多久视为冷场（秒），跨天对话重新起话题
SESSION_STALE_SECONDS = 12 * 3600
# 「上次聊过」最多带几条当时的事件（整场搬进来会挤上下文，只要够唤起记忆就行）
PREV_SESSION_EVENTS_MAX = 2

_lock = threading.Lock()

# ==================== 存储 ====================

def _load() -> dict:
    if not SESSION_MEMORY_FILE.exists():
        return {}
    try:
        content = SESSION_MEMORY_FILE.read_text(encoding="utf-8").strip()
        return json.loads(content) if content else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    SESSION_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _raw_session(user_id: str) -> dict:
    """读**原始**记录——不做过期过滤。过期判定和「上次聊过」都要用它自己算。"""
    data = _load()
    sess = data.get(user_id) if isinstance(data, dict) else None
    return sess if isinstance(sess, dict) else {}


def session_gap_seconds(user_id: str) -> float | None:
    """距这个会话上次活动过了多少秒；没有记录 / 时间戳缺失或坏掉 → None。"""
    last = str(_raw_session(user_id).get("last_active") or "")
    if not last:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(last)).total_seconds()
    except (ValueError, TypeError):
        return None


def get_session(user_id: str) -> dict:
    """读取某用户的会话状态。没有、或已冷场时返回空会话。

    冷场判定：太久没聊（SESSION_STALE_SECONDS，默认 12 小时）→ 旧话题不适用，
    返回空会话重新起。

    ⚠️ **拿不到时间戳的一律按冷场处理**（老记录/手改记录没有 last_active）。
    旧写法是 `if last:` 才做判定——缺字段的记录**直接跳过检查、永久免疫**，
    永远被当【当前会话】注入。2026-09-27 查过线上 106 条都有该字段，所以还没炸；
    但那是运气不是设计，随手动改数据随时会踩。
    """
    sess = _raw_session(user_id)
    if not sess:
        return {"topic": "", "events": [], "last_active": ""}
    gap = session_gap_seconds(user_id)
    if gap is None or gap > SESSION_STALE_SECONDS:
        return {"topic": "", "events": [], "last_active": ""}
    return sess


def previous_session_note(user_id: str) -> str:
    """上一场会话**已经过期**时，给一条带时间的「上次聊过」提示；否则空串。

    ⚠️ **必须在 probe_session 之前调用**：probe 会把这条记录的 last_active
    改写成"现在"，之后再问就永远得到"没过期"。

    为什么需要它：过期会话原来是**静默丢掉**的（get_session 直接返回空）——
    时间戳只决定了"丢不丢"，没决定"怎么定性"，于是她连"上次你不是说要加粉丝群吗"
    都说不出来。这里补上后半句：**不丢，但把时间说出来**，让她知道那是隔了多久的旧事。
    """
    sess = _raw_session(user_id)
    topic = str(sess.get("topic") or "").strip()
    events = [e.strip() for e in (sess.get("events") or [])
              if isinstance(e, str) and e.strip()][-PREV_SESSION_EVENTS_MAX:]
    if not topic and not events:
        return ""
    gap = session_gap_seconds(user_id)
    if gap is not None and gap <= SESSION_STALE_SECONDS:
        return ""   # 还在有效期内 → 走正常的【当前会话】注入，别重复说一遍
    when = f"{humanize_gap(gap)}前" if gap is not None else ""
    body = []
    if topic:
        body.append(f"聊到「{topic}」")
    if events:
        body.append("当时：" + "；".join(events))
    return (
        f"【上次聊过】{when}你们" + "，".join(body) + "。"
        f"（隔了这么久，**这不是现在正在聊的话题**：想提可以自然提一句「上次…」，"
        f"但别当成刚发生的事往下接，也别拿它当当前语境的上下文。）"
    )


# ==================== 提示词 ====================

SESSION_PROBE_PROMPT = """你是会话话题追踪器。用户在聊天中刚发了一条新消息，下面给出【上一轮已知话题】和【最近对话】。

任务：判断这条新消息在当前语境下的完整含义，以及它是否带来了话题转变。

【上一轮已知话题】{prev_topic}（为空表示新会话）

【最近对话】
{history}

【用户刚发的消息】{user_msg}

输出 JSON：
{{
  "topic": "当前话题的一句话概括（如'用户在撒娇，灰泽满在傲娇推拉''聊灰泽满的香水'；话题没变就沿用上一轮的，变了就换成新的）",
  "topic_changed": true 或 false,
  "new_event": "这条消息值得记住的关键事件，一句话（如'用户发比爱心示好'）；纯寒暄无事件则 null",
  "expanded_query": "如果这条消息很短（≤4字）**或是指代性的**（必须结合前面聊的话题才能理解，如'能读给我听听吗''那个呢''然后呢'——单看这句不知道指什么），补全成既能检索到相关记忆、又能帮模型理解的一句话（如'用户让灰泽满念她写的小说乌色月'）；其他情况则 null"
}}
要求：
- topic 要能体现**对话调性**（在干嘛、什么氛围），不是只复述内容
- **topic 只概括双方实际说的话，不要添加对话里没有的设定或推断**：用户没提"时差/异地/对方在哪/身份"，就别写"时差/异地"——用户只说时间是几点，就写"聊时间/几点/作息"；拿不准就平实地概括内容，宁可朴素不要加戏（曾踩坑：用户说"现在是墨尔本时间凌晨六点"，被脑补成"调侃时差"并注入带偏回复）
- topic_changed 判定标准（**新主题优先**）：只要用户这条消息是在**问/聊一个新的具体主题**（如"你最近有在用香水吗""你会不会游泳"），即使语气还延续之前的氛围，也视为转话题 → topic_changed=true，topic 换成这个新主题。只有当消息是**同一主题下的继续**（如上一轮聊香水、这轮"那你喜欢哪个牌子"）才 topic_changed=false 沿用。
- new_event 只记有意义的互动（示好/情绪/承诺/分享），寒暄问候不记
- **expanded_query 表情识别铁律**：如果消息是**纯表情/纯符号**（emoji 或[表情：xx]），必须先按**表情的标准含义**识别，不要从对话历史臆测：
  · 😅 = 无语/无奈/尴尬（不是傲娇调侃）
  · 😭 = 委屈/难过/哭
  · 🥲/😢 = 强颜欢笑/难受
  · 😂 = 笑/好笑
  · 🙏 = 感谢/拜托
  · ❤️/😍/🥰 = 爱意/喜欢
  · 😳/😳 = 害羞/尴尬
  · 🤔 = 疑惑
  补全句应表达"用户发了【表情含义】"这个意思（如"用户发了个无语的表情"），用于检索记忆理解用户情绪，不要加引号。
- 非表情的短消息（如"咋这样""真的吗"）**或指代性消息**（如"能读给我听听吗""那个呢"——单看不知道指什么）才结合语境补全成完整句；普通长消息（自带话题）返回 null。
只输出 JSON，不要多余内容。"""


# ==================== LLM 调用 ====================

async def _llm(client, prompt: str, max_tokens: int = 200, temperature: float = 0.2) -> str:
    """调 DeepSeek。失败返回空串（调用方降级）。"""
    try:
        resp = await client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **THINKING_DISABLED,
        )
        content = extract_chat_content(resp)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content
    except Exception as e:
        print(f"⚠️ 会话记忆 LLM 调用失败: {e}")
        return ""


# ==================== 更新会话（每轮对话后调用） ====================

async def probe_session(user_id: str, user_msg: str, history_text: str, client) -> str:
    """对话前同步探测会话：判断话题延续/转换、累计事件、扩充短 query。

    一次 LLM 调用同时完成三件事（避免对话后异步更新导致的话题滞后一轮）：
    1. 判断这条消息是否转话题 → 转则清空旧事件、起新话题
    2. 累计关键事件
    3. 若消息 ≤4 字，补成完整句（返回给调用方做检索 query）

    返回：短消息的扩充句（长消息原样返回）。
    注意：这是**对话前**调用，所以第 N 轮注入的就是第 N 轮自己的话题。
    """
    msg = (user_msg or "").strip()
    if not msg:
        return msg
    prev = get_session(user_id)
    prev_topic = prev.get("topic", "")
    prompt = SESSION_PROBE_PROMPT.format(
        prev_topic=prev_topic or "（新会话）",
        history=history_text or "（无）",
        user_msg=msg,
    )
    content = await _llm(client, prompt, max_tokens=250, temperature=0.2)
    expanded = msg
    if not content:
        return expanded
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return expanded

    topic = str(parsed.get("topic", "")).strip()
    if topic.lower() in ("null", "none"):
        topic = ""  # 兜底：模型把"无话题"写成字符串 null 时不存
    changed = bool(parsed.get("topic_changed"))
    new_event = parsed.get("new_event")
    if new_event and isinstance(new_event, str):
        new_event = new_event.strip()

    # 转话题：清空旧事件，起新话题
    events = [] if changed else list(prev.get("events", []))
    if new_event and new_event != "null":
        if new_event not in events:
            events.append(new_event)
    events = events[-MAX_EVENTS_PER_SESSION:]

    with _lock:
        data = _load()
        data[user_id] = {
            "topic": topic or prev_topic,
            "events": events,
            "last_active": datetime.now().isoformat(),
        }
        _save(data)

    # 检索 query 扩充：短消息（≤4字）或指代性消息（必须结合话题才能懂，如"能读给我听听吗"）
    # 时，用模型补全的完整句做检索——否则指代性消息检索不到任何记忆（曾导致"念乌色月"时模型没
    # 记忆注入、只能编造内容）。
    ex = parsed.get("expanded_query")
    if ex and isinstance(ex, str):
        ex = ex.strip()
        if 0 < len(ex) <= 80 and ex != msg:
            expanded = ex
    return expanded


# ==================== 表情消息判定 ====================

# 常见 emoji 的 Unicode 范围（判断消息是否纯表情）
_EMOJI_RANGES = [
    (0x1F600, 0x1F64F),  # 表情符号
    (0x1F300, 0x1F5FF),  # 符号/图形
    (0x1F680, 0x1F6FF),  # 交通/地图
    (0x1F900, 0x1F9FF),  # 补充表情
    (0x2600, 0x27BF),    # 杂项符号
    (0xFE00, 0xFE0F),    # 变体选择符
    (0x200D, 0x200D),    # 零宽连接符（复合 emoji）
]


def is_emoji_msg(msg: str) -> bool:
    """判断消息是否纯表情（emoji 或 QQ 表情码 [表情：xx]）。

    纯表情消息不表达"要检索什么话题"，只表达情绪，应跳过语义检索，
    只把表情含义作为语气提示注入给模型。
    """
    text = (msg or "").strip()
    if not text:
        return False
    # QQ 表情码（[表情：xxx]）
    if text.startswith("[表情：") and text.endswith("]"):
        return True
    # 纯 emoji / 符号
    for ch in text:
        cp = ord(ch)
        if not any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES) and not ch.isspace():
            return False
    return True


# ==================== 注入上下文 ====================

def build_session_context(user_id: str) -> str:
    """生成【当前会话】注入文本。无有效会话返回空串。"""
    sess = get_session(user_id)
    topic = sess.get("topic", "")
    events = sess.get("events", [])
    if not topic and not events:
        return ""
    parts = []
    if topic:
        parts.append(f"当前话题：{topic}")
    if events:
        parts.append("本场发生：" + "；".join(events))
    return "\n".join(parts)


# ==================== 调试入口 ====================

def debug_dump() -> dict:
    """打印当前所有会话状态（调试用）。"""
    return _load()
