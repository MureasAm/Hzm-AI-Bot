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
import os
import re
import threading
from pathlib import Path
from datetime import datetime

from .memory import archive_legacy_file
from . import injection as inj
from .constants import PROJECT_ROOT, THINKING_DISABLED
from .config import log_cache_usage, _get_model_name, extract_chat_content
# 时长措辞（"3天"/"2小时"）与长期/短期记忆共用一份，别各写一套（memory re-export 自根模块）
from .memory import humanize_gap

SESSION_MEMORY_FILE = PROJECT_ROOT / "user_memory" / "session.json"   # 【旧】单文件：已拆，留作迁移源
SESSION_DIR = PROJECT_ROOT / "user_memory" / "session"                # 【现】会话记忆：一个会话一个文件

# 短消息阈值：≤4 字视为"口语回应"，需要上下文扩充
SHORT_QUERY_MAX_CHARS = 4
# 单用户保留的最大事件数（防无限膨胀）
MAX_EVENTS_PER_SESSION = 6
# 会话冷场阈值（秒）：**按"距上一条消息"算，3 小时**。
# ⚠️ 2026-10-10 改（推翻 2026-10-05 的"按话题年龄 topic_since、24h"）：
#    原顾虑是"last_active 每轮刷新 → 话题永不老化"，但那是把"话题跨天延续"当成常态了。
#    真实对话是**一阵一阵的**：停几个小时再回来，就是新的一场。所以改成
#    "几小时内没消息 → 这场过去了"，阈值也从 24h 收到 3h。
#    topic_since 仍保留（用于"这个话题几点开始的"），但**不再决定过期**。
SESSION_STALE_SECONDS = 3 * 3600
# 情绪（mood）与话题同寿命：一起在 3 小时后消散（2026-10-10，用户定）。
MOOD_STALE_SECONDS = SESSION_STALE_SECONDS
# 「上次聊过」提示的阈值（2026-10-16 改：**和话题消散那条对齐**）。
# ⚠️ 原来它单独一条、而且是 **12 小时**，理由是"按 3h 提示太吵"。但那留出一个
#    **3~12 小时的盲区**：话题 3h 就消散了、提示又还没到 → **这一场对谁都不存在**。
#    实测线上 245 个真实会话里 **54 个正落在盲区**；而且方向是反的——
#    8 小时前聊的什么都不说，13 小时前反而说"上次聊过"。越近越相关，行为却相反。
#
#    现在对齐：**一场满 3 小时就算过去了，同时降级成【上次聊过】**，两块无缝。
#    不会因此"太吵"：这条提示只会在**隔了这么久之后回来的第一句**上出现一次
#    （`previous_session_note` 在 `probe_session` 之前调用，probe 一跑就把
#      last_active 刷成现在，下一句 gap≈0 就不注了）。
# ⚠️ 这两个常量**必须相等**——不相等就又漏出盲区。测试里有一条专门盯着
#    （`test_no_gap_between_the_two_blocks`）。
PREV_SESSION_NOTE_SECONDS = SESSION_STALE_SECONDS
# 「上次聊过」最多带几条当时的事件（整场搬进来会挤上下文，只要够唤起记忆就行）
PREV_SESSION_EVENTS_MAX = 2

_lock = threading.Lock()

# ==================== 存储 ====================
# 一个会话一个文件（2026-10-09）：跟短期记忆/聊天落盘同一套路，理由也一样——
# 每次读写只碰一个会话（O(1)），人工查看方便，删某人的记录＝删一个文件。
# 旧单文件 session.json 首次运行自动拆分，原件改名 .migrated 留档。

_migrated = False


def _file_for(user_id: str) -> Path:
    """某个会话的文件。id 只可能是 QQ 号/群号，仍然过滤一遍防路径穿越。"""
    return SESSION_DIR / f"{re.sub(r'[^\w.-]', '_', str(user_id))}.json"


def _read_json(path: Path):
    """读一个 JSON 文件；缺失/空/坏 → None。"""
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        return json.loads(content) if content else None
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data) -> None:
    """写 JSON 文件（先写 .tmp 再 replace，避免写一半被读到半个文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _ensure_migrated() -> None:
    """把旧的单文件 session.json 拆成一人一个文件（只跑一次，原件改名存档，不删）。"""
    global _migrated
    if _migrated:
        return
    _migrated = True
    if not SESSION_MEMORY_FILE.exists():
        return
    data = _read_json(SESSION_MEMORY_FILE)
    if not isinstance(data, dict):
        return
    n = 0
    try:
        for uid, sess in data.items():
            if not isinstance(sess, dict) or not sess:
                continue
            f = _file_for(uid)
            if f.exists():      # 已经拆过就别覆盖
                continue
            _write_json(f, sess)
            n += 1
        archived = archive_legacy_file(SESSION_MEMORY_FILE)
        print(f"[会话记忆] 旧单文件已拆分：{n} 个会话 → {SESSION_DIR}"
              f"（原件存为 {archived.name}）")
    except OSError as e:
        print(f"⚠️ 会话记忆拆分失败（退回旧单文件读，不影响聊天）: {e}")


def use_storage(root) -> None:
    """把会话记忆整体指到 `root` 目录下（**测试/离线脚本专用**）。

    一次改齐三样（目录 / 旧单文件路径 / 迁移标志）——只改一个会漏掉隔离，
    或更糟：让迁移去动真实那份 session.json。
    """
    global SESSION_DIR, SESSION_MEMORY_FILE, _migrated
    root = Path(root)
    SESSION_DIR = root / "session"
    SESSION_MEMORY_FILE = root / "session.json"
    _migrated = True


def _load() -> dict:
    """**调试/兼容**用：合并所有会话（生产路径不用它，只用 `_raw_session`）。"""
    _ensure_migrated()
    out = {}
    if SESSION_DIR.exists():
        for f in SESSION_DIR.glob("*.json"):
            v = _read_json(f)
            if isinstance(v, dict) and v:
                out[f.stem] = v
    return out


def _raw_session(user_id: str) -> dict:
    """读**原始**记录——不做过期过滤。过期判定和「上次聊过」都要用它自己算。"""
    _ensure_migrated()
    sess = _read_json(_file_for(user_id))
    if not isinstance(sess, dict) and SESSION_MEMORY_FILE.exists():
        # 兜底：万一旧单文件还没拆成功，仍能从它里面读到这个会话
        legacy = _read_json(SESSION_MEMORY_FILE)
        sess = legacy.get(str(user_id)) if isinstance(legacy, dict) else None
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

    冷场判定（2026-10-10 改）：**按"距上一条消息"（last_active）**，阈值 3 小时。
    停几小时没说话 = 这一场已经过去了（见 SESSION_STALE_SECONDS 的说明）。
    旧写法按"话题年龄"（topic_since, 24h），对"聊一阵停一阵"的日常太宽松。

    ⚠️ **拿不到时间戳的一律按冷场处理**（老记录/手改记录没有时间字段）。
    旧写法是 `if last:` 才做判定——缺字段的记录**直接跳过检查、永久免疫**。2026-09-27 查过
    线上 106 条都有该字段，所以还没炸；但那是运气不是设计，随手动改数据随时会踩。
    """
    sess = _raw_session(user_id)
    if not sess:
        return {"topic": "", "events": [], "last_active": ""}
    last = str(sess.get("last_active") or "")
    if not last:
        return {"topic": "", "events": [], "last_active": ""}
    try:
        gap = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
    except (ValueError, TypeError):
        return {"topic": "", "events": [], "last_active": ""}
    if gap > SESSION_STALE_SECONDS:
        return {"topic": "", "events": [], "last_active": ""}
    return sess


def previous_session_note(user_id: str) -> str:
    """隔得够久（> `PREV_SESSION_NOTE_SECONDS`，**与话题消散同为 3h**）时，给一条带时间的
    「上次聊过」提示；否则空串。

    ⚠️ 这个阈值**必须和 `SESSION_STALE_SECONDS` 相等**（2026-10-16 改）。
    原来它单独是 12h，中间 3~12h 就成了盲区：第 8 块已经不注、第 9 块还没到，
    实测线上 54 个会话落在那儿。现在对齐——**一场满 3 小时就算过去了，同时降级成「上次聊过」**。

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
    if gap is not None and gap <= PREV_SESSION_NOTE_SECONDS:
        return ""   # 没隔够久 → 不提示（话题可能已按 3h 消散，但那不算"旧事重提"）
    when = f"{humanize_gap(gap)}前" if gap is not None else ""
    body = []
    if topic:
        body.append(f"聊到「{topic}」")
    if events:
        body.append("当时：" + "；".join(events))
    return inj.T("prev_session", when=when, body="，".join(body))


def set_mood(user_id: str, mood: str, *, now: datetime | None = None) -> None:
    """写入当下情绪（手动/测试用；生产路径由 probe_session 顺带写入）。"""
    text = (mood or "").strip()
    if not text:
        return
    now = now or datetime.now()
    with _lock:
        sess = _raw_session(user_id)
        if not isinstance(sess, dict):
            sess = {}
        sess["mood"] = text
        sess["mood_at"] = now.isoformat()
        sess.setdefault("topic", "")
        sess.setdefault("events", [])
        sess.setdefault("last_active", now.isoformat())
        _write_json(_file_for(user_id), sess)


def mood_note(user_id: str) -> str:
    """当下情绪的注入文本；超过 3 小时（或没有）返回空串。

    注入位置**紧挨用户消息**（跟【当前时间】一样放尾部）——它只影响这一轮语气，
    不该跟【会话记忆】那类中段内容混在一起。
    """
    sess = _raw_session(user_id)
    mood = str(sess.get("mood") or "").strip()
    if not mood:
        return ""
    at = str(sess.get("mood_at") or "")
    gap = ""
    if at:
        try:
            secs = (datetime.now() - datetime.fromisoformat(at)).total_seconds()
        except (ValueError, TypeError):
            secs = None
        if secs is not None and secs > MOOD_STALE_SECONDS:
            return ""
        if secs is not None:
            g = humanize_gap(secs)
            if g:
                gap = f"（{g}前说的）"
    return inj.T("mood", mood=mood, gap=gap)


# ==================== 提示词 ====================

SESSION_PROBE_PROMPT = """你是会话话题追踪器。用户在聊天中刚发了一条新消息，下面给出【上一轮已知话题】和【最近对话】。

任务：判断这条新消息在当前语境下的完整含义，以及它是否带来了话题转变。


输出 JSON：
{{
  "topic": "当前话题的一句话概括（如'用户在撒娇，灰泽满在傲娇推拉''聊灰泽满的香水'；话题没变就沿用上一轮的，变了就换成新的）",
  "topic_changed": true 或 false,
  "new_event": "这条消息值得记住的关键事件，一句话（如'用户发比爱心示好'）；纯寒暄无事件则 null",
  "mood": "用户**此刻**的情绪状态，一句话（如'因为考试失利而失落''和队友闹别扭后很委屈'）；看不出来 / 没有明显情绪则 null",
  "entities": "用户这句话提到/问到的**作品、作者、地点、动物**等专名（最多 3 个，写规范名，如'空中杀人现场''东野圭吾''袋鼠'）；没有或只是普通闲聊则 null",
  "expanded_query": "如果这条消息很短（≤4字）**或是指代性的**（必须结合前面聊的话题才能理解，如'能读给我听听吗''那个呢''然后呢'——单看这句不知道指什么），补全成既能检索到相关记忆、又能帮模型理解的一句话（如'用户让灰泽满念她写的小说乌色月'）；其他情况则 null"
}}
要求：
- topic 要能体现**对话调性**（在干嘛、什么氛围），不是只复述用户那句话。
  ✅「用户和灰泽满互相斗嘴骂人」「用户要求灰泽满说晚安，灰泽满傲娇反驳，氛围轻松打闹」
     ——写得出"**这一场在干嘛**"。
  ❌「用户惊讶发现有语言功能」「用户向灰泽满分享买了生日礼物」
     ——这是把**他那句话又说了一遍**，没有"在干嘛"。（2026-10-16 量过：全库约 7% 是这种。）
- **topic 只概括双方实际说的话，不要添加对话里没有的设定或推断**：用户没提"时差/异地/对方在哪/身份"，就别写"时差/异地"——用户只说时间是几点，就写"聊时间/几点/作息"；拿不准就平实地概括内容，宁可朴素不要加戏（曾踩坑：用户说"现在是墨尔本时间凌晨六点"，被脑补成"调侃时差"并注入带偏回复）
- topic_changed 判定标准（**新主题优先**）：只要用户这条消息是在**问/聊一个新的具体主题**（如"你最近有在用香水吗""你会不会游泳"），即使语气还延续之前的氛围，也视为转话题 → topic_changed=true，topic 换成这个新主题。只有当消息是**同一主题下的继续**（如上一轮聊香水、这轮"那你喜欢哪个牌子"）才 topic_changed=false 沿用。
- new_event 只记有意义的互动（示好/情绪/承诺/分享），寒暄问候不记
- mood 只在用户**明显**带情绪时写（失落/焦虑/兴奋/委屈…）；纯寒暄、看不出情绪就 null。
  **别把"话题"当情绪**（聊考试 ≠ 焦虑），也**别把灰泽满自己的语气**算成用户的情绪。
- entities 只收**专名**（书名/作者/地点/动物），别把普通名词（"书""作家""动物"）当实体；
  用户没提具体名字、拿不准就 null。作品名写用户说的那个，作者写全名。
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
- **扩写要保留这句话的言语行为**：提醒 / 质问 / 纠正 / 撒娇 / 试探，不要改写成中性陈述。
  例：用户说「开会呀，你忘了吗？」→「用户在提醒灰泽满：自己今天要开会（之前说过的安排），问她是不是忘了」
- 扩写**优先保留用户原词**（「跑步」就写跑步，不换成“正在跑步”之类的近义词），并带上上下文里的具体信息（谁、什么时候、什么事）；上下文确实没说的不要编。
- 若用户在提醒/追问一件之前提过的事：先看【现在】和【本场已知的事】，能对上就写出来（「用户在提醒灰泽满：<那件事>，问她是不是忘了」）；对不上就写「某件之前说过的事」，**不要猜具体内容**。
- 用户只是在提醒/重复之前说过的事 → new_event 为 null（别记成“用户分享了…”）。
  · 补全**只写「这句话在当前语境下是什么意思」**，别编对话里没出现的细节。
    ❌「用户问灰泽满关于小说乌色月**中那个内容的细节**」——"那个内容"是凭空加的；
      补成「用户追问小说乌色月的事」就够（2026-10-16 探针上出现的）。
【下面是要判的那一轮】

【现在】{now}
【上一轮已知话题】{prev_topic}（为空表示新会话）

【最近对话】
{history}

【本场已知的事】{events}

【用户刚发的消息】{user_msg}

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
        log_cache_usage(resp, "话题探测")
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

# 「同一件事换了个说法」的去重阈值（二字重合度）。
# 实测（2026-10-16）：用户 1620265686 的事件表里躺着**三条要说晚安语音**——
#   「用户让灰泽满发晚安语音」／「用户再次要求发晚安语音」／「用户再次要求灰泽满发晚安语音」
# 原来的判据是 `new_event not in events`（**字符串完全相等**），换个说法就进得去。
# 事件本来就是"一句话描述"，模型每轮措辞都会飘 —— 精确比对几乎等于没去重，
# 而这 6 个格子是全库共用的，三条重复等于一半被浪费。
_EVENT_DUP_RATIO = 0.5


def _bigrams(text: str) -> set:
    s = re.sub(r"[^一-鿿A-Za-z0-9]", "", text or "")
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _is_duplicate_event(new_event: str, events: list) -> bool:
    """新事件和已有事件说的是同一件事吗（二字重合度超阈值就算）。

    拿真实数据定过阈值：三条"要晚安语音"两两重合 0.50 / 0.80（**判重**），
    而「用户骂灰泽满傻逼」和「用户对灰泽满笑骂了一句你滚吧」只有 0.43（**不判重**）。
    """
    g_new = _bigrams(new_event)
    if not g_new:
        return False
    for e in events:
        g = _bigrams(str(e))
        if g and len(g_new & g) / max(1, min(len(g_new), len(g))) >= _EVENT_DUP_RATIO:
            return True
    return False


_WEEKDAYS = "一二三四五六日"


def _now_label(now: datetime | None = None) -> str:
    """「现在」给探测看：日期 + 星期 + 几点 + 时段（早上/晚上…）。

    为什么要给探测看时间（2026-10-20）：「你忘了吗」这种提醒句，只有配上“现在是早上 +
    他早上固定要做什么”才能被理解；探测原来完全不知道现在几点。
    """
    now = now or datetime.now()
    h = now.hour
    if 5 <= h < 11:
        period = "早上"
    elif 11 <= h < 14:
        period = "中午"
    elif 14 <= h < 18:
        period = "下午"
    elif 18 <= h < 23:
        period = "晚上"
    else:
        period = "深夜"
    return f"{now.strftime('%Y-%m-%d')} 周{_WEEKDAYS[now.weekday()]} {now.strftime('%H:%M')}（{period}）"


async def probe_session(user_id: str, user_msg: str, history_text: str, client,
                        debug: bool = False, now: datetime | None = None):
    """对话前同步探测会话：判断话题延续/转换、累计事件、扩充短 query。

    一次 LLM 调用同时完成三件事（避免对话后异步更新导致的话题滞后一轮）：
    1. 判断这条消息是否转话题 → 转则清空旧事件、起新话题
    2. 累计关键事件
    3. 若消息 ≤4 字，补成完整句（返回给调用方做检索 query）

    返回：短消息的扩充句（长消息原样返回）。
    注意：这是**对话前**调用，所以第 N 轮注入的就是第 N 轮自己的话题。

    `debug=True` 时返回 `(扩充句, 模型原始 JSON)` —— 给 `scripts/session_probe.py` 看
    `topic_changed` / `mood` 这些**没被存下来**的判断，生产路径不受影响。
    """
    msg = (user_msg or "").strip()
    if not msg:
        return (msg, {}) if debug else msg
    prev = get_session(user_id)
    prev_topic = prev.get("topic", "")
    events = "；".join(
        str(e).strip() for e in (prev.get("events") or []) if str(e).strip()
    ) or "（无）"
    prompt = SESSION_PROBE_PROMPT.format(
        prev_topic=prev_topic or "（新会话）",
        history=history_text or "（无）",
        user_msg=msg,
        events=events,
        now=_now_label(now),
    )
    content = await _llm(client, prompt, max_tokens=320, temperature=0.2)
    expanded = msg
    if not content:
        return (expanded, {}) if debug else expanded
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return (expanded, {}) if debug else expanded

    topic = str(parsed.get("topic", "")).strip()
    if topic.lower() in ("null", "none"):
        topic = ""  # 兜底：模型把"无话题"写成字符串 null 时不存
    changed = bool(parsed.get("topic_changed"))
    new_event = parsed.get("new_event")
    if new_event and isinstance(new_event, str):
        new_event = new_event.strip()
    # 当下情绪（2026-10-10 从长期记忆搬到这里）：只在这一轮确实读出情绪时覆盖。
    mood = parsed.get("mood")
    mood = mood.strip() if isinstance(mood, str) else ""
    if mood.lower() in ("null", "none"):
        mood = ""
    # 关联实体（2026-10-20）：作品/作者/地点/动物的专名，供【关联背景】沿边走。
    # 这里只做形状收口（list[str]、去空、≤3、单个 ≤20 字）；认不认识、怎么展开由 retrieval 决定。
    raw_entities = parsed.get("entities")
    entities: list = []
    if isinstance(raw_entities, list):
        for item in raw_entities:
            s = str(item or "").strip()
            if s and s.lower() not in ("null", "none") and len(s) <= 20 and s not in entities:
                entities.append(s)
    parsed["entities"] = entities[:3]

    # 转话题：清空旧事件，起新话题
    events = [] if changed else list(prev.get("events", []))
    if new_event and new_event != "null" and not _is_duplicate_event(new_event, events):
        events.append(new_event)
    events = events[-MAX_EVENTS_PER_SESSION:]

    with _lock:
        # topic_since = **这个话题是什么时候开始的**（2026-10-05 加）。
        # 为什么需要：last_active 每轮都刷新，所以它只等于"距上一条消息多久"，
        # **不表示"这个话题有多旧"**。没有它，模型就算拿到时间也无法判断
        # "这场对话是不是早就结束了"——它看不到话题的年龄。
        # 话题延续（changed=False）时**保持不变**，转话题时才重置。
        prev_since = str(prev.get("topic_since") or "")
        if not changed and prev_since:
            topic_since = prev_since
        else:
            topic_since = datetime.now().isoformat()
        now_iso = datetime.now().isoformat()
        record = {
            "topic": topic or prev_topic,
            "events": events,
            "last_active": now_iso,
            "topic_since": topic_since,
        }
        # mood 只在读出情绪时覆盖；没读出就沿用上一轮的（直到 3h 自然消散）。
        # 注意：cold session 时 prev 是空的 → 旧 mood 也随之丢弃，符合"和话题一起消散"。
        if mood:
            record["mood"] = mood
            record["mood_at"] = now_iso
        elif prev.get("mood"):
            record["mood"] = str(prev.get("mood"))
            record["mood_at"] = str(prev.get("mood_at") or "")
        _write_json(_file_for(user_id), record)

    # 检索 query 扩充：短消息（≤4字）或指代性消息（必须结合话题才能懂，如"能读给我听听吗"）
    # 时，用模型补全的完整句做检索——否则指代性消息检索不到任何记忆（曾导致"念乌色月"时模型没
    # 记忆注入、只能编造内容）。
    ex = parsed.get("expanded_query")
    if ex and isinstance(ex, str):
        ex = ex.strip()
        if 0 < len(ex) <= 80 and ex != msg:
            expanded = ex
    return (expanded, parsed) if debug else expanded


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
    """生成会话记忆的注入文本。无有效会话返回空串。

    ⚠️ 2026-10-05 改：**不再叫「当前话题」/「本场发生」**。
    旧措辞在断言"这是正在进行的对话"（模型读到"当前"就不会去判时间），
    而它可能是几小时前那场、早就自然结束了。

    现在只陈述事实：**话题是什么 + 它什么时候开始的 + 期间发生了什么**；
    "这场还继不继续"交给模型看时间自己判（用户定的原则：不设阈值）。
    """
    sess = get_session(user_id)
    topic = sess.get("topic", "")
    events = sess.get("events", [])
    if not topic and not events:
        return ""
    # 话题年龄：优先 topic_since；老记录没这个字段就退回 last_active
    # （那等于"距上次说话多久"，比话题真实年龄小，但聊胜于无）
    since = str(sess.get("topic_since") or sess.get("last_active") or "")
    age = ""
    if since:
        try:
            secs = (datetime.now() - datetime.fromisoformat(since)).total_seconds()
            g = humanize_gap(secs)
            if g:
                age = f"（这个话题是{g}前开始的）"
        except (ValueError, TypeError):
            age = ""
    parts = []
    if topic:
        parts.append(f"话题：{topic}{age}")
    if events:
        parts.append("期间：" + "；".join(events))
    return "\n".join(parts)


# ==================== 调试入口 ====================

def debug_dump() -> dict:
    """打印当前所有会话状态（调试用）。"""
    return _load()
