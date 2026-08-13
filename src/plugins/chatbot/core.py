"""消息处理主循环。

组装系统提示词（基础人设 + 人格规则 + 行为指令 + RAG + 长期/短期记忆），
调用 DeepSeek 生成回复，并异步更新长期记忆。
"""
import json
import random
import asyncio
import re

from nonebot import get_driver
from openai import AsyncOpenAI

from .constants import (
    SYSTEM_PROMPT_FILE, DEEPSEEK_BASE_URL, ZHIPU_BASE_URL, TERMS_FILE,
    DEFAULT_MODEL, THINKING_DISABLED,
    CHAT_TEMPERATURE, CHAT_FREQUENCY_PENALTY, CHAT_MAX_TOKENS,
    MEMORY_EXTRACT_TEMPERATURE, MEMORY_EXTRACT_MAX_TOKENS,
    VOICE_SAMPLE_REPLY_TRIM_CHARS,
)
from .persona import load_persona_rules, build_global_persona_context
from .memory import (
    get_user_history, append_user_history,
    get_user_memory, update_user_memory, build_memory_context,
    _format_profile_summary, MEMORY_EXTRACT_PROMPT,
)
from .rag import embed_query
from .retrieval import (
    retrieve_corpus, retrieve_voice_samples, retrieve_phrases,
    retrieve_preferences, retrieve_core_stories, fuse_and_truncate, select_behavior_item,
)
from .constants import (
    PHRASE_PHASES_MAX, SPLIT_MIN_LEN, SPLIT_MAX_PARTS, SPLIT_MERGE_MIN_CHARS,
    SPLIT_DELAY_BASE_MS, SPLIT_DELAY_PER_CHAR_MS,
    SPLIT_DELAY_MIN_MS, SPLIT_DELAY_MAX_MS, SPLIT_DELAY_JITTER,
)
from . import context_probe
from .session_memory import (
    get_session, probe_session, build_session_context, is_emoji_msg,
)

# ==================== 💬 经典梗硬匹配库 ====================
LEGENDARY_REPLIES = {
    "爱不爱绿冻": [
        "早就说过很爱了...你要是感觉不到，灰泽满也不知道该怎么证明了",
        "爱是也可以的，不爱也可以的~（但其实是爱的，别不信）",
    ],
    "在和谁说话": [
        "在和..在和你说话哦~",
        "在和弹幕说话~"
    ],
    # "爱不爱"的跟进句：绿冻说感觉不到爱（带 LLM 确认，结合上下文判断是不是指感情）
    "没感觉": [
        "那灰泽满要怎么表现你才感觉得到？抱一下？",
        "感觉不到吗……那灰泽满多说几次，真的很爱",
    ],
    "感觉不": [
        "感觉不出来？那灰泽满只能天天说了，说到你信为止",
    ],
    # "绿冻永远是第二选择"（真实直播梗）：语义检索够不到口语化关系提问（实测~0.55），
    # 用关键词子串精确触发。回复从她的原话出发。
    "排第几": [
        "绿冻永远是灰泽满的第二选择，这还用排吗",
        "别排了，绿冻永远是灰泽满的第二选择",
    ],
    "你的粉丝": [
        "不只是粉丝……绿冻永远是灰泽满的第二选择",
        "谁说是普通粉丝了，绿冻永远是灰泽满的第二选择",
    ],
    "在乎我": [
        "在乎的，绿冻永远是灰泽满的第二选择",
        "绿冻永远是灰泽满的第二选择，怎么可能不在乎",
    ],
    "我重要": [
        "重要啊，绿冻永远是灰泽满的第二选择",
        "怎么会不重要，绿冻永远是灰泽满的第二选择",
    ],
    "绿冻算什么": [
        "绿冻算什么……绿冻是灰泽满的第二选择啊",
    ],
    # 问"能不能叫你妈妈"（征求/求宠）：才16岁怎么做父母（先于"妈妈"命中，dict 顺序靠前）
    "叫你妈妈": [
        "灰泽满才16岁，怎么做父母啊",
        "16岁当妈？灰泽满自己还是小孩呢",
        "别别别，灰泽满担不起这个称呼",
    ],
    "叫你一声": [
        "灰泽满才16岁，怎么做父母啊",
        "16岁当妈？灰泽满自己还是小孩呢",
    ],
    "叫我妈妈": [
        "灰泽满才16岁，怎么做父母啊",
        "16岁当妈？灰泽满自己还是小孩呢",
    ],
    # 直接喊"妈妈"（玩梗撒娇，不征求意见）：单独回个😅
    "妈妈": [
        "😅",
        "……😅",
    ],
}

# ==================== 梗库双路由：LLM 语境确认 ====================
# 仅宽泛关键词配确认（防误触发）；'排第几''绿冻算什么'等特定词不配。
# 每个 trigger → 一个确认 prompt 模板；含 {context} 的会附带最近对话（判断需要上下文的语境）。
_CONFIRM_SECOND = (
    "判断这条消息是否属于'绿冻向灰泽满表达自我怀疑：自己只是普通粉丝、不被在乎、"
    "在灰泽满心里没有位置'的语境，需要灰泽满用'绿冻永远是第二选择'来安抚。\n\n"
    "属于（回复'是'）的例子：\n"
    "- 我在你心里排第几\n- 我只是你的粉丝吧\n- 你是不是根本不在乎我\n- 你觉得我重要吗\n"
    "- 我只是个路人而已吧\n- 我在你心里有位置吗\n\n"
    "不属于（回复'否'）的例子：\n"
    "- 你的粉丝好热情啊（是在夸粉丝，不是绿冻自我怀疑）\n"
    "- 这个任务我重要吗（指任务，不是绿冻本人）\n"
    "- 你还在乎我们宿舍吗（指宿舍/他人，不是绿冻本人）\n\n"
    "结合最近对话判断语境——如果最近在聊感情/关系/被冷落，消息里的'重要吗''在乎吗'就是指灰泽满对TA的感情。\n\n"
    "最近对话：\n{context}\n\n消息：{msg}\n只回复：是 或 否"
)
_CONFIRM_LOVE = (
    "判断这条消息是否在'绿冻质疑灰泽满的感情（爱不爱、在不在乎、能不能感觉到爱）'的语境。\n"
    "结合最近对话判断——如果最近在聊感情/爱不爱，消息里的'没感觉''感觉不到'就是指感情。\n\n"
    "属于（回复'是'）的例子：\n"
    "- 可是我没感觉出来（前面在聊爱不爱）\n- 你说爱我但我感觉不到\n\n"
    "不属于（回复'否'）的例子：\n"
    "- 这首歌我没感觉（指歌）\n- 这个菜没味道（指菜）\n\n"
    "最近对话：\n{context}\n\n消息：{msg}\n只回复：是 或 否"
)
_CONFIRM_MOM = (
    "判断这条消息是否在'绿冻玩梗喊灰泽满妈妈'——把灰泽满当妈/叫妈妈求宠/认妈'的语境，需要灰泽满装傻不接。\n\n"
    "属于（回复'是'）的例子：\n"
    "- 妈妈！\n- 喊你一声妈妈行不行\n- 我能叫你妈妈吗\n- 妈！我要抱抱\n\n"
    "不属于（回复'否'）的例子：\n"
    "- 我妈让我早点睡（指用户自己的妈妈）\n- 我妈妈叫我睡觉了（指用户自己的妈妈）\n"
    "- 你妈妈也是这么说你的吗（指灰泽满的妈妈/满妈）\n"
    "- 帮我妈个忙（'妈'是动词，非称呼灰泽满）\n\n"
    "最近对话：\n{context}\n\n消息：{msg}\n只回复：是 或 否"
)

LEGENDARY_CONFIRMS = {
    "你的粉丝": _CONFIRM_SECOND,
    "在乎我": _CONFIRM_SECOND,
    "我重要": _CONFIRM_SECOND,
    "没感觉": _CONFIRM_LOVE,
    "感觉不": _CONFIRM_LOVE,
    "妈妈": _CONFIRM_MOM,
}


async def _legendary_confirmed(user_msg: str, prompt_template: str, history: str = "") -> bool:
    """LLM 判断关键词命中的消息是否真是目标梗的语境（双路由第二层，防误触发）。

    关键词命中是低频事件，为它加一次便宜 LLM 判断成本可控；确认失败默认放行（不阻塞）。
    history 用于需要上下文的确认（如'没感觉出来'是否指感情）。
    """
    try:
        content = prompt_template
        if "{context}" in content:
            content = content.replace("{context}", history or "（无）").replace("{msg}", user_msg)
        else:
            content = content.replace("{msg}", user_msg)
        deepseek_client, _ = _get_clients()
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=10,
            **THINKING_DISABLED,
        )
        return "是" in (resp.choices[0].message.content or "")
    except Exception as e:
        print(f"⚠️ 梗确认失败（默认放行）: {e}")
        return True


# ==================== 🎭 行为意图分类（L3：LLM 判意图，不再用 embedding 猜） ====================
# 背景：embedding 聚的是"句式"不是"意图"——'灰泽满你唱歌好听'（夸）和
# '灰泽满你怎么又迟到了'（质问）句式相同，在向量空间挤成一团，余弦匹配会把
# 夸奖误判成质疑（实测 0.70+）。治本：行为归属交给 LLM 理解（复用 LEGENDARY 双路由的
# LLM 判定模式），判别性词汇（敷衍/骗/鸽/迟到）仍作关键词兜底。
# 行为判定用原始用户消息 + 最近对话，一次调用（temperature 0），失败降级为不触发。

BEHAVIOR_CLASSIFY_PROMPT = """你是灰泽满的行为意图分类器。判断用户刚发的这条消息是否明确落入某个"行为触发场景"。只有明确匹配才选，拿不准一律 null（宁可不触发，不误触发）。

可选行为（name：触发情境）：
{behavior_defs}

判定要点：
- 只看用户这条消息本身的内容和语气，结合最近对话判断语境。
- "被夸"：消息确实在夸灰泽满（声音/外貌/才能/表现/生日祝福/唱歌好听等）。
- "被质疑/失约被催"：用户在质问、戳穿或催问灰泽满（骗人/敷衍/迟到/没播/鸽）。
- "被越界"：玩笑/幻想触及个人边界（黄段子/低俗/过度幻想等）。
- "冷场"：提及或营造社交尴尬/冷场，要求灰泽满救场。
- "立Flag/感性流露/主动抛梗"：消息必须明显对应那个情境（立Flag=灰泽满刚立承诺被打脸；感性流露=真诚感谢/脆弱；主动抛梗=明确要求灰泽满抛话题/来段子）。
- 普通闲聊、提问、寒暄、表情、玩梗 → null。
- 拿不准 → null。

最近对话：
{history}

用户消息：{user_msg}

只输出 JSON：{{"behavior": "<可选行为name>" 或 null}}"""


async def classify_behavior(deepseek_client, user_msg: str, history_text: str, behaviors: list) -> str:
    """LLM 判定用户消息落入哪个行为场景；拿不准或失败返回空串（不触发任何行为）。

    返回的行为名必须是 behaviors 里的真实 name（防模型编造）。
    """
    if not behaviors or not user_msg:
        return ""
    # 行为定义带真实粉丝话样例：领域黑话（如"黑黑的/造黄桃"是越界梗）光靠 trigger 描述 LLM 认不出，
    # 给真实样例当参照（素材驱动），分类更准（实测 b9 越界从漏判变命中）。
    defs = []
    for b in behaviors:
        if not b.get("name"):
            continue
        line = f"- {b['name']}：{b.get('trigger', '')}"
        for s in b.get("samples", [])[:2]:
            u = (s.get("user") or "").strip()
            if u:
                line += f"\n    例：{u}"
        defs.append(line)
    prompt = BEHAVIOR_CLASSIFY_PROMPT.format(
        behavior_defs="\n".join(defs), history=history_text or "（无）", user_msg=user_msg,
    )
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
            **THINKING_DISABLED,
        )
        content = (resp.choices[0].message.content or "").strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(content)
        behavior = str(parsed.get("behavior") or "").strip()
        names = {b.get("name") for b in behaviors}
        return behavior if behavior in names else ""
    except Exception as e:
        print(f"⚠️ 行为意图分类失败（降级不触发）: {e}")
        return ""


# ==================== 🎭 基础人设提示词 ====================
if SYSTEM_PROMPT_FILE.exists():
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    raise FileNotFoundError(f"❌ 未找到 {SYSTEM_PROMPT_FILE}")

# ==================== 🛠️ API 客户端（惰性初始化） ====================
# 延迟到首次使用时才读取 config 并创建客户端，
# 保证模块可被独立导入（便于测试），不依赖 NoneBot 已初始化。
_clients_cache = None


def _get_clients():
    """返回 (deepseek_client, zhipu_client)，首次调用时创建。"""
    global _clients_cache
    if _clients_cache is not None:
        return _clients_cache

    _global_config = get_driver().config
    deepseek_api_key = getattr(_global_config, "openai_api_key", None)
    deepseek_api_base = getattr(_global_config, "openai_api_base", DEEPSEEK_BASE_URL)
    zhipu_api_key = getattr(_global_config, "zhipu_api_key", None)

    if not deepseek_api_key:
        raise ValueError("❌ 未检测到 OPENAI_API_KEY")
    if not zhipu_api_key:
        raise ValueError("❌ 未检测到 ZHIPU_API_KEY")

    _clients_cache = (
        AsyncOpenAI(api_key=deepseek_api_key, base_url=deepseek_api_base),
        AsyncOpenAI(api_key=zhipu_api_key, base_url=ZHIPU_BASE_URL),
    )
    return _clients_cache


def _get_model_name() -> str:
    """解析对话模型名：优先 config.openai_model，回退到 DEFAULT_MODEL。"""
    try:
        return getattr(get_driver().config, "openai_model", None) or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def _trim_text(text: str, max_chars: int) -> str:
    """裁剪长文本，超长加省略号。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "……"


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（DP，O(n*m)）。复读检测用。"""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _is_echo_reply(reply: str, recent_bot_replies: list, min_ratio: float = 0.6) -> bool:
    """判断新回复是否复读了最近自己说过的话（复读机防护）。

    复读是模型对情境相关句的执着（会在短期记忆里看到自己的原话再引用），
    提示词的【防复读】拦不住，只能确定性检测：与最近回复完全一致，
    或最长公共子串覆盖较短者的 60% 以上（如"你这话说的……灰泽满刚醒"
    复读后半句"灰泽满刚醒"）。太短的句子（<8字）不判，避免误伤"晚安"类。
    """
    reply = (reply or "").strip()
    if not reply:
        return False
    for old in (recent_bot_replies or [])[-3:]:
        old = (old or "").strip()
        if not old:
            continue
        if reply == old:
            return True
        shorter = min(len(reply), len(old))
        if shorter < 8:
            continue
        if _lcs_len(reply, old) >= shorter * min_ratio:
            return True
    return False


def _split_sentences(text: str) -> list:
    """按句末标点切分（。！？）；省略号仅在"前后都有内容"时才当边界。

    省略号常表"无语/语气"（如"啊……这……"）和犹豫前缀（如"灰泽满……"），不能乱切；
    只有省略号后面跟着新内容（≥5 字）**且前面也有足够内容**（≥4 字）才当停顿边界
    （如"……那你说说"这类，前文太短就不是边界，避免把"灰泽满……"单独切一条发出去）。
    """
    parts = []
    i = 0
    for m in re.finditer(r'[。！？]|…+', text):
        end = m.end()
        if m.group(0).startswith("…"):
            rest = text[end:].lstrip()
            before = text[i:m.start()].rstrip()
            # 后文太少=语气/无语；前文太短=犹豫前缀（"灰泽满……"），都不是边界
            if len(rest) < 5 or len(before) < 4:
                continue
        parts.append(text[i:end])
        i = end
    if i < len(text):
        parts.append(text[i:])
    return [p for p in parts if p.strip()]


def _limit_commas(parts: list) -> list:
    """每段至多 1 个逗号；超了从最后一个逗号拆开（逗号去掉），避免长串逗号连句。"""
    result = []
    for p in parts:
        while True:
            commas = [idx for idx, ch in enumerate(p) if ch in "，,"]
            if len(commas) < 2:
                break
            idx = commas[-1]
            result.append(p[:idx].strip())
            p = p[idx + 1:].strip()
        result.append(p)
    return [x for x in result if x]


def split_reply(reply: str, min_len: int = SPLIT_MIN_LEN,
                max_parts: int = SPLIT_MAX_PARTS) -> list:
    """把长回复按句子断开发送（打字感）。短回复/单句不拆，返回单元素列表。

    切分规则：
    - 句末标点（。！？）切分；省略号仅在后面还有新内容（≥5 字）时才切，保住"啊……"这类无语表达
    - 逗号限制：每段至多 1 个逗号，超了从最后一个逗号拆开（防长串逗号连句）
    - 聊天习惯不打句号：切分后去掉句尾"。"（保留？！…）
    - 超出 max_parts 并入最后一段，避免刷屏
    """
    text = reply.strip().rstrip("。")  # 整条回复末尾的句号也去掉
    if not text or len(text) < min_len:
        return [text]

    parts = _split_sentences(text)
    parts = _limit_commas(parts)
    parts = [p.strip().rstrip("。") for p in parts if p and p.strip()]

    # 纯括号段（（小声嘀咕））并入上一段，避免单独发一条"舞台说明"——它没有实质内容，
    # 单独发会让对方接不住（实测：单独"（小声嘀咕）"→ 对方问"你在说什么"→ 模型瞎编）
    merged = []
    for p in parts:
        if merged and re.fullmatch(r'[（(][^）)]*[）)]', p):
            merged[-1] += p
        else:
            merged.append(p)
    parts = merged

    # 短段并入下一段（业界：merge small chunks with neighbors，防"哦？""那倒是稀奇……"
    # 这种微消息单独成条——按标点机械切碎=表演感/戏剧节拍，短惊讶应是一条自然消息）。
    # 最后一段过短则并入前一段。
    merged_short = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and len(parts[i]) < SPLIT_MERGE_MIN_CHARS:
            parts[i + 1] = parts[i] + parts[i + 1]
        else:
            merged_short.append(parts[i])
        i += 1
    parts = merged_short
    if len(parts) > 1 and len(parts[-1]) < SPLIT_MERGE_MIN_CHARS:
        parts[-2] += parts[-1]
        parts.pop()

    if len(parts) <= 1:
        return [text]

    if len(parts) > max_parts:
        # 超限的分段用换行连接（不能直接拼接——会把两句焊成一句没标点的 run-on，
        # 实测"好像是媒体研究。灰泽满只想打死自己"被拼成"研究灰泽满只想…"）
        merged = "\n".join(p for p in parts[max_parts - 1:] if p).strip()
        parts = parts[:max_parts - 1] + [merged]
    return parts


def split_delay(part_text: str) -> float:
    """句间发送延迟（秒）：按段落长度模拟打字 + ±15% 随机抖动，避免机械等长。"""
    ms = SPLIT_DELAY_BASE_MS + SPLIT_DELAY_PER_CHAR_MS * len(part_text)
    ms = max(SPLIT_DELAY_MIN_MS, min(ms, SPLIT_DELAY_MAX_MS))
    ms *= random.uniform(1 - SPLIT_DELAY_JITTER, 1 + SPLIT_DELAY_JITTER)
    return round(ms / 1000.0, 3)


async def summarize_batch(msgs: list) -> str:
    """把一批消息归纳成一两句话（说了什么 + 语气 + 意图），供模型理解整批。

    只有攒批 ≥2 条才归纳（单条零额外延迟）；只归纳原文已有信息，不编造；
    失败返回空串（调用方忽略，原文照旧）。
    """
    if len(msgs) < 2:
        return ""
    texts = []
    for t, v in msgs:
        if t.strip():
            texts.append(t)
        if v:
            texts.append(f"[图片：{v}]")
    prompt = (
        "以下是用户刚刚连发的几条消息，请用一两句话概括：他们在说什么、什么语气、大致想表达什么。\n"
        "要求：只归纳原文已有的信息，不要编造；不要逐条复述；如果是纯寒暄就直接说。\n\n"
        f"消息：\n" + "\n".join(texts)
    )
    try:
        deepseek_client, _ = _get_clients()
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
            **THINKING_DISABLED,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ 批量归纳失败（忽略）: {e}")
        return ""


# 灰泽满以外的人（第三人称名字）：回复里出现这些名字时，"她/他"可能指别人，不动
_OTHER_PERSON_NAMES_CACHE = None


def _get_other_person_names() -> set:
    """取"灰泽满以外的人"的名字集合：terms 的 person/family/relation 分类 + 常见补充。

    用于 clean_reply 的"她/他自指兜底"保护——回复里出现这些名字时，"她/他"
    大概率指这个人而不是灰泽满自己，故不替换。动态取自 terms，新增人物不用记两处。
    """
    global _OTHER_PERSON_NAMES_CACHE
    if _OTHER_PERSON_NAMES_CACHE is not None:
        return _OTHER_PERSON_NAMES_CACHE
    names = {"女同学", "女仆女同学", "弥希", "真绯瑠", "瑞雅", "塔菲"}
    for t in load_terms():
        if t.get("category") in ("person", "family", "relation") and t.get("keyword"):
            names.add(t["keyword"])
            names.update(str(a) for a in t.get("aliases", []) if a)
    _OTHER_PERSON_NAMES_CACHE = names
    return names


def clean_reply(reply: str) -> str:
    """输出清洗：去括号前缀、整条至多 1 个括号、省略号归一并限频。

    人设规则是"每轮回复至多一个括号、省略号是例外"，但声音样本里带（小声）（心虚），
    模型 few-shot 会学着用。这里做确定性过滤：
    - 剥掉开头"（咽口水）"这类前缀；整条超过 1 个括号时只保留第一个（符合人设额度）
    - 省略号归一（.../…… 串 → ……），整条至多 2 次——保住"啊……这……"这种无语表达，
      掐掉刷屏式……。温度靠语气词/自嘲承载，不靠符号堆砌。
    """
    text = reply.strip()
    # 去掉开头的连续括号前缀，如 （咽口水）你看...
    while text.startswith("（"):
        idx = text.find("）")
        if idx == -1:
            break
        text = text[idx + 1:].lstrip()
    if not text:
        return reply  # 剥光了就回原样，避免空回复
    # 若仍有 ≥2 个括号，只保留第一个，其余删除
    matches = list(re.finditer(r'（[^）]*）', text))
    if len(matches) >= 2:
        first = matches[0]
        before = text[:first.start()]
        kept = first.group(0)
        after = re.sub(r'（[^）]*）', '', text[first.end():])
        text = before + kept + after
    # 省略号纪律：归一连续省略号；剥掉开头省略号（"……那你说说"→"那你说说"，纯"……"无语保留）；
    # 开头"词+省略号"犹豫（"早……灰泽满刚醒"）→ 省略号换逗号（"早，灰泽满刚醒"），保留语气词；
    # 整条至多保留前 2 个
    text = re.sub(r'\.{3,}|…+', '……', text)
    text = re.sub(r'^……(?=\S)', '', text)
    # 开头"短语+省略号"犹豫（"那倒不是……""只是……""早……"）→ 省略号换逗号。
    # 覆盖 1-5 个中文字（短短语/语气词/犹豫开头），只转"后面还有≥4字实质内容"的，
    # 保住"啊……这……"（无语）这类省略号后没内容的。
    text = re.sub(r'^([一-鿿]{1,5})……(?=.{4,})', r'\1，', text)
    if not text:
        text = '……'
    ell_pos = [m.start() for m in re.finditer('……', text)]
    if len(ell_pos) > 2:
        cut = ell_pos[2]
        text = text[:cut] + text[cut:].replace('……', '')
    # 结巴消融："那、那"→"那那"（真人快速打字是叠字，不是顿号；顿号结巴是 RP 腔）
    text = re.sub(r'(.)、\1', r'\1\1', text)
    # 自指"她/他"兜底：对话里灰泽满用名字自称，不该出现"她"指自己。
    # 仅当回复里没出现其他第三人称名字时才替换（这时的"她/他"几乎必指灰泽满自己）。
    # "她"必是代词安全替换；"他"避开"其他/他人/他家"这类复合词。
    if not any(n in text for n in _get_other_person_names()):
        text = re.sub(r"她", "灰泽满", text)
        text = re.sub(r"(?<!其|无)他(?!人|家|国|乡|方|日)", "灰泽满", text)
    return text


def _is_emotion_only_query(query: str) -> bool:
    """判断 probe 补全的检索 query 是否为"纯情绪"描述（如'用户发了个偷笑的表情'）。

    表情/情绪为主的短消息（如'可惜🤭'）会被 probe 按表情规则补全成这类纯情绪句——
    它没有话题词，做语义检索会误命中无关样本（实测'用户发了个偷笑的表情'→ voice_sample
    peer_5'新衣服'，导致'可惜🤭'被回'喜欢新衣服'）。对齐纯表情消息的既有设计：
    跳过语义检索，补全句只作语气提示（query_hint）注入。
    """
    q = (query or "").strip()
    return q.startswith("用户发") and "表情" in q


def _compose_record_msg(user_msg: str, vision_desc: str) -> str:
    """短期记忆里记录的用户消息：纯图片用视觉描述兜底，图文都有则拼接。"""
    text = user_msg.strip()
    if not vision_desc:
        return user_msg
    if text:
        return f"{text}（附图：{vision_desc}）"
    return f"[发送图片] {vision_desc}"


def _split_fused(fused_items):
    """把融合结果按源分组：behavior / corpus / voice_sample / phrase。"""
    behaviors, corpus, samples, phrases = [], [], [], []
    for it in fused_items:
        if it.source == "behavior":
            behaviors.append(it)
        elif it.source == "corpus":
            corpus.append(it)
        elif it.source == "voice_sample":
            samples.append(it)
        elif it.source == "phrase":
            phrases.append(it)
    return behaviors, corpus, samples, phrases


# ==================== 名词库（terms/lorebook） ====================

_terms_cache = None


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


def build_terms_note(user_msg: str) -> str:
    """根据用户消息命中名词库：核心词(always)每次注入 + 命中词(关键词/别名/正则)注入。

    返回注入文本【灰泽满的世界】。客观打底 + 带灰泽满态度，让模型既懂词义又有正确的相处态度。

    命中检查：
    - priority=always：每次注入
    - keyword / aliases：子串命中
    - pattern（可选）：正则命中。用于子串匹配做不到的场景——如'区'单字撞'小区/地区'，
      用 `满区|(这么|太|好|很|真…)\\s*区` 只命中'满区'或形容词用法（'你怎么这么区'），
      不误触普通名词里的'区'。
    """
    terms = load_terms()
    if not terms:
        return ""
    msg = user_msg or ""
    notes = []
    for t in terms:
        kw = t.get("keyword", "")
        if not kw:
            continue
        keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
        pattern = t.get("pattern")
        hit = (t.get("priority") == "always"
               or any(k in msg for k in keys)
               or (pattern and re.search(pattern, msg)))
        if hit:
            parts = [t.get("meaning", "")]
            if t.get("reaction"):
                parts.append(f"被提到时：{t['reaction']}")
            notes.append(f"{kw}：{'；'.join(parts)}")
    return "；".join(notes) if notes else ""


def build_message_list(user_msg: str, global_persona: str, fused_items: list,
                       memory_context: str, user_history: list,
                       vision_desc: str = "", weather_city: str = "",
                       batch_summary: str = "", preference_items: list = None,
                       core_stories: list = None, session_context: str = "",
                       query_hint: str = "") -> list:
    """按优先级组装发送给模型的消息列表。

    fused_items 为三路融合后的 RetrievalItem 列表，按源分组注入。
    vision_desc 为用户消息附带的图片视觉描述（可选）。
    weather_city 为该用户所在城市（空则用全局默认天气城市）。
    batch_summary 为一批消息的智能归纳（可选，提示层）。
    preference_items 为命中相关偏好的条目列表（第 5 路语义检索，可选）。
    core_stories 为命中的核心记忆（印象最深的结晶，可选）。
    session_context 为会话级记忆（当前话题 + 本场事件，可选）。
    query_hint 为短消息的语境扩充（可选）：模型理解短消息用，正文仍是原 msg。
    """
    messages = []
    base_system = SYSTEM_PROMPT
    if global_persona:
        base_system += "\n\n" + global_persona
    messages.append({"role": "system", "content": base_system})

    # 偏好档案（第 5 路语义检索）：聊到相关话题才注入；与语料/记忆冲突时以偏好为准
    if preference_items:
        prefs_text = "；".join(
            f"{p.get('category', '')}：{p.get('text', '')}" for p in preference_items
        )
        if prefs_text:
            messages.append({
                "role": "system",
                "content": f"【灰泽满的偏好】{prefs_text}（这是她稳定真实的偏好，若与直播记忆/聊天记录冲突，以本条为准）",
            })

    # 核心记忆（印象最深的结晶）：粉丝常提及的过去故事，命中才注入
    if core_stories:
        story_text = "；".join(
            f"{s.get('category', '')}：{s.get('text', '')}" for s in core_stories
        )
        if story_text:
            messages.append({
                "role": "system",
                "content": f"【她的核心记忆】{story_text}（这是她过去最深刻的经历，粉丝常拿这些开玩笑。**回应时自然带出，别整段复述**：被问你的书/事迹时，大方承认、可提议念给TA听或说个片段概括，别一口气把原文/全文念出来——除非对方明确说'念一下'）",
            })

    # 名词库（terms/lorebook）：核心词 always 注入 + 命中用户消息的词注入——让模型懂"绿冻/枪神8/slg"这类词并带对的态度
    terms_note = build_terms_note(user_msg)
    if terms_note:
        messages.append({
            "role": "system",
            "content": f"【灰泽满的世界】{terms_note}（这些是她世界的词，遇到时按定义理解并带着对应的态度，别当成普通的词）",
        })

    # 会话级记忆（当前话题 + 本场事件）：让模型接得住会话调性
    if session_context:
        messages.append({
            "role": "system",
            "content": f"【当前会话】{session_context}\n（这是你们这一场对话的调性和发生过的事，回应时要自然地顺着这个语境，不要生硬提及）",
        })

    # 感知源①：当前时间/农历/天气（始终注入，占预算极少；天气按该用户所在城市）
    now_context = context_probe.get_now_context(city=weather_city)
    if now_context:
        messages.append({"role": "system", "content": now_context})

    behaviors, corpus, samples, phrases = _split_fused(fused_items)

    # 行为指令（source=behavior）
    if behaviors:
        behavior_text = "\n\n".join(it.text for it in behaviors if it.text)
        if behavior_text:
            messages.append({
                "role": "system",
                "content": f"【当前情境下的行为指令】请严格按此模式回应：\n{behavior_text}"
            })

    # 直播记忆（source=corpus）：只当"背景记忆"，不参与风格示范
    if corpus:
        context = "\n".join(f"- {it.text}" for it in corpus if it.text)
        if context:
            messages.append({
                "role": "system",
                "content": f"【她经历过的相关背景】以下是她经历过的相关的事，仅当对当前话题有帮助时自然提及；不要模仿里面的叙述口吻，不要整段复述。说话风格看下面的样本：\n{context}"
            })

    # 长期记忆注入
    if memory_context:
        messages.append({
            "role": "system",
            "content": f"【关于这个绿冻的长期记忆】\n{memory_context}"
        })

    # 短期记忆注入
    if user_history:
        if isinstance(user_history, list):
            context = "\n".join(user_history)
            # 一致性规则 + 防复读：灰泽满自己的话是历史背景，用户没主动追问就不重复
            context += (
                "\n\n【一致性规则】解释同一件事（如'今天为什么没播'）时，借口要与之前保持一致，"
                "不要前后矛盾；但被问到新问题（如'明天会不会播'）时正常回答，不要强行沿用旧借口。\n"
                "【防复读】以上对话中，灰泽满自己说过的话（如'天气冷''在忙什么'）只是历史背景，"
                "用户没有主动追问时，不要反复重复提起；自然顺着用户当前的问题回答即可，"
                "不要每轮都重提自己之前提过的事。"
            )
            label = "【最近对话记录】"
        else:
            context = f'我说："{user_history}"'
            label = "【关于这个绿冻的上一轮记忆】"
        messages.append({
            "role": "system",
            "content": f"{label}\n{context}"
        })

    # 措辞指纹（source=phrase）：同一意思用她的真实原话锚定，不自创措辞
    if phrases:
        phrase_blocks = []
        for it in phrases:
            usage = it.extra.get("usage", "")
            phs = it.extra.get("phrases", [])[:PHRASE_PHASES_MAX]
            if phs:
                block = f"· {it.extra.get('meaning', it.item_id)}：{'、'.join(phs)}"
                if usage:
                    block += f"（{usage}）"
                phrase_blocks.append(block)
        if phrase_blocks:
            messages.append({
                "role": "system",
                "content": "【她的固定说法】以下情景她说这些话。表达同类意思时用这些原话组织，不要自创解释性措辞：\n" + "\n".join(phrase_blocks)
            })

    # 声音样本 few-shot（source=voice_sample）：示范灰泽满"怎么说话"
    if samples:
        messages.append({
            "role": "system",
            "content": "【灰泽满的说话方式参考】以下是她真实的对话片段。只学其中的语气、断句、省略号、自称（灰泽满/hzm）和措辞。括号是她的'心里话标注'，只在情绪顶点才用一个（如（小声）），日常回复默认一个都不用。内容要针对当前话题，不要复述、也不要套用示例里的具体内容（人物/礼物/衣服/事件等）。日常回复保持短句（30字内），简短干脆。"
        })
        for it in samples:
            user_part = it.extra.get("user", "")
            reply_part = it.extra.get("reply", "")
            if user_part and reply_part:
                messages.append({"role": "user", "content": user_part})
                messages.append({"role": "assistant", "content": _trim_text(reply_part, VOICE_SAMPLE_REPLY_TRIM_CHARS)})

    # 极简长度提醒：一句一停，不展开
    messages.append({
        "role": "system",
        "content": "【回复节奏】日常闲聊：一句话说完就停，不再补第二句。30字内。"
    })

    # 感知源②：图片消息——把视觉描述并入用户消息，避免空消息让模型以为"对方没说话"
    final_user = user_msg
    if vision_desc:
        final_user = f"{user_msg}\n[图片：{vision_desc}]" if user_msg.strip() else f"[图片：{vision_desc}]"

    # 感知源③：批量归纳（用户连发多条时，作为理解整批的提示层，原文仍完整保留）
    if batch_summary:
        messages.append({
            "role": "system",
            "content": f"【这批消息的归纳】{batch_summary}"
        })

    # 短消息语境提示：用户消息 ≤4 字时，把扩充后的完整语义作为提示给模型
    # （正文仍是原 msg，这里帮模型理解"咋这样"这种短句的真实含义）
    if query_hint:
        # 纯表情消息：按表情的真实情绪回应，体现情绪该有的态度，不被当前话题绑架
        if is_emoji_msg(final_user):
            emoji_hint = (
                f"【用户发了表情】{query_hint}\n"
                "用户只发了一个表情，没有任何文字。请按这个表情的真实情绪回应，并体现这种情绪该有的态度：\n"
                "- 无语/无奈（😅）→ 略带攻击性地反击，类似'感觉你不是很服气？'\n"
                "- 委屈/哭（😭）→ 心软安慰，但不要套用当前话题的模板（如别硬扯'夸你可爱'）\n"
                "- 其他情绪 → 按情绪的自然反应回应\n"
                "不要复述表情，不要顺着当前话题硬接，只回应这个情绪。"
            )
            messages.append({"role": "system", "content": emoji_hint})
        else:
            messages.append({
                "role": "system",
                "content": f"【用户这条消息的语境】{query_hint}\n（上面是这条消息在当前语境下的完整意思——短消息或指代性消息（如'能读给我听听吗'）需要结合前文才能理解，按这个理解回复；不要复述这句话）"
            })

    messages.append({"role": "user", "content": final_user})
    return messages


async def generate_reply(messages: list) -> str:
    """调用 DeepSeek 生成回复；失败时返回带错误的兜底文本。"""
    try:
        deepseek_client, _ = _get_clients()
        response = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=messages,
            temperature=CHAT_TEMPERATURE,
            frequency_penalty=CHAT_FREQUENCY_PENALTY,
            max_tokens=CHAT_MAX_TOKENS,
            **THINKING_DISABLED,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"哎呀，hzm脑子卡了一下……（错误: {e}）"
    return reply if reply else "……（沉默，可能是信号不好）"


def _repair_llm_json(text: str) -> str:
    """修复 LLM 常见的不规范 JSON（DeepSeek 偶发），尽力让 json.loads 能过。

    常见病：键没加双引号（{name: "x"}）、单引号键/值、尾逗号、markdown 围栏、前后杂质。
    修不好的原样返回，交给调用方兜底（重试/丢弃）。
    """
    if not text:
        return text
    t = text.strip()
    # 剥 markdown 代码围栏
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # 只取最外层 {…} / […]（剥掉前后杂质，如模型先写"好的"）
    start = min((i for i in (t.find("{"), t.find("[")) if i != -1), default=-1)
    end = max(t.rfind("}"), t.rfind("]"))
    if start != -1 and end > start:
        t = t[start:end + 1]
    # 键补双引号：单引号键 {'a': …} 和裸键 {a: …}
    t = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', t)
    t = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', t)
    # 单引号字符串值 → 双引号
    t = re.sub(r":\s*'([^']*)'", lambda m: ': "' + m.group(1).replace('"', '\\"') + '"', t)
    # 尾逗号 ,} / ,]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def _parse_memory_extract(content: str) -> dict:
    """把记忆提取 LLM 的输出解析为 dict：剥围栏 + 修复不规范 JSON。

    内容为 "null" 返回 {}；修复后仍不是合法 JSON 则抛异常（调用方重试一次）。
    """
    content = (content or "").strip()
    if content == "null":
        return {}
    return json.loads(_repair_llm_json(content))


async def update_memory_task(user_id: str, user_msg: str, reply: str, user_memory_card: dict):
    """异步提取并更新长期记忆。"""
    # 密度门控：太短/纯表情的消息不值得提取（省成本减噪音）
    msg = (user_msg or "").strip()
    if not msg or len(msg) < 4:
        return
    if msg.startswith("[表情：") and msg.endswith("]"):
        return
    try:
        deepseek_client, _ = _get_clients()
    except Exception as e:
        print(f"[长期记忆] 客户端初始化失败: {e}")
        return

    # 用可读画像摘要替代原生 JSON dump，让模型能可靠 dedup/冲突检测
    current_summary = _format_profile_summary(user_memory_card)
    prompt = MEMORY_EXTRACT_PROMPT.format(
        current_summary=current_summary,
        user_msg=user_msg,
        reply=reply
    )
    # V1：停用 self_fact 提取。灰泽满的"自我"应来自真人素材（voice_samples/corpus），
    # 而不是聊天时临时编造的自我披露，防止 AI 自嗨污染长期人格。
    prompt += "\n【本轮的强制规则】new_self_fact 一律返回 null。只提取关于用户的信息（new_impression / new_user_fact），不要从灰泽满的回复中提取任何自我披露内容。"

    content = None
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=MEMORY_EXTRACT_TEMPERATURE,
            max_tokens=MEMORY_EXTRACT_MAX_TOKENS,
            **THINKING_DISABLED,
        )
        content = resp.choices[0].message.content.strip()
        print(f"[长期记忆] 提取结果: {content}")
        if content and content.strip() != "null":
            updates = _parse_memory_extract(content)
            if updates:
                update_user_memory(user_id, updates)
    except Exception as e:
        # 首次失败（多为 JSON 不规范/偶发）：严格格式重试一次。记忆提取是异步后台任务，重试不阻塞聊天。
        print(f"[长期记忆] 首次解析失败（{e}），严格格式重试一次")
        try:
            strict_prompt = prompt + (
                "\n【强制格式】输出必须是严格 JSON：所有键和字符串值都加双引号，"
                "null 不带引号，键后冒号，不要 markdown 围栏，不要任何额外文字。"
            )
            resp2 = await deepseek_client.chat.completions.create(
                model=_get_model_name(),
                messages=[{"role": "user", "content": strict_prompt}],
                temperature=MEMORY_EXTRACT_TEMPERATURE,
                max_tokens=MEMORY_EXTRACT_MAX_TOKENS,
                **THINKING_DISABLED,
            )
            content2 = resp2.choices[0].message.content.strip()
            print(f"[长期记忆] 重试提取: {content2}")
            updates = _parse_memory_extract(content2)
            if updates:
                update_user_memory(user_id, updates)
        except Exception as e2:
            print(f"[长期记忆] 重试仍失败，本轮记忆丢弃。首次原始输出: {content!r}")
            import traceback
            traceback.print_exc()


async def handle_chat(user_id: str, user_msg: str, vision_desc: str = "",
                      batch_summary: str = "") -> str:
    """处理一条用户消息，返回机器人回复。vision_desc 为图片描述；batch_summary 为批量归纳。"""
    deepseek_client, zhipu_client = _get_clients()

    # --- 会话级记忆：对话前同步探测（判断话题延续/转换 + 短 query 扩充） ---
    # 必须在组装消息前完成，这样本轮注入的就是本轮自己的话题，不滞后一轮。
    query_text = user_msg.strip()
    user_history = get_user_history(user_id)
    history_text = "\n".join(user_history[-6:]) if user_history else ""
    retrieval_query = query_text
    if query_text:
        retrieval_query = await probe_session(user_id, query_text, history_text, deepseek_client)
        if retrieval_query != query_text:
            print(f"[会话记忆] 短 query 扩充: 「{query_text}」→「{retrieval_query}」")
    # 话题/事件已在本轮探测中更新，取最新会话状态
    session_context = build_session_context(user_id)

    # --- 🃏 经典梗硬匹配（双路由：关键词粗筛 + LLM 语境确认，防误触发） ---
    _confirm_history = ""
    for trigger, replies in LEGENDARY_REPLIES.items():
        if trigger in user_msg:
            confirm_tpl = LEGENDARY_CONFIRMS.get(trigger)
            if confirm_tpl:
                if not _confirm_history:
                    _confirm_history = "\n".join(get_user_history(user_id)[-4:])
                if not await _legendary_confirmed(user_msg, confirm_tpl, history=_confirm_history):
                    print(f"[梗] 关键词 {trigger!r} 命中但 LLM 未确认，落到正常管线")
                    break  # 不是目标语境，放弃梗，走正常回复
            reply = random.choice(replies)
            # 梗匹配也记入短期记忆 + 异步长期记忆，避免后续对话"失忆"
            append_user_history(user_id, user_msg, reply)
            card = get_user_memory(user_id)
            asyncio.create_task(update_memory_task(user_id, user_msg, reply, card))
            return reply

    # --- 🎭 人格规则 ---
    traits, styles, behaviors = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)

    # --- 🔍 检索 + 融合（query 只算 1 次 embedding） ---
    # 纯图片消息（无文字）不做检索：让灰泽满直接评价图片，避免语料/行为劫持图片内容
    # 纯表情消息（emoji/[表情：xx]）也不做语义检索：表情只表达情绪不表达话题，
    # 扩充句会作为语气提示注入，但检索 memory 会跑偏（如😭命中"被夸"样本）
    if query_text and not is_emoji_msg(query_text):
        # 纯情绪消息（如'可惜🤭'，被 probe 补全成'用户发了个偷笑的表情'）无话题词，
        # 语义检索会误命中无关样本（实测→peer_5'新衣服'）。对齐纯表情设计：跳过检索，
        # 补全句只作语气提示（query_hint）注入。
        if _is_emotion_only_query(retrieval_query):
            fused_items = []
            preference_items = []
            core_stories = []
        else:
            # L3：行为归属用 LLM 判意图（不再用 embedding 猜——embedding 按句式聚团，
            # 会把'灰泽满你唱歌好听'（夸）和'灰泽满你怎么又迟到'（质问）挤在一起误判）。
            # 判别词（敷衍/骗/鸽/迟到/黄桃/擦边…）命中则直接命中（可靠且省一次 LLM 调用）；
            # 否则 LLM 判意图，判别词在 LLM 拿不准时兜底。
            kw_item = select_behavior_item(query_text, "", behaviors)
            if kw_item:
                behavior_items = [kw_item]
            else:
                behavior_intent = await classify_behavior(deepseek_client, query_text, history_text, behaviors)
                if behavior_intent:
                    print(f"[行为] LLM 判定: {behavior_intent}")
                behavior_item = select_behavior_item(query_text, behavior_intent, behaviors)
                behavior_items = [behavior_item] if behavior_item else []
            query_vector = await embed_query(zhipu_client, retrieval_query or query_text)
            corpus_items = retrieve_corpus(retrieval_query or query_text, query_vector)
            sample_items = retrieve_voice_samples(retrieval_query or query_text, query_vector)
            phrase_items = retrieve_phrases(retrieval_query or query_text, query_vector)
            fused_items = fuse_and_truncate(corpus_items, sample_items, behavior_items, phrase_items)
            preference_items = retrieve_preferences(retrieval_query or query_text, query_vector)  # 第 5 路：偏好
            core_stories = retrieve_core_stories(retrieval_query or query_text, query_vector)     # 核心记忆（结晶）
    else:
        fused_items = []
        preference_items = []
        core_stories = []

    # --- 🧠 确定性两路记忆 ---
    user_memory_card = get_user_memory(user_id)
    memory_context = build_memory_context(user_memory_card)
    weather_city = (user_memory_card or {}).get("weather_city", "") or ""

    # --- 🧩 构建消息列表 ---
    # query_hint：短消息（≤4字）的语境扩充，仅当扩充句与原文不同时传入，帮模型理解短句
    query_hint = retrieval_query if (retrieval_query and retrieval_query != query_text) else ""
    messages = build_message_list(
        user_msg, global_persona, fused_items, memory_context, user_history,
        vision_desc=vision_desc, weather_city=weather_city, batch_summary=batch_summary,
        preference_items=preference_items, core_stories=core_stories,
        session_context=session_context, query_hint=query_hint,
    )

    # --- 🤖 调用大模型 ---
    reply = await generate_reply(messages)

    # --- 🔁 复读机防护（确定性，治本）---
    # 模型会对情境相关句执着复读（自己说过的话进短期记忆后再被引用），提示词拦不住。
    # 检测到与最近自己说过的话重复，就加"别复读"提示强制重新生成（最多 2 次）。
    recent_bot = [ln[4:] for ln in get_user_history(user_id) if ln.startswith("灰泽满：")]
    if _is_echo_reply(reply, recent_bot):
        print(f"[防复读] 检测到复读『{reply[:20]}』，强制重新生成")
        nudge = {
            "role": "system",
            "content": f"警告：你刚说过『{reply}』，几乎原样复读会让人反感。用完全不同的说法重新回复这条消息，别重复这句话。",
        }
        for _ in range(2):
            reply = await generate_reply(list(messages) + [nudge])
            if not _is_echo_reply(reply, recent_bot):
                break

    # --- 💾 更新短期记忆（带锁）：图片消息把视觉描述记进去，后续才记得聊过什么图 ---
    record_msg = _compose_record_msg(user_msg, vision_desc)
    append_user_history(user_id, record_msg, reply)

    # --- 📝 异步更新长期记忆（会话级记忆已在对话前 probe_session 同步更新） ---
    asyncio.create_task(update_memory_task(user_id, record_msg, reply, user_memory_card))

    return reply
