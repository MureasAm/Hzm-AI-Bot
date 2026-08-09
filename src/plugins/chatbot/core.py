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
    SYSTEM_PROMPT_FILE, DEEPSEEK_BASE_URL, ZHIPU_BASE_URL,
    DEFAULT_MODEL, THINKING_DISABLED,
    CHAT_TEMPERATURE, CHAT_FREQUENCY_PENALTY, CHAT_MAX_TOKENS,
    MEMORY_EXTRACT_TEMPERATURE, MEMORY_EXTRACT_MAX_TOKENS,
    VOICE_SAMPLE_REPLY_TRIM_CHARS,
)
from .persona import load_persona_rules, build_global_persona_context
from .memory import (
    get_user_history, append_user_history,
    get_user_memory, update_user_memory, build_memory_context,
    MEMORY_EXTRACT_PROMPT,
)
from .rag import embed_query
from .retrieval import (
    retrieve_corpus, retrieve_voice_samples, retrieve_behaviors, retrieve_phrases,
    fuse_and_truncate,
)
from .constants import (
    PHRASE_PHASES_MAX, SPLIT_MIN_LEN, SPLIT_MAX_PARTS,
    SPLIT_DELAY_BASE_MS, SPLIT_DELAY_PER_CHAR_MS,
    SPLIT_DELAY_MIN_MS, SPLIT_DELAY_MAX_MS, SPLIT_DELAY_JITTER,
)
from . import context_probe

# ==================== 💬 经典梗硬匹配库 ====================
LEGENDARY_REPLIES = {
    "爱不爱绿冻": [
        "早就说过很爱了...",
        "爱是也可以的，不爱也可以的~"
    ],
    "在和谁说话": [
        "在和..在和你说话哦~",
        "在和弹幕说话~"
    ]
}

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


def split_reply(reply: str, min_len: int = SPLIT_MIN_LEN,
                max_parts: int = SPLIT_MAX_PARTS) -> list:
    """把长回复按句子断开发送（打字感）。短回复/单句不拆，返回单元素列表。

    按句末标点（。！？…）切分；连续标点归并（"……"不断开）；
    超出 max_parts 的碎片并入最后一段，避免刷屏。
    聊天习惯不打句号：切分后去掉句尾的"。"（保留 ？！…）。
    """
    text = reply.strip().rstrip("。")  # 整条回复末尾的句号也去掉
    if not text or len(text) < min_len:
        return [text]

    parts = re.findall(r'[^。！？…]*[。！？…]+', text)
    tail = text[len(''.join(parts)):].strip()
    if tail:
        parts.append(tail)
    # 去句号：聊天不打句号，其他标点保留
    parts = [p.strip().rstrip("。") for p in parts if p and p.strip()]

    if len(parts) <= 1:
        return [text]

    if len(parts) > max_parts:
        parts = parts[:max_parts - 1] + [''.join(parts[max_parts - 1:]).strip()]
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


def clean_reply(reply: str) -> str:
    """输出清洗：去掉开头（动作/情绪）括号前缀，整条至多保留 1 个括号。

    人设规则是"每轮回复至多一个括号、日常通常不用"，但声音样本里带（小声）（心虚），
    模型 few-shot 会学着用。这里做确定性过滤：剥掉开头"（咽口水）"这类前缀，
    整条超过 1 个括号时只保留第一个（符合人设额度）。温度靠语气词/自嘲/省略号承载。
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
    return text


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


def build_message_list(user_msg: str, global_persona: str, fused_items: list,
                       memory_context: str, user_history: list,
                       vision_desc: str = "", weather_city: str = "",
                       batch_summary: str = "") -> list:
    """按优先级组装发送给模型的消息列表。

    fused_items 为三路融合后的 RetrievalItem 列表，按源分组注入。
    vision_desc 为用户消息附带的图片视觉描述（可选）。
    weather_city 为该用户所在城市（空则用全局默认天气城市）。
    batch_summary 为一批消息的智能归纳（可选，提示层）。
    """
    messages = []
    base_system = SYSTEM_PROMPT
    if global_persona:
        base_system += "\n\n" + global_persona
    messages.append({"role": "system", "content": base_system})

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
            # 增加一致性强制指令
            context += "\n\n【一致性规则】解释同一件事（如'今天为什么没播'）时，借口要与之前保持一致，不要前后矛盾；但被问到新问题（如'明天会不会播'）时正常回答，不要强行沿用旧借口。"
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
            "content": "【灰泽满的说话方式参考】以下是她真实的对话片段。模仿其中的语气、断句、省略号、自称（灰泽满/hzm）和措辞。括号是她的'心里话标注'，只在情绪顶点才用一个（如（小声）），日常回复默认一个都不用。内容要针对当前话题不要复述示例。日常回复保持短句（30字内），简短干脆。"
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


async def update_memory_task(user_id: str, user_msg: str, reply: str, user_memory_card: dict):
    """异步提取并更新长期记忆。"""
    try:
        deepseek_client, _ = _get_clients()
        current_summary = json.dumps(user_memory_card, ensure_ascii=False) if user_memory_card else "无"
        prompt = MEMORY_EXTRACT_PROMPT.format(
            current_summary=current_summary,
            user_msg=user_msg,
            reply=reply
        )
        # V1：停用 self_fact 提取。灰泽满的"自我"应来自真人素材（voice_samples/corpus），
        # 而不是聊天时临时编造的自我披露，防止 AI 自嗨污染长期人格。
        prompt += "\n【本轮的强制规则】new_self_fact 一律返回 null。只提取关于用户的信息（new_impression / new_user_fact），不要从灰泽满的回复中提取任何自我披露内容。"
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=MEMORY_EXTRACT_TEMPERATURE,
            max_tokens=MEMORY_EXTRACT_MAX_TOKENS,
            **THINKING_DISABLED,
        )
        content = resp.choices[0].message.content.strip()
        print(f"[长期记忆] 提取结果: {content}")
        if content and content != "null":
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            updates = json.loads(content)
            update_user_memory(user_id, updates)
    except Exception as e:
        print(f"[长期记忆] 更新失败: {e}")
        import traceback
        traceback.print_exc()


async def handle_chat(user_id: str, user_msg: str, vision_desc: str = "",
                      batch_summary: str = "") -> str:
    """处理一条用户消息，返回机器人回复。vision_desc 为图片描述；batch_summary 为批量归纳。"""
    # --- 🃏 经典梗硬匹配 ---
    for trigger, replies in LEGENDARY_REPLIES.items():
        if trigger in user_msg:
            return random.choice(replies)

    # --- 🎭 人格规则 ---
    traits, styles, behaviors = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)

    # --- 🔍 检索 + 融合（query 只算 1 次 embedding） ---
    # 纯图片消息（无文字）不做检索：让灰泽满直接评价图片，避免语料/行为劫持图片内容
    query_text = user_msg.strip()
    _, zhipu_client = _get_clients()
    if query_text:
        query_vector = await embed_query(zhipu_client, query_text)
        corpus_items = retrieve_corpus(query_text, query_vector)
        sample_items = retrieve_voice_samples(query_text, query_vector)
        behavior_items = retrieve_behaviors(query_text, query_vector, behaviors)
        phrase_items = retrieve_phrases(query_text, query_vector)
        fused_items = fuse_and_truncate(corpus_items, sample_items, behavior_items, phrase_items)
    else:
        fused_items = []

    # --- 🧠 确定性两路记忆 ---
    user_memory_card = get_user_memory(user_id)
    memory_context = build_memory_context(user_memory_card)
    user_history = get_user_history(user_id)
    weather_city = (user_memory_card or {}).get("weather_city", "") or ""

    # --- 🧩 构建消息列表 ---
    messages = build_message_list(
        user_msg, global_persona, fused_items, memory_context, user_history,
        vision_desc=vision_desc, weather_city=weather_city, batch_summary=batch_summary,
    )

    # --- 🤖 调用大模型 ---
    reply = await generate_reply(messages)

    # --- 💾 更新短期记忆（带锁）：图片消息把视觉描述记进去，后续才记得聊过什么图 ---
    record_msg = _compose_record_msg(user_msg, vision_desc)
    append_user_history(user_id, record_msg, reply)

    # --- 📝 异步更新长期记忆 ---
    asyncio.create_task(update_memory_task(user_id, record_msg, reply, user_memory_card))

    return reply
