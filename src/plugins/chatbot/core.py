"""消息处理主循环。

组装系统提示词（基础人设 + 人格规则 + 行为指令 + RAG + 长期/短期记忆），
调用 DeepSeek 生成回复，并异步更新长期记忆。
"""
import json
import random
import asyncio

from nonebot import get_driver
from openai import AsyncOpenAI

from .constants import (
    SYSTEM_PROMPT_FILE, DEEPSEEK_BASE_URL, ZHIPU_BASE_URL,
    DEFAULT_MODEL, THINKING_DISABLED,
    CHAT_TEMPERATURE, CHAT_MAX_TOKENS,
    MEMORY_EXTRACT_TEMPERATURE, MEMORY_EXTRACT_MAX_TOKENS,
)
from .persona import load_persona_rules, build_global_persona_context, match_behaviors_semantic
from .memory import (
    get_user_history, append_user_history,
    get_user_memory, update_user_memory, build_memory_context,
    MEMORY_EXTRACT_PROMPT,
)
from .rag import embed_query, retrieve_semantic_contexts

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


def build_message_list(user_msg: str, global_persona: str, matched_behavior: str,
                       retrieved_context: str, memory_context: str, user_history: list) -> list:
    """按优先级组装发送给模型的消息列表。"""
    messages = []
    base_system = SYSTEM_PROMPT
    if global_persona:
        base_system += "\n\n" + global_persona
    messages.append({"role": "system", "content": base_system})

    if matched_behavior:
        messages.append({
            "role": "system",
            "content": f"【当前情境下的行为指令】请严格按此模式回应：\n{matched_behavior}"
        })

    if retrieved_context:
        messages.append({
            "role": "system",
            "content": f"【历史记忆片段（模仿语气，勿复读）】:\n{retrieved_context}"
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
            context += "\n\n【强制规则】请先阅读以上对话记录。如果你之前已经给过某个借口（如'被作业封印''泡面洒了''睡过头'），本轮必须沿用同一个借口，禁止在相邻几轮中编造不同的借口。如果你之前承诺过直播时间，不要更改。"
            label = "【最近对话记录】"
        else:
            context = f'我说："{user_history}"'
            label = "【关于这个绿冻的上一轮记忆】"
        messages.append({
            "role": "system",
            "content": f"{label}\n{context}"
        })

    messages.append({"role": "user", "content": user_msg})
    return messages


async def generate_reply(messages: list) -> str:
    """调用 DeepSeek 生成回复；失败时返回带错误的兜底文本。"""
    try:
        deepseek_client, _ = _get_clients()
        response = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=messages,
            temperature=CHAT_TEMPERATURE,
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


async def handle_chat(user_id: str, user_msg: str) -> str:
    """处理一条用户消息，返回机器人回复。"""
    # --- 🃏 经典梗硬匹配 ---
    for trigger, replies in LEGENDARY_REPLIES.items():
        if trigger in user_msg:
            return random.choice(replies)

    # --- 💾 短期记忆 ---
    user_history = get_user_history(user_id)

    # --- 🎭 人格规则 + 行为匹配 ---
    traits, styles, behaviors = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)
    # 一次 embedding：query 向量供行为匹配与 RAG 共用
    _, zhipu_client = _get_clients()
    query_vector = await embed_query(zhipu_client, user_msg)
    matched_behavior = await match_behaviors_semantic(user_msg, query_vector, behaviors)

    # --- 📚 RAG 记忆 ---
    retrieved_context = await retrieve_semantic_contexts(user_msg, query_vector)

    # --- 🧠 长期记忆 ---
    user_memory_card = get_user_memory(user_id)
    memory_context = build_memory_context(user_memory_card)

    # --- 🧩 构建消息列表 ---
    messages = build_message_list(
        user_msg, global_persona, matched_behavior, retrieved_context, memory_context, user_history
    )

    # --- 🤖 调用大模型 ---
    reply = await generate_reply(messages)

    # --- 💾 更新短期记忆（带锁） ---
    append_user_history(user_id, user_msg, reply)

    # --- 📝 异步更新长期记忆 ---
    asyncio.create_task(update_memory_task(user_id, user_msg, reply, user_memory_card))

    return reply
