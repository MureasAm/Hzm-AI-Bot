from nonebot import on_message, get_driver
from nonebot.adapters.onebot.v11 import Bot, Event, Message
from nonebot.params import EventPlainText
from openai import AsyncOpenAI
import json
import random
import asyncio
from pathlib import Path
import sys




# 确保能导入项目根目录下的 memory_manager 模块
sys.path.append(str(Path(__file__).parent.parent.parent))
from memory_manager import (
    get_user_memory, update_user_memory, build_memory_context, 
    MEMORY_EXTRACT_PROMPT
)

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
SYSTEM_PROMPT_FILE = Path("persona/system_prompt.txt")
if SYSTEM_PROMPT_FILE.exists():
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    raise FileNotFoundError("❌ 未找到 persona/system_prompt.txt")

# ==================== 📁 文件路径配置 ====================
MEMORY_FILE = Path("memory.json")                # 用户短期记忆
VECTOR_FILE = Path("corpus_vectors.json")        # 直播记忆向量库

# 人格规则文件
TRAITS_FILE = Path("persona/persona_traits.json")
STYLES_FILE = Path("persona/persona_styles.json")
BEHAVIORS_FILE = Path("persona/persona_behaviors.json")

# ==================== 🛠️ API 配置 ====================
global_config = get_driver().config

deepseek_api_key = getattr(global_config, "openai_api_key", None)
deepseek_api_base = getattr(global_config, "openai_api_base", "https://api.deepseek.com/v1")
model_name = getattr(global_config, "openai_model", "deepseek-chat")

zhipu_api_key = getattr(global_config, "zhipu_api_key", None)

if not deepseek_api_key:
    raise ValueError("❌ 未检测到 OPENAI_API_KEY")
if not zhipu_api_key:
    raise ValueError("❌ 未检测到 ZHIPU_API_KEY")

deepseek_client = AsyncOpenAI(api_key=deepseek_api_key, base_url=deepseek_api_base)
zhipu_client = AsyncOpenAI(api_key=zhipu_api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")

# ==================== 📐 余弦相似度 ====================
def cosine_similarity(v1, v2) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if not norm_v1 or not norm_v2:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# ==================== 🧠 人格规则加载 ====================
def load_persona_rules():
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

async def match_behaviors_semantic(user_query: str, behaviors: list, threshold: float = 0.65) -> str:
    if not behaviors:
        return ""

    triggers = []
    for b in behaviors:
        t = b.get("trigger", "")
        triggers.append(t if t else "")

    try:
        query_resp = await zhipu_client.embeddings.create(
            model="embedding-3",
            input=user_query
        )
        query_vec = query_resp.data[0].embedding
    except Exception as e:
        print(f"⚠️ 获取用户消息 embedding 失败: {e}")
        return ""

    best_sim = -1
    best_rule = None
    for idx, b in enumerate(behaviors):
        t = triggers[idx]
        if not t:
            continue
        try:
            resp = await zhipu_client.embeddings.create(
                model="embedding-3",
                input=t
            )
            trigger_vec = resp.data[0].embedding
            sim = cosine_similarity(query_vec, trigger_vec)
            if sim > best_sim:
                best_sim = sim
                best_rule = b
        except Exception as e:
            print(f"⚠️ 计算 trigger embedding 失败: {e}")
            continue

    if best_rule and best_sim >= threshold:
        name = best_rule.get("name", "")
        desc = best_rule.get("response", "")
        trigger_desc = best_rule.get("trigger", "")
        parts = []
        if name:
            parts.append(f"【{name}】")
        if trigger_desc:
            parts.append(f"触发情境：{trigger_desc}")
        if desc:
            parts.append(f"回应模式：{desc}")
        return "\n".join(parts)
    return ""

# ==================== 📚 直播记忆 RAG 检索 ====================
async def retrieve_semantic_contexts(user_query: str, top_k: int = 2) -> str:
    if not VECTOR_FILE.exists():
        return ""
    try:
        with open(VECTOR_FILE, "r", encoding="utf-8") as f:
            vector_db = json.load(f)
        if not vector_db:
            return ""

        response = await zhipu_client.embeddings.create(
            model="embedding-3",
            input=user_query
        )
        query_vector = response.data[0].embedding

        scored = []
        for item in vector_db:
            sim = cosine_similarity(query_vector, item["vector"])
            scored.append((sim, item["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        valid = [text for score, text in scored[:top_k] if score > 0.35]
        if not valid:
            return ""
        return "\n".join([f"- {text}" for text in valid])
    except Exception as e:
        print(f"⚠️ RAG 检索失败: {e}")
        return ""

# ==================== 💬 消息处理主逻辑 ====================
chat = on_message(priority=10, block=True)

@chat.handle()
async def handle_chat(bot: Bot, event: Event, user_msg: str = EventPlainText()):
    user_id = event.get_user_id()
    print(f"[收到消息] user={user_id}, msg={user_msg}")

    # --- 🃏 经典梗硬匹配 ---
    for trigger, replies in LEGENDARY_REPLIES.items():
        if trigger in user_msg:
            await chat.finish(Message(random.choice(replies)))
            return

    # --- 💾 短期记忆（最近 3 轮对话） ---
    memory = {}
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    memory = json.loads(content)
        except (json.JSONDecodeError, Exception):
            memory = {}
    user_memory = memory.get(user_id, [])

    # --- 🎭 人格规则 ---
    traits, styles, behaviors = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)
    matched_behavior = await match_behaviors_semantic(user_msg, behaviors)

    # --- 📚 RAG 记忆 ---
    retrieved_context = await retrieve_semantic_contexts(user_msg, top_k=2)

    # --- 🧠 长期记忆 ---
    user_memory_card = get_user_memory(user_id)
    memory_context = build_memory_context(user_memory_card)

    # --- 🧩 构建消息列表 ---
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
    if user_memory:
        if isinstance(user_memory, list):
            context = "\n".join(user_memory)
            # 增加一致性强制指令
            context += "\n\n【强制规则】请先阅读以上对话记录。如果你之前已经给过某个借口（如'被作业封印''泡面洒了''睡过头'），本轮必须沿用同一个借口，禁止在相邻几轮中编造不同的借口。如果你之前承诺过直播时间，不要更改。"
            label = "【最近对话记录】"
        else:
            context = f'我说："{user_memory}"'
            label = "【关于这个绿冻的上一轮记忆】"
        messages.append({
            "role": "system",
            "content": f"{label}\n{context}"
        })

    messages.append({"role": "user", "content": user_msg})

    # --- 🤖 调用大模型 ---
    try:
        response = await deepseek_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.85,
            max_tokens=150
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"哎呀，hzm脑子卡了一下……（错误: {e}）"

    if not reply:
        reply = "……（沉默，可能是信号不好）"

    # --- 💾 更新短期记忆（同时记录用户和 AI 的发言，保留最近 3 轮共 6 条） ---
    history = memory.get(user_id, [])
    if isinstance(history, str):
        history = [history] if history else []
    history.append(f"用户：{user_msg}")
    history.append(f"灰泽满：{reply}")
    if len(history) > 6:
        history = history[-6:]
    memory[user_id] = history
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    # --- 📝 异步更新长期记忆 ---
    async def update_memory_task():
        try:
            current_summary = json.dumps(user_memory_card, ensure_ascii=False) if user_memory_card else "无"
            prompt = MEMORY_EXTRACT_PROMPT.format(
                current_summary=current_summary,
                user_msg=user_msg,
                reply=reply
            )
            resp = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
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

    asyncio.create_task(update_memory_task())

    # --- 📤 回复 ---
    await chat.finish(Message(reply))