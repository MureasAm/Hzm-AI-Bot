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
from .persona import load_persona_rules, build_global_persona_context, load_schedule
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
from .routing import (
    LEGENDARY_REPLIES, LEGENDARY_CONFIRMS, legendary_confirmed, classify_behavior,
)
from .reply_style import (
    split_reply, split_delay, clean_reply, is_echo_reply, is_emotion_only_query, _trim_text,
)
from .session_memory import (
    get_session, probe_session, build_session_context, is_emoji_msg,
)


# ==================== 🎭 基础人设提示词 ====================
if SYSTEM_PROMPT_FILE.exists():
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    raise FileNotFoundError(f"❌ 未找到 {SYSTEM_PROMPT_FILE}")

# ==================== 🛠️ API 客户端（惰性初始化） ====================
# 挪到了 config.py（基础设施层）。core 从这里 import，routing 也从 config 取，避免循环依赖。
from .config import _get_clients, _get_model_name  # noqa: E402


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
# load_terms 挪到了 persona.py（人格数据加载的归属地），reply_style / build_terms_note 都从 persona 取。
from .persona import load_terms  # noqa: E402


def build_terms_note(user_msg: str, denied_terms: set | None = None) -> str:
    """根据用户消息命中名词库：核心词(always)每次注入 + 命中词(关键词/别名/正则)注入。

    返回注入文本【灰泽满的世界】。客观打底 + 带灰泽满态度，让模型既懂词义又有正确的相处态度。

    命中检查：
    - priority=always：每次注入
    - keyword / aliases：子串命中
    - pattern（可选）：正则命中。用于子串匹配做不到的场景——如'区'单字撞'小区/地区'，
      用 `满区|(这么|太|好|很|真…)\\s*区` 只命中'满区'或形容词用法（'你怎么这么区'），
      不误触普通名词里的'区'。

    双向（v2）：
    - usage：该词的"主动用词规则"——回复里表达这个概念时用这个词（意思→词）。
    - usage_triggers：概念触发词。消息出现这些词（关键词没命中）时只注入 usage，
      让模型讨论这个概念时主动用黑话称呼，而不是只会听懂。
    usage 只在"关键词命中 或 概念词出现"时才注入（不常驻），避免"每条都蹦黑话"。

    denied_terms：LLM 语境确认后应剔除的词条 keyword 集合（见 confirm_ambiguous_terms）。
    """
    denied = denied_terms or set()
    terms = load_terms()
    if not terms:
        return ""
    msg = user_msg or ""
    notes = []
    for t in terms:
        kw = t.get("keyword", "")
        if not kw:
            continue
        if kw in denied:
            continue
        keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
        pattern = t.get("pattern")
        usage = t.get("usage")
        key_hit = any(k in msg for k in keys) or (pattern and re.search(pattern, msg))
        concept_hit = any(w in msg for w in (t.get("usage_triggers") or []))
        hit = t.get("priority") == "always" or key_hit
        if hit:
            parts = [t.get("meaning", "")]
            if t.get("reaction"):
                parts.append(f"被提到时：{t['reaction']}")
            if usage and (key_hit or concept_hit):
                parts.append(f"用词规则：{usage}")
            notes.append(f"{kw}：{'；'.join(parts)}")
        elif usage and concept_hit:
            # 概念命中：消息没提关键词，但提到概念词 → 只注入用词规则
            notes.append(f"用词规则：{usage}")
    return "；".join(notes) if notes else ""


def _hits_on_demand_term(msg: str) -> bool:
    """消息是否命中某个 on-demand 术语（关键词/别名/正则）。

    命中说明这是个已知黑话，短 query 扩充（probe）可能猜错含义
    （如"富区"被猜成"富拉尔基区"），应跳过扩充、用术语自己的定义。
    """
    for t in load_terms():
        kw = t.get("keyword", "")
        if not kw or t.get("priority") == "always":
            continue
        keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
        pattern = t.get("pattern")
        if any(k in msg for k in keys) or (pattern and re.search(pattern, msg)):
            return True
    return False


async def confirm_ambiguous_terms(user_msg: str, deepseek_client=None) -> set:
    """LLM 语境确认（词→意思的准确性守卫）。

    对 `confirm:true` 且本轮命中的术语，用一次便宜 LLM 判断"这个词在这个语境下是否真指它定义的含义"，
    返回应剔除的 keyword 集合（交给 build_terms_note 跳过）。防止 pattern/短词命中误触
    （如'满区'命中'怎么这么区'，可能是普通形容词用法而非黑话）。

    失败/无客户端默认放行（返回空集，不丢注入）——和 legendary_confirmed 同策略。
    """
    msg = (user_msg or "").strip()
    if not msg:
        return set()
    terms = load_terms()
    candidates = []
    for t in terms:
        kw = t.get("keyword", "")
        if not kw or not t.get("confirm"):
            continue
        keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
        pattern = t.get("pattern")
        if any(k in msg for k in keys) or (pattern and re.search(pattern, msg)):
            candidates.append(t)
    if not candidates:
        return set()
    if deepseek_client is None:
        try:
            deepseek_client, _ = _get_clients()
        except Exception:
            return set()
    lines = "\n".join(f"- {t['keyword']}：{t.get('meaning', '')[:80]}" for t in candidates)
    prompt = (
        "你是角色语境的判断器。下面是一批角色黑话/专名词条，用户消息命中了它们的关键词。\n"
        "判断：每个词条在**当前语境下**是否真的指它定义的含义。\n"
        "注意：发消息的人通常是粉丝/熟人，命中大多是黑话本义；**只有当语境明显指向其他意思时才剔除**。\n"
        "（例：'这个是好区'=好地段，不是'满区'粉丝黑话→剔除；'真的好区'=调侃'好菜/拉胯'，是黑话本义→不剔除。）\n\n"
        f"用户消息：{msg}\n\n词条：\n{lines}\n\n"
        '只输出 JSON：{"exclude": ["词条A"]}，exclude 只列**明显不是**该含义的词条；'
        '都适用输出 {"exclude": []}'
    )
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            **THINKING_DISABLED,
        )
        content = (resp.choices[0].message.content or "").strip()
        if "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        exclude = set(data.get("exclude", []))
        return {t["keyword"] for t in candidates if t["keyword"] in exclude}
    except Exception as e:
        print(f"⚠️ 术语语境确认失败（放行）: {e}")
        return set()


def build_message_list(user_msg: str, global_persona: str, fused_items: list,
                       memory_context: str, user_history: list,
                       vision_desc: str = "", weather_city: str = "",
                       batch_summary: str = "", preference_items: list = None,
                       core_stories: list = None, session_context: str = "",
                       query_hint: str = "", denied_terms: set | None = None) -> list:
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

    # 周表 + 近况（地面真值）：被问"明天来吗/这周/几点播"以它为准。
    # 记忆里带"明天/下周"的话是过去某场直播当时的说法，可能早过期，不能当现在的安排。
    sched = load_schedule()
    weekly = sched.get("weekly") if sched else None
    if weekly:
        lines = "、".join(f"{x.get('day')} {x.get('time')}" for x in weekly if x.get('day'))
        note = (sched.get("近况") or "").strip()
        note_txt = ""
        if note:
            note_txt = f"。近况：{note}（近况更新于 {sched.get('近况_updated', '?')}，过期就忽略）"
        messages.append({
            "role": "system",
            "content": f"【灰泽满的周表】她的固定直播安排：{lines}{note_txt}。"
                       f"被问'明天/这周/几点播/来不来直播'时，以这个周表为准回答（带她的嘴硬风格），"
                       f"不要拿直播记忆里过去某场的旧安排当现在的计划。",
        })

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
    terms_note = build_terms_note(user_msg, denied_terms=denied_terms)
    if terms_note:
        messages.append({
            "role": "system",
            "content": f"【灰泽满的世界】{terms_note}（这些是她世界的词，遇到时按定义理解并带着对应的态度，别当成普通的词；其中『用词规则』是说话习惯，回复里表达对应概念时主动用她的黑话称呼）",
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
                "content": f"【她经历过的相关背景】以下是她过去直播里经历过的事（背景记忆，都是曾经发生的，不是现在）。"
                           f"只当'她记得的经历'自然带出（聊到相关话题时'之前那次…'），不整段复述、不模仿里面的叙述口吻。"
                           f"**别拿背景记忆编当下的因果**——"
                           f"① 她一贯的毛病（迟到/睡过头/拖延/临时鸽/熬夜）是她的常态：被问'怎么又迟到/又鸽/为什么迟到'这类时，"
                           f"直接认领常态就好，嘴硬自洽地接（'灰泽满迟到还要理由？''老套理由：睡过头/网络''老毛病了别问'），"
                           f"不需要也编不出'这一次'的具体原因，别硬解释；"
                           f"② 某次具体的旧记忆（那次和谁连麦、那次赶作业到半夜）只当讲古素材，**绝不能拿来当这次迟到/鸽的理由去编因果**；"
                           f"若确实在说当下且感知没给原因，就大方说不知道/打哈哈，别从旧事现编一个。"
                           f"说话风格看下面的样本：\n{context}"
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
                      batch_summary: str = "", is_group: bool = False) -> str:
    """处理一条用户消息，返回机器人回复。vision_desc 为图片描述；batch_summary 为批量归纳。

    is_group=True 时按群会话处理：user_id 传群号（会话历史按群记），
    不建用户记忆卡（群不是单个用户），回复对象由调用方按群发送。
    """
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
    # 命中已知术语：probe 的短 query 扩充可能猜错含义（如"富区"→"富拉尔基区"），
    # 术语自己已定义含义，回退用原文，避免错误扩充污染检索与语境提示
    if query_text and _hits_on_demand_term(query_text):
        retrieval_query = query_text
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
                if not await legendary_confirmed(user_msg, confirm_tpl, history=_confirm_history):
                    print(f"[梗] 关键词 {trigger!r} 命中但 LLM 未确认，落到正常管线")
                    break  # 不是目标语境，放弃梗，走正常回复
            reply = random.choice(replies)
            # 梗匹配也记入短期记忆 + 异步长期记忆，避免后续对话"失忆"
            append_user_history(user_id, user_msg, reply)
            if not is_group:  # 群会话不建用户记忆卡
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
        if is_emotion_only_query(retrieval_query):
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
    user_memory_card = {} if is_group else get_user_memory(user_id)
    memory_context = "" if is_group else build_memory_context(user_memory_card)
    weather_city = (user_memory_card or {}).get("weather_city", "") or ""

    # --- 🧩 构建消息列表 ---
    # query_hint：短消息（≤4字）的语境扩充，仅当扩充句与原文不同时传入，帮模型理解短句
    query_hint = retrieval_query if (retrieval_query and retrieval_query != query_text) else ""
    # 术语语境确认：confirm:true 的命中做一次便宜 LLM 判断，剔除误触词条（词→意思守卫）
    denied_terms = await confirm_ambiguous_terms(user_msg, deepseek_client)
    messages = build_message_list(
        user_msg, global_persona, fused_items, memory_context, user_history,
        vision_desc=vision_desc, weather_city=weather_city, batch_summary=batch_summary,
        preference_items=preference_items, core_stories=core_stories,
        session_context=session_context, query_hint=query_hint,
        denied_terms=denied_terms,
    )

    # --- 🤖 调用大模型 ---
    reply = await generate_reply(messages)

    # --- 🔁 复读机防护 ---
    # 模型会对情境相关句执着复读：自己上轮的话进短期记忆后，用户把同一抱怨又说一遍时，
    # 它会整段照搬上上轮的解释（隔几句也一样，故窗口放宽到 8，不只最近 3）。
    # 检测到就强制"换动作"重生成：光"换个说法"不够，得 pivot——
    # 对方大概率在重复同一句/同一情绪，应认怂答应去做/自嘲/点破，而不是把解释再说一遍。
    recent_bot = [ln[4:] for ln in get_user_history(user_id) if ln.startswith("灰泽满：")]
    if is_echo_reply(reply, recent_bot, window=8):
        print(f"[防复读] 与最近自己说过的话重复『{reply[:20]}』，强制换说法")
        nudge = {
            "role": "system",
            "content": f"警告：你刚说过『{reply}』，几乎原样复读会很生硬。"
                       f"对方很可能在重复同一句/同一情绪。别再复读上轮的解释，换一个动作："
                       f"认怂答应去做、自嘲一句、或直接点破'你是不是在闹我'。重新回复这条消息。",
        }
        for _ in range(3):
            reply = await generate_reply(list(messages) + [nudge])
            if not is_echo_reply(reply, recent_bot, window=8):
                break

    # --- 💾 更新短期记忆（带锁）：图片消息把视觉描述记进去，后续才记得聊过什么图 ---
    record_msg = _compose_record_msg(user_msg, vision_desc)
    append_user_history(user_id, record_msg, reply)

    # --- 📝 异步更新长期记忆（会话级记忆已在对话前 probe_session 同步更新） ---
    if not is_group:  # 群会话不建用户记忆卡
        asyncio.create_task(update_memory_task(user_id, record_msg, reply, user_memory_card))

    return reply