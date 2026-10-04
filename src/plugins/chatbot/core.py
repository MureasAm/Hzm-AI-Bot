"""消息处理主循环。

组装系统提示词（基础人设 + 人格规则 + 行为指令 + RAG + 长期/短期记忆），
调用 DeepSeek 生成回复，并异步更新长期记忆。
"""
import json
import os
import random
import asyncio
import re
import time

from nonebot import get_driver
from openai import AsyncOpenAI

from .constants import (
    SYSTEM_PROMPT_FILE, DEEPSEEK_BASE_URL, ZHIPU_BASE_URL,
    DEFAULT_MODEL, THINKING_DISABLED,
    CHAT_TEMPERATURE, CHAT_FREQUENCY_PENALTY, CHAT_MAX_TOKENS,
    MEMORY_EXTRACT_TEMPERATURE, MEMORY_EXTRACT_MAX_TOKENS,
)
from .persona import (
    load_persona_rules, build_global_persona_context, load_schedule,
)
from .memory import (
    get_user_history, append_user_history,
    get_user_memory, update_user_memory, build_memory_context,
    _format_profile_summary, MEMORY_EXTRACT_PROMPT,
    get_last_turn_gap_seconds, humanize_gap,
)
from .rag import embed_query
from .retrieval import (
    retrieve_corpus_candidates,
    load_phrase_groups, select_phrase_groups,
    retrieve_preferences, retrieve_core_stories, fuse_and_truncate, select_behavior_item,
)
from .corpus_judge import judge_corpus
from .constants import (
    PHRASE_PHASES_MAX, SPLIT_MIN_LEN, SPLIT_MAX_PARTS, SPLIT_MERGE_MIN_CHARS,
    SPLIT_DELAY_BASE_MS, SPLIT_DELAY_PER_CHAR_MS,
    SPLIT_DELAY_MIN_MS, SPLIT_DELAY_MAX_MS, SPLIT_DELAY_JITTER,
)
from . import context_probe
from . import group_memory
from .routing import (
    LEGENDARY_REPLIES, LEGENDARY_CONFIRMS, legendary_confirmed, legendary_hit, classify_l3,
    _repair_llm_json,
)
from .reply_style import (
    split_reply, split_delay, clean_reply, is_echo_reply, is_emotion_only_query,
    _trim_text,
)
from .session_memory import (
    probe_session, build_session_context, is_emoji_msg, previous_session_note,
)


# ==================== 🎭 基础人设提示词 ====================
if SYSTEM_PROMPT_FILE.exists():
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    raise FileNotFoundError(f"❌ 未找到 {SYSTEM_PROMPT_FILE}")

# ==================== 🛠️ API 客户端（惰性初始化） ====================
# 挪到了 config.py（基础设施层）。core 从这里 import，routing 也从 config 取，避免循环依赖。
from .config import _get_clients, _get_model_name, extract_chat_content  # noqa: E402


def _detach_behavior_from_rrf() -> bool:
    """**行为/措辞不进 RRF** —— `DETACH_BEHAVIOR=1` 打开。**缺省关 = 现状**（见文末"为什么没开"）。

    ## 这个开关改什么（用户 2026-09-30 提出）

    > "behaviors 有没有必要进入检索呢？被 LLM 判断中难道不是就可以直接读取文件内容？"

    对，那两路**已经不是检索了**：`select_behavior_item` / `select_phrase_groups` 干的是
    「按 L3 判出的名字**查一条**」——没有相似度、没有排名、分数写死 1.0。
    而 RRF 是给「检索」造的（把几路的**排名**融合成一份有序列表，再取 top-6 / 卡 1200 字预算）。

    把查表结果丢进排行榜的后果：**它占 top-6 名额**。实测挤掉率 0%，但那是靠
    `behavior` 权重 1.5 最高 + 名额够用这两个巧合撑着的；哪天名额紧或四路全命中，
    "该做的动作"会**静默**消失。

    ```
    关（现状）：fuse_and_truncate(corpus, sample, behavior, phrase)   ← 四路抢 6 个名额
    开        ：fuse_and_truncate(corpus, sample, [], []) + behavior + phrase   ← 截断后追加
    ```

    ## A/B 的结论（2026-09-30，`scripts/detach_behavior_ab.py`）

    - **不删任何东西**：行为/措辞两版都照注入，变的只是"6 个名额归谁"
    - 代价：注入量平均 **+6 字**（−10~+24，≈免费）
    - **15 轮真实 A/B（靶子情境），用户判"都差不多"** → 感知上无损失
    - 评测不变：retrieval-eval **47/48**、regression --check 跑通过 **8/8**

    ## ⚠️ 那为什么没开（缺省关）

    有个**指向"可能有害"的信号没查清**：

    ```
    整套 regression --check：新版 5 次里挂 2 次 ｜ 旧版 3 次里挂 0 次   （p≈0.5，不显著）
    单跑那条用例（no_fixed_opener_when_deflecting）：**两版都 5/5 过** → 复现不了
    ```

    可疑机制：新版让 **voice_samples 多进 1~2 条**（名额不再被行为/措辞占），
    样本多了模型**可能更容易盯住某一条的开头** → 反而更同质。而那条用例判的正是
    "反复求交往时不能每轮同一个词开头"。

    **收益不足以冒这个险**：它**不修任何已知故障**，只是结构更正确（现状实测零代价）。
    → 按"信号未清就不动"处理。**查清了再开**（见 `待办清单.md` §0.9）。
    """
    return os.environ.get("DETACH_BEHAVIOR", "0") == "1"


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
        return extract_chat_content(resp)
    except Exception as e:
        print(f"⚠️ 批量归纳失败（忽略）: {e}")
        return ""


# 转发摘要的输入上限：转发动辄几十条、几千字，先截再喂，防把上下文挤爆
FORWARD_FOR_SUMMARY_CHARS = 4000


async def summarize_forward(raw: str) -> str:
    """把别人转发的聊天记录压成一两句，供她理解后自然回应。

    **为什么不让模型直接读原文**：转发记录动辄几十条、几千字，全塞进上下文会挤爆
    （而且她要的是"知道这帮人聊了啥"，不是逐条读）。所以先摘要、只注入摘要。

    失败返回空串（调用方降级成"内容较长没细看"，原文不入上下文）。
    """
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        deepseek_client, _ = _get_clients()
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content":
                       "下面是一段 QQ 群里转发的聊天记录（多人对话）。用**一两句话**概括他们聊了什么，"
                       "供别人理解后自然接话。\n"
                       "要求：只概括不评价、不加戏；说清'谁在跟谁聊什么、大致什么氛围'；不要逐条复述。\n\n"
                       f"聊天记录：\n{text[:FORWARD_FOR_SUMMARY_CHARS]}\n\n"
                       "只输出摘要本身，不要任何前缀。"}],
            temperature=0.3,
            max_tokens=120,
            **THINKING_DISABLED,
        )
        return extract_chat_content(resp).strip()
    except Exception as e:
        print(f"⚠️ 转发摘要失败（忽略）: {e}")
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


def _qq_face_of(msg: str) -> str:
    """`[表情：可怜]` → `可怜`；不是纯 QQ 表情返回空串。

    QQ 内置表情的含义**就是它的名字**，不需要模型去"理解"——见 handle_chat 里的用法。
    """
    t = (msg or "").strip()
    if t.startswith("[表情：") and t.endswith("]"):
        return t[len("[表情："):-1].strip()
    return ""


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
        content = extract_chat_content(resp)
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
                       query_hint: str = "", denied_terms: set | None = None,
                       group_context: str = "", history_gap_note: str = "",
                       prev_session_note: str = "", same_request: dict = None) -> list:
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

    # 周表（地面真值）：被问"明天来吗/这周/几点播"以它为准。
    # 记忆里带"明天/下周"的话是过去某场直播当时的说法，可能早过期，不能当现在的安排。
    # 这里**只有 weekly**（weekday 制，到周自动对，永不过期）——曾有的「近况」字段已整条删除，
    # 它治不了"用旧记忆回答当下"（漏点在 voice_samples），且手动维护必烂。见 constants.py 的说明。
    sched = load_schedule()
    weekly = sched.get("weekly") if sched else None
    if weekly:
        lines = "、".join(f"{x.get('day')} {x.get('time')}" for x in weekly if x.get('day'))
        messages.append({
            "role": "system",
            "content": f"【灰泽满的周表】她的固定直播安排：{lines}。"
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
                "content": f"【她的核心记忆】{story_text}（这是她过去最深刻的经历，粉丝常拿这些开玩笑。被问你的书/事迹时，大方承认、可提议念给TA听或说个片段概括，别一口气把原文/全文念出来——除非对方明确说'念一下'）",
            })

    # 群聊现场（轻量群记忆注入）：在场成员 + 群内近况。群 = 好几个不同的绿冻在场，
    # 回给刚才直接点名搭话的那位，别把不同的人当成同一个"你"。
    if group_context:
        messages.append({
            "role": "system",
            "content": f"【群聊现场】这是群聊。{group_context}",
        })

    # 名词库（terms/lorebook）：核心词 always 注入 + 命中用户消息的词注入——让模型懂"绿冻/枪神8/slg"这类词并带对的态度
    terms_note = build_terms_note(user_msg, denied_terms=denied_terms)
    if terms_note:
        messages.append({
            "role": "system",
            "content": f"【灰泽满的世界】{terms_note}（其中『用词规则』是说话习惯，回复里表达对应概念时主动用她的黑话称呼）",
        })

    # 会话级记忆（当前话题 + 本场事件）：让模型接得住会话调性
    if session_context:
        messages.append({
            "role": "system",
            # A 类包装语已删（2026-09-30）：原来尾巴还有「（这是你们这一场对话的调性和发生过的事，
            # 回应时要自然地顺着这个语境，不要生硬提及）」——§3.5 实测"管怎么用"零作用。
            # ⚠️ 下面那条 `prev_session_note` 的注释说的"那句写的是'你们这一场对话'"指的就是被删的这句，
            #    删掉它**不影响**那个设计（它防的是"把几天前的东西塞进当前会话"，与包装语无关）。
            "content": f"【当前会话】{session_context}",
        })

    # 上一场会话（已过期，带"隔了多久"）：**不能并进上面那块**——
    # 那句写的是"你们这一场对话"，把几天前的东西塞进去她会当成现在。
    # 单独一条、且自己带着时间定性（见 session_memory.previous_session_note）。
    if prev_session_note:
        messages.append({"role": "system", "content": prev_session_note})

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
                # A 类包装语已删（2026-09-30）：原来是「请严格按此模式回应：」。
                # ⚠️ 同句式有**已知有害**的先例：它曾把 behaviors 里的 `"'呃…'起头"` 变成 10/10 照抄
                #    （而 86 条真实素材里 0 条那样开头）。那批措辞规定已经删了，**这句强制语一直留着**。
                "content": f"【当前情境下的行为指令】\n{behavior_text}"
            })

    # 直播记忆（source=corpus）：只当"背景记忆"，不参与风格示范
    if corpus:
        context = "\n".join(f"- {it.text}" for it in corpus if it.text)
        if context:
            messages.append({
                "role": "system",
                "content": f"【她经历过的相关背景】以下是她过去直播里经历过的事（背景记忆，都是曾经发生的，不是现在）。"
                           # ── A 类包装语已删（2026-09-30，§3.5 实测"管怎么用"零作用）──
                           # 删掉的是：「只当'她记得的经历'自然带出…不整段复述、不模仿里面的叙述口吻。」
                           # 留下的是下面这句「别拿背景记忆编当下的因果」——它管的是**事实/时间**，
                           # 对应真实故障（「搬家」被当现在用，差 80 天），不是"怎么用"。
                           f"**别拿背景记忆编当下的因果**——"
                           f"① 她一贯的毛病（迟到/睡过头/拖延/临时鸽/熬夜）是她的常态：被问'怎么又迟到/又鸽/为什么迟到'这类时，"
                           f"直接认领常态就好，嘴硬自洽地接，"
                           f"不需要也编不出'这一次'的具体原因，别硬解释；"
                           f"② 某次具体的旧记忆（那次和谁连麦、那次赶作业到半夜）只当讲古素材，**绝不能拿来当这次迟到/鸽的理由去编因果**；"
                           f"若确实在说当下且感知没给原因，就大方说不知道/打哈哈，别从旧事现编一个。"
                           f"\n{context}"
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
            if history_gap_note:
                # 只加一行汇总，不给每行打时间戳——要的是"知道隔了多久"这个概念，
                # 不是让她复述"3天前"。同一场对话内 humanize_gap 返回空串，不注入。
                context = f"（距离上一轮对话已经过去{history_gap_note}了）\n" + context
            # 一致性（只管事实）+ 防复读（别逐字复读）。
            # 旧的【一致性规则】写的是"借口要与之前保持一致"——那是**直接教它复读**：
            # 一旦第一次用了某个借口（"威严""搬家"），规则就把它锁死，错误只固化不自我纠正
            # （实测"威严"和"搬家"各被锁了一轮又一轮）。改成**只约束事实别打架**，明确不管说法与态度。
            # 旧【防复读】的豁免条件是"用户没有主动追问时"——而用户反复追问正是复读高发场景，
            # 等于在最需要它的地方自动失效。改成"可以正面回答，但换个说法，别逐字搬上轮那句"。
            context += (
                "\n\n【别自相矛盾】同一件事的**事实**别前后打架——说过的原因、安排、答应过的事"
                "（如'今天为什么没播''这周播不播'），下一轮别改口成另一套。\n"
                "这条只管**事实**，不管**说法和态度**：被反复追问同一件事时，换个说法、换个角度、"
                "或者干脆松口/推进，都比把上轮那句原样再说一遍自然。**宁可换个方式说同一件事，也别逐字复读。**\n"
                "【防复读】以上对话中，灰泽满自己说过的话只是历史背景。用户重复问同一件事时，"
                "她可以正面回答，但别把上轮那句原样搬出来——顺着他当前的说法自然接，"
                "也别每轮都重提自己之前提过的事。"
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
                # A 类包装语已删（2026-09-30）：原来是「表达同类意思时用这些原话组织，不要自创解释性措辞：」。
                # 而且 phrases 本来就是**碎片**（实测 0/16 被抄）——这条约束在管一件不会发生的事。
                "content": "【她的固定说法】以下情景她说这些话：\n" + "\n".join(phrase_blocks)
            })

    # ⛔【灰泽满的说话方式参考】整段已删（2026-10-04）——voice_samples 通道取消。
    #   原因（实测，n=32）：
    #     ① 它按【话题】检索，必然捞到"同话题的完整回答" → 把**事实**灌进对话：
    #        "你今天吃什么了"→捞到"今天怎么没吃饭"→她答"还没吃呢"→下一轮自相矛盾；
    #        "外面下雨了"→她答"刚淋着跑回来的"（编的）；"最近怎么样"→"要写出百年孤独了"（近乎逐字搬）
    #     ② assistant 通道注入时，样本和真实历史混在一起（样本 user 与真实 user 相邻，
    #        模型读成"连着两轮用户发言"）→ **与【当前时间】"正在直播中"冲突 13/32**（system 一行只 2/32），
    #        样本词泄漏 4/32（system 一行 1/32），反问率 38%（system 一行 50%）
    #   这批原句里**该留的 18 条已并入 behaviors 的 samples**（按情景触发，不按话题乱捞）。

    # ⛔【回复节奏】已删（2026-09-30）——C 类：与骨架【说话节奏】重复
    #   （骨架写的是"默认短句，一句一个想法，说清楚就停"），纯浪费预算。
    #   短句仍由骨架软引导 + `split_reply` 分段 + 语音兜底，**没有丢机制**。

    # 连续索要（客观事实，不含指令）：她需要知道"这已经是第几次了"。
    # 为什么必须有（2026-10-04 从真实记录发现）：上下文只带最近 10 条，
    # 跨轮的连续索要她完全看不见 → 每轮都当成"第一次被要" → 换着理由挡 5 轮后突然松口，
    # 松口之后又失去"我从没答应过"的立场（3076669330 要"宝宝"/要语音两段都是这样）。
    # ⚠️ 这里**只给事实**（第几轮、之前在要什么），**不给"该不该答应"**——
    #    "怎么应对"是 behaviors 的事；而且实测"管怎么用"的说明无效、事实类信息有效。
    if same_request and int(same_request.get("count") or 1) >= 2:
        what = same_request.get("what") or "同一件事"
        messages.append({
            "role": "system",
            "content": f"【客观情况】用户已经连续 {same_request['count']} 轮在向灰泽满要{what}了"
                       f"（换着说法也算同一件事）。前面 {same_request['count'] - 1} 轮都没有答应。",
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
            # 这里**只做"别读错情绪"的正确性约束，不规定她该怎么反应**——
            # 「什么情绪 → 她怎么回应」属于数据层，该由 behaviors 的条目 + 真人样本承担。
            # （我曾在这里写过"委屈/可怜 → 她该放软、别嘴硬顶着"，那是提示词补丁，已回退：
            #   同一个会话里我刚把 😅 那行的"规定动作"删掉，转手又加一条，方向自相矛盾。
            #   见 待办清单.md / 交接文档：纯表情消息目前会跳过 L3，所以数据层压根没参与。）
            emoji_hint = (
                f"【用户发了表情】{query_hint}\n"
                "用户只发了一个表情，没有任何文字。**以这个表情本身的情绪为准**，别读成别的情绪"
                "（尤其别因为她上一句在嘴硬，就把这个表情读成'他在怼我/他对我无语'）。\n"
                "按这个情绪自然回应，**具体怎么反应以注入的【行为指令】与她的样本为准，这里不规定**。\n"
                "不要复述表情，不要顺着当前话题硬接，只回应这个情绪。"
            )
            messages.append({"role": "system", "content": emoji_hint})
        else:
            messages.append({
                "role": "system",
                "content": f"【用户这条消息的语境】{query_hint}\n（上面是这条消息在当前语境下的完整意思——短消息或指代性消息（如'能读给我听听吗'）需要结合前文才能理解，按这个理解回复）"
            })

    messages.append({"role": "user", "content": final_user})
    return messages


# 生成失败时的兜底文案（**只发给用户，绝不进记忆**——见 is_fallback_reply）
_FALLBACK_REPLY = "哎呀，hzm脑子卡了一下……"
_FALLBACK_SILENT = "……（沉默，可能是信号不好）"


def is_fallback_reply(text: str) -> bool:
    """这句是不是"没生成出来"的兜底文案。

    用途：兜底文案不该写进记忆。踩坑：上游偶发 `choices: null` 那一轮，
    兜底文案被当成"她的回复"存进了短期记忆，于是她的历史里永久留着一句
    `哎呀，hzm脑子卡了一下……（错误: ...）`——**AI 腔的故障信息变成了"她说过的话"，
    还会作为 few-shot 喂回去**。一次上游抖动 = 一份永久污染。
    """
    return (text or "").strip() in (_FALLBACK_REPLY, _FALLBACK_SILENT)


async def generate_reply(messages: list) -> str:
    """调用 DeepSeek 生成回复；失败时返回兜底文案（调用方负责"别把它记进记忆"）。"""
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
        reply = extract_chat_content(response)
    except Exception as e:
        # 异常原文不再塞进回复：那是内部信息，会原样发给 QQ 上的每个用户。
        # 打到控制台即可，她这边只回一句人话。
        print(f"⚠️ 生成失败: {e}")
        reply = _FALLBACK_REPLY
    return reply if reply else _FALLBACK_SILENT


def _parse_memory_extract(content: str) -> dict:
    """把记忆提取 LLM 的输出解析为 dict：剥围栏 + 修复不规范 JSON。

    内容为 "null" 返回 {}；修复后仍不是合法 JSON 则抛异常（调用方重试一次）。

    `_repair_llm_json` 已挪到 routing.py（L3 也要用，且 core 反向 import 会循环）。
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
        content = extract_chat_content(resp)
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
            content2 = extract_chat_content(resp2)
            print(f"[长期记忆] 重试提取: {content2}")
            updates = _parse_memory_extract(content2)
            if updates:
                update_user_memory(user_id, updates)
        except Exception as e2:
            print(f"[长期记忆] 重试仍失败，本轮记忆丢弃。首次原始输出: {content!r}")
            import traceback
            traceback.print_exc()


async def gather_retrieval(query_text: str, retrieval_query: str, history_text: str,
                           deepseek_client, zhipu_client,
                           is_user_msg: bool = True) -> dict:
    """规范取素材：一段文本该配的东西一次捞齐（人设 + 行为/措辞 + 直播记忆 + 风格样本 + 偏好 + 核心记忆）。

    **聊天和主动发言共用这一份**。抽出来的原因：主动发言原来是自己在
    `proactive.py` 里另拼一套（system_prompt + 自己写的规则），**素材层整层没接**，
    输出就成了"干说、没有生活感"。两处各拼一套必然漂，所以收成一个入口
    （周表/感知/名词库那些"每轮都注入"的层在 build_message_list 里，这里管"按内容取的"）。

    `is_user_msg`：要取素材的这段文本，**是不是"用户说的话"**？
      · True（聊天，默认）：该走的都走——L3 按"用户这句想干嘛"判行为/措辞；
        corpus 交 LLM 判"用户是不是在问她这段"。
      · False（主动发言）：她发的动态/微博**不是用户说的话**，上面两步的前提不成立
        （拿它去问"用户是不是在问她这段"，答案天然是否），所以跳过；其余按**话题**取的
        路（风格样本/偏好/核心记忆）照旧，并按"她自己的陈述句"用不设阈值的取法。

    ⚠️ 后续若要给主动发言也接直播记忆，得另写一个判据（"这条内容和她的哪段经历有关"），
    不能直接复用 corpus_judge 的问法。
    """
    traits, styles, behaviors = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)
    result = {"global_persona": global_persona, "fused_items": [],
              "preference_items": [], "core_stories": [], "same_request": None}

    # 纯图片消息（无文字）不做检索：让灰泽满直接评价图片，避免语料/行为劫持图片内容
    # 纯表情消息（emoji/[表情：xx]）也不做语义检索：表情只表达情绪不表达话题，
    # 扩充句会作为语气提示注入，但检索 memory 会跑偏（如😭命中"被夸"样本）
    if not (query_text and not is_emoji_msg(query_text)):
        return result
    # 纯情绪消息（如'可惜🤭'，被 probe 补全成'用户发了个偷笑的表情'）无话题词，
    # 语义检索会误命中无关样本（实测→peer_5'新衣服'）。对齐纯表情设计：跳过检索，
    # 补全句只作语气提示（query_hint）注入。
    if is_emotion_only_query(retrieval_query):
        return result

    try:
        await _fill_retrieval(result, query_text, retrieval_query, history_text,
                              deepseek_client, zhipu_client, behaviors, is_user_msg)
    except Exception as e:
        # **检索层炸了不能让她变哑巴**：素材按"没检索到"处理，回复/主动发言继续。
        # 踩坑（2026-09-27）：智谱 embedding key 过期 → retrieve_voice_samples 抛
        # TypeError('NoneType' object is not iterable)，而调用点没有兜底 →
        # **整条回复任务死掉，她一个字都不回**（含群聊，看起来像"不接话"）。
        # 兜底放这里=两条链路（聊天/主动发言）一起受保护。
        print(f"⚠️ 检索层异常（按'没检索到'继续）: {e}")
    return result


async def _fill_retrieval(result: dict, query_text: str, retrieval_query: str, history_text: str,
                          deepseek_client, zhipu_client, behaviors, is_user_msg: bool) -> None:
    """`gather_retrieval` 的干活部分（单独一层，方便上面统一兜底）。结果写进 result。"""
    behavior_items, phrase_items, corpus_items = [], [], []
    # L3：归属用 LLM 判意图（不再用 embedding 猜——embedding 按句式聚团，
    # 会把'灰泽满你唱歌好听'（夸）和'灰泽满你怎么又迟到'（质问）挤在一起误判）。
    # **一次调用同时判"行为"（该怎么做）和"措辞"（该用哪些词）**——两者是同一个问题的两半。
    # 判别词（敷衍/骗/鸽/迟到/黄桃/擦边…）退**兜底**位：L3 判不出来时才用
    # （这才是它 docstring 里写的定位；以前被当成"省一次调用"的短路，
    #   但实测 138 条真实消息里它 0 次命中，而短路会顺手跳过措辞分类）。
    if is_user_msg:
        phrase_groups = load_phrase_groups()
        l3 = await classify_l3(deepseek_client, query_text, history_text, behaviors, phrase_groups)
        if l3["behavior"]:
            print(f"[行为] LLM 判定: {l3['behavior']}")
        if l3["phrases"]:
            print(f"[措辞] LLM 判定: {'、'.join(l3['phrases'])}")
        sr = l3.get("same_request") or {}
        if int(sr.get("count") or 1) >= 2:
            result["same_request"] = sr
            print(f"[连续索要] 第 {sr['count']} 轮：{sr.get('what') or '（未说明）'}")
        behavior_item = select_behavior_item(query_text, l3["behavior"], behaviors)
        behavior_items = [behavior_item] if behavior_item else []
        phrase_items = select_phrase_groups(l3["phrases"], phrase_groups)

    query_vector = await embed_query(zhipu_client, retrieval_query or query_text)
    if is_user_msg:
        # corpus：**全部交 LLM 判**。门/钩子只决定"把哪些条递过去看"，**不再有直通**。
        # 为什么取消直通（2026-09-29）：实测词重叠这把尺子分不开"说的是谁"——
        #   误报 n7「宝宝我明天要早起我先去睡了」（用户说**自己**）对 #271：ov=0.40
        #   真阳性 co9「你不是以前有喜欢的男学霸吗」（问**她**）对 #42：ov=0.143
        #   **误报的重叠度反而更高** → 阈值只是在拿一个换另一个（试过 0.5，co9/co10 一起红）。
        # 判不出来一律不带（宁可漏不可错，见 corpus_judge 模块头）。
        candidates = retrieve_corpus_candidates(retrieval_query or query_text, query_vector)
        corpus_items = await judge_corpus(deepseek_client, query_text, candidates)

    # 声音样本那一路已删（2026-10-04）：sample_items 恒为空，见 retrieval.py 顶部说明。
    sample_items: list = []

    if _detach_behavior_from_rrf():
        # 行为/措辞**不进 RRF**：它们是 L3 判出类别后**按名字查表**的结果（没有相似度、
        # 分数写死 1.0），不是"从一堆里挑最像的"那种检索。RRF 是给检索造的（融合多路排名），
        # 把查表结果丢进去，它会**占 top-6 名额和 1200 字预算**。
        # → 截断**之后**再追加，永不参与名额竞争；名额全还给 corpus + voice_sample。
        # 注入位置由 `_split_fused` 按段类型决定，**不受这里顺序影响**（见 build_message_list）。
        # ⚠️ `fuse_and_truncate` 的签名是 (corpus, sample, behavior, phrase)，
        # 四个都是位置参数（`behavior_items` 没有默认值）——所以要显式传空表，
        # 不能只传前两个（踩过：少传参数会让**整层检索抛异常**，
        # 然后被 gather_retrieval 的兜底"按没检索到继续"吞掉，A/B 结果全废）。
        result["fused_items"] = (fuse_and_truncate(corpus_items, sample_items, [], [])
                                 + behavior_items + phrase_items)
    else:
        result["fused_items"] = fuse_and_truncate(corpus_items, sample_items,
                                                  behavior_items, phrase_items)
    # 第 5 路：偏好（关键词命中）／核心记忆（结晶）
    result["preference_items"] = retrieve_preferences(retrieval_query or query_text)
    result["core_stories"] = retrieve_core_stories(retrieval_query or query_text, query_vector)


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
    # 距上一轮多久（此刻还没 append 本轮，最后一条就是上一轮）。同一场对话内为空串
    _gap_sec = get_last_turn_gap_seconds(user_id)
    history_gap_note = humanize_gap(_gap_sec) if _gap_sec is not None else ""
    # 上一场会话（已过期）→「上次聊过」提示。**必须在 probe_session 之前取**——
    # probe 会把 last_active 改写成"现在"，之后再问就永远判成"没过期"。
    _prev_session_note = previous_session_note(user_id)
    retrieval_query = query_text
    if query_text:
        retrieval_query = await probe_session(user_id, query_text, history_text, deepseek_client)
        if retrieval_query != query_text:
            print(f"[会话记忆] 短 query 扩充: 「{query_text}」→「{retrieval_query}」")
    # 命中已知术语：probe 的短 query 扩充可能猜错含义（如"富区"→"富拉尔基区"），
    # 术语自己已定义含义，回退用原文，避免错误扩充污染检索与语境提示
    if query_text and _hits_on_demand_term(query_text):
        retrieval_query = query_text
    # 纯 QQ 表情（[表情：可怜]）：**含义就在名字里，别让 LLM 再猜**。
    # 踩坑：probe 会拿表情去"理解"，带上下文时容易猜偏（实测把它读成"无语"，
    # 于是回复变成"无语了？灰泽满又没说错什么"——而用户发的是"可怜"）。
    # 直接在代码层还原名字，确定性、且省一次 LLM 猜。
    _face = _qq_face_of(query_text)
    if _face:
        retrieval_query = f"用户发了一个「{_face}」的QQ表情"
        print(f"[表情] 按字面含义注入（不猜）: {retrieval_query}")
    # 话题/事件已在本轮探测中更新，取最新会话状态
    session_context = build_session_context(user_id)

    # --- 🃏 经典梗硬匹配（双路由：关键词粗筛 + LLM 语境确认，防误触发） ---
    _confirm_history = ""
    for trigger, entry in LEGENDARY_REPLIES.items():
        if legendary_hit(trigger, user_msg):
            replies = entry.get("replies") or []
            if not replies:
                continue
            confirm_tpl = LEGENDARY_CONFIRMS.get(trigger)
            if confirm_tpl:
                if not _confirm_history:
                    _confirm_history = "\n".join(get_user_history(user_id)[-4:])
                if not await legendary_confirmed(user_msg, confirm_tpl, history=_confirm_history):
                    print(f"[梗] 关键词 {trigger!r} 命中但 LLM 未确认，继续查后续触发词")
                    continue  # 只跳过这一条；同句还可能命中别的更明确梗，别一并放弃
            reply = random.choice(replies)
            # 梗匹配也记入短期记忆 + 异步长期记忆，避免后续对话"失忆"
            append_user_history(user_id, user_msg, reply)
            if not is_group:  # 群会话不建用户记忆卡
                card = get_user_memory(user_id)
                asyncio.create_task(update_memory_task(user_id, user_msg, reply, card))
            return reply

    # --- 🎭 人格规则 + 🔍 检索融合（query 只算 1 次 embedding）---
    # 素材统一从 gather_retrieval 取（主动发言也走同一个入口，别再各拼一套）
    ctx = await gather_retrieval(query_text, retrieval_query, history_text,
                                 deepseek_client, zhipu_client)
    global_persona = ctx["global_persona"]
    fused_items = ctx["fused_items"]
    preference_items = ctx["preference_items"]
    core_stories = ctx["core_stories"]

    # --- 🧠 确定性两路记忆 ---
    user_memory_card = {} if is_group else get_user_memory(user_id)
    memory_context = "" if is_group else build_memory_context(user_memory_card)
    # 群聊现场事实：在场成员 + 群内近况（轻量群记忆，只在本群注入）
    group_context = ""
    if is_group:
        gm = group_memory.get_group(user_id)   # user_id 此时 = 群号
        members = [n for n in (gm.get("members") or {}).values() if n]
        names = "、".join(members[-12:]) or "（还没太熟的几个绿冻）"
        # 带相对时间注入：群近况事件存了 t，以前注入时丢掉 → 上周的事被当"现在"。
        # 和长期记忆 last_seen 是同一个坑（存了时间戳、注入时丢）。
        events = []
        for e in (gm.get("events") or [])[:3]:
            txt = (e.get("text") or "").strip()
            if not txt:
                continue
            gap = humanize_gap(time.time() - float(e.get("t") or 0)) if e.get("t") else ""
            events.append(f"{txt}（{gap}前）" if gap else txt)
        evtxt = "；".join(events) or "（刚进群，还没太熟）"
        group_context = f"在场成员：{names}。群内近况：{evtxt}"
    weather_city = (user_memory_card or {}).get("weather_city", "") or ""

    # --- 🧩 构建消息列表 ---
    # 天气预热：组消息前异步把城市 LocationID+天气取进缓存，build 时同步读缓存零阻塞
    await context_probe.warm_weather(weather_city)
    # query_hint：短消息（≤4字）的语境扩充，仅当扩充句与原文不同时传入，帮模型理解短句
    query_hint = retrieval_query if (retrieval_query and retrieval_query != query_text) else ""
    # 术语语境确认：confirm:true 的命中做一次便宜 LLM 判断，剔除误触词条（词→意思守卫）
    denied_terms = await confirm_ambiguous_terms(user_msg, deepseek_client)
    messages = build_message_list(
        user_msg, global_persona, fused_items, memory_context, user_history,
        vision_desc=vision_desc, weather_city=weather_city, batch_summary=batch_summary,
        preference_items=preference_items, core_stories=core_stories,
        session_context=session_context, query_hint=query_hint,
        denied_terms=denied_terms, group_context=group_context,
        history_gap_note=history_gap_note, prev_session_note=_prev_session_note,
        same_request=ctx.get("same_request"),
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
                       f"认怂答应去做、自嘲一句、或者直接点破对方又在闹。重新回复这条消息。",
        }
        for _ in range(3):
            reply = await generate_reply(list(messages) + [nudge])
            if not is_echo_reply(reply, recent_bot, window=8):
                break

    # （这里曾有"防措辞固化"：find_repeat_word 检测到反复用同一个词就强制换措辞重生成。
    #  2026-09-12 删除——实测它 75% 的触发是在拦她自己的自称"灰泽满"，
    #  详见 reply_style.py 顶部注释。）

    # --- 💾 更新短期记忆（带锁）：图片消息把视觉描述记进去，后续才记得聊过什么图 ---
    # 存**清洗后**的版本（clean_reply 平时在 chat_window 里、本函数返回之后才跑）：
    # 否则她带换行/多括号的原始输出会存进记忆，再作为"她自己怎么说话"的 few-shot 喂回去，
    # 变成自我强化回路——后处理每次都得再洗一遍，坏习惯却一直被喂养。
    record_msg = _compose_record_msg(user_msg, vision_desc)
    if is_fallback_reply(reply):
        # 这一轮压根没生成出来（上游异常），**整轮都不记**：
        # 记了就等于让她"记得自己说过"一句故障文案，还会当 few-shot 喂回去。
        # 用户的这句话也随之不留——但她这轮并没有真的回应过，不记比记错干净。
        print("[记忆] 本轮是兜底回复（生成失败），不写入短期/长期记忆")
        return reply
    append_user_history(user_id, record_msg, clean_reply(reply))

    # --- 📝 异步更新长期记忆（会话级记忆已在对话前 probe_session 同步更新） ---
    if not is_group:  # 群会话不建用户记忆卡
        asyncio.create_task(update_memory_task(user_id, record_msg, reply, user_memory_card))

    return reply