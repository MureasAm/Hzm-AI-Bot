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
from dataclasses import replace

from .bg import spawn
from .constants import (
    SYSTEM_PROMPT_FILE, THINKING_DISABLED,
    CHAT_TEMPERATURE, CHAT_FREQUENCY_PENALTY, CHAT_MAX_TOKENS,
)
from .persona import (
    load_persona_rules, load_schedule, load_terms,
)
from .memory import (
    get_user_history, get_user_history_timed, get_user_history_entries,
    _strip_timed, _timed_prefix,
    append_user_history,
    get_last_turn_gap_seconds, humanize_gap, is_bot_line, bot_line_text,
)
from .rag import embed_query
from .retrieval import (
    retrieve_corpus_candidates,
    retrieve_preferences, retrieve_core_stories, fuse_and_truncate, select_behavior_item,
    resolve_associations,
)
from .corpus_judge import judge_corpus
from .chatlog import log_corpus_hits, log_block_hits, log_injection_trace, log_leak_guard
from .config import get_llm_route, get_flag
from .constants import SPLIT_DELAY_MIN_MS, SPLIT_DELAY_MAX_MS  # noqa: F401 — test_core 校验分段延迟上下限时读它
from . import context_probe
from . import group_memory
from . import injection as inj
from . import daily_memory
from . import event_memory
from . import memory_recall
from . import memory_v2
from . import self_speech
from . import song_requests
from . import songs
from . import need_triggers
from . import wiki_lookup
from .material_retrieval import retrieve_contextual_materials
from .retrieval_context import RetrievalContext
from .llm_json import parse_json_object
from .memory_context import filter_memory_hints
from .routing import (
    LEGENDARY_REPLIES, LEGENDARY_CONFIRMS, legendary_confirmed, legendary_hit, classify_l3,
)
from .reply_style import (
    clean_reply, is_echo_reply, is_emotion_only_query, uses_time_hook, starts_with_name,
    uses_hollow_opener, repeated_formula, is_meta_silence, planning_leak_reason,
    extract_action_tags, Reply,
)
from .session_memory import (
    probe_session, build_session_context, get_session, is_emoji_msg, previous_session_note,
    mood_note, actions_note, last_song, voice_preference_note,
)
from .term_hints import terms_hint_block, term_match


_MEMORY_V2_STORE = memory_v2.ShadowMemoryStore()


def injection_v3_enabled() -> bool:
    """注入 v3（registry/frame + 稳定卡/事件检索）。默认 0：旧链路逐字不变，便于回退。"""
    return get_flag("INJECTION_V3", "0").strip().lower() not in ("0", "false", "no", "off")


def history_as_assistant_enabled() -> bool:
    """私聊历史改成原生 user/assistant turns（2026-11-19 默认开）。

    只去掉「历史逐条转述」；引用改写、跨天日期本地化、群聊转述都还保留。
    回退：`.env.prod` 设 `HISTORY_AS_ASSISTANT=0`。
    """
    return get_flag("HISTORY_AS_ASSISTANT", "1").strip().lower() not in (
        "0", "false", "no", "off")


def wiki_lookup_enabled() -> bool:
    """未知专名去萌娘百科查一次。默认 0：先关着，验证后再开。"""
    return get_flag("WIKI_LOOKUP", "0").strip().lower() in ("1", "true", "yes", "on")


def _known_wiki_terms() -> set:
    """terms 已维护的词不再查 wiki——wiki 只补长尾，不抢名词库的活。

    只认 keyword，不认 aliases：alias 用来让词条命中（如「小松绿」→「后辈」），
    但 wiki 仍可补这个词的客观资料（社团/期数/出道时间），两者互补。
    """
    known: set = set()
    # 她自己不需要查百科；否则用户一句「小松绿和灰泽满」可能先查到灰泽满
    known.update({"灰泽满", "Hazel", "灰泽满Hazel"})
    for t in load_terms():
        kw = str(t.get("keyword") or "").strip()
        if kw:
            known.add(kw)
    return known


# ==================== 🎭 基础人设提示词 ====================
if SYSTEM_PROMPT_FILE.exists():
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    raise FileNotFoundError(f"❌ 未找到 {SYSTEM_PROMPT_FILE}")

# ==================== 🛠️ API 客户端（惰性初始化） ====================
# 挪到了 config.py（基础设施层）。core 从这里 import，routing 也从 config 取，避免循环依赖。
from .config import log_cache_usage, _get_clients, _get_model_name, extract_chat_content  # noqa: E402
from .config import _get_reply_clients, _get_reply_model  # noqa: E402  —— 主回复专用链（2026-11-14）
from .config import aux_clients_or, aux_model  # noqa: E402  —— 辅助链（2026-10-29）


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

    可疑机制：新版让 **corpus 多进 1~2 条**（top-6 名额不再被行为占），
    素材多了模型**可能更容易盯住某一条的开头** → 反而更同质。而那条用例判的正是
    "反复求交往时不能每轮同一个词开头"。

    **收益不足以冒这个险**：它**不修任何已知故障**，只是结构更正确（现状实测零代价）。
    → 按"信号未清就不动"处理。**查清了再开**（见 `docs/历史/待办清单.md` §0.9）。
    """
    return os.environ.get("DETACH_BEHAVIOR", "0") == "1"


async def summarize_batch(msgs: list) -> str:
    """把一批消息归纳成一两句话（说了什么 + 语气 + 意图），供模型理解整批。

    只有**≥2 条有文字**的消息才归纳（单条/只有图片不归纳）；只归纳原文已有信息，不编造；
    失败返回空串（调用方忽略，原文照旧）。
    """
    if sum(1 for t, _ in msgs if str(t or "").strip()) < 2:
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
        "消息：\n" + "\n".join(texts)
    )
    prompt += terms_hint_block("\n".join(texts), max_terms=2, max_chars=200)
    try:
        deepseek_client, _ = aux_clients_or(_get_clients)
        resp = await deepseek_client.chat.completions.create(
            model=aux_model(_get_model_name()),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "批量归纳")
        return extract_chat_content(resp)
    except Exception as e:
        print(f"⚠️ 批量归纳失败（忽略）: {e}")
        return ""


async def summarize_group_batch(items: list) -> str:
    """把一批群聊消息归纳成「对话线」：谁在对谁说什么、几条线、停在哪句话头上。

    `items` = [(昵称, 文本, 图片描述), ...]。和 `summarize_batch` 的区别是**带发言人**：
    群里有好几个人，只说“他们在聊什么”分不清谁在接谁；她要接在同一条线上，
    就必须知道最后一句是谁抛给谁的（2026-10-23，见 docs/历史/交接文档.md）。

    只有 **≥2 条有文字**的消息才归纳（单条/只有图片不归纳——本批本身就是那句）；
    失败返回空串（调用方不注入，原文照旧）。
    """
    if sum(1 for _, t, _ in items if str(t or "").strip()) < 2:
        return ""
    lines = []
    for name, text, desc in items:
        body = " ".join(
            part for part in (str(text or "").strip(), f"[图片：{desc}]" if desc else "") if part
        )
        if body:
            lines.append(f"{name}：{body}")
    if not lines:
        return ""
    prompt = (
        "下面是群里刚聊的一批消息。用 1–3 句话说明：谁在对谁说什么、这批有几条对话线、"
        "最后停在哪句话头上。\n"
        "要求：只归纳原文已有的信息，不要编造、不要评价；别逐条复述；"
        "如果大家各说各的、没有接同一条线，就直接说没有。\n\n"
        "消息：\n" + "\n".join(lines)
    )
    prompt += terms_hint_block("\n".join(lines), max_terms=2, max_chars=200)
    try:
        deepseek_client, _ = aux_clients_or(_get_clients)
        resp = await deepseek_client.chat.completions.create(
            model=aux_model(_get_model_name()),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=240,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "群聊归纳")
        return extract_chat_content(resp)
    except Exception as e:
        print(f"⚠️ 群聊对话线归纳失败（忽略）: {e}")
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
        deepseek_client, _ = aux_clients_or(_get_clients)
        resp = await deepseek_client.chat.completions.create(
            model=aux_model(_get_model_name()),
            messages=[{"role": "user", "content":
                       "下面是一段 QQ 群里转发的聊天记录（多人对话）。用**一两句话**概括他们聊了什么，"
                       "供别人理解后自然接话。\n"
                       "要求：只概括不评价、不加戏；说清'谁在跟谁聊什么、大致什么氛围'；不要逐条复述。\n\n"
                       f"{terms_hint_block(text, max_terms=2, max_chars=200)}\n"
                       f"聊天记录：\n{text[:FORWARD_FOR_SUMMARY_CHARS]}\n\n"
                       "只输出摘要本身，不要任何前缀。"}],
            temperature=0.3,
            max_tokens=240,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "转发归纳")
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
    """把融合结果按源分组：behavior / corpus。"""
    behaviors, corpus = [], []
    for it in fused_items:
        if it.source == "behavior":
            behaviors.append(it)
        elif it.source == "corpus":
            corpus.append(it)
    return behaviors, corpus


# ==================== 名词库（terms/lorebook） ====================
# load_terms 挪到了 persona.py（人格数据加载的归属地），reply_style / build_terms_note 都从 persona 取。
from .persona import load_terms  # noqa: E402


def build_terms_note(user_msg: str, denied_terms: set | None = None,
                     hit_out: list | None = None, selected_terms: set | None = None) -> str:
    """根据用户消息命中名词库：核心词(always)每次注入 + 命中词(关键词/别名/正则)注入。

    返回注入文本【灰泽满的世界】。客观打底 + 带灰泽满态度，让模型既懂词义又有正确的相处态度。

    hit_out：传入 list 时，把**真正命中并注入了词义**的词条 keyword 依次追加进去
    （条目级注入轨迹用；概念命中的 usage 规则没有词条名，不计）。

    命中检查：
    - priority=always：每次注入
    - keyword / aliases：子串命中
    - pattern（可选）：正则命中。用于子串匹配做不到的场景——如'区'单字撞'小区/地区'，
      用 `满区|(这么|太|好|很|真…)\\s*区` 只命中'满区'或形容词用法（'你怎么这么区'），
      不误触普通名词里的'区'。

    双向（v2）：
    - usage：该词的"主动用词规则"——回复里表达这个概念时用这个词（意思→词）。
    - usage_triggers：概念触发词。消息出现这些词（关键词没命中）时注入 **meaning + usage**，
      让模型既知道这个词是什么意思，也知道可以主动用它，而不是只会听懂。
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
        if selected_terms is not None and kw not in selected_terms:
            continue
        usage = t.get("usage")
        key_hit, concept_hit = term_match(t, msg)
        approved = selected_terms is not None and kw in selected_terms
        hit = approved or t.get("priority") == "always" or key_hit
        if hit:
            parts = [t.get("meaning", "")]
            if t.get("reaction"):
                parts.append(f"被提到时：{t['reaction']}")
            if usage and (approved or key_hit or concept_hit):
                parts.append(f"用词规则：{usage}")
            notes.append(f"{kw}：{'；'.join(parts)}")
            if hit_out is not None:
                hit_out.append(kw)
        elif usage and concept_hit:
            # 概念命中：消息没提关键词，但提到概念词 → 给 meaning + 用词规则
            parts = [t.get("meaning", ""), f"用词规则：{usage}"]
            notes.append(f"{kw}（相关概念）：{'；'.join(p for p in parts if p)}")
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
        if term_match(t, msg)[0]:
            return True
    return False


async def confirm_ambiguous_terms(user_msg: str, aux_client=None) -> set:
    """LLM 语境确认（词→意思的准确性守卫）。

    对 `confirm:true` 且本轮命中的术语，用一次便宜 LLM 判断"这个词在这个语境下是否真指它定义的含义"，
    返回应剔除的 keyword 集合（交给 build_terms_note 跳过）。防止 pattern/短词命中误触
    （如'满区'命中'怎么这么区'，可能是普通形容词用法而非黑话）。

    ⚠️ **必须传辅助链客户端**：下面用的是 `aux_model(...)`（u-ai 的 gemini）。
    2026-11-10 实测：传主客户端（DeepSeek 官方端点）会把 gemini 模型名发过去，
    每次都 400 "The supported API model names are ..."，并把这层打进 3 分钟冷却。

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
        hit = term_match(t, msg)[0]
        if hit:
            candidates.append(t)
    if not candidates:
        return set()
    if aux_client is None:
        try:
            aux_client, _ = aux_clients_or(_get_clients)
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
        resp = await aux_client.chat.completions.create(
            model=aux_model(_get_model_name()),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=320,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "术语消歧")
        data = parse_json_object(extract_chat_content(resp))
        values = data.get("exclude", [])
        exclude = {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()
        return {t["keyword"] for t in candidates if t["keyword"] in exclude}
    except Exception as e:
        print(f"⚠️ 术语语境确认失败（放行）: {e}")
        return set()


def build_message_list(user_msg: str, fused_items: list,
                       user_history: list,
                       vision_desc: str = "", weather_city: str = "",
                       batch_summary: str = "", preference_items: list = None,
                       core_stories: list = None, session_context: str = "",
                       query_hint: str = "", pragmatic_hint: str = "", denied_terms: set | None = None,
                       group_context: str = "", history_gap_note: str = "",
                       prev_session_note: str = "", same_request: dict = None,
                       profile_context: str = "", commitment_context: str = "",
                       emotion_context: str = "", daily_context: str = "",
                       user_transcript: str = "", nickname_hint: str = "",
                       log_hits: bool = False, trace: dict | None = None,
                       association_context: str = "", association_entities: list | None = None,
                       wiki_context: str = "",
                       is_group: bool = False, group_address_note: str = "",
                       group_thread_note: str = "", name_budget: str = "",
                       group_batch_note: str = "",
                       habit_hint: str = "", now_context: str = "",
                       time_hint: str = "", prefetch_context: str = "", recall_context: str = "",
                       request_context: str = "",
                       reminders_context: str = "",
                       state_stable: str = "", state_dynamic: str = "",
                       event_context: str = "",
                       history_turns: list | None = None,
                       show_actions: bool = False, selected_terms: set | None = None,
                       song_context: str = "") -> list:
    """按优先级组装发送给模型的消息列表。

    fused_items 为融合后的 RetrievalItem 列表（corpus + behavior），按源分组注入。
    vision_desc 为用户消息附带的图片视觉描述（可选）。
    weather_city 为该用户所在城市（空则用全局默认天气城市）。
    batch_summary 为一批消息的智能归纳（可选，提示层）。
    preference_items 为命中相关偏好的条目列表（第 5 路语义检索，可选）。
    core_stories 为命中的核心记忆（印象最深的结晶，可选）。
    session_context 为会话级记忆（当前话题 + 本场事件，可选）。
    query_hint 为短消息的语境扩充（可选）：模型理解短消息用，正文仍是原 msg。
    profile_context / commitment_context / emotion_context 为 Memory V2 的三段注入
    （见 docs/记忆v2设计.md §三）：相处方式靠前、未完成约定中段、当下情绪紧挨用户消息。
    log_hits=True 时把**真正进提示词**的 corpus 条目记进 `data/corpus_hits.jsonl`，
    并把 12–19 的实际开火块名记进 `data/block_hits.jsonl`
    （只有真实聊天路径传 True；离线实验保持 False，免得污染统计）。

    trace：条目级注入轨迹（`docs/数据迭代.md` §2）。传一个 dict 进来，这里把
    **实际注入了哪些块、哪些条目**填进 `blocks` / `items` / `values`；写盘由
    `handle_chat` 统一做（离线/实验路径不传就不留痕）。只记 id/名称/值，不记整段提示词。

    association_context：关联背景（作品→作者→喜好）的关系句，见
    `retrieval.resolve_associations`；association_entities 是轨迹里的实体清单。

    is_group：**私聊/群聊的唯一真值**（来自 NapCat 的 message_type，别拿 to_me 或
    group_context 是否为空反推）。决定第 00 块注入哪段场景文案。

    group_address_note：群聊本批谁点名了她（系统元信息）。**不写进用户消息**——
    写进去模型会偶发把标记原样吐出来（2026-10-21 实测 2/216）。
    **只在有人点名时才有值**；没人点名就不注入任何“没点名”的信息（2026-10-23）。

    group_thread_note：群聊这批的对话线归纳（谁在对谁说、停在哪句话头上），
    紧跟在群块后注入，让她接在同一条线上。

    name_budget：称呼预算提醒（`name_budget_note` 渲染好的整句）。
    位置紧跟相处方式——它是对“怎么称呼这个人”的当场修正，不该被后面的素材盖过。

    habit_hint：这个用户当前时段/话题相关的`habit_*`记忆提示（`build_habit_hint`）。
    只对拥有该习惯的用户有值；放在用户消息前，让短问候也能接住个人习惯。

    history_turns：私聊原生历史轮次（`build_native_history_turns`）。提供时**跳过**
    `user_history` 的 system 块，在本轮感知前插入 header + native turns +
    【别自相矛盾】；群聊不传，仍走原来的转述 system 块。
    """
    # 场景（第 00 块）：**只有群聊注入**。
    # 私聊场景已移回骨架（2026-10-21 用户定：骨架保持原句
    # 「正在 QQ 上与绿冻聊天 / 你是在直接与对方聊天」）。群聊才需要额外提醒
    # “有多个绿冻、只有点名才是跟她说”——这句不能写进骨架，否则私聊也会带上。
    if is_group:
        messages = [{"role": "system",
                     "content": SYSTEM_PROMPT + "\n\n" + inj.T("scene", "group")}]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    hit_blocks: list[str] = []   # 12–19 本轮回实际注入了哪些块（真实聊天才落盘）

    # 轨迹：按**实际注入顺序**记块名；items 只收真正渲染进提示词的条目
    # （RRF/预算截掉的不算，和 log_corpus_hits 的纪律一致）。
    def _mark(name: str) -> None:
        if trace is not None:
            trace.setdefault("blocks", []).append(name)

    def _items(name: str, values: list) -> None:
        values = [str(v) for v in values if v]
        if trace is not None and values:
            trace.setdefault("items", {})[name] = values

    if is_group:
        hit_blocks.append("scene_group")
        _mark("scene_group")

    # 周表（地面真值）：被问"明天来吗/这周/几点播"以它为准。
    # 记忆里带"明天/下周"的话是过去某场直播当时的说法，可能早过期，不能当现在的安排。
    # 这里**只有 weekly**（weekday 制，到周自动对，永不过期），没有会过期的自由文本字段。
    sched = load_schedule()
    weekly = sched.get("weekly") if sched else None
    if weekly:
        messages.append({"role": "system", "content": inj.T("schedule", weekly=inj.format_weekly(weekly))})
        _mark("schedule")

    # 稳定用户卡（2026-11-05）：私聊每轮靠前注入；事件/当天/情绪不进卡。
    if state_stable:
        messages.append({"role": "system", "content": state_stable})
        hit_blocks.append("state_stable")
        _mark("state_stable")
        if trace is not None:
            trace.setdefault("values", {})["state_stable"] = state_stable[:300]

    # Memory V2【靠前】相处方式：偏好→行为句 + 话题相关事实（个体适配，优先级最高）。
    # 位置靠前是有意的：它是"针对这个人的长期适配"，不该被后面的通用素材盖过。
    if profile_context:
        messages.append({"role": "system", "content": profile_context})
        _mark("profile")
        if trace is not None:
            trace.setdefault("values", {})["profile"] = profile_context[:300]

    # 称呼预算（2026-10-23）：最近几轮叫得太多时，这轮前置提醒别再叫。
    # 紧跟在相处方式之后：`address_style.nickname` 会鼓励称呼，这里是对它的当场配平。
    if name_budget:
        messages.append({"role": "system", "content": name_budget})
        _mark("name_budget")

    # 长期请求（2026-10-27）：用户希望她以后怎么对待自己；是背景事实，不是本轮待办。
    if request_context:
        messages.append({"role": "system", "content": request_context})
        _mark("request")

    # 偏好档案（第 5 路语义检索）：聊到相关话题才注入；与语料/记忆冲突时以偏好为准
    if preference_items:
        prefs_text = "；".join(
            f"{p.get('category', '')}：{p.get('text', '')}" for p in preference_items
        )
        if prefs_text:
            messages.append({"role": "system", "content": inj.T("preferences", items=prefs_text)})
            _mark("preferences")
            _items("preferences", [p.get("id") or p.get("category") for p in preference_items])

    # 核心记忆（印象最深的结晶）：粉丝常提及的过去故事，命中才注入
    if core_stories:
        story_text = "；".join(
            f"{s.get('category', '')}：{s.get('text', '')}" for s in core_stories
        )
        if story_text:
            messages.append({"role": "system", "content": inj.T("core_stories", items=story_text)})
            _mark("core_stories")
            _items("core_stories", [s.get("id") or s.get("category") for s in core_stories])

    # 群聊现场（轻量群记忆注入）：在场成员 + 群内近况。群 = 好几个不同的绿冻在场，
    # 回给刚才直接点名搭话的那位，别把不同的人当成同一个"你"。
    # ⚠️「好几个绿冻，别当成同一个人」和「只认点名/@」两条**挂在这一块上**（2026-10-16 从骨架搬来）：
    #    它们只在群聊成立，而私聊时这块压根不注入——留在骨架里就是一条永远在场的群聊规则。
    if group_context:
        messages.append({"role": "system", "content": inj.T("group", context=group_context)})
        _mark("group")

    # 群聊对话线：这批谁在对谁说、停在哪句话头上（群聊归纳，2026-10-23）。
    # 放在群块后面、点名情况前面——先知道场子，再知道谁在跟她说话。
    if group_thread_note:
        messages.append({"role": "system",
                         "content": inj.T("group", "thread_note", note=group_thread_note)})
        _mark("group_thread")

    # 本批谁点名了她：系统元信息（用户消息里只留干净的「昵称：文本」）。
    # ⚠️ 只有**有人点名**时才有值并被注入；没人点名时一个字都不注——
    #    「本批无人点名」本身会被模型复述成「没点我，才不管」（2026-10-23）。
    if group_address_note:
        messages.append({"role": "system",
                         "content": inj.T("group", "address_note", note=group_address_note)})
        _mark("group_addr")

    # 本批新消息的回复方式：编号 + “整段回” + <reply:n>。
    # 只在群聊且确实是一批多条时注；单条消息不需要额外指令。
    if group_batch_note:
        messages.append({"role": "system",
                         "content": inj.T("group", "batch_note", count=group_batch_note)})
        _mark("group_batch")

    # 名词库（terms/lorebook）：核心词 always 注入 + 命中用户消息的词注入——让模型懂"绿冻/枪神8/slg"这类词并带对的态度
    _term_hits: list = []
    terms_note = build_terms_note(user_msg, denied_terms=denied_terms,
                                  hit_out=_term_hits if trace is not None else None,
                                  selected_terms=selected_terms)
    if terms_note:
        messages.append({"role": "system", "content": inj.T("terms", note=terms_note)})
        _mark("terms")
        _items("terms", _term_hits)

    # 关联背景（作品 → 作者 → 喜好）：紧跟在名词库之后，段头文案在 07_terms/block.json。
    # 为什么单独一段：terms 是"这个词什么意思"，这里补的是"这个名字牵出什么"——
    # 用户只报作品名时，模型要能自己沿「作品 → 作者 → 她的喜好」接上，不用逐本收书。
    if association_context:
        messages.append({"role": "system",
                         "content": inj.T("terms", "associations", note=association_context)})
        _mark("associations")
        _items("associations", association_entities or [])

    # 萌娘百科按需背景（未知专名）：只作事实参考，不混进名词库/关联背景。
    # 放在关联背景之后、会话记忆之前——先懂词，再接话。
    if wiki_context:
        messages.append({"role": "system",
                         "content": inj.T("terms", "wiki_note", note=wiki_context)})
        _mark("wiki")

    # 会话级记忆（当前话题 + 本场事件）：让模型接得住会话调性
    if session_context:
        # ⚠️ 段头原为【当前会话】——那在**断言"这是正在进行的对话"**（2026-10-05 改）。
        #    它可能是几小时前那场、早就自然结束了，所以改成中性的【会话记忆】，
        #    里面只陈述"话题是什么 + 什么时候开始的"，判断交给模型。
        messages.append({"role": "system", "content": inj.T("session", context=session_context)})
        _mark("session")

    # 上一场会话（已过期，带"隔了多久"）：**不能并进上面那块**——
    # 那句写的是"你们这一场对话"，把几天前的东西塞进去她会当成现在。
    # 单独一条、且自己带着时间定性（见 session_memory.previous_session_note）。
    if prev_session_note:
        messages.append({"role": "system", "content": prev_session_note})
        _mark("prev_session")

    # ⚠️ 感知（当前时间/农历/天气）**挪到最末尾**（2026-10-08），见下面 final_user 之前那处。
    #    它是唯一"每分钟必变"的块：原来坐在第 5 位，导致它后面的**每一块**
    #    （行为指令/直播记忆/长期记忆/短期记忆…）在 DeepSeek 的**前缀缓存**里永远 miss。
    #    实测（2026-10-11，真发请求读 prompt_cache_hit_tokens；数据见 docs/历史/交接文档.md）：
    #    **稳定前缀就是最前面两块**（system_prompt 3413 字 + 周表 ≈ 2176 token）——从第 3 块
    #    （相处方式/偏好/名词库…）起**每轮都变**（检索/记忆/profile 都按当轮 query 选），
    #    所以分界线在第 2 块、不是"倒数第二块"（那句旧注释是错的）。
    #    ⚠️ 因此**重排块序不是杠杆**：能救的已经救了（感知在末尾），再往后挪任何块都没有收益——
    #    想提高命中只有"减少注入量 / 换模型 / 接受"三条路。

    behaviors, corpus = _split_fused(fused_items)

    # 行为指令（source=behavior）
    if behaviors:
        behavior_text = "\n\n".join(it.text for it in behaviors if it.text)
        if behavior_text:
            messages.append({"role": "system", "content": inj.T("behaviors", text=behavior_text)})
            _mark("behaviors")
            _items("behaviors", [it.item_id for it in behaviors if it.text])

    # 直播记忆（source=corpus）：只当"背景记忆"，不参与风格示范
    if corpus:
        # ⚠️ **每条自己带时间定性前缀**（2026-10-15，用户定的）。
        #    段头那句总说明（"都是曾经发生的"）实测管不住模型——项目自己的教训是：
        #    说明写在整块外面，模型读单条时不会往回看。加在**每条自己身上**才对。
        #    ⚠️ 走**注入层**不改数据：corpus 每条都进 embedding，数据层加前缀 = 给 421 个
        #    向量加同一个常数偏移、稀释条间区分度，还要重算全库。要回退，改 block.json 的 item。
        context = "\n".join(inj.T("corpus", "item", text=it.text) for it in corpus if it.text)
        if context:
            messages.append({"role": "system", "content": inj.T("corpus", items=context)})
            _mark("corpus")
            _items("corpus", [it.item_id for it in corpus if it.text])
            # 命中留痕：只记**真正进了提示词**的条目（分析库利用率用，失败静默）。
            # log_hits 由真实聊天路径打开；再加通道闸——回归/AB/回放等离线脚本
            # 直接调 handle_chat 时 `_ROUTE` 是空的，不能混进线上统计。
            if log_hits and get_llm_route() in ("私", "群"):
                log_corpus_hits(user_msg, [it for it in corpus if it.text])

    # Memory V2【中段】未完成约定：别忘、要兑现（system 中段，紧挨记忆区）。
    if commitment_context:
        messages.append({"role": "system", "content": commitment_context})
        hit_blocks.append("commitment")
        _mark("commitment")

    # ⚠️ 这里原来注「长期记忆」（第 13 块 `memory`）。2026-10-17 整段删掉：
    #    它的内容来自 v1 的长期记忆卡（根 `memory_manager.py` → `user_memory/long_term/*`），
    #    而 v1 已于 2026-10-12 整个删除，生产者不存在了；`memory_context` 在调用侧
    #    恒为空串，`if` 永远不成立。它想装的东西现在由 03（相处方式 + 相关事实）
    #    和 12（未完成约定）承担，接上反而是同一批事实一轮出现两次。
    #    参数已一并移除（含 20 处调用点）。

    # 短期记忆注入：**用户 + 她自己**最近 5 轮的完整上下文（2026-10-18 用户定）。
    # 2026-10-14 曾把她自己的话从注入里过滤掉（去自反馈：实测"灰泽满"起手黏性
    # 51.7% vs 17.4%）；用户 2026-10-18 明确要求恢复完整上下文——她需要看到
    # 自己上一轮说了什么，才能接得住、不前后打架。记忆里两边本来就都记着。
    if user_history and not history_turns:
        if isinstance(user_history, list):
            context = "\n".join(ln for ln in user_history if ln)
            if context:
                # ⚠️ 时间戳**不再**在这里加一行汇总（2026-10-05 改）——
                #   旧做法是把「距上一轮多久」当一行汇总拼在整块开头，结果 gap=10秒 和 gap=3小时
                #   注入出来的历史**长得一模一样**，她分不出"刚说完"和"三小时前说的"。
                #   现在改成：**每条消息自己带发送时间**（由 memory.get_user_history_timed 渲染：
                #   「今天/昨天 + 时刻 + 距现在多久」），只给事实、不给"该怎么理解"的说明，判断交给模型。
                #   history_gap_note 参数保留（调用方还在传），但不再拼进历史块。
                text = inj.T("history", context=context) + inj.T("history", "contradiction")
                messages.append({"role": "system", "content": text})
                hit_blocks.append("history")
                _mark("history")
        else:
            messages.append({"role": "system",
                             "content": inj.T("history", "fallback", history=user_history)})
            _mark("history")

    # 事件背景（2026-11-05）：用户与灰泽满之间发生过的事，按指涉/实体/向量取；
    # 放在历史之后——它是背景，不能挤掉“最近几轮到底说了什么”。
    if event_context:
        messages.append({"role": "system", "content": event_context})
        hit_blocks.append("events")
        _mark("events")
        if trace is not None:
            trace.setdefault("values", {})["events"] = event_context[:300]

    # 连续索要（客观事实，不含指令）：她需要知道"这已经是第几次了"。
    # 为什么必须有（2026-10-04 从真实记录发现）：上下文只带最近 10 条，
    # 跨轮的连续索要她完全看不见 → 每轮都当成"第一次被要" → 换着理由挡 5 轮后突然松口，
    # 松口之后又失去"我从没答应过"的立场（3076669330 要"宝宝"/要语音两段都是这样）。
    # ⚠️ 这里**只给事实**（第几轮、之前在要什么），**不给"该不该答应"**——
    #    "怎么应对"是 behaviors 的事；而且实测"管怎么用"的说明无效、事实类信息有效。
    if same_request and int(same_request.get("count") or 1) >= 2:
        what = same_request.get("what") or "同一件事"
        messages.append({"role": "system", "content": inj.T(
            "same_request", count=same_request["count"], what=what,
            prev=same_request["count"] - 1)})
        hit_blocks.append("same_request")
        _mark("same_request")


    # 感知源②：图片消息——把视觉描述并入用户消息，避免空消息让模型以为"对方没说话"。
    # user_transcript 是 chat_window 按消息顺序拼好的完整批次（含图片标记、已过滤表情包）；
    # 有它就优先用，避免把 vision_desc 再当尾部块重复/抢话题。
    final_user = user_transcript or user_msg
    if not user_transcript and vision_desc:
        final_user = f"{user_msg}\n[图片：{vision_desc}]" if user_msg.strip() else f"[图片：{vision_desc}]"

    # 感知源③：批量归纳（用户连发多条时，作为理解整批的提示层，原文仍完整保留）
    if batch_summary:
        messages.append({"role": "system", "content": inj.T("batch", summary=batch_summary)})
        hit_blocks.append("batch")
        _mark("batch")

    # 短消息语境提示：用户消息 ≤4 字时，把扩充后的完整语义作为提示给模型
    # （正文仍是原 msg，这里帮模型理解"咋这样"这种短句的真实含义）
    if pragmatic_hint:
        # 语用注释（玩笑/反讽/预演）：和短消息补全不同，它作用于**任意长度**的消息。
        # 单独一个变体，避免 query_hint 的“短消息/指代性消息”措辞误导模型。
        messages.append({"role": "system", "content": inj.T("query_hint", "pragmatic", hint=pragmatic_hint)})
        hit_blocks.append("query_hint")
        _mark("query_hint")
    elif query_hint:
        # 纯表情消息：按表情的真实情绪回应，体现情绪该有的态度，不被当前话题绑架
        if is_emoji_msg(final_user):
            # 这里**只做"别读错情绪"的正确性约束，不规定她该怎么反应**——
            # 「什么情绪 → 她怎么回应」属于数据层，该由 behaviors 的条目 + 真人样本承担。
            # （我曾在这里写过"委屈/可怜 → 她该放软、别嘴硬顶着"，那是提示词补丁，已回退：
            #   同一个会话里我刚把 😅 那行的"规定动作"删掉，转手又加一条，方向自相矛盾。
            #   见 docs/历史/待办清单.md / 交接文档：纯表情消息目前会跳过 L3，所以数据层压根没参与。）
            messages.append({"role": "system", "content": inj.T("query_hint", "emoji", hint=query_hint)})
        else:
            messages.append({"role": "system", "content": inj.T("query_hint", "short", hint=query_hint)})
        hit_blocks.append("query_hint")
        _mark("query_hint")

    # 原生历史先于本轮感知，避免旧回复里的报时离当前提问更近。
    # 时间只标在用户消息上，assistant 正文沿用她的原话。
    if history_turns:
        messages.append({"role": "system", "content": inj.T("history", "turns_head")})
        messages.extend(history_turns)
        messages.append({"role": "system",
                         "content": inj.T("history", "contradiction").strip()})
        hit_blocks.append("history")
        _mark("history")

    # 感知源①：当前时间/农历/天气（始终注入，占预算极少；天气按该用户所在城市）。
    # **放在最后**：它是唯一"每分钟必变"的块，而 DeepSeek 的缓存按**前缀**匹配——
    # 它放中间会把后面所有块（行为/直播记忆/长期短期记忆…）一起拖成 miss。
    # 语义上也更顺：紧挨着用户这句，"现在几点"最贴近此刻。
    # v3：动态段由 state_dynamic 提供（日期/星期/直播/天气/话题/情绪）。
    if time_hint:
        messages.append({"role": "system", "content": time_hint})
        hit_blocks.append("time_hint")
        _mark("time_hint")
    # 当前状态动态段（时间/天气/直播/话题）：稳定段已靠前，这一段必须最后。
    if state_dynamic:
        messages.append({"role": "system", "content": state_dynamic})
        hit_blocks.append("state_dynamic")
        _mark("state_dynamic")
        if trace is not None:
            trace.setdefault("values", {})["state_dynamic"] = state_dynamic[:300]
    else:
        now_context = now_context or context_probe.get_now_context(city=weather_city)
        if now_context:
            messages.append({"role": "system", "content": now_context})
            hit_blocks.append("now")
            _mark("now")
            if trace is not None:
                trace.setdefault("values", {})["now"] = now_context.split("\n", 1)[0][:80]

    # Memory V2【最后】当下情绪：紧挨用户消息（跟【当前时间】一样放尾部），只影响这一轮语气。
    # （文案在 persona/injection/19_mood/block.json；session_memory.mood_note 渲染。）
    if emotion_context:
        messages.append({"role": "system", "content": emotion_context})
        hit_blocks.append("mood")
        _mark("mood")
        if trace is not None:
            trace.setdefault("values", {})["mood"] = emotion_context[:80]

    # 称呼提示（2026-10-19）：只在用户**这一轮**叫了她的外号时出现。
    # 默认仍自称"灰泽满"；这一条是一次性的情境提示，不是长期偏好。
    if nickname_hint:
        messages.append({"role": "system", "content": nickname_hint})
        hit_blocks.append("nickname")
        _mark("nickname")

    # 【他今天的日常】（第 20 块）：**每轮都注**，放最尾——跟【当前时间】同理，
    # 它是"每分钟都可能变"的动态段，放中间会把后面的块一起拖成缓存 miss。
    # 条目 <2 条时 daily_memory 自己会返回空串（宁可不提）。
    if daily_context:
        messages.append({"role": "system", "content": inj.T("daily", context=daily_context)})
        hit_blocks.append("daily")
        _mark("daily")
        if trace is not None:
            trace.setdefault("values", {})["daily"] = True

    # 个人固定习惯（`habit_*` 记忆）：**只对拥有这条记忆的用户**出现在用户消息前，
    # 让“早/起床了”这类短问候也能接住（不是全局 behaviors）。
    if habit_hint:
        messages.append({"role": "system", "content": habit_hint})
        hit_blocks.append("habit_hint")
        _mark("habit_hint")

    if reminders_context:
        messages.append({"role": "system", "content": reminders_context})
        hit_blocks.append("reminders")
        _mark("reminders")
        if trace is not None:
            trace.setdefault("values", {})["reminders"] = reminders_context[:300]

    # 可用动作（第 23 块，2026-11-11 试点）：只列能力，不写“什么时候该用”。
    # 主动发言路径不注（它不是聊天，主模型也不执行这条链路的动作）。
    if show_actions:
        # 类别用**真实表情组名**，别让模型照着示例瞎编（编的类别会被执行层丢掉）
        try:
            from .stickers import load_stickers
            groups = sorted({str(s.get("group") or "").strip()
                             for s in load_stickers() if s.get("group")})
        except Exception:
            groups = []
        action_note = inj.T("actions", groups="、".join(groups) or "（暂无表情）")
        if song_context and not is_group:
            action_note += "\n" + song_context
        messages.append({"role": "system", "content": action_note})
        hit_blocks.append("actions")
        _mark("actions")

    if prefetch_context:
        messages.append({"role": "system", "content": prefetch_context})
        hit_blocks.append("prefetch")
        _mark("prefetch")

    if recall_context:
        messages.append({"role": "system", "content": recall_context})
        hit_blocks.append("reply_evidence")
        _mark("reply_evidence")

    # 自称方式（第 21 块，2026-10-27 用户定）：骨架里不再写整段自称规则，
    # 只在**所有注入之后、用户消息之前**留一句弱约束。实测（60 条真实消息）：
    # 名字自指 28%、第一人称 7%、才不 13%、名字+才 7%、名字起手 2%——
    # 比整段规则更自然，也避开了“名字+才”的固定句式。允许少量“我”，别再改回整段。
    messages.append({"role": "system", "content": inj.T("selfref")})
    hit_blocks.append("selfref")
    _mark("selfref")

    # 运行时块级开火留痕（scene_* / 12–19；真实聊天路径才写，离线实验/回归不污染）
    if log_hits and get_llm_route() in ("私", "群"):
        log_block_hits(hit_blocks)

    messages.append({"role": "user", "content": final_user})
    return messages


# 生成失败时的兜底文案（**只发给用户，绝不进记忆**——见 is_fallback_reply）
# 2026-10-14 换：原来那句"哎呀，hzm脑子卡了一下……"把**故障**摆到了台面上（还是她自己承认故障）。
# 用户要求改成"她这会儿不在"——一句**体面的缺席**，把"上游挂了/超时/被内容审核拦了"都盖过去。
_FALLBACK_REPLY = "小满现在不在哦~"
_FALLBACK_SILENT = "小满现在不在哦~"

# 历史兜底文案：老记录里可能还留着，识别时一起认（否则会被当成"她说过的话"喂回去）
_LEGACY_FALLBACKS = ("哎呀，hzm脑子卡了一下……", "……（沉默，可能是信号不好）")


def is_fallback_reply(text: str) -> bool:
    """这句是不是"没生成出来"的兜底文案。

    用途：兜底文案不该写进记忆。踩坑：上游偶发 `choices: null` 那一轮，
    兜底文案被当成"她的回复"存进了短期记忆，于是她的历史里永久留着一句
    `哎呀，hzm脑子卡了一下……（错误: ...）`——**AI 腔的故障信息变成了"她说过的话"，
    还会作为 few-shot 喂回去**。一次上游抖动 = 一份永久污染。
    （2026-10-14 换文案后，老文案仍要认得——历史记录里就存着。）
    """
    t = (text or "").strip()
    return t in (_FALLBACK_REPLY, _FALLBACK_SILENT) or t in _LEGACY_FALLBACKS


def build_native_history_turns(user_id: str, *, limit: int = 10) -> list[dict]:
    """把私聊短期记忆渲染成原生 user/assistant turns（2026-11-19）。

    为什么不再逐条转述：真实历史本来就是「用户说一句、她说一句」，
    assistant 才是它原本的通道；实测 29 对样本里，B 的追问自述重合 0.23 vs 转述 0.13，
    复读和「刚说了」都没有增加。
    仍然保留的两件小事：
      - 用户消息前带发送时间；她说的话不带前缀（避免她学会在正文里写时间）。
      - 跨天的「今天/明天」落成绝对日期（确定性替换，不调模型）。
    兜底文案不放进历史——那是生成失败的占位，不是她说过的话。
    """
    turns: list[dict] = []
    for entry in get_user_history_entries(user_id)[-limit:]:
        text = str(entry.get("text") or "").strip()
        t = entry.get("t")
        lead = _timed_prefix(t) if t is not None else ""
        if is_bot_line(text):
            body = bot_line_text(text).strip()
            if not body or is_fallback_reply(body):
                continue
            body = self_speech._localize_day_words(body, lead)
            role = "assistant"
        else:
            plain = _strip_timed(text)
            if not plain.startswith("用户："):
                continue
            body = plain[len("用户："):].strip()
            if not body:
                continue
            body = self_speech._localize_day_words(body, lead)
            if lead:
                body = f"{lead}{body}"
            role = "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] = turns[-1]["content"] + "\n" + body
        else:
            turns.append({"role": role, "content": body})
    return turns


_PLANNING_REWRITE_PROMPT = (
    "上一次回复混入了分析、计划或旁白，已经被丢弃。"
    "现在只输出灰泽满要直接发给对方的那句话：不要复述任何规则、不要列举备选回复、"
    "不要写「参考腔调」、不要评价自己的回复，也不要写动作或神态。"
)


async def _rewrite_planning_reply(client, messages: list, original: str, reason: str) -> str:
    """规划式泄露的兜底：带一条强约束重写一次；仍不合格就回兜底文案。

    为什么重写而不是删掉命中的句子：泄露往往**整条都是草稿**，没有可保留的台词。
    为什么不是直接兜底：偶发故障不该让用户看到"她不在"，先给模型一次机会。
    """
    try:
        retry_messages = list(messages) + [
            {"role": "system", "content": _PLANNING_REWRITE_PROMPT}]
        response = await client.chat.completions.create(
            model=_get_reply_model(),
            messages=retry_messages,
            temperature=0.6,
            frequency_penalty=CHAT_FREQUENCY_PENALTY,
            max_tokens=CHAT_MAX_TOKENS,
            **THINKING_DISABLED,
        )
        log_cache_usage(response, "主回复重写")
        rewritten = extract_chat_content(response)
        if rewritten and not planning_leak_reason(rewritten):
            log_leak_guard(original, reason, rewritten, "rewritten")
            return rewritten
        log_leak_guard(original, reason, rewritten, "still_bad")
        return _FALLBACK_REPLY
    except Exception as e:
        print(f"⚠️ 规划式回复重写失败: {e}")
        log_leak_guard(original, reason, "", "error")
        return _FALLBACK_REPLY


async def generate_reply(messages: list) -> str:
    """调用 DeepSeek 生成回复；失败时返回兜底文案（调用方负责"别把它记进记忆"）。"""
    try:
        deepseek_client, _ = _get_reply_clients()
        response = await deepseek_client.chat.completions.create(
            model=_get_reply_model(),
            messages=messages,
            temperature=CHAT_TEMPERATURE,
            frequency_penalty=CHAT_FREQUENCY_PENALTY,
            max_tokens=CHAT_MAX_TOKENS,
            **THINKING_DISABLED,
        )
        # ⚠️ 主回复是**最大的 token 消费者**，但以前漏了插桩——日志里那三行 `主回复`
        #    其实是批量归纳/转发归纳/术语消歧三个小助手（2026-10-11 查明）。
        #    不打出这一行，"缓存到底命中多少"根本没法归因。
        log_cache_usage(response, "主回复")
        reply = extract_chat_content(response)
        leak_reason = planning_leak_reason(reply)
        if leak_reason and get_flag("LEAK_GUARD", "1") != "0":
            print(f"⚠️ 主回复混入规划/旁白（{leak_reason}），丢弃重写")
            reply = await _rewrite_planning_reply(deepseek_client, messages, reply, leak_reason)
    except Exception as e:
        # 异常原文不再塞进回复：那是内部信息，会原样发给 QQ 上的每个用户。
        # 打到控制台即可，她这边只回一句人话。
        print(f"⚠️ 生成失败: {e}")
        reply = _FALLBACK_REPLY
    return reply if reply else _FALLBACK_SILENT


async def update_memory_v2_shadow_task(
    user_id: str,
    user_msg: str,
    reply: str,
    session_id: str,
    reminder_hint: dict | None = None,
    *,
    source_turn_id: str = "",
    delivery_verified: bool = False,
) -> list[dict]:
    """Extract one private turn into the opt-in V2 shadow store.

    This path is deliberately fail-closed and has no influence on the reply.
    Keeping the feature check inside the task also makes direct/offline callers
    safe when the environment flag is absent.

    `reminder_hint`：L3 判出的定时提醒请求。命中且时间明确时**绕过成本闸门**——
    "18:20 提醒我"通常很短，正常会被 worth_extracting 拦掉，但这条必须落库。
    """
    if not memory_v2.shadow_enabled():
        return []
    msg = (user_msg or "").strip()
    if not msg or is_emoji_msg(msg):
        return []
    force_extract = bool(reminder_hint and not reminder_hint.get("ambiguous"))
    # 成本闸门只控制模型提取，短句里的明确称呼仍走本地记录。
    use_model = (force_extract or get_flag("MEMORY_EXTRACT_GATE", "1") == "0"
                 or memory_v2.worth_extracting(msg))
    try:
        deepseek_client, zhipu_client = _get_clients() if use_model else (None, None)
        # 可选的提取模型覆盖（2026-10-27 成本 A/B）：
        # 默认跟随主模型；设 MEMORY_EXTRACT_MODEL=... 可只给提取换模型，不动主回复。
        model = get_flag("MEMORY_EXTRACT_MODEL", "") or _get_model_name()
        return await memory_v2.extract_and_ingest(
            client=deepseek_client,
            model=model,
            store=_MEMORY_V2_STORE,
            user_id=user_id,
            user_text=user_msg,
            assistant_text=reply,
            session_id=session_id,
            zhipu_client=zhipu_client,
            reminder_hint=reminder_hint,
            source_turn_id=source_turn_id,
            delivery_verified=delivery_verified,
        )
    except Exception:
        return []


# 用户所在地（天气感知用）：从 **V2 的 location 事实**取。V2 存的是"在曼谷""在泰国留学"
# 这种，取值前要洗成纯地名。（v1 卡的 weather_city 回落已于 2026-10-12 随 v1 清理撤掉——
# v1 只有 4 个用户带 weather_city，而 V2 有 10 个，回落价值趋近于零。）
_WEATHER_CITY_KEYS = {"location", "city"}
_WEATHER_CITY_PREFIX = re.compile(r"^(?:在|住在|现居|目前住在|老家在)")
_WEATHER_CITY_TAIL = re.compile(r"[，,。;；].*$|(?:留学|定居|工作|生活|读书|上学|上班).*$")


def _weather_city_for(user_id: str) -> str:
    """用户所在城市：V2 的 location/city 事实；取不到返回空串（天气用默认城市）。

    洗出来的不像地名（太长）就别用——天气查不到只是少一行天气，不致命；
    但把一个长句当地名填进去，等于把这个功能废掉。
    """
    try:
        memories = _MEMORY_V2_STORE.snapshot(user_id).get("memories", [])
    except Exception:
        memories = []
    for memory in memories:
        if memory.get("kind") != "fact" or memory.get("status") not in {"confirmed", "active"}:
            continue
        if str(memory.get("key")) not in _WEATHER_CITY_KEYS:
            continue
        value = _WEATHER_CITY_PREFIX.sub("", str(memory.get("value") or "").strip())
        value = _WEATHER_CITY_TAIL.sub("", value).strip()
        if 2 <= len(value) <= 8:
            return value
    return ""


_USER_NAME_KEYS = {"preferred_name", "name", "nickname"}

# 名字记忆的"外壳词"：V2 的值常常被写成整句（"用户希望被称呼为星屿哥""希望被叫小温"），
# 得把这层剥掉才拿得到名字本身。⚠️ **名单要够全**——漏一个词，这里就返回整句，
# 下面那个"别老拿名字开头"的守卫会因为比对不上而**静默失效**。
# 踩坑（2026-10-12 实测）：名单里只有"称呼为/被称为/希望被称为"，没有"希望被叫"，
# 于是小温那条返回 '希望被叫小温' → 守卫一次都没拦，她 **5/5 回复都用"小温，"开头**
# （修好提取后 5/5 → 1/5，守卫每次都能拦下）。
_NAME_WRAPPERS = (
    "用户", "对方", "他", "她", "希望", "要求", "接受", "想要", "想让", "让",
    "被", "称呼为", "称呼", "称为", "称作", "自称", "名叫", "名字是", "名字叫",
    "叫", "称", "是",
)


def _bare_name_from(value: str) -> str:
    """从记忆值里取出裸名字：「希望被叫小温」→「小温」。

    值常被写成整句（提取模型不听话），所以两步：① **只取第一个逗号前那截**——
    后面多半是"不叫lumen_example"这类补充说明；② 反复剥掉外壳词。剥不干净就返回原样
    （调用方有长度检查兜底）。
    """
    v = value.split("，")[0].split(",")[0].strip()
    changed = True
    while changed:
        changed = False
        for w in _NAME_WRAPPERS:
            if v.startswith(w) and len(v) > len(w):
                v = v[len(w):].strip("：: 　")
                changed = True
    return v.strip("：: 　\"'“”‘’（）()")


def _neutralize_group_history(lines: list[str]) -> list[str]:
    """注入前把「灰泽满（回甲）：…」的**具体名字**换成中性的「回群友」。

    为什么要改注释而不是加回复守卫（2026-11-01 用户指出）：她自己的回复会进历史被当样本；
    `（回X）` 标签会让模型学到“回复=先叫群友名字”，进而频繁名字起手。根在喂回去的注释，
    不在生成后的句子。这里**保留“她在回某位群友”这个关系**（理解不丢），只把具体名字去掉；
    `chat_log` / 短期记忆原文仍保留回复对象，供排查与统计。
    """
    out = []
    for ln in lines or []:
        def _repl(m):
            label = m.group(2)
            if label.startswith("回群友"):
                return m.group(0)   # 已经是中性标注（可能带 ·转述），别把它洗掉
            return f"{m.group(1) or ''}灰泽满（回群友）："
        out.append(re.sub(r"^(\[[^\]]+\]\s*)?灰泽满（(回[^）]*)）：", _repl, ln))
    return out




def _user_name_for(user_id: str) -> str:
    """用户希望被怎么称呼——用来判"她是不是又在拿名字当开头"。

    只从 V2 的 preferred_name/name/nickname 取。V2 的值常常是一整句
    （"用户希望被称呼为星屿哥，或者lumen_example"），所以取第一个逗号前、再剥掉外壳、
    且要求长度像名字（剥不干净的整句宁可返回空——返回错的还会**骗过**守卫）。
    （v1 卡 `user_name` 的首选层已于 2026-10-12 随 v1 清理撤掉。）
    """
    try:
        memories = _MEMORY_V2_STORE.snapshot(user_id).get("memories", [])
    except Exception:
        memories = []
    for memory in memories:
        if memory.get("kind") != "fact" or memory.get("status") not in {"confirmed", "active"}:
            continue
        if str(memory.get("key")) not in _USER_NAME_KEYS:
            continue
        value = _bare_name_from(str(memory.get("value") or ""))
        if 1 <= len(value) <= 12:
            return value
    return ""


# 称呼预算（2026-10-23）：最近 5 轮（10 行）里她叫对方名字的次数达到阈值，
# 就前置提醒“这轮尽量别再叫”。阈值可用 NAME_BUDGET_MIN_CALLS 覆盖（A/B 用），
# 设 0 等于关掉。
# 默认 1（5 轮内出现过就提醒）：真实回放 A/B（20 条，同一份提示词）含名字回复
# 基线 35% → ≥1 档 5%（降 86%）、≥2 档 15%（降 57%）、最近 2 条档 20%（降 43%，未达标）。
# 见 docs/历史/交接文档.md 2026-10-23。
NAME_BUDGET_WINDOW_LINES = 10
NAME_BUDGET_MIN_CALLS = 1


def _has_address_request(user_id: str) -> bool:
    """这个用户有“以后多叫名字”的长期 request_address 吗？"""
    try:
        for m in _MEMORY_V2_STORE.snapshot(user_id).get("memories", []):
            if m.get("kind") != "request" or m.get("key") != "request_address":
                continue
            if m.get("status") not in {"confirmed", "active"}:
                continue
            exp = m.get("expires_at")
            if exp:
                try:
                    from datetime import datetime as _dt
                    if _dt.fromisoformat(str(exp)) <= _dt.now():
                        continue
                except (TypeError, ValueError):
                    pass
            return True
    except Exception:
        pass
    return False


def name_budget_note(user_id: str) -> str:
    """最近几轮她叫对方名字太多次 → 返回一条**前置提醒**；否则空串。

    为什么需要它（2026-10-23）：名字一旦进短期记忆会自我强化——实测某用户前 100 轮
    0 次，第 103 轮起一路涨到最近 14 轮 14/14。现有 `starts_with_name` 守卫只拦
    “名字开头”这一条窄路径，管不住句中反复称呼；而 `address_style.nickname` 那条
    偏好又会鼓励她称呼。所以补一条**事实型**的前置提醒（只给次数，不下命令）。

    ⚠️ **只统计她自己说的话**（`is_bot_line`）：用户消息里提到名字不等于她在叫，
    算进来会把“用户自己说了一次名字”误判成她称呼滥用。
    """
    name = _user_name_for(user_id)
    if not name:
        return ""
    try:
        min_calls = int(os.environ.get("NAME_BUDGET_MIN_CALLS", str(NAME_BUDGET_MIN_CALLS)))
    except ValueError:
        min_calls = NAME_BUDGET_MIN_CALLS
    if min_calls <= 0:
        return ""
    # 用户明确要求“以后多叫名字”时，放宽预算：只在他已经连叫 3 次时才提醒。
    if _has_address_request(user_id):
        min_calls = max(min_calls, 3)
    hist = get_user_history(user_id)[-NAME_BUDGET_WINDOW_LINES:]
    calls = sum(bot_line_text(ln).count(name) for ln in hist if is_bot_line(ln))
    if calls < min_calls:
        return ""
    return inj.T("profile", "name_budget", name=name, n=calls)


async def gather_retrieval(query_text: str, retrieval_query: str, history_text: str,
                           deepseek_client, zhipu_client,
                           is_user_msg: bool = True,
                           extra_preference_ids: list | None = None,
                           context: RetrievalContext | None = None, *,
                           song_text: str = "") -> dict:
    """回复按对话语境筛选材料；动态广播保留原有公共素材检索。"""
    context = context or RetrievalContext(
        query_text, resolved_text=retrieval_query, history=history_text,
        is_user_msg=is_user_msg)
    unconfirmed_terms = {t["keyword"] for t in load_terms() if t.get("keyword")}
    result = {"fused_items": [], "preference_items": [], "core_stories": [],
              "same_request": None, "reminder": None,
              "denied_terms": unconfirmed_terms.copy(),
              "approved_terms": set(), "association_context": "", "association_entities": [],
              "wiki_entities": [], "entity_mentions": {},
              "embedding_available": True, "selection_trace": {"available": True}, "context": context}
    if not (query_text and not is_emoji_msg(query_text)) or is_emotion_only_query(retrieval_query):
        if is_user_msg:
            result["context"] = replace(context, needs=())
        return result
    try:
        await _fill_retrieval(result, query_text, retrieval_query, history_text,
                              deepseek_client, zhipu_client, load_persona_rules(), is_user_msg,
                              extra_preference_ids, context=context, song_text=song_text)
    except Exception as error:
        if is_user_msg:
            # 异常不能把未判需求的 None 继续传给事件/个人提示，亦不保留半成品材料。
            l3 = result.get("l3") or {}
            confirmed_behavior = [item for item in result.get("fused_items", [])
                                  if l3.get("classification_ok") is True
                                  and getattr(item, "source", "") == "behavior"
                                  and getattr(item, "item_id", "") == l3.get("behavior")]
            result.update(
                context=replace(context, needs=()), fused_items=confirmed_behavior,
                corpus_items=[], behavior_items=[], preference_items=[], core_stories=[],
                denied_terms=unconfirmed_terms, approved_terms=set(), association_context="",
                association_entities=[], wiki_entities=[], entity_mentions={},
            )
        result["selection_trace"] = {"available": False, "error": type(error).__name__}
        print(f"⚠️ 检索层异常（按未确认材料继续）: {type(error).__name__}")
    return result


async def _fill_retrieval(result: dict, query_text: str, retrieval_query: str, history_text: str,
                          deepseek_client, zhipu_client, behaviors, is_user_msg: bool,
                          extra_preference_ids: list | None = None,
                          context: RetrievalContext | None = None, *, song_text: str = "") -> None:
    """回复先判资料需求，再召回和确认；动态广播只检索公共素材。"""
    classifier_kwargs = {"context": context} if context is not None else {}
    if song_text:
        classifier_kwargs["song_text"] = song_text
    context = context or RetrievalContext(query_text, resolved_text=retrieval_query,
                                           history=history_text, is_user_msg=is_user_msg)
    l3 = {"behavior": "", "same_request": {"what": "", "count": 1},
          "reminder": None, "classification_ok": True}
    if is_user_msg:
        l3, query_vector = await asyncio.gather(
            classify_l3(deepseek_client, query_text, history_text, behaviors, **classifier_kwargs),
            embed_query(zhipu_client, context.query),
        )
        sr = l3.get("same_request") or {}
        if int(sr.get("count") or 1) >= 2:
            result["same_request"] = sr
        result["reminder"] = l3.get("reminder")
    else:
        query_vector = await embed_query(zhipu_client, context.query)
    result["l3"] = l3
    result["embedding_available"] = bool(query_vector)
    if is_user_msg and "needs_ok" in l3:
        context = replace(context, needs=tuple(l3.get("needs") or ()) if l3["needs_ok"] else (),
                          response_task=str(l3.get("task") or "")[:240] if l3["needs_ok"] else "",
                          response_scope=str(l3.get("scope") or "") if l3["needs_ok"] else "")
    result["context"] = context
    if not is_user_msg:
        # 动态改写广播没有某位好友的交流现场，保持原有公共素材取法。
        result["preference_items"] = retrieve_preferences(context.query, extra_ids=extra_preference_ids)
        result["core_stories"] = retrieve_core_stories(context.query, query_vector)
        result["denied_terms"] = None
        result.pop("approved_terms", None)
        result["selection_trace"] = {"available": True, "mode": "broadcast"}
        return
    behavior_name = l3.get("behavior", "")
    if l3.get("song_request") and behavior_name == "被推着要求时打太极":
        behavior_name = ""
    behavior = (select_behavior_item(query_text, behavior_name, behaviors,
                                    allow_fallback=l3.get("classification_ok") is False)
                if is_user_msg else None)
    # 成功的L3已按同一语境确认；失败后的关键词候选不能自行获得注入权限。
    semantic_behavior = [behavior] if behavior and l3.get("behavior") else []
    fallback = behavior if behavior and not l3.get("behavior") else None
    materials = await retrieve_contextual_materials(
        context, query_vector, aux_clients_or(lambda: (deepseek_client, None))[0],
        extra_preference_ids=extra_preference_ids, behavior_candidate=fallback)
    result.update(materials)
    behavior_items = semantic_behavior + materials["behavior_items"]
    corpus_items = materials["corpus_items"]
    if _detach_behavior_from_rrf():
        result["fused_items"] = fuse_and_truncate(corpus_items) + behavior_items
    else:
        result["fused_items"] = fuse_and_truncate(corpus_items, behavior_items)


def _reminder_request_note(req: dict | None) -> str:
    """把 L3 判出的提醒请求渲染成一条给主模型的短提示。

    - 时间明确：让她在回复里**明确确认一次**，这样 memory_v2 才会抽到 commitment；
    - 时间不明确：让她先反问，**不要**拿一个猜的时间去答应。
    """
    if not isinstance(req, dict) or not req.get("what"):
        return ""
    what = str(req.get("what") or "").strip()
    if req.get("ambiguous"):
        clarify = str(req.get("clarify") or "是几点提醒你？").strip()
        return (f"【提醒请求】用户想让你提醒他：{what}；时间不明确。"
                f"先在回复里反问（{clarify}），不要直接答应具体时间。")
    when = str(req.get("when_hint") or req.get("due_at") or "到点").strip()
    return (f"【提醒请求】用户想让你在 {when} 提醒他：{what}。"
            f"回复里明确确认一次（例如“好，{when}提醒你”），不要只回“嗯”。")


async def handle_chat(user_id: str, user_msg: str, vision_desc: str = "",
                      batch_summary: str = "", is_group: bool = False,
                      user_transcript: str = "", record_text: str = "",
                      group_addressed: bool = True, group_address_note: str = "",
                      group_thread_note: str = "", group_batch_note: str = "",
                      reply_to: str = "", *, timed_user_transcript: str = "",
                      defer_memory: bool = False, quoted_message_ids: tuple[str, ...] = (),
                      song_text: str | None = None) -> str:
    """处理一条用户消息，返回机器人回复。vision_desc 为图片描述；batch_summary 为批量归纳。

    is_group=True 时按群会话处理：user_id 传群号（会话历史按群记），
    不建用户记忆卡（群不是单个用户），回复对象由调用方按群发送。

    user_transcript 是本批完整文本（群聊带发言人和批内编号）。
    私聊的 timed_user_transcript 只用于模型请求的最后一条 user 消息。
    defer_memory 让私聊调用方在确认发送后写入实际发出的文本。
    record_text 可覆盖落盘的正文，供群聊去掉批次标记后写入记忆。

    group_addressed：这批群里有没有人点名/@ 她（接话门的确定性判定）。为 False 时，
    如果模型输出的是“没点我/保持潜水”这类**元回复**，视为这批不说话（只记用户侧、不发）。
    group_address_note：本批点名情况的系统元信息（见 build_message_list）；**只有点名时才有值**。
    group_thread_note：本批群聊对话线归纳（见 summarize_group_batch）。
    reply_to：她这句在回群里哪位（写进短期记忆的「灰泽满（回X）：」，只用于群聊）。
    """
    deferred_private = defer_memory and not is_group
    deepseek_client, zhipu_client = _get_clients()
    aux_client, _ = aux_clients_or(_get_clients)   # 辅助链（2026-10-29）：话题探测/直播记忆判走它

    # 短期记忆的说话人标签：群聊写成「群聊：甲：…」/「灰泽满（回甲）：…」，
    # 私聊保持原来的「用户：」/「灰泽满：」——历史一读就知道是群聊、她回的是谁，
    # 不会把好几个群友的消息糊成同一个“用户”（2026-10-23）。
    _user_label = "群聊" if is_group else "用户"
    _reply_label = f"灰泽满（回{reply_to}）" if (is_group and reply_to) else "灰泽满"

    # --- 会话级记忆：对话前同步探测（判断话题延续/转换 + 短 query 扩充） ---
    # 必须在组装消息前完成，这样本轮注入的就是本轮自己的话题，不滞后一轮。
    query_text = user_msg.strip()
    clock_question = need_triggers.asks_current_time(query_text)
    recall_enabled = not is_group and memory_v2.injection_enabled()
    history_question = recall_enabled and memory_recall.needs_recall(
        query_text, reference_ids=quoted_message_ids)
    v3 = injection_v3_enabled()
    # ⚠️ 带每条时间戳的版本（2026-10-05）：让模型看出"刚说完"和"三小时前说的"。
    #    两个消费者都用它：① 注入层（【最近对话记录】）② L3 判 same_request
    #    ——L3 必须知道那几轮之间隔了多久，否则隔 8 小时的一句会被算成"连着"。
    user_history_timed = get_user_history_timed(user_id)
    # 探测、L3、生成都用**去标签版**群聊历史：`灰泽满（回甲）` 的“回谁”只留在落盘日志里，
    # 不喂回模型——它会学成“回复先叫名字”（2026-11-01 用户指出根因在注释回喂）。
    history_for_model = _neutralize_group_history(user_history_timed) if is_group else user_history_timed
    # 探测和 L3 都看**完整**短期记忆（10 条 / 5 轮）；原来只给 8 条，
    # 短句/提醒要指代的上一句可能刚好落在被裁掉的那两条里（2026-10-20）。
    history_text = "\n".join(history_for_model) if history_for_model else ""
    show_actions = get_flag("ACTION_TAGS", "1").strip().lower() not in ("0", "false", "no", "off")
    song_context = ""
    song_kwargs = {}
    song_options = {}
    if not is_group and show_actions and song_text is not None:
        catalog = songs.load_songs()
        previous_reply = bot_line_text(history_for_model[-1]) if (
            history_for_model and is_bot_line(history_for_model[-1])) else ""
        song_context = inj.T("actions", "song_available" if catalog else "song_empty")
        if song_requests.may_request_song(song_text, catalog, previous_reply):
            song_kwargs["song_text"] = song_text
    # 距上一轮多久（此刻还没 append 本轮，最后一条就是上一轮）。同一场对话内为空串
    _gap_sec = get_last_turn_gap_seconds(user_id)
    history_gap_note = humanize_gap(_gap_sec) if _gap_sec is not None else ""
    # 上一场会话（已过期）→「上次聊过」提示。**必须在 probe_session 之前取**——
    # probe 会把 last_active 改写成"现在"，之后再问就永远判成"没过期"。
    _prev_session_note = previous_session_note(user_id, include_when=not v3)
    retrieval_query = query_text
    probe_meta: dict = {}
    if query_text:
        # debug=True 只是为了拿到**这次探测已经算出来的** entities（作品/作者/地点/动物）——
        # 不多调一次模型；生产路径的返回值仍是扩充后的检索 query。
        retrieval_query, probe_meta = await probe_session(
            user_id, query_text, history_text, aux_client, debug=True,
            is_group=is_group, group_addressed=group_addressed)
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
    _session_state = get_session(user_id)
    memory_scope_id = str(
        _session_state.get("topic_since")
        or _session_state.get("last_active")
        or f"turn-{int(time.time())}"
    )

    turn_context = RetrievalContext.from_probe(
        query_text, retrieval_query, history_text, probe_meta,
        is_group=is_group, addressed=group_addressed if is_group else True)

    # --- 🃏 经典梗：所有入口都要按同一份语境确认，关键词不能直接决定回复 ---
    _confirm_history = ""
    legendary_entries = () if history_question else LEGENDARY_REPLIES.items()
    for trigger, entry in legendary_entries:
        if legendary_hit(trigger, user_msg):
            replies = entry.get("replies") or []
            if not replies:
                continue
            confirm_tpl = LEGENDARY_CONFIRMS.get(trigger) or (
                f"判断用户本轮是否确实在向灰泽满触发「{trigger}」这类固定应答。\n"
                f"候选回复：{json.dumps(replies, ensure_ascii=False)}\n"
                "核对说的是谁、引用范围、否定和真实意图；不要因为出现关键词就确认。\n"
                "当前消息：{msg}\n最近对话：{context}\n只回答是或否。")
            if not _confirm_history:
                _confirm_history = "\n".join(get_user_history(user_id)[-4:])
            if not await legendary_confirmed(user_msg, confirm_tpl, history=_confirm_history,
                                               context=turn_context):
                continue
            reply = random.choice(replies)
            if deferred_private:
                result = Reply(reply)
                result.memory_context = {
                    "user_text": user_msg, "session_id": memory_scope_id, "reminder_hint": None,
                }
                return result
            # 梗匹配也记入短期记忆 + 异步长期记忆，避免后续对话"失忆"
            append_user_history(user_id, user_msg, reply,
                                user_label=_user_label, reply_label=_reply_label)
            if not is_group:  # 群会话不建用户记忆卡
                spawn(update_memory_v2_shadow_task(
                    user_id, user_msg, reply, memory_scope_id,
                ))
            return reply

    # 各路候选在共享入口一次确认；关联与百科也不能只凭专名就放行。
    ctx = await gather_retrieval(query_text, retrieval_query, history_text,
                                 deepseek_client, zhipu_client, context=turn_context, **song_kwargs)
    turn_context = ctx.get("context", turn_context)
    if song_kwargs and (song_request := (ctx.get("l3") or {}).get("song_request")):
        song_note, song_options = song_requests.plan_song(song_request, catalog, last_song(user_id))
        song_context += "\n" + song_note
    fused_items = ctx["fused_items"]
    preference_items = ctx["preference_items"]
    core_stories = ctx["core_stories"]
    reminder_req = ctx.get("reminder")

    wiki_context = ""
    if wiki_lookup_enabled() and ctx.get("wiki_entities"):
        try:
            wiki_context = await asyncio.wait_for(
                wiki_lookup.wiki_context_for(
                    query_text, ctx["wiki_entities"], _known_wiki_terms(),
                    context=turn_context, allowed_entities=ctx["wiki_entities"],
                    entity_mentions=ctx.get("entity_mentions")), timeout=6.0)
        except asyncio.TimeoutError:
            print("⚠️ wiki 查询超时（忽略，不阻塞回复）")
        except Exception as e:
            print(f"⚠️ wiki 查询异常（忽略）: {str(e)[:100]}")
    if wiki_context:
        print(f"[wiki] 注入背景: {wiki_context[:60]}")

    # --- 🧠 长期记忆：**只有 V2**（2026-10-12 起 v1 整个删除）---
    # v1 的卡（user_memory/long_term/*.json）连同 memory_manager.py 一起清掉了；
    # 名字/城市/长期事实全从 V2 取（见 _user_name_for / _weather_city_for）。
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
    weather_city = _weather_city_for(user_id)
    # v3 天气 owner=user 时，**这一句明确提到的地点优先于 V2 旧 location**
    # （如用户问“坡县几度”，probe 扩写成“新加坡…”，不能拿旧城市答）。
    weather_city_turn = weather_city
    if v3:
        weather_city_turn = (need_triggers.place_from_text(retrieval_query)
                             or need_triggers.place_from_text(query_text)
                             or weather_city)

    # 注入 v3：显式触发预取（系统侧，不占工具轮）；时间拆成 date/live 常驻 + 本场时段一次。
    prefetch_context = ""
    now_context = ""
    time_hint = ""
    if v3:
        try:
            prefetch_context, _ = await asyncio.wait_for(
                need_triggers.prefetch(
                    user_id=user_id, msg=query_text, store=_MEMORY_V2_STORE,
                    user_city=weather_city_turn, deepseek_client=deepseek_client,
                    zhipu_client=zhipu_client, is_group=is_group,
                ),
                timeout=3.0,
            )
        except Exception:
            prefetch_context = ""
        now_context = context_probe.frame_now()
        is_new_session = (not user_history_timed) or (_gap_sec is None) or (_gap_sec > 3 * 3600)
        time_hint = context_probe.session_time_hint(
            user_msg=query_text, is_new_session=is_new_session)

    # --- 🧠 稳定卡 / 事件 / 动态状态（2026-11-05）---
    # 稳定信息（称呼/身份/地点/相处/仪式/兴趣）每轮靠前；事件按指涉/实体/向量取；
    # 时间/天气/话题/情绪并进动态段放最后。旧的 profile/now/session/mood 不再单独注入，
    # 避免同一件事在提示词里出现两遍。
    state_stable = ""
    state_dynamic = ""
    card_text = ""
    event_context = ""
    profile_context = commitment_context = daily_context = ""
    request_context = ""
    emotion_context = ""
    memory_ids: list = []   # 本轮实际注入的记忆 id（注入轨迹用；群聊为 空）
    if not is_group and memory_v2.injection_enabled():
        if v3:
            card = _MEMORY_V2_STORE.build_state_card(user_id, query=turn_context.query)
            if card:
                card_text = card
                state_stable = inj.T("state", "stable", lines=card)
        else:
            profile_context = _MEMORY_V2_STORE.build_preference_context(
                user_id, retrieval_query, used_ids=memory_ids, include_facts=True)
            request_context = _MEMORY_V2_STORE.build_request_context(
                user_id, used_ids=memory_ids)
            if daily_memory.enabled():
                daily_context = daily_memory.context(user_id)
        commitment_context = _MEMORY_V2_STORE.build_commitment_context(
            user_id, retrieval_query, used_ids=memory_ids)
    # 事件背景：明确指涉过去/评价，或实体/话题命中时取 1–2 条；群聊不取。
    if not is_group:
        _event_query = (query_text or "") + "\n" + (retrieval_query or "")
        _explicit = bool(re.search(
            r"(上次|之前|那天|那次|还记得|记得|我说过|我们说好|你答应|你忘了|"
            r"以前|当时|评价|什么样的人|了解我)", _event_query))
        try:
            _events = await asyncio.wait_for(
                event_memory.retrieve(
                    user_id, _event_query,
                    entities=(probe_meta or {}).get("entities") if isinstance(probe_meta, dict) else None,
                    explicit=_explicit, zhipu_client=zhipu_client,
                    judge_client=deepseek_client, judge_model=_get_model_name(),
                    limit=2 if _explicit else 1,
                    avoid_text=history_text, context=turn_context,
                ),
                timeout=4.0,
            )
        except Exception:
            _events = []
        if _events:
            if v3 and card_text:
                # 同一件事既在稳定卡、又被事件检索捞到时：把时间并进卡里，事件不再重复注。
                card_text, _events = memory_v2.merge_events_into_card(card_text, _events)
                state_stable = inj.T("state", "stable", lines=card_text) if card_text else ""
            if _events:
                event_context = inj.T("events", "main", context=event_memory.render_context(_events))
    # 称呼证据：本轮直接称呼或追问以往叫法/答复时按词召回。
    nickname_hint = ""
    recall_context = ""
    matched_nicknames: list[str] = []
    if recall_enabled:
        nickname_turn_ids: list[str] = []
        nickname_hint = _MEMORY_V2_STORE.nickname_hint(
            user_id, user_msg, history=user_history_timed, matched_terms=matched_nicknames,
            source_turn_ids=nickname_turn_ids)
        recall_context = memory_recall.build_recall_context(
            _MEMORY_V2_STORE, user_id, user_msg,
            nicknames=matched_nicknames, nickname_turn_ids=nickname_turn_ids,
            reference_ids=quoted_message_ids,
        )
    if not is_group and memory_recall.is_commitment_question(user_msg):
        reminder_req = None
    # 称呼预算：最近几轮她叫对方名字太多 → 这轮前置提醒别再叫。
    # **只对私聊**：群聊没有“单独对某个人称呼”的语境，切进来只会让她更奇怪。
    name_budget = "" if is_group else name_budget_note(user_id)
    # 个人固定习惯：只对私聊、且这个用户确实有相关 habit_* 记忆时出现。
    habit_hint = ""
    ritual_hint = ""
    if not is_group and memory_v2.injection_enabled():
        recent_bot_text = "\n".join(
            bot_line_text(ln) for ln in get_user_history(user_id) if is_bot_line(ln)
        )
        habit_hint = _MEMORY_V2_STORE.build_habit_hint(
            user_id, query_text, recent_bot_text=recent_bot_text)
        if v3:
            # 仪式内容在稳定卡的「仪式」行；这里只在他这轮说到场景词时加一句“照做、别漏”。
            # 旧链路的【他的固定期待】带完整内容，但它没有迁到 v3（2026-11-18 补）。
            ritual_hint = _MEMORY_V2_STORE.build_ritual_hint(user_id, query_text)
    if not is_group and not v3:
        emotion_context = mood_note(user_id)

    if not is_group:
        # 词面触发只形成候选，避免引用/第三人台词强行触发个人仪式和回查。
        hints = await filter_memory_hints(aux_client, turn_context, {
            "commitment_context": commitment_context, "ritual_hint": ritual_hint,
            "habit_hint": habit_hint, "nickname_hint": nickname_hint, "recall_context": recall_context})
        commitment_context, ritual_hint, habit_hint, nickname_hint, recall_context = (
            hints.get(key, "") for key in ("commitment_context", "ritual_hint", "habit_hint", "nickname_hint", "recall_context"))

    # v3 关系收口：时效内容（约定/习惯/称呼预算/昵称提示）并成一条【本轮提醒】放用户消息前。
    reminders_context = ""
    if v3:
        if request_context:
            profile_context = (profile_context + "\n\n" + request_context).strip()
            request_context = ""
        parts = [x for x in (commitment_context, ritual_hint, habit_hint,
                             name_budget, nickname_hint) if x]
        if parts:
            reminders_context = "【本轮提醒】\n" + "\n".join(parts)
        commitment_context = ritual_hint = habit_hint = name_budget = nickname_hint = ""

    # 定时提醒请求（2026-11-17）：L3 判出后必须让她明确确认一次，
    # memory_v2 才会把这条当成 commitment 落库并带 due_at。
    reminder_note = _reminder_request_note(reminder_req)
    if reminder_note:
        reminders_context = (reminders_context + "\n" + reminder_note).strip() \
            if reminders_context else "【本轮提醒】\n" + reminder_note

    # 动态段：时间/天气/直播/话题/情绪；历史与连续索要稍后并进来。
    state_parts: list = []
    if v3:
        # 群聊也并进【当前状态】：时间/时段/本场话题；天气和情绪只对私聊有意义。
        if not is_group:
            try:
                await asyncio.wait_for(context_probe.warm_weather(weather_city_turn), timeout=1.5)
            except Exception:
                pass
        _frame = now_context or context_probe.frame_now()
        if _frame:
            state_parts.append(_frame.replace("【当前时间】", "", 1))
        if history_gap_note:
            state_parts.append(f"距上次聊天：{history_gap_note}前")
        if not is_group:
            _weather = context_probe.weather_now(weather_city_turn)
            if _weather:
                state_parts.append(_weather)
        if time_hint:
            state_parts.append("时段：" + time_hint.replace("【本场时段】", "", 1))
        if session_context:
            state_parts.append(session_context)
        if _prev_session_note:
            state_parts.append(_prev_session_note)
        # 她上一轮做过的动作（表情包/语音）——执行层写的回执，主动让她“知道”自己做过。
        _actions = actions_note(user_id)
        if _actions:
            state_parts.append(_actions)
        _voice_pref = voice_preference_note(user_id)
        if _voice_pref:
            state_parts.append(_voice_pref)
        if not is_group:
            _mood = mood_note(user_id)
            if _mood:
                state_parts.append("情绪：" + _mood.replace("【用户此刻的情绪】", "", 1))
        # 历史与连续索要稍后并进来；这里先不生成 state_dynamic。
        now_context = ""
        session_context = ""
        emotion_context = ""
        daily_context = ""
        time_hint = ""

    # --- 🧩 构建消息列表 ---
    # 天气预热：组消息前异步把城市 LocationID+天气取进缓存，build 时同步读缓存零阻塞
    # v3 的天气并进【当前状态】动态段，按需预热。
    if not v3:
        await context_probe.warm_weather(weather_city)
    # query_hint：短消息（≤4字）的语境扩充，仅当扩充句与原文不同时传入，帮模型理解短句
    query_hint = retrieval_query if (retrieval_query and retrieval_query != query_text) else ""
    # 语用注释（2026-10-31）：玩笑/反讽/预演这类“不是字面”的消息，单独给一条提示。
    # 与 query_hint 分开：它作用于任意长度，措辞也不能写成“短消息补全”。
    pragmatic_hint = ""
    _pa = str(probe_meta.get("speech_act") or "")
    _pn = str(probe_meta.get("pragmatic_note") or "").strip()
    _sc = str(probe_meta.get("scene") or "").strip()
    _st = str(probe_meta.get("implied_stance") or "").strip()
    try:
        _pc = float(probe_meta.get("pragmatic_confidence") or 0)
    except (TypeError, ValueError):
        _pc = 0.0
    if _pn and _pc >= 0.6 and _pa in ("tease", "anticipation", "roleplay", "other") \
            and not (is_group and not group_addressed):
        # ⚠️ 群聊未点名时**禁用语用注释**：probe 容易把群友之间的「你/她/这事」
        #    误判成在说灰泽满（2026-10-31 967761489 22:36 实锤：
        #    「还关门/有什么我们不能听的吗」被注成「调侃灰泽满背着大家做什么」）。
        pragmatic_hint = ((f"场景：{_sc}。" if _sc else "")
                          + _pn
                          + (f"（更自然的反应：{_st}）" if _st else ""))
        print(f"[语用] {_pa} → {pragmatic_hint[:60]}")
    # 术语语境确认：confirm:true 的命中做一次便宜 LLM 判断，剔除误触词条（词→意思守卫）
    denied_terms = ctx.get("denied_terms")
    # 条目级注入轨迹：build 时填"实际注入了哪些块/条目"，回复生成后再补称呼是否被用，
    # 由下面统一落盘（只有 私/群 真实聊天路径写，和 block_hits 同一个闸）。
    injection_trace: dict = {}
    if ctx.get("selection_trace"):
        injection_trace["retrieval"] = ctx["selection_trace"]
    if pragmatic_hint:
        injection_trace.setdefault("values", {})["pragmatic"] = pragmatic_hint[:200]
    # 历史注入（2026-11-19）：私聊默认走 assistant 原生轮次，不再逐条转述。
    # 只去掉「历史转述」；用户当前消息里引用她的话仍由 annotate_quotes 改写。
    # 群聊保持原样（多发言人 + 转述），回退设 HISTORY_AS_ASSISTANT=0。
    use_native_history = (not is_group) and history_as_assistant_enabled()
    history_turns = build_native_history_turns(user_id) if use_native_history else None
    hist_for_injection, gen_user_msg = user_history_timed, user_msg
    if use_native_history:
        if self_speech.enabled():
            gen_user_msg = await self_speech.annotate_quotes(user_msg, deepseek_client)
        hist_for_injection = []
    elif self_speech.enabled():
        hist_for_injection, gen_user_msg = await self_speech.annotate_turn(
            user_history_timed, user_msg, deepseek_client)
    if is_group:
        # annotate_turn 会把 `（回X）` 换成 `（转述）`；若它没启用/失败，这里再兜底去一次。
        hist_for_injection = _neutralize_group_history(hist_for_injection)
    # v3 私聊：最近对话 + 连续索要并进【当前状态】动态段，和当前话题放在一起。
    history_arg = hist_for_injection
    same_request_arg = ctx.get("same_request")
    if v3:
        if isinstance(hist_for_injection, list):
            _hist_text = "\n".join(ln for ln in hist_for_injection if ln)
        else:
            _hist_text = str(hist_for_injection or "")
        if _hist_text:
            state_parts.append(inj.T("history", context=_hist_text)
                              + inj.T("history", "contradiction"))
            history_arg = []
        _sr = same_request_arg or {}
        if int(_sr.get("count") or 1) >= 2:
            state_parts.append(inj.T(
                "same_request", count=_sr["count"],
                what=_sr.get("what") or "同一件事", prev=_sr["count"] - 1))
            same_request_arg = None
        if state_parts:
            state_dynamic = inj.T("state", "dynamic", parts="\n".join(state_parts))
    messages = build_message_list(
        gen_user_msg, fused_items, history_arg,
        history_turns=history_turns,
        vision_desc=vision_desc, weather_city=weather_city, batch_summary=batch_summary,
        preference_items=preference_items,
        core_stories=core_stories,
        session_context=session_context, query_hint=query_hint,
        pragmatic_hint=pragmatic_hint,
        denied_terms=denied_terms, selected_terms=ctx.get("approved_terms"), group_context=group_context,
        history_gap_note=history_gap_note,
        prev_session_note="" if v3 else _prev_session_note,
        same_request=same_request_arg,
        profile_context=profile_context, commitment_context=commitment_context,
        emotion_context=emotion_context, daily_context=daily_context,
        user_transcript=user_transcript, nickname_hint=nickname_hint,
        log_hits=True, trace=injection_trace,
        association_context=ctx.get("association_context", ""),
        association_entities=ctx.get("association_entities"),
        wiki_context=wiki_context,
        is_group=is_group, group_address_note=group_address_note,
        group_thread_note=group_thread_note, name_budget=name_budget,
        group_batch_note=group_batch_note,
        habit_hint=habit_hint, now_context=now_context, time_hint=time_hint,
        prefetch_context=prefetch_context, recall_context=recall_context,
        request_context=request_context, reminders_context=reminders_context,
        state_stable=state_stable, state_dynamic=state_dynamic,
        event_context=event_context,
        show_actions=show_actions, song_context=song_context,
    )

    # --- 🤖 调用大模型 ---
    if timed_user_transcript:
        messages[-1]["content"] = timed_user_transcript
    reply = await generate_reply(messages)

    # --- 🔁 复读机防护 ---
    # 模型会对情境相关句执着复读：自己上轮的话进短期记忆后，用户把同一抱怨又说一遍时，
    # 它会整段照搬上上轮的解释（隔几句也一样，故窗口放宽到 8，不只最近 3）。
    # 检测到就强制"换动作"重生成：光"换个说法"不够，得 pivot——
    # 对方大概率在重复同一句/同一情绪，应认怂答应去做/自嘲/点破，而不是把解释再说一遍。
    recent_bot = [bot_line_text(ln) for ln in get_user_history(user_id) if is_bot_line(ln)]
    # 用户明确问时间时，同一分钟的答案可以和上一轮相同。
    if not clock_question and is_echo_reply(reply, recent_bot, window=8):
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

    # 「时间钩子」重复：熬夜场景下她反复拿"现在几点了/还不睡"当开头（逐字复读抓不到——
    # 说法不同、公共子串不够长）。触发条件很窄：**这条提了时间 且 最近两条里也提过**。
    # 只换一句更具体的提醒，重生 2 次；判据本身只认时间词，不碰她别的口癖。
    if not clock_question and uses_time_hook(reply) and any(
            uses_time_hook(prev) for prev in recent_bot[-2:]):
        print(f"[防复读] 又拿时间当切入点『{reply[:20]}』，换角度")
        nudge = {
            "role": "system",
            "content": "你最近几条回复都在拿'几点了/还不睡'当开头（'凌晨三点…'那类）。"
                       "对方已经听过了，再说一遍很腻。这条**不许提时间**、也不要点评对方熬夜，"
                       "换一个完全不同的切入点重新回复。",
        }
        for _ in range(2):
            candidate = await generate_reply(list(messages) + [nudge])
            reply = candidate
            if not (uses_time_hook(reply) and any(uses_time_hook(prev) for prev in recent_bot[-2:])):
                break

    # 「固定模板」重复：`灰泽满才没…` / `突然说这个…` 是**模型级模板**（2026-10-20 A/B：
    # 同一批 prompt 下 gemini-3.1 10/20 出现、gemini-3.8 0/20、deepseek 1/20），
    # 不是 behaviors 例句。只在**最近 4 条里有 2 条同模板**时才拦，单条不管。
    opener = repeated_formula(reply, recent_bot)
    if opener:
        print(f"[防复读] 固定模板『{opener}…』连续出现，换说法")
        nudge = {
            "role": "system",
            "content": f"你最近几条回复都在用『{opener}…』这个固定句式，对方已经听腻了。"
                       f"这条**不要用这个句式开场**，换一个动作：直接承认、反问、换话题、"
                       f"或者自嘲一句。重新回复这条消息。",
        }
        for _ in range(2):
            candidate = await generate_reply(list(messages) + [nudge])
            reply = candidate
            if not repeated_formula(reply, recent_bot):
                break

    # 「名字开头」重复：名字进记忆只是"种子"，真正让它停不下来的是**正反馈**——她的回复进
    # 【最近对话记录】→ 被当样本学回去。实测某用户前 100 轮 0 次、第 103 轮（用户第一次说出
    # 那个名字）起一路涨到最近 14 轮 14/14。触发很窄：**这条以名字开头 且 最近两条里也有一条**。
    user_name = _user_name_for(user_id)
    if user_name and starts_with_name(reply, user_name) and \
            any(starts_with_name(prev, user_name) for prev in recent_bot[-2:]):
        print(f"[防复读] 又拿名字当开头『{reply[:20]}』，换起手")
        nudge = {
            "role": "system",
            "content": f"你最近几条回复都拿对方的名字（{user_name}）当开头，太频繁了、像客服。"
                       f"这条**不要用名字开头**，直接从内容说起。",
        }
        for _ in range(2):
            candidate = await generate_reply(list(messages) + [nudge])
            reply = candidate
            if not (user_name and starts_with_name(reply, user_name)):
                break

    # 「看看」空壳起手：与名字/时间同源——**自强化**出来的固定开头。
    # 实测（2026-10-12 某用户 537 条里 108 条）退化轨迹："看看时间"（正常）→"看看这个"→"看看你"
    # →"看看，"（没宾语了），一旦开始就成串（最长连 29 条）。触发很窄：**这条以"看看"起手
    # 且最近两条里也有一条**——在这个用户身上能拦下 92%。
    if uses_hollow_opener(reply) and any(uses_hollow_opener(prev) for prev in recent_bot[-2:]):
        print(f"[防复读] 又拿'看看'当起手『{reply[:20]}』，换个开头")
        nudge = {
            "role": "system",
            "content": "你最近几条回复都拿'看看…'当开头（'灰泽满看看时间''灰泽满看看你'那类），"
                       "对方已经看腻了。这条**不要用'看看'起头**，直接从内容说起。",
        }
        for _ in range(2):
            candidate = await generate_reply(list(messages) + [nudge])
            reply = candidate
            if not (uses_hollow_opener(reply) and any(uses_hollow_opener(prev) for prev in recent_bot[-2:])):
                break

    # 「太长」重说一遍已于 2026-11-05 删除（用户定：长度交给模型自己判断）。
    # 原因见 constants.CHAT_MAX_TOKENS 的注释：守卫会把明确要展开的回答整段丢弃重写，
    # 真实聊天里变成"我要说了→其实什么都没说"；默认短回复由骨架 §06 定基调即可。

    # --- 🧾 条目级注入轨迹（`docs/数据迭代.md` §2）---
    # 闸和 block_hits 一致：只有真实聊天（通道 私/群）写，离线回放/实验不污染。
    if get_llm_route() in ("私", "群"):
        ids = list(dict.fromkeys(str(i) for i in memory_ids if i))   # 去重保序
        if ids:
            injection_trace.setdefault("items", {})["memory"] = ids
        injection_trace["session"] = str(user_id)
        injection_trace["route"] = get_llm_route()
        if v3 and matched_nicknames:
            injection_trace.setdefault("blocks", []).append("nickname")
        injection_trace["reply_used_nickname"] = any(nickname in reply for nickname in matched_nicknames)
        log_injection_trace(injection_trace)

    # --- 💾 更新短期记忆（带锁）：图片消息把视觉描述记进去，后续才记得聊过什么图 ---
    # 存**清洗后**的版本（clean_reply 平时在 chat_window 里、本函数返回之后才跑）：
    # 否则她带换行/多括号的原始输出会存进记忆，再作为"她自己怎么说话"的 few-shot 喂回去，
    # 变成自我强化回路——后处理每次都得再洗一遍，坏习惯却一直被喂养。
    record_msg = record_text or user_transcript or _compose_record_msg(user_msg, vision_desc)
    if is_fallback_reply(reply):
        # 这一轮压根没生成出来（上游异常），**整轮都不记**：
        # 记了就等于让她"记得自己说过"一句故障文案，还会当 few-shot 喂回去。
        # 用户的这句话也随之不留——但她这轮并没有真的回应过，不记比记错干净。
        print("[记忆] 本轮是兜底回复（生成失败），不写入短期/长期记忆")
        return reply
    # 动作标记（[[语音]] / [[表情:类别]]）先剥掉：它绝不能进记忆/转述/发送。
    # 剥出来的意图挂到 Reply 上，由 chat_window 执行（语音/表情包）。
    no_tag_reply, actions = extract_action_tags(reply)
    if "song" in actions:
        delivery = song_options.get(actions.pop("song"))
        if delivery is not None and not song_requests.declined_song(no_tag_reply, delivery.mode):
            actions["song"] = delivery
        elif not no_tag_reply and not is_group:
            no_tag_reply = ("这段没选好，你想听哪首呀？" if song_options
                            else "刚刚那句没回好，能再说一下吗？")
    # 未点名的群聊批次：模型如果把内部标记说成了回复（「没点我」「（不说话，潜水）」），
    # 视为这批她不说话——只记用户侧，不发。点名后的“关我什么事”是正常斗嘴，不拦。
    if is_group and not group_addressed and is_meta_silence(clean_reply(no_tag_reply)):
        print("[群聊] 模型输出的是“没点名/潜水”式元回复 → 这批不发，只记用户侧")
        append_user_history(user_id, record_msg, "", user_label=_user_label)
        return ""
    result = Reply(no_tag_reply, actions)
    if deferred_private:
        result.memory_context = {
            "user_text": record_msg, "session_id": memory_scope_id, "reminder_hint": reminder_req,
        }
        return result
    cleaned_reply = clean_reply(no_tag_reply)
    append_user_history(user_id, record_msg, cleaned_reply,
                        user_label=_user_label, reply_label=_reply_label)
    # 预转述她这句：下一轮注入直接命中缓存，把转述开销挪出关键路径。
    # 私聊已切 assistant 原生轮次，不再需要；群聊仍走转述，保留预转述。
    if self_speech.enabled() and (is_group or not history_as_assistant_enabled()):
        spawn(self_speech.prewarm(cleaned_reply))

    # --- 📝 异步更新长期记忆（V2；会话级记忆已在对话前 probe_session 同步更新） ---
    if not is_group:  # 群会话不建用户记忆卡
        spawn(update_memory_v2_shadow_task(
            user_id, record_msg, no_tag_reply, memory_scope_id,
            reminder_hint=reminder_req,
        ))

    return result
