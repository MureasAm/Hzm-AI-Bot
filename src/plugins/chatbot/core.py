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

from .constants import (
    SYSTEM_PROMPT_FILE, THINKING_DISABLED,
    CHAT_TEMPERATURE, CHAT_FREQUENCY_PENALTY, CHAT_MAX_TOKENS,
    LENGTH_GUARD_MAX_CHARS,
)
from .persona import (
    load_persona_rules, load_schedule,
)
from .memory import (
    get_user_history, get_user_history_timed, append_user_history,
    get_last_turn_gap_seconds, humanize_gap,
)
from .rag import embed_query
from .retrieval import (
    retrieve_corpus_candidates,
    retrieve_preferences, retrieve_core_stories, fuse_and_truncate, select_behavior_item,
    resolve_associations,
)
from .corpus_judge import judge_corpus
from .chatlog import log_corpus_hits, log_block_hits, log_injection_trace
from .config import get_llm_route
from .constants import SPLIT_DELAY_MIN_MS, SPLIT_DELAY_MAX_MS  # noqa: F401 — test_core 校验分段延迟上下限时读它
from . import context_probe
from . import group_memory
from . import injection as inj
from . import daily_memory
from . import memory_v2
from .routing import (
    LEGENDARY_REPLIES, LEGENDARY_CONFIRMS, legendary_confirmed, legendary_hit, classify_l3,
)
from .reply_style import (
    clean_reply, is_echo_reply, is_emotion_only_query, uses_time_hook, starts_with_name,
    uses_hollow_opener, repeated_formula,
)
from .session_memory import (
    probe_session, build_session_context, get_session, is_emoji_msg, previous_session_note,
    mood_note,
)


_MEMORY_V2_STORE = memory_v2.ShadowMemoryStore()


# ==================== 🎭 基础人设提示词 ====================
if SYSTEM_PROMPT_FILE.exists():
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    raise FileNotFoundError(f"❌ 未找到 {SYSTEM_PROMPT_FILE}")

# ==================== 🛠️ API 客户端（惰性初始化） ====================
# 挪到了 config.py（基础设施层）。core 从这里 import，routing 也从 config 取，避免循环依赖。
from .config import log_cache_usage, _get_clients, _get_model_name, extract_chat_content  # noqa: E402


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
    try:
        deepseek_client, _ = _get_clients()
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "批量归纳")
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
                     hit_out: list | None = None) -> str:
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
            if hit_out is not None:
                hit_out.append(kw)
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
        log_cache_usage(resp, "术语消歧")
        content = extract_chat_content(resp)
        if "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        exclude = set(data.get("exclude", []))
        return {t["keyword"] for t in candidates if t["keyword"] in exclude}
    except Exception as e:
        print(f"⚠️ 术语语境确认失败（放行）: {e}")
        return set()


def build_message_list(user_msg: str, fused_items: list,
                       user_history: list,
                       vision_desc: str = "", weather_city: str = "",
                       batch_summary: str = "", preference_items: list = None,
                       core_stories: list = None, session_context: str = "",
                       query_hint: str = "", denied_terms: set | None = None,
                       group_context: str = "", history_gap_note: str = "",
                       prev_session_note: str = "", same_request: dict = None,
                       profile_context: str = "", commitment_context: str = "",
                       emotion_context: str = "", daily_context: str = "",
                       user_transcript: str = "", nickname_hint: str = "",
                       log_hits: bool = False, trace: dict | None = None,
                       association_context: str = "", association_entities: list | None = None,
                       is_group: bool = False) -> list:
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
    """
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
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

    # 场景（第 00 块）：私聊/群聊**二选一**，紧跟骨架、在周表之前。
    # 从骨架里搬出来的是"一对一"那句——群聊里说"你"前必须知道有几个人在场。
    scene_key = "group" if is_group else "private"
    messages.append({"role": "system", "content": inj.T("scene", scene_key)})
    hit_blocks.append(f"scene_{scene_key}")
    _mark(f"scene_{scene_key}")

    # 周表（地面真值）：被问"明天来吗/这周/几点播"以它为准。
    # 记忆里带"明天/下周"的话是过去某场直播当时的说法，可能早过期，不能当现在的安排。
    # 这里**只有 weekly**（weekday 制，到周自动对，永不过期），没有会过期的自由文本字段。
    sched = load_schedule()
    weekly = sched.get("weekly") if sched else None
    if weekly:
        messages.append({"role": "system", "content": inj.T("schedule", weekly=inj.format_weekly(weekly))})
        _mark("schedule")

    # Memory V2【靠前】相处方式：偏好→行为句 + 话题相关事实（个体适配，优先级最高）。
    # 位置靠前是有意的：它是"针对这个人的长期适配"，不该被后面的通用素材盖过。
    if profile_context:
        messages.append({"role": "system", "content": profile_context})
        _mark("profile")

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

    # 名词库（terms/lorebook）：核心词 always 注入 + 命中用户消息的词注入——让模型懂"绿冻/枪神8/slg"这类词并带对的态度
    _term_hits: list = []
    terms_note = build_terms_note(user_msg, denied_terms=denied_terms,
                                  hit_out=_term_hits if trace is not None else None)
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
    if user_history:
        if isinstance(user_history, list):
            context = "\n".join(ln for ln in user_history if ln)
            if context:
                # ⚠️ 时间戳**不再**在这里加一行汇总（2026-10-05 改）——
                #   旧做法是把「距上一轮多久」当一行汇总拼在整块开头，结果 gap=10秒 和 gap=3小时
                #   注入出来的历史**长得一模一样**，她分不出"刚说完"和"三小时前说的"。
                #   现在改成：**每条消息自己带 [多久前]**（由 memory.get_user_history_timed 渲染），
                #   只给事实、不给"该怎么理解"的说明，判断交给模型。
                #   history_gap_note 参数保留（调用方还在传），但不再拼进历史块。
                text = inj.T("history", context=context) + inj.T("history", "contradiction")
                messages.append({"role": "system", "content": text})
                hit_blocks.append("history")
                _mark("history")
        else:
            messages.append({"role": "system",
                             "content": inj.T("history", "fallback", history=user_history)})
            _mark("history")

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
    if query_hint:
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

    # 感知源①：当前时间/农历/天气（始终注入，占预算极少；天气按该用户所在城市）。
    # **放在最后**：它是唯一"每分钟必变"的块，而 DeepSeek 的缓存按**前缀**匹配——
    # 它放中间会把后面所有块（行为/直播记忆/长期短期记忆…）一起拖成 miss。
    # 语义上也更顺：紧挨着用户这句，"现在几点"最贴近此刻。
    now_context = context_probe.get_now_context(city=weather_city)
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


def length_guard_enabled() -> bool:
    """长度守卫开关：环境变量 LENGTH_GUARD=0 可关掉（快速回滚用），缺省开。"""
    return os.environ.get("LENGTH_GUARD", "1") != "0"


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
        # ⚠️ 主回复是**最大的 token 消费者**，但以前漏了插桩——日志里那三行 `主回复`
        #    其实是批量归纳/转发归纳/术语消歧三个小助手（2026-10-11 查明）。
        #    不打出这一行，"缓存到底命中多少"根本没法归因。
        log_cache_usage(response, "主回复")
        reply = extract_chat_content(response)
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
) -> list[dict]:
    """Extract one private turn into the opt-in V2 shadow store.

    This path is deliberately fail-closed and has no influence on the reply.
    Keeping the feature check inside the task also makes direct/offline callers
    safe when the environment flag is absent.
    """
    if not memory_v2.shadow_enabled():
        return []
    msg = (user_msg or "").strip()
    if not msg or is_emoji_msg(msg):
        return []
    try:
        deepseek_client, _ = _get_clients()
        return await memory_v2.extract_and_ingest(
            client=deepseek_client,
            model=_get_model_name(),
            store=_MEMORY_V2_STORE,
            user_id=user_id,
            user_text=user_msg,
            assistant_text=reply,
            session_id=session_id,
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

# 名字记忆的"外壳词"：V2 的值常常被写成整句（"用户希望被称呼为草草子哥""希望被叫小温"），
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
    后面多半是"不叫hikami"这类补充说明；② 反复剥掉外壳词。剥不干净就返回原样
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


def _user_name_for(user_id: str) -> str:
    """用户希望被怎么称呼——用来判"她是不是又在拿名字当开头"。

    只从 V2 的 preferred_name/name/nickname 取。V2 的值常常是一整句
    （"用户希望被称呼为草草子哥，或者hikami"），所以取第一个逗号前、再剥掉外壳、
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


async def gather_retrieval(query_text: str, retrieval_query: str, history_text: str,
                           deepseek_client, zhipu_client,
                           is_user_msg: bool = True,
                           extra_preference_ids: list | None = None) -> dict:
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
    behaviors = load_persona_rules()
    result = {"fused_items": [],
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
                              deepseek_client, zhipu_client, behaviors, is_user_msg,
                              extra_preference_ids)
    except Exception as e:
        # **检索层炸了不能让她变哑巴**：素材按"没检索到"处理，回复/主动发言继续。
        # 踩坑（2026-09-27）：智谱 embedding key 过期 → 检索层抛
        # TypeError('NoneType' object is not iterable)，而调用点没有兜底 →
        # **整条回复任务死掉，她一个字都不回**（含群聊，看起来像"不接话"）。
        # 兜底放这里=两条链路（聊天/主动发言）一起受保护。
        print(f"⚠️ 检索层异常（按'没检索到'继续）: {e}")
    return result


async def _fill_retrieval(result: dict, query_text: str, retrieval_query: str, history_text: str,
                          deepseek_client, zhipu_client, behaviors, is_user_msg: bool,
                          extra_preference_ids: list | None = None) -> None:
    """`gather_retrieval` 的干活部分（单独一层，方便上面统一兜底）。结果写进 result。"""
    behavior_items, corpus_items = [], []
    # L3：归属用 LLM 判意图（不再用 embedding 猜——embedding 按句式聚团，
    # 会把'灰泽满你唱歌好听'（夸）和'灰泽满你怎么又迟到'（质问）挤在一起误判）。
    # 同一次调用还判"连续第几轮在要同一件事"（same_request）。
    # 判别词（敷衍/骗/鸽/迟到/黄桃/擦边…）退**兜底**位：L3 判不出来时才用。
    if is_user_msg:
        l3 = await classify_l3(deepseek_client, query_text, history_text, behaviors)
        if l3["behavior"]:
            print(f"[行为] LLM 判定: {l3['behavior']}")
        sr = l3.get("same_request") or {}
        if int(sr.get("count") or 1) >= 2:
            result["same_request"] = sr
            print(f"[连续索要] 第 {sr['count']} 轮：{sr.get('what') or '（未说明）'}")
        behavior_item = select_behavior_item(query_text, l3["behavior"], behaviors)
        behavior_items = [behavior_item] if behavior_item else []

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

    if _detach_behavior_from_rrf():
        # 行为**不进 RRF**：它是 L3 判出类别后**按名字查表**的结果（没有相似度、
        # 分数写死 1.0），不是"从一堆里挑最像的"那种检索。RRF 是给检索造的（融合多路排名），
        # 把查表结果丢进去，它会**占 top-6 名额和 1200 字预算**。
        # → 截断**之后**再追加，永不参与名额竞争；名额全还给 corpus。
        # 注入位置由 `_split_fused` 按段类型决定，**不受这里顺序影响**（见 build_message_list）。
        result["fused_items"] = fuse_and_truncate(corpus_items) + behavior_items
    else:
        result["fused_items"] = fuse_and_truncate(corpus_items, behavior_items)
    # 第 5 路：偏好（关键词命中）／核心记忆（结晶）
    # 关联边（has_preference）指过来的偏好排最前——"顺着作品/作者落到她的喜好上"比碰巧撞词更该带
    result["preference_items"] = retrieve_preferences(
        retrieval_query or query_text, extra_ids=extra_preference_ids)
    result["core_stories"] = retrieve_core_stories(retrieval_query or query_text, query_vector)


async def handle_chat(user_id: str, user_msg: str, vision_desc: str = "",
                      batch_summary: str = "", is_group: bool = False,
                      user_transcript: str = "", record_text: str = "") -> str:
    """处理一条用户消息，返回机器人回复。vision_desc 为图片描述；batch_summary 为批量归纳。

    is_group=True 时按群会话处理：user_id 传群号（会话历史按群记），
    不建用户记忆卡（群不是单个用户），回复对象由调用方按群发送。

    user_transcript = **给模型看的**本批完整文本（群聊带 [对灰泽满]/[未点名] 标记）；
    record_text = **记进短期记忆的**版本（群聊是干净的「昵称：文本」）——两者分开，
    免得批次级标记被当成对话历史喂回去。
    """
    deepseek_client, zhipu_client = _get_clients()

    # --- 会话级记忆：对话前同步探测（判断话题延续/转换 + 短 query 扩充） ---
    # 必须在组装消息前完成，这样本轮注入的就是本轮自己的话题，不滞后一轮。
    query_text = user_msg.strip()
    # ⚠️ 带每条时间戳的版本（2026-10-05）：让模型看出"刚说完"和"三小时前说的"。
    #    两个消费者都用它：① 注入层（【最近对话记录】）② L3 判 same_request
    #    ——L3 必须知道那几轮之间隔了多久，否则隔 8 小时的一句会被算成"连着"。
    user_history_timed = get_user_history_timed(user_id)
    history_text = "\n".join(user_history_timed[-8:]) if user_history_timed else ""
    # 距上一轮多久（此刻还没 append 本轮，最后一条就是上一轮）。同一场对话内为空串
    _gap_sec = get_last_turn_gap_seconds(user_id)
    history_gap_note = humanize_gap(_gap_sec) if _gap_sec is not None else ""
    # 上一场会话（已过期）→「上次聊过」提示。**必须在 probe_session 之前取**——
    # probe 会把 last_active 改写成"现在"，之后再问就永远判成"没过期"。
    _prev_session_note = previous_session_note(user_id)
    retrieval_query = query_text
    probe_meta: dict = {}
    if query_text:
        # debug=True 只是为了拿到**这次探测已经算出来的** entities（作品/作者/地点/动物）——
        # 不多调一次模型；生产路径的返回值仍是扩充后的检索 query。
        retrieval_query, probe_meta = await probe_session(
            user_id, query_text, history_text, deepseek_client, debug=True)
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
                asyncio.create_task(update_memory_v2_shadow_task(
                    user_id, user_msg, reply, memory_scope_id,
                ))
            return reply

    # --- 🔗 关联背景（作品 → 作者 → 喜好）---
    # 用**用户原话**匹配已知实体，再补上本轮 probe 识别出的专名（未知作品 → 作者/地点）。
    # 必须放在检索前：边上 has_preference 指向的偏好要并进偏好检索（如 东野 → author）。
    # 识别不到实体、或走完没有可说的关系 → note 为空，后面什么都不注。
    _assoc = resolve_associations(query_text, probe_meta.get("entities"))
    if _assoc["note"]:
        print(f"[关联] {_assoc['entities']} → {_assoc['note'][:40]}")

    # --- 🎭 人格规则 + 🔍 检索融合（query 只算 1 次 embedding）---
    # 素材统一从 gather_retrieval 取（主动发言也走同一个入口，别再各拼一套）
    ctx = await gather_retrieval(query_text, retrieval_query, history_text,
                                 deepseek_client, zhipu_client,
                                 extra_preference_ids=_assoc["preference_ids"])
    fused_items = ctx["fused_items"]
    preference_items = ctx["preference_items"]
    core_stories = ctx["core_stories"]

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

    # --- 🧠 Memory V2 注入（可选；与影子提取分开的开关，默认关，见 docs/记忆v2设计.md §上线路径）---
    # 两段分开放：相处方式靠前 / 未完成约定中段。
    profile_context = commitment_context = daily_context = ""
    memory_ids: list = []   # 本轮实际注入的记忆 id（注入轨迹用；群聊为 空）
    if not is_group and memory_v2.injection_enabled():
        profile_context = _MEMORY_V2_STORE.build_preference_context(
            user_id, retrieval_query, used_ids=memory_ids)
        commitment_context = _MEMORY_V2_STORE.build_commitment_context(
            user_id, retrieval_query, used_ids=memory_ids)
        if daily_memory.enabled():
            daily_context = daily_memory.context(user_id)
    # 称呼提示：用户这轮叫了 her_nickname 才注入；不做长期自称偏好。
    nickname_hint = ""
    if not is_group and memory_v2.injection_enabled():
        nickname_hint = _MEMORY_V2_STORE.nickname_hint(user_id, user_msg)
    # 当下情绪住在**会话记忆**里（2026-10-10 从长期记忆搬来），跟话题一起 3h 消散。
    # 它**不受 v2 开关控制**——情绪是会话级感知，生产里就该生效；注入位置仍是紧挨用户消息。
    emotion_context = "" if is_group else mood_note(user_id)

    # --- 🧩 构建消息列表 ---
    # 天气预热：组消息前异步把城市 LocationID+天气取进缓存，build 时同步读缓存零阻塞
    await context_probe.warm_weather(weather_city)
    # query_hint：短消息（≤4字）的语境扩充，仅当扩充句与原文不同时传入，帮模型理解短句
    query_hint = retrieval_query if (retrieval_query and retrieval_query != query_text) else ""
    # 术语语境确认：confirm:true 的命中做一次便宜 LLM 判断，剔除误触词条（词→意思守卫）
    denied_terms = await confirm_ambiguous_terms(user_msg, deepseek_client)
    # 条目级注入轨迹：build 时填"实际注入了哪些块/条目"，回复生成后再补称呼是否被用，
    # 由下面统一落盘（只有 私/群 真实聊天路径写，和 block_hits 同一个闸）。
    injection_trace: dict = {}
    messages = build_message_list(
        user_msg, fused_items, user_history_timed,
        vision_desc=vision_desc, weather_city=weather_city, batch_summary=batch_summary,
        preference_items=preference_items, core_stories=core_stories,
        session_context=session_context, query_hint=query_hint,
        denied_terms=denied_terms, group_context=group_context,
        history_gap_note=history_gap_note, prev_session_note=_prev_session_note,
        same_request=ctx.get("same_request"),
        profile_context=profile_context, commitment_context=commitment_context,
        emotion_context=emotion_context, daily_context=daily_context,
        user_transcript=user_transcript, nickname_hint=nickname_hint,
        log_hits=True, trace=injection_trace,
        association_context=_assoc["note"], association_entities=_assoc["entities"],
        is_group=is_group,
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

    # 「时间钩子」重复：熬夜场景下她反复拿"现在几点了/还不睡"当开头（逐字复读抓不到——
    # 说法不同、公共子串不够长）。触发条件很窄：**这条提了时间 且 最近两条里也提过**。
    # 只换一句更具体的提醒，重生 2 次；判据本身只认时间词，不碰她别的口癖。
    if uses_time_hook(reply) and any(uses_time_hook(prev) for prev in recent_bot[-2:]):
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

    # 「太长」重说一遍（2026-10-14）——**长度是模型内的老问题，这是那个"可靠解法"**。
    # 为什么不是提示词：骨架里"默认往短里走"那段只能定基调，实测管不住（换模型后尤其明显：
    # gemini 默认 36 字，而她历史上的水平是 ~29）。为什么不是 max_tokens：硬压会腰斩半句。
    # 实测（20 条真实消息，阈值 40 字）：gemini 35.7 → **25.4 字**（≈历史水平），只动 1/4 的轮次；
    # ds 一句都没超、完全不触发。形状照上面那几道闸：确定性触发 + 追加提醒 + 重生成。
    if length_guard_enabled() and len(reply) > LENGTH_GUARD_MAX_CHARS:
        print(f"[长度] {len(reply)} 字超过 {LENGTH_GUARD_MAX_CHARS}，带提醒重说一遍")
        nudge = {
            "role": "system",
            "content": "你刚才那句太长了。**重说一遍，短一半**：只留最要紧的那一句，"
                       "别丢信息、别丢态度，也别补新的话。",
        }
        for _ in range(2):
            candidate = await generate_reply(list(messages) + [nudge])
            reply = candidate
            if is_fallback_reply(reply) or len(reply) <= LENGTH_GUARD_MAX_CHARS:
                break

    # --- 🧾 条目级注入轨迹（`docs/数据迭代.md` §2）---
    # 闸和 block_hits 一致：只有真实聊天（通道 私/群）写，离线回放/实验不污染。
    if get_llm_route() in ("私", "群"):
        ids = list(dict.fromkeys(str(i) for i in memory_ids if i))   # 去重保序
        if ids:
            injection_trace.setdefault("items", {})["memory"] = ids
        injection_trace["session"] = str(user_id)
        injection_trace["route"] = get_llm_route()
        try:
            nickname = _MEMORY_V2_STORE.her_nickname(user_id) if nickname_hint else ""
        except Exception:
            nickname = ""
        injection_trace["reply_used_nickname"] = bool(nickname and nickname in reply)
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
    append_user_history(user_id, record_msg, clean_reply(reply))

    # --- 📝 异步更新长期记忆（V2；会话级记忆已在对话前 probe_session 同步更新） ---
    if not is_group:  # 群会话不建用户记忆卡
        asyncio.create_task(update_memory_v2_shadow_task(
            user_id, record_msg, reply, memory_scope_id,
        ))

    return reply
