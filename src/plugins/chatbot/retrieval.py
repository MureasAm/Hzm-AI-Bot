"""V3 检索抽象层：三路检索 + RRF 融合 + 预算控制。

三路检索（全部同步 CPU，query 向量由调用方传入，全程只调 1 次 embedding）：
- corpus        直播记忆（persona/world/corpus_vectors.json）
- voice_sample  风格样本（persona/speech/voice_sample_vectors.json）
- behavior      行为触发（persona/behavior/trigger_vectors.json）

融合后按字符预算截断，只注入本轮真正相关的信息。
"""
import os
import json
from dataclasses import dataclass, field

from .constants import (
    VOICE_SAMPLE_VECTOR_FILE, PHRASE_VECTOR_FILE, PREFERENCE_VECTOR_FILE, CORE_STORY_VECTOR_FILE,
    RAG_THRESHOLD, CORPUS_TOP_N,
    CORPUS_KEYWORD_FLOOR, CORPUS_STRONG_KEYWORD,
    VOICE_SAMPLE_THRESHOLD, VOICE_SAMPLE_TOP_N, VOICE_SAMPLE_KEEPALIVE, VOICE_SAMPLE_MIN_K,
    VOICE_SAMPLE_KEEPALIVE_MIN_SIM,
    PHRASE_THRESHOLD, PHRASE_TOP_N, PHRASE_PHASES_MAX,
    PREFERENCE_THRESHOLD, PREFERENCE_TOP_N,
    CORE_STORY_THRESHOLD, CORE_STORY_TOP_N,
    RRF_K, SOURCE_WEIGHTS, RETRIEVAL_TOPK,
    RETRIEVAL_BUDGET_CHARS, MAX_RETRIEVAL_ITEM_CHARS,
)
from .rag import cosine_similarity, load_vector_db
from .persona import load_trigger_vectors, _format_behavior_rule


@dataclass
class RetrievalItem:
    """一条检索候选。source 区分来源，extra 承载注入所需信息。"""
    source: str                       # "corpus" | "voice_sample" | "behavior"
    item_id: str                      # 样本 id / 行为 name / corpus 序号
    score: float                      # 原始余弦相似度
    rank: int = 0                     # 本路内排名（1-based），RRF 时填充
    fusion_score: float = 0.0         # RRF 融合分
    text: str = ""                    # 注入文本：corpus=陈述 / behavior=指令 / sample=""
    extra: dict = field(default_factory=dict)  # voice_sample → {"user","reply","type"}


# ==================== 统一打分核心 ====================

def _score_candidates(query_vector, entries, threshold, top_n,
                      source, id_of, text_of, extra_of=None) -> list:
    """通用打分：低于阈值丢弃，按分数降序取 top_n。

    entries: 可迭代的 {"vector": [...], ...}
    """
    if not query_vector:
        return []
    scored = []
    for entry in entries:
        sim = cosine_similarity(query_vector, entry["vector"])
        if sim < threshold:
            continue
        scored.append(RetrievalItem(
            source=source,
            item_id=id_of(entry),
            score=sim,
            text=text_of(entry),
            extra=extra_of(entry) if extra_of else {},
        ))
    scored.sort(key=lambda it: it.score, reverse=True)
    return scored[:top_n]


# ==================== V6 corpus 关键词门 ====================
# 纯 cosine 对"问句 vs 陈述式"嵌入有鸿沟（相关 0.51 / 无关 0.62 无法用单一阈值分开），
# 且 embedding 按句式聚团（"灰泽满你…"问句不论夸骂都挤一起）。加区分性关键词门：
# 只有 query 与 statement 有实质词重叠才放行，语义阈值敢降也不乱锁。

# 领域停用字：过滤区分性 bigram 时剔除的高频词（灰泽满/绿冻/直播/你我他的…）
_GATE_STOP_CHARS = set("灰泽满绿冻直播你我他的了吗呢吧啊嗯哈是不是不什么怎么和去很在就没都")
# 名字/高频实体：先整词剔除再算 bigram（"灰泽"残留在每句陈述里会污染重叠度）
_GATE_STRIP_TOKENS = ("灰泽满", "灰泽满Hazel", "hzm", "绿冻", "满神", "小满", "满姐")


def _gate_bigrams(text: str) -> set:
    """区分性 bigram：去名字/实体 + 去领域停用字。"""
    for tok in _GATE_STRIP_TOKENS:
        text = text.replace(tok, "")
    text = text.replace(" ", "")
    out = set()
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a in _GATE_STOP_CHARS or b in _GATE_STOP_CHARS:
            continue
        out.add(a + b)
    return out


def _corpus_keyword_overlap(query: str, statement: str) -> float:
    """query 的区分性 bigram 被 statement 覆盖的比例（0~1）。"""
    qb = _gate_bigrams(query)
    if not qb:
        return 0.0
    tb = _gate_bigrams(statement)
    return len(qb & tb) / len(qb)


def _corpus_gate_pass(query: str, statement: str, sim: float) -> bool:
    """corpus 放行判定：强关键词直接过；否则语义达标 + 有区分性词重叠才过。"""
    ov = _corpus_keyword_overlap(query, statement)
    if ov >= CORPUS_STRONG_KEYWORD:
        return True
    return sim >= RAG_THRESHOLD and ov >= CORPUS_KEYWORD_FLOOR


# ==================== 三路 retriever ====================

def retrieve_corpus(user_query: str, query_vector,
                    threshold: float = RAG_THRESHOLD,
                    top_n: int = CORPUS_TOP_N) -> list:
    """直播记忆检索。item_id 用序号，text=场景化陈述。带 V6 关键词门。"""
    db = load_vector_db()
    if not db or not user_query or not query_vector:
        return []
    scored = []
    for i, it in enumerate(db):
        sim = cosine_similarity(query_vector, it["vector"])
        if not _corpus_gate_pass(user_query, it["text"], sim):
            continue
        scored.append(RetrievalItem(source="corpus", item_id=str(i), score=sim, text=it["text"]))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]


# 声音样本向量缓存（模块级，一次性加载）
_sample_vectors = None


def load_voice_sample_vectors() -> list:
    """读 persona/speech/voice_sample_vectors.json（缓存）。环境变量 VOICE_SAMPLES=0 时返回 []。"""
    global _sample_vectors
    if _sample_vectors is not None:
        return _sample_vectors
    if os.environ.get("VOICE_SAMPLES", "1") == "0" or not VOICE_SAMPLE_VECTOR_FILE.exists():
        _sample_vectors = []
        return _sample_vectors
    try:
        with open(VOICE_SAMPLE_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", []) if isinstance(data, dict) else []
        _sample_vectors = [s for s in samples
                           if isinstance(s, dict) and s.get("vector") and s.get("reply")]
    except (json.JSONDecodeError, OSError):
        _sample_vectors = []
    return _sample_vectors


def _ensure_min_samples(items: list, samples: list, query_vector) -> list:
    """保底：阈值过滤后为空时，注入全体最高分 1 条，保住声音风格不断档。

    但保底也要设门槛（VOICE_SAMPLE_KEEPALIVE_MIN_SIM）：如果最高分样本
    相关度太低（日常短句 vs 直播时间样本 ≈ 0.55），宁可风格断档也不注入
    无关样本——否则会像"9点是你那边"那样，把直播时间样本硬塞给日常话题。
    """
    if items or not VOICE_SAMPLE_KEEPALIVE or not samples:
        return items
    entries = [{"vector": s["vector"], "id": s["id"],
                "text": "", "extra": {"user": s["user"], "reply": s["reply"], "type": s.get("type", "")}}
               for s in samples]
    # 找全体最高分，若低于保底门槛则不注入（宁缺毋滥）
    best = max((cosine_similarity(query_vector, s["vector"]) for s in entries), default=-1.0)
    if best < VOICE_SAMPLE_KEEPALIVE_MIN_SIM:
        return []
    return _score_candidates(query_vector, entries, -1.0, VOICE_SAMPLE_MIN_K,
                             "voice_sample", lambda e: e["id"], lambda e: e["text"],
                             lambda e: e["extra"])


def retrieve_voice_samples(user_query: str, query_vector,
                           threshold: float = VOICE_SAMPLE_THRESHOLD,
                           top_n: int = VOICE_SAMPLE_TOP_N) -> list:
    """风格样本检索。extra 含 user/reply，供 few-shot 注入。"""
    samples = load_voice_sample_vectors()
    if not samples:
        return []
    entries = [{"vector": s["vector"], "id": s["id"], "text": "",
                "extra": {"user": s["user"], "reply": s["reply"], "type": s.get("type", "")}}
               for s in samples]
    items = _score_candidates(query_vector, entries, threshold, top_n,
                              "voice_sample", lambda e: e["id"], lambda e: e["text"],
                              lambda e: e["extra"])
    return _ensure_min_samples(items, samples, query_vector)


# 口语质问关键词 → 强制命中行为（语义检索对"敷衍/鸽/迟到"这类口语有~0.55天花板且易错配，
# 对齐 LEGENDARY 的思路：固定质问词 → 固定反应）
_BEHAVIOR_KEYWORDS = {
    "被质疑时心虚辩解": ["敷衍", "又骗", "在骗", "装的", "撒谎", "骗子"],
    "失约被催时认栽滑跪": ["又迟到", "说好的", "又鸽", "鸽了", "放鸽子", "爽约", "又没播"],
    # 被越界：领域黑话（黄桃梗/擦边/低俗）是判别性内容词，LLM 未必认识"黄桃"这类梗，
    # 关键词兜底可靠（实测 LLM 对"造个黄桃吧"判 null，加兜底后命中）
    "被越界时冷静推开": ["黄桃", "擦边", "低俗", "黄段子", "开黄腔", "色色", "涩涩"],
}


def select_behavior_item(user_msg: str, behavior_intent: str, behaviors: list) -> "RetrievalItem | None":
    """L3：把行为意图（LLM 判定）转成行为注入项。

    优先级：LLM 意图 > 关键词兜底（判别词：敷衍/骗/鸽/迟到…，当 LLM 拿不准时保底）。
    返回 None 表示本轮不注入任何行为。替代旧的纯语义检索（embedding 按句式聚团，
    把'夸奖'和'质问'这类同句式消息挤在一起会误判——见 ROADMAP/L3 说明）。
    """
    if not behaviors or not user_msg:
        return None
    # ① LLM 判定的意图（权威）
    if behavior_intent:
        for b in behaviors:
            if b.get("name") == behavior_intent:
                return RetrievalItem(source="behavior", item_id=behavior_intent, score=1.0,
                                     text=_format_behavior_rule(b))
    # ② 关键词兜底：LLM 拿不准但消息含判别词（"你是不是在敷衍我"这类），保底命中
    for b in behaviors:
        name = b.get("name", "")
        kws = _BEHAVIOR_KEYWORDS.get(name)
        if kws and any(k in user_msg for k in kws):
            return RetrievalItem(source="behavior", item_id=name, score=1.0,
                                 text=_format_behavior_rule(b))
    return None


# 措辞指纹向量缓存（模块级，一次性加载）
_phrase_vectors = None


def load_phrase_vectors() -> list:
    """读 persona/speech/phrase_vectors.json（缓存）。环境变量 PHRASES=0 时返回 []。"""
    global _phrase_vectors
    if _phrase_vectors is not None:
        return _phrase_vectors
    if os.environ.get("PHRASES", "1") == "0" or not PHRASE_VECTOR_FILE.exists():
        _phrase_vectors = []
        return _phrase_vectors
    try:
        with open(PHRASE_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        groups = data.get("phrase_groups", []) if isinstance(data, dict) else []
        _phrase_vectors = [g for g in groups if isinstance(g, dict) and g.get("vector") and g.get("phrases")]
    except (json.JSONDecodeError, OSError):
        _phrase_vectors = []
    return _phrase_vectors


def retrieve_phrases(user_query: str, query_vector,
                     threshold: float = PHRASE_THRESHOLD,
                     top_n: int = PHRASE_TOP_N) -> list:
    """措辞指纹检索。extra 含 phrases/usage，供注入。"""
    groups = load_phrase_vectors()
    if not groups:
        return []
    entries = [{"vector": g["vector"], "id": g["id"], "text": "",
                "extra": {"meaning": g.get("meaning", ""), "phrases": g.get("phrases", []),
                          "usage": g.get("usage", "")}}
               for g in groups]
    return _score_candidates(query_vector, entries, threshold, top_n,
                             "phrase", lambda e: e["id"], lambda e: e["text"],
                             lambda e: e["extra"])


# ==================== 偏好检索（第 5 路） ====================

# 偏好向量缓存（模块级，一次性加载）
_pref_vectors = None


def load_preference_vectors() -> list:
    """读 persona/world/preference_vectors.json（缓存）。"""
    global _pref_vectors
    if _pref_vectors is not None:
        return _pref_vectors
    if not PREFERENCE_VECTOR_FILE.exists():
        _pref_vectors = []
        return _pref_vectors
    try:
        with open(PREFERENCE_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        _pref_vectors = [e for e in entries if e.get("vector") and e.get("text")]
    except (json.JSONDecodeError, OSError):
        _pref_vectors = []
    return _pref_vectors


def retrieve_preferences(user_query: str, query_vector,
                         threshold: float = PREFERENCE_THRESHOLD,
                         top_n: int = PREFERENCE_TOP_N) -> list:
    """偏好语义检索（第 5 路）：命中与当前消息相关的偏好条目。

    返回 [{id, category, text, score}]，供【灰泽满的偏好】注入。
    不进 RRF 融合、不占检索预算——偏好是身份事实层，命中才带，避免与风格样本抢预算。
    """
    entries = load_preference_vectors()
    if not entries or not query_vector:
        return []
    scored = []
    for e in entries:
        sim = cosine_similarity(query_vector, e["vector"])
        if sim < threshold:
            continue
        scored.append((sim, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{
        "id": e.get("id", ""),
        "category": e.get("category", ""),
        "text": e.get("text", ""),
        "score": round(sim, 3),
    } for sim, e in scored[:top_n]]


# ==================== 核心记忆检索（印象最深的结晶） ====================

# 核心记忆向量缓存（模块级，一次性加载）
_core_story_vectors = None


def load_core_story_vectors() -> list:
    """读 persona/world/core_story_vectors.json（缓存）。"""
    global _core_story_vectors
    if _core_story_vectors is not None:
        return _core_story_vectors
    if not CORE_STORY_VECTOR_FILE.exists():
        _core_story_vectors = []
        return _core_story_vectors
    try:
        with open(CORE_STORY_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        stories = data.get("stories", []) if isinstance(data, dict) else []
        _core_story_vectors = [s for s in stories if s.get("vector") and s.get("text")]
    except (json.JSONDecodeError, OSError):
        _core_story_vectors = []
    return _core_story_vectors


def retrieve_core_stories(user_query: str, query_vector,
                          threshold: float = CORE_STORY_THRESHOLD,
                          top_n: int = CORE_STORY_TOP_N) -> list:
    """核心记忆检索：命中与当前消息相关的核心故事（比 corpus 阈值低，更容易浮现）。

    返回 [{id, category, text, score}]，注入【她的核心记忆】。
    这些是直播以来印象最深的结晶，独立检索避免被 273 条普通背景淹没。
    """
    stories = load_core_story_vectors()
    if not stories or not query_vector:
        return []
    scored = []
    for s in stories:
        sim = cosine_similarity(query_vector, s["vector"])
        if sim < threshold:
            continue
        scored.append((sim, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{
        "id": s.get("id", ""),
        "category": s.get("category", ""),
        "text": s.get("text", ""),
        "score": round(sim, 3),
    } for sim, s in scored[:top_n]]


# ==================== RRF 融合 ====================

def rrf_fuse(ranked_lists: list, k: int = RRF_K,
             weights: dict = SOURCE_WEIGHTS) -> list:
    """加权 Reciprocal Rank Fusion。三路来源互不重叠，Σ 退化为单加数。"""
    fused = []
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            item.rank = rank
            w = weights.get(item.source, 1.0)
            item.fusion_score = w / (k + rank)
            fused.append(item)
    fused.sort(key=lambda it: (-it.fusion_score, -it.score))
    return fused


# ==================== 预算控制 ====================

def _item_cost(it: RetrievalItem) -> int:
    if it.source == "voice_sample":
        return len(it.extra.get("user", "")) + len(it.extra.get("reply", ""))
    if it.source == "phrase":
        # 措辞组成本 = 注入的短语总长（按 PHRASE_PHASES_MAX 裁剪后）
        return sum(len(p) for p in it.extra.get("phrases", [])[:PHRASE_PHASES_MAX])
    return len(it.text)


def truncate_by_budget(items: list, budget_chars: int = RETRIEVAL_BUDGET_CHARS,
                       max_item_chars: int = MAX_RETRIEVAL_ITEM_CHARS) -> list:
    """按融合序贪心保留，超预算丢弃低优先级条目。"""
    total, kept = 0, []
    for it in items:
        cost = _item_cost(it)
        if cost > max_item_chars:
            cost = max_item_chars
        if total + cost > budget_chars:
            continue
        kept.append(it)
        total += cost
    return kept


def fuse_and_truncate(corpus_items, sample_items, behavior_items, phrase_items=None) -> list:
    """完整融合流程：RRF → 条数截断 → 字符预算截断。

    当 VOICE_SAMPLE_PREFER_SHORT=True 时，short 档声音样本获得权重加成，
    让模型优先看到短句范例（控制回复长度）。
    """
    from .constants import VOICE_SAMPLE_PREFER_SHORT, SOURCE_WEIGHTS as _W

    if phrase_items is None:
        phrase_items = []

    weights = dict(_W)
    if VOICE_SAMPLE_PREFER_SHORT:
        # 给 short 样本额外权重，让短句范例更可能进入 top-k
        for it in sample_items:
            if it.extra.get("length", "short") == "short":
                weights["voice_sample"] = weights.get("voice_sample", 1.0) + 0.3
                break  # 任一 short 存在即加权整路

    fused = rrf_fuse([corpus_items, sample_items, behavior_items, phrase_items], weights=weights)
    fused = fused[:RETRIEVAL_TOPK]
    return truncate_by_budget(fused)
