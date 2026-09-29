"""检索抽象层：六路检索 + RRF 融合 + 预算控制。

六路来源（全部同步 CPU，query 向量由调用方传入，全程只调 1 次 embedding）：
- corpus        背景记忆（persona/world/corpus_vectors.json）        → 走 RRF
- voice_sample  风格样本（persona/speech/voice_sample_vectors.json） → 走 RRF
- behavior      行为触发（L3 LLM 意图分类 + behavior_keywords 判别词兜底，不走 embedding）→ 走 RRF
- phrase        措辞指纹（**按 L3 判出的组 id 取**，不再向量检索）    → 走 RRF
- preference    偏好事实（**按 keywords 子串命中**，不再向量检索）    → 命中才带，不走 RRF
- core_story    核心记忆（persona/world/core_story_vectors.json）    → 命中才带，不走 RRF

**判据怎么选**（2026-09-26 定，别再搞混）：说法千变万化、要从 N 条里挑 → 向量 + LLM 判（corpus）；
类别少、任务其实是"给这句话归类" → 分类（behavior/phrase）；类别名能被用户直接说出来 → 子串命中（preference/terms）。
**phrase 和 preference 原本都用向量，都是"用检索工具干分类/关键词的活"**，已改掉（两路的地板都压在正例上）。

corpus/voice_sample/behavior/phrase 四路走加权 RRF 融合（score = w/(k+rank)），
再条数截断 + 字符预算截断，只注入本轮真正相关的信息；
preference/core_story 是身份事实层，命中才带、不占检索预算。
"""
import os
import json
import re
from dataclasses import dataclass, field

from .constants import (
    PROJECT_ROOT,
    VOICE_SAMPLE_VECTOR_FILE, CORE_STORY_VECTOR_FILE,
    PHRASES_FILE, PREFERENCES_FILE, CORPUS_KEYWORDS_FILE,
    RAG_THRESHOLD, CORPUS_TOP_N, CORPUS_CANDIDATE_N, CORPUS_LEXICAL_EXTRA_N,
    CORPUS_KEYWORD_FLOOR, CORPUS_STRONG_KEYWORD,
    VOICE_SAMPLE_THRESHOLD, VOICE_SAMPLE_TOP_N, VOICE_SAMPLE_KEEPALIVE, VOICE_SAMPLE_MIN_K,
    VOICE_SAMPLE_KEEPALIVE_MIN_SIM,
    PHRASE_TOP_N, PHRASE_PHASES_MAX,
    PREFERENCE_TOP_N,
    CORE_STORY_THRESHOLD, CORE_STORY_TOP_N,
    RRF_K, SOURCE_WEIGHTS, RETRIEVAL_TOPK,
    RETRIEVAL_BUDGET_CHARS, MAX_RETRIEVAL_ITEM_CHARS,
)
from .rag import cosine_similarity, load_vector_db
from .persona import _format_behavior_rule


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


# corpus 的「钩子」：用户提到这些具体词时，**把该条递进候选池**（确定性命中，不靠语义相似）
# ⚠️ 不是"直通注入"了——2026-09-29 起判定全交 LLM，钩子只负责"递过去看"。
# 为什么需要：实测 322 条里有一大批**没有任何召回入口**——用户不可能说出一句话恰好把它们
# 勾出来，于是永远不会被注入。语义检索解决不了这个：那是"像不像"，这里缺的是"**提没提到**"。
# 数据：persona/world/corpus_keywords.json（键是 statement_final 的序号，由 build_corpus_keywords.py 生成）。
_corpus_keywords_cache = None


def load_corpus_keywords() -> dict:
    global _corpus_keywords_cache
    if _corpus_keywords_cache is not None:
        return _corpus_keywords_cache
    if not CORPUS_KEYWORDS_FILE.exists():
        _corpus_keywords_cache = {}
        return _corpus_keywords_cache
    try:
        with open(CORPUS_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("keywords", {}) if isinstance(data, dict) else {}
        _corpus_keywords_cache = {str(k): v for k, v in raw.items() if isinstance(v, list) and v}
    except (json.JSONDecodeError, OSError):
        _corpus_keywords_cache = {}
    return _corpus_keywords_cache


def _corpus_gate_pass(query: str, statement: str, sim: float) -> bool:
    """corpus 的「**这个条目值得让 LLM 看一眼**」判定（强关键词重叠 / 语义达标+区分性词重叠）。

    ⚠️ **它曾经的含义是"直通注入"（放行就进提示词、跳过 LLM 判）。2026-09-29 已改为"只进候选池"。**

    为什么必须改：实测**基于词重叠/词频的尺子分不开"说的是谁"**——
      · 误报 n7「宝宝我明天要早起我先去睡了」（用户在说**自己**）对 #271：`ov=0.40`
      · 真阳性 co9「你不是以前有喜欢的男学霸吗」（在问**她**）对 #42：`ov=0.143`
    **误报的重叠度反而更高**，所以任何 FLOOR 都是在拿一个换另一个（试过 0.5：
    n7 挡住了，co9/co10 一起被打回红）。区别在"**说的是谁**"——那是 LLM 判的事，
    词重叠（甚至 bigram 文档频率）都拿不到这个信息。
    → 杠杆只剩一个：**别让它绕过判定**。门/钩子现在只决定"递哪些条给 LLM 看"。

    判据本身保持不变——**当筛选用它够好**：挑出来的确实都跟这句话沾边，
    只是"沾边"不等于"在问她"。
    """
    ov = _corpus_keyword_overlap(query, statement)
    if ov >= CORPUS_STRONG_KEYWORD:
        return True
    return sim >= RAG_THRESHOLD and ov >= CORPUS_KEYWORD_FLOOR


# ==================== 三路 retriever ====================

def retrieve_corpus(user_query: str, query_vector,
                    threshold: float = RAG_THRESHOLD,
                    top_n: int = CORPUS_TOP_N) -> list:
    """直播记忆检索：只返回**门放行**的条目，item_id 用序号，text=场景化陈述。

    ⚠️ **线上注入路径已经不用它了**（2026-09-29）。现在它是**诊断/实验用**：
    `judge_experiment` 拿它当"关键词门、不加 LLM 判"的对照方法；
    `coverage_check` / `threshold_scan` 用它量开火率与噪声地板。
    线上注入走的是 **`retrieve_corpus_candidates` + `judge_corpus`**（见 `core.handle_chat`）——
    门放行的条目也在这条路上，只是**照样要过 LLM 判**（理由见 `_corpus_gate_pass`）。
    """
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


def retrieve_corpus_candidates(user_query: str, query_vector,
                               top_n: int = CORPUS_CANDIDATE_N) -> list:
    """corpus 的**候选召回**：给 LLM 判定用的输入（见 corpus_judge）。

    三部分拼起来（**顺序 = 相关性顺序**，判定池的大小则封顶）：
      ① 按余弦取的 top-N —— **为什么敢不过阈值**：实测正例对正确那条的**排名**是对的
         （「你多高啊」→ 身高那条排第 1），只是分数（0.443）低于噪声地板
         —— **排序有用、阈值没用**。所以把打分交给排序，把"相关不相关"交给 LLM 判。
      ② 钩子命中的（用户确实提到了那些词）
      ③ 区分性词重叠 ≥ FLOOR 的（原来"门放行直通"那批，`CORPUS_LEXICAL_EXTRA_N` 封顶）
    ⚠️ ②③ **只看词面、不看语义分**：要救的正是"语义分低但用户确实提到了"的那批。
    """
    db = load_vector_db()
    if not db or not user_query or not query_vector:
        return []
    scored = [RetrievalItem(source="corpus", item_id=str(i),
                            score=cosine_similarity(query_vector, it["vector"]),
                            text=it["text"])
              for i, it in enumerate(db)]
    scored.sort(key=lambda x: x.score, reverse=True)
    top = scored[:top_n]
    # **钩子命中的也塞进候选**——它们语义分低（正是"没有入口"那批），
    # 但用户确实提到了里面的词，值得让 LLM 判一次"是不是在聊这件事"。
    # 让 LLM 看一眼的还有两类"**语义分低、但词面上确实沾边**"的条目：
    #   ① **钩子命中的** —— 用户确实提到了那些词（解决"没有入口"）
    #   ② **区分性词重叠 ≥ FLOOR 的** —— 原来"门放行直通"的那批。门不再直通了，
    #      它们必须走这里进候选池，否则"取消直通"就等于**丢召回**。
    # ⚠️ 判据**只看词面，不看语义分**（所以不用 _corpus_gate_pass，那个还要求 sim≥0.48）：
    #    要救的恰恰是语义分低的那批（co9 的 #42：ov=0.143、sim 不到 0.48）。
    #    实测若沿用 _corpus_gate_pass，co9 的候选只有 7 条且不含 #42 → 照样红。
    keywords_map = load_corpus_keywords()
    got = {it.item_id for it in top}
    lexical = []
    for it in scored:
        if it.item_id in got:
            continue
        kws = keywords_map.get(it.item_id)
        hooked = bool(kws and any(k in user_query for k in kws))
        ov = _corpus_keyword_overlap(user_query, it.text)
        if hooked or ov >= CORPUS_KEYWORD_FLOOR:
            lexical.append((2.0 if hooked else ov, it))   # 钩子命中优先于"只是撞了几个词"
    lexical.sort(key=lambda x: -x[0])
    for _, it in lexical[:CORPUS_LEXICAL_EXTRA_N]:         # 封顶，别把判定池灌爆
        top.append(it)
    return top


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
                "text": "", "extra": {"user": s["user"], "reply": s["reply"], "type": s.get("type", ""), "length": s.get("length", "short")}}
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
    """风格样本检索。extra 含 user/reply，供 few-shot 注入。

    ⚠️ `not query_vector` 这个早退**不能省**：embedding 服务挂掉时
    `rag.embed_query` 会返回 None（它自己吞异常），而保底注入那条路
    （`_ensure_min_samples` 算全体最高分）没判 None —— `cosine_similarity(None, …)`
    里的 `zip(None, …)` 直接抛 TypeError。
    实测踩坑（2026-09-27）：用户智谱 embedding key 过期 → 每次带文字的消息
    都在这里抛异常，而 core.handle_chat 那一段**没有 try 兜底** → 整条回复任务死掉，
    **她一个字都不回**（不只主动发言）。
    另外两路（retrieve_corpus / retrieve_core_stories）开头都有同样的判断，
    就这路漏了——这里的写法是为了跟它们对齐：拿不到向量就当作"没检索到"，不是崩。
    """
    samples = load_voice_sample_vectors()
    if not samples or not query_vector:
        return []
    entries = [{"vector": s["vector"], "id": s["id"], "text": "",
                "extra": {"user": s["user"], "reply": s["reply"], "type": s.get("type", ""), "length": s.get("length", "short")}}
               for s in samples]
    items = _score_candidates(query_vector, entries, threshold, top_n,
                              "voice_sample", lambda e: e["id"], lambda e: e["text"],
                              lambda e: e["extra"])
    return _ensure_min_samples(items, samples, query_vector)


# 行为判别词 → 强制命中行为（语义检索对"敷衍/鸽/迟到"这类口语有~0.55天花板且易错配，
# 对齐 LEGENDARY 的思路：固定判别词 → 固定反应）。
# 数据化：从 persona/behavior/behavior_keywords.json 加载，填 JSON 不用改代码。
_BEHAVIOR_KEYWORDS_FILE = PROJECT_ROOT / "persona" / "behavior" / "behavior_keywords.json"
_behavior_keywords_cache = None


def load_behavior_keywords() -> dict:
    """加载行为判别词（行为名 → 判别词列表）。行为名必须与 behaviors.json 的 name 一致。"""
    global _behavior_keywords_cache
    if _behavior_keywords_cache is not None:
        return _behavior_keywords_cache
    if not _BEHAVIOR_KEYWORDS_FILE.exists():
        _behavior_keywords_cache = {}
        return _behavior_keywords_cache
    try:
        data = json.loads(_BEHAVIOR_KEYWORDS_FILE.read_text(encoding="utf-8"))
        _behavior_keywords_cache = {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError):
        _behavior_keywords_cache = {}
    return _behavior_keywords_cache


_BEHAVIOR_KEYWORDS = load_behavior_keywords()


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


# ==================== 措辞指纹（按 id 取组，不再向量检索） ====================
# 为什么去掉了向量：措辞组的 trigger 是 **6~13 字的类别标签**（"被夸奖、被称赞时"），
# 不是自然语言句子——拿它跟用户消息算余弦是**用检索工具干分类的活**。实测：
# 噪声地板 0.575 / 正例 0.573，中位数 0.461 都过阈值 → **100% 开火，等于没有判据**。
# 现在改由 L3 的 LLM 分类给出组 id（和 behaviors 同一套：类别少、说法固定就该分类不该检索），
# 这里只负责按 id 把组取出来。

_phrase_groups_cache = None


def load_phrase_groups() -> list:
    """读 persona/speech/phrases.json（源文件，缓存）。环境变量 PHRASES=0 时返回 []。

    注意读的是**源文件**（phrases.json），不是 precompute 出来的向量缓存——
    这一路不再用向量，`phrase_vectors.json` 已停用（见 交接文档/注入链路）。
    """
    global _phrase_groups_cache
    if _phrase_groups_cache is not None:
        return _phrase_groups_cache
    if os.environ.get("PHRASES", "1") == "0" or not PHRASES_FILE.exists():
        _phrase_groups_cache = []
        return _phrase_groups_cache
    try:
        with open(PHRASES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        groups = data.get("phrase_groups", []) if isinstance(data, dict) else []
        _phrase_groups_cache = [g for g in groups if isinstance(g, dict) and g.get("id") and g.get("phrases")]
    except (json.JSONDecodeError, OSError):
        _phrase_groups_cache = []
    return _phrase_groups_cache


def select_phrase_groups(group_ids, groups: list = None, top_n: int = PHRASE_TOP_N) -> list:
    """把 L3 判出来的措辞组 id 转成注入项（不再有阈值/相似度）。

    extra 含 meaning/phrases/usage，供【她的固定说法】注入。
    """
    if not group_ids:
        return []
    groups = load_phrase_groups() if groups is None else groups
    by_id = {g.get("id"): g for g in groups}
    out = []
    for gid in group_ids:
        g = by_id.get(gid)
        if not g:
            continue                      # 模型编的 id 直接丢
        out.append(RetrievalItem(source="phrase", item_id=gid, score=1.0, text="",
                                 extra={"meaning": g.get("meaning", ""),
                                        "phrases": g.get("phrases", []),
                                        "usage": g.get("usage", "")}))
        if len(out) >= top_n:
            break
    return out


# ==================== 偏好检索（第 5 路） ====================

# 偏好缓存（模块级，一次性加载）
_preferences_cache = None


def load_preferences() -> list:
    """读 persona/world/preferences.json（源文件，缓存）。"""
    global _preferences_cache
    if _preferences_cache is not None:
        return _preferences_cache
    if not PREFERENCES_FILE.exists():
        _preferences_cache = []
        return _preferences_cache
    try:
        with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        _preferences_cache = [e for e in entries if e.get("text") and e.get("keywords")]
    except (json.JSONDecodeError, OSError):
        _preferences_cache = []
    return _preferences_cache


def retrieve_preferences(user_query: str, top_n: int = PREFERENCE_TOP_N) -> list:
    """偏好命中（第 5 路）：**keyword 子串 或 pattern 正则**命中，就带该条。

    返回 [{id, category, text, score}]，供【灰泽满的偏好】注入。
    不进 RRF 融合、不占检索预算——偏好是身份事实层，命中才带，避免与风格样本抢预算。

    **为什么不用向量了**：偏好是"类别 + 一串用户能直接说出的词"（食物 / 水果 / 猫狗…），
    跟 terms 同构。而实测余弦的噪声地板 0.565 **插在正例正中间**（正例 0.501/0.552/0.590），
    任何阈值都分不开（试过提到 0.63，噪声挡住了正例一起被杀）。改用确定性命中：
    零成本、可解释、可人工审、能写测试。

    **pattern（可选）**：正则命中，和 terms.json 同一套（见 core.py 的名词库匹配）。
    子串做不到的场景——如"吃"会撞"吃什么药/吃瓜"，用负向断言排除。**子串法的天花板就在这**：
    只能靠挑更多具体词逼近，不可能根治；真要好得换分类器。
    """
    if not user_query:
        return []
    q = user_query.lower()
    hits = []
    for e in load_preferences():
        kw_hit = any(str(k).lower() in q for k in e.get("keywords", []))
        pat = e.get("pattern")
        try:
            pat_hit = bool(pat) and re.search(pat, user_query, re.IGNORECASE) is not None
        except re.error:
            pat_hit = False          # 写坏的正则不该炸掉整路
        if kw_hit or pat_hit:
            hits.append({"id": e.get("id", ""), "category": e.get("category", ""),
                         "text": e.get("text", ""), "score": 1.0})
        if len(hits) >= top_n:
            break
    return hits


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
