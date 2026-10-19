"""检索抽象层：两路 RRF 融合 + 两路直达 + 预算控制。

来源（全部同步 CPU，query 向量由调用方传入，全程只调 1 次 embedding）：
- corpus        背景记忆（persona/injection/11_corpus/corpus_vectors.json）        → 走 RRF
- behavior      行为触发（L3 LLM 意图分类 + behavior_keywords 判别词兜底，不走 embedding）→ 走 RRF
- preference    偏好事实（**按 keywords 子串命中**，不再向量检索）    → 命中才带，不走 RRF
- core_story    核心记忆（persona/injection/05_core_stories/core_story_vectors.json）    → 命中才带，不走 RRF

**判据怎么选**（2026-09-26 定，别再搞混）：说法千变万化、要从 N 条里挑 → 向量 + LLM 判（corpus）；
类别少、任务其实是"给这句话归类" → 分类（behavior）；类别名能被用户直接说出来 → 子串命中（preference/terms）。
**preference 原本也用向量，是"用检索工具干关键词的活"**，已改掉（地板压在正例上）。

corpus/behavior 两路走加权 RRF 融合（score = w/(k+rank)），再条数截断 + 字符预算截断，
只注入本轮真正相关的信息；preference/core_story 是身份事实层，命中才带、不占检索预算。

（voice_sample / phrase 两路已于 2026-10-04 随对应文件删除。）
"""
import json
import re
from dataclasses import dataclass, field

from .constants import (
    CORE_STORY_VECTOR_FILE,
    PREFERENCES_FILE, CORPUS_KEYWORDS_FILE, BEHAVIOR_KEYWORDS_FILE, ASSOCIATIONS_FILE,
    RAG_THRESHOLD, CORPUS_TOP_N, CORPUS_CANDIDATE_N, CORPUS_LEXICAL_EXTRA_N,
    CORPUS_KEYWORD_FLOOR, CORPUS_STRONG_KEYWORD,
    PREFERENCE_TOP_N,
    CORE_STORY_THRESHOLD, CORE_STORY_TOP_N,
    RRF_K, SOURCE_WEIGHTS, RETRIEVAL_TOPK,
    RETRIEVAL_BUDGET_CHARS, MAX_RETRIEVAL_ITEM_CHARS,
)
from .rag import cosine_similarity, load_vector_db
from .persona import _format_behavior_rule
from . import injection as inj


@dataclass
class RetrievalItem:
    """一条检索候选。source 区分来源，extra 承载注入所需信息。"""
    source: str                       # "corpus" | "behavior"
    item_id: str                      # 样本 id / 行为 name / corpus 序号
    score: float                      # 原始余弦相似度
    rank: int = 0                     # 本路内排名（1-based），RRF 时填充
    fusion_score: float = 0.0         # RRF 融合分
    text: str = ""                    # 注入文本：corpus=陈述 / behavior=指令
    extra: dict = field(default_factory=dict)  # 各路附带的注入所需信息


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
# 数据：persona/injection/11_corpus/corpus_keywords.json（键是 statement_final 的序号，由 build_corpus_keywords.py 生成）。
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


# 行为判别词 → 强制命中行为（语义检索对"敷衍/鸽/迟到"这类口语有~0.55天花板且易错配，
# 对齐 LEGENDARY 的思路：固定判别词 → 固定反应）。
# 数据化：从 persona/injection/10_behaviors/behavior_keywords.json 加载，填 JSON 不用改代码。
_behavior_keywords_cache = None


def load_behavior_keywords() -> dict:
    """加载行为判别词（行为名 → 判别词列表）。行为名必须与 behaviors.json 的 name 一致。"""
    global _behavior_keywords_cache
    if _behavior_keywords_cache is not None:
        return _behavior_keywords_cache
    if not BEHAVIOR_KEYWORDS_FILE.exists():
        _behavior_keywords_cache = {}
        return _behavior_keywords_cache
    try:
        data = json.loads(BEHAVIOR_KEYWORDS_FILE.read_text(encoding="utf-8"))
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





# ==================== 偏好检索（第 5 路） ====================

# 偏好缓存（模块级，一次性加载）
_preferences_cache = None


def load_preferences() -> list:
    """读 persona/injection/04_preferences/preferences.json（源文件，缓存）。"""
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


# ==================== 关联背景（作品 → 作者 → 喜好） ====================
# 数据在 persona/injection/07_terms/associations.json（**只存实体别名 + 边**）：
# 事实仍在 terms.json，态度仍在 04_preferences，这里只负责"沿边走"并把关系句渲染出来。
# 为什么不让 terms 自己展开：terms 是"一个词什么意思"的关键词匹配，**按话题索引**；
# 关联要的是"用户只报作品名，也能自己接上作者/地点"。两者判据不同，别混在一处。

_associations_cache = None
_MAX_ASSOCIATION_SEEDS = 3      # 一次最多从几个命中实体出发（防一条消息带出一串）
_MAX_ASSOCIATION_ENTITIES = 4   # 轨迹/实体清单上限：种子 + 作者 + 2 本代表作
_ASSOCIATION_NOTE_BUDGET = 300  # 【关联背景】整段字符预算


def load_associations() -> dict:
    """读 associations.json（缓存）。坏文件/缺文件返回空结构——注入链不能因此挂掉。"""
    global _associations_cache
    if _associations_cache is not None:
        return _associations_cache
    empty = {"entities": {}, "edges": []}
    if not ASSOCIATIONS_FILE.exists():
        _associations_cache = empty
        return _associations_cache
    try:
        data = json.loads(ASSOCIATIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        _associations_cache = empty
        return _associations_cache
    if not isinstance(data, dict):
        _associations_cache = empty
        return _associations_cache
    entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    edges = [e for e in (data.get("edges") or [])
             if isinstance(e, dict) and e.get("from") and e.get("to")]
    _associations_cache = {
        "entities": {str(k): v for k, v in entities.items() if isinstance(v, dict)},
        "edges": edges,
    }
    return _associations_cache


def _canonical_entity(name: str) -> str:
    """实体名/别名 → 实体 id；不认识返回空串。"""
    name = (name or "").strip()
    if not name:
        return ""
    entities = load_associations()["entities"]
    if name in entities:
        return name
    for eid, meta in entities.items():
        if any(str(a).strip() == name for a in (meta.get("aliases") or [])):
            return eid
    return ""


def match_entities(text: str, extra_entities=None) -> list:
    """消息里命中的已知实体（别名子串）+ 探测给出的实体（规范化后）。

    `match:false` 的实体（如《真相》——平时是普通词）不参与子串匹配，
    但探测**明确**给出它时仍然认（那是"用户真在聊这本书"）。
    """
    data = load_associations()
    entities = data["entities"]
    found: list = []
    lowered = (text or "").lower()
    for eid, meta in entities.items():
        if meta.get("match", True) is False:
            continue
        keys = [eid] + [str(a) for a in (meta.get("aliases") or []) if a]
        if any(k.lower() in lowered for k in keys):
            found.append(eid)
    for raw in extra_entities or []:
        eid = _canonical_entity(str(raw))
        if eid and eid not in found:
            found.append(eid)
    return found[:_MAX_ASSOCIATION_SEEDS]


def resolve_associations(text: str, extra_entities=None) -> dict:
    """沿关联边走最多 2 跳，返回 `{entities, note, preference_ids}`。

    · entities：这轮实际用到的实体（种子 + 沿边到达的作者/代表作），供注入轨迹定位；
    · note：注入【关联背景】的关系句（**不含**偏好原文——态度由偏好块自己注，避免一轮出现两次）；
    · preference_ids：边 `has_preference` 指向的偏好 id，由调用方并进偏好检索。
    识别不到实体、或走完没有任何可说的关系 → note 为空串（不注入）。
    """
    data = load_associations()
    entities = data["entities"]
    seeds = match_entities(text, extra_entities)
    if not seeds:
        return {"entities": [], "note": "", "preference_ids": []}

    outgoing: dict = {}
    for edge in data["edges"]:
        outgoing.setdefault(edge["from"], []).append(edge)

    used = list(seeds)
    sentences: list = []
    pref_ids: list = []
    rendered_rep: set = set()
    rendered_work: set = set()

    def _use(name: str) -> None:
        if name and name not in used and len(used) < _MAX_ASSOCIATION_ENTITIES:
            used.append(name)

    def _add(sentence: str) -> None:
        if sentence and sentence not in sentences:
            sentences.append(sentence)

    def _pref(target: str) -> None:
        pid = str(target).split(":", 1)[1] if ":" in str(target) else ""
        if pid and pid not in pref_ids:
            pref_ids.append(pid)

    def _representative(author: str) -> None:
        """作者的代表作（最多 2 本、跳过已经说过的书）。每个作者只说一次。"""
        if author in rendered_rep:
            return
        works = [e["to"] for e in outgoing.get(author, [])
                 if e.get("rel") == "representative_work" and e["to"] not in rendered_work][:2]
        if not works:
            return
        rendered_rep.add(author)
        for w in works:
            rendered_work.add(w)
            _use(w)
        _add(inj.T("terms", "associations_representative",
                   author=author, works="".join(f"《{w}》" for w in works)))

    for seed in seeds:
        for edge in outgoing.get(seed, []):
            rel, target = edge.get("rel"), edge["to"]
            if rel == "written_by":
                rendered_work.add(seed)
                _use(target)
                _add(inj.T("terms", "associations_written_by", work=seed, author=target))
                _representative(target)
                for pe in outgoing.get(target, []):
                    if pe.get("rel") == "has_preference":
                        _pref(pe["to"])
            elif rel == "located_in":
                _use(target)
                _add(inj.T("terms", "associations_located_in", subject=seed, place=target))
            elif rel == "has_preference":
                _pref(target)
        if entities.get(seed, {}).get("kind") == "author":
            _representative(seed)

    note = ""
    for sentence in sentences:
        if len(note) + len(sentence) > _ASSOCIATION_NOTE_BUDGET:
            break
        note += sentence
    return {"entities": used if note else [], "note": note, "preference_ids": pref_ids}


def retrieve_preferences(user_query: str, top_n: int = PREFERENCE_TOP_N,
                         extra_ids: list | None = None) -> list:
    """偏好命中（第 5 路）：**keyword 子串 或 pattern 正则**命中，就带该条。

    返回 [{id, category, text, score}]，供【灰泽满的偏好】注入。
    不进 RRF 融合、不占检索预算——偏好是身份事实层，命中才带，避免与风格样本抢预算。

    `extra_ids`：关联边（`has_preference`）指过来的偏好 id，**排在最前**——
    它代表"这轮话题顺着作品/作者落到了她的喜好上"，比碰巧撞上的关键词更该带。

    **为什么不用向量了**：偏好是"类别 + 一串用户能直接说出的词"（食物 / 水果 / 猫狗…），
    跟 terms 同构。而实测余弦的噪声地板 0.565 **插在正例正中间**（正例 0.501/0.552/0.590），
    任何阈值都分不开（试过提到 0.63，噪声挡住了正例一起被杀）。改用确定性命中：
    零成本、可解释、可人工审、能写测试。

    **pattern（可选）**：正则命中，和 terms.json 同一套（见 core.py 的名词库匹配）。
    子串做不到的场景——如"吃"会撞"吃什么药/吃瓜"，用负向断言排除。**子串法的天花板就在这**：
    只能靠挑更多具体词逼近，不可能根治；真要好得换分类器。
    """
    if not user_query and not extra_ids:
        return []
    q = (user_query or "").lower()
    entries = load_preferences()
    hits = []
    seen = set()
    if extra_ids:
        by_id = {str(e.get("id", "")): e for e in entries}
        for pid in extra_ids:
            e = by_id.get(str(pid))
            if not e or not e.get("text") or str(pid) in seen:
                continue
            hits.append({"id": e.get("id", ""), "category": e.get("category", ""),
                         "text": e.get("text", ""), "score": 1.0})
            seen.add(str(pid))
            if len(hits) >= top_n:
                return hits
    for e in entries:
        if str(e.get("id", "")) in seen:
            continue
        kw_hit = any(str(k).lower() in q for k in e.get("keywords", []))
        pat = e.get("pattern")
        try:
            pat_hit = bool(pat) and re.search(pat, user_query, re.IGNORECASE) is not None
        except re.error:
            pat_hit = False          # 写坏的正则不该炸掉整路
        if kw_hit or pat_hit:
            hits.append({"id": e.get("id", ""), "category": e.get("category", ""),
                         "text": e.get("text", ""), "score": 1.0})
            seen.add(str(e.get("id", "")))
        if len(hits) >= top_n:
            break
    return hits


# ==================== 核心记忆检索（印象最深的结晶） ====================

# 核心记忆向量缓存（模块级，一次性加载）
_core_story_vectors = None


def load_core_story_vectors() -> list:
    """读 persona/injection/05_core_stories/core_story_vectors.json（缓存）。"""
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


def fuse_and_truncate(corpus_items, behavior_items=None) -> list:
    """完整融合流程：RRF → 条数截断 → 字符预算截断。

    只融合 **corpus**（向量+LLM 判）与 **behavior**（按名字查表）。
    """
    from .constants import SOURCE_WEIGHTS as _W

    fused = rrf_fuse([corpus_items, behavior_items or []], weights=dict(_W))
    fused = fused[:RETRIEVAL_TOPK]
    return truncate_by_budget(fused)
