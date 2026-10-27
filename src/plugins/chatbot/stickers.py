# -*- coding: utf-8 -*-
"""表情包：判断她这句话配哪张表情，十分对应时才发。

**为什么要先打标**：表情包文件名是一串哈希，模型不知道每张画的是什么、什么时候该发。
`scripts/label_stickers.py` 离线用视觉模型问出 desc/tags/use_when，存
`persona/decision/stickers.json`。加新表情：丢进 assets/stickers/ 再跑一次那个脚本。

**为什么用 LLM 判、不用向量阈值**：要的是"**十分**对应"。阈值调不准——
群聊接话门那边已经吃过教训（同一份数据、判据稍动就从 0% 跳到 74%）。
24 张的清单塞进提示词很便宜，让模型直接判"有没有十分对应的、没有就 null"更稳。

**频率**：真人也不会每轮都甩表情包。调用方负责冷却（见 chat_window 的 `_UserWindow`）。
"""
import json
import os
import random
import re
import time

from .constants import PROJECT_ROOT, THINKING_DISABLED, STICKER_FILE
from .config import log_cache_usage, _get_model_name, extract_chat_content, _get_clients, get_flag
from .rag import embed_query, cosine_similarity

#: 表情包决策流水（统计用）。**唯一的数据源**——在此之前"发了哪张"只 print 到控制台，
#: 落盘一份才谈得上"哪些图从没被用过 / 哪一类被反复用"。读它的是 `scripts/sticker_stats.py`。
#: 不入库（见 .gitignore）；一行失败静默，绝不因为统计拖垮发送。
STICKER_LOG = PROJECT_ROOT / "data" / "sticker_log.jsonl"

_stickers_cache = None


def log_event(event: str, *, session="", kind: str = "", sticker: dict | None = None,
              why: str = "") -> None:
    """记一行表情包决策。event ∈ sent / none / cooldown / sent_failed。"""
    try:
        rec = {
            "t": round(time.time(), 3),
            "session": str(session or ""),
            "kind": kind,
            "event": event,
            "id": (sticker or {}).get("id", ""),
            "group": (sticker or {}).get("group", ""),
            "why": (why or "")[:60],
        }
        STICKER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(STICKER_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ⚠️ **别重排这段提示词**（2026-10-09）：`{{reply}}/{{avoid}}` 这些**每轮都变**的变量必须留在
# 后面。DeepSeek 的缓存按**前缀**匹配，变量放前面（原来在 14%）会让后面整份表情目录
# （~1700 字）永远不命中——实测变量在前的写法命中率 **0%**，挪到后面 **68%**。
STICKER_PROMPT = """灰泽满刚说了一句话，判断要不要给她配一张表情包。

规则：
- **只有表情和这句话十分对应时才发。宁可不发**——发错比不发难看得多。
- 大多数回复都不配表情包（真人也是聊好几轮才用一张）。
- 表情是**配她自己这句话的情绪**，不是评论别人。

【可选表情包】
{options}

【她刚说的】
{reply}{avoid}

只输出 JSON：{{"id": "选中的表情 id" 或 null, "why": "不超过15字"}}"""


def sticker_enabled() -> bool:
    """环境变量 STICKER=0 可整体关掉（快速回滚用），缺省开。"""
    return os.environ.get("STICKER", "1") != "0"


# ==================== 非 LLM 快路（2026-10-27） ====================
# 表情包是装饰性判断：先规则闸掉明显不值得的回复，再用 embedding 选组；
# 高置信度直接发，低置信度才回退 LLM（默认关闭，A/B 通过后开）。
STICKER_FAST_MIN = 0.58
STICKER_FAST_MARGIN = 0.04
_GROUP_VECTORS = None
_GROUP_VEC_PATH = PROJECT_ROOT / "data" / "sticker_group_vectors.json"

_NEUTRAL_RE = re.compile(
    r"^(嗯|哦|噢|好|好的|行|可以|知道|知道了|收到|没问题|ok|okay)[。！!…\.]*$", re.I)
_FACT_RE = re.compile(
    r"(周表|周[一二三四五六日天]|几点|几号|开播|下播|直播时间|时差|上课|下课|作业|考试|签到)")
_EMO_RE = re.compile(r"(才不|才没|少来|哼|呜|哭|笑|哈|？|\?|！|!|…|嘛|啦|呢|啊|呀|吧|~)")


def fast_enabled() -> bool:
    return get_flag("STICKER_FAST", "0").strip().lower() in ("1", "true", "on", "yes")


def fast_fallback_enabled() -> bool:
    return get_flag("STICKER_FAST_FALLBACK", "1").strip().lower() in ("1", "true", "on", "yes")


def _fast_float(name: str, default: float) -> float:
    try:
        return float(get_flag(name, str(default)))
    except (TypeError, ValueError):
        return default


def fast_gate(reply: str) -> bool:
    """这条回复值不值得进入表情包判断。只拦明显不值得的，宁漏勿错。"""
    t = (reply or "").strip()
    if not t or len(t) < 4:
        return False
    if "小满现在不在" in t:
        return False
    if _NEUTRAL_RE.match(t):
        return False
    if _FACT_RE.search(t) and not _EMO_RE.search(t):
        return False
    return True


def _group_texts(items: list) -> dict:
    groups = {}
    for s in items:
        groups.setdefault(s.get("group") or "其他", []).append(s)
    out = {}
    for g, ss in groups.items():
        parts = []
        for s in ss:
            tag = "/".join(s.get("tags") or [])
            parts.append(f"{s.get('desc', '')}（{tag}）适合：{s.get('use_when', '')}")
        out[g] = f"{g}：" + "；".join(parts)
    return out


async def _load_group_vectors(zhipu_client=None) -> dict:
    global _GROUP_VECTORS
    if _GROUP_VECTORS is not None:
        return _GROUP_VECTORS
    items = load_stickers()
    texts = _group_texts(items)
    if not texts:
        _GROUP_VECTORS = {}
        return _GROUP_VECTORS
    try:
        if _GROUP_VEC_PATH.exists():
            data = json.loads(_GROUP_VEC_PATH.read_text(encoding="utf-8"))
            if data.get("texts") == texts and isinstance(data.get("vectors"), dict):
                _GROUP_VECTORS = data["vectors"]
                return _GROUP_VECTORS
    except Exception:
        pass
    if zhipu_client is None:
        _, zhipu_client = _get_clients()
    vectors = {}
    for g, text in texts.items():
        v = await embed_query(zhipu_client, text)
        if v:
            vectors[g] = v
    _GROUP_VECTORS = vectors
    try:
        _GROUP_VEC_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GROUP_VEC_PATH.write_text(json.dumps({"texts": texts, "vectors": vectors},
                                              ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return vectors


async def fast_pick(reply: str, avoid_groups: list | None = None,
                    zhipu_client=None) -> dict | None:
    """Embedding 选组；只有高置信度才返回，否则 None（交给 LLM 或直接不发）。"""
    q = await embed_query(zhipu_client, reply) if zhipu_client is not None else None
    if q is None:
        if zhipu_client is None:
            _, zhipu_client = _get_clients()
        q = await embed_query(zhipu_client, reply)
    if not q:
        return None
    vectors = await _load_group_vectors(zhipu_client)
    if not vectors:
        return None
    scored = sorted(((cosine_similarity(q, v), g) for g, v in vectors.items()),
                    key=lambda x: -x[0])
    if not scored:
        return None
    min_score = _fast_float("STICKER_FAST_MIN", STICKER_FAST_MIN)
    margin = _fast_float("STICKER_FAST_MARGIN", STICKER_FAST_MARGIN)
    top_score, top_group = scored[0]
    if top_score < min_score:
        return None
    if len(scored) > 1 and top_score - scored[1][0] < margin:
        return None
    avoid = set(avoid_groups or [])
    for score, group in scored:
        if score < min_score:
            break
        if group in avoid:
            continue
        candidates = [s for s in load_stickers() if (s.get("group") or "其他") == group]
        if candidates:
            hit = dict(random.choice(candidates))
            hit["_source"] = "vector"
            return hit
    return None


def load_stickers() -> list:
    """加载表情标注（模块级缓存）。文件缺失/写坏时返回空列表，调用方当作没表情。"""
    global _stickers_cache
    if _stickers_cache is not None:
        return _stickers_cache
    _stickers_cache = []
    if STICKER_FILE.exists():
        try:
            data = json.loads(STICKER_FILE.read_text(encoding="utf-8"))
            items = data.get("stickers", []) if isinstance(data, dict) else []
            # 文件不存在于磁盘的条目要剔掉：标注是离线生成的，图可能被手动删过
            _stickers_cache = [
                s for s in items
                if isinstance(s, dict) and s.get("id") and s.get("file")
                and (PROJECT_ROOT / s["file"]).exists()
            ]
        except (json.JSONDecodeError, OSError):
            _stickers_cache = []
    return _stickers_cache


def _options_text(items: list) -> str:
    lines = []
    for s in items:
        tag = "/".join(s.get("tags") or [])
        grp = s.get("group") or ""
        lines.append(f"- {s['id']}［{grp}］：{s.get('desc', '')}（{tag}）适合：{s.get('use_when', '')}")
    return "\n".join(lines)


async def pick_sticker(deepseek_client, reply: str, avoid_groups: list | None = None,
                       session="", kind: str = "", zhipu_client=None) -> dict | None:
    """给这句话挑一张表情；没有十分对应的返回 None。

    `avoid_groups`：**最近发过的类别**（不是单张）——别连着发观感相近的图。
    用户 2026-10-04 定：同一类里可能有好几张（委屈 3 张、无语无奈 4 张），
    按单张去重挡不住"换一张还是同一个味道"。

    失败一律返回 None（**不发**）——发错表情比不发难看得多，这里和接话门相反，
    宁可少发不可发错。
    """
    if not sticker_enabled() or not (reply or "").strip():
        return None
    items = load_stickers()
    if not items:
        return None

    # 快路①：明显不值得判断的回复直接跳过 LLM（仅 STICKER_FAST=1 时启用）。
    if fast_enabled() and not fast_gate(reply):
        log_event("none", session=session, kind=kind, why="fast_gate")
        return None
    # 快路②：embedding 选组；高置信度直接发，低置信度才回退 LLM。
    if fast_enabled():
        hit = await fast_pick(reply, avoid_groups, zhipu_client)
        if hit:
            print(f"[表情包] 向量选 {hit['id']}［{hit.get('group', '')}］")
            return hit
        if not fast_fallback_enabled():
            log_event("none", session=session, kind=kind, why="fast_vector")
            return None

    avoid = ""
    if avoid_groups:
        names = "、".join(dict.fromkeys(avoid_groups))
        avoid = f"\n\n【最近已经发过的类别，别再选这些类别里的】{names}"

    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": STICKER_PROMPT.format(
                reply=reply.strip(), options=_options_text(items), avoid=avoid)}],
            temperature=0,
            max_tokens=80,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "表情包")
        content = extract_chat_content(resp)
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        content = content.strip()
        if content[:4].lower() == "json":
            content = content[4:].strip()
        data = json.loads(content)
        sid = data.get("id")
        if not sid:
            log_event("none", session=session, kind=kind, why=str(data.get("why") or ""))
            return None
        hit = next((s for s in items if s["id"] == sid), None)  # 拦住模型编的 id
        if hit is None:
            print(f"⚠️ 表情判定给了不存在的 id={sid!r}，忽略")
            log_event("none", session=session, kind=kind, why=f"bad_id:{sid}")
            return None
        # 「最近发过的类别」在提示词里也说了，但**模型不一定听**（实测 2 次里 1 次照选）。
        # 所以这里再用代码拦一道：宁可这轮不发，也不连着甩同一类的。
        if hit.get("group") and hit["group"] in (avoid_groups or []):
            print(f"[表情包] 判定选了最近发过的类别「{hit['group']}」，这轮不发")
            log_event("none", session=session, kind=kind, sticker=hit, why="avoid_group")
            return None
        print(f"[表情包] 选 {sid}［{hit.get('group', '')}］（{str(data.get('why') or '')[:20]}）")
        return hit
    except Exception as e:
        print(f"⚠️ 表情包判定失败（不发）: {e}")
        log_event("none", session=session, kind=kind, why=f"error:{e}")
        return None
