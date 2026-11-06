# -*- coding: utf-8 -*-
"""用户事件记忆：发生在用户与灰泽满之间、或用户亲口告诉她的具体事。

**为什么单开**（2026-11-05 用户定）：稳定信息（称呼/身份/相处/仪式/兴趣）住稳定卡；
“某天发生了什么”不是稳定属性，需要带日期、状态和主体，按话题/指代检索。
它的组织方式向 `persona/injection/11_corpus` 看齐：一个目录、事件正文、关键词/实体、向量。

存储（不入库）：
`user_memory/events/<uid>/events.json`、`vectors.json`、`keywords.json`

事件字段：
    id, date, subject(user|her|shared), source(user_told|shared|her_told),
    what, status(ongoing|ended|planned|unknown),
    type(health|relationship|study_work|pet|conflict|promise|shared_moment|other),
    entities[], confidence, expires_at, created_at, last_observed_at, history[]

状态是**当前状态**：车祸发生但伤还在 = ongoing；好了 = ended。
更新同一件事时只留一条事件，旧状态进 history。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from .constants import PROJECT_ROOT
from .rag import cosine_similarity, embed_query

EVENT_DIR = PROJECT_ROOT / "user_memory" / "events"
_LOCK = threading.Lock()

SUBJECTS = {"user", "her", "shared"}
SOURCES = {"user_told", "shared", "her_told"}
STATUSES = {"ongoing", "ended", "planned", "unknown"}
TYPES = {"health", "relationship", "study_work", "pet", "conflict", "promise",
         "shared_moment", "other"}

_STATUS_ZH = {"ongoing": "进行中", "ended": "已结束", "planned": "计划中",
              "unknown": "状态未知"}
_SOURCE_ZH = {"user_told": "他说的", "shared": "你们一起", "her_told": "她说的"}

# 明显是模板/段子/角色扮演的开头或形态：宁可不记，别把玩梗当经历。
_TEMPLATE_RE = re.compile(
    r"(重新介绍(一下)?自己|丫头[，,]|今天我给大家|文案|模板|复制|转发|段子|"
    r"角色扮演|我是[^，。]{0,6}(扮演|饰演)|声明|公告)")
_EXPLICIT_REF_RE = re.compile(
    r"(上次|之前|那天|那次|还记得|记得|我说过|你给我说|我们说好|你答应|"
    r"你忘了|以前|当时|刚刚说的|刚才说的)")

MAX_EVENTS = 200
MAX_WHAT_CHARS = 160
MAX_ENTITY_CHARS = 12


def _dir(user_id: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", str(user_id or ""))
    return EVENT_DIR / safe


def _event_file(user_id: str) -> Path:
    return _dir(user_id) / "events.json"


def _vectors_file(user_id: str) -> Path:
    return _dir(user_id) / "vectors.json"


def _keywords_file(user_id: str) -> Path:
    return _dir(user_id) / "keywords.json"


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save(path: Path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        print(f"⚠️ 事件记忆写入失败: {e}")


def _iso(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _date_text(value) -> str:
    """事件日期只保留 YYYY-MM-DD；解析不出就空串（不编）。"""
    text = str(value or "").strip()
    m = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def _norm_entities(values) -> list[str]:
    out = []
    for v in values or []:
        s = re.sub(r"[^\w一-鿿-]", "", str(v or "").strip())
        if 1 <= len(s) <= MAX_ENTITY_CHARS and s not in out:
            out.append(s)
        if len(out) >= 6:
            break
    return out


def _valid_event(raw: dict, now: datetime) -> dict | None:
    if not isinstance(raw, dict):
        return None
    what = str(raw.get("what") or "").strip()
    if len(what) < 4 or len(what) > MAX_WHAT_CHARS:
        return None
    if _TEMPLATE_RE.search(what):
        return None
    subject = str(raw.get("subject") or "").strip()
    if subject not in SUBJECTS:
        return None
    source = str(raw.get("source") or "").strip()
    if source not in SOURCES:
        source = "shared" if subject == "shared" else ("user_told" if subject == "user" else "her_told")
    status = str(raw.get("status") or "").strip()
    if status not in STATUSES:
        status = "unknown"
    etype = str(raw.get("type") or "").strip()
    if etype not in TYPES:
        etype = "other"
    try:
        confidence = max(0.0, min(float(raw.get("confidence") or 0.8), 1.0))
    except (TypeError, ValueError):
        confidence = 0.8
    expires = _iso(raw.get("expires_at"))
    if expires is not None and expires <= now:
        return None
    return {
        "date": _date_text(raw.get("date")),
        "subject": subject,
        "source": source,
        "what": what,
        "status": status,
        "type": etype,
        "entities": _norm_entities(raw.get("entities")),
        "confidence": confidence,
        "expires_at": expires.isoformat() if expires else "",
    }


def _event_id(user_id: str, event: dict) -> str:
    raw = f"{user_id}|{event['date']}|{event['type']}|{event['what']}"
    return "evt_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]


def _same_entity(a: dict, b: dict) -> bool:
    ea, eb = set(a.get("entities") or []), set(b.get("entities") or [])
    return bool(ea & eb)


def _near_in_time(a: str, b: str, days: int = 30) -> bool:
    da, db = _iso(a), _iso(b)
    if da is None or db is None:
        return True          # 日期缺失时允许更新，交给模型判断
    return abs((da - db).days) <= days


def _load_events(user_id: str) -> list[dict]:
    data = _load(_event_file(user_id), {"events": []})
    events = data.get("events") if isinstance(data, dict) else None
    return events if isinstance(events, list) else []


def _save_events(user_id: str, events: list[dict]) -> None:
    events = events[-MAX_EVENTS:]
    _save(_event_file(user_id), {"schema_version": 1, "events": events})
    kw: dict[str, list[str]] = {}
    for e in events:
        for ent in e.get("entities") or []:
            kw.setdefault(ent, [])
            if e["id"] not in kw[ent]:
                kw[ent].append(e["id"])
    _save(_keywords_file(user_id), {"entities": kw})


def recent_summary(user_id: str, limit: int = 8) -> str:
    """给提取提示词看：已记过的事件，方便判断“更新同一条”还是“新事件”。"""
    events = _active_events(user_id)[:limit]
    if not events:
        return "（还没有记过事件）"
    return "\n".join(
        f"- {e['id']} | {e.get('date', '')} | {_STATUS_ZH.get(e.get('status'), '')} | {e.get('what', '')}"
        for e in events
    )


def add_events(user_id: str, events: list[dict], *, source: dict | None = None,
               now: datetime | None = None) -> list[dict]:
    """提取候选 → 事件库。模型可用 `update_of` 声明这是同一件事的更新。"""
    now = now or datetime.now()
    accepted: list[dict] = []
    with _LOCK:
        stored = _load_events(user_id)
        by_id = {e.get("id"): e for e in stored}
        for raw in (events or [])[:10]:
            event = _valid_event(raw, now)
            if event is None:
                continue
            update_of = str(raw.get("update_of") or "").strip()
            target = by_id.get(update_of)
            if target is not None and target.get("type") == event["type"] and _same_entity(target, event) \
                    and _near_in_time(target.get("date", ""), event["date"]):
                target.setdefault("history", []).append({
                    "t": now.isoformat(),
                    "status": target.get("status"),
                    "what": target.get("what"),
                })
                for key in ("date", "subject", "source", "what", "status", "type",
                            "entities", "confidence", "expires_at"):
                    target[key] = event[key]
                target["last_observed_at"] = now.isoformat()
                accepted.append(target)
                continue
            event["id"] = _event_id(user_id, event)
            event["created_at"] = now.isoformat()
            event["last_observed_at"] = now.isoformat()
            event["history"] = []
            # 完全重复（同 id）只刷新时间，不新增
            dup = by_id.get(event["id"])
            if dup is not None:
                dup["last_observed_at"] = now.isoformat()
                accepted.append(dup)
                continue
            stored.append(event)
            by_id[event["id"]] = event
            accepted.append(event)
        _save_events(user_id, stored)
    return accepted


def _active_events(user_id: str, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now()
    out = []
    for e in _load_events(user_id):
        exp = _iso(e.get("expires_at"))
        if exp is not None and exp <= now:
            continue
        out.append(e)
    out.sort(key=lambda e: (e.get("last_observed_at") or "", e.get("date") or ""), reverse=True)
    return out


async def refresh_vectors(user_id: str, zhipu_client, *, force: bool = False) -> None:
    """为事件生成/更新向量；事件文本变了才重算（corpus 同款）。"""
    if zhipu_client is None:
        return
    events = _load_events(user_id)
    vectors = _load(_vectors_file(user_id), {"items": {}})
    items = vectors.get("items") if isinstance(vectors, dict) else {}
    changed = False
    for e in events:
        text = f"{e.get('date', '')} {e.get('what', '')} {' '.join(e.get('entities') or [])}".strip()
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        old = items.get(e["id"])
        if old and old.get("hash") == digest and not force:
            continue
        try:
            vec = await embed_query(zhipu_client, text)
        except Exception:
            continue
        if vec:
            items[e["id"]] = {"hash": digest, "vec": vec}
            changed = True
    if changed:
        _save(_vectors_file(user_id), {"items": items})


def _bigrams(text: str) -> set:
    s = re.sub(r"[^一-鿿A-Za-z0-9]", "", text or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _overlap(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    return len(A & B) / min(len(A), len(B)) if A and B else 0.0


async def retrieve(user_id: str, query: str, *, entities=None, explicit: bool = False,
                   zhipu_client=None, judge_client=None, judge_model: str | None = None,
                   limit: int | None = None, now: datetime | None = None,
                   avoid_text: str = "") -> list[dict]:
    """按“明确指涉/实体命中优先，向量兜底”取事件；最多 2 条。"""
    query = str(query or "").strip()
    if not query:
        return []
    events = _active_events(user_id, now=now)
    if avoid_text:
        # 最近 10 轮已经出现过的同一实体，别再作为“背景事件”注一遍。
        events = [e for e in events
                  if not any(ent in avoid_text for ent in (e.get("entities") or []))]
    if not events:
        return []
    wants = _norm_entities(entities) + _norm_entities(re.findall(r"[一-鿿]{2,6}", query)[:6])
    scored: list[tuple[float, dict]] = []
    vectors = (_load(_vectors_file(user_id), {"items": {}}) or {}).get("items") or {}
    qvec = None
    if zhipu_client is not None and vectors:
        try:
            qvec = await embed_query(zhipu_client, query)
        except Exception:
            qvec = None
    for e in events:
        text = f"{e.get('what', '')} {' '.join(e.get('entities') or [])}"
        ent_hit = bool(set(e.get("entities") or []) & set(wants))
        score = 3.0 if ent_hit else 0.0
        score += _overlap(query, text)
        if qvec and e["id"] in vectors:
            try:
                score += 1.5 * cosine_similarity(qvec, vectors[e["id"]]["vec"])
            except Exception:
                pass
        if score <= 0:
            continue
        scored.append((score, e, ent_hit))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1].get("last_observed_at") or ""))
    picked = [e for _, e, ent_hit in scored
              if (ent_hit or _EXPLICIT_REF_RE.search(query) or explicit)]
    if not picked:
        picked = [e for _, e, _ in scored[:1]]
    cap = 2 if (explicit or _EXPLICIT_REF_RE.search(query)) else 1
    picked = picked[:cap if limit is None else min(limit, cap)]
    # 向量兜底且没有实体/指涉锚点时，让 judge 再确认一次；失败不放行。
    if judge_client is not None and not any(
            set(e.get("entities") or []) & set(wants) for e in picked):
        picked = await _judge(judge_client, judge_model, query, picked)
    return picked[:cap]


async def _judge(client, model: str | None, query: str, events: list[dict]) -> list[dict]:
    if not events:
        return []
    try:
        from .config import _get_model_name, extract_chat_content, log_cache_usage
        from .constants import THINKING_DISABLED
        model = model or _get_model_name()
        numbered = "\n".join(
            f"{i + 1}. {e.get('date', '')} {e.get('what', '')}" for i, e in enumerate(events))
        prompt = (
            "判断这些历史事件和当前这句话是否真的相关。只按内容判断，不要因为撞到同一个词就保留；"
            "拿不准就不保留。\n\n【历史事件】\n" + numbered +
            "\n\n【当前这句话】\n" + query +
            '\n\n只输出 JSON：{"keep": [编号…]}；都不相关输出 {"keep": []}')
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=80, **THINKING_DISABLED)
        log_cache_usage(resp, "事件判定")
        content = extract_chat_content(resp)
        if "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        keep = json.loads(content).get("keep") or []
        return [e for i, e in enumerate(events, start=1) if i in keep]
    except Exception:
        return []


def render_event(e: dict) -> str:
    return (f"[{e.get('date') or '日期未知'} | {_STATUS_ZH.get(e.get('status'), '')} | "
            f"{_SOURCE_ZH.get(e.get('source'), '')}] {e.get('what', '')}")


def render_context(events: list[dict]) -> str:
    if not events:
        return ""
    return "\n".join(render_event(e) for e in events)


__all__ = ["EVENT_DIR", "add_events", "refresh_vectors", "recent_summary",
           "retrieve", "render_event", "render_context"]