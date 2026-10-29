#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`recall`：注入 v3 的唯一取数工具。

设计见 `outputs/injection_v3_registry_recall_spec.md` §5。
- 一个工具、一个 schema；kind 是提示不是硬过滤。
- 不允许传 user_id；调用方绑定当前用户/群。
- 天气/精确感知必须能解析出 owner/time，否则不返回（fail-closed）。
- 空结果明确告诉模型“没有查到，不要编”。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import context_probe, daily_memory, memory_v2, persona, retrieval
from .rag import cosine_similarity, embed_query

TOOL_NAME = "recall"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "查询你记得的事。只在回复真的需要时调用；不确定就不调用。"
            "kind 省略=在所有记忆里找。问天气时必须给 subject 和 when；"
            "解析不了就不返回。返回空表示没有查到，不要编。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "用自然语言写要找什么，例如“用户的名字”“今天天气”“她喜欢的食物”"},
                "kind": {"type": "string",
                         "enum": ["fact", "preference", "interaction_preference", "story",
                                  "corpus", "term", "habit", "commitment", "daily",
                                  "weather", "perception"]},
                "subject": {"type": "string", "enum": ["user", "her"]},
                "when": {"type": "string", "enum": ["now", "today", "session", "any"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 6},
            },
            "required": ["query"],
        },
    },
}

ALL_KINDS = ("fact", "preference", "interaction_preference", "story", "corpus",
             "term", "habit", "commitment", "daily", "weather", "perception")
KIND_ALIASES = {"core_story": "story", "core_stories": "story", "perceptions": "perception"}


@dataclass
class RecallItem:
    id: str
    kind: str
    subject: str
    validity: str
    text: str
    authority: str = "canon"
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "subject": self.subject,
                "validity": self.validity, "text": self.text}


def _s(v) -> str:
    return str(v or "").strip()


def _bigrams(text: str) -> set:
    t = re.sub(r"[^一-龥A-Za-z0-9]", "", text or "")
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _overlap(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def _kind(value) -> str:
    k = _s(value)
    return KIND_ALIASES.get(k, k)


def _subject_from_query(query: str) -> str:
    q = query or ""
    her = ("你那边", "你那里", "你那", "你的名字", "你叫什么", "你住哪", "你家",
           "灰泽满", "hzm", "澳洲", "墨尔本")
    user = ("我这边", "我这里", "我们这", "我这", "我今天", "我那边", "我叫什么",
            "我是谁", "我记得")
    if any(x in q for x in user):
        return "user"
    if any(x in q for x in her):
        return "her"
    return ""


def _validity_of_memory(m: dict) -> str:
    scope = _s(m.get("temporal_scope"))
    if scope == "past":
        return "past"
    if m.get("expires_at"):
        return "day"
    return "stable"


_NAME_HINTS = ("名字", "叫什么", "称呼", "叫我", "昵称", "我是谁", "记得我")
_LOCATION_HINTS = ("在哪", "哪里", "城市", "地点", "住", "所在地", "时区")
_STYLE_HINTS = ("相处", "态度", "怎么对我", "称呼", "昵称", "亲昵", "安慰", "怼", "温柔")
_HABIT_HINTS = ("习惯", "每天", "平时", "早上", "晚上", "作息", "跑步", "起床")
_COMMIT_HINTS = ("答应", "约定", "说好", "承诺", "还没", "什么时候")


def _memory_items(store, user_id: str, query: str, kind: str,
                  now: datetime) -> list[RecallItem]:
    q = query or ""
    active = store._active_memories(user_id, now=now)
    out: list[RecallItem] = []

    def add(m: dict, score: float, text: str | None = None):
        out.append(RecallItem(
            id=str(m.get("id") or f"{m.get('kind')}:{m.get('key')}"),
            kind=kind,
            subject="user",
            validity=_validity_of_memory(m),
            text=text or f"{m.get('key')}：{m.get('value')}",
            authority="user_stated",
            score=score,
            meta={"key": m.get("key")},
        ))

    if kind == "fact":
        for m in active:
            if m.get("kind") != "fact" or _s(m.get("key")).startswith("habit_"):
                continue
            key = _s(m.get("key"))
            value = _s(m.get("value"))
            score = memory_v2._fact_relevance(q, m)
            if key in {"preferred_name", "name", "nickname"} and any(h in q for h in _NAME_HINTS):
                score = max(score, 1.0)
            if key in {"location", "city", "timezone"} and any(h in q for h in _LOCATION_HINTS):
                score = max(score, 1.0)
            if score > 0:
                add(m, score)

    elif kind == "interaction_preference":
        for m in active:
            if m.get("kind") != "interaction_preference":
                continue
            score = memory_v2._relevance(q, m)
            if any(h in q for h in _STYLE_HINTS):
                score = max(score, 0.5)
            if score > 0:
                add(m, score, text=f"{m.get('key')}：{m.get('value')}")

    elif kind == "commitment":
        for m in active:
            if m.get("kind") != "commitment":
                continue
            score = memory_v2._relevance(q, m)
            if any(h in q for h in _COMMIT_HINTS):
                score = max(score, 0.5)
            if score > 0:
                add(m, score, text=f"约定：{m.get('value')}")

    elif kind == "habit":
        for m in active:
            key = _s(m.get("key"))
            if m.get("kind") != "fact" or not key.startswith("habit_"):
                continue
            score = memory_v2._habit_relevance(q, m)
            if any(h in q for h in _HABIT_HINTS):
                score = max(score, 0.5)
            if score > 0:
                add(m, score, text=f"习惯：{m.get('value')}")

    out.sort(key=lambda it: (-it.score, it.id))
    return out


def _daily_items(user_id: str, now: datetime, query: str) -> list[RecallItem]:
    entries = daily_memory.recall_items(user_id, now=now)
    out = []
    for e in entries:
        text = daily_memory.render_entry(e, now)
        if not text:
            continue
        score = _overlap(query, text)
        out.append(RecallItem(id=f"daily:{e.get('t')}", kind="daily", subject="user",
                              validity="today", text=text, authority="user_stated",
                              score=score))
    out.sort(key=lambda it: (-it.score, it.id))
    return out


def _preference_items(query: str) -> list[RecallItem]:
    hits = retrieval.retrieve_preferences(query)
    if not hits and any(h in (query or "") for h in ("喜欢", "爱吃", "偏好", "口味", "讨厌")):
        entries = retrieval.load_preferences()
        hits = [{"id": e.get("id", ""), "category": e.get("category", ""),
                 "text": e.get("text", ""), "score": 0.4} for e in entries[:4]]
    return [RecallItem(id=f"pref:{h.get('id')}", kind="preference", subject="her",
                       validity="stable", text=f"{h.get('category', '')}：{h.get('text', '')}",
                       authority="canon", score=float(h.get("score") or 0.0))
            for h in hits]


def _core_story_items(query: str, query_vector) -> list[RecallItem]:
    hits = retrieval.retrieve_core_stories(query, query_vector) if query_vector else []
    if not hits and query:
        stories = retrieval.load_core_story_vectors() or []
        scored = sorted((( _overlap(query, s.get("text", "")), s) for s in stories),
                        key=lambda x: -x[0])
        hits = [{"id": s.get("id", ""), "category": s.get("category", ""),
                 "text": s.get("text", ""), "score": sc}
                for sc, s in scored[:3] if sc > 0]
    return [RecallItem(id=f"story:{h.get('id')}", kind="story", subject="her",
                       validity="past", text=h.get("text", ""), authority="canon",
                       score=float(h.get("score") or 0.0))
            for h in hits]


async def _corpus_items(query: str, query_vector, deepseek_client) -> list[RecallItem]:
    if not query_vector:
        return []
    candidates = retrieval.retrieve_corpus_candidates(query, query_vector)
    if not candidates:
        return []
    if deepseek_client is not None:
        try:
            from .corpus_judge import judge_corpus
            from .config import aux_clients_or
            # 直播记忆判走辅助链（2026-10-29）；没配 AUX_* 时退回调用方给的客户端
            judge_client, _ = aux_clients_or(lambda: (deepseek_client, None))
            picked = await judge_corpus(judge_client, query, candidates)
        except Exception:
            picked = []
    else:
        picked = []
    return [RecallItem(id=f"corpus:{it.item_id}", kind="corpus", subject="her",
                       validity="past", text=it.text, authority="canon",
                       score=float(it.score or 0.0))
            for it in picked]


def _term_items(query: str) -> list[RecallItem]:
    q = query or ""
    out = []
    for t in persona.load_terms() or []:
        if t.get("priority") == "always":
            continue
        kw = _s(t.get("keyword"))
        keys = [kw] + [str(a) for a in (t.get("aliases") or []) if a]
        pattern = t.get("pattern")
        hit = any(k and k in q for k in keys) or (pattern and re.search(pattern, q))
        if not hit:
            continue
        parts = [_s(t.get("meaning"))]
        if t.get("reaction"):
            parts.append(f"被提到时：{t['reaction']}")
        if t.get("usage"):
            parts.append(f"用词规则：{t['usage']}")
        out.append(RecallItem(id=f"term:{kw}", kind="term", subject="world",
                              validity="stable", text=f"{kw}：{'；'.join(p for p in parts if p)}",
                              authority="canon", score=1.0))
    return out


async def _weather_items(subject: str, user_city: str, now: datetime) -> list[RecallItem]:
    if subject not in {"user", "her"}:
        return []
    city = user_city if subject == "user" else ""
    if subject == "user" and not city:
        return []  # 不拿她的城市冒充对方所在地
    await context_probe.warm_weather(city)
    text = context_probe.weather_now(city)
    if not text:
        return []
    return [RecallItem(id=f"weather:{subject}:{now.strftime('%Y%m%d%H')}",
                       kind="weather", subject=subject, validity="now",
                       text=text, authority="api", score=1.0)]


def _perception_items(now: datetime) -> list[RecallItem]:
    text = context_probe.perception_now(now=now)
    if not text:
        return []
    return [RecallItem(id=f"perception:{now.strftime('%Y%m%d%H%M')}",
                       kind="perception", subject="her", validity="now",
                       text=text, authority="api", score=1.0)]


async def recall(
    *,
    user_id: str,
    query: str,
    kind: str | None = None,
    subject: str | None = None,
    when: str | None = None,
    limit: int = 6,
    now: datetime | None = None,
    store=None,
    user_city: str = "",
    deepseek_client=None,
    zhipu_client=None,
    is_group: bool = False,
) -> list[RecallItem]:
    """按统一 schema 取数。返回按 score 排序的 RecallItem；空 = 没查到。"""
    now = now or datetime.now()
    query = _s(query)
    if not query:
        return []
    kind = _kind(kind)
    when = _s(when) or "any"
    limit = max(1, min(int(limit or 6), 6))
    requested = [kind] if kind else list(ALL_KINDS)
    explicit_subject = _s(subject) or _subject_from_query(query)
    # 泛问“今天几度/天气怎么样”默认问的是对方所在地；若对方城市未知，
    # `_weather_items` 会直接返回空，不拿她的城市冒充。
    if kind == "weather" and not explicit_subject:
        explicit_subject = "user"

    items: list[RecallItem] = []
    query_vector = None
    if any(k in requested for k in ("story", "corpus")) and zhipu_client is not None:
        try:
            query_vector = await embed_query(zhipu_client, query)
        except Exception:
            query_vector = None

    for k in requested:
        if k == "fact" and store is not None and user_id and not is_group:
            items += _memory_items(store, user_id, query, "fact", now)
        elif k == "interaction_preference" and store is not None and user_id:
            items += _memory_items(store, user_id, query, "interaction_preference", now)
        elif k == "commitment" and store is not None and user_id and not is_group:
            items += _memory_items(store, user_id, query, "commitment", now)
        elif k == "habit" and store is not None and user_id and not is_group:
            items += _memory_items(store, user_id, query, "habit", now)
        elif k == "daily" and user_id and not is_group:
            items += _daily_items(user_id, now, query)
        elif k == "preference":
            items += _preference_items(query)
        elif k == "story":
            items += _core_story_items(query, query_vector)
        elif k == "corpus":
            items += await _corpus_items(query, query_vector, deepseek_client)
        elif k == "term":
            items += _term_items(query)
        elif k == "weather":
            items += await _weather_items(explicit_subject, user_city, now)
        elif k == "perception":
            items += _perception_items(now)

    # 主体/时间过滤
    subject = explicit_subject
    if kind == "weather" and subject not in {"user", "her"}:
        return []
    filtered = []
    for it in items:
        if subject and it.kind in {"fact", "daily", "commitment", "habit",
                                   "interaction_preference", "weather"}:
            if it.subject != subject:
                continue
        if when == "now" and it.validity not in {"now"}:
            continue
        if when == "today" and it.validity not in {"today", "now"}:
            continue
        if when == "session" and it.validity not in {"session", "today", "now"}:
            continue
        filtered.append(it)

    # 去重 + 多样性（跨类型时每类最多 2 条）
    filtered.sort(key=lambda it: (-it.score, it.kind, it.id))
    out, seen, per_kind = [], set(), {}
    for it in filtered:
        if it.id in seen:
            continue
        if not kind:
            if per_kind.get(it.kind, 0) >= 2:
                continue
        seen.add(it.id)
        per_kind[it.kind] = per_kind.get(it.kind, 0) + 1
        out.append(it)
        if len(out) >= limit:
            break
    return out


async def recall_for_tool(arguments: dict, *, user_id: str, store, user_city: str,
                          deepseek_client=None, zhipu_client=None,
                          is_group: bool = False) -> tuple[str, list[RecallItem]]:
    """tool loop 的入口：解析 JSON 参数，返回 (给模型看的文本, 条目)。"""
    try:
        args = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
    except Exception:
        args = {}
    query = _s(args.get("query"))
    if not query:
        return "没有查到；不要编造。", []
    try:
        limit = int(args.get("limit") or 6)
    except Exception:
        limit = 6
    items = await recall(
        user_id=user_id, query=query, kind=args.get("kind"),
        subject=args.get("subject"), when=args.get("when"), limit=limit,
        store=store, user_city=user_city, deepseek_client=deepseek_client,
        zhipu_client=zhipu_client, is_group=is_group,
    )
    if not items:
        return "没有查到；不要编造。", []
    return json.dumps([it.as_dict() for it in items], ensure_ascii=False), items


__all__ = ["TOOL_NAME", "TOOL_SCHEMA", "RecallItem", "recall", "recall_for_tool"]