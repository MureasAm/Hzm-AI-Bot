"""萌娘百科按需查询：实体名 → 一段短背景（运行时、缓存、失败静默）。

只走 MediaWiki 的 `prop=extracts`——实测萌娘百科只放行这个 action；
`list=search` / `action=parse` 都返回 `action-notallowed`，HTML 搜索页还有 Cloudflare。
所以这里不做搜索：按实体名精确查，带 `redirects=1` 让「小圆」这类别名落到正式词条。
查不到 / 超时 / 消歧义都返回 None，绝不影响正常聊天。
"""
import asyncio
import json
import os
import re
import time

import httpx

from .constants import PROJECT_ROOT

WIKI_APIS = (
    "https://zh.moegirl.org.cn/api.php",
    "https://mzh.moegirl.org.cn/api.php",
    "https://moegirl.uk/api.php",
)
WIKI_CACHE_FILE = PROJECT_ROOT / "data" / "wiki_cache.json"
WIKI_TTL = 30 * 24 * 3600          # 基础字段（社团/期数/出道）30 天
WIKI_MISS_TTL = 3 * 24 * 3600      # 查不到也缓存 3 天，免得每轮都打
WIKI_FRESH_RE = re.compile(r"最近|最新|现在|这周|本周|今天|刚刚|新动态|更新|近来|近期|最近在")
WIKI_TIMEOUT = 5.0
WIKI_MAX_CHARS = 120
_WIKI_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_STOP_TERMS = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这个", "那个",
    "什么", "怎么", "为什么", "哪里", "现在", "今天", "明天", "哈哈", "真的",
}
_cache: dict | None = None


def _norm(term: str) -> str:
    return re.sub(r"\s+", " ", str(term or "")).strip()


def _cache_key(term: str) -> str:
    return _norm(term).casefold()


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        raw = json.loads(WIKI_CACHE_FILE.read_text(encoding="utf-8"))
        _cache = raw.get("items") if isinstance(raw, dict) else {}
    except Exception:
        _cache = {}
    return _cache


def _save_cache() -> None:
    items = _load_cache()
    if len(items) > 500:   # 只留最近 300 条，避免缓存文件无限长
        keep = sorted(items.items(), key=lambda kv: kv[1].get("ts", 0), reverse=True)[:300]
        items.clear()
        items.update(keep)
    try:
        WIKI_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = WIKI_CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, WIKI_CACHE_FILE)
    except OSError as e:
        print(f"⚠️ wiki 缓存写失败（忽略）: {e}")


def clean_extract(text: str) -> str:
    """把 MediaWiki extract 清成一段可直接注入的短文本。"""
    t = str(text or "")
    t = re.sub(r"\s*=+\s*[^=\n]{1,40}\s*=+\s*", " ", t)   # 去掉 == 标题 ==
    t = t.replace("​", "").replace("﻿", "")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= WIKI_MAX_CHARS:
        return t
    cut = t[:WIKI_MAX_CHARS]
    idx = max(cut.rfind("。"), cut.rfind("！"), cut.rfind("？"), cut.rfind("；"))
    if idx >= WIKI_MAX_CHARS * 0.5:
        cut = cut[:idx + 1]
    else:
        cut = cut.rstrip()
    return cut.rstrip("，,、 ") + "…"


def _cache_get(term: str) -> dict | None:
    item = _load_cache().get(_cache_key(term))
    if not isinstance(item, dict):
        return None
    ttl = WIKI_MISS_TTL if item.get("missing") else WIKI_TTL
    if time.time() - float(item.get("ts", 0)) > ttl:
        return None
    return item


def _cache_put(term: str, data: dict) -> None:
    _load_cache()[_cache_key(term)] = {**data, "ts": time.time()}
    _save_cache()


def _pick_page(data: dict) -> tuple[str, str, str] | None:
    """从 query 响应里取 (标题, 清洗后摘要, 重定向后的正式标题)。"""
    query = data.get("query") or {}
    pages = query.get("pages") or {}
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page or not page.get("extract"):
        return None
    title = str(page.get("title") or "").strip()
    extract = clean_extract(page.get("extract") or "")
    if not title or not extract:
        return None
    if "消歧义" in title or "可以指" in extract:
        return None
    redirects = query.get("redirects") or []
    resolved = str(redirects[0].get("to") or title) if redirects else title
    return title, extract, resolved


async def _query_one(api: str, term: str) -> dict:
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "exintro": "1", "explaintext": "1", "titles": term, "redirects": "1",
    }
    async with httpx.AsyncClient(timeout=WIKI_TIMEOUT, headers={"User-Agent": _WIKI_UA},
                                 follow_redirects=True, trust_env=False) as client:
        resp = await client.get(api, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    picked = _pick_page(resp.json())
    if not picked:
        return {"missing": True}
    title, extract, resolved = picked
    return {"title": title, "extract": extract, "resolved": resolved, "api": api}


async def lookup(term: str, force: bool = False) -> dict | None:
    """按实体名查一次；命中/未命中都缓存。失败返回 None。

    `force=True` 跳过缓存（用户问“最近/最新/现在”时用），查完仍然写回缓存。
    三个镜像**并发**查：有的镜像没有新词条（如 moegirl.uk），不能让它的 missing
    覆盖主站命中；只有全部镜像都明确返回 missing 才缓存「查不到」。超时不缓存，
    下一轮还可能恢复。
    """
    term = _norm(term)
    if not term:
        return None
    if not force:
        cached = _cache_get(term)
        if cached is not None:
            return None if cached.get("missing") else cached

    last_errors: list = []
    for attempt in range(2):
        results = await asyncio.gather(
            *(_query_one(api, term) for api in WIKI_APIS), return_exceptions=True)
        hit = None
        missing = 0
        errors: list = []
        for api, res in zip(WIKI_APIS, results):
            if isinstance(res, Exception):
                errors.append(f"{api}: {type(res).__name__}")
                continue
            if res.get("missing"):
                missing += 1
                continue
            if hit is None:   # 按 WIKI_APIS 顺序优先主站
                hit = res
        if hit is not None:
            _cache_put(term, hit)
            print(f"[wiki] 「{term}」→「{hit['title']}」：{hit['extract'][:40]}")
            return hit
        if missing == len(WIKI_APIS):
            _cache_put(term, {"missing": True})
            return None
        last_errors = errors
        if attempt == 0:
            await asyncio.sleep(0.6)   # 瞬时超时/抖动，短等再试一次
    print(f"⚠️ wiki 查询失败（不缓存，下一轮可重试）: 「{term}」 {last_errors[:2]}")
    return None


def _eligible(entity: str, user_msg: str, known: set) -> bool:
    e = _norm(entity)
    if not e or len(e) < 2 or len(e) > 20:
        return False
    if e in _STOP_TERMS or e.casefold() in known:
        return False
    # 只查用户这句话里真的出现过的词，模型脑补出来的实体不查
    return e in (user_msg or "") or e.casefold() in (user_msg or "").casefold()


async def wiki_context_for(user_msg: str, entities, known_terms=None,
                           max_items: int = 1) -> str:
    """从 probe 的 entities 里挑第一个可查的，返回一段「标题：摘要」。

    每轮最多查 `max_items` 个；查不到 / 超时 / 已知词都直接跳过，返回空串。
    """
    known = {_cache_key(x) for x in (known_terms or set()) if str(x).strip()}
    force = bool(WIKI_FRESH_RE.search(user_msg or ""))
    picked = 0
    notes = []
    for entity in entities or []:
        if picked >= max_items:
            break
        if not _eligible(entity, user_msg, known):
            continue
        data = await lookup(entity, force=force)
        if not data:
            continue
        name = data.get("title") or _norm(entity)
        if name != _norm(entity):
            note = f"{_norm(entity)}（百科词条：{name}）：{data['extract']}"
        else:
            note = f"{name}：{data['extract']}"
        notes.append(note)
        picked += 1
    return "；".join(notes)


def reset_cache_for_tests() -> None:
    """仅测试用：丢掉模块内缓存，让 monkeypatch 的缓存路径生效。"""
    global _cache
    _cache = None