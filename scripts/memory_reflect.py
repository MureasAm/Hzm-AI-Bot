# -*- coding: utf-8 -*-
"""批量反思：扫一个用户的多天聊天记录，找**跨天重复**的仪式/习惯/偏好。

为什么有它（2026-10-23 用户提出）：逐轮提取看不到“连续很多天”，而固定仪式、作息、
习惯恰恰是日复一日里长出来的（905860780 的晚安仪式、2151983431 的早操都是这么漏的）。

两阶段 map-reduce：
  1) 每天一份 digest：当天反复出现的要求/仪式/习惯/情绪触发点（缓存到
     `user_memory/reflection/<user>/<日期>.json`，只跑新增的天）；
  2) 把所有 digest 合成一次跨天反思：≥2 天或 ≥3 次、会改变以后回应的模式。

默认**只产候选**到 `outputs/reflection/<user>.json`，人工看过后再加 `--apply` 写库。
候选到记忆条目的映射是确定性的（`candidate_to_item`）：ritual/habit/preference/fact。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIGEST_DIR = ROOT / "user_memory" / "reflection"
OUT_DIR = ROOT / "outputs" / "reflection"
MAX_DAY_CHARS = 7000
MAX_MSG_CHARS = 160

DAY_PROMPT = """下面是一个用户和“灰泽满”在某一天的聊天记录。请只做**记录**，不下长期结论。
找出当天**反复出现或明确的要求/仪式/习惯/情绪触发点**，以及可能影响以后回应的用户事实。
规则：
- 只写当天真实出现的内容，标注出现次数或原话；一次性闲聊不要写。
- 不要评价、不要脑补；不要记录角色扮演/玩梗里的设定。
- 同一活动有**多个频率**时分开写（“每天早上跑操”和“每周三次 3 公里”不是一件事）。
- 特别留意：反复要求同一件事（固定说法、固定流程）、固定时段的行为、称呼要求、
  雷区、明确的喜好/厌恶。
只输出 JSON：
{{"items":[{{"kind":"ritual|habit|preference|fact|boundary","value":"一句话","evidence":"原话/次数"}}]}}

聊天记录：
{chat}
"""

REFLECT_PROMPT = """下面是同一个用户连续多天与“灰泽满”的聊天要点（按天）。请做**跨天反思**，
找出值得进入长期记忆的**重复模式**，而不是一次性事件。

判据：
- 至少跨 2 天、或同一模式出现 3 次以上；
- 要能改变灰泽满以后怎么回应（固定仪式、固定说法、固定时段习惯、称呼、雷区、长期偏好）；
- 一次性的经历/情绪不算长期记忆（除非明确改变了相处方式）；
- 角色扮演/玩梗里的设定不要当事实；
- **habit 必须写清频率 + 原因**（每天/每周几次、为什么），不能只写“有…习惯”；
  同一活动多个频率要分开列（例如“每天早上跑操”和“每周三次 3 公里”是两条）；
- 合并前检查：两条只差频率/原因时，它们很可能是两个事实，不要合成一条。

输出 JSON：
{{"memories":[
  {{"kind":"ritual|habit|preference|fact|boundary",
    "key":"小写英文键，具体到事",
    "value":"简洁中文；ritual 必须写成「场景：内容」",
    "days":出现天数,"count":出现次数,
    "evidence":"跨天证据，引用不同天的说法"}}
]}}
只输出 JSON，不要 markdown。

按天要点：
{digests}
"""


def _slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return s or "item"


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _bootstrap():
    import nonebot
    try:
        nonebot.get_driver()
    except ValueError:
        nonebot.init()


def load_days(path: Path, days: int) -> dict:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    by_day = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = float(d.get("t") or 0)
        day = datetime.fromtimestamp(t).strftime("%Y-%m-%d")
        if day < cutoff:
            continue
        u = str(d.get("user") or "").strip()[:MAX_MSG_CHARS]
        a = str(d.get("reply") or "").strip()[:MAX_MSG_CHARS]
        if u or a:
            by_day[day].append(f"用户：{u}\n灰泽满：{a}")
    return dict(sorted(by_day.items()))


async def _call(client, prompt: str, max_tokens: int = 1600) -> str:
    from src.plugins.chatbot.config import _get_model_name
    from src.plugins.chatbot.constants import THINKING_DISABLED

    resp = await client.chat.completions.create(
        model=_get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
        **THINKING_DISABLED,
    )
    return resp.choices[0].message.content or ""


def candidate_to_item(cand: dict) -> tuple[dict | None, str]:
    """把反思候选映射成 memory_v2 能校验的条目；不支持/不合格 → (None, 原因)。"""
    from src.plugins.chatbot import memory_v2 as mv

    kind = str(cand.get("kind") or "").strip()
    key = _slug(cand.get("key"))
    value = str(cand.get("value") or "").strip()
    if not value:
        return None, "empty_value"
    base = {
        "value": value, "action": "upsert", "valid_for": "none",
        "explicit": True, "confidence": float(cand.get("confidence") or 0.8),
        "actor": "user",
    }
    if kind == "ritual":
        if "：" not in value and ":" not in value:
            return None, "ritual_needs_cue"
        key = key if key.startswith("ritual") else f"ritual_{key}"
        return {**base, "kind": "interaction_preference", "key": key,
                "temporal_scope": "timeless"}, ""
    if kind == "habit":
        if not re.search(r"每天|每周|一周|每早|每晚|每周末|日常|平时|近期|最近|次", value):
            # 只写“有早起跑步的习惯”会把频率/原因丢掉，宁可不自动写，留给人看。
            return None, "habit_missing_frequency"
        key = key if key.startswith("habit") else f"habit_{key}"
        return {**base, "kind": "fact", "key": key,
                "temporal_scope": "timeless"}, ""
    if kind == "preference":
        # 名字类候选在反思里常被写成 preference；抽取引号里的裸名字，
        # 交给 store 的名字闸门校验（source 里有“叫我”才过）。
        if (key in ("preferred_name", "user_nickname", "nickname")
                or "call_me" in key or "user_call" in key or "nickname" in key):
            m = re.search(r"[“\"「]([^”\"」]{1,12})[”\"」]", value)
            if not m:
                return None, "name_not_extractable"
            return {**base, "kind": "fact", "key": "preferred_name",
                    "value": m.group(1), "temporal_scope": "timeless"}, ""
        if key not in mv._PREFERENCE_VALUES and not mv._is_ritual_key(key):
            # 带频率的偏好候选（如“喜欢索要摸摸，高频日常”）落成 habit 事实；
            # 没有频率就留给人看，别硬塞进封闭枚举。
            if re.search(r"每天|每周|一周|日常|平时|近期|最近|次", value):
                habit_key = key if key.startswith("habit") else f"habit_{key}"
                return {**base, "kind": "fact", "key": habit_key,
                        "temporal_scope": "timeless"}, ""
            return None, "unknown_preference_key"
        if key in mv._PREFERENCE_VALUES and value not in mv._PREFERENCE_VALUES[key]:
            return None, "invalid_preference_value"
        return {**base, "kind": "interaction_preference", "key": key,
                "temporal_scope": "timeless"}, ""
    if kind in ("fact", "boundary"):
        return {**base, "kind": "fact", "key": key,
                "temporal_scope": "timeless"}, ""
    return None, "unsupported_kind"


async def reflect_user(user: str, days: int, *,
                       apply: bool = False, refresh: bool = False,
                       out_dir: Path = OUT_DIR) -> dict:
    from src.plugins.chatbot.config import _get_clients

    path = ROOT / "data" / "chat_log" / f"{re.sub(r'[^0-9]', '', user)}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"没有聊天记录：{path}")
    by_day = load_days(path, days)
    if not by_day:
        return {"user": user, "days": 0, "candidates": [], "applied": []}

    client, _ = _get_clients()
    cache_dir = DIGEST_DIR / user
    cache_dir.mkdir(parents=True, exist_ok=True)
    digests = []
    for day, lines in by_day.items():
        cache = cache_dir / f"{day}.json"
        if cache.exists() and not refresh:
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            chat = "\n".join(lines)[:MAX_DAY_CHARS]
            data = _parse_json(await _call(client, DAY_PROMPT.format(chat=chat)))
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        digests.append({"day": day, "items": data.get("items") or []})
        print(f"[digest] {day}: {len(digests[-1]['items'])} 条")

    result = _parse_json(await _call(
        client, REFLECT_PROMPT.format(digests=json.dumps(digests, ensure_ascii=False)),
        max_tokens=2000))
    cands = result.get("memories") or []
    mapped, rejected = [], []
    for c in cands:
        item, why = candidate_to_item(c)
        if item:
            mapped.append({"candidate": c, "item": item})
        else:
            rejected.append({"candidate": c, "reason": why})

    applied = []
    backup = ""
    if apply and mapped:
        from src.plugins.chatbot.memory_v2 import ShadowMemoryStore

        store = ShadowMemoryStore()
        if store.state_file.exists():
            backup = str(store.state_file.with_name(
                store.state_file.name + f".bak-reflect-{datetime.now():%Y%m%d-%H%M%S}"))
            shutil.copy2(store.state_file, backup)
            print(f"[reflect] 写库前已备份：{backup}")
        last_day = max(by_day)
        observed = f"{last_day}T23:59:00"
        for m in mapped:
            src = {
                "session_id": f"reflect-{user}-{last_day}",
                "observed_at": observed,
                "user_text": str(m["candidate"].get("evidence") or m["candidate"].get("value") or "")[:300],
                "assistant_text": "",
            }
            try:
                accepted = store.ingest(user, [m["item"]], source=src)
                applied.append({"key": m["item"]["key"], "accepted": bool(accepted)})
            except Exception as e:
                applied.append({"key": m["item"].get("key"), "error": repr(e)})

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{user}.json"
    out.write_text(json.dumps({
        "user": user, "days": len(by_day),
        "candidates": cands, "mapped": mapped,
        "rejected": rejected, "applied": applied, "backup": backup,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"user": user, "days": len(by_day), "candidates": len(cands),
            "mapped": len(mapped), "rejected": len(rejected),
            "applied": applied, "backup": backup, "out": str(out)}


def run(args) -> dict:
    _bootstrap()
    return asyncio.run(reflect_user(
        args.user, args.days, apply=args.apply, refresh=args.refresh,
        out_dir=Path(args.out)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="QQ 号")
    ap.add_argument("--days", type=int, default=30, help="回看多少天（默认 30）")
    ap.add_argument("--apply", action="store_true", help="把候选写进真实 Memory V2（默认只出候选）")
    ap.add_argument("--refresh", action="store_true", help="忽略 digest 缓存，重跑所有天")
    ap.add_argument("--out", default=str(OUT_DIR), help="候选输出目录（默认 outputs/reflection）")
    args = ap.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
