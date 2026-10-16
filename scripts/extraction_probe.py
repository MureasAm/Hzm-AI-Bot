#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打**真模型**的提取探针：喂一个设定好的场景，看它到底记住了什么。

## 为什么要有它

`tests/test_memory_v2.py` 用的是**假客户端**——它只能验"给了这个 JSON，代码处理得对不对"，
验不了"**模型会不会这么说**"。而我们要修的恰恰是后者：

- 同一件事换了个说法，模型会不会另起一个新名字？（→ 感冒记三份）
- 遇到新的事，模型会不会又起一个 `moment` 这种空壳名？（→ 五条 moment）

这两件事**只能拿真模型量**。所以这个脚本每次跑都要花几次 API 调用。

## 怎么读结果

每个场景打印三样：**预置**（库里本来有什么）→ **模型新写的**（它认为该记什么）→ **入库后**
（过了代码校验、合并完之后剩什么）。最后一行是这条场景的判据过没过。

用法：
    python scripts/extraction_probe.py                 # 全部场景
    python scripts/extraction_probe.py --only cold     # 只跑 key 里含 cold 的
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")

from src.plugins.chatbot import memory_v2  # noqa: E402
from src.plugins.chatbot.config import _get_clients, _get_model_name  # noqa: E402

NOW = datetime(2026, 10, 16, 20, 0, 0)


def _seed_item(kind, key, value, scope="current", valid_for="none"):
    return {"kind": kind, "key": key, "value": value, "explicit": True,
            "confidence": 0.9, "temporal_scope": scope, "valid_for": valid_for}


# 每个场景：预置了什么 / 这轮说了什么 / 期望什么
SCENARIOS = [
    {
        "name": "cold_同一件事换个说法（合）",
        "seed": [_seed_item("fact", "recent_cold", "用户最近感冒了", valid_for="14d")],
        "user": "感冒好像还挺严重的，浑身不舒服一直咳嗽",
        "assistant": "又严重了？先别硬撑。",
        "expect": "不新增第二条感冒——沿用原 key，库里只剩一条感冒",
        "check": lambda before, alive, new: sum(1 for m in alive if "感冒" in m["value"]) == 1,
    },
    {
        "name": "cold2_同一个近况续期（合）",
        "seed": [_seed_item("fact", "health_recent_cold", "用户最近感冒了", valid_for="14d")],
        "user": "还是没好，这几天一直在吃药",
        "assistant": "吃药有用吗？",
        "expect": "沿用 health_recent_cold，不新起名字",
        "check": lambda before, alive, new: sum(
            1 for m in alive if "感冒" in m["value"] or "吃药" in m["value"]) == 1,
    },
    {
        "name": "split_全新的一件事（拆）",
        "seed": [],
        "user": "我女儿上个月出车祸走了，我现在一闭眼就是她",
        "assistant": "……",
        "expect": "key 必须是类别+具体事（family_… 之类），不能是 moment/info",
        "check": lambda before, alive, new: all(
            m["key"] not in memory_v2._GENERIC_FACT_KEYS for m in new if m["kind"] == "fact"),
    },
    {
        "name": "split2_库里有 moment，来了件不相干的事（拆）",
        "seed": [_seed_item("fact", "moment", "用户首次表白，灰泽满震惊", scope="past")],
        "user": "我家猫昨天死了，养了十一年",
        "assistant": "……十一年啊。",
        "expect": "新条目不能用 moment，要能和老的那条分开",
        "check": lambda before, alive, new: all(
            m["key"] not in memory_v2._GENERIC_FACT_KEYS for m in new if m["kind"] == "fact"),
    },
    {
        "name": "gate_只能从语气里猜的（闸门）",
        "seed": [],
        "user": "哦，随便吧，你说了算。",
        "assistant": "那就这么定了。",
        "expect": "没有可核实的明确表达 → 不该记下任何 fact（或 explicit=false 被拦）",
        "check": lambda before, alive, new: not [m for m in new if m["kind"] == "fact"],
    },
    {
        "name": "two_一轮里两件不同的事（拆）",
        "seed": [],
        "user": "我下周三要考试，还有我搬回广州了",
        "assistant": "考试和搬家撞一起了？",
        "expect": "两条，各有各的具体名字，不是同一个空壳名",
        "check": lambda before, alive, new: (
            len([m for m in new if m["kind"] == "fact"]) >= 2
            and len({m["key"] for m in new}) == len(new)),
    },
    {
        "name": "same_同一件事在一句话里说了两遍（合）",
        "seed": [],
        "user": "我感冒了，好难受啊，这感冒真烦人",
        "assistant": "别硬撑，喝点水。",
        "expect": "只出一条，不能因为说了三遍就记三条",
        "check": lambda before, alive, new: sum(1 for m in new if m["kind"] == "fact") == 1,
    },
    {
        "name": "kin_同类但是两件事（拆）",
        "seed": [_seed_item("fact", "family_daughter_accident", "女儿上个月因车祸离世", scope="past")],
        "user": "我妈前天住院了，心脏不好",
        "assistant": "……最近怎么都赶在一起了。",
        "expect": "起一个和女儿那条不同、但同属 family 的名字",
        "check": lambda before, alive, new: all(
            m["key"] not in memory_v2._GENERIC_FACT_KEYS and m["key"] != "family_daughter_accident"
            for m in new if m["kind"] == "fact"),
    },
    {
        "name": "rel_关系推进的第二回合（拆）",
        "seed": [_seed_item("fact", "relationship_confession", "用户首次表白，灰泽满震惊", scope="past")],
        "user": "我们交往吧，我是认真的",
        "assistant": "……你认真的？",
        "expect": "不能用 moment，也不能直接把表白那条顶掉（是两件事）",
        "check": lambda before, alive, new: all(
            m["key"] not in memory_v2._GENERIC_FACT_KEYS and m["key"] != "relationship_confession"
            for m in new if m["kind"] == "fact"),
    },
    {
        "name": "correct_用户纠正旧信息（合·顶替）",
        "seed": [_seed_item("fact", "location", "杭州", scope="timeless")],
        "user": "我不在杭州了，我一直在南京",
        "assistant": "行，记着了。",
        "expect": "location 被顶替成南京，库里只剩一条",
        "check": lambda before, alive, new: (
            sum(1 for m in alive if m["key"] == "location") == 1
            and any("南京" in m["value"] for m in alive if m["key"] == "location")),
    },
    {
        "name": "legacy_库里已经有两条重复的感冒（存量局限性）",
        "seed": [_seed_item("fact", "recent_cold", "用户最近感冒了", valid_for="14d"),
                 _seed_item("fact", "recent_illness", "感冒很严重，浑身不舒服", valid_for="14d")],
        "user": "感冒还是没好，一直咳嗽",
        "assistant": "还没好？",
        "expect": "能识别出「这是已记过的事」，别再添第三条（存量那两条得靠清理脚本）",
        "check": lambda before, alive, new: sum(1 for m in alive if "感冒" in m["value"]) == 2,
    },
]


async def run_one(client, model, sc) -> bool:
    tmp = tempfile.mkdtemp()
    store = memory_v2.ShadowMemoryStore(
        state_file=os.path.join(tmp, "s.json"),
        audit_file=os.path.join(tmp, "a.jsonl"),
        context_budget_chars=700,
    )
    uid = "probe-user"
    src = {"session_id": "u1:s1", "observed_at": NOW.isoformat(),
           "user_text": sc["user"], "assistant_text": sc["assistant"]}

    if sc["seed"]:
        store.ingest(uid, sc["seed"], source=src, now=NOW)
    before = store.snapshot(uid)["memories"]

    new = await memory_v2.extract_and_ingest(
        client=client, model=model, store=store, user_id=uid,
        user_text=sc["user"], assistant_text=sc["assistant"],
        session_id="probe:s1", observed_at=NOW,
    )
    after = store.snapshot(uid)["memories"]
    alive = [m for m in after if m.get("status") not in ("superseded", "expired")]

    print(f"\n{'=' * 72}\n【{sc['name']}】{sc['expect']}")
    print(f"  他说：{sc['user']}")
    print(f"  预置：")
    for m in before:
        print(f"      {m['kind']:<22}{m['key']:<28}{m['value'][:30]}  ({m.get('status')})")
    print(f"  模型这轮新写的：")
    if not new:
        print("      （没有）")
    for m in new:
        print(f"      {m['kind']:<22}{m['key']:<28}{m['value'][:30]}  ({m.get('status')})")
    print(f"  入库后活着的：")
    for m in alive:
        print(f"      {m['kind']:<22}{m['key']:<28}{m['value'][:30]}  ({m.get('status')})")

    try:
        ok = bool(sc["check"](before, alive, new))
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"  ⚠️ 判据本身出错：{e}")
    print(f"  → {'✅ 过' if ok else '❌ 不过'}")
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑名字里含这些词的场景（逗号分隔，如 cold,split）")
    args = ap.parse_args()

    client, _ = _get_clients()
    model = _get_model_name()
    print(f"模型：{model}")

    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    scs = [s for s in SCENARIOS if not keys or any(k in s["name"] for k in keys)]
    results = []
    for sc in scs:
        try:
            results.append((sc["name"], await run_one(client, model, sc)))
        except Exception as e:                               # noqa: BLE001
            print(f"\n❌ 【{sc['name']}】跑挂了：{type(e).__name__}: {e}")
            results.append((sc["name"], False))

    print(f"\n{'=' * 72}\n汇总：{sum(1 for _, ok in results if ok)}/{len(results)} 过")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))