#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性清理记忆库里的存量坏数据：改掉空壳名字、合并同一件事的多条。

## 治的是哪两类（都是 2026-10-16 之前在库里攒下来的）

1. **拆**：`key` 是空壳名（`moment` / `health` / `info` …）——什么都能装，既认不了亲也查不出来。
   实测全库 **20 条 `moment`，涉及 9 个用户**。
2. **合**：同一件事被起了不同名字，各存一份。实测某用户的感冒记成了
   `recent_cold` / `recent_illness` / `recent_insomnia` 三条，一次注入全端出来。

**提取端的提示词已经改好了（以后不会再产生新的），但这个脚本治的是已经躺在库里的。**

## 怎么做的

每个用户**一次调用**：把它手上所有活着的事实（`key=value` + 内部 id）整个交给模型，
让它回答两件事：

    {"actions": [
       {"ids": ["a"],            "new_key": "health_recent_cold"},           ← 只是改名
       {"ids": ["b", "c"],       "new_key": "family_pet_died",
        "merged_value": "…",     "keep": "b"}                                ← 合并成一条
    ]}

脚本只动三样东西：`key` / `value` / `status`，**其它字段一律不碰**。
改完会**重算 `id`**（id 是 `key+value` 的哈希，不重算的话下次同样的内容会被当成新条目又存一遍）。

## 用法

    python scripts/memory_v2_repair.py                # 只看，不写（默认）
    python scripts/memory_v2_repair.py --write        # 落盘，先自动备份
    python scripts/memory_v2_repair.py --user 3076669330 --write   # 只修一个人
"""
import argparse
import asyncio
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

_KEY_OK = re.compile(r"^[a-z][a-z0-9_]{0,47}$")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")

from src.plugins.chatbot import memory_v2  # noqa: E402
from src.plugins.chatbot.config import _get_clients, _get_model_name  # noqa: E402
from src.plugins.chatbot.constants import THINKING_DISABLED  # noqa: E402

STATE = ROOT / "user_memory" / "memory_v2_shadow.json"

PROMPT = """你在整理一份"关于某个人"的长期记忆。下面每一行是一条记录，格式：
  编号 ｜ 栏目名 ｜ 内容

请做两件事，**只输出 JSON**：

1. **把空壳栏目名换成具体的**。空壳名是指"什么都能装"的那种（`moment` / `info` / `health` /
   `situation` / `event` / `cold_status` 这类），它们既认不了亲也查不出来。
   ⛔ **已经能看出是哪一件事的栏目名，一个字都不要动**（`preferred_name` / `location` /
   `timezone_offset` / `roommate_relationship` / `daughter_death`…）。
   **只动空壳名**——把具体名换成光秃秃的类别名是**倒退**。
   换成 `类别_具体事`：住地 `location` ／ 身份 `identity` ／ 健康 `health` ／ 家人 `family` ／
   工作学习 `work` ／ 关系 `relationship` ／ 喜好 `interest` ／ 习惯 `habit` ／ 过往事件 `past_event`。
   例：`moment` 装着"用户向灰泽满求婚" → `relationship_proposal`；
   `cold_status` 装着"用户感冒了" → `health_cold`。
   ⚠️ **一条记录只改名的，`new_key` 里必须带上"是哪件事"**（`health_cold` 而不是 `health`）。
   ⚠️ **同一个人的两条记录不能改成同一个 `new_key`**——那是两件事，名字得分开
   （`relationship_first_confession` / `relationship_second_confession`）。
2. **把"其实是同一件事的两遍记录"合并成一条**。合并时由你写一条新的内容，把几条并进去，别丢东西。
   ⚠️ **默认只改名，不要合并。** 合并是有风险的动作。
   只有当你确信这几行**说的是同一件事、只是重复记了两遍**时才合并。
   ⚠️ **同一类事情发生过好几次的，一律不合并**——那是几件事，不是一件事的几遍记录：
   · 第一次告白 / 第二次求交往 / 求婚 —— 三件事，**拆开**
   · 想念一个失去的人：白天提过一次 / 深夜又提过一次 —— 说的是**同一个人同一份感情**，**可以并**
   （踩坑 2026-10-16 预览：把"发文案告白 / 当面告白 / 挽留"三件不同的互动并成了一条，丢细节。）

输出格式：
{{"actions": [
  {{"ids": ["编号"], "new_key": "新栏目名"}},
  {{"ids": ["编号1", "编号2"], "new_key": "新栏目名", "merged_value": "合并后的内容", "keep": "编号1"}}
]}}
- 只改名就只给 `ids`（一个）+ `new_key`
- 要合并就给 `ids`（两个以上）+ `new_key` + `merged_value` + `keep`（留哪条当主）
- **不动的那几条不要出现在 actions 里**

【这个人现在的记录】
{listing}
"""


def _mid(kind, key, value):
    return hashlib.sha256(f"{kind}\0{key}\0{value}".encode("utf-8")).hexdigest()[:20]


def _alive_facts(user: dict) -> list:
    """活着的 fact。⚠️ 库里**有没写 `id` 的条目**（老的/手加的）——缺了就现算一个补上，
    否则 `ingest` 按 id 找永远找不到它，同样的内容会被反复存成新条目。"""
    out = []
    for m in user.get("memories", []):
        if m.get("status") not in ("confirmed", "active") or m.get("kind") != "fact":
            continue
        if not m.get("id"):
            m["id"] = _mid(m["kind"], m.get("key", ""), m.get("value", ""))
        out.append(m)
    return out


def _needs_work(uid: str, user: dict, generic: set) -> bool:
    facts = _alive_facts(user)
    if any(m["key"] in generic for m in facts):
        return True
    # 同一件事的不同名字：内容上的二字组高度重合
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            ga, gb = memory_v2._ngrams(a["value"]), memory_v2._ngrams(b["value"])
            if ga and gb and len(ga & gb) / max(1, min(len(ga), len(gb))) >= 0.34:
                return True
    return False


async def propose(client, model, facts: list) -> list:
    listing = "\n".join(
        f"{m['id'][:8]} ｜ {m['key']} ｜ {m['value']}" for m in facts)
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(listing=listing)}],
        temperature=0,
        max_tokens=1500,
        **THINKING_DISABLED,
    )
    txt = (resp.choices[0].message.content or "").strip()
    txt = txt.replace("```json", "").replace("```", "").strip()
    if txt[:4].lower() == "json":
        txt = txt[4:].strip()
    try:
        return json.loads(txt).get("actions") or []
    except json.JSONDecodeError:
        return []


async def main() -> int:
    ap = argparse.ArgumentParser(description="清理记忆库里的存量坏数据")
    ap.add_argument("--write", action="store_true", help="落盘（缺省只看）")
    ap.add_argument("--user", default="", help="只处理这个 user_id")
    args = ap.parse_args()

    state = json.loads(STATE.read_text(encoding="utf-8"))
    users = state.get("users", {})
    generic = memory_v2._GENERIC_FACT_KEYS
    client, _ = _get_clients()
    model = _get_model_name()

    todo = [(uid, u) for uid, u in users.items()
            if (not args.user or uid == args.user) and _needs_work(uid, u, generic)]
    print(f"有活要干的有 {len(todo)} 个用户（模型 {model}）\n")

    changed_users = 0
    total_actions = 0
    skipped = 0
    for uid, user in todo:
        facts = _alive_facts(user)
        by_short = {m["id"][:8]: m for m in facts}
        actions = await propose(client, model, facts)
        if not actions:
            continue
        print(f"── {uid}（{len(facts)} 条）──")
        applied = 0
        used_keys: dict[str, str] = {}
        for act in actions:
            ids = [str(i)[:8] for i in act.get("ids") or []]
            targets = [by_short[i] for i in ids if i in by_short]
            new_key = str(act.get("new_key") or "").strip()
            if not targets or not new_key:
                continue
            # 守卫①：不许改成空壳名（把具体名退化成 `health` 这种是倒退）
            if new_key in generic or not _KEY_OK.match(new_key):
                print(f"   ⏭️ 跳过：`{new_key}` 是空壳名/不合规")
                skipped += 1
                continue
            # 守卫②：同名撞车——那就是两件事，名字得分开
            if new_key in used_keys:
                print(f"   ⏭️ 跳过：`{new_key}` 和另一条撞名了（它们是两件事）")
                skipped += 1
                continue
            if all(t["key"] == new_key for t in targets) and len(targets) == 1:
                continue                      # 名字没变、也没合并 —— 不用记一笔
            used_keys[new_key] = ids[0]
            keep = by_short.get(str(act.get("keep") or "")[:8], targets[0])
            merged = str(act.get("merged_value") or "").strip()
            print(f"   {[t['key'] for t in targets]}  →  {new_key}")
            for t in targets:
                print(f"        · {t['value'][:44]}")
            if merged:
                print(f"      合并后 → {merged[:56]}")
            applied += 1                      # 预览也要计数，否则最后那行汇总永远是 0
            if args.write:
                for t in targets:
                    if t is keep and merged:
                        t["value"] = merged
                    t["key"] = new_key
                    if t is not keep and len(targets) > 1:
                        t["status"] = "superseded"
                        t["invalidated_at"] = datetime.now().isoformat()
                    t["id"] = _mid(t["kind"], t["key"], t["value"])   # id 是 key+value 的哈希，必须重算
        if applied:
            changed_users += 1
            total_actions += applied
        print()

    if not args.write:
        print(f"（预览模式，没落盘。以上 {total_actions} 组改动、涉及 {changed_users} 个用户"
              + (f"；另有 {skipped} 条被守卫拦下" if skipped else "") + "。加 --write 才写。）")
        return 0

    if not total_actions:
        print("没有要改的。")
        return 0
    backup = STATE.with_suffix(f".json.bak-repair-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(STATE, backup)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[OK] 改了 {total_actions} 组 / {changed_users} 个用户")
    print(f"     备份：{backup.name}")
    print("     ⚠️ 记得【重启 bot】生效（记忆有进程内缓存）。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))