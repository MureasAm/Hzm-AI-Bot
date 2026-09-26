#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""入口审计：**给每条素材造"用户会怎么说"，测判据能不能把它勾出来。**

## 为什么需要

实测：`terms` 26 个词只有 3 个被真实对话命中过、`legendary` 20 组只有 2 组命中过。
**从没被触发过的条目 = 从没被验证过的条目**——它们的正确性完全是假设。
等真实对话来撞，可能等几个月；而且真撞上时才发现匹配不上就晚了。

所以主动造：**让 LLM 写"用户在不知道答案时会怎么问这件事"**，再拿它去跑**真实的匹配函数**。
- **确定性路（terms/legendary）**：命中=判据能工作；不命中=**关键词表和真实说法脱节**（可修）
- **语义路（core_story）**：命中=能召回；不命中=入口不够（可加关键词）

⚠️ 生成的问法是 LLM 造的，**不是真实数据**——所以结论是"判据**能不能**工作"，不是"用户**会不会**这么说"。
（两者都要，但前者便宜、后者要等。见 `注入设计原理.md` 第 4.4 节的区分。）

用法：
    python scripts/entry_audit.py terms
    python scripts/entry_audit.py legendary
    python scripts/entry_audit.py core_story
    python scripts/entry_audit.py all
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from openai import AsyncOpenAI  # noqa: E402
from src.plugins.chatbot import retrieval as R  # noqa: E402
from src.plugins.chatbot.rag import embed_query  # noqa: E402
from src.plugins.chatbot.persona import load_terms  # noqa: E402
from src.plugins.chatbot.routing import LEGENDARY_REPLIES  # noqa: E402
from src.plugins.chatbot.constants import DEEPSEEK_BASE_URL, ZHIPU_BASE_URL, THINKING_DISABLED  # noqa: E402

N_ASKS = 3
PROMPT = """有一个虚拟主播「灰泽满」。下面是关于她的**一条资料**。

【资料】{desc}

请写 {n} 句**用户可能对她说的话**——说完这句之后，这条资料就该被想起来/被用上。

⚠️ 像 QQ 打字，短，口语。**用户并不知道这条资料**，所以不会直接复述它。
⚠️ 不要提"资料""条目"这种词，就是普通粉丝会打的话。
⚠️ {n} 句角度尽量不同。

只输出 JSON 数组：["…", "…", "…"]"""


def _key(name: str) -> str:
    for line in (ROOT / ".env.prod").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(name):
            return line.split("=", 1)[1].replace('"', "").strip()
    return ""


def _match_term(t: dict, msg: str) -> bool:
    kw = t.get("keyword", "")
    keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
    pat = t.get("pattern")
    return any(k in msg for k in keys) or bool(pat and re.search(pat, msg))


async def audit_terms(ds, model, zp, sem):
    terms = [t for t in load_terms() if t.get("priority") != "always" and t.get("keyword")]
    return await _run(ds, model, zp, sem, "terms", [
        {"id": t["keyword"], "desc": f"名词「{t['keyword']}」：{t.get('meaning','')[:60]}",
         "check": (lambda m, _t=t: _match_term(_t, m))} for t in terms])


async def audit_legendary(ds, model, zp, sem):
    from src.plugins.chatbot.routing import legendary_hit   # 用真实匹配函数，别自己复现
    return await _run(ds, model, zp, sem, "legendary", [
        {"id": trig,
         "desc": f"经典梗「{trig}」→ 固定回应：{(entry.get('replies') or [''])[0][:50]}",
         "check": (lambda m, _t=trig: legendary_hit(_t, m))}
        for trig, entry in LEGENDARY_REPLIES.items()])


async def audit_core_story(ds, model, zp, sem):
    cs = json.loads((ROOT / "persona/world/core_stories.json").read_text(encoding="utf-8"))
    stories = cs["stories"] if isinstance(cs, dict) and "stories" in cs else cs
    return await _run(ds, model, zp, sem, "core_story", [
        {"id": s["id"], "desc": f"{s.get('category','')}：{s.get('text','')[:70]}",
         "vec": True} for s in stories])


async def _run(ds, model, zp, sem, name, entries):
    print(f"\n{'='*74}\n【{name}】{len(entries)} 条 ｜ 每条造 {N_ASKS} 个问法，测判据能不能勾出来\n")

    async def gen(desc):
        async with sem:
            try:
                r = await ds.chat.completions.create(
                    model=model, messages=[{"role": "user",
                        "content": PROMPT.format(desc=desc, n=N_ASKS)}],
                    temperature=0.8, max_tokens=200, **THINKING_DISABLED)
                c = (r.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
                return [a for a in json.loads(c) if isinstance(a, str) and a.strip()]
            except Exception:
                return []

    ok, bad = [], []
    for e in entries:
        asks = await gen(e["desc"])
        if e.get("vec"):                       # 语义路：走真实检索
            hits = []
            for a in asks:
                v = await embed_query(zp, a)
                got = R.retrieve_core_stories(a, v)
                hits.append(any(g["id"] == e["id"] for g in got))
        else:                                  # 确定性路：走真实匹配
            hits = [e["check"](a) for a in asks]
        n_hit = sum(hits)
        (ok if n_hit else bad).append((e["id"], n_hit, len(asks), asks))

    print(f"  能勾出来（≥1 个问法命中）: {len(ok)}/{len(entries)} = {len(ok)/len(entries)*100:.0f}%")
    print(f"  勾不出来（判据没工作）:    {len(bad)}/{len(entries)}")
    if bad:
        print("\n  ⚠️ 勾不出来的条目（判据与真实说法脱节）：")
        for eid, _, n, asks in bad[:12]:
            print(f"     · {eid}")
            for a in asks[:2]:
                print(f"         ← 「{a}」")
    if ok:
        print("\n  ✅ 能勾出来的样例：")
        for eid, k, n, asks in ok[:5]:
            print(f"     · {eid}  ({k}/{n})  ← 「{asks[0][:40]}」")
    return {"name": name, "total": len(entries), "ok": len(ok),
            "bad": [{"id": e, "asks": a} for e, _, _, a in bad]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["terms", "legendary", "core_story", "all"])
    args = ap.parse_args()

    ds = AsyncOpenAI(api_key=_key("OPENAI_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    zp = AsyncOpenAI(api_key=_key("ZHIPU_API_KEY"), base_url=ZHIPU_BASE_URL)
    model = _key("OPENAI_MODEL") or "deepseek-flash"
    sem = asyncio.Semaphore(5)

    fns = {"terms": audit_terms, "legendary": audit_legendary, "core_story": audit_core_story}
    which = list(fns) if args.which == "all" else [args.which]
    res = [await fns[w](ds, model, zp, sem) for w in which]

    out = ROOT / "outputs" / "entry_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已存 {out}")
    print("\n=== 汇总 ===")
    for r in res:
        print(f"  {r['name']:<12} 能勾出来 {r['ok']}/{r['total']} = {r['ok']/r['total']*100:.0f}%")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
