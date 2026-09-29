#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 corpus 每条经历挂「用户会提到的词」当**钩子**——解决"有经历但没人问得出来"。

## 为什么

实测：322 条里 **141 条没有任何召回入口**（`entry_audit` 那段审计）——
用户不可能说出一句话恰好把它们勾出来，**它们永远不会被注入**。

语义检索（向量）解决不了这个：那是"像不像"，而这里缺的是"**提没提到**"。
所以挂**确定性的词钩子**（和 terms 的 aliases 同一个机制）。

## 钩子必须窄，不能宽

- ✅ 具体：「三谋」「生日纪念视频」「纹身」
- ❌ 泛词：直播、她、上次、那个（到处都出现，一挂就乱触发）

## 挂了之后怎么验证（三步，缺一不可）

1. **对不对**：`python scripts/entry_audit.py` 那套——造"用户会怎么说"，跑真匹配看能不能勾出来
2. **有没有真发生**：拿 `data/chat_log/` 回放，看真实聊天里被勾出来过几次
3. **勾错了没有**：同回放，看误报

⚠️ 本脚本只**产出数据**（`persona/world/corpus_keywords.json`），不改线上。

用法：
    python scripts/build_corpus_keywords.py --limit 10   # 先看 10 条
    python scripts/build_corpus_keywords.py              # 全量（只处理没钩子的）
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from openai import AsyncOpenAI  # noqa: E402
from src.plugins.chatbot.constants import DEEPSEEK_BASE_URL, THINKING_DISABLED  # noqa: E402

SRC = ROOT / "persona" / "world" / "statement_final.json"
OUT = ROOT / "persona" / "world" / "corpus_keywords.json"

PROMPT = """下面是"灰泽满"（虚拟主播）的一段经历，她用**语义检索**找不到，需要一个"钩子"——
用户提到某个具体词时，这条经历就该被想起来。

【经历】
{text}

请给出 **2~5 个用户可能提到的具体词/短语**（2~6 字），提到任一个就该把这段想起来。

⚠️ 必须**具体**：人名、物品、事件名、专有说法。
   ✅ 好：「三谋」「生日纪念视频」「纹身」「抢票」「塔菲」
   ❌ 差：「直播」「上次」「那个」「事情」——到处都出现，一挂就乱触发
⚠️ 词要**用户真的会打出来**的，不是概括这段经历的书面词。

只输出 JSON 数组：["…", "…"]"""


def _key(n: str) -> str:
    for line in (ROOT / ".env.prod").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(n):
            return line.split("=", 1)[1].replace('"', "").strip()
    return ""


# ⚠️ 泛词黑名单：**一挂就乱触发**（每条经历都沾"灰泽满""直播"，挂了等于没挂还添噪声）。
# 模型即使被明确要求"别用泛词"，实测还是会塞进来（#4/#5/#6 都带了"灰泽满"），所以代码兜一道。
GENERIC = (
    "灰泽满", "hzm", "绿冻", "直播", "粉丝", "观众", "弹幕", "上次", "那天", "当时",
    "事情", "东西", "时候", "自己", "可以", "什么",
)


def _clean(kws: list) -> list:
    """去泛词、去太长（>6 字用户不会打）、去纯数字时间。"""
    out, seen = [], set()
    for k in kws:
        k = k.strip()
        if not (2 <= len(k) <= 6):
            continue
        if any(g in k for g in GENERIC):     # 含泛词就丢（哪怕只是包含）
            continue
        if k.replace(":", "").isdigit():
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.loads(SRC.read_text(encoding="utf-8"))["statements"]
    texts = [x["statement"] for x in items]
    old = {}
    if OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8")).get("keywords", {})

    todo = [(i, t) for i, t in enumerate(texts) if str(i) not in old]
    if args.limit:
        todo = todo[: args.limit]
    print(f"待处理 {len(todo)} 条（已有钩子的 {len(old)} 条跳过）\n")

    ds = AsyncOpenAI(api_key=_key("OPENAI_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    model = _key("OPENAI_MODEL") or "deepseek-flash"
    sem = asyncio.Semaphore(5)

    async def one(i, t):
        async with sem:
            try:
                r = await ds.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": PROMPT.format(text=t)}],
                    temperature=0.7, max_tokens=150, **THINKING_DISABLED)
                c = (r.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
                kws = _clean([k for k in json.loads(c) if isinstance(k, str)])
                return i, kws
            except Exception as e:
                return i, []

    res = await asyncio.gather(*[one(i, t) for i, t in todo])
    got = [(i, k) for i, k in res if k]
    print(f"生成成功 {len(got)}/{len(todo)}\n")
    for i, k in got[:8]:
        print(f"  #{i} {k}")
        print(f"      ← {texts[i][:66]}")

    for i, k in got:
        old[str(i)] = k
    OUT.write_text(json.dumps({
        "_readme": "corpus 的「钩子」：用户提到这些**具体**词时，**把该条递过去让 LLM 判一眼**。\n"
                   "⚠️ 2026-09-29 起**不再直通注入** —— 门/钩子只决定候选，判定全交 LLM"
                   "（见 retrieval._corpus_gate_pass 的注释）。\n"
                   "由 scripts/build_corpus_keywords.py 生成（**合并式**：已有钩子的条目会跳过，"
                   "所以手工补的不会被覆盖）；键是 statement_final 里的**序号**。\n"
                   "⚠️ 只放**具体**的词（人名/物品/事件名），泛词会到处误触发（现在的表现是'灌爆候选池'）。\n"
                   "⚠️ 挂完要三步验证：entry_audit（能不能勾出）→ 回放 chat_log（真发生没）→ 看误报。\n"
                   "⚠️ **手工补的钩子记在这里**（生成器不会覆盖，但也没人知道你为什么加）。",
        "keywords": old,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n✅ 已写 {OUT}（共 {len(old)} 条有钩子）")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
