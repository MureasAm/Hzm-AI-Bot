#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 corpus 每条生成「用户可能怎么说」当**索引键**——而且**自己验证**。

## 为什么

corpus 存的是**第三人称叙述**（"灰泽满被问到身高，直接说1米6…"），把
「触发情境」和「内容」揉在一起了。而检索该匹配的只是**触发情境**。
实测：`「你多高啊」→ cos 0.443`（被阈值挡），
     `「灰泽满被问到身高」→ cos 0.790`（同构了，余弦完全够用）。

**所以不是"该换 LLM 判"，是"库和 query 形态不同构"。**
`behaviors` 之所以准，就是因为它的索引是 `samples[].user`（粉丝会怎么说），
不是 `trigger`（概括描述）。

## 关键设计：**自验证**

生成 6 个问法 → **每个都拿去检索，看能不能命中它自己** → **只留能命中的**。
  - 能命中 → 这个问法真的能把这条捞出来，留
  - 命不中 → 丢弃（它是个"看着像但检索不到"的问法）
  - **一个都命不中** → 这条素材**本来就不容易被问到**，标记出来（顺带成了筛选信号）

**为什么必须自验证**：实测过——同一条经历生成 3 个问法，中 1 不中 2。
不问自验证就入库，等于把一堆"看着像"的噪声当索引。

用法：
    python scripts/build_corpus_asks.py                # 全量
    python scripts/build_corpus_asks.py --limit 60     # 先跑 60 条看效果
    python scripts/build_corpus_asks.py --dry          # 只打印，不写文件
"""
import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nonebot  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

nonebot.init()
from src.plugins.chatbot.rag import embed_query  # noqa: E402
from src.plugins.chatbot.retrieval import load_vector_db, cosine_similarity  # noqa: E402
from src.plugins.chatbot.constants import ZHIPU_BASE_URL, DEEPSEEK_BASE_URL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.prod"
OUT_FILE = ROOT / "persona" / "world" / "corpus_asks.json"

N_ASKS = 6          # 每条先生成几个候选（多的会被自验证滤掉）
CONCURRENCY = 6
# 自验证的判定线：这个问法**问出来之后，第一条必须是它自己**（专指）。
# 实测对比（60 条样张）：
#   top-8 → 78% 覆盖，但平均留下 5.2 个/条（几乎不过滤，泛问法也过）
#   top-1 → 70% 覆盖，平均 3.0 个/条（**过滤掉一半**，留下的都是专指的）
# 取 top-1：宁可少留，也别把"泛泛能撞进前八"的问法当索引——那会带进一堆无关条目。
KEEP_TOPK = 1

PROMPT = """下面是"灰泽满"（虚拟主播）的一段经历。

【她的经历】
{text}

写出 **{n} 条用户可能说出来的话**——说完这句之后，她就该想起上面这段经历。

⚠️ 最重要：**像用户"还不知道这段经历"时会怎么开口**，不是概括这段经历。
   ✅ 好："你多高啊" / "小满是不是很矮"
   ❌ 差："你上次是不是被问到身高了"  ← 这是概括，用户不会这么问
⚠️ **短**，像 QQ 打字，别加原文没有的信息（别写"你声音都变了"这种你自己脑补的）
⚠️ {n} 条角度尽量不同（问事实的 / 提到相关人事物的 / 起话头的）

只输出 JSON 数组：["…", "…", …]"""


def _key(name: str) -> str:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].replace('"', "").strip()
    return ""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    ap.add_argument("--dry", action="store_true", help="只打印不写文件")
    args = ap.parse_args()

    ds = AsyncOpenAI(api_key=_key("OPENAI_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    zp = AsyncOpenAI(api_key=_key("ZHIPU_API_KEY"), base_url=ZHIPU_BASE_URL)
    model = _key("OPENAI_MODEL") or "deepseek-flash"

    sf = json.loads((ROOT / "persona/world/statement_final.json").read_text(encoding="utf-8"))
    texts = [x["statement"] for x in sf]
    if args.limit:
        texts = texts[: args.limit]
    db = load_vector_db()          # 与 texts 同序
    # 预先把每条的向量取出来，自验证时本地比（省 embedding）
    vecs = [it["vector"] for it in db[: len(texts)]]
    print(f"共 {len(texts)} 条，每条生成 {N_ASKS} 个问法，再自验证\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    results = {}

    async def one(i: int):
        async with sem:
            try:
                r = await ds.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT.format(text=texts[i], n=N_ASKS)}],
                    temperature=0.8, max_tokens=300,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                t = (r.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
                asks = [a for a in json.loads(t) if isinstance(a, str) and a.strip()]
            except Exception:
                return i, [], 0
            kept, tried = [], 0
            for a in asks[:N_ASKS]:
                tried += 1
                v = await embed_query(zp, a.strip())
                top = sorted(range(len(vecs)), key=lambda j: -cosine_similarity(v, vecs[j]))[:KEEP_TOPK]
                if i in top:                       # ← 这个问法真能把这条捞出来
                    kept.append(a.strip())
            return i, kept, tried

    for fut in asyncio.as_completed([one(i) for i in range(len(texts))]):
        i, kept, tried = await fut
        results[i] = kept

    n_any = sum(1 for i in range(len(texts)) if results.get(i))
    n_kept = sum(len(v) for v in results.values())
    print(f"**{n_any}/{len(texts)} = {n_any/len(texts):.0%} 至少有一个可用问法**")
    print(f"共保留 {n_kept} 条问法（平均每条 {n_kept/max(1,n_any):.1f} 个）\n")

    print("=== 一个都没命中的（这条素材检索不到，是筛选信号）===")
    for i in range(len(texts)):
        if not results.get(i):
            print(f"  ({len(texts[i])}字) {texts[i][:60]}")

    print("\n=== 有问法的样例 ===")
    shown = 0
    for i in range(len(texts)):
        if results.get(i) and shown < 5:
            print(f"  【{texts[i][:44]}】")
            for a in results[i]:
                print(f"      ← 「{a}」")
            shown += 1

    if not args.dry:
        payload = {
            "_readme": "corpus 的「用户可能怎么说」索引（scripts/build_corpus_asks.py 生成）。"
                       "每条都已经过**自验证**：拿这个问法去检索，能把它自己捞回来才留下。"
                       "用途：检索时 query 对 statement 向量和 asks 向量都算，取最高分——"
                       "解决『口语问句 vs 第三人称叙述』不同构导致真信号被阈值挡掉的问题。",
            "asks": {texts[i]: results[i] for i in range(len(texts)) if results.get(i)},
        }
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n✅ 写入 {OUT_FILE}")


asyncio.run(main())
