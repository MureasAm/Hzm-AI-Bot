#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus 形式改造：把"她当时怎么说的"从背景记忆里去掉，只留"发生了什么"。

## 为什么改

`statement_final.json`（322 条直播经历）的形式是**第三人称陈述**。但实测
（`scripts/form_experiment.py`，见 `docs/评测与实验.md` 第三节）：
**自足的内容会被模型整句搬走**（③ 事实条目 逐字 16/16、② 陈述 9/16），
而 corpus 的注入包装语写着"别整段复述"——**包装语实测无效**。

体检：**157/322 = 49% 含引号**，也就是把"她当时怎么说的"写进了背景记忆。
那部分是 `voice_samples` / `phrases` 的活。

**新判据**：这条引语**换成别的说法行不行？**
  - 行 → 措辞不重要 → **只留事实**（本例要做的）
  - 不行（标志性台词/梗） → 那本来就该在 `legendary` / `phrases` → 标出来人工搬

## 安全设计

- **绝不直接改 `statement_final.json`**：那份是线上一手源（322 条只有 111 条可追溯、
  「合并」这步还是手工的），误覆盖不可逆。本脚本只**产出对照表**，人工看过再落。
- 逐条输出 before/after + 删了什么，供人肉审。
- 判"引语就是本条的核心"的**不自动改**，单独标出来让人决定。

用法：
    python scripts/rewrite_corpus.py --limit 15        # 先看 15 条
    python scripts/rewrite_corpus.py                   # 全量（只写过哪些需改）
    python scripts/rewrite_corpus.py --only quote      # 只处理"含引语"的
    python scripts/rewrite_corpus.py --apply           # 人工审过之后，写回源文件
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
from src.plugins.chatbot.constants import DEEPSEEK_BASE_URL, THINKING_DISABLED  # noqa: E402

SRC = ROOT / "persona" / "world" / "statement_final.json"
REVIEW = ROOT / "outputs" / "corpus_rewrite_review.json"
ENV_FILE = ROOT / ".env.prod"

MUST_FIX_CHARS = 300       # 注入时单条上限，超了会被截断
TARGET_CHARS = 80          # 改写目标长度（现在中位数就是 80）

# ⚠️ 单引号也算引语——第一版只认双引号，漏了 85 条（含 `'好变态啊'` 这种她说的原话）。
# 单引号在这批数据里有两种用途：① 标她说的原话（要处理）② 标被解释的词/别人的话（要留），
# 所以**检出后交给 LLM 分辨**，不在这里一刀切。
# 「 『 “ ” ‘ ’ " '  —— 用显式码点，避免引号在字符串里的转义坑
QUOTE = re.compile("[「『“”‘’\"']")
WHEN = re.compile(r'(今天|昨天|明天|最近|刚刚|这周|这阵子)')


def _key(name: str) -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(name):
            return line.split("=", 1)[1].replace('"', "").strip()
    return ""


def needs_fix(t: str) -> list:
    """这条要做什么手术？（空列表 = 不用动）"""
    tags = []
    if QUOTE.search(t):
        tags.append("引语")
    if len(t) > MUST_FIX_CHARS:
        tags.append("超长")
    if WHEN.search(t):
        tags.append("时效词")
    return tags


PROMPT = """下面是"灰泽满"（虚拟主播）的一条**背景记忆**素材。

【原文】
{text}

这条素材的用途是让她**记得发生过这件事**。用的时候**不该让她照搬里面的措辞**——
"她怎么说话"由另外的文件（风格样本/措辞库）负责。

请改写：**只留"发生了什么"，去掉"她当时怎么说的"**。

规则：
- ⚠️ **删掉"她说的原话"**。要保留的是那句话**传达的信息**，不是措辞本身。
  例：「她说"明天要开始准备首播了"」→「她说要开始准备首播了」
  例：「她自嘲'本质不熟'」→「她自嘲其实不熟」
- ✅ **但这些引号要留着**（不是她的话，删了就丢信息）：
  · 被解释的词：「'瓜娃子'是傻子的意思」← 词本身必须留着
  · 别人的话：「被弹幕说'物欲在不同地方'」「同学教她说'八嘎'」
  · 作品名/专名：「'灰之魔女'DLC」「被称为'绿茶'」
- ⚠️ 删掉"用什么样的语气/口吻"这类描写（那也是措辞范畴）。
- ✅ 保留：发生了什么事、结果如何、她的**态度/立场**（不含措辞）。
- ⚠️ **不要加原文没有的信息**，不要脑补。
- ⚠️ 仍是**第三人称**叙述（"灰泽满…"/"她…"），不要写第一人称。
- ⚠️ 尽量压到 {target} 字以内。超了就砍细节，别砍事实。
- ⚠️ **如果这条的核心就是那一句话本身**（比如某个梗的固定应答、她的标志性台词），
  那就**别改**，把 `quote_is_the_point` 设为 true。

只输出 JSON：
{{"quote_is_the_point": false, "rewritten": "…", "dropped": "不超过20字，说明删了什么"}}"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="any", choices=["any", "quote"])
    ap.add_argument("--apply", action="store_true", help="把审过的结果写回源文件")
    ap.add_argument("--skip-json", default=None,
                    help="跳过这份旧对照表里已处理过的 idx（重跑时别把改过的再改一遍）")
    args = ap.parse_args()

    data = json.loads(SRC.read_text(encoding="utf-8"))
    # 兼容两种形状：裸 list，或 {"_readme":..., "statements":[...]}（2026-09-27 起）
    items = data["statements"] if isinstance(data, dict) else data
    texts = [x["statement"] for x in items]

    if args.apply:
        if not REVIEW.exists():
            print(f"❌ 没有对照表 {REVIEW}，先跑一次不带 --apply 的")
            return
        rev = json.loads(REVIEW.read_text(encoding="utf-8"))
        n = 0
        for item in rev["items"]:
            i = item["idx"]
            if item.get("quote_is_the_point") or not item.get("rewritten"):
                continue
            texts[i] = item["rewritten"]
            n += 1
        # 保持原结构写回（别把 _readme 冲掉——那是这份文件的角色说明）
        items = [{"statement": t} for t in texts]
        out = {"_readme": data["_readme"], "statements": items} if isinstance(data, dict) else items
        SRC.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"✅ 已写回 {n} 条 → {SRC}")
        print("⚠️ 别忘了重算向量：python scripts/run_tool.py generate-vectors -i persona/world/statement_final.json")
        print("⚠️ 以及跑 retrieval_eval 确认没退化")
        return

    done = set()
    if args.skip_json:
        prev = json.loads(Path(args.skip_json).read_text(encoding="utf-8"))
        done = {x["idx"] for x in prev.get("items", [])
                if x.get("rewritten") and not x.get("quote_is_the_point")}
        print(f"跳过已处理过的 {len(done)} 条（{args.skip_json}）")

    todo = [(i, t) for i, t in enumerate(texts)
            if i not in done and needs_fix(t) and (args.only != "quote" or QUOTE.search(t))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"需要处理 {len(todo)} 条（共 {len(texts)}）\n")

    client = AsyncOpenAI(api_key=_key("OPENAI_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    model = _key("OPENAI_MODEL") or "deepseek-flash"
    sem = asyncio.Semaphore(5)

    async def one(i, t):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT.format(text=t, target=TARGET_CHARS)}],
                    temperature=0, max_tokens=400, **THINKING_DISABLED,
                )
                c = (r.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
                d = json.loads(c)
                return {"idx": i, "tags": needs_fix(t), "before": t,
                        "quote_is_the_point": bool(d.get("quote_is_the_point")),
                        "rewritten": (d.get("rewritten") or "").strip(),
                        "dropped": (d.get("dropped") or "")[:40]}
            except Exception as e:
                return {"idx": i, "tags": needs_fix(t), "before": t, "error": str(e)[:80]}

    items = list(await asyncio.gather(*[one(i, t) for i, t in todo]))
    items.sort(key=lambda x: x["idx"])

    ok = [x for x in items if x.get("rewritten") and not x.get("quote_is_the_point")]
    keep = [x for x in items if x.get("quote_is_the_point")]
    err = [x for x in items if x.get("error")]
    print(f"改写 {len(ok)} 条 ｜ 标记'引语就是核心'不改 {len(keep)} 条 ｜ 失败 {len(err)} 条\n")

    for x in items[:12]:
        print(f"--- #{x['idx']}  [{'+'.join(x['tags'])}]")
        print(f"  前: {x['before'][:88]}")
        if x.get("quote_is_the_point"):
            print("  后: ⚠️ 标记为「引语就是核心」，不改（该搬去 legendary/phrases？）")
        else:
            print(f"  后: {x.get('rewritten','')[:88]}")
            print(f"  删: {x.get('dropped','')}")

    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(json.dumps({"total": len(items), "items": items},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 对照表已存 {REVIEW}")
    print("   人工审过后跑：python scripts/rewrite_corpus.py --apply")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
