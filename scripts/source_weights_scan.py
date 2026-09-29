#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 `SOURCE_WEIGHTS` 找实测依据（`注入设计原理.md` §七-4）。

## 先说一个发现：**现在的评测测不到这个旋钮**

`retrieval-eval` 的判据是"**每一路各自**有没有召回对的条目"（`check_case(case, got)`，
`got` 是 snapshot 的**原始**各路结果）。而 `SOURCE_WEIGHTS` **只影响**

- 融合后的**排序**（`fusion_score = w / (k + rank)`，逐条赋值）
- 由此决定的 **top-6 截断** 和 **1200 字预算截断**

—— 它**不改变任何一路的召回**。所以拿 `retrieval-eval` 扫权重，扫出来的永远是同一条平线。

## 所以判据要换成：「融合 + 截断之后，期望的条目还在不在」

那才是权重真正决定的事：**模型最后看到的是哪几条。**

## 两段式（省 API）

```
--dump    跑一次真实检索，把**各路排名**存下来        （要 API：48 条 × 1 次）
--sweep   离线用不同权重重新融合、重新判分            （零成本，可随便扫）
```

## 用法
    python scripts/source_weights_scan.py --dump
    python scripts/source_weights_scan.py --sweep
"""
import argparse
import asyncio
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nonebot  # noqa: E402
try:
    nonebot.init()
except Exception:
    pass

import _common                        # noqa: E402
import retrieval_eval as RE           # noqa: E402
from src.plugins.chatbot import retrieval as R      # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules  # noqa: E402
from src.plugins.chatbot.constants import (         # noqa: E402
    SOURCE_WEIGHTS, RETRIEVAL_TOPK, VOICE_SAMPLE_PREFER_SHORT, RRF_K)

DUMP = _common.OUTPUTS_DIR / "eval" / "weights_dump.json"
ROUTES = ("corpus", "voice_sample", "behavior", "phrase")


def _item_to_dict(it) -> dict:
    # ⚠️ preference / core_story 是**裸 dict**（不是 RetrievalItem）——
    #    和 `retrieval_eval.check_case` 里的 `is_dict_style` 对得上。
    if isinstance(it, dict):
        return {k: it.get(k) for k in ("id", "text", "score", "desc", "keywords") if k in it}
    ex = it.extra or {}
    return {"source": it.source, "item_id": str(it.item_id), "score": it.score,
            "text": it.text,
            "extra": {k: ex.get(k) for k in ("user", "reply", "length", "phrases", "type")
                      if k in ex}}


def _dict_to_item(d: dict):
    it = R.RetrievalItem(source=d["source"], item_id=d["item_id"],
                         score=d["score"], text=d["text"])
    it.extra = d.get("extra") or {}
    return it


async def do_dump() -> int:
    from openai import AsyncOpenAI
    from src.plugins.chatbot.config import ZHIPU_BASE_URL, DEEPSEEK_BASE_URL
    key, dk = RE._zhipu_key(), RE._deepseek_key()
    if not key:
        print("❌ 没有 ZHIPU_API_KEY（.env.prod）")
        return 1
    client = AsyncOpenAI(api_key=key, base_url=ZHIPU_BASE_URL)
    ds = AsyncOpenAI(api_key=dk, base_url=DEEPSEEK_BASE_URL) if dk else None
    _, _, behaviors = load_persona_rules()
    cases = RE.load_cases()
    out = []
    for i, case in enumerate(cases, 1):
        got, l3 = await RE.snapshot(case["query"], client, ds, behaviors)
        if got is None:
            print(f"  [{i}] SKIP {case['id']}（embedding 失败）")
            continue
        out.append({"id": case["id"], "query": case["query"],
                    "expect": case.get("expect", {}),
                    "routes": {s: [_item_to_dict(x) for x in (got.get(s) or [])]
                               for s in ROUTES},
                    "preference": [_item_to_dict(x) for x in (got.get("preference") or [])],
                    "core_story": [_item_to_dict(x) for x in (got.get("core_story") or [])]})
        print(f"  [{i}/{len(cases)}] {case['id']}  "
              + " ".join(f"{s}={len(got.get(s) or [])}" for s in ROUTES))
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    DUMP.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已存各路排名：{DUMP}（{len(out)} 条）")
    return 0


def _score_combo(dump: list, weights: dict, prefer_short: bool) -> tuple:
    """用给定权重重新融合 + 截断，再按**同一份期望**判分。→ (通过数, 总条数)"""
    passed = 0
    for rec in dump:
        lists = []
        for s in ROUTES:
            items = [_dict_to_item(d) for d in rec["routes"].get(s, [])]
            lists.append(items)
        w = dict(weights)
        if prefer_short:
            for it in lists[1]:                      # voice_sample
                if it.extra.get("length", "short") == "short":
                    w["voice_sample"] = w.get("voice_sample", 1.0) + 0.3
                    break
        fused = R.rrf_fuse(lists, k=RRF_K, weights=w)[:RETRIEVAL_TOPK]
        fused = R.truncate_by_budget(fused)
        got = {s: [] for s in ("corpus", "voice_sample", "behavior", "phrase",
                               "preference", "core_story")}
        for it in fused:
            got.setdefault(it.source, []).append(it)
        # preference / core_story 不参与 RRF，原样带上
        got["preference"] = list(rec.get("preference", []))     # 原样（dict 形状）
        got["core_story"] = list(rec.get("core_story", []))
        if RE.check_case({"id": rec["id"], "query": rec["query"],
                          "expect": rec["expect"]}, got)["passed"]:
            passed += 1
    return passed, len(dump)


def do_sweep() -> int:
    if not DUMP.exists():
        print(f"❌ 先跑 --dump（缺 {DUMP}）")
        return 1
    dump = json.loads(DUMP.read_text(encoding="utf-8"))
    print(f"离线扫描：{len(dump)} 条 case，判据 = 「融合+截断后期望条目还在」\n")

    base_passed, n = _score_combo(dump, SOURCE_WEIGHTS, VOICE_SAMPLE_PREFER_SHORT)
    print(f"现状 SOURCE_WEIGHTS = {SOURCE_WEIGHTS}")
    print(f"  → 通过 {base_passed}/{n} = {base_passed/n*100:.0f}%\n")

    grid = {"behavior": [1.0, 1.5, 2.0], "corpus": [0.7, 1.0, 1.4],
            "voice_sample": [0.7, 1.0, 1.4], "phrase": [0.7, 1.2, 1.7]}
    rows = []
    for b, c, v, p in itertools.product(grid["behavior"], grid["corpus"],
                                        grid["voice_sample"], grid["phrase"]):
        w = {"behavior": b, "corpus": c, "voice_sample": v, "phrase": p}
        ps, _ = _score_combo(dump, w, VOICE_SAMPLE_PREFER_SHORT)
        rows.append((ps, w))
    rows.sort(key=lambda x: -x[0])
    print(f"{'通过':>5}  {'behavior':>9}{'corpus':>8}{'voice_sample':>13}{'phrase':>8}")
    print("-" * 46)
    for ps, w in rows[:10]:
        tag = "  ← 现状" if w == SOURCE_WEIGHTS else ""
        print(f"{ps:>5}  {w['behavior']:>9}{w['corpus']:>8}{w['voice_sample']:>13}"
              f"{w['phrase']:>8}{tag}")
    best = rows[0][0]
    same = [w for ps, w in rows if ps == best]
    print(f"\n最好组合通过 {best}/{n}；并列的有 {len(same)} 个组合")
    print(f"现状 {base_passed}/{n}"
          + ("　✅ 已在最优档（这个旋钮**不用动**）" if base_passed == best
             else f"　⚠️ 比最好低 {best - base_passed} 条 → 值得改，但先看并列有多少"))

    # 每路单独扫（看单路权重有没有更灵敏的方向）
    print(f"\n【单路扫描】（其余保持现状）")
    for route, vals in grid.items():
        line = []
        for val in vals:
            w = dict(SOURCE_WEIGHTS)
            w[route] = val
            ps, _ = _score_combo(dump, w, VOICE_SAMPLE_PREFER_SHORT)
            line.append(f"{val}→{ps}")
        print(f"  {route:<14}" + "　".join(line))

    (DUMP.parent / "weights_scan.json").write_text(json.dumps(
        {"n": n, "base": {"weights": SOURCE_WEIGHTS, "passed": base_passed},
         "grid": [{"passed": ps, "weights": w} for ps, w in rows]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n⚠️ 读法：**并列多 = 这个旋钮不敏感**（换哪个都差不多）→ 那就别动它。")
    print(f"   只有「现状明显低于最好、且最好那一档很窄」才值得改。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    _common.ensure_utf8_stdout()
    if a.dump:
        return asyncio.run(do_dump())
    if a.sweep:
        return do_sweep()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
