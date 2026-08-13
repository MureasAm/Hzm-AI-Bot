#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索层评测：量化六路检索（corpus/voice_sample/behavior/phrase 走 RRF + preference/core_story 命中才带）
在不同 query 下的命中率，让阈值调参从"人肉肉眼看"变成"可测量的回归"。

用法：
    python scripts/retrieval_eval.py              # 跑全部 case
    python scripts/retrieval_eval.py --case b1    # 只看某个 case
    python scripts/retrieval_eval.py --verbose    # 打印每个 source 命中的 top 项 + 分数

标注集：data/retrieval_eval_cases.json
每条 case = query + 对各路的期望：
    "behavior"/"voice_sample"/"phrase"/"preference"/"core_story" → {"should": [ids]} 期望命中
    "behavior"/"voice_sample" 等 → {"should_not_hit": true}    期望不命中（负例）
    "corpus" → {"contains": [关键词]}                          期望 top-k 文本含关键词

不读线上记忆、不调对话模型，只调智谱 embedding（与线上同款），只读向量缓存。只读不改。
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

from src.plugins.chatbot import retrieval  # noqa: E402  (可无 NoneBot 直接导入)
from src.plugins.chatbot.rag import embed_query  # noqa: E402
from src.plugins.chatbot.constants import ZHIPU_BASE_URL, DEEPSEEK_BASE_URL  # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules  # noqa: E402
from src.plugins.chatbot.retrieval import select_behavior_item  # noqa: E402
from src.plugins.chatbot.core import classify_behavior  # noqa: E402  (L3：LLM 判行为意图)

CASES_FILE = PROJECT_ROOT / "scripts" / "retrieval_eval_cases.json"
ENV_FILE = PROJECT_ROOT / ".env.prod"
OUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "retrieval"


def _env_key(name: str) -> str:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name):
                return line.split("=", 1)[1].replace('"', "").strip()
    return ""


def _zhipu_key() -> str:
    return _env_key("ZHIPU_API_KEY")


def _deepseek_key() -> str:
    return _env_key("OPENAI_API_KEY")


# ==================== 单条 case 判定 ====================

def _ids_of(items, is_dict_style: bool):
    if is_dict_style:
        return [str(i.get("id", "")) for i in items]
    return [str(i.item_id) for i in items]


def check_case(case: dict, got: dict) -> dict:
    """按期望判定一条 case。got: source -> 检索返回项列表。"""
    expect = case.get("expect", {})
    passed = True
    checks = []
    for source, cond in expect.items():
        items = got.get(source, [])
        dict_style = source in ("preference", "core_story")
        if "should" in cond:
            ids = _ids_of(items, dict_style)
            hit = [i for i in cond["should"] if i in ids]
            ok = bool(hit)
            detail = f"期望{cond['should']} 命中{hit} 实际{ids[:3]}"
        elif "should_not_hit" in cond:
            ok = len(items) == 0
            detail = f"期望不命中 实际{len(items)}条"
        elif "contains" in cond:
            texts = [getattr(it, "text", "") for it in items] if source == "corpus" else []
            ok = any(any(k in t for k in cond["contains"]) for t in texts)
            detail = f"期望文本含{cond['contains']} 实际top文本: {[t[:22] for t in texts[:2]]}"
        else:
            ok, detail = True, "（无判定条件）"
        passed = passed and ok
        checks.append({"source": source, "ok": ok, "cond": cond, "detail": detail})
    return {"id": case["id"], "query": case["query"], "passed": passed, "checks": checks}


def _summarize_got(got: dict, top: int = 3) -> dict:
    """压缩各 source 命中项（id + 分数），供 --verbose 打印。"""
    out = {}
    for source, items in got.items():
        if source in ("preference", "core_story"):
            out[source] = [(str(i.get("id", "")), round(i.get("score", 0), 3)) for i in items[:top]]
        elif source == "corpus":
            out[source] = [(it.item_id, round(it.score, 3), it.text[:18]) for it in items[:top]]
        else:
            out[source] = [(it.item_id, round(it.score, 3)) for it in items[:top]]
    return out


# ==================== 主流程 ====================

async def run(cases, verbose: bool, only_id: str = None):
    key = _zhipu_key()
    if not key:
        print("❌ 未找到 ZHIPU_API_KEY（.env.prod）——评测需要与线上同款 embedding")
        return 1
    client = AsyncOpenAI(api_key=key, base_url=ZHIPU_BASE_URL)
    ds_client = AsyncOpenAI(api_key=_deepseek_key(), base_url=DEEPSEEK_BASE_URL) if _deepseek_key() else None

    _, _, behaviors = load_persona_rules()
    print("🔍 检索评测 | 行为走 L3（LLM 判意图 + 关键词兜底），其余走 embedding")

    results = []
    for case in cases:
        if only_id and only_id not in case["id"]:
            continue
        query = case["query"]
        qv = await embed_query(client, query)
        if not qv:
            print(f"[SKIP] {case['id']}  {query}  （embedding 失败）")
            continue
        got = {
            "corpus": retrieval.retrieve_corpus(query, qv),
            "voice_sample": retrieval.retrieve_voice_samples(query, qv),
            "phrase": retrieval.retrieve_phrases(query, qv),
            "preference": retrieval.retrieve_preferences(query, qv),
            "core_story": retrieval.retrieve_core_stories(query, qv),
            "behavior": [],
        }
        # L3：行为走 LLM 判意图 → 关键词兜底（不再有 embedding 基线）
        behavior_intent = ""
        if "behavior" in case.get("expect", {}):
            behavior_intent = await classify_behavior(ds_client, query, "", behaviors) if ds_client else ""
            item = select_behavior_item(query, behavior_intent, behaviors)
            got["behavior"] = [item] if item else []

        verdict = check_case(case, got)
        results.append(verdict)

        flag = "PASS" if verdict["passed"] else "FAIL"
        print(f"\n[{flag}] {case['id']}  {query}")
        for c in verdict["checks"]:
            print(f"      {c['source']:>11}: {'✓' if c['ok'] else '✗'} {c['detail']}")
        if behavior_intent:
            print(f"      · 行为 L3 判定: {behavior_intent}")
        if verbose and not verdict["passed"]:
            for source, items in _summarize_got(got).items():
                if items:
                    print(f"      · 实际 {source}: {items}")

    # ---- 汇总：按 source 的命中率 ----
    total = {"behavior": [0, 0], "corpus": [0, 0], "voice_sample": [0, 0], "phrase": [0, 0],
             "preference": [0, 0], "core_story": [0, 0]}
    for r in results:
        for c in r["checks"]:
            total[c["source"]][1] += 1
            if c["ok"]:
                total[c["source"]][0] += 1
    passed_cases = sum(1 for r in results if r["passed"])
    print(f"\n=== 汇总 ===")
    print(f"case 通过率: {passed_cases}/{len(results)}")
    for src, (ok, n) in total.items():
        if n:
            print(f"  {src:>12}: {ok}/{n}  ({ok / n * 100:.0f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "eval_report.json"
    out_file.write_text(json.dumps({
        "case_total": len(results),
        "case_passed": passed_cases,
        "source_stats": {k: {"pass": v[0], "total": v[1]} for k, v in total.items()},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 报告已保存: {out_file}")
    return 0


def load_cases() -> list:
    """加载标注集（query → 期望命中的样本），供 run_tool 与本脚本共用。"""
    return json.loads(CASES_FILE.read_text(encoding="utf-8")).get("cases", [])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="检索层评测")
    ap.add_argument("--case", default=None, help="只看某个 case id 前缀")
    ap.add_argument("--verbose", action="store_true", help="失败 case 打印实际命中项")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(load_cases(), verbose=args.verbose, only_id=args.case)))


if __name__ == "__main__":
    main()
