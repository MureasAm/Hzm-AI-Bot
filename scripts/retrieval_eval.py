#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索层评测：量化检索层（corpus/behavior 走 RRF + preference/core_story 命中才带）

⚠️ voice_sample / phrase 两路已于 2026-10-04 删除（前者整条通道删，后者并入 behaviors）。
phrase 相关的期望已从用例集里清掉——它恒为空，留着是假绿灯。
在不同 query 下的命中率，让阈值调参从"人肉肉眼看"变成"可测量的回归"。

用法：
    python scripts/retrieval_eval.py              # 跑全部 case
    python scripts/retrieval_eval.py --case b1    # 只看某个 case
    python scripts/retrieval_eval.py --verbose    # 打印每个 source 命中的 top 项 + 分数

标注集：`scripts/retrieval_eval_cases.json`（**唯一的** query→期望 存储；
原 `scripts/judge_labels.json` 已于 2026-09-29 并入，判据实验也从这份推导 want）
每条 case = query + 对各路的期望，**四档从强到弱**（按"你能确定到什么程度"选）：
    {"should": [ids]}        该路必须命中**具体某条**（最强）
    {"contains": [关键词]}   该路 top-k 的文本要含关键词（corpus 用）
    {"should_fire": true}    该路只要有东西就行、**不钉是哪条**（说不出是哪条时用这个）
    {"should_not_hit": true} 该路必须**为空**（负例）
⚠️ 别把"现在返回的东西"钉成 should —— 那是把现状当真理，是反过来的循环论证。

不读线上记忆、只读向量缓存、不写任何数据文件。调两类模型：
  · 智谱 embedding（与线上同款）
  · deepseek（behavior 的 L3 意图分类、corpus 的语义判——**与线上同一条路**，
    评测必须镜像线上，否则测的不是真正跑的东西）
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

# ⚠️ 必须先 init NoneBot：`src/plugins/chatbot/__init__.py` 在模块级就调用
# `@get_driver().on_bot_connect`，不 init 直接导入会崩 "NoneBot has not been initialized"。
# （本文件曾注释"可无 NoneBot 直接导入"，那句后来失效了 —— 于是整个脚本静默跑不起来。）
import nonebot  # noqa: E402
nonebot.init()

from src.plugins.chatbot import retrieval  # noqa: E402
from src.plugins.chatbot.rag import embed_query  # noqa: E402
from src.plugins.chatbot.constants import ZHIPU_BASE_URL, DEEPSEEK_BASE_URL  # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules  # noqa: E402
from src.plugins.chatbot.retrieval import select_behavior_item  # noqa: E402
from src.plugins.chatbot.routing import classify_l3  # noqa: E402  (L3：一次判行为 + 措辞)
from src.plugins.chatbot.corpus_judge import judge_corpus  # noqa: E402  (corpus 语义判)

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
        elif "should_fire" in cond:
            # 该路只要有东西就行，不钉具体是哪条（说不出是哪条时的诚实写法）
            ok = bool(items)
            detail = f"期望该路非空 实际{len(items)}条"
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


async def corpus_items(query: str, qv, ds_client) -> list:
    """corpus（**必须与 core.handle_chat 保持一致**，否则评测测的不是线上的东西）：
    门/钩子只决定候选，**判定全部交 LLM**（2026-09-29 起取消"门放行直通"，
    理由见 `retrieval._corpus_gate_pass`）。
    没有 deepseek client 时退回"只看门"的旧行为（纯 embedding 环境用）。
    """
    if not ds_client:
        return retrieval.retrieve_corpus(query, qv)
    return await judge_corpus(ds_client, query, retrieval.retrieve_corpus_candidates(query, qv))


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


# ==================== 一条消息走六路的实况（评测与问题驱动入口共用） ====================

async def snapshot(query: str, client, ds_client, behaviors) -> tuple:
    """跑一遍**线上真实的六路检索**，返回 `(got, l3)`；`got` 是 source -> 命中项。

    ⚠️ 这里必须与 `core.handle_chat` 同一条路 —— 评测（本文件）和问题驱动入口
    （`problem_cases.py`）都调它，所以"线上到底怎么检索"**只有这一处实现**。
    别在别处再写一遍：自己写一遍"检索应该怎么做"，追的就不是线上跑的东西了（循环论证）。

    `got` 为 None 表示 embedding 失败（调用方自己决定怎么报）。
    """
    qv = await embed_query(client, query)
    if not qv:
        return None, {"behavior": "", "phrases": []}
    # L3：一次调用同时判行为 + 措辞（与 core.handle_chat 同一条路）
    l3 = (await classify_l3(ds_client, query, "", behaviors)
          if ds_client else {"behavior": "", "phrases": []})
    got = {
        "corpus": await corpus_items(query, qv, ds_client),
        "preference": retrieval.retrieve_preferences(query),
        "core_story": retrieval.retrieve_core_stories(query, qv),
        "behavior": [],
    }
    # 行为：LLM 判意图 → 判别词兜底（不再有 embedding 基线）
    item = select_behavior_item(query, l3["behavior"], behaviors)
    got["behavior"] = [item] if item else []
    return got, l3


def format_snapshot(got: dict, l3: dict, full_text: bool = False) -> list:
    """把实况渲染成人读的行（给 `--verbose` 和问题驱动入口共用）。"""
    lines = []
    if l3.get("behavior"):
        lines.append(f"行为 L3 判定：{l3['behavior']}")
    for source, items in got.items():
        if not items:
            lines.append(f"{source:>12}: （空）")
        elif source in ("preference", "core_story"):
            lines.append(f"{source:>12}: " + "、".join(
                f"{i.get('id','')}({i.get('score',0):.3f})" for i in items[:3]))
        elif source == "corpus":
            lines.append(f"{source:>12}: " + "、".join(
                f"#{it.item_id}({it.score:.3f})“{(it.text or '')[:40 if not full_text else 200]}”"
                for it in items[:3]))
        else:
            lines.append(f"{source:>12}: " + "、".join(
                f"{it.item_id}({it.score:.3f})“{(getattr(it,'text','') or '')[:40]}”" for it in items[:3]))
    return lines


# ==================== 主流程 ====================

async def run(cases, verbose: bool, only_id: str = None):
    key = _zhipu_key()
    if not key:
        print("❌ 未找到 ZHIPU_API_KEY（.env.prod）——评测需要与线上同款 embedding")
        return 1
    client = AsyncOpenAI(api_key=key, base_url=ZHIPU_BASE_URL)
    ds_client = AsyncOpenAI(api_key=_deepseek_key(), base_url=DEEPSEEK_BASE_URL) if _deepseek_key() else None

    behaviors = load_persona_rules()
    print("🔍 检索评测 | 行为走 L3（LLM 判意图 + 关键词兜底），其余走 embedding")

    results = []
    for case in cases:
        if only_id and only_id not in case["id"]:
            continue
        query = case["query"]
        got, l3 = await snapshot(query, client, ds_client, behaviors)
        if got is None:
            print(f"[SKIP] {case['id']}  {query}  （embedding 失败）")
            continue
        behavior_intent = l3["behavior"]

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
    total = {"behavior": [0, 0], "corpus": [0, 0],
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
