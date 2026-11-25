#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享材料检索评测：复用 gather_retrieval 的选择、融合与预算结果。

⚠️ voice_sample / phrase 两路已于 2026-10-04 删除（前者整条通道删，后者并入 behaviors）。
phrase 相关的期望已从用例集里清掉——它恒为空，留着是假绿灯。
在不同 query 下的命中率，让阈值调参从"人肉肉眼看"变成"可测量的回归"。

用法：
    python scripts/retrieval_eval.py              # 跑全部 case
    python scripts/retrieval_eval.py --case b1    # 只看某个 case
    python scripts/retrieval_eval.py --verbose    # 打印每个 source 命中的 top 项 + 分数

标注集：`scripts/retrieval_eval_cases.json`（**唯一的** query→期望 存储；
原 `scripts/judge_labels.json` 已于 2026-09-29 并入，判据实验也从这份推导 want）
每条 case = query + 对各路的期望（多个条件必须全部成立）：
    {"should": [ids]}        该路必须命中**具体某条**（最强）
    {"contains": [关键词]}   普通来源命中任一备选词；facts 须覆盖全部给定信息
    {"should_fire": true}    该路只要有东西就行、**不钉是哪条**（说不出是哪条时用这个）
    {"should_not_hit": true} 该路必须**为空**（负例）
    {"must_not": [ids]}      禁止具体条目；可单独使用，也可与上述条件同时要求成立
    {"allowed": [ids]}       可为空；若有条目，只能来自所列范围，不自动要求命中
facts 只汇总最终已确认并实际可用的事实材料，不含候选、预算丢弃项、行为风格例句或模型最终回复。
⚠️ 别把"现在返回的东西"钉成 should —— 那是把现状当真理，是反过来的循环论证。

不运行会话探测、用户记忆、回复生成或发送，不修改机器人状态。
共享检索入口会使用 embedding、L3 与材料选择客户端；报告写到 outputs，
可用 --output 指定 JSON 文件。这不是完整聊天运行时的端到端评测。
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
try:
    nonebot.get_driver()
except ValueError:
    nonebot.init()

from src.plugins.chatbot import retrieval  # noqa: E402
from src.plugins.chatbot.config import _get_clients  # noqa: E402
from src.plugins.chatbot.constants import ZHIPU_BASE_URL  # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules  # noqa: E402
from src.plugins.chatbot.core import build_terms_note, gather_retrieval  # noqa: E402
from src.plugins.chatbot.retrieval_context import RetrievalContext  # noqa: E402
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


# ==================== 单条 case 判定 ====================

def _ids_of(items, is_dict_style: bool):
    if is_dict_style:
        return [str(i.get("id", "")) for i in items]
    return [str(i.item_id) for i in items]


def _text_of(item) -> str:
    return str((item.get("text") if isinstance(item, dict) else getattr(item, "text", "")) or "")


def _valid_condition(cond) -> bool:
    supported = {"should", "contains", "should_fire", "should_not_hit", "must_not", "allowed"}
    if not isinstance(cond, dict) or not cond or set(cond) - supported:
        return False
    for key in ("should", "contains", "must_not", "allowed"):
        if key not in cond:
            continue
        value = cond[key]
        if (not isinstance(value, list)
                or any(not isinstance(item, str) or not item.strip() for item in value)):
            return False
        if not value and (key in {"should", "contains"} or (key == "must_not" and len(cond) == 1)):
            return False
    return all(cond[key] is True for key in ("should_fire", "should_not_hit") if key in cond)


def check_case(case: dict, got: dict) -> dict:
    """按期望判定一条 case。got: source -> 检索返回项列表。"""
    expect = case.get("expect", {})
    if not expect:
        return {"id": case["id"], "query": case["query"], "passed": False,
                "checks": [{"source": "_case", "ok": False, "cond": {},
                            "detail": "缺少期望标注，未执行有效检查", "missing": True}]}
    passed = True
    checks = []
    for source, cond in expect.items():
        items = got.get(source, [])
        missing = source not in got or got[source] is None
        dict_style = source in ("preference", "core_story", "associations", "terms", "facts")
        if not _valid_condition(cond):
            ok, detail, missing = False, "缺少有效或包含不支持的判定条件", True
        elif missing:
            ok, detail = False, "该路结果缺失，不能按空结果判定"
        else:
            ids = _ids_of(items, dict_style)
            outcomes, details = [], []
            if "should" in cond:
                hit = [item_id for item_id in cond["should"] if item_id in ids]
                outcomes.append(bool(hit))
                details.append(f"期望{cond['should']} 命中{hit} 实际{ids[:3]}")
            if "should_fire" in cond:
                outcomes.append(bool(items))
                details.append(f"期望该路非空 实际{len(items)}条")
            if "should_not_hit" in cond:
                outcomes.append(not items)
                details.append(f"期望不命中 实际{len(items)}条")
            if "contains" in cond:
                texts = [_text_of(item) for item in items]
                hit = [word for word in cond["contains"] if any(word in text for text in texts)]
                # 旧 corpus.contains 是同义词备选；facts 是跨材料的必需信息清单。
                outcomes.append(len(hit) == len(cond["contains"]) if source == "facts" else bool(hit))
                details.append(f"期望文本含{cond['contains']} 实际覆盖{hit}")
            if "allowed" in cond:
                unexpected = [item_id for item_id in ids if item_id not in cond["allowed"]]
                outcomes.append(not unexpected)
                details.append(f"允许为空或仅{cond['allowed']} 范围外{unexpected}")
            if "must_not" in cond:
                forbidden_hits = [item_id for item_id in cond["must_not"] if item_id in ids]
                outcomes.append(not forbidden_hits)
                details.append(f"禁止{cond['must_not']} 误中{forbidden_hits}")
            ok, detail = all(outcomes), "；".join(details)
        passed = passed and ok
        checks.append({"source": source, "ok": ok, "cond": cond, "detail": detail,
                       "missing": missing})
    return {"id": case["id"], "query": case["query"], "passed": passed, "checks": checks}


async def corpus_items(query: str, qv, ds_client) -> list:
    """保留旧诊断脚本的 corpus 单路接口；完整评测使用共享 gather_retrieval。"""
    if not ds_client:
        return []
    return await judge_corpus(ds_client, query, retrieval.retrieve_corpus_candidates(query, qv))


def _summarize_got(got: dict, top: int = 3) -> dict:
    """压缩各 source 命中项（id + 分数），供 --verbose 打印。"""
    out = {}
    for source, items in got.items():
        if source in ("preference", "core_story", "associations", "terms", "facts"):
            out[source] = [(str(i.get("id", "")), round(i.get("score", 0), 3)) for i in items[:top]]
        elif source == "corpus":
            out[source] = [(it.item_id, round(it.score, 3), it.text[:18]) for it in items[:top]]
        else:
            out[source] = [(it.item_id, round(it.score, 3)) for it in items[:top]]
    return out


# ==================== 共享材料检索实况（评测与问题驱动入口共用） ====================

async def snapshot(query: str, client, ds_client, behaviors, *, history: str = "",
                   resolved_query: str = "", probe_meta: dict | None = None,
                   is_group: bool = False, addressed: bool = True) -> tuple:
    """复用共享检索结果返回 `(got, l3)`，不运行探测器或完整聊天链。

    可提供已有历史、指代补全和 probe 元数据；behaviors 参数仅保留旧调用兼容，
    行为数据由共享入口加载。嵌入失败仍返回其他来源；l3 的下划线字段记录降级。
    """
    context = RetrievalContext.from_probe(query, resolved_query, history, probe_meta,
                                         is_group=is_group, addressed=addressed)
    result = await gather_retrieval(
        query, context.query, history, ds_client, client, context=context)
    l3 = dict(result.get("l3") or {"behavior": "", "phrases": []})
    trace = result.get("selection_trace") or {}
    reasons = []
    if result.get("embedding_available") is False:
        reasons.append("embedding_unavailable")
    if trace.get("available") is False:
        reasons.append("selection_unavailable")
    classification_available = l3.get("classification_ok")
    if classification_available is False:
        reasons.append("l3_unavailable")
    needs_available = l3.get("needs_ok")
    if needs_available is False:
        reasons.append("needs_unavailable")
    l3.update(_embedding_available=result.get("embedding_available"),
              _selection_trace=trace, _classification_available=classification_available,
              _needs_available=needs_available, _degraded=bool(reasons), _degraded_reasons=reasons)
    term_hits: list = []
    term_text = build_terms_note(query, denied_terms=result.get("denied_terms") or set(), hit_out=term_hits,
                                selected_terms=result.get("approved_terms") or set())
    fused = result.get("fused_items") or []
    got = {
        "corpus": [item for item in fused if item.source == "corpus"],
        "preference": result.get("preference_items") or [],
        "core_story": result.get("core_stories") or [],
        "behavior": [item for item in fused if item.source == "behavior"],
        "associations": [{"id": e, "score": 1.0} for e in result.get("association_entities") or []],
        "terms": [{"id": kw, "score": 1.0} for kw in term_hits],
    }
    facts = [{"id": f"{source}:{item.get('id') if isinstance(item, dict) else item.item_id}",
              "source": source, "text": _text_of(item)}
             for source in ("corpus", "preference", "core_story")
             for item in got[source] if _text_of(item)]
    for source, text in (("terms", term_text), ("associations", result.get("association_context"))):
        if text:
            facts.append({"id": source, "source": source, "text": text})
    got["facts"] = facts
    return got, l3


def format_snapshot(got: dict, l3: dict, full_text: bool = False) -> list:
    """把实况渲染成人读的行（给 `--verbose` 和问题驱动入口共用）。"""
    lines = []
    if l3.get("_degraded"):
        lines.append("检索降级：" + "、".join(l3.get("_degraded_reasons") or ["unavailable"]))
    if l3.get("behavior"):
        lines.append(f"行为 L3 判定：{l3['behavior']}")
    for source, items in got.items():
        if not items:
            lines.append(f"{source:>12}: （空）")
        elif source in ("preference", "core_story", "associations", "terms", "facts"):
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

async def run(cases, verbose: bool, only_id: str = None, *, output: Path | str | None = None):
    key = _zhipu_key()
    setup_reasons = []
    if not key:
        print("⚠️ 未找到 ZHIPU_API_KEY——仍逐条评测，并标记 embedding 不可用")
        setup_reasons.append("embedding_client_unavailable")
    client = AsyncOpenAI(api_key=key, base_url=ZHIPU_BASE_URL) if key else None
    # ⚠️ 对话客户端**必须借线上那个**（`config._get_clients()`），别自己拿
    #    `OPENAI_API_KEY + DEEPSEEK_BASE_URL` 拼。
    #    这里原来就是那么拼的：2026-10-14 线上把 `OPENAI_*` 换成了**中转站**（linkapi），
    #    而下面写死的仍是 DeepSeek 官方的 base_url —— 等于把中转站的 key 发给 DeepSeek 官方，
    #    **每一条 L3 / corpus 判定都 401**，脚本却照常打 PASS/FAIL、照常出汇总。
    #    （实测 2026-10-15：48 条只过 33 条、behavior 7/18，全是这个 401 造出来的假红。）
    #    借线上那个还顺带拿到降级链，与「corpus 一路镜像线上」的定位一致。
    try:
        ds_client, _ = _get_clients()
    except Exception as error:
        ds_client = None
        setup_reasons.append(f"chat_client_unavailable:{type(error).__name__}")

    behaviors = load_persona_rules()
    print("🔍 检索评测 | 共享材料选择、融合与实际预算结果；降级用例保留在分母")

    results = []
    for case in cases:
        if only_id and only_id not in case["id"]:
            continue
        query = case["query"]
        try:
            got, l3 = await snapshot(
                query, client, ds_client, behaviors, history=case.get("history", ""),
                resolved_query=case.get("resolved_query", ""), probe_meta=case.get("probe_meta"),
                is_group=case.get("is_group", False), addressed=case.get("addressed", True))
        except Exception as error:
            got, l3 = None, {"_degraded_reasons": [f"snapshot_error:{type(error).__name__}"]}
        reasons = list(setup_reasons) + list(l3.get("_degraded_reasons") or [])
        if got is None:
            got = {}
            reasons.append("retrieval_results_unavailable")
        degraded = bool(reasons or l3.get("_degraded"))
        behavior_intent = l3.get("behavior", "")

        verdict = check_case(case, got)
        verdict["retrieval_passed"] = verdict["passed"]
        verdict["degraded"] = degraded
        verdict["degraded_reasons"] = reasons
        verdict["diagnostics"] = {k: v for k, v in l3.items() if k.startswith("_")}
        verdict["approved_facts"] = got.get("facts")
        verdict["passed"] = verdict["passed"] and not degraded
        results.append(verdict)

        flag = "DEGRADED" if degraded else ("PASS" if verdict["passed"] else "FAIL")
        print(f"\n[{flag}] {case['id']}  {query}")
        if reasons:
            print("      · 降级原因: " + "、".join(reasons))
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
             "preference": [0, 0], "core_story": [0, 0],
             "associations": [0, 0], "terms": [0, 0]}
    source_degraded = {}
    for r in results:
        for c in r["checks"]:
            total.setdefault(c["source"], [0, 0])
            total[c["source"]][1] += 1
            if r["degraded"]:
                source_degraded[c["source"]] = source_degraded.get(c["source"], 0) + 1
            elif c["ok"]:
                total[c["source"]][0] += 1
    passed_cases = sum(1 for r in results if r["passed"])
    degraded_cases = sum(1 for r in results if r["degraded"])
    failed_cases = len(results) - passed_cases - degraded_cases
    checks = [check for result in results for check in result["checks"]]
    missing_checks = sum(bool(check["missing"]) for check in checks)
    print(f"\n=== 汇总 ===")
    print(f"case 通过率: {passed_cases}/{len(results)}")
    print(f"失败: {failed_cases} | 降级: {degraded_cases} | 缺失检查: {missing_checks}/{len(checks)}")
    for src, (ok, n) in total.items():
        if n:
            print(f"  {src:>12}: {ok}/{n}  ({ok / n * 100:.0f}%) 降级{source_degraded.get(src, 0)}")

    out_file = Path(output) if output is not None else OUT_DIR / "eval_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "case_total": len(results),
        "case_passed": passed_cases,
        "case_failed": failed_cases,
        "case_degraded": degraded_cases,
        "check_total": len(checks),
        "check_missing": missing_checks,
        "source_stats": {k: {"pass": v[0], "total": v[1], "degraded": source_degraded.get(k, 0)}
                         for k, v in total.items()},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 报告已保存: {out_file}")
    return int(not results or failed_cases > 0 or degraded_cases > 0 or missing_checks > 0)


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
    ap.add_argument("--output", type=Path, default=None, help="JSON 报告文件路径")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(load_cases(), verbose=args.verbose, only_id=args.case, output=args.output)))


if __name__ == "__main__":
    main()
