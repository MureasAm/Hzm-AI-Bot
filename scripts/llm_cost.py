#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 用量分账：把 `data/llm_usage.jsonl` 聚合成"哪条通道花了多少"。

## 为什么有它（2026-10-13）

用户要量清**群聊到底占多少花费**（bot 10-12 单日 ¥8.43，假设是"群里把群友说的话
都读进去"占了大头）。stdout 那行 `[缓存] …` 看完就滚没了，而且不带输出 token、
也分不出私聊/群聊——聚合不出"哪条路贵"。

所以运行时（`chatbot/config.py:log_cache_usage`）每次调用追加一行 JSONL：
`{"t":…, "tag":"主回复", "route":"群", "prompt":…, "hit":…, "miss":…, "output":…}`。
本脚本按 **通道**（私/群/主动）和 **通道×tag** 聚合。

## 关于 ¥

token 那几列是**实测**；¥ 用下面 `PRICE_*` 那套单价算（deepseek-v4.1，含高峰加倍，
用户 2026-10-14 给的官方价）。**调价就改那几个常量**，`--price-*` 可临时覆盖。

⚠️ **v4.1 的价让"谁贵"完全换了个样子**：命中 0.02、未命中 1、输出 4（元/百万），
也就是**命中只有未命中的 1/50、输出是它的 4 倍**。所以别看 prompt 总量，
**要看未命中输入和输出**——10-14 那次重算，主回复只占 22%，而缓存命中率只有 26% 的
`群×接话门`排到了第三。

## 用法

    .venv/Scripts/python.exe scripts/llm_cost.py                 # 全量
    .venv/Scripts/python.exe scripts/llm_cost.py --days 1        # 最近 24 小时
    .venv/Scripts/python.exe scripts/llm_cost.py --by-day        # 按天分开看（对比 10-12 那天）
    .venv/Scripts/python.exe scripts/llm_cost.py --json          # 给别的脚本吃
    .venv/Scripts/python.exe scripts/llm_cost.py --routes         # 只看通道汇总
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

USAGE_FILE = _common.DATA_DIR / "llm_usage.jsonl"

# 单价（元 / 百万 token）——**只在这里改**。用户 2026-10-14 给的 deepseek-v4.1 官方价：
#   缓存命中输入 0.02 ｜ 未命中输入 1 ｜ 输出 4
# ⚠️ 命中只有未命中的 **1/50**、输出是未命中的 **4 倍**——所以"谁贵"跟 v3 时代完全不同：
#    别再用"prompt 总量"排序，要看 miss 和输出。
PRICE_HIT, PRICE_MISS, PRICE_OUT = 0.02, 1.0, 4.0

# **按模型的单价**（元/百万，命中 / 未命中 / 输出）。
#
# 为什么非有不可（2026-10-14）：降级链上线后一本账里**混着好几个模型**，
# 全按一套价算会把账算歪——比如中转的 gemini，未命中输入只有 0.375，
# 是 deepseek 的 1/2.7，按 deepseek 的价报出来会凭空翻倍。
#
# ⚠️ **高峰加倍只对 deepseek 官方成立**（用户 2026-10-14 给的口径是 deepseek 的），
# 中转站那几家没这条——所以加倍只加在 `PEAK_MODELS` 里的模型上。
PRICE_BY_MODEL = {
    # 官方 deepseek（FALLBACK_*，链的最后一层）
    "deepseek-flash": (0.02, 1.0, 4.0),
    # 中转站（价格由用户 2026-10-14 提供）
    # ⚠️ 两个站挂着**同名的 gemini 但价不一样**——刚好模型名不同，能分开：
    "gemini-3.1-flash-lite-preview": (0.0125, 0.125, 0.75),   # krill（现挂）
    "gemini-3.1-flash-lite": (0.0125, 0.125, 0.75),            # linkapi（与 krill 同价）
    "gemini-3.8-flash": (0.0375, 0.375, 1.88),
    "claude-haiku-4-5-20251001": (0.0342, 0.342, 1.71),
    "deepseek-v4.1-flash": (0.014, 0.7, 2.8),      # 中转七折
    "gpt-5.6-luna": (0.02, 0.2, 1.2),
}

# 只有这几个模型吃高峰加倍（deepseek 官方的口径）；中转站的不加
PEAK_MODELS = {"deepseek-flash"}

# 中转站这些模型**把缓存量重复算进了 prompt**（2026-10-14 实测）：
# 同一段提示词连问两次，第一次 `prompt=1646`，第二次 `prompt=3182`——正好 1646+1536（cached）。
# 所以对它们要先减掉那一份，否则输入量凭空翻倍。
DOUBLE_COUNTS_CACHE = {"deepseek-v4.1-flash"}


def _prices_for(rec: dict, fallback):
    """这条记录用哪套单价：**认识这个模型就用它自己的**，不认识就用传进来的默认/命令行覆盖。"""
    return PRICE_BY_MODEL.get(str(rec.get("model") or ""), fallback)

# 高峰时段（本地时间，左闭右开）+ 倍率：用户 2026-10-14 给的口径
# ——「9~12 以及 14~18 价钱双倍」。**按每条记录自己的小时算**，不是按整本账算。
PEAK_HOURS = ((9, 12), (14, 18))
PEAK_MULT = 2.0


def _mult_for(ts: float, model: str = "") -> float:
    """这条记录落在高峰段吗（本地时间）。**只对 `PEAK_MODELS` 里的模型加倍**。"""
    if model and model not in PEAK_MODELS:
        return 1.0
    hour = datetime.fromtimestamp(ts).hour
    return PEAK_MULT if any(a <= hour < b for a, b in PEAK_HOURS) else 1.0

# 没设通道的那部分（启动时的预热、桥接推送里没标 route 的调用…）
NO_ROUTE = "（未标）"

# 每个 tag 在干什么 —— 报告末尾直接附上，省得看数字的人回头翻代码。
# ⚠️ 加了新的 `log_cache_usage(resp, "新名字")` 就把这里补一行，否则报告里会只剩一个数字。
TAG_DOCS = {
    "主回复": ("生成她真正发给用户的那句话", "core.generate_reply —— **唯一用户看得见的调用**", "每轮 1 次"),
    "话题探测": ("判话题延续/切换 + 短消息补全语义", "session_memory.probe_session（判完同步注入）", "每轮 1 次"),
    "行为分类": ("拿用户这句去 17 条 behaviors 里判「这是哪种情景」+ 数连续索要第几轮",
                 "routing.classify_l3", "每轮 1 次"),
    "梗确认": ("梗库关键词命中后，判「这句真是那个梗吗」", "routing.legendary_confirmed", "关键词命中才跑（低频）"),
    "直播记忆判": ("从检索到的候选里判「用户是不是在问她这段经历」", "corpus_judge.judge_corpus", "每轮 1 次"),
    "接话门": ("群里这批要不要开口", "group_gate.should_reply_in_group", "**群独有**，每批 1 次"),
    "记忆提取": ("把这一轮里的名字/事实/承诺写进长期记忆（异步，不影响回复）",
                 "memory_v2.extract_and_ingest", "私聊每轮 1 次"),
    "表情包": ("判「有没有一张表情跟这句话十分对应」", "stickers.pick_sticker", "有冷却，约每 3 轮 1 次"),
    "批量归纳": ("把用户连发的几条压成一两句话", "core.summarize_batch", "攒批 ≥2 条才跑"),
    "转发归纳": ("把转发的聊天记录压成一两句", "core.summarize_forward", "有转发才跑"),
    "术语消歧": ("判命中的黑话在**当前语境**下是不是那个意思", "core.confirm_ambiguous_terms", "命中且标了 confirm 才跑"),
    "主动发言": ("动态/微博 → 私聊里那句（同一个 tag 含两步：先判「说不说得清」，再生成）",
                 "proactive.compose_proactive", "每条动态 1~2 次"),
}


def iter_records(path: Path):
    """逐条产出账本记录，坏行跳过（绝不因为一条脏数据让整个分析挂掉）。"""
    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            yield rec


def _tokens(rec: dict) -> tuple[int, int, int]:
    """(prompt, 命中, 未命中, 输出) —— 上游没给缓存字段时，prompt 全算**未命中**。

    为什么要这个兜底：网关/别的模型可能不返回 `prompt_cache_*`，那些记录的 hit/miss
    都是 0——直接按 0 算会把它们的 prompt 从账单里凭空抹掉（总量凭空变小）。
    """
    prompt = int(rec.get("prompt") or 0)
    hit = int(rec.get("hit") or 0)
    miss = int(rec.get("miss") or 0)
    out = int(rec.get("output") or 0)
    if str(rec.get("model") or "") in DOUBLE_COUNTS_CACHE:
        prompt = max(prompt - hit, 0)          # 先把它多算的那份减回去
    if hit + miss == 0:
        miss = prompt                          # 完全没有缓存字段 → 整段都算未命中
    elif miss == 0 and prompt > hit:
        # 报了命中却没报未命中（2026-10-14 实测 1182/1842 条 v4.1 记录是这样）——
        # 不改的话剩下那截**一分钱都不算**，2.52M 输入 token 会凭空消失。
        miss = prompt - hit
    return prompt, hit, miss, out


def _yen(hit: int, miss: int, out: int, prices, mult: float = 1.0) -> float:
    ph, pm, po = prices
    return (hit / 1e6 * ph + miss / 1e6 * pm + out / 1e6 * po) * mult


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_yen(y: float) -> str:
    """小额给 4 位小数——只跑了半天时全是 ¥0.00 的话，那一列就白看了。"""
    return f"¥{y:.2f}" if y >= 0.01 else f"¥{y:.4f}"


def _table(rows: list[tuple], headers: list[str], aligns: list[str]) -> str:
    """手搓一张对齐的表（不引第三方依赖，输出要给用户直接看）。"""
    widths = [len(h) for h in headers]
    cells = []
    for row in rows:
        c = [str(x) for x in row]
        cells.append(c)
        for i, x in enumerate(c):
            widths[i] = max(widths[i], sum(2 if ord(ch) > 127 else 1 for ch in x))
    def pad(s, w, a):
        cur = sum(2 if ord(ch) > 127 else 1 for ch in s)
        fill = " " * max(0, w - cur)
        return fill + s if a == "r" else s + fill
    out = ["  " + "  ".join(pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))]
    out.append("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for c in cells:
        out.append("  " + "  ".join(pad(c[i], widths[i], aligns[i]) for i in range(len(c))))
    return "\n".join(out)


def summarize(records: list[dict], prices, peak: bool = True) -> dict:
    """按通道 / 通道×tag / 天聚合。每格是 [调用, prompt, hit, miss, output, ¥]。

    ¥ **逐条算**（按每条自己的小时决定要不要加倍），不是"先把 token 加总再乘"——
    否则高峰/低谷混在一本账里就没法定价了。
    """
    by_route = defaultdict(lambda: [0, 0, 0, 0, 0, 0.0])
    by_pair = defaultdict(lambda: [0, 0, 0, 0, 0, 0.0])
    by_day = defaultdict(lambda: [0, 0, 0, 0, 0, 0.0])
    models = set()
    for rec in records:
        prompt, hit, miss, out = _tokens(rec)
        route = str(rec.get("route") or "") or NO_ROUTE
        tag = str(rec.get("tag") or "（无 tag）")
        t = float(rec.get("t") or 0)
        day = datetime.fromtimestamp(t).strftime("%Y-%m-%d")
        model = str(rec.get("model") or "")
        y = _yen(hit, miss, out, _prices_for(rec, prices),
                 _mult_for(t, model) if peak else 1.0)
        for bucket, key in ((by_route, route), (by_pair, (route, tag)), (by_day, day)):
            b = bucket[key]
            b[0] += 1
            b[1] += prompt
            b[2] += hit
            b[3] += miss
            b[4] += out
            b[5] += y
        if rec.get("model"):
            models.add(str(rec["model"]))
    return {"route": by_route, "pair": by_pair, "day": by_day, "models": models}


def _print_bucket(title: str, bucket: dict, prices, total_yen: float, limit: int | None = None,
                  baseline: dict | None = None, hours: float | None = None):
    """打一张聚合表。

    `baseline`（通道 → 该通道"主回复"的调用数）给了就多两列"次/回复"——**这是"为什么贵"的
    最直接的一列**：私聊每轮要跑 5 个调用，群聊每批 4 个。`hours` 给了就多一列"折合¥/天"。
    """
    rows = []
    for key, vals in bucket.items():
        calls, prompt, hit, miss, out, y = vals      # ¥ 在聚合时逐条算好了（含高峰加倍）
        label = "×".join(key) if isinstance(key, tuple) else key
        rows.append((y, label, calls, prompt, hit, out, key))
    rows.sort(reverse=True)
    if limit:
        rows = rows[:limit]

    print(f"\n{title}")
    headers = ["通道/条目", "调用", "prompt", "命中率", "输出", "估算¥", "占比"]
    aligns = ["l", "r", "r", "r", "r", "r", "r"]
    if baseline is not None:
        headers[2:2] = ["次/回复", "prompt/次", "输出/次", "¥/次"]
        aligns[2:2] = ["r", "r", "r", "r"]
    if hours:
        headers.insert(len(headers) - 1, "折合¥/天")
        aligns.insert(len(aligns) - 1, "r")

    table = []
    for y, label, calls, prompt, hit, out, key in rows:
        share = f"{y / total_yen:.0%}" if total_yen > 0 else "—"
        rate = f"{hit / prompt:.0%}" if prompt else "—"
        per = "—"
        if baseline is not None:
            route = key[0] if isinstance(key, tuple) else key
            base = (baseline or {}).get(route, 0)
            per = f"{calls / base:.1f}" if base else "—"
        row = [label, calls, _fmt_tok(prompt), rate, _fmt_tok(out), _fmt_yen(y), share]
        if baseline is not None:
            row[2:2] = [per, _fmt_tok(round(prompt / calls)) if calls else "—",
                        _fmt_tok(round(out / calls)) if calls else "—",
                        _fmt_yen(y / calls) if calls else "—"]
        if hours:
            row.insert(len(row) - 1, _fmt_yen(y * 24 / hours))
        table.append(tuple(row))
    print(_table(table, headers, aligns))


def _print_tag_docs(bucket: dict):
    """每个 tag 在干什么（只列报告里真出现过的 tag）。"""
    seen = []
    for key in bucket:
        tag = key[1] if isinstance(key, tuple) else key
        if tag not in seen:
            seen.append(tag)
    print("\n这些调用分别在干什么（按报告里出现的 tag）：")
    for tag in seen:
        if tag in TAG_DOCS:
            what, where, freq = TAG_DOCS[tag]
            print(f"  {tag:<6} {what}")
            print(f"  {'':<6}   ↳ {where} ｜ {freq}")
        else:
            print(f"  {tag:<6} ⚠️ 这个 tag 还没写进 scripts/llm_cost.py 的 TAG_DOCS，去补一行")


def main(argv=None) -> int:
    _common.ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="llm_cost", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(USAGE_FILE), help="账本路径（默认 data/llm_usage.jsonl）")
    ap.add_argument("--days", type=float, default=None, help="只看最近 N 天（小数可，如 0.5）")
    ap.add_argument("--date", default=None, help="只看某一天，如 2026-10-13")
    ap.add_argument("--by-day", action="store_true", help="按天分开看")
    ap.add_argument("--routes", action="store_true", help="只看通道汇总（不列 tag 明细）")
    ap.add_argument("--top", type=int, default=25, help="tag 明细最多列几行（默认 25）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（给别的脚本吃）")
    ap.add_argument("--no-peak", action="store_true",
                    help="不按高峰时段加倍（默认加：9~12 与 14~18 双倍）")
    ap.add_argument("--price-hit", type=float, default=PRICE_HIT, help="缓存命中单价 元/百万")
    ap.add_argument("--price-miss", type=float, default=PRICE_MISS, help="缓存未命中单价 元/百万")
    ap.add_argument("--price-out", type=float, default=PRICE_OUT, help="输出单价 元/百万")
    args = ap.parse_args(argv)
    prices = (args.price_hit, args.price_miss, args.price_out)

    path = Path(args.file)
    records = list(iter_records(path))
    if args.days is not None:
        cutoff = time.time() - args.days * 86400
        records = [r for r in records if float(r.get("t") or 0) >= cutoff]
    if args.date:
        records = [r for r in records
                   if datetime.fromtimestamp(float(r.get("t") or 0)).strftime("%Y-%m-%d") == args.date]

    if not records:
        print(f"📒 账本里没有可用记录：{path}")
        print("   （记账从 2026-10-13 起才有。bot 得先跑起来、真的产生过 LLM 调用；"
              " 被 LLM_USAGE=0 关掉时也不会记。）")
        return 0

    agg = summarize(records, prices, peak=not args.no_peak)

    def totals(bucket):
        calls = sum(v[0] for v in bucket.values())
        prompt = sum(v[1] for v in bucket.values())
        hit = sum(v[2] for v in bucket.values())
        miss = sum(v[3] for v in bucket.values())
        out = sum(v[4] for v in bucket.values())
        return calls, prompt, hit, miss, out, sum(v[5] for v in bucket.values())

    calls, prompt, hit, miss, out, yen = totals(agg["route"])

    if args.json:
        print(json.dumps({
            "calls": calls, "prompt": prompt, "hit": hit, "miss": miss,
            "output": out, "yen": round(yen, 4), "prices": prices,
            "by_route": {k: v for k, v in agg["route"].items()},
            "by_pair": {f"{a}×{b}": v for (a, b), v in agg["pair"].items()},
            "by_day": {k: v for k, v in agg["day"].items()},
        }, ensure_ascii=False, indent=2))
        return 0

    ts = [float(r.get("t") or 0) for r in records]
    fmt = lambda x: datetime.fromtimestamp(x).strftime("%m-%d %H:%M")
    hours = max((max(ts) - min(ts)) / 3600, 0.1)
    # 「次/回复」的基准：每条通道里"主回复"跑了几次 = 这条通道回复了几轮
    baseline = {route: v[0] for (route, tag), v in agg["pair"].items() if tag == "主回复"}
    print(f"📒 LLM 用量分账 —— {path}")
    print(f"   区间 {fmt(min(ts))} ~ {fmt(max(ts))}（{hours:.1f} 小时）｜ 记录 {len(records)} 条"
          f" ｜ 模型 {'/'.join(sorted(agg['models'])) or '（未记）'}")
    print(f"\n   总计：调用 {calls} 次 ｜ prompt {_fmt_tok(prompt)} tok"
          f"（命中 {_fmt_tok(hit)} / {hit / prompt:.0%}，未命中 {_fmt_tok(miss)}）"
          f" ｜ 输出 {_fmt_tok(out)} tok")
    print(f"   估算花费 {_fmt_yen(yen)}（这本账的 {hours:.1f} 小时）"
          f"｜ **折合 {_fmt_yen(yen * 24 / hours)}/天**")
    known = sorted({str(r.get("model") or "") for r in records} & set(PRICE_BY_MODEL))
    print(f"   **按模型定价**：认识的模型用自己的价（{ '、'.join(known) or '无' }），"
          f"其余用 命中 {prices[0]} / 未命中 {prices[1]} / 输出 {prices[2]} 元每百万")
    print(f"   ⚠️ 单价是手填的、不是账单；对不上就拿 --price-* 覆盖。"
          f"⚠️ 中转站有的通道不报缓存（实测 haiku / gemini-3.8-flash 全是 0），那些按全价估")
    if not args.no_peak:
        peak_recs = [r for r in records
                     if _mult_for(float(r.get("t") or 0), str(r.get("model") or "")) > 1]
        peak_y = sum(_yen(*_tokens(r)[1:], _prices_for(r, prices),
                          _mult_for(float(r.get("t") or 0), str(r.get("model") or "")))
                     for r in peak_recs)
        off_y = yen - peak_y
        print(f"   高峰段（9~12 / 14~18，×{PEAK_MULT:g}）：{len(peak_recs)} 次调用花了 "
              f"{_fmt_yen(peak_y)}（占 {peak_y / yen:.0%}）；其余 {_fmt_yen(off_y)}")
        print(f"   ⚠️ 省钱最直接的两根杠杆是**未命中输入**和**输出**（命中只有 1/50 价）："
              f"本账未命中 {_fmt_tok(miss)} tok、输出 {_fmt_tok(out)} tok")

    if args.by_day:
        _print_bucket("按天：", agg["day"], prices, totals(agg["day"])[5])
    _print_bucket("按通道：", agg["route"], prices, yen, baseline=baseline, hours=hours)
    if not args.routes:
        _print_bucket(f"按通道×tag（前 {args.top}，按花费排序）：",
                      agg["pair"], prices, yen, limit=args.top,
                      baseline=baseline, hours=hours)
        # 「次/回复」列在群里要按"批"读：群聊 4.9 次/回复 是因为大部分批次过了门但没开口
        print("\n  ⚠️ 群里「次/回复」偏大是正常的：接话门对**每一批**都跑，多数批次过了门但没开口。")
        _print_tag_docs(agg["pair"])
    print("\n（token 是实测；¥ 是按上面那套单价估的——**换模型/调价后要改脚本里的 PRICE_***）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
