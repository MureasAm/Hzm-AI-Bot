# -*- coding: utf-8 -*-
"""历史注入通道 A/B：她的原话走 assistant，vs 现在的「转述 + system」。

只变一个变量：最近几轮历史怎么进提示词。
  A（现状）：system 里的【最近对话记录】，她的句子先用 self_speech 转述。
  B（assistant）：同样的历史原句，改成真正的 user / assistant 消息。
  C（混合）：最近 2 轮走 assistant 原句，更早的仍走 A。

其余全部相同：同一份人格、同一张用户卡、同一条当前用户消息、同一套生成参数。
每一轮还会追加一次「你刚才跟我说什么了？」的追问，量她记不记得自己说过的话。

用法：
    .venv/Scripts/python.exe scripts/run_tool.py history-channel-ab --turns 10 --samples 1
    .venv/Scripts/python.exe scripts/run_tool.py history-channel-ab --dry   # 只看提示词差异

产物：outputs/history_channel_ab/report.md + results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

try:  # Windows 控制台可能是 GBK；日志里有 ⚠️ 会直接崩
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "history_channel_ab"
PROBE = "你刚才跟我说什么了？"
_RECENT_ECHO_RE = re.compile(r"刚(?:才)?(?:不是)?说了|不是说过|问两遍|说过几遍|刚说")


def _load_turns(uid: str) -> list[dict]:
    path = ROOT / "data" / "chat_log" / f"{uid}.jsonl"
    rows = []
    for line in path.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        u = str(d.get("user") or "").strip()
        a = str(d.get("reply") or "").strip()
        if len(u) < 2 or len(a) < 4:
            continue
        rows.append({"t": float(d.get("t") or 0), "user": u, "reply": a})
    rows.sort(key=lambda r: r["t"])
    return rows


def _pick(rows: list[dict], n: int) -> list[tuple[int, dict]]:
    """取最近 n 个「前面至少有 3 轮」的轮次，保持时间顺序。"""
    return [(i, rows[i]) for i in range(3, len(rows))][-n:]


def _history_lines(rows: list[dict]) -> list[str]:
    from src.plugins.chatbot import memory
    out = []
    for r in rows:
        lead = memory._timed_prefix(r["t"])
        out.append(f"{lead}用户：{r['user']}")
        out.append(f"{lead}灰泽满：{r['reply']}")
    return out


def _ngrams(text: str, n: int = 3) -> set[str]:
    s = re.sub(r"[^一-龥A-Za-z0-9]", "", text or "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def _jaccard(a: str, b: str, n: int = 3) -> float:
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _state_block(state: str) -> str:
    from src.plugins.chatbot import injection as inj
    return inj.T("state", "stable", lines=state) if state else ""


def _native_turns(rows: list[dict]) -> list[dict]:
    from src.plugins.chatbot import memory
    out = []
    for r in rows:
        lead = memory._timed_prefix(r["t"])
        out.append({"role": "user", "content": f"{lead}{r['user']}"})
        out.append({"role": "assistant", "content": r["reply"]})
    return out


async def _paraphrase(rows: list[dict], user: str):
    from src.plugins.chatbot import self_speech
    return await self_speech.annotate_turn(_history_lines(rows), user)


def _insert_before_final_user(messages: list[dict], block: list[dict]) -> list[dict]:
    out, inserted = [], False
    for m in messages:
        if m.get("role") == "user" and not inserted:
            out.extend(block)
            inserted = True
        out.append(m)
    return out


_PLACEHOLDER = "__AB_HISTORY_PLACEHOLDER__"


def _replace_history_at_position(messages: list[dict], native_block: list[dict],
                                 *, keep_old_lines: bool = False) -> list[dict]:
    """把 A 版的历史块位置换成 native turns，**保持块的位置不变**。

    不这么做的话，B 会被插到最后一条用户消息前，比 A 更“近”，
    测出来的差异就不只是通道，而是通道 + 位置。
    """
    from src.plugins.chatbot import injection as inj
    contradiction = inj.T("history", "contradiction").strip()
    for i, m in enumerate(messages):
        content = str(m.get("content", ""))
        if m.get("role") != "system" or not content.startswith("【最近对话记录】"):
            continue
        old_lines: list[str] = []
        if keep_old_lines:
            body = content.split("\n", 1)[1] if "\n" in content else ""
            for ln in body.split("\n"):
                s = ln.strip()
                if not s or s == _PLACEHOLDER or s.startswith("【别自相矛盾】"):
                    continue
                old_lines.append(ln)
        block: list[dict] = []
        if old_lines:
            block.append({"role": "system",
                          "content": "【最近对话记录】\n" + "\n".join(old_lines)})
        block.extend(native_block)
        if contradiction:
            block.append({"role": "system", "content": contradiction})
        return messages[:i] + block + messages[i + 1:]
    block = list(native_block)
    if contradiction:
        block.append({"role": "system", "content": contradiction})
    return _insert_before_final_user(messages, block)


async def _prepare_a(user: str, state: str, rows: list[dict]) -> tuple[list[dict], list[str], bool]:
    from src.plugins.chatbot import core, memory
    hist, gen_user = await _paraphrase(rows, user)
    miss = False
    for ln in hist:
        plain = memory._strip_timed(ln)
        if plain.startswith("灰泽满：") or plain.startswith("灰泽满（回"):
            miss = True
            break
    return core.build_message_list(
        gen_user, [], hist, state_stable=_state_block(state), log_hits=False), hist, miss


def _prepare_b(user: str, state: str, rows: list[dict]) -> list[dict]:
    from src.plugins.chatbot import core
    # 用占位历史块保留 A 的位置，再把它换成 native turns。
    messages = core.build_message_list(
        user, [], [_PLACEHOLDER], state_stable=_state_block(state), log_hits=False)
    return _replace_history_at_position(messages, _native_turns(rows), keep_old_lines=False)


async def _prepare_c(user: str, state: str, rows: list[dict]) -> list[dict]:
    """最近 2 轮 native，更早的转述 system；位置仍与 A 相同。"""
    from src.plugins.chatbot import core
    if len(rows) <= 2:
        return _prepare_b(user, state, rows)
    old, recent = rows[:-2], rows[-2:]
    hist_old, gen_user = await _paraphrase(old, user)
    messages = core.build_message_list(
        gen_user, [], hist_old + [_PLACEHOLDER],
        state_stable=_state_block(state), log_hits=False)
    return _replace_history_at_position(messages, _native_turns(recent), keep_old_lines=True)


async def _probe_a(history_lines: list[str], reply: str, state: str) -> str:
    from src.plugins.chatbot import core, self_speech
    base = list(history_lines) + [f"灰泽满：{reply}"]
    hist, probe_user = await self_speech.annotate_turn(base, PROBE)
    msgs = core.build_message_list(
        probe_user, [], hist, state_stable=_state_block(state), log_hits=False)
    return await core.generate_reply(msgs)


async def _probe_native(messages: list[dict], reply: str) -> str:
    from src.plugins.chatbot import core
    msgs = list(messages) + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": PROBE},
    ]
    return await core.generate_reply(msgs)


async def _run_one(uid: str, idx: int, row: dict, rows: list[dict], state: str,
                   samples: int, arms: str = "ABC") -> dict:
    from src.plugins.chatbot import core, reply_style
    hist_rows = rows[max(0, idx - 5):idx]
    prior_bot = [r["reply"] for r in hist_rows if not core.is_fallback_reply(r["reply"])]
    msgs_a = hist_a = None
    paraphrase_miss = False
    if "A" in arms:
        msgs_a, hist_a, paraphrase_miss = await _prepare_a(row["user"], state, hist_rows)
    msgs_b = _prepare_b(row["user"], state, hist_rows) if "B" in arms else None
    msgs_c = await _prepare_c(row["user"], state, hist_rows) if "C" in arms else None
    result = {"user": uid, "t": row["t"], "user_msg": row["user"],
              "actual_reply": row["reply"], "paraphrase_miss": paraphrase_miss,
              "history": [{"user": r["user"], "reply": r["reply"]} for r in hist_rows],
              "arms": {}}
    for arm, msgs in (("A", msgs_a), ("B", msgs_b), ("C", msgs_c)):
        if msgs is None:
            continue
        gens = []
        for _ in range(samples):
            reply = await core.generate_reply(msgs)
            fallback = core.is_fallback_reply(reply)
            probe = ""
            if not fallback:
                if arm == "A":
                    probe = await _probe_a(hist_a, reply, state)
                else:
                    probe = await _probe_native(msgs, reply)
            gens.append({
                "reply": reply,
                "probe": probe,
                "fallback": fallback,
                "echo": (not fallback) and reply_style.is_echo_reply(reply, prior_bot, window=8),
                "echo_overlap": round(max((_jaccard(reply, old) for old in prior_bot),
                                          default=0.0), 3) if not fallback else 0.0,
                "recent_echo": (not fallback) and bool(_RECENT_ECHO_RE.search(reply)),
                "probe_overlap": round(_jaccard(probe, reply, n=2), 3) if probe else 0.0,
            })
        result["arms"][arm] = gens
    return result


async def run(users: list[str], turns: int, samples: int,
              arms: str = "ABC", concurrency: int = 3) -> dict:
    import nonebot
    nonebot.init()
    from src.plugins.chatbot import memory_v2
    store = memory_v2.ShadowMemoryStore()
    all_rows = []
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(uid: str, idx: int, row: dict, rows: list[dict], state: str):
        async with sem:
            print(f"[ab] {uid} {row['t']:.0f} {row['user'][:20]}", flush=True)
            return await _run_one(uid, idx, row, rows, state, samples, arms)

    for uid in users:
        rows = _load_turns(uid)
        state = store.build_state_card(uid)
        tasks = [_one(uid, idx, row, rows, state) for idx, row in _pick(rows, turns)]
        all_rows.extend(await asyncio.gather(*tasks))
    return {"users": users, "turns": turns, "samples": samples,
            "arms": arms, "rows": all_rows}


def _summary(result: dict) -> dict:
    agg = {}
    # 只统计「A 的转述全部成功」的轮次：转述失败会退回原句，A 就变成 B 了。
    clean_rows = [r for r in result["rows"] if not r.get("paraphrase_miss")]
    present = [a for a in "ABC"
               if any(a in r.get("arms", {}) for r in result["rows"])] or ["A", "B"]
    for arm in present:
        all_gens = [g for row in clean_rows for g in row["arms"].get(arm, [])]
        gens = [g for g in all_gens if not g.get("fallback")]
        n = len(gens) or 1
        agg[arm] = {
            "n": len(gens),
            "fallback": round((len(all_gens) - len(gens)) / (len(all_gens) or 1), 3),
            "echo": round(sum(1 for g in gens if g["echo"]) / n, 3),
            "echo_overlap": round(sum(g["echo_overlap"] for g in gens) / n, 3),
            "recent_echo": round(sum(1 for g in gens if g["recent_echo"]) / n, 3),
            "probe_overlap": round(sum(g["probe_overlap"] for g in gens) / n, 3),
        }
    return agg


def write_report(result: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = _summary(result)
    lines = ["# 历史注入通道 A/B（初步）", "",
             f"- 用户：{', '.join(result['users'])}",
             f"- 轮次：{result['turns']} / 用户；每轮生成 {result['samples']} 次",
             f"- 转述失败（A 退回原句）的轮次：{sum(1 for r in result['rows'] if r.get('paraphrase_miss'))}"
             f" / {len(result['rows'])}，已从汇总里剔除",
             "", "## 汇总", "",
             "| 组 | n | 兜底率 | 复读命中 | 与前文最大重合 | 说「刚说了」 | 追问重合 |",
             "|---|---|---|---|---|---|---|"]
    for arm, s in summary.items():
        lines.append(f"| {arm} | {s['n']} | {s['fallback']:.0%} | {s['echo']:.0%} | "
                     f"{s['echo_overlap']:.2f} | {s['recent_echo']:.0%} | {s['probe_overlap']:.2f} |")
    lines += ["", "## 样本", ""]
    picked = []
    for uid in result["users"]:
        picked += [r for r in result["rows"] if r["user"] == uid][:5]
    for row in picked:
        lines += [f"### {row['user']} @ {row['t']:.0f}"
                  + ("（A 转述失败，已排除汇总）" if row.get("paraphrase_miss") else ""), "",
                  f"- 用户：{row['user_msg']}",
                  f"- 她当时真实回的：{row['actual_reply']}",
                  f"- 前文最后两轮：{json.dumps(row['history'][-2:], ensure_ascii=False)}"]
        for arm in ("A", "B", "C"):
            gens = row["arms"].get(arm) or []
            if not gens:
                continue
            g = gens[0]
            lines += [f"- {arm} 生成：{g['reply']}" + ("（兜底）" if g.get("fallback") else ""),
                      f"  - 追问回答：{g['probe']}",
                      f"  - 复读={g['echo']} 重合={g['echo_overlap']} 追问重合={g['probe_overlap']}"]
        lines.append("")
    path = OUT_DIR / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    # 另存一份“可读摘要”，只保留汇总 + 更多原文，方便直接看。
    summary_lines = ["# 历史注入通道 A/B · 可读摘要", "",
                     f"- 数据：{', '.join(result['users'])} 最近 {result['turns']} 轮/人，"
                     f"每轮生成 {result['samples']} 次",
                     f"- 有效轮次：{len([r for r in result['rows'] if not r.get('paraphrase_miss')])}"
                     f"（转述失败 {sum(1 for r in result['rows'] if r.get('paraphrase_miss'))} 轮已剔除）",
                     "", "## 汇总", "",
                     "| 组 | n | 兜底率 | 复读命中 | 与前文最大重合 | 说「刚说了」 | 追问重合 |",
                     "|---|---|---|---|---|---|---|"]
    for arm, s in summary.items():
        summary_lines.append(
            f"| {arm} | {s['n']} | {s['fallback']:.0%} | {s['echo']:.0%} | "
            f"{s['echo_overlap']:.2f} | {s['recent_echo']:.0%} | {s['probe_overlap']:.2f} |")
    summary_lines += ["", "## 对话原文（节选）", ""]
    for row in picked[:10]:
        summary_lines += [f"### {row['user']} · 用户：{row['user_msg'][:60]}", "",
                          f"- 她当时真实回复：{row['actual_reply'][:100]}"]
        for arm in ("A", "B", "C"):
            g = (row["arms"].get(arm) or [{}])[0]
            summary_lines.append(
                f"- {arm}：{str(g.get('reply', ''))[:110]}"
                + ("（兜底）" if g.get("fallback") else "")
                + f"\n  - 追问：{str(g.get('probe', ''))[:110]}（重合 {g.get('probe_overlap', 0)}）")
        summary_lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="900000001,900000001")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--arms", default="ABC", help="要跑哪几组，如 ABC / AB / B")
    ap.add_argument("--concurrency", type=int, default=3, help="并发轮数（默认 3）")
    ap.add_argument("--dry", action="store_true", help="只打印每组提示词条数，不调模型")
    args = ap.parse_args(argv)
    users = [u.strip() for u in args.users.split(",") if u.strip()]
    if args.dry:
        import nonebot
        nonebot.init()
        for uid in users:
            rows = _load_turns(uid)
            for idx, row in _pick(rows, min(2, args.turns)):
                print(uid, row["user"][:20], "history rows", len(rows[max(0, idx - 5):idx]))
        return 0
    arms = "".join(a for a in "ABC" if a in args.arms.upper()) or "ABC"
    result = asyncio.run(run(users, args.turns, args.samples,
                             arms=arms, concurrency=args.concurrency))
    path = write_report(result)
    print(json.dumps(_summary(result), ensure_ascii=False, indent=1))
    print(f"报告：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
