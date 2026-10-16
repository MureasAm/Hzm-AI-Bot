#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""骨架 A/B：**同一套链路，只换 `system_prompt.txt`**，把两版的输出摆在一起。

## 为什么有它（2026-10-15）

`outputs/_sp_new.txt` 那版重写草稿 2026-10-12 就跑过一次 A/B，但 ① 那次是 **deepseek** 打的，
现在线上换成 gemini 了；② 草稿**一直没上线**（`docs/历史/待办清单.md` §六"骨架草稿待拍"）。
这一轮把它正式换上去，所以要在**现在的模型上**重新量一次。

## 设计（只变一个变量）

对每条 query：
  ① 走**真实检索链路**取一次 `fused_items`（只取一次 → 两版看到同一份素材）
  ② 用 A 版骨架 `build_message_list` → 生成
  ③ 用 B 版骨架 `build_message_list` → 生成
两条 messages **只有第 1 段（骨架）不同**。

判据（**确定性、一眼可验**，不用模型当裁判——见 `make_ab_review.py` 那条教训）：
  · 平均字数 / 超 40 字条数（她要"短"）
  · 括号率 / 省略号率（配额：各 ≤ 1/5）
  · 以"灰泽满"起手的比例（自反馈固化指标）
  · 回归用例的禁词命中数（客服腔 / 咨询腔 / 编细节）

用法：
    python scripts/run_tool.py skeleton-ab                    # 默认 10 条真实消息 × 2 版
    python scripts/run_tool.py skeleton-ab --n 20 --runs 2
    python scripts/run_tool.py skeleton-ab --b outputs/_sp_v3.txt
"""
import argparse
import asyncio
import json
import random
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")

import src.plugins.chatbot.core as core              # noqa: E402
import src.plugins.chatbot.memory as mem             # noqa: E402
from src.plugins.chatbot.config import _get_clients  # noqa: E402

CASES_FILE = ROOT / "scripts" / "regression_cases.json"
A_FILE = ROOT / "persona" / "core" / "system_prompt.txt"
DEFAULT_B = ROOT / "outputs" / "_sp_v3.txt"


def _real_messages(n: int, seed: int) -> list:
    """从真实私聊记录里抽 n 条用户消息（只取用户那半句，不含她那半」。"""
    import glob
    rows = []
    for f in glob.glob(str(ROOT / "data" / "chat_log" / "*.jsonl")):
        for ln in open(f, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("kind") != "private":
                continue
            u = (d.get("user") or "").strip()
            # 只要"一句话"的：多行/带昵称前缀/引用/太长的一律跳过
            if not u or "\n" in u or "：" in u or u.startswith("[引用") or len(u) > 30:
                continue
            rows.append(u)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


async def _one(query: str, prompt_a: str, prompt_b: str, client, zhipu) -> dict:
    """同一条 query：检索一次，两版骨架各生成一次。"""
    ctx = await core.gather_retrieval(
        query_text=query, retrieval_query=query, history_text="",
        deepseek_client=client, zhipu_client=zhipu, is_user_msg=True)
    fused = ctx["fused_items"]
    out = {}
    for tag, prompt in (("A", prompt_a), ("B", prompt_b)):
        core.SYSTEM_PROMPT = prompt
        msgs = core.build_message_list(query, fused, [])
        out[tag] = await core.generate_reply(msgs)
    return out


def _stats(replies: list) -> dict:
    n = len(replies)
    if not n:
        return {}
    return {
        "n": n,
        "均长": round(sum(len(r) for r in replies) / n, 1),
        "超40字": sum(1 for r in replies if len(r) > 40),
        "括号率": round(sum(1 for r in replies if "（" in r or "(" in r) / n, 3),
        "省略号率": round(sum(1 for r in replies if "……" in r or "..." in r) / n, 3),
        "名字起手率": round(sum(1 for r in replies if re.match(r"^\s*[（(【]?\s*(灰泽满|hzm)", r)) / n, 3),
    }


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="抽多少条真实消息")
    ap.add_argument("--runs", type=int, default=1, help="每条跑几轮")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--b", default=str(DEFAULT_B), help="B 版骨架文件")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import nonebot.log as _log
    _log.logger.remove()

    prompt_a = A_FILE.read_text(encoding="utf-8")
    prompt_b = Path(args.b).read_text(encoding="utf-8")
    print(f"A（线上）＝{len(prompt_a)} 字   B（草稿）＝{len(prompt_b)} 字\n")

    client, zhipu = _get_clients()
    core.update_memory_v2_shadow_task = lambda *a, **k: asyncio.sleep(0)
    mem.use_storage(Path(tempfile.mkdtemp(prefix="skelab_")))

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    queries = _real_messages(args.n, args.seed) + [c["user"] for c in cases]

    rows, acc = [], {"A": [], "B": []}
    for q in queries:
        for _ in range(args.runs):
            try:
                r = await _one(q, prompt_a, prompt_b, client, zhipu)
            except Exception as e:
                print(f"  ⚠️ {q!r} 失败：{e}")
                continue
            rows.append({"query": q, **r})
            acc["A"].append(r["A"])
            acc["B"].append(r["B"])
            print(f"U: {q}")
            print(f"  甲: {r['A']!r}")
            print(f"  乙: {r['B']!r}")

    print("\n" + "=" * 60)
    print(f"{'指标':<12}{'A（线上）':>14}{'B（草稿）':>14}")
    for k in _stats(acc["A"]):
        print(f"{k:<12}{str(_stats(acc['A'])[k]):>14}{str(_stats(acc['B']).get(k)):>14}")

    # 禁词判据：两版各犯几次规
    print("\n--- 回归用例禁词命中（越少越好）---")
    for cid in ("no_assistant_tone", "no_therapist_tone_when_sad"):
        forbid = next((c.get("forbid") for c in cases if c["id"] == cid), [])
        for tag in ("A", "B"):
            rs = [r[tag] for r in rows if r["query"] in {c["user"] for c in cases if c["id"] == cid}]
            hit = sum(1 for r in rs for w in forbid if w in r)
            print(f"  {cid:<28} {tag}: {hit} 次命中 / {len(rs)} 条")

    out = ROOT / "outputs" / "eval" / "skeleton_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"a_chars": len(prompt_a), "b_chars": len(prompt_b),
                               "b_file": args.b, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n✅ 明细已保存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))