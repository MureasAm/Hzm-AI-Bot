#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B：**行为/措辞进不进 RRF**，同一批消息两版并排给你看。

## 背景（用户 2026-09-30 提出）

> "behaviors 有没有必要进入检索呢？被 LLM 判断中难道不是就可以直接读取文件内容，
> 不需要检索了吗？"

对。`select_behavior_item` / `select_phrase_groups` 就是**按名字查表**（没有相似度、
分数写死 1.0），它们早就不检索了。但结果仍被丢进 RRF 排行榜，**占 top-6 名额和预算**。

## 这一版做了什么

```
关（现状）：fuse_and_truncate(corpus, sample, behavior, phrase)   ← 四路一起排名、一起抢名额
开（新版）：fuse_and_truncate(corpus, sample) + behavior + phrase  ← 查表结果截断后追加
```

开关是运行时读环境变量 `DETACH_BEHAVIOR`，所以可以在**同一个进程里**跑两版、比同一批消息。

## 怎么给你看

- **只展示"行为或措辞真的命中了"的轮次**——没命中的话两版完全一样，看它没意义
- **左右打乱**（不然你会被"哪个是新版"带偏），key 存在 json 里
- 顺带报**注入量涨了多少字**（这是这一版要付的代价）

用法：
    python scripts/detach_behavior_ab.py --n 12
    python scripts/detach_behavior_ab.py --n 12 --only-diff   # 只留两版回复不一样的
"""
import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nonebot  # noqa: E402
nonebot.init()
from nonebot.adapters.onebot.v11 import Adapter as OB11  # noqa: E402
nonebot.get_driver().register_adapter(OB11)
nonebot.load_plugins("src/plugins")

import _common                              # noqa: E402
from src.plugins.chatbot import core         # noqa: E402
from src.plugins.chatbot import memory as mem  # noqa: E402

OUT = _common.OUTPUTS_DIR / "eval" / "detach_behavior_ab.json"

_cap, _orig = {}, None


async def _capture(messages):
    _cap["msgs"] = messages
    return "（占位）"


def _load_real(n: int) -> list:
    rows = []
    for line in _common.chatlog_lines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        u = (r.get("user") or "").strip()
        if (u and r.get("kind") == "private" and len(u) <= 40 and "\n" not in u
                and str(r.get("session", "")).isdigit()):
            rows.append(u)
    if not rows:
        return []
    step = max(1, len(rows) // n)
    out, seen = [], set()
    for u in rows[::step]:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= n:
            break
    return out


def _size(msgs) -> int:
    return sum(len(m.get("content") or "") for m in msgs)


async def _run_one(msg: str, detach: bool):
    """跑一次链路（拿 messages）+ 生成一条回复。返回 (回复, 注入总字数, 有哪些段)。"""
    os.environ["DETACH_BEHAVIOR"] = "1" if detach else "0"
    _cap.clear()
    try:
        await core.handle_chat(f"ab_{'d' if detach else 'k'}", msg)
    except Exception as e:
        return None, 0, [], ""
    msgs = _cap.get("msgs")
    if not msgs:
        return None, 0, [], ""
    secs = [m.get("content", "")[:14] for m in msgs if m.get("role") == "system"]
    # 注入内容的**规范串**（用来判"这次改动到底动了输入没有"——回复差异可能是生成的随机性）
    fingerprint = "\n".join(m.get("content", "") for m in msgs)
    reply = ""
    for _ in range(3):
        reply = await _orig([dict(m) for m in msgs])
        if reply and reply.strip() and reply.strip() != "（占位）":
            break
    return reply, _size(msgs), secs, fingerprint


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--only-diff", action="store_true", help="只留两版回复不一样的")
    ap.add_argument("--from-scenarios", action="store_true",
                    help="用靶子情境（被夸/被示好/让立flag/被捧高）当消息——真实消息里行为只命中 ~8%，"
                         "那样 12 条只能看到 1 轮。判据是「哪版更像她」，与场景怎么来无关。")
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()
    _common.ensure_utf8_stdout()

    global _orig
    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    core.update_memory_v2_shadow_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="detachab_")
    mem.use_storage(Path(tmp))
    _orig = core.generate_reply
    core.generate_reply = _capture

    if a.from_scenarios:
        doc = json.loads((Path(__file__).resolve().parent
                          / "scenario_prompts.json").read_text(encoding="utf-8"))
        pool = []
        for k, v in doc.items():
            if not k.startswith("_"):
                pool += v["prompts"]
        msgs = pool[:a.n]
    else:
        msgs = _load_real(a.n)
    if not msgs:
        print("❌ chat_log 里取不到真实消息（CHATLOG=0？）")
        return 1
    print(f"🔬 行为/措辞「进 RRF」vs「不进 RRF」：{len(msgs)} 条真实消息\n")

    rows, skipped, fired = [], 0, 0
    for i, msg in enumerate(msgs, 1):
        r_old, s_old, sec_old, fp_old = await _run_one(msg, detach=False)
        r_new, s_new, sec_new, fp_new = await _run_one(msg, detach=True)
        if not r_old or not r_new:
            skipped += 1
            continue
        hit = any(("行为指令" in s or "固定说法" in s) for s in sec_old)
        if hit:
            fired += 1
        input_diff = fp_old != fp_new
        rows.append({"msg": msg, "old": r_old, "new": r_new,
                     "old_chars": s_old, "new_chars": s_new, "behavior_fired": hit,
                     "input_diff": input_diff,
                     "diff": r_old.strip() != r_new.strip()})
        print(f"  [{i}/{len(msgs)}] {msg[:18]:<20}"
              f"{'行为/措辞命中' if hit else '未命中':<10}"
              f"{'注入**变了**' if input_diff else '注入没变（两版输入一模一样）':<26}"
              f"注入 {s_old}→{s_new} 字")

    # ⚠️ 只展示**注入内容真的变了**的轮次——否则两版输入一模一样，
    #    回复差异只是生成的随机性，看了会误判。
    show = [r for r in rows if r["input_diff"]]
    # 左右打乱
    rnd = random.Random(a.seed)
    key = []
    for k, r in enumerate(show, 1):
        left, right = ("new", "old") if rnd.random() < 0.5 else ("old", "new")
        key.append({"i": k, "left": left, "right": right, "msg": r["msg"]})

    print(f"\n{'=' * 78}")
    print(f"【两版并排】行为/措辞命中的 {fired} 轮里，展示 {len(show)} 轮"
          f"（跳过 {skipped} 条跑失败）")
    print(f"⚠️ 左右是打乱的，你看不出哪个是新版。**凭感觉判哪版更像她**。\n")
    for k, r in enumerate(show, 1):
        L, R = key[k - 1]["left"], key[k - 1]["right"]
        print(f"--- {k}. 用户说：{r['msg']}")
        print(f"    甲：{r[L]}")
        print(f"    乙：{r[R]}")
        print()
    inc = [r["new_chars"] - r["old_chars"] for r in rows if r["behavior_fired"]]
    if inc:
        print(f"【代价】注入总字数：命中轮平均 +{sum(inc) / len(inc):.0f} 字"
              f"（{min(inc):+d} ~ {max(inc):+d}）")
    diff_n = sum(1 for r in rows if r["diff"])
    n_in = sum(1 for r in rows if r["input_diff"])
    print(f"【关键数】{n_in}/{len(rows)} 轮的**注入内容真的变了**"
          f"（其余轮两版输入一模一样 → 回复差异只是生成的随机性，不算数）")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "key": key}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n✅ 明细（含左右 key，别在判之前看）：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
