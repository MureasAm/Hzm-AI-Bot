#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""换模型的 A/B：拿**真实对话**跑同一份提示词，把两个模型的输出摆在一起看。

## 为什么有它（2026-10-14）

用户拿了一个中转站的 key（只有 `gpt-5.6-luna` 一个模型），要判断"值不值得换掉 deepseek"。
价格能算（见 `docs/下一步.md` A2 节），但**"像不像灰泽满"只能看**——所以拿真实历史消息
组装同一份提示词，两个模型各生成一遍，再跟**她当时真实发出去的那句**并排摆。

## 两边的端点从哪来

A 侧＝**线上正在用的那套**（`OPENAI_API_BASE/KEY/MODEL`）；B 侧＝`FALLBACK_*`
（那就是线上**自动兜底**用的同一个端点——它是 deepseek 官方）。

## 用法

    .venv/Scripts/python.exe scripts/run_tool.py model-ab                    # 12 条真实私聊
    .venv/Scripts/python.exe scripts/run_tool.py model-ab --n 20
    .venv/Scripts/python.exe scripts/run_tool.py model-ab --judge            # 判据 A/B（行为分类）
    .venv/Scripts/python.exe scripts/run_tool.py model-ab --only 主回复       # 只跑某类 tag

⚠️ 会真调两个模型（每条 2 次）。**提示词是现组装的**：检索/行为/语料块用的是**现在**的
语料按这条消息重新检索的（历史会话状态没法重建），记忆/历史传空——**两个模型看到的是
同一份，所以对比公平**，但它不等于当时线上那条完整提示词。
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import _common  # noqa: E402


def _load_real_turns(n: int, route: str):
    """从 chat_log 里挑真实轮次：(用户消息, 她真实回的, 会话)。"""
    rows = []
    for line in _common.chatlog_lines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sess = str(r.get("session") or "")
        if not sess.isdigit():                      # 滤掉 harness 会话（u1/u9 那种）
            continue
        kind = "群" if r.get("kind") == "group" else "私"
        if route and kind != route:
            continue
        u = (r.get("user") or "").strip()
        a = (r.get("reply") or "").strip()
        if len(u) < 2 or len(a) < 4:
            continue
        rows.append((u, a, sess))
    random.seed(20261014)                            # 取样固定，便于复跑对比
    random.shuffle(rows)
    return rows[:n]


async def run_reply(n: int, route: str, model_b: str) -> None:
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter
    nonebot.init()
    nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
    nonebot.load_plugins("src/plugins")

    from openai import AsyncOpenAI
    from src.plugins.chatbot import core
    from src.plugins.chatbot.config import _get_clients, _get_model_name
    from src.plugins.chatbot.constants import THINKING_DISABLED

    base = _common.get_api_key("FALLBACK_API_BASE")
    key = _common.get_api_key("FALLBACK_API_KEY")
    if not (base and key):
        print("❌ 没配对照端点（.env.prod 里的 FALLBACK_API_BASE / FALLBACK_API_KEY）")
        return
    other = AsyncOpenAI(base_url=base, api_key=key)
    ds, zhipu = _get_clients()
    model_a = _get_model_name()

    turns = _load_real_turns(n, route)
    print(f"A = {model_a}（线上） ｜ B = {model_b}（中转站） ｜ 真实历史消息 {len(turns)} 条\n")

    fail = {"A": 0, "B": 0}
    len_a, len_b, len_real = [], [], []
    for i, (user_msg, real, sess) in enumerate(turns, 1):
        # 用**现在**的语料按这条消息重新检索，两个模型共用同一份提示词
        ctx = await core.gather_retrieval(user_msg, user_msg, "", ds, zhipu)
        msgs = core.build_message_list(
            user_msg, ctx["fused_items"], "", [],
            preference_items=ctx["preference_items"], core_stories=ctx["core_stories"])
        outs = {}
        for name, client, model in (("A", ds, model_a), ("B", other, model_b)):
            t0 = time.time()
            try:
                r = await client.chat.completions.create(
                    model=model, messages=msgs, temperature=0.85,
                    frequency_penalty=0.3, max_tokens=150, **THINKING_DISABLED)
                txt = (r.choices[0].message.content or "").strip()
                if not txt:
                    fail[name] += 1
                    txt = f"（空，finish_reason={r.choices[0].finish_reason}）"
                outs[name] = f"{txt}   〔{time.time()-t0:.1f}s〕"
            except Exception as e:
                fail[name] += 1
                outs[name] = f"❌ {type(e).__name__}: {str(e)[:70]} 〔{time.time()-t0:.1f}s〕"
        len_real.append(len(real))
        len_a.append(len(outs["A"].split("   〔")[0]))
        len_b.append(len(outs["B"].split("   〔")[0]))
        print(f"—{i}— 用户：{user_msg[:60]}")
        print(f"   她当时真实说的：{real[:90]}")
        print(f"   A[{model_a}]：{outs['A'][:110]}")
        print(f"   B[{model_b}]：{outs['B'][:110]}\n")
    avg = lambda xs: sum(xs) / max(len(xs), 1)
    print(f"失败/空：A {fail['A']}/{len(turns)} ｜ B {fail['B']}/{len(turns)}")
    print(f"平均长度：真实 {avg(len_real):.0f} 字 ｜ A {avg(len_a):.0f} 字 ｜ B {avg(len_b):.0f} 字")


async def run_judge(n: int, model_b: str) -> None:
    """判据 A/B：拿真实用户消息跑行为分类，看两个模型选出来的类别一不一致。"""
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter
    nonebot.init()
    nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
    nonebot.load_plugins("src/plugins")

    from openai import AsyncOpenAI
    from src.plugins.chatbot import routing
    from src.plugins.chatbot.persona import load_persona_rules
    from src.plugins.chatbot.config import _get_clients
    from src.plugins.chatbot.constants import THINKING_DISABLED

    other = AsyncOpenAI(base_url=_common.get_api_key("FALLBACK_API_BASE"),
                        api_key=_common.get_api_key("FALLBACK_API_KEY"))
    ds, _ = _get_clients()
    beh = load_persona_rules()
    turns = _load_real_turns(n, "")
    same = 0
    for i, (user_msg, real, _s) in enumerate(turns, 1):
        # ⚠️ 判据函数内部自己取模型名（`_get_model_name()`），所以"换模型"要在模块层打补丁——
        # 这也正是**线上要换就必须加一个"判据模型"配置**的原因。
        routing._get_model_name = lambda: "deepseek-flash"
        a = await routing.classify_l3(ds, user_msg, "", beh)
        routing._get_model_name = lambda m=model_b: m
        b = await routing.classify_l3(other, user_msg, "", beh)
        ok = a["behavior"] == b["behavior"]
        same += ok
        print(f"  {'✓' if ok else '✗'} {user_msg[:44]:<46} A={a['behavior'] or '（无）':<12} B={b['behavior'] or '（无）'}")
    print(f"\n  一致率 {same}/{len(turns)} = {same/max(len(turns),1):.0%}")


async def run_replay(n_sessions: int, turns: int, model_b: str) -> None:
    """**持续上下文回放**：拿真实会话按顺序一轮一轮跑，历史喂"模型自己上一轮说的话"。

    为什么必须有它（用户 2026-10-14 指出）：前面那种"每条独立跑一遍"的 A/B 是**冷启动**，
    模型看不到前文、也没有"自己刚说过什么"这个变量——而**生产里最要紧的恰恰是这个**
    （口癖自强化、越聊越像自己、长度漂移都是**跨轮**现象）。
    所以这里两种跑法并排：

      cold＝只发这一条（= 之前的对比）；warm＝把前几轮的"用户：…／灰泽满：<它自己上轮写的>"喂回去

    回放时历史用**它自己生成的**回复（自回归），跟生产一致：她说过的话会被记进短期记忆。
    """
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter
    nonebot.init()
    nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
    nonebot.load_plugins("src/plugins")

    from openai import AsyncOpenAI
    from src.plugins.chatbot import core
    from src.plugins.chatbot.config import _get_clients, _get_model_name
    from src.plugins.chatbot.constants import THINKING_DISABLED
    from src.plugins.chatbot.reply_style import clean_reply

    other = AsyncOpenAI(base_url=_common.get_api_key("FALLBACK_API_BASE"),
                        api_key=_common.get_api_key("FALLBACK_API_KEY"))
    ds, zhipu = _get_clients()
    model_a = _get_model_name()

    # 挑一段连续的真实会话（同一会话、时间相邻、条数够）
    bysess = {}
    for line in _common.chatlog_lines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not str(r.get("session") or "").isdigit() or r.get("kind") == "group":
            continue
        if len((r.get("user") or "").strip()) < 2 or len((r.get("reply") or "").strip()) < 4:
            continue
        bysess.setdefault(str(r["session"]), []).append(r)
    segs = []
    for sess, rs in sorted(bysess.items(), key=lambda kv: -len(kv[1])):
        rs.sort(key=lambda r: r.get("t") or 0)
        if len(rs) >= turns + 2:
            segs.append(rs[:turns])
        if len(segs) >= n_sessions:
            break

    async def gen(client, model, msgs, **kw):
        r = await client.chat.completions.create(model=model, messages=msgs,
                                                 temperature=0.85, max_tokens=150, **kw)
        return clean_reply((r.choices[0].message.content or "").strip())

    def head3(s):
        return (s or "").strip()[:3]

    for si, seg in enumerate(segs, 1):
        print(f"\n{'='*72}\n会话 {si}（{len(seg)} 轮，会话 id={seg[0]['session'][:6]}…）\n{'='*72}")
        # 三条臂：cold=无历史 ｜ user=只喂用户侧（= **今天线上**的形态）｜ self=用户+她自己说过的话（= 昨天以前）
        hist = {"cold": [], "user": [], "self": []}
        out = {"cold": [], "user": [], "self": []}
        for i, rec in enumerate(seg, 1):
            u = (rec.get("user") or "").strip()
            real = (rec.get("reply") or "").strip()
            ctx = await core.gather_retrieval(u, u, "", ds, zhipu)
            print(f"〔{i}〕用户：{u}")
            print(f"      她真实：{real[:64]}")
            for mode in ("cold", "user", "self"):
                msgs = core.build_message_list(u, ctx["fused_items"], "", list(hist[mode]))
                try:
                    a = await gen(ds, model_a, msgs, frequency_penalty=0.3, **THINKING_DISABLED)
                except Exception as e:
                    a = f"❌{type(e).__name__}"
                try:
                    b = await gen(other, model_b, msgs)
                except Exception as e:
                    b = f"❌{type(e).__name__}"
                out[mode].append((a, b))
                hist["user"] += [f"用户：{u}"]
                hist["self"] += [f"用户：{u}", f"灰泽满：{b}"]
                # cold 不累积
                print(f"      {mode:<4} ds ：{a[:60]}")
                print(f"      {mode:<4} gem：{b[:60]}")
        # 统计
        print(f"\n  ——— 会话 {si} 统计 ———")
        for mode in ("cold", "user", "self"):
            for idx, name in ((0, model_a), (1, model_b)):
                xs = [x[idx] for x in out[mode] if not x[idx].startswith("❌")]
                if not xs:
                    continue
                rep = sum(1 for a, b in zip(xs, xs[1:]) if head3(a) == head3(b))
                br = sum(len([c for c in x if c == "（"]) for x in xs)
                print(f"    {mode:<4} {name:<30} 均长 {sum(map(len,xs))/len(xs):>5.1f} 字"
                      f" ｜ 括号 {br/len(xs):.1f}/条 ｜ 相邻同开头 {rep}/{max(len(xs)-1,1)}")


def main(argv=None) -> int:
    import os
    _common.ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="model_ab", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12, help="取几条真实轮次（默认 12）")
    ap.add_argument("--route", default="私", choices=["私", "群", ""], help="私聊/群聊（默认私）")
    ap.add_argument("--model-b", default=None, help="B 侧模型（缺省读 .env.prod 的 FALLBACK_MODEL）")
    ap.add_argument("--judge", action="store_true", help="跑判据 A/B（行为分类一致率）")
    ap.add_argument("--replay", action="store_true", help="持续上下文回放（cold vs warm 并排）")
    ap.add_argument("--sessions", type=int, default=2, help="回放几段真实会话（默认 2）")
    ap.add_argument("--turns", type=int, default=8, help="每段回放几轮（默认 8）")
    args = ap.parse_args(argv)
    model_b = args.model_b or _common.get_api_key("FALLBACK_MODEL") or "gpt-5.6-luna"
    if args.judge:
        return asyncio.run(run_judge(args.n, model_b)) or 0
    if args.replay:
        return asyncio.run(run_replay(args.sessions, args.turns, model_b)) or 0
    return asyncio.run(run_reply(args.n, args.route, model_b)) or 0


if __name__ == "__main__":
    sys.exit(main())
