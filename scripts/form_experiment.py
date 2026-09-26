#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""形式对照实验：**同一个事实，写成不同形式，看哪种传得进去、哪种会被整句抄。**

## 为什么做这个

素材"长什么形式"决定了模型会怎么用它——这是本仓最核心、也最没被系统化的一条。
已有零散实测（都记在 `注入链路.md` / `待办清单.md`）：
  · 同一个事实，走 voice_samples（assistant turn）3/5 被采用，走 corpus（system）0/5
  · `'呃…'起头` 写进 system → 10/10 说"呃"，其中 8 条逐字相同
  · 给问答对 → 整句抄（7/8 逐字同一句）；给词表 → 只能织进自己的句子
这个脚本把上面三条**放进同一个受控实验**里重跑一遍，一次量出"形式 → 行为"的对应。

## 设计

- **合成事实**：实验用的事实**不在她的任何人格数据里**（否则基线会泄漏，测不准）。
  所以任何"用上了"都只能来自本次注入。
- **只变一个变量**：其余全部走真实链路 `build_message_list`（真人格 + 真检索），
  只额外挂一层，条件之间只有那一层不同。
- **两个观测量**：
  - **采用率**：回复有没有体现"她讨厌香菜"（关键词判定）
  - **逐字率**：回复有没有把注入里的特征短语（"闻到就想跑"）原样搬出来
- **不写任何记忆**：直接调 `build_message_list` + `generate_reply`，不经过 handle_chat。

用法：
    python scripts/form_experiment.py                 # 默认 N=8
    python scripts/form_experiment.py --n 12 --fact 猫
    python scripts/form_experiment.py --dry           # 只打印会注入什么，不调生成
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

from src.plugins.chatbot import core  # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules, build_global_persona_context  # noqa: E402
from src.plugins.chatbot.constants import SYSTEM_PROMPT_FILE  # noqa: E402

# ==================== 实验素材：合成事实（不在人格数据里） ====================
# 每条 = 一个问题 + 用来判"有没有用上/抄没抄"的词 + 若干种"形式"。
#
# ⚠️ **实验设计教训（第一版踩的坑）**：最初只用了「香菜」——结果**基线（不注入）就有 6/8
# "用上了"**：因为"挑食的人设 → 不吃香菜"是模型的强先验，判据（回复里出现"香菜"）
# 把先验当成了注入的效果。**用话题词当判据会被模型先验污染**。
# 所以：① 主判据改成**特征短语逐字率**（注入里埋一个编造的说法，基线不可能出现）；
#       ② 另加一条**模型猜不到**的事实（数字）来验证"采用率"这一列。
FACTS = {
    # 事实 A：有强先验（模型本来就会答"不吃"）——只看"抄没抄"，不看"用没用"
    "香菜": {
        "query": "你爱吃香菜吗",
        "want": ["香菜"],              # ⚠️ 这一列对 A 不可信（被先验污染），只看 copied
        "signature": "闻到就想跑",     # 注入里埋的特征短语：原样出现 = 整句抄
        "prior": True,
        "forms": {
            "① 不注入（基线）": None,
            "② system 第三人称陈述\n（corpus 的形式）":
                "【她经历过的相关背景】灰泽满被问到香菜，说自己完全接受不了，闻到就想跑。",
            "③ system 事实条目\n（preferences 的形式）":
                "【灰泽满的偏好】食物：不吃香菜，闻到就想跑（这是她稳定真实的偏好）",
            "④ system 词表\n（phrases 的形式）":
                "【她的固定说法】\n· 被问到香菜时：香菜不行、闻到就想跑",
            "⑤ system 指令式\n（behaviors 的形式）":
                "【当前情境下的行为指令】请严格按此模式回应：\n【被问到香菜时】"
                "触发情境：用户问灰泽满吃不吃香菜\n回应模式：明确说接受不了，闻到就想跑",
            "⑥ assistant 问答对\n（voice_samples 的形式）":
                ("__QA__", "你爱吃香菜吗", "香菜不行，闻到就想跑"),
        },
    },
    # 事实 B：**模型猜不到**（任意事实，无先验）——"采用率"这一列对它才可信
    "数字": {
        "query": "你最喜欢哪个数字啊",
        "want": ["7", "七"],
        "signature": "一直排第一",
        "prior": False,
        "forms": {
            "① 不注入（基线）": None,
            "② system 第三人称陈述\n（corpus 的形式）":
                "【她经历过的相关背景】被粉丝问到最喜欢的数字，灰泽满说自己最喜欢 7，在她这儿一直排第一。",
            "③ system 事实条目\n（preferences 的形式）":
                "【灰泽满的偏好】数字：最喜欢 7，一直排第一（这是她稳定真实的偏好）",
            "④ system 词表\n（phrases 的形式）":
                "【她的固定说法】\n· 被问到喜欢的数字时：7、一直排第一",
            "⑤ system 指令式\n（behaviors 的形式）":
                "【当前情境下的行为指令】请严格按此模式回应：\n【被问到喜欢的数字时】"
                "触发情境：用户问灰泽满最喜欢哪个数字\n回应模式：说 7，一直排第一",
            "⑥ assistant 问答对\n（voice_samples 的形式）":
                ("__QA__", "你最喜欢哪个数字啊", "7 啊，一直排第一"),
            # ⑦ 与 ④ 的唯一区别：词表**不构成一句完整的话**，只是可织进去的措辞碎片。
            #    项目文档里写着"给词表 → 只能织进自己的句子，产出新句"，
            #    但 ④ 实测 8/8 逐字抄 —— 想搞清是"形式"的功劳还是"自足句"的功劳。
            "⑦ system 碎片词表\n（phrases 的另一种可能形式）":
                "【她的固定说法】\n· 被问到喜欢的数字时：说法是「排第一」「头号」「铁打的」这几种",
            # ⑧ **包装语对照**：内容与 ④ 完全相同（同一句自足的话），只换外层的包裹语。
            #    ④ 的包装是 phrases 式（"用这些原话组织"= 鼓励照说）；
            #    ⑧ 换成 corpus 式（"只当背景记忆…不整段复述"= 明确叫它别抄）。
            #    问题：**包装语到底管不管用，还是内容形式说了算？**
            "⑧ 同内容·换包装语\n（corpus 式的包装 + ④ 的内容）":
                "【她经历过的相关背景】以下是她过去直播里经历过的事（背景记忆，都是曾经发生的）。"
                "只当'她记得的经历'自然带出，不整段复述、不模仿里面的叙述口吻。\n- 被问到喜欢的数字，她答：7，一直排第一。",
            # ---- 2026-09-27 追加：直接回答"voice_samples 的 user 字段该怎么写" ----
            # ⑨ 与 ⑥ 的唯一区别：**user 字段从"具体问句"换成"情境描述"**。
            #    动机：真实的 voice_samples 里 43/86 条 user 是「粉丝问：<具体问题>」，
            #    与真实提问撞车时她会答"刚不是说过了嘛"（见 ⑥ 的副作用）。改成情境描述
            #    应该能治撞车，但**会不会同时削弱问答对的效力**？——这就是本次要量的。
            "⑨ 问答对·user 写情境描述\n（voice_samples 的另一写法）":
                ("__QA__", "粉丝在问灰泽满最喜欢哪个数字", "7 啊，一直排第一"),
            # ⑩ 只有 assistant、没有配对的 user —— 她"自说自话"式示范
            "⑩ 只有 assistant（无配对 user）":
                ("__ASSIST__", "7，一直排第一"),
            # ⑪ 纯情境描述：无指令语气、无问答配对（区别于 ⑤ 的"请严格按此模式回应"）
            "⑪ system 纯情境描述\n（无指令语气）":
                "【关于灰泽满】被问到最喜欢的数字时，她会说 7，一直排第一。",
        },
    },
}


def _messages_for(fact: dict, form_val) -> list:
    """真实链路搭好 messages，再按条件挂一层。只变这一层，其余完全一致。"""
    traits, styles, _ = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)
    msgs = core.build_message_list(fact["query"], global_persona, [], "", [])
    last_user = msgs[-1]                               # 真正的"用户这句话"

    if form_val is None:                               # ① 基线：什么都不加
        return msgs
    if isinstance(form_val, tuple):
        tag, *rest = form_val
        if tag == "__ASSIST__":                        # ⑩ 只有 assistant，无配对 user
            return msgs[:-1] + [
                {"role": "assistant", "content": rest[0]},
                {"role": "system", "content": "【回复节奏】日常闲聊一句话说完就停，别硬凑第二、第三句长段。"},
                last_user,
            ]
        _, user_part, reply_part = form_val            # ⑥⑨ assistant 问答对
        return msgs[:-1] + [
            {"role": "user", "content": user_part},
            {"role": "assistant", "content": reply_part},
            {"role": "system", "content": "【回复节奏】日常闲聊一句话说完就停，别硬凑第二、第三句长段。"},
            last_user,
        ]
    # 其余：在"这句话"之前插一条 system（各层真实注入位置就在这一带）
    return msgs[:-1] + [{"role": "system", "content": form_val}, last_user]


async def run_fact(name: str, fact: dict, n: int, dry: bool):
    print(f"\n{'='*72}\n事实：{name} ｜ 提问：「{fact['query']}」 ｜ 每种形式各生成 {n} 次\n")
    results = {}
    for label, val in fact["forms"].items():
        lab = label.replace("\n", " ")
        if dry:
            print(f"--- {lab}\n    {(val if not isinstance(val, tuple) else val[1] + ' / ' + val[2])}\n")
            continue
        sem = asyncio.Semaphore(5)          # 并发跑，别一条条等
        async def one():
            async with sem:
                return await core.generate_reply(_messages_for(fact, val))
        replies = list(await asyncio.gather(*[one() for _ in range(n)]))
        used = sum(1 for r in replies if any(w in r for w in fact["want"]))
        copied = sum(1 for r in replies if fact["signature"] in r)
        # 「她以为刚说过了」——问答对/assistant 形式的副作用（user 字段与真实提问撞车时）
        ECHO = re.compile(r"刚(不是)?说(过|了)|问过了|问两遍|复读|都说了")
        echoed = sum(1 for r in replies if ECHO.search(r))
        results[lab] = {"used": used, "copied": copied, "echoed": echoed, "replies": replies}
        print(f"--- {lab}")
        print(f"    采用 {used}/{n}   逐字抄 {copied}/{n}   以为'刚说过' {echoed}/{n}")
        for r in replies[:4]:
            print(f"      · {r[:60]}")
        if fact.get("prior"):
            print("    ⚠️ 这条事实模型有强先验，「采用」列不可信，只看「逐字」列")
    return results


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="每种形式生成几次")
    ap.add_argument("--fact", default="all", help="用哪条合成事实（all = 全部）")
    ap.add_argument("--dry", action="store_true", help="只打印注入内容，不调生成")
    ap.add_argument("--out", default="outputs/_form_experiment.json")
    args = ap.parse_args()

    names = list(FACTS) if args.fact == "all" else [args.fact]
    allres = {}
    for nm in names:
        allres[nm] = await run_fact(nm, FACTS[nm], args.n, args.dry)
    if args.dry:
        return

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": args.n, "results": allres}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n✅ 原始回复已存 {out}")
    for nm in names:
        res = allres[nm]
        print(f"\n=== {nm} ===  （提问：「{FACTS[nm]['query']}」）")
        print(f"{'形式':<34}{'采用率':>9}{'逐字率':>9}{'以为刚说过':>12}")
        print("-" * 66)
        for k, v in res.items():
            print(f"{k:<34}{v['used']}/{args.n:<7}{v['copied']}/{args.n:<7}{v.get('echoed', 0)}/{args.n}")
        if FACTS[nm].get("prior"):
            print("⚠️ 这条事实模型有强先验——「采用率」列不可信，只看「逐字率」")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
