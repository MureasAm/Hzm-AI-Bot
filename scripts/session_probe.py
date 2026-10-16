#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""会话探针（第 8/17/19 块共用的那次 LLM 调用）的**准确率探针**。

## 为什么要连着几轮测

`probe_session` 是**有状态**的：第 N 轮的判断依赖"上一轮的话题"和"最近对话"。
单独喂一句测不出"转话题"判得对不对——那本来就是个**相对**判断。
所以这里模拟整段对话，一轮一轮喂，逐轮看它判成什么。

它一次产 5 样，其中 **3 样分别注到 3 个块**：
    topic / new_event  → 第 8 块（会话记忆）
    expanded_query     → 第 17 块（这条消息的语境）
    mood               → 第 19 块（用户此刻的情绪）
（另外两样 topic_changed / 是否记事件，是内部控制。）

用法：
    python scripts/session_probe.py                 # 全部场景
    python scripts/session_probe.py --only topic    # 只跑名字里含 topic 的
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")

from src.plugins.chatbot import session_memory as SM  # noqa: E402
from src.plugins.chatbot.config import _get_clients  # noqa: E402


# 每个场景：一串轮次。每轮可以写期望，脚本自己判过没过。
#   changed: 期望的 topic_changed（True/False/None=不判）
#   mood:    "yes" 该有情绪 / "no" 不该有 / None 不判
#   exp:     "yes" 该给出补全句 / "no" 不该给 / None 不判
#   exp_has: 补全句里应该出现的词（给了就顺便查）
#   topic_has: 话题里应该出现的词
SCENARIOS = [
    {
        "name": "topic_同一主题延续 vs 转话题",
        "turns": [
            {"u": "你最近有在用香水吗", "a": "问这个干嘛，灰泽满身上香不香关你什么事。",
             "changed": True, "topic_has": "香水", "mood": "no", "exp": "no"},
            {"u": "那你喜欢哪个牌子的", "a": "不告诉你，你自己猜。",
             "changed": False, "topic_has": "香水", "mood": "no", "exp": "no"},
            {"u": "好吧，换个话题，你玩不玩王者荣耀", "a": "玩过，怎么了。",
             "changed": True, "topic_has": "王者", "mood": "no", "exp": "no"},
        ],
    },
    {
        "name": "exp_短消息与指代性消息的补全",
        "turns": [
            {"u": "我写的小说叫乌色月", "a": "哦？你自己写的？",
             "changed": True, "exp": "no"},
            {"u": "能念给我听听吗", "a": "……你要听啊。",
             "changed": False, "exp": "yes", "exp_has": "乌色月"},
            {"u": "那个呢", "a": "哪个。",
             "changed": False, "exp": "yes"},
            {"u": "咋这样", "a": "咋样了。",
             "changed": False, "exp": "yes"},
        ],
    },
    {
        "name": "emoji_纯表情按标准含义识别",
        "turns": [
            {"u": "今天又被老板骂了", "a": "……又被骂了？",
             "changed": True, "mood": "yes"},
            {"u": "😅", "a": "笑什么。",
             "changed": False, "exp": "yes", "exp_has_any": ["无语", "无奈", "尴尬"],
             "note": "prompt 里 😅 允许 无语/无奈/尴尬 三个说法，不能只认一个"},
            {"u": "[表情：可怜]", "a": "装可怜啊。",
             "changed": False, "exp": "yes", "exp_has": "可怜"},
        ],
    },
    {
        "name": "mood_情绪该记的记、不该记的别记",
        "turns": [
            {"u": "我下周三要考试", "a": "哦，考试。",
             "changed": True, "mood": "no",
             "note": "聊考试 ≠ 焦虑——不能把话题当情绪"},
            {"u": "可是我好怕考不好啊，压力好大", "a": "……你这次认真了？",
             "changed": False, "mood": "yes"},
            {"u": "对了，现在几点了", "a": "自己看手机。",
             "changed": True, "mood": "no"},
        ],
    },
    {
        "name": "event_寒暄不记、有意义的互动才记",
        "turns": [
            {"u": "早上好呀", "a": "早。",
             "changed": True, "event": "no",
             "note": "纯寒暄不该进事件"},
            {"u": "我今天给你买了生日礼物！", "a": "……你买什么了。",
             "changed": True, "event": "yes",
             "note": "寒暄→送礼，确实换话题了（这里原来我期望 False，是**我写错了**——"
                     "按规则聊了新主题就该是 True）"},
        ],
    },
]


def judge(sc, i, out) -> list:
    """返回 [(条目, 过没过, 说明)]。"""
    t = sc["turns"][i]
    res = []
    if t.get("changed") is not None:
        ok = bool(out["changed"]) == t["changed"]
        res.append(("topic_changed", ok, f"期望 {t['changed']} 实际 {out['changed']}"))
    if t.get("topic_has"):
        ok = t["topic_has"] in (out["topic"] or "")
        res.append(("话题含词", ok, f"期望含「{t['topic_has']}」实际「{out['topic']}」"))
    if t.get("mood") in ("yes", "no"):
        has = bool(out["mood"])
        ok = has == (t["mood"] == "yes")
        res.append(("情绪", ok, f"期望{t['mood']} 实际「{out['mood']}」"))
    if t.get("exp") in ("yes", "no"):
        has = bool(out["expanded"]) and out["expanded"] != t["u"]
        ok = has == (t["exp"] == "yes")
        res.append(("补全", ok, f"期望{t['exp']} 实际「{out['expanded']}」"))
    if t.get("exp_has"):
        ok = t["exp_has"] in (out["expanded"] or "")
        res.append(("补全含词", ok, f"期望含「{t['exp_has']}」实际「{out['expanded']}」"))
    if t.get("exp_has_any"):
        ok = any(w in (out["expanded"] or "") for w in t["exp_has_any"])
        res.append(("补全含词", ok,
                    f"期望含{t['exp_has_any']}之一 实际「{out['expanded']}」"))
    if t.get("event") in ("yes", "no"):
        has = bool(out["event"])
        ok = has == (t["event"] == "yes")
        res.append(("事件", ok, f"期望{t['event']} 实际「{out['event']}」"))
    return res


async def run_scenario(client, sc, uid) -> list:
    """跑一个场景的所有轮次，返回所有判据结果。"""
    print(f"\n{'=' * 74}\n【{sc['name']}】")
    history, results = [], []
    for i, t in enumerate(sc["turns"]):
        hist_text = "\n".join(history[-8:]) or "（无）"
        expanded, parsed = await SM.probe_session(
            user_id=uid, user_msg=t["u"], history_text=hist_text,
            client=client, debug=True)

        def _clean(v):
            v = str(v or "").strip()
            return "" if v.lower() in ("null", "none") else v

        out = {
            "topic": _clean(parsed.get("topic")),
            "changed": bool(parsed.get("topic_changed")),
            "event": _clean(parsed.get("new_event")),
            "mood": _clean(parsed.get("mood")),
            "expanded": expanded,
        }

        print(f"  轮 {i + 1}  用户：{t['u']}")
        print(f"        转话题：{'是' if out['changed'] else '否'} ｜ 话题：{out['topic']}")
        print(f"        补全句：{out['expanded']}"
              f"{'（同原文，没补）' if out['expanded'] == t['u'] else ''}")
        print(f"        情绪：{out['mood'] or '（无）'} ｜ 事件：{out['event'] or '（无）'}")
        if t.get("note"):
            print(f"        ⚠️ 这轮特别要看：{t['note']}")
        for name, ok, why in judge(sc, i, out):
            results.append((f"{sc['name']}/轮{i + 1}/{name}", ok, why))
        history += [f"user：{t['u']}", f"灰泽满：{t['a']}"]
    return results


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每个场景跑几遍（temperature=0.2 有抖动，判'提示词改了有没有用'必须看多次）")
    ap.add_argument("--show-topics", action="store_true", help="只打印每轮判出的话题，方便横着比")
    args = ap.parse_args()

    # ⚠️ 把会话记忆指到临时目录——**探针绝不能往真实的 user_memory/session/ 里写**
    #    （回归测试就是因为漏了这一步，在真实目录里留了 78 个 regr_* 假会话）
    tmp = tempfile.mkdtemp(prefix="hzm_session_probe_")
    SM.use_storage(Path(tmp))

    client, _ = _get_clients()
    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    scs = [s for s in SCENARIOS if not keys or any(k in s["name"] for k in keys)]

    all_results = []
    for rep in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'#' * 74}\n# 第 {rep + 1} 遍\n{'#' * 74}")
        for n, sc in enumerate(scs):
            SM.use_storage(Path(tmp) / f"r{rep}u{n}")     # 每次都从干净会话开始
            all_results += await run_scenario(client, sc, f"probe_{rep}_{n}")

    print(f"\n{'=' * 74}")
    ok = sum(1 for _, o, _ in all_results if o)
    print(f"判据 {ok}/{len(all_results)} 过\n")
    for name, o, why in all_results:
        if not o:
            print(f"  ❌ {name}  —— {why}")
    print()
    for name, o, why in all_results:
        if o:
            print(f"  ✅ {name}")
    return 0 if ok == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))