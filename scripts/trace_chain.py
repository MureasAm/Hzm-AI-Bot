#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""链路追踪：**对一条消息，把整条链的每个决策都打印出来。**

## 为什么需要它

整条链（判别 → 文件内容 → 注入 → 生成 → 回复）平时是**看不见的**——
出问题时只能靠猜"是哪一层"。这个工具把它变成可见的。

它同时是**端到端对比的底座**：改动前后跑同一条消息，diff 一眼看出哪层变了。

## ⚠️ 实现原则：**包装真实函数，不重新实现**

如果这里自己写一遍"检索应该怎么做"，那追踪的就不是线上跑的东西了——**那是循环论证**
（项目踩过这个坑：合成场景自测 8/8 却毫无意义）。
所以做法是：把 `core` 里那些函数的**名字换成带日志的包装**，再调真的 `handle_chat`。
调用关系、参数、顺序全部是线上的。

## 用法

    python scripts/trace_chain.py "你多高啊"              # 只看判别与注入
    python scripts/trace_chain.py "你多高啊" --reply      # 连最终回复一起生成
    python scripts/trace_chain.py "今天怎么样" --group     # 按群聊链路（走接话门）
    python scripts/trace_chain.py --baseline outputs/trace_baseline   # 存一组基线
    python scripts/trace_chain.py --diff outputs/trace_baseline       # 改动后对比

## ⚠️ diff 有噪声——别把单次差异当结论

实测：**同一条消息连跑两次，「那你呢」的 corpus 候选就不一样**。原因：
**会话记忆的 probe（短查询扩充）是 LLM 生成的**，每次扩写不同 → 检索用的 query 就不同 → 下游全变。

所以：
- **单条消息、单次 diff 的差异 = 提示，不是证据**
- 要确认某个改动真的有效，看 **① 多跑几次是否稳定复现**，或 **② 用 `judge_experiment.py` 在整份标注集上量指标**
- **对所有消息都出现的、方向一致的差异**才是真信号
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

import src.plugins.chatbot.core as core  # noqa: E402
from src.plugins.chatbot import memory  # noqa: E402

TRACE_USER = "_trace_only_"      # 用完就清，别污染真实记忆
LOG = []


def _wrap(owner, name, label, extract=None):
    """把 owner.name 换成带日志的包装。owner 是模块，name 是模块里的函数名。"""
    orig = getattr(owner, name, None)
    if orig is None:
        return

    async def aw(*a, **k):
        out = await orig(*a, **k)
        LOG.append({"layer": label, "value": extract(out) if extract else out})
        return out

    def sw(*a, **k):
        out = orig(*a, **k)
        LOG.append({"layer": label, "value": extract(out) if extract else out})
        return out

    setattr(owner, name, aw if asyncio.iscoroutinefunction(orig) else sw)


def _items(items):
    """RetrievalItem 列表 → 精简可打印结构。"""
    out = []
    for it in items or []:
        out.append({
            "source": getattr(it, "source", "?"),
            "id": str(getattr(it, "item_id", "")),
            "score": round(getattr(it, "score", 0.0), 3),
            "text": (getattr(it, "text", "") or "")[:70],
            "extra": {k: v for k, v in (getattr(it, "extra", {}) or {}).items()
                      if k in ("user", "reply", "meaning", "phrases")},
        })
    return out


_TRAPS_INSTALLED = False


def install_traps():
    """把 core 里的关键函数换成带日志的包装。**只包不实现**。

    ⚠️ 只能装一次：重复调用会把**已经包过的函数再包一层**，日志就重了
    （踩过：baseline 模式每条消息都装一次，结果同一次判定被记了 N 遍）。
    """
    global _TRAPS_INSTALLED
    if _TRAPS_INSTALLED:
        return
    _TRAPS_INSTALLED = True
    _wrap(core, "classify_l3", "L3分类")
    _wrap(core, "retrieve_corpus", "corpus·门口直通", _items)
    _wrap(core, "retrieve_corpus_candidates", "corpus·候选", _items)
    _wrap(core, "judge_corpus", "corpus·LLM判", _items)
    _wrap(core, "retrieve_voice_samples", "voice_sample", _items)
    _wrap(core, "select_phrase_groups", "phrase", _items)
    _wrap(core, "retrieve_preferences", "preference",
          lambda v: [{"id": x.get("id"), "text": (x.get("text") or "")[:50]} for x in (v or [])])
    _wrap(core, "retrieve_core_stories", "core_story", _items)
    _wrap(core, "select_behavior_item", "behavior",
          lambda b: None if not b else {"id": b.item_id, "text": (b.text or "")[:60]})
    _wrap(core, "fuse_and_truncate", "融合后", _items)
    _wrap(core, "generate_reply", "生成结果")
    # 梗库硬路由（命中就直接回，不过检索）
    _wrap(core, "legendary_confirmed", "梗库确认")


def cleanup():
    """清掉本次追踪写进记忆的东西——追踪是只读操作，不该留下痕迹。"""
    try:
        import json as _j
        for f in ("user_memory/short_term.json", "user_memory/session.json", "user_memory/long_term.json"):
            p = ROOT / f
            if not p.exists():
                continue
            d = _j.loads(p.read_text(encoding="utf-8"))
            rm = [k for k in d if str(k).startswith("_trace")]
            for k in rm:
                d.pop(k)
            if rm:
                p.write_text(_j.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ 清理追踪痕迹失败：{e}")


# ==================== 端到端基线（改动前后对比） ====================
# 一组**有代表性的消息**：覆盖各条路开火/不开火的各种情形。
# 改动前跑一次存下来，改动后跑一次 diff——哪一层变了、变成什么，一眼可见。
MESSAGES = [
    ("日常闲聊", "今天在干嘛呀"),
    ("她有这段经历", "你多高啊"),
    ("她有这段经历", "你和室友关系怎么样"),
    ("被夸", "你唱歌好好听啊"),
    ("被质疑/失约", "说好的八点直播呢？都八点四十了"),
    ("求推荐（问品味）", "小满 给我推荐一首歌"),
    ("问偏好", "你爱吃什么呀"),
    ("求语音（不该检索）", "我要听你的声音"),
    ("纯表情（跳过检索）", "[表情：比心]"),
    ("无关事务（不该命中）", "感冒吃什么药"),
    ("无关事务（不该命中）", "明天几点开会"),
    ("短消息/指代", "那你呢"),
]


async def _run_one(msg: str, group: bool, reply: bool) -> dict:
    LOG.clear()
    install_traps()
    uid = "_trace_group_" if group else TRACE_USER
    if not reply:
        async def _skip(messages):
            LOG.append({"layer": "注入的消息数组",
                        "value": [{"role": m["role"], "len": len(m["content"]),
                                   "head": m["content"][:40]} for m in messages]})
            return "（跳过生成）"
        core.generate_reply = _skip
    try:
        r = await core.handle_chat(uid, msg, is_group=group)
    finally:
        cleanup()
    return {"msg": msg, "group": group, "reply": r, "log": LOG}


def _diff(base: dict, now: dict) -> list:
    """比两次追踪，返回变化列表。只关心"哪层的结果变了"。"""
    out = []
    b = {x["layer"]: x["value"] for x in base.get("log", [])}
    n = {x["layer"]: x["value"] for x in now.get("log", [])}
    for layer in sorted(set(b) | set(n)):
        bv, nv = b.get(layer), n.get(layer)
        if json.dumps(bv, ensure_ascii=False, sort_keys=True) != json.dumps(nv, ensure_ascii=False, sort_keys=True):
            out.append((layer, bv, nv))
    if base.get("reply") != now.get("reply"):
        out.append(("最终回复", base.get("reply"), now.get("reply")))
    return out


def _brief(v, n=3):
    if isinstance(v, list):
        if v and isinstance(v[0], dict) and "id" in v[0]:
            return "[" + ", ".join(str(x.get("id")) for x in v[:n]) + (f" …共{len(v)}" if len(v) > n else "") + "]"
        return f"<{len(v)} 项>"
    return str(v)[:60]


async def run_baseline(outdir: str, reply: bool):
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    print(f"\n跑 {len(MESSAGES)} 条代表消息 → {d}\n")
    for i, (tag, msg) in enumerate(MESSAGES, 1):
        rec = await _run_one(msg, False, reply)
        rec["tag"] = tag
        (d / f"{i:02d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        fired = [x["layer"] for x in rec["log"]
                 if x["layer"] in ("L3分类", "behavior", "phrase", "preference", "core_story")
                 and x["value"] and x["value"] != {"behavior": "", "phrases": []}]
        print(f"  {i:02d} [{tag}] {msg[:22]:<24} 开火: {fired or '无'}")
    print(f"\n✅ 基线已存 {d}（下次改动用 --diff {outdir} 对比）")


async def run_diff(basedir: str, reply: bool):
    d = Path(basedir)
    if not d.exists():
        print(f"❌ 没有基线目录 {d}，先跑 --baseline {basedir}")
        return
    print(f"\n和基线 {d} 对比，跑 {len(MESSAGES)} 条\n")
    changed_any = False
    for i, (tag, msg) in enumerate(MESSAGES, 1):
        f = d / f"{i:02d}.json"
        if not f.exists():
            continue
        base = json.loads(f.read_text(encoding="utf-8"))
        now = await _run_one(msg, False, reply)
        diffs = _diff(base, now)
        if diffs:
            changed_any = True
            print(f"\n{'!'*3} [{tag}] 「{msg}」有变化：")
            for layer, bv, nv in diffs:
                print(f"      {layer:<20} {_brief(bv):<34} → {_brief(nv)}")
    if not changed_any:
        print("\n✅ 全部一致，没有变化")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("msg", nargs="?", default=None, help="用户消息（不填则用 --baseline/--diff）")
    ap.add_argument("--reply", action="store_true", help="真的生成回复（否则跳过生成）")
    ap.add_argument("--group", action="store_true", help="按群聊链路走（会过接话门）")
    ap.add_argument("--json", default=None, help="把追踪结果存成 JSON（供前后 diff）")
    ap.add_argument("--baseline", default=None, help="跑一组代表消息并存成基线目录")
    ap.add_argument("--diff", default=None, help="和基线目录对比（改动前后跑）")
    args = ap.parse_args()

    if args.baseline:
        return await run_baseline(args.baseline, args.reply)
    if args.diff:
        return await run_diff(args.diff, args.reply)
    if not args.msg:
        ap.error("要么给一条消息，要么用 --baseline/--diff")

    if not args.reply:                       # 不生成 → 掐掉真实调用，省时间也避免噪声
        async def _skip(messages):
            LOG.append({"layer": "注入的消息数组",
                        "value": [{"role": m["role"], "len": len(m["content"]),
                                   "head": m["content"][:40]} for m in messages]})
            return "（--reply 未开，跳过生成）"
        core.generate_reply = _skip

    install_traps()
    uid = TRACE_USER if not args.group else "_trace_group_"
    print(f"\n{'═'*76}\n 链路追踪 ｜ {'群聊' if args.group else '私聊'} ｜ 消息：「{args.msg}」\n{'═'*76}")
    print(f"【入口】{'群聊' if args.group else '私聊'} · 无引用 · 无图片 · 无表情")
    print("【读秒窗口】（本工具从 handle_chat 开始，不含读秒攒批）")
    if args.group:
        # 接话门在 chat_window 里（handle_chat 之前），这里单独跑一次给它一个位置
        from src.plugins.chatbot.group_gate import should_reply_in_group
        from src.plugins.chatbot.config import _get_clients
        ds, _ = _get_clients()
        try:
            take = await should_reply_in_group(ds, "", args.msg)
            print(f"【群聊接话门】{'接' if take else '不接'}")
        except Exception as e:
            print(f"【群聊接话门】（判定失败，按'接'放行）{e}")
    else:
        print("【群聊接话门】（私聊不走这个门）")
    try:
        reply = await core.handle_chat(uid, args.msg, is_group=args.group)
    finally:
        cleanup()

    # ---- 打印 ----
    for rec in LOG:
        layer = rec["layer"]
        v = rec["value"]
        if layer == "L3分类" and isinstance(v, dict):
            print(f"\n【L3 分类】behavior={v.get('behavior') or '∅'}   phrase={v.get('phrases') or '∅'}")
        elif layer.startswith("corpus"):
            n = len(v) if isinstance(v, list) else 1
            print(f"\n【{layer}】{n} 条")
            for it in (v or [])[:6] if isinstance(v, list) else []:
                print(f"    #{it['id']:<5} {it['score']:>6}  {it['text'][:56]}")
        elif layer in ("voice_sample", "phrase", "core_story", "behavior"):
            if v is None:
                print(f"\n【{layer}】未命中")
            else:
                arr = v if isinstance(v, list) else [v]
                print(f"\n【{layer}】{len(arr)} 条")
                for it in arr:
                    d = it.get("extra") or {}
                    tip = d.get("reply") or d.get("meaning") or it.get("text", "")
                    print(f"    {it['id']:<24} {str(tip)[:50]}")
        elif layer == "preference":
            print(f"\n【preference】{len(v or [])} 条  " +
                  " ｜ ".join(x.get("id", "") for x in (v or [])))
        elif layer == "融合后":
            print(f"\n【RRF 融合后】{len(v or [])} 条（top6 + 1200 字预算内）")
            for it in v or []:
                print(f"    {it['source']:<14} {it['id']:<22} {it['text'][:40]}")
        elif layer == "注入的消息数组":
            print(f"\n【注入的消息数组】{len(v)} 段")
            for m in v:
                print(f"    [{m['role']:>9}] {m['len']:>5}字  {m['head']}")
        else:
            print(f"\n【{layer}】{str(v)[:200]}")

    print(f"\n{'─'*76}\n【最终回复】{reply}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"msg": args.msg, "group": args.group, "reply": reply, "log": LOG},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 追踪已存 {args.json}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
