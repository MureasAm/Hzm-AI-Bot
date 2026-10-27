#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""表情包使用情况统计。

## 为什么有它

"优化表情包内容"这件事，第一个卡点是**没有数据**：在她说话配表情那条链路上，
"发了哪张"原来只 `print` 到控制台，关掉窗口就没了 —— 于是"哪些图从没被用过""哪一类
被反复用"全靠感觉。

2026-10-15 起 `chat_window._maybe_send_sticker` 会把**每一次判断**写进
`data/sticker_log.jsonl`（五种事件：cooldown / topic_used / none / sent / sent_failed），
本脚本就是读它的那个。

## 事件口径

- `cooldown`：冷却没到，**这一轮压根没问模型**（`STICKER_COOLDOWN_TURNS=5`）。
  它记录的是"被冷却吃掉的机会"——这个数大了，说明想统计"该发却没发"就无从谈起。
- `topic_used`：这个话题已经发过一张，按规则跳过，**这一轮也没问模型**
  （2026-10-24 起；同话题一张，另加发后 5 轮冷却）。
- `none`：问了模型，但**没发**。三种原因：模型说 null / 模型给了个不存在的 id /
  判定选中了"最近发过的类别"被代码拦下（`avoid_group`）。
- `sent`：真的发出去了。
- `sent_failed`：选出来了，但发送抛异常。

## 用法

    python scripts/sticker_stats.py                 # 全部历史
    python scripts/sticker_stats.py --since 14      # 只看最近 14 天
    python scripts/sticker_stats.py --json          # 原始聚合结果（给别的脚本用）
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB_FILE = ROOT / "persona" / "decision" / "stickers.json"
LOG_FILE = ROOT / "data" / "sticker_log.jsonl"


# ---------------------------------------------------------------- 纯函数（可测）

def load_library(path: Path = None) -> tuple[list, list]:
    """读表情库；返回 (可用条目, 图文件缺失的条目)。文件坏掉返回 ([], [])。"""
    p = path or LIB_FILE
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []
    items = data.get("stickers", []) if isinstance(data, dict) else []
    ok, missing = [], []
    for s in items:
        if not isinstance(s, dict) or not s.get("id"):
            continue
        f = s.get("file")
        (ok if f and (ROOT / f).exists() else missing).append(s)
    return ok, missing


def load_events(since_days: float = 0, path: Path = None) -> list:
    """读决策流水；坏行跳过（一行坏不能拖垮统计）。"""
    p = Path(path or LOG_FILE)
    if not p.exists():
        return []
    cut = time.time() - since_days * 86400 if since_days else 0
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = rec.get("t")
        if not isinstance(t, (int, float)):
            continue
        if t < cut:
            continue
        out.append(rec)
    return out


def summarize(lib: list, events: list) -> dict:
    """把库 + 流水聚合成一份可直接打印的统计。"""
    events = sorted(events, key=lambda r: r["t"])
    by_event = Counter(e.get("event", "?") for e in events)
    sent = [e for e in events if e.get("event") == "sent"]

    per_id = Counter(e.get("id", "") for e in sent if e.get("id"))
    per_group = Counter(e.get("group", "") for e in sent if e.get("group"))

    # 库里每组几张 → 用来判断"这一组是不是只有一两张在扛"
    lib_by_group = defaultdict(list)
    for s in lib:
        lib_by_group[s.get("group") or "(无组)"].append(s.get("id"))

    asked = by_event.get("none", 0) + by_event.get("sent", 0) + by_event.get("sent_failed", 0)
    none_reasons = Counter()
    for e in events:
        if e.get("event") != "none":
            continue
        w = e.get("why", "")
        if w.startswith("bad_id"):
            none_reasons["模型编了不存在的 id"] += 1
        elif w == "avoid_group":
            none_reasons["选中了最近发过的类别，被代码拦下"] += 1
        elif w.startswith("error:"):
            none_reasons["判定调用出错"] += 1
        else:
            none_reasons["模型判定没有十分对应的（正常）"] += 1

    unused = [s for s in lib if s["id"] not in per_id]

    return {
        "lib_total": len(lib),
        "lib_by_group": {g: sorted(v) for g, v in sorted(lib_by_group.items())},
        "events_total": len(events),
        "by_event": dict(by_event),
        "asked": asked,
        "sent": len(sent),
        "send_rate": (len(sent) / asked) if asked else 0.0,
        "per_id": dict(per_id),
        "per_group": dict(per_group),
        "none_reasons": dict(none_reasons),
        "unused": [s["id"] for s in unused],
        "first_ts": events[0]["t"] if events else None,
        "last_ts": events[-1]["t"] if events else None,
    }


# ---------------------------------------------------------------- 打印

def _bar(n: int, top: int, width: int = 24) -> str:
    return "█" * max(1, round(n / top * width)) if top else ""


def render(st: dict, lib: list, missing: list) -> str:
    L = []
    a = L.append
    a("=" * 68)
    a("表情包库")
    a("=" * 68)
    a(f"可用 {st['lib_total']} 张" + (f"；⚠️ 图文件缺失 {len(missing)} 张（"
        + "、".join(m['id'] for m in missing[:5]) + "…）" if missing else ""))
    for g, ids in st["lib_by_group"].items():
        a(f"  {g:<12} {len(ids):>2} 张")
    a("")

    if st["events_total"] == 0:
        a("=" * 68)
        a("使用情况：还没有数据")
        a("=" * 68)
        a(f"日志文件：{LOG_FILE.relative_to(ROOT)}")
        a("它从 bot 下一次发消息开始记录（四种事件：cooldown / none / sent / sent_failed）。")
        a("跑几天再回来，这里会出现：每张被发的次数、从没被用过的是哪几张、")
        a("模型判定'没有十分对应的'占多少、被冷却吃掉了多少轮。")
        return "\n".join(L)

    import datetime as _dt
    fmt = lambda t: _dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    a("=" * 68)
    a(f"使用情况（{fmt(st['first_ts'])} ~ {fmt(st['last_ts'])}）")
    a("=" * 68)
    ev = st["by_event"]
    a(f"  记录轮次      {st['events_total']}")
    a(f"  冷却没问到    {ev.get('cooldown', 0):>5}   ← 冷却期内的轮次，这轮压根没问模型")
    a(f"  问了模型      {st['asked']:>5}")
    a(f"  真发出去      {st['sent']:>5}   发送率 {st['send_rate']:.1%}"
      + (f"（另有发送失败 {ev.get('sent_failed', 0)}）" if ev.get("sent_failed") else ""))
    a("")

    if st["none_reasons"]:
        a("--- 没发的原因 ---")
        tot = sum(st["none_reasons"].values())
        for k, v in sorted(st["none_reasons"].items(), key=lambda kv: -kv[1]):
            a(f"  {k:<30} {v:>4}  ({v/tot:.0%})")
        a("")

    if st["per_id"]:
        a("--- 每张被发出去的次数 ---")
        top = max(st["per_id"].values())
        names = {s["id"]: s for s in lib}
        for sid, n in sorted(st["per_id"].items(), key=lambda kv: -kv[1]):
            s = names.get(sid, {})
            a(f"  {sid:<14}{n:>4}  {_bar(n, top):<24}{s.get('group','?'):<10}{s.get('desc','')[:26]}")
        a("")

    if st["per_group"]:
        a("--- 按类别（看是不是只有一两张在扛）---")
        n_stickers = {g: len(ids) for g, ids in st["lib_by_group"].items()}
        for g, n in sorted(st["per_group"].items(), key=lambda kv: -kv[1]):
            k = n_stickers.get(g, 0)
            a(f"  {g:<12} 发出 {n:>4} 次   ／ 库里 {k} 张 → 平均每张 {n/k:.1f} 次")
        a("")

    if st["unused"]:
        names = {s["id"]: s for s in lib}
        a(f"--- 从没被发过的 {len(st['unused'])} 张 ---")
        for sid in st["unused"]:
            s = names.get(sid, {})
            a(f"  {sid:<14}［{s.get('group','?'):<8}］{s.get('desc','')[:30]}")
            a(f"      标注的适用场景：{s.get('use_when','')[:60]}")
        a("")
        a("  ↑ 没人用过，通常是两种原因之一：① 这张图的场景跟别的图重叠，模型总能找到更好的；")
        a("     ② `use_when` 写得跟别的图没差别（模型分不出该选谁）。改标注比换图便宜。")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="表情包使用情况统计")
    ap.add_argument("--since", type=float, default=0, help="只看最近 N 天（默认全部）")
    ap.add_argument("--json", action="store_true", help="输出聚合结果 JSON")
    ap.add_argument("--log", default=None, help="日志路径（缺省 data/sticker_log.jsonl）")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    lib, missing = load_library()
    events = load_events(args.since, Path(args.log) if args.log else None)
    st = summarize(lib, events)

    if args.json:
        print(json.dumps(st, ensure_ascii=False, indent=1))
    else:
        print(render(st, lib, missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())