#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""撞车检查：behaviors 的 samples.user 和回归/评测用例的 query 是否同话题。

为什么需要它：实测两次踩同一个坑——
  样本的 user 与测试输入几乎一字不差 → 模型把样本当标准答案 → 逐字复读（回归直接挂）。
  这不是"检索太准"，是样本和输入撞车。写数据时没人会记得去比这个，所以做成工具。

判据：**只看特有词**（去掉停用词、角色自称、泛化人称后的 bigram），
      特有词重合 ≥ 50% 才算撞车——否则"灰泽满""是不是"这种到处都是的词会刷一堆误报。
用法：python scripts/check_collision.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 高频词：这些词重合不代表"同话题"，只会制造误报
COMMON = set("""的了呢吧吗啊嗯哈是不没都我你他她它们这那有个上下里说问想会能要可以给对来到去做用把被让但又还也真太
就是灰泽满hzm小满满姐绿冻粉丝观众弹幕直播一下怎么什么为什么不是没有可以自己一个这个那个时候现在最近今天明天
觉得感觉知道看看说说问问行吧好吧那那嗯嗯哈哈""")

SPLIT_RE = re.compile(r"[\s，。！？、；：\"'（）()\[\]【】~…,.!?;:\-—]+")


def tokens(t: str) -> set:
    """切成≥2字的片段（按标点/空格），再去掉高频词。"""
    parts = [p for p in SPLIT_RE.split(str(t)) if len(p) >= 2]
    out = set()
    for p in parts:
        p = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", p)
        if len(p) < 2:
            continue
        for i in range(len(p) - 1):
            bg = p[i:i + 2]
            if bg not in COMMON:
                out.add(bg)
    return out


def overlap(a: str, b: str) -> float:
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def main():
    behaviors = json.loads(
        (ROOT / "persona" / "behavior" / "behaviors.json").read_text(encoding="utf-8")
    )["behaviors"]

    queries = []
    for f in ["scripts/retrieval_eval_cases.json", "scripts/regression_cases.json"]:
        p = ROOT / f
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        items = d if isinstance(d, list) else (d.get("cases") or d.get("regression") or [])
        for x in items:
            q = x.get("query") or x.get("user") or x.get("msg")
            if q:
                queries.append((x.get("id", "?"), str(q)))
    print(f"比对：{len(behaviors)} 条 behaviors × {len(queries)} 条用例（只在特有词上比）\n")

    bad = 0
    for b in behaviors:
        for s in b.get("samples") or []:
            u = str(s.get("user") or "")
            if not u:
                continue
            best = max(((overlap(u, q), cid, q) for cid, q in queries), default=(0, "", ""))
            if best[0] >= 0.5:
                bad += 1
                print(f"⚠️ 特有词撞车 {best[0]:.0%}  [{b['name']}]")
                print(f"     样本 user : {u}")
                print(f"     用例 {best[1]:16s}: {best[2]}")
    print()
    if not bad:
        print("✅ 没有发现同话题撞车的样本")
    else:
        print(f"共 {bad} 处。**同话题撞车会让模型把样本当标准答案逐字搬——改样本，别改用例。**")


if __name__ == "__main__":
    main()
