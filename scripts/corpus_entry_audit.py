#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**corpus 的「召回入口」审计**（`注入设计原理.md` §七-3，文档标的"下一件该做的"）。

## 判据（就一句）

> **用户会说什么话，能把这条素材勾出来？**

`corpus_asks.json` 里那 485 条假问法**当索引用失败了**，但**当审计正合适**——
因为它生成时**逐条自验证过**（拿这个问法去检索，能把它自己捞回来才留下）。

所以：**有 asks 的 = 有入口；一条假问法都生不出来的 = 疑似无入口。**

⚠️ **别按位置对齐。** `asks` 只有 181 条、而 corpus 有 322 条——它是"有入口"那些的
**子序列**（没入口的被跳过了）。而且 corpus 在 2026-09-26 做过形式改造（文本变了），
所以 asks 的键是**改造前**的旧文本，**对不上新文本**。
→ 用**单调的相似度对齐**（按顺序找最佳匹配），并且**报对齐质量**——
  本仓教训：「从旧标注集继承条目时，要确认它指的素材真的存在」。

## 用法
    python scripts/corpus_entry_audit.py
    python scripts/corpus_entry_audit.py --min-overlap 0.30
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

WORLD = _common.WORLD_DIR
OUT = _common.OUTPUTS_DIR / "eval" / "corpus_entry_audit.json"
OUT_MD = _common.OUTPUTS_DIR / "corpus_no_entry.md"


def _bg(s: str) -> set:
    s = re.sub(r"[^一-龥]", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def _ov(a: str, b: str) -> float:
    ba = _bg(a)
    return 0.0 if not ba else len(ba & _bg(b)) / len(ba)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-overlap", type=float, default=0.30,
                    help="对齐时接受的最低重合度（低于它的当作对不上，会报出来）")
    args = ap.parse_args()
    _common.ensure_utf8_stdout()

    st = [x["statement"] for x in json.loads(
        (WORLD / "statement_final.json").read_text(encoding="utf-8"))["statements"]]
    asks = json.loads((WORLD / "corpus_asks.json").read_text(encoding="utf-8"))["asks"]
    kw = json.loads((WORLD / "corpus_keywords.json").read_text(encoding="utf-8"))["keywords"]
    ask_keys = list(asks)

    # ==================== 单调相似度对齐 ====================
    pairs, cursor, weak = [], 0, []
    for k in ask_keys:
        best_i, best_o = None, 0.0
        for i in range(cursor, len(st)):
            o = _ov(k, st[i])
            if o > best_o:
                best_i, best_o = i, o
        if best_i is None or best_o < args.min_overlap:
            weak.append((k, best_o))
            continue
        pairs.append((best_i, best_o))
        cursor = best_i + 1

    print(f"corpus {len(st)} 条 ｜ asks {len(ask_keys)} 条（= 生成时自验证通过的）")
    print(f"对齐：成功 {len(pairs)} 条 ｜ 对不上 {len(weak)} 条")
    ovs = sorted(o for _, o in pairs)
    if ovs:
        print(f"  对齐重合度：中位 {ovs[len(ovs) // 2]:.2f}　最低 {ovs[0]:.2f}　"
              f"低于 0.5 的 {sum(1 for o in ovs if o < 0.5)} 条")
    if weak:
        print(f"  ⚠️ 对不上的（要么旧文本改了太多、要么它本来就没入口）：")
        for k, o in weak[:5]:
            print(f"     重合 {o:.2f}　{k[:40]}")

    has_entry = {i for i, _ in pairs}
    no_entry = [i for i in range(len(st)) if i not in has_entry]

    # ==================== 入口的两种：asks（假问法）/ keywords（钩子） ====================
    hooked = {i for i in range(len(st)) if kw.get(str(i))}
    print(f"\n【分类】")
    print(f"  有入口（有假问法）           {len(has_entry):>4} 条")
    print(f"  无入口 但**有钩子**          {len(set(no_entry) & hooked):>4} 条"
          f"　← 用户提到钩子里的词时还能被递过去")
    print(f"  **无入口 且 无钩子**         {len(set(no_entry) - hooked):>4} 条"
          f"　← ⚠️ **用户说什么话都勾不出它**")

    bare = sorted(set(no_entry) - hooked)

    # ==================== ★ 用**真实对话**验证入口 ====================
    # 「有入口」和「真的被勾出来过」是两回事（文档原话：判据"能不能"工作 vs 用户"会不会"这么说）。
    # 前者是 LLM 造的问法/生成的钩子表（都**没经过真实数据检验**），后者只有真实对话能答。
    chat = _common.PROJECT_ROOT / "data" / "chat_log" / "chat.jsonl"
    real_hits, real_entries = {}, set()
    n_msgs = 0
    if chat.exists():
        for line in chat.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            u = (r.get("user") or "").strip()
            if not u:
                continue
            n_msgs += 1
            for i in range(len(st)):
                words = kw.get(str(i)) or []
                if any(w and w in u for w in words):
                    real_hits[i] = real_hits.get(i, 0) + 1
                    real_entries.add(i)
    print(f"\n【★ 真实对话验证】扫了 {n_msgs} 条真实用户消息")
    if n_msgs:
        print(f"  钩子词**真的被用户说到过**的条目：{len(real_entries)} / {len(st)} 条"
              f"（{len(real_entries) / len(st) * 100:.0f}%）")
        never = [i for i in range(len(st)) if i not in real_entries]
        print(f"  → **从没被真实勾出过**：{len(never)} 条（{len(never) / len(st) * 100:.0f}%）")
        top = sorted(real_hits.items(), key=lambda kv: -kv[1])[:6]
        if top:
            print(f"  被勾得最多的：")
            for i, c in top:
                print(f"    ×{c:<3} #{i:<4}{(kw.get(str(i)) or [''])[0][:8]:<10}{st[i][:40]}")
    else:
        never = []

    print(f"\n【无入口且无钩子的 {len(bare)} 条】前 12 条：")
    for i in bare[:12]:
        print(f"  #{i:<4}{st[i][:58]}")

    if bare:
        lines = ["# corpus 里「用户说什么话都勾不出」的条目", "",
                 f"> 判据：既没有自验证通过的假问法（`corpus_asks.json`），也没有钩子"
                 f"（`corpus_keywords.json`）。共 **{len(bare)}** 条 / 322。",
                 "> 生成：`python scripts/corpus_entry_audit.py`", "",
                 "| # | 条目 | 建议 |", "|---|---|---|"]
        for i in bare:
            lines.append(f"| {i} | {st[i][:70]} | 删 / 或补钩子 |")
        OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_statements": len(st),
        "n_asks": len(ask_keys),
        "aligned": [{"idx": i, "overlap": o} for i, o in pairs],
        "weak": [{"key": k, "overlap": o} for k, o in weak],
        "no_entry": no_entry,
        "no_entry_no_hook": bare,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n⚠️ 读法：")
    print(f"  · 「无入口且无钩子」= **用户说什么话都勾不出它** → 它永远不会进上下文。")
    print(f"    处置两条：**删**（省向量库空间）或**补钩子**（想留就得给入口）。")
    print(f"  · 「有钩子」的不算没入口——钩子是词面匹配，用户提到那个词就会把它递过去。")
    print(f"  · ⚠️ 对齐是**相似度**做的：抽查上面「对不上」那几条，别当它是零。")
    print(f"\n✅ 清单：{OUT_MD}\n✅ 明细：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
