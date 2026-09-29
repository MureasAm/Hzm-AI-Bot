#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把消融出来的两版**摆给人看**，由人来定夺。

## 为什么改成这样（这条是用户纠正我的）

我原来拿一个 LLM 当裁判判"哪句更像灰泽满"。**但它根本不知道灰泽满是谁**——
它只看过我塞给它的 20 句话，**和我一样没有样本**。所以那种结论建立在一个
"不懂她的人"的偏好上，**不成立**。

**真正知道她是谁的人是用户。** 所以：

    生成两版 → **摆给用户看** → **用户定夺** （不再请任何模型当裁判）

## 关键细节

- **左右顺序打乱**（本仓实测过位置会显著影响判断），并**记下哪边是有文件的那版**
- 每对后面留勾选：`[ ] 甲  [ ] 乙  [ ] 差不多`——用户只回答一个问题
- 统计：把勾好的读回来数（`--tally`），算出"用户选有文件的那版"的比例

用法：
    python scripts/make_ab_review.py                    # 生成待判清单
    python scripts/make_ab_review.py --file persona_ablation.json
    python scripts/make_ab_review.py --tally            # 用户勾完后回读统计
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALDIR = ROOT / "outputs" / "eval"
OUT = ROOT / "outputs" / "ab_review.md"
SEED = 20260929          # 固定种子：左右顺序可复现，不是每次跑都变


def _collect(fname: str, only: str = None):
    """收集 (段名, 有文件的那版, 没文件的那版, 用户那句) 四元组。"""
    p = EVALDIR / fname
    if not p.exists():
        return []
    rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])
    out = []
    for r in rows:
        msg = r.get("msg") or ""
        if r.get("with") and r.get("without"):          # traits/styles 那种形态
            out.append(("【性格基底】+【语言风格】", r["with"], r["without"], msg))
        for s, abl in (r.get("ablation") or {}).items():  # 按段消融那种形态
            if isinstance(abl, str) and abl.strip():
                if only and only not in s:
                    continue
                out.append((s, r.get("reply", ""), abl, msg))
    return out


def generate(files, only):
    items = []
    for f in files:
        items += _collect(f, only)
    rnd = random.Random(SEED)
    L = ["# 两版对照 —— 请你看哪一版更像灰泽满", "",
         "> **每对里，一句是「带着那个文件」的，一句是「删掉那个文件」的**（左右已打乱，你别猜）。",
         "> 你只回答一个问题：**哪一句更像她？** 后面加个 `x` 即可。",
         "> 全部判完跑 `python scripts/make_ab_review.py --tally`，我统计出结果。", "",
         f"> 共 {len(items)} 对 ｜ 生成于固定种子（左右顺序可复现）", "", "---", ""]
    key = []
    for i, (section, with_f, without_f, msg) in enumerate(items, 1):
        a, b = (with_f, without_f) if rnd.random() < 0.5 else (without_f, with_f)
        which = "with" if a is with_f else "without"
        key.append({"i": i, "section": section, "with_is": which, "msg": msg})
        L += [f"### {i}. 用户说：{msg}", "",
              f"**甲**：{a}", "",
              f"**乙**：{b}", "",
              f"<!-- ans:{i} -->  哪个更像她？  [ ] 甲   [ ] 乙   [ ] 差不多", "", "---", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    (EVALDIR / "ab_review_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 待判清单：{OUT}（{len(items)} 对）")
    print("   你看完把 `[ ] 甲` 之类改成 `[x] 甲`，然后跑 --tally")
    return 0


def tally():
    if not OUT.exists():
        print(f"❌ 没有 {OUT}")
        return 1
    key = {k["i"]: k for k in json.loads((EVALDIR / "ab_review_key.json").read_text(encoding="utf-8"))}
    txt = OUT.read_text(encoding="utf-8")
    blocks = re.split(r"<!-- ans:(\d+) -->", txt)
    win_with = win_without = tie = 0
    detail = {}
    for i in range(1, len(blocks), 2):
        idx = int(blocks[i])
        line = blocks[i + 1]
        k = key.get(idx, {})
        pick = ("甲" if re.search(r"\[[xX✓]\]\s*甲", line) else
                "乙" if re.search(r"\[[xX✓]\]\s*乙", line) else
                "差不多" if re.search(r"\[[xX✓]\]\s*差不多", line) else None)
        if not pick:
            continue
        sec = k.get("section", "?")
        d = detail.setdefault(sec, [0, 0, 0])
        if pick == "差不多":
            tie += 1; d[2] += 1
        elif (pick == "甲" and k.get("with_is") == "with") or (pick == "乙" and k.get("with_is") == "without"):
            win_with += 1; d[0] += 1
        else:
            win_without += 1; d[1] += 1
    n = win_with + win_without + tie
    if n == 0:
        print("❌ 一对都没勾（把 [ ] 改成 [x] 了吗？）")
        return 1
    decided = win_with + win_without
    print(f"\n【你的判断】共 {n} 对（其中『差不多』 {tie} 对）")
    print(f"  选「**带着那个文件的那版**」：{win_with}")
    print(f"  选「**删掉那个文件的那版**」：{win_without}")
    if decided:
        print(f"\n  → 在有明确偏好的 {decided} 对里，**{win_with/decided*100:.0f}% 你觉得带着它更像她**")
        print("     （明显高于一半 = 那个文件在起作用；一半附近 = 看不出差别）")
    print("\n【按段拆开】")
    for sec, (w, wo, t) in sorted(detail.items()):
        print(f"  {sec[:26]:<28} 带它更像 {w}｜删掉更像 {wo}｜差不多 {t}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", default=None)
    ap.add_argument("--section", default=None)
    ap.add_argument("--tally", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.tally:
        sys.exit(tally())
    files = args.file or ["persona_ablation.json", "skeleton_ablation.json"]
    sys.exit(generate(files, args.section))
