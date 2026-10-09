#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""骨架按小节消融 × **确定性判据**：删掉某一节后，她还守不守规矩。

## 为什么换这套判据（前因）

风格判据（"哪句更像她"）在**细微差别**上不可靠 —— 实测它给出的结论
跟我、跟用户用眼睛看的**不一致**。而骨架的真实职责其实是**守规矩**：
「绝不用"她"指自己」「不懂就说不懂」「别当客服」「别编具体的事」。

**这些能用禁词判据测，不需要任何模型当裁判** —— 而且**你自己就能验**：
看回复里有没有那个词。

## 测法

拿仓库现成的规矩用例（`scripts/regression_cases.json`）当**输入**
（它们就是"该触发某条规矩"的话），对每个用例：
  ① 正常生成 N 次 → 数犯规次数（基线）
  ② 逐个删掉骨架的某一小节 → 各生成 N 次 → 数犯规次数
**删掉哪一节后犯规明显变多 → 那一节在管这条规矩。**

判据就是 `forbid`（禁词）和 `forbid_pattern`（正则）——**一眼可验**。

用法：
    python scripts/skeleton_rule_ablation.py                  # 全部规矩用例
    python scripts/skeleton_rule_ablation.py --runs 4
    python scripts/skeleton_rule_ablation.py --case no_assistant_tone
"""
import argparse
import asyncio
import json
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

import src.plugins.chatbot.core as core            # noqa: E402
import src.plugins.chatbot.memory as mem           # noqa: E402

CASES_FILE = ROOT / "scripts" / "regression_cases.json"
OUT = ROOT / "outputs" / "eval" / "skeleton_rule_ablation.json"
_SKEL_SEC = re.compile(r"^## 【", re.M)

_cap = {}
_ORIG = None


async def _capture(messages):
    _cap["msgs"] = messages
    return "（占位）"


def _units(base: str):
    """骨架拆成可单独删的小节（+ 性格基底/语言风格那一坨）。"""
    skel = core.SYSTEM_PROMPT
    tail = base[len(skel):] if base.startswith(skel) else ""
    idx = [m.start() for m in _SKEL_SEC.finditer(skel)]
    units = []
    if idx:
        units.append(("骨架开头那段", skel[: idx[0]]))
    for i, st in enumerate(idx):
        en = idx[i + 1] if i + 1 < len(idx) else len(skel)
        units.append((skel[st: st + 16].split("\n")[0].strip().replace("## ", ""), skel[st:en]))
    if tail.strip():
        units.append(("性格基底+语言风格", tail))
    return units


def _violations(reply: str, case: dict) -> int:
    """这条回复犯了几次规（禁词命中 + 正则命中）。**一眼可验**。"""
    r = reply or ""
    n = 0
    for w in (case.get("forbid") or []):
        if w in r:
            n += 1
    pat = case.get("forbid_pattern")
    if pat:
        try:
            if re.search(pat, r):
                n += 1
        except re.error:
            pass
    return n


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    doc = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = [c for c in doc["cases"]
             if (c.get("forbid") or c.get("forbid_pattern"))
             and (not args.case or args.case == c["id"])]
    if not cases:
        print("❌ 没有可测的规矩用例"); return 1

    core.update_memory_v2_shadow_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="skelrule_")
    mem.use_storage(Path(tmp))
    global _ORIG
    _ORIG = core.generate_reply
    core.generate_reply = _capture

    print(f"📏 骨架按小节消融 × 禁词判据：{len(cases)} 条规矩 × {args.runs} 次\n")
    rows = []
    for c in cases:
        _cap.clear()
        uid = f"skr_{c['id']}"
        # 多轮场景：把 history 垫进去（和 regression_test 一样的做法）
        for h in (c.get("history") or []):
            mem.append_user_history(uid, h.get("user", ""), h.get("reply", ""))
        try:
            await core.handle_chat(uid, c["user"], is_group=bool(c.get("is_group")))
        except Exception as e:
            print(f"  [{c['id']}] 失败：{type(e).__name__}"); continue
        msgs = _cap.get("msgs")
        if not msgs:
            continue
        base0 = msgs[0].get("content") or ""
        units = _units(base0)
        rec = {"case": c["id"], "desc": c.get("desc", ""), "user": c["user"],
               "base_v": 0, "units": {}}
        for _ in range(args.runs):
            rec["base_v"] += _violations(await _ORIG([dict(m) for m in msgs]), c)
        for title, text in units:
            ab = [dict(m) for m in msgs]
            ab[0]["content"] = base0.replace(text, "", 1)
            if ab[0]["content"] == base0:
                continue
            v = 0
            for _ in range(args.runs):
                v += _violations(await _ORIG(ab), c)
            rec["units"][title] = v
        rows.append(rec)
        print(f"  [{c['id']}] 基线犯规 {rec['base_v']}/{args.runs} 次")

    print(f"\n{'=' * 78}\n【结果】删掉哪一节后，她开始犯规？（数字 = 犯规次数，次数越大越差）\n")
    # 每节在多少条规矩上让犯规变多
    worse = {}
    for r in rows:
        for title, v in r["units"].items():
            if v > r["base_v"]:
                worse.setdefault(title, []).append((r["case"], r["base_v"], v))
    if not worse:
        print("⚠️ **没有任何一节**的缺失让犯规变多 —— 在这批规矩上，骨架各节都在守（或都无所谓）。")
    else:
        for title, lst in sorted(worse.items(), key=lambda kv: -len(kv[1])):
            print(f"\n【{title}】→ 删掉后在这些规矩上犯规变多：")
            for cid, b, v in lst:
                print(f"    · {cid}：基线 {b} 次 → 删掉后 {v} 次")

    print("\n⚠️ 判据是**禁词/正则**，你自己就能验：去看 `outputs/eval/skeleton_rule_ablation.json`")
    print("   里那条回复到底有没有出现那个词。**不需要信任何模型当裁判**。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"runs": args.runs, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"✅ 明细已保存: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
